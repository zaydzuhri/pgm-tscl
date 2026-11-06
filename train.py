"""Finetune a causal LM (the Student) while a Teacher chooses which difficulty buckets each batch comes from.

    uv run train.py --config configs/sft.json
    uv run train.py --config configs/tscl_window.json --seed 1 --total_steps 1000   # CLI overrides JSON

We follow the *batch POMDP* formulation of TSCL (Matiisen et al. 2017, Sec 2.2), since this is
supervised learning:

  - tasks       = difficulty buckets of MATH (e.g. Level 1..5)
  - action a_t  = a probability distribution over the N buckets; training batches are sampled from it
  - observation = per-bucket scores o_t = (x^(1)_t, ..., x^(N)_t) on a held-out validation set
                  that contains every bucket
  - reward      = how the scores change (the slope of the learning curve), worked out inside each Teacher

This gives a doubly nested loop:

    o_0 = observe(student)                                  # scores before any training
    for each teacher timestep t:                            # OUTER loop: one teacher decision
        a_t = teacher.distribution()
        repeat teacher_interval times:                      # INNER loop: ordinary finetuning steps
            batch ~ a_t ; one optimizer step on the student
        o_t = observe(student) ; teacher.update(o_t)        # teacher learns from the score change

Fair comparison: every run, whatever its teacher, uses the same number of training examples
(total_steps x batch_size x grad_accum), the same LR schedule and the same test-eval steps. Only the
mix of buckets in each batch differs. Ordinary finetuning is just the `proportional` teacher, which
never changes its distribution. The teacher's extra cost (val forward passes) is logged separately
as `teacher_eval_examples`.
"""
import argparse
import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import data
import teachers


@dataclass
class Config:
    # --- Student (model) and tasks (data) ---
    model: str = "Qwen/Qwen3-0.6B"
    dataset: str = "math"
    bucket_by: str = "level"            # how to split data into tasks: level | subject | level_subject | none
    n_val_per_bucket: int = 32          # held-out examples per task, used only to score the Student for the Teacher
    max_len: int = 1024
    # --- Teacher ---
    teacher: str = "proportional"       # see teachers.TEACHERS; "proportional" = ordinary finetuning
    teacher_kwargs: dict = field(default_factory=dict)   # e.g. {"K": 10, "alpha": 0.1, "eps": 0.1}
    teacher_interval: int = 10          # optimizer steps per teacher timestep (inner loop length)
    score: str = "val_loss"             # what the Teacher observes per task: val_loss | val_acc | train_loss
    always_score: bool = False          # also score baselines each round (only for plotting learning curves)
    # --- Training budget (keep these identical across runs you compare) ---
    total_steps: int = 500              # total optimizer steps
    batch_size: int = 4                 # micro-batch size
    grad_accum: int = 8                 # examples per optimizer step = batch_size * grad_accum
    lr: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    grad_ckpt: bool = False
    # --- Evaluation / logging ---
    eval_every: int = 100               # optimizer steps between test-set evaluations (same for every teacher)
    eval_at_start: bool = True
    test_size: int = 500                # 0 = full test set
    gen_batch_size: int = 64
    max_new_tokens: int = 1024
    log_every: int = 10
    seed: int = 0
    data_seed: int = 0                  # fixes the train/val split, so all runs see the same data
    out_dir: str = ""                   # default: runs/<config-name>_s<seed>
    save_model: bool = False


def load_config():
    """Config values come from: dataclass defaults, then the --config JSON file, then any --key value CLI flags."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="JSON file with Config fields")
    for f in dataclasses.fields(Config):
        t = {bool: lambda s: s.lower() in ("1", "true", "yes"), dict: json.loads}.get(f.type, f.type)
        p.add_argument(f"--{f.name}", type=t, default=None)
    args = vars(p.parse_args())
    cfg_path = args.pop("config")
    values = {}
    if cfg_path:
        with open(cfg_path) as fh:
            values = json.load(fh)
        unknown = set(values) - {f.name for f in dataclasses.fields(Config)}
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
    values.update({k: v for k, v in args.items() if v is not None})
    cfg = Config(**values)
    if not cfg.out_dir:
        name = os.path.splitext(os.path.basename(cfg_path))[0] if cfg_path else cfg.teacher
        cfg.out_dir = os.path.join("runs", f"{name}_s{cfg.seed}")
    return cfg


# ----------------------------------------------------------------------------- Student (model)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Weights are kept in fp32; on CUDA the forward pass runs in bf16 autocast.
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32).to(device)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return tok, model


def forward_loss(model, batch):
    """Next-token cross-entropy on the solution tokens only.

    Returns (sum of token losses over the batch, mean loss of each example). The per-example losses
    let us track the loss of each bucket. To save memory, logits are only computed at positions that
    have a label.
    """
    hidden = model.base_model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]               # token t predicts token t+1
    has_label = labels != -100                    # prompt and padding positions are -100
    logits = model.get_output_embeddings()(hidden[:, :-1][has_label]).float()
    token_loss = F.cross_entropy(logits, labels[has_label], reduction="none")
    example_of_token = has_label.nonzero()[:, 0]
    loss_sum_per_example = torch.zeros(labels.shape[0], device=hidden.device).index_add_(0, example_of_token, token_loss)
    return token_loss.sum(), loss_sum_per_example / has_label.sum(1).clamp(min=1)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def generate(model, tok, problems, cfg, device, amp):
    """Greedy decoding for a list of problems."""
    model.eval()
    tok.padding_side = "left"
    stop = [tok.eos_token_id] + [i for i in [tok.convert_tokens_to_ids("<|im_end|>")]
                                 if isinstance(i, int) and i != tok.unk_token_id]
    outs = []
    for i in range(0, len(problems), cfg.gen_batch_size):
        prompts = [data.format_prompt(tok, p) for p in problems[i:i + cfg.gen_batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with amp:
            g = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                               temperature=None, top_p=None, top_k=None, use_cache=True,
                               eos_token_id=stop, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    model.train()
    return outs


def accuracy(model, tok, examples, cfg, device, amp):
    """List of True/False: did the Student's final \\boxed{} answer match the reference?"""
    outs = generate(model, tok, [e["problem"] for e in examples], cfg, device, amp)
    return [data.is_correct(o, e["answer"]) for o, e in zip(outs, examples)]


def test_eval(model, tok, test, cfg, device, amp):
    """Final-task performance on the test set (MATH-500): overall, per level, and per subject.
    Only for reporting; the Teacher never sees it."""
    correct = accuracy(model, tok, test, cfg, device, amp)
    res = {"test_acc": float(np.mean(correct))}
    for key in ("level", "subject"):
        groups = {}
        for c, e in zip(correct, test):
            groups.setdefault(data.bucket_key(e, key), []).append(c)
        res[f"test_acc_by_{key}"] = {k: float(np.mean(v)) for k, v in sorted(groups.items())}
    return res


@torch.no_grad()
def val_scores(model, tok, val_raw, val_enc, cfg, device, amp):
    """The Teacher's observation o_t = (x^(1)_t, ..., x^(N)_t): one score per task on the validation set.

    Higher is better. The paper uses validation accuracy. That needs generation, which is slow for an
    LLM, so by default we use -validation loss (one forward pass). Set score="val_acc" to use accuracy.
    """
    if cfg.score == "val_acc":
        return np.array([np.mean(accuracy(model, tok, bucket, cfg, device, amp)) for bucket in val_raw])
    model.eval()
    scores = []
    for bucket in val_enc:
        losses = []
        for i in range(0, len(bucket), cfg.batch_size):
            batch = to_device(data.collate(bucket[i:i + cfg.batch_size], tok.pad_token_id), device)
            with amp:
                _, example_losses = forward_loss(model, batch)
            losses += example_losses.tolist()
        scores.append(-float(np.mean(losses)))
    model.train()
    return np.array(scores)


def train_loss_scores(round_loss_sum, round_count, prev_scores):
    """Free alternative observation (score="train_loss"): -mean training loss of each task during
    the last round. A task that was not sampled this round keeps its previous score."""
    scores = np.full(len(round_count), np.nan)
    sampled = round_count > 0
    scores[sampled] = -round_loss_sum[sampled] / round_count[sampled]
    if prev_scores is not None:
        scores[~sampled] = prev_scores[~sampled]
    scores[np.isnan(scores)] = np.nanmean(scores)   # never seen yet: use the average of the others
    return scores


# ----------------------------------------------------------------------------- main: the Teacher-Student loop
def main():
    cfg = load_config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2)
    logf = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")

    def log(kind, **kw):
        logf.write(json.dumps({"type": kind, **kw}) + "\n")
        logf.flush()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    # --- Student ---
    tok, model = load_model(cfg, device)
    model.train()

    # --- Tasks: training data split into N buckets, plus a val set per bucket and a test set ---
    td = data.load_task_data(cfg.dataset, bucket_by=cfg.bucket_by, n_val=cfg.n_val_per_bucket,
                             data_seed=cfg.data_seed)
    N = len(td.names)
    train_enc = [[data.encode(tok, ex, cfg.max_len, i) for ex in bucket] for i, bucket in enumerate(td.train)]
    val_enc = [[data.encode(tok, ex, cfg.max_len, i) for ex in bucket] for i, bucket in enumerate(td.val)]
    test = td.test[:cfg.test_size] if cfg.test_size else td.test
    sizes = [len(b) for b in train_enc]
    print(f"device={device} buckets={dict(zip(td.names, sizes))} val/bucket={cfg.n_val_per_bucket} test={len(test)}")

    # --- Teacher, plus the sampler that turns its distribution into actual batches ---
    sampler = data.BucketSampler(train_enc, seed=cfg.seed)
    teacher = teachers.make_teacher(cfg.teacher, N, sizes, seed=cfg.seed, **cfg.teacher_kwargs)
    use_scores = teacher.needs_scores or cfg.always_score

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * cfg.total_steps), cfg.total_steps)

    # Budget counters. Compare runs at equal `examples` (or `tokens`), not equal wall-clock.
    step, t, examples, tokens, teacher_eval_examples = 0, 0, 0, 0, 0

    def counters():
        return dict(step=step, examples=examples, tokens=tokens, teacher_eval_examples=teacher_eval_examples)

    def run_test_eval():
        t0 = time.time()
        res = test_eval(model, tok, test, cfg, device, amp)
        log("eval", **counters(), **res)
        print(f"[eval] step {step} test_acc={res['test_acc']:.4f} by_level={res['test_acc_by_level']} ({time.time() - t0:.0f}s)")

    if cfg.eval_at_start:
        run_test_eval()

    # o_0: score every task once before training, so the first reward (a score change) can be computed.
    scores = None
    if use_scores and cfg.score != "train_loss":
        scores = val_scores(model, tok, td.val, val_enc, cfg, device, amp)
        teacher_eval_examples += sum(len(b) for b in td.val)
        teacher.update(scores)
        log("round", round=0, **counters(), scores=scores.tolist(), names=td.names)

    t0 = time.time()
    while step < cfg.total_steps:
        # ===== OUTER LOOP: one Teacher timestep t =====
        # Action a_t: the Teacher picks a probability distribution over the N tasks.
        task_probs = teacher.distribution()

        # Per-task training loss during this round (logged, and used when score="train_loss").
        round_loss_sum, round_count = np.zeros(N), np.zeros(N)

        # ===== INNER LOOP: the Student trains for `teacher_interval` optimizer steps on batches drawn from a_t =====
        for _ in range(min(cfg.teacher_interval, cfg.total_steps - step)):
            # Sample which task each example comes from (according to a_t), then take examples from those buckets.
            batch = sampler.sample(task_probs, cfg.batch_size * cfg.grad_accum)
            micro_batches = [to_device(data.collate(batch[i:i + cfg.batch_size], tok.pad_token_id), device)
                             for i in range(0, len(batch), cfg.batch_size)]

            # Standard gradient step (with gradient accumulation). The loss is averaged over every
            # solution token in the full batch.
            n_tok = sum(int((mb["labels"][:, 1:] != -100).sum()) for mb in micro_batches)
            loss_sum = 0.0
            for mb in micro_batches:
                with amp:
                    token_loss_sum, example_losses = forward_loss(model, mb)
                (token_loss_sum / n_tok).backward()
                loss_sum += token_loss_sum.item()
                task_ids = mb["bucket"].cpu().numpy()
                np.add.at(round_loss_sum, task_ids, example_losses.detach().float().cpu().numpy())
                np.add.at(round_count, task_ids, 1)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            examples += len(batch)
            tokens += n_tok
            if step % cfg.log_every == 0:
                loss = loss_sum / n_tok
                log("train", **counters(), loss=loss, grad_norm=float(gnorm), lr=sched.get_last_lr()[0])
                print(f"step {step}/{cfg.total_steps} loss={loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"task_probs={np.round(task_probs, 3).tolist()} ({(time.time() - t0) / step:.2f}s/step)")
            # Test evals happen at fixed optimizer steps, so every teacher is measured at the same budget.
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                run_test_eval()

        # ===== Back in the OUTER LOOP: the Teacher observes o_t and updates =====
        # (Inside teacher.update: reward = score change / learning-curve slope, then the policy is updated.)
        t += 1
        if use_scores:
            if cfg.score == "train_loss":
                scores = train_loss_scores(round_loss_sum, round_count, scores)
            else:
                scores = val_scores(model, tok, td.val, val_enc, cfg, device, amp)
                teacher_eval_examples += sum(len(b) for b in td.val)
            teacher.update(scores)
        log("round", round=t, **counters(), probs=task_probs.tolist(), names=td.names,
            scores=None if scores is None else scores.tolist(),
            drawn=sampler.drawn.tolist(), unique_seen=[len(s) for s in sampler.seen], **teacher.state())

    if cfg.save_model:
        model.save_pretrained(os.path.join(cfg.out_dir, "model"))
        tok.save_pretrained(os.path.join(cfg.out_dir, "model"))
    logf.close()


if __name__ == "__main__":
    main()
