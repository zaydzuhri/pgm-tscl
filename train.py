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
(total_steps x batch_size), the same LR schedule and the same test-eval steps. Only the mix of
buckets in each batch differs. Ordinary finetuning is just the `proportional` teacher, which never
changes its distribution. The teacher's extra cost (val forward passes) is logged separately as
`teacher_eval_examples`.
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
    teacher_kwargs: dict = field(default_factory=dict)   # e.g. {"K": 10, "alpha": 0.1, "epsilon": 0.1}
    teacher_interval: int = 10          # optimizer steps per teacher timestep (inner loop length)
    score: str = "val_loss"             # what the Teacher observes per task: val_loss | val_acc | train_loss
    always_score: bool = False          # also score baselines each round (only for plotting learning curves)
    # --- Training budget (keep these identical across runs you compare) ---
    total_steps: int = 500              # total optimizer steps
    batch_size: int = 32                # examples per optimizer step (use grad_ckpt if memory is tight)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON file with Config fields")
    for config_field in dataclasses.fields(Config):
        parse_type = {bool: lambda text: text.lower() in ("1", "true", "yes"),
                      dict: json.loads}.get(config_field.type, config_field.type)
        parser.add_argument(f"--{config_field.name}", type=parse_type, default=None)
    cli_args = vars(parser.parse_args())
    config_path = cli_args.pop("config")
    values = {}
    if config_path:
        with open(config_path) as config_file:
            values = json.load(config_file)
        unknown_keys = set(values) - {config_field.name for config_field in dataclasses.fields(Config)}
        if unknown_keys:
            raise ValueError(f"unknown config keys: {unknown_keys}")
    values.update({key: value for key, value in cli_args.items() if value is not None})
    config = Config(**values)
    if not config.out_dir:
        run_name = os.path.splitext(os.path.basename(config_path))[0] if config_path else config.teacher
        config.out_dir = os.path.join("runs", f"{run_name}_s{config.seed}")
    return config


# ----------------------------------------------------------------------------- Student (model)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Weights are kept in fp32; on CUDA the forward pass runs in bf16 autocast.
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.float32).to(device)
    if config.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return tokenizer, model


def forward_loss(model, batch):
    """Next-token cross-entropy on the solution tokens only.

    Returns (mean token loss over the batch, mean loss of each example). The per-example losses
    let us track the loss of each bucket. To save memory, logits are only computed at positions that
    have a label.
    """
    hidden_states = model.base_model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]               # token t predicts token t+1
    has_label = labels != -100                    # prompt and padding positions are -100
    logits = model.get_output_embeddings()(hidden_states[:, :-1][has_label]).float()
    token_loss = F.cross_entropy(logits, labels[has_label], reduction="none")
    example_of_token = has_label.nonzero()[:, 0]
    loss_sum_per_example = torch.zeros(labels.shape[0], device=hidden_states.device).index_add_(
        0, example_of_token, token_loss)
    return token_loss.mean(), loss_sum_per_example / has_label.sum(1).clamp(min=1)


def to_device(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def generate(model, tokenizer, problems, config, device, autocast):
    """Greedy decoding for a list of problems."""
    model.eval()
    tokenizer.padding_side = "left"
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_token_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        stop_token_ids.append(im_end_id)
    completions = []
    for start in range(0, len(problems), config.gen_batch_size):
        prompts = [data.format_prompt(tokenizer, problem) for problem in problems[start:start + config.gen_batch_size]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast:
            generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False,
                                       temperature=None, top_p=None, top_k=None, use_cache=True,
                                       eos_token_id=stop_token_ids, pad_token_id=tokenizer.pad_token_id)
        prompt_length = inputs["input_ids"].shape[1]
        completions += tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
    model.train()
    return completions


def accuracy(model, tokenizer, examples, config, device, autocast):
    """List of True/False: did the Student's final \\boxed{} answer match the reference?"""
    completions = generate(model, tokenizer, [example["problem"] for example in examples], config, device, autocast)
    return [data.is_correct(completion, example["answer"]) for completion, example in zip(completions, examples)]


def test_eval(model, tokenizer, test_examples, config, device, autocast):
    """Final-task performance on the test set (MATH-500): overall, per level, and per subject.
    Only for reporting; the Teacher never sees it."""
    correct = accuracy(model, tokenizer, test_examples, config, device, autocast)
    results = {"test_acc": float(np.mean(correct))}
    for group_by in ("level", "subject"):
        groups = {}
        for was_correct, example in zip(correct, test_examples):
            groups.setdefault(data.bucket_key(example, group_by), []).append(was_correct)
        results[f"test_acc_by_{group_by}"] = {name: float(np.mean(values)) for name, values in sorted(groups.items())}
    return results


@torch.no_grad()
def val_scores(model, tokenizer, val_examples, val_encoded, config, device, autocast):
    """The Teacher's observation o_t = (x^(1)_t, ..., x^(N)_t): one score per task on the validation set.

    Higher is better. The paper uses validation accuracy. That needs generation, which is slow for an
    LLM, so by default we use -validation loss (one forward pass). Set score="val_acc" to use accuracy.
    """
    if config.score == "val_acc":
        return np.array([np.mean(accuracy(model, tokenizer, bucket, config, device, autocast))
                         for bucket in val_examples])
    model.eval()
    scores = []
    for bucket in val_encoded:
        losses = []
        for start in range(0, len(bucket), config.batch_size):
            batch = to_device(data.collate(bucket[start:start + config.batch_size], tokenizer.pad_token_id), device)
            with autocast:
                _, example_losses = forward_loss(model, batch)
            losses += example_losses.tolist()
        scores.append(-float(np.mean(losses)))
    model.train()
    return np.array(scores)


def train_loss_scores(round_loss_sum, round_count, previous_scores):
    """Free alternative observation (score="train_loss"): -mean training loss of each task during
    the last round. A task that was not sampled this round keeps its previous score."""
    scores = np.full(len(round_count), np.nan)
    sampled = round_count > 0
    scores[sampled] = -round_loss_sum[sampled] / round_count[sampled]
    if previous_scores is not None:
        scores[~sampled] = previous_scores[~sampled]
    scores[np.isnan(scores)] = np.nanmean(scores)   # never seen yet: use the average of the others
    return scores


# ----------------------------------------------------------------------------- main: the Teacher-Student loop
def main():
    config = load_config()
    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "config.json"), "w") as config_file:
        json.dump(dataclasses.asdict(config), config_file, indent=2)
    log_file = open(os.path.join(config.out_dir, "log.jsonl"), "a")

    def log(record_type, **fields):
        log_file.write(json.dumps({"type": record_type, **fields}) + "\n")
        log_file.flush()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    # --- Student ---
    tokenizer, model = load_model(config, device)
    model.train()

    # --- Tasks: training data split into N buckets, plus a val set per bucket and a test set ---
    task_data = data.load_task_data(config.dataset, bucket_by=config.bucket_by,
                                    n_val=config.n_val_per_bucket, data_seed=config.data_seed)
    num_tasks = len(task_data.names)
    train_encoded = [[data.encode(tokenizer, example, config.max_len, task_id) for example in bucket]
                     for task_id, bucket in enumerate(task_data.train)]
    val_encoded = [[data.encode(tokenizer, example, config.max_len, task_id) for example in bucket]
                   for task_id, bucket in enumerate(task_data.val)]
    test_examples = task_data.test[:config.test_size] if config.test_size else task_data.test
    bucket_sizes = [len(bucket) for bucket in train_encoded]
    print(f"device={device} buckets={dict(zip(task_data.names, bucket_sizes))} "
          f"val/bucket={config.n_val_per_bucket} test={len(test_examples)}")

    # --- Teacher, plus the sampler that turns its distribution into actual batches ---
    sampler = data.BucketSampler(train_encoded, seed=config.seed)
    teacher = teachers.make_teacher(config.teacher, num_tasks, bucket_sizes, seed=config.seed, **config.teacher_kwargs)
    use_scores = teacher.needs_scores or config.always_score

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(config.warmup_ratio * config.total_steps),
                                                config.total_steps)

    # Budget counters. Compare runs at equal `examples` (or `tokens`), not equal wall-clock.
    step, teacher_step, examples, tokens, teacher_eval_examples = 0, 0, 0, 0, 0

    def counters():
        return dict(step=step, examples=examples, tokens=tokens, teacher_eval_examples=teacher_eval_examples)

    def run_test_eval():
        start_time = time.time()
        results = test_eval(model, tokenizer, test_examples, config, device, autocast)
        log("eval", **counters(), **results)
        print(f"[eval] step {step} test_acc={results['test_acc']:.4f} by_level={results['test_acc_by_level']} "
              f"({time.time() - start_time:.0f}s)")

    if config.eval_at_start:
        run_test_eval()

    # o_0: score every task once before training, so the first reward (a score change) can be computed.
    scores = None
    if use_scores and config.score != "train_loss":
        scores = val_scores(model, tokenizer, task_data.val, val_encoded, config, device, autocast)
        teacher_eval_examples += sum(len(bucket) for bucket in task_data.val)
        teacher.update(scores)
        log("round", round=0, **counters(), scores=scores.tolist(), names=task_data.names)

    train_start_time = time.time()
    while step < config.total_steps:
        # ===== OUTER LOOP: one Teacher timestep t =====
        # Action a_t: the Teacher picks a probability distribution over the N tasks.
        task_probs = teacher.distribution()

        # Per-task training loss during this round (logged, and used when score="train_loss").
        round_loss_sum, round_count = np.zeros(num_tasks), np.zeros(num_tasks)

        # ===== INNER LOOP: the Student trains for `teacher_interval` optimizer steps on batches drawn from a_t =====
        for _ in range(min(config.teacher_interval, config.total_steps - step)):
            # Sample which task each example comes from (according to a_t), then take examples from those buckets.
            batch = to_device(data.collate(sampler.sample(task_probs, config.batch_size), tokenizer.pad_token_id),
                              device)

            # From here on it is a normal finetuning step.
            with autocast:
                loss, example_losses = forward_loss(model, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Bookkeeping: per-task training loss for this round.
            task_ids = batch["bucket"].cpu().numpy()
            np.add.at(round_loss_sum, task_ids, example_losses.detach().float().cpu().numpy())
            np.add.at(round_count, task_ids, 1)

            step += 1
            examples += config.batch_size
            tokens += int((batch["labels"] != -100).sum())
            if step % config.log_every == 0:
                learning_rate = scheduler.get_last_lr()[0]
                log("train", **counters(), loss=loss.item(), grad_norm=float(grad_norm), lr=learning_rate)
                seconds_per_step = (time.time() - train_start_time) / step
                print(f"step {step}/{config.total_steps} loss={loss.item():.4f} lr={learning_rate:.2e} "
                      f"task_probs={np.round(task_probs, 3).tolist()} ({seconds_per_step:.2f}s/step)")
            # Test evals happen at fixed optimizer steps, so every teacher is measured at the same budget.
            if step % config.eval_every == 0 or step == config.total_steps:
                run_test_eval()

        # ===== Back in the OUTER LOOP: the Teacher observes o_t and updates =====
        # (Inside teacher.update: reward = score change / learning-curve slope, then the policy is updated.)
        teacher_step += 1
        if use_scores:
            if config.score == "train_loss":
                scores = train_loss_scores(round_loss_sum, round_count, scores)
            else:
                scores = val_scores(model, tokenizer, task_data.val, val_encoded, config, device, autocast)
                teacher_eval_examples += sum(len(bucket) for bucket in task_data.val)
            teacher.update(scores)
        log("round", round=teacher_step, **counters(), probs=task_probs.tolist(), names=task_data.names,
            scores=None if scores is None else scores.tolist(),
            drawn=sampler.drawn.tolist(), unique_seen=[len(seen) for seen in sampler.seen], **teacher.state())

    if config.save_model:
        model.save_pretrained(os.path.join(config.out_dir, "model"))
        tokenizer.save_pretrained(os.path.join(config.out_dir, "model"))
    log_file.close()


if __name__ == "__main__":
    main()
