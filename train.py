"""Finetune a causal LM (the Student) while a Teacher chooses which difficulty buckets each batch comes from.

    uv run train.py --config configs/sft.json
    uv run train.py --config configs/tscl_window.json --seed 1 --total_train_steps 1000   # CLI overrides JSON

We follow the *batch POMDP* formulation of TSCL (Matiisen et al. 2017, Sec 2.2), since this is
supervised learning:

  - tasks       = difficulty buckets of MATH (e.g. Level 1..5)
  - action a_t  = a probability distribution over the N buckets; training batches are sampled from it
œ  - observation = per-bucket accuracies o_t = (x^(1)_t, ..., x^(N)_t) on a held-out validation set
                  that contains every bucket
  - reward      = how the scores change (the slope of the learning curve), worked out inside each Teacher

Training is one ordinary loop over optimizer steps; the Teacher acts every `teacher_every` steps:

    o_0 = observe(student) ; teacher.update(o_0)            # scores before any training
    a = teacher.distribution()
    for step in 1..total_train_steps:
        batch ~ a ; one optimizer step on the student
        if step % teacher_every == 0:                       # one Teacher timestep t
            o_t = observe(student) ; teacher.update(o_t)    # teacher learns from the score change
            a = teacher.distribution()                      # action a_t for the next teacher_every steps

Teachers that never look at scores (proportional, uniform) skip the observation entirely, so they
cost exactly as much as plain finetuning.

Fair comparison: every run, whatever its teacher, uses the same number of training examples
(total_train_steps x batch_size), the same LR schedule and the same test-eval steps. Only the mix of
buckets in each batch differs. Ordinary finetuning is just the `proportional` teacher, which never
changes its distribution. The teacher's extra cost (val generations) is logged separately as
`teacher_eval_examples`.
"""
import argparse
import copy
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
    model: str = "Qwen/Qwen3-0.6B-Base"
    dataset: str = "math"
    bucket_by: str = "level"            # how to split data into tasks: level | subject | level_subject | none
    n_val_per_bucket: int = 32          # held-out examples per task; their accuracy is what the Teacher observes
    max_len: int = 2048
    # --- Teacher ---
    teacher: str = "proportional"       # see teachers.TEACHERS; "proportional" = ordinary finetuning
    teacher_kwargs: dict = field(default_factory=dict)   # e.g. {"K": 10, "alpha": 0.1, "epsilon": 0.1}
    teacher_every: int = 10             # optimizer steps between Teacher timesteps (observe o_t, pick a new a_t)
    always_score: bool = False          # also score baselines every teacher_every steps (only for plotting learning curves)
    # --- Training budget (keep these identical across runs you compare) ---
    total_train_steps: int = 500        # optimizer steps in the whole run
    batch_size: int = 32                # examples per optimizer step (use grad_ckpt if memory is tight)
    lr: float = 2e-5                    # TRL SFT default; Qwen3-0.6B full-FT model cards use 2e-5 to 4e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    grad_ckpt: bool = False
    # --- Evaluation / logging ---
    eval_every: int = 100               # optimizer steps between accuracy evals on test, val and a train subset (same for every teacher)
    n_train_acc_per_bucket: int = 32    # training examples per task whose accuracy is also measured at each eval (0 = skip)
    val_loss_every: int = 50            # optimizer steps between validation-loss measurements (0 = never)
    eval_at_start: bool = True
    test_size: int = 500                # 0 = full test set
    gen_batch_size: int = 64
    max_new_tokens: int = 1024
    log_every: int = 10
    seed: int = 0
    data_seed: int = 0                  # fixes the train/val split, so all runs see the same data
    out_dir: str = ""                   # default: runs/<config-name>_s<seed>
    save_model: bool = False
    wandb_project: str = "pgm-tscl"     # empty string = no wandb logging (log.jsonl is always written)
    wandb_entity: str = ""              # empty = your default wandb entity


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
    # Standard bf16 mixed precision (what HF Trainer's bf16=True does): weights, gradients and AdamW state
    # stay in fp32, and the forward/backward pass runs in bf16 autocast (see main). Loading in fp32 must be
    # explicit, because transformers otherwise loads Qwen3 in its checkpoint dtype (bf16).
    model = AutoModelForCausalLM.from_pretrained(config.model, dtype=torch.float32).to(device)
    if config.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return tokenizer, model


def forward_loss(model, batch):
    """Mean next-token cross-entropy over the solution tokens (prompt and padding are labelled -100).
    Same loss as model(**batch).loss, but logits are only computed where there is a label, which
    saves a lot of memory with Qwen's 150k-token vocabulary.

    Shapes: B = batch size, L = padded sequence length, D = hidden size, V = vocab size,
            T = number of labelled (solution) tokens in the whole batch.
    Input:  batch["input_ids"], batch["attention_mask"], batch["labels"]: (B, L)
    Output: scalar loss tensor ()
    """
    hidden_states = model.base_model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"]).last_hidden_state  # (B, L, D)
    labels = batch["labels"][:, 1:]               # (B, L-1)  token t predicts token t+1
    has_label = labels != -100                    # (B, L-1)  bool, True on the T solution tokens
    selected_hidden = hidden_states[:, :-1][has_label]                          # (T, D)
    logits = model.get_output_embeddings()(selected_hidden).float()             # (T, V)  upcast for the loss, as HF does
    return F.cross_entropy(logits, labels[has_label])                           # (T, V), (T,) -> ()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_accuracy(model, tokenizer, examples, config, device):
    """Greedy-decode each problem and check its final \\boxed{} answer against the reference.
    Returns a list of True/False, one per example.

    Generation uses a bf16 copy of the weights (like HF's bf16_full_eval); the fp32 training weights are untouched.

    Shapes (per generation batch): B = up to gen_batch_size, P = padded prompt length,
                                   G = number of generated tokens (<= max_new_tokens)."""
    generation_model = copy.deepcopy(model).to(torch.bfloat16).eval()
    tokenizer.padding_side = "left"

    correct = []
    for start in range(0, len(examples), config.gen_batch_size):
        chunk = examples[start:start + config.gen_batch_size]
        prompts = [data.format_prompt(example["problem"]) for example in chunk]
        # inputs["input_ids"], inputs["attention_mask"]: (B, P), left-padded so all prompts end at the same position
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        generated = generation_model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False,
                                              temperature=None, top_p=None, top_k=None, use_cache=True,
                                              eos_token_id=tokenizer.eos_token_id,
                                              pad_token_id=tokenizer.pad_token_id)                   # (B, P + G)
        prompt_length = inputs["input_ids"].shape[1]                                                       # P
        completions = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)       # (B, G) -> B strings
        correct += [data.is_correct(completion, example["answer"]) for completion, example in zip(completions, chunk)]
    del generation_model
    return correct


@torch.no_grad()
def evaluate_loss(model, tokenizer, encoded_buckets, config, device, autocast):
    """Loss on held-out examples: the same mean next-token cross-entropy over solution tokens as the
    training loss (teacher forcing, no generation, so it is cheap).
    Input:  encoded_buckets: N lists of encoded examples (one list per task)
    Output: (per-task loss (N,), overall loss ()), each a mean over solution tokens.
            A task with no usable examples gets NaN."""
    model.eval()
    loss_sums = np.zeros(len(encoded_buckets))        # (N,) summed token losses per task
    token_counts = np.zeros(len(encoded_buckets))     # (N,) number of solution tokens per task
    for task_id, bucket in enumerate(encoded_buckets):
        for start in range(0, len(bucket), config.batch_size):
            batch = data.collate(bucket[start:start + config.batch_size], tokenizer.pad_token_id)
            batch = {key: tensor.to(device) for key, tensor in batch.items()}    # input_ids, attention_mask, labels: (B, L)
            num_tokens = int((batch["labels"] != -100).sum())                    # T, solution tokens in this batch
            with autocast:
                loss = forward_loss(model, batch)                                # () mean over the T tokens
            loss_sums[task_id] += loss.item() * num_tokens
            token_counts[task_id] += num_tokens
    model.train()
    per_task_loss = np.where(token_counts > 0, loss_sums / np.maximum(token_counts, 1), np.nan)   # (N,)
    return per_task_loss, loss_sums.sum() / token_counts.sum()


# ----------------------------------------------------------------------------- main: the Teacher-Student loop
def main():
    config = load_config()
    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "config.json"), "w") as config_file:
        json.dump(dataclasses.asdict(config), config_file, indent=2)
    log_file = open(os.path.join(config.out_dir, "log.jsonl"), "w")   # a re-run of the same config starts a fresh log
    wandb_run = None
    if config.wandb_project:
        import wandb
        wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity or None,
                               name=os.path.basename(os.path.normpath(config.out_dir)),
                               config=dataclasses.asdict(config), dir=config.out_dir)

    def log(record_type, **fields):
        log_file.write(json.dumps({"type": record_type, **fields}) + "\n")
        log_file.flush()
        if wandb_run is None:
            return
        # wandb wants flat numbers, so per-bucket lists and dicts become one metric per bucket,
        # e.g. probs=[...] in a "round" record -> round/probs/L1, round/probs/L2, ...
        # The budget counters (examples, tokens, teacher_eval_examples) stay unprefixed so any of
        # them can be picked as the x-axis in the wandb UI.
        metrics = {}
        for key, value in fields.items():
            if key in ("step", "round", "names") or value is None:
                continue
            if key in ("examples", "tokens", "teacher_eval_examples"):
                metrics[key] = value
            elif isinstance(value, dict):
                metrics.update({f"{record_type}/{key}/{name}": number for name, number in value.items()})
            elif isinstance(value, list):
                metrics.update({f"{record_type}/{key}/{name}": number for name, number in zip(task_data.names, value)})
            else:
                metrics[f"{record_type}/{key}"] = value
        wandb_run.log(metrics, step=fields["step"])

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

    def encode_buckets(buckets):
        """Tokenize every example of every bucket. Examples whose solution was cut off entirely by max_len
        are dropped: they have no tokens to compute a loss on, and a batch made only of them would give NaN."""
        encoded_buckets = [[data.encode(tokenizer, example, config.max_len, task_id) for example in bucket]
                           for task_id, bucket in enumerate(buckets)]
        return [[encoded for encoded in bucket if any(label != -100 for label in encoded["labels"])]
                for bucket in encoded_buckets]

    # The training set is too large to tokenize up front, so the sampler tokenizes each example when it is
    # drawn. Examples whose prompt alone fills max_len are dropped here instead (no solution tokens to train
    # on; a batch made only of them would give NaN). Prompts are shared by many solutions, so only the unique
    # problems are tokenized.
    prompt_lengths = {problem: len(tokenizer(data.format_prompt(problem), add_special_tokens=False).input_ids)
                      for problem in {example["problem"] for bucket in task_data.train for example in bucket}}
    train_buckets = [[example for example in bucket if prompt_lengths[example["problem"]] < config.max_len]
                     for bucket in task_data.train]
    val_encoded = encode_buckets(task_data.val)       # for the validation loss
    test_examples = task_data.test[:config.test_size] if config.test_size else task_data.test
    # All validation examples in one flat list, so observing the scores is one batched generation over
    # every task instead of one per task. val_task_ids[k] is the task of val_examples[k].
    val_examples = [example for bucket in task_data.val for example in bucket]
    val_task_ids = np.array([task_id for task_id, bucket in enumerate(task_data.val) for _ in bucket])  # (N * n_val,)
    # A fixed subset of the training examples, to compare accuracy on data the Student trains on against
    # held-out data (val, test). Buckets are already shuffled by data_seed, so the first ones are a random subset.
    train_acc_examples = [example for bucket in task_data.train for example in bucket[:config.n_train_acc_per_bucket]]
    train_acc_task_ids = np.array([task_id for task_id, bucket in enumerate(task_data.train)
                                   for _ in bucket[:config.n_train_acc_per_bucket]])                       # (N * n_train_acc,)
    bucket_sizes = [len(bucket) for bucket in train_buckets]
    print(f"device={device} buckets={dict(zip(task_data.names, bucket_sizes))} "
          f"val/bucket={config.n_val_per_bucket} test={len(test_examples)}")

    # --- Teacher, plus the sampler that turns its distribution into actual batches ---
    sampler = data.BucketSampler(train_buckets, seed=config.seed,
                                 encode=lambda example, task_id: data.encode(tokenizer, example, config.max_len, task_id))
    teacher = teachers.make_teacher(config.teacher, num_tasks, bucket_sizes, seed=config.seed, **config.teacher_kwargs)
    use_scores = teacher.needs_scores or config.always_score

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    num_warmup_steps = math.ceil(config.warmup_ratio * config.total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, config.total_train_steps)

    # Budget counters. Compare runs at equal `examples` (or `tokens`), not equal wall-clock.
    step, teacher_step, examples, tokens, teacher_eval_examples = 0, 0, 0, 0, 0

    def counters():
        return dict(step=step, examples=examples, tokens=tokens, teacher_eval_examples=teacher_eval_examples)

    def per_task_accuracy(correct, task_ids):
        """(K,) bool correctness and (K,) task ids -> (N,) accuracy per task (NaN for a task with no examples)."""
        return [float(correct[task_ids == task_id].mean()) if (task_ids == task_id).any() else float("nan")
                for task_id in range(num_tasks)]

    def run_eval():
        """Accuracy (greedy decoding) on three sets, in one batched generation:
          - test (MATH-500): the final task, overall and per level / subject. The Teacher never sees it.
          - val: the held-out examples the Teacher observes, per task.
          - train: a fixed subset of the training examples, per task. A gap between train and val/test
            accuracy shows how much the Student is fitting its training data instead of generalizing."""
        start_time = time.time()
        eval_examples = test_examples + val_examples + train_acc_examples
        print(f"[eval] step {step}: generating answers for {len(test_examples)} test + {len(val_examples)} val + "
              f"{len(train_acc_examples)} train problems (up to {config.max_new_tokens} new tokens each)...")
        correct = np.array(evaluate_accuracy(model, tokenizer, eval_examples, config, device))   # (K,) bool
        test_correct = correct[:len(test_examples)]
        val_correct = correct[len(test_examples):len(test_examples) + len(val_examples)]
        train_correct = correct[len(test_examples) + len(val_examples):]
        results = {"test_acc": float(test_correct.mean())}
        for group_by in ("level", "subject"):
            groups = {}
            for was_correct, example in zip(test_correct, test_examples):
                groups.setdefault(data.bucket_key(example, group_by), []).append(was_correct)
            results[f"test_acc_by_{group_by}"] = {name: float(np.mean(values)) for name, values in sorted(groups.items())}
        results["val_acc"] = float(val_correct.mean())
        results["val_acc_by_bucket"] = per_task_accuracy(val_correct, val_task_ids)
        if len(train_correct):
            results["train_acc"] = float(train_correct.mean())
            results["train_acc_by_bucket"] = per_task_accuracy(train_correct, train_acc_task_ids)
        log("eval", **counters(), **results)
        print(f"[eval] step {step} test_acc={results['test_acc']:.4f} val_acc={results['val_acc']:.4f} "
              f"train_acc={results.get('train_acc', float('nan')):.4f} by_level={results['test_acc_by_level']} "
              f"({time.time() - start_time:.0f}s)")

    def run_val_loss():
        """Loss on the held-out validation examples, overall and per task. These are the same examples the
        Teacher's observation is measured on (as accuracy), and they are never trained on."""
        start_time = time.time()
        per_task_loss, overall_loss = evaluate_loss(model, tokenizer, val_encoded, config, device, autocast)
        log("val", **counters(), loss=float(overall_loss), loss_by_bucket=per_task_loss.tolist())
        print(f"[val] step {step} loss={overall_loss:.4f} by_bucket={np.round(per_task_loss, 4).tolist()} "
              f"({time.time() - start_time:.0f}s)")

    def observe_and_update_teacher():
        """One Teacher timestep: observe o_t (per-task val accuracy), let the Teacher update on it
        (inside teacher.update: reward = score change / learning-curve slope, then the policy update),
        and log it. Only called for teachers that use scores (or with always_score)."""
        nonlocal teacher_eval_examples
        start_time = time.time()
        val_correct = np.array(evaluate_accuracy(model, tokenizer, val_examples, config, device))  # (N * n_val,) bool
        scores = np.array([val_correct[val_task_ids == task_id].mean() for task_id in range(num_tasks)])  # (N,) one accuracy per task
        teacher_eval_examples += len(val_examples)
        print(f"[teacher] round {teacher_step} scores={np.round(scores, 3).tolist()} ({time.time() - start_time:.0f}s)")
        teacher.update(scores)
        log("round", round=teacher_step, **counters(), scores=scores.tolist(), names=task_data.names, **teacher.state())

    if config.eval_at_start:
        run_eval()
    if config.val_loss_every:
        run_val_loss()                  # validation loss before any training

    # o_0: score every task once before training, so the first reward (a score change) can be computed.
    # The score of task i is the Student's accuracy on that task's validation examples.
    if use_scores:
        observe_and_update_teacher()
    # Action a_0: the Teacher's first distribution over the N tasks. It is kept until the next Teacher timestep.
    task_probs = teacher.distribution()                                         # (N,) sums to 1

    train_start_time = time.time()
    for step in range(1, config.total_train_steps + 1):
        # Sample which task each example comes from (according to the current action), then take examples
        # from those buckets.  batch: input_ids, attention_mask, labels: (B, L); bucket: (B,)   with B = batch_size
        batch = data.collate(sampler.sample(task_probs, config.batch_size), tokenizer.pad_token_id)
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        # From here on it is a normal finetuning step.
        with autocast:
            loss = forward_loss(model, batch)                                   # () scalar
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        examples += config.batch_size
        tokens += int((batch["labels"] != -100).sum())                          # (B, L) bool -> count of solution tokens
        if step % config.log_every == 0:
            learning_rate = scheduler.get_last_lr()[0]
            # probs = the action in use; drawn / unique_seen / refills = per-bucket data usage so far
            log("train", **counters(), loss=loss.item(), grad_norm=float(grad_norm), lr=learning_rate,
                probs=task_probs.tolist(), drawn=sampler.drawn.tolist(),
                unique_seen=[len(seen) for seen in sampler.seen], refills=sampler.refills.tolist())
            seconds_per_step = (time.time() - train_start_time) / step
            print(f"step {step}/{config.total_train_steps} loss={loss.item():.4f} lr={learning_rate:.2e} "
                  f"task_probs={np.round(task_probs, 3).tolist()} ({seconds_per_step:.2f}s/step)")
        if config.val_loss_every and (step % config.val_loss_every == 0 or step == config.total_train_steps):
            run_val_loss() # log held out validation loss
        # Accuracy evals happen at fixed optimizer steps, so every teacher is measured at the same budget.
        if step % config.eval_every == 0 or step == config.total_train_steps:
            run_eval()

        # Teacher timestep t: observe o_t, update, and choose the action a_t for the next teacher_every steps.
        # Skipped after the last step, since no training is left to use a new action.
        if step % config.teacher_every == 0 and step < config.total_train_steps:
            teacher_step += 1
            if use_scores:
                observe_and_update_teacher()
            task_probs = teacher.distribution()                                 # (N,) sums to 1

    if config.save_model:
        model.save_pretrained(os.path.join(config.out_dir, "model"))
        tokenizer.save_pretrained(os.path.join(config.out_dir, "model"))
    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
