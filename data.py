"""Data: loading datasets into difficulty buckets, tokenization, the bucket sampler, and answer checking.

A dataset loader returns a TaskData: training examples grouped into named buckets (the TSCL
"tasks"), a small held-out validation set per bucket (used only for the teacher's observations),
and a test set (used only for reporting). To add a dataset, write a loader and register it in DATASETS.
"""
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


@dataclass
class TaskData:
    names: list   # bucket names, length N
    train: list   # N lists of examples
    val: list     # N lists of examples, disjoint from train
    test: list    # flat list of examples


def bucket_key(example, bucket_by):
    if bucket_by == "level":
        return f"L{example['level']}"
    if bucket_by == "subject":
        return example["subject"]
    if bucket_by == "level_subject":
        return f"{example['subject']}/L{example['level']}"
    if bucket_by == "none":
        return "all"
    raise ValueError(f"unknown bucket_by={bucket_by}")


def split_into_buckets(train_rows, test_rows, bucket_by, n_val, data_seed=0):
    groups = {}
    for row in train_rows:
        groups.setdefault(bucket_key(row, bucket_by), []).append(row)
    names = sorted(groups)
    random_generator = random.Random(data_seed)
    train, val = [], []
    for name in names:
        group = groups[name][:]
        random_generator.shuffle(group)
        val.append(group[:n_val])
        train.append(group[n_val:])
    return TaskData(names, train, val, test_rows)


# ----------------------------------------------------------------------------- datasets
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def load_math(bucket_by="level", n_val=32, data_seed=0):
    """Hendrycks MATH train split (7.5k problems) bucketed by level and/or subject; MATH-500 as test set."""
    train_rows = []
    for subject in MATH_SUBJECTS:
        for row in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
            level = row["level"].replace("Level ", "")
            answer = last_boxed(row["solution"])
            if not level.isdigit() or answer is None:  # a handful are "Level ?" or have no \boxed answer
                continue
            train_rows.append(dict(problem=row["problem"], solution=row["solution"], answer=answer,
                                   level=int(level), subject=row["type"]))
    test_rows = [dict(problem=row["problem"], solution=row["solution"], answer=row["answer"],
                      level=int(row["level"]), subject=row["subject"])
                 for row in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    return split_into_buckets(train_rows, test_rows, bucket_by, n_val, data_seed)


DATASETS = {"math": load_math}


def load_task_data(name, **kwargs) -> TaskData:
    return DATASETS[name](**kwargs)


# ----------------------------------------------------------------------------- tokenization
def format_prompt(problem):
    """Plain-text prompt for a base model; the model continues with the solution."""
    return f"Problem: {problem}{PROMPT_SUFFIX}\nSolution: "


def encode(tokenizer, example, max_len, bucket):
    """Prompt tokens are masked out of the loss (-100); only the solution is trained on."""
    prompt_ids = tokenizer(format_prompt(example["problem"]), add_special_tokens=False).input_ids
    solution_ids = tokenizer(example["solution"] + tokenizer.eos_token, add_special_tokens=False).input_ids
    return dict(input_ids=(prompt_ids + solution_ids)[:max_len],
                labels=([-100] * len(prompt_ids) + solution_ids)[:max_len],
                bucket=bucket)


def collate(examples, pad_token_id):
    """Right-pad a list of encoded examples into a batch of tensors.

    Shapes: B = len(examples), L = length of the longest example in the batch.
    Output: input_ids, labels, attention_mask: (B, L) long;  bucket: (B,) long (task id of each example)
    """
    max_length = max(len(example["input_ids"]) for example in examples)                   # L
    input_ids = torch.full((len(examples), max_length), pad_token_id, dtype=torch.long)  # (B, L)
    labels = torch.full((len(examples), max_length), -100, dtype=torch.long)             # (B, L)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)          # (B, L)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(example["input_ids"])
        labels[row, :length] = torch.tensor(example["labels"])
        attention_mask[row, :length] = 1
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                bucket=torch.tensor([example["bucket"] for example in examples]))


class BucketSampler:
    """Draws training examples given the teacher's distribution over buckets.

    Each bucket is its own shuffled queue (sampling without replacement, reshuffled when it runs
    out), so with the `proportional` teacher this matches ordinary shuffled finetuning.
    """

    def __init__(self, buckets, seed=0):
        self.buckets = buckets
        self.random_generator = np.random.default_rng(seed)
        self.queues = [[] for _ in buckets]
        self.drawn = np.zeros(len(buckets), dtype=np.int64)    # total draws per bucket
        self.seen = [set() for _ in buckets]                   # unique examples seen per bucket
        self.refills = np.zeros(len(buckets), dtype=np.int64)  # times each bucket ran out and started repeating

    def _next_example(self, bucket):
        if not self.queues[bucket]:  # queue empty: fill it with a fresh shuffled ordering of the bucket
            if self.seen[bucket]:    # not the first fill, so every example has been used once: repetition starts
                self.refills[bucket] += 1
            self.queues[bucket] = self.random_generator.permutation(len(self.buckets[bucket])).tolist()
        index = self.queues[bucket].pop()
        self.seen[bucket].add(index)
        return self.buckets[bucket][index]

    def sample(self, probs, batch_size):
        """Choose a bucket for each example according to `probs`, then take examples from those buckets."""
        probs = np.asarray(probs, dtype=np.float64)                                  # (N,)
        counts = self.random_generator.multinomial(batch_size, probs / probs.sum())  # (N,) examples per bucket, sums to batch_size
        batch = [self._next_example(bucket) for bucket, count in enumerate(counts) for _ in range(count)]
        self.random_generator.shuffle(batch)
        self.drawn += counts
        return batch


# ----------------------------------------------------------------------------- answer checking
def last_boxed(text):
    """Return the contents of the last \\boxed{...} (or \\fbox{...}) in text, or None."""
    box_start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if box_start < 0:
        return None
    open_brace = text.find("{", box_start)
    if open_brace < 0:
        return None
    depth = 0
    for position in range(open_brace, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[open_brace + 1:position]
    return None


def _normalize_answer(answer):
    for old, new in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                     ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                     ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        answer = answer.replace(old, new)
    answer = re.sub(r"\\(?:text|mbox|mathrm)\{(.*?)\}", r"\1", answer)
    answer = answer.replace(" ", "").rstrip(".")
    return re.sub(r"^[a-zA-Z]=", "", answer)


def is_correct(completion, gold_answer):
    """Exact match after normalization, falling back to math_verify's symbolic equivalence check."""
    predicted_answer = last_boxed(completion)
    if predicted_answer is None:
        return False
    if _normalize_answer(predicted_answer) == _normalize_answer(gold_answer):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse("\\boxed{" + gold_answer + "}"), parse("\\boxed{" + predicted_answer + "}")))
    except Exception:
        return False
