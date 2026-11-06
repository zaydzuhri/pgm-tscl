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


def bucket_key(ex, bucket_by):
    if bucket_by == "level":
        return f"L{ex['level']}"
    if bucket_by == "subject":
        return ex["subject"]
    if bucket_by == "level_subject":
        return f"{ex['subject']}/L{ex['level']}"
    if bucket_by == "none":
        return "all"
    raise ValueError(f"unknown bucket_by={bucket_by}")


def split_into_buckets(rows, test, bucket_by, n_val, data_seed=0):
    groups = {}
    for r in rows:
        groups.setdefault(bucket_key(r, bucket_by), []).append(r)
    names = sorted(groups)
    rng = random.Random(data_seed)
    train, val = [], []
    for k in names:
        g = groups[k][:]
        rng.shuffle(g)
        val.append(g[:n_val])
        train.append(g[n_val:])
    return TaskData(names, train, val, test)


# ----------------------------------------------------------------------------- datasets
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def load_math(bucket_by="level", n_val=32, data_seed=0):
    """Hendrycks MATH train split (7.5k problems) bucketed by level and/or subject; MATH-500 as test set."""
    rows = []
    for subj in MATH_SUBJECTS:
        for r in load_dataset("EleutherAI/hendrycks_math", subj, split="train"):
            lvl = r["level"].replace("Level ", "")
            ans = last_boxed(r["solution"])
            if not lvl.isdigit() or ans is None:  # a handful are "Level ?" or have no \boxed answer
                continue
            rows.append(dict(problem=r["problem"], solution=r["solution"], answer=ans,
                             level=int(lvl), subject=r["type"]))
    test = [dict(problem=r["problem"], solution=r["solution"], answer=r["answer"],
                 level=int(r["level"]), subject=r["subject"])
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    return split_into_buckets(rows, test, bucket_by, n_val, data_seed)


DATASETS = {"math": load_math}


def load_task_data(name, **kw) -> TaskData:
    return DATASETS[name](**kw)


# ----------------------------------------------------------------------------- tokenization
def format_prompt(tok, problem):
    msgs = [{"role": "user", "content": problem + PROMPT_SUFFIX}]
    if tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return f"Problem: {problem}{PROMPT_SUFFIX}\nSolution: "


def encode(tok, ex, max_len, bucket):
    """Prompt tokens are masked out of the loss (-100); only the solution is trained on."""
    p = tok(format_prompt(tok, ex["problem"]), add_special_tokens=False).input_ids
    r = tok(ex["solution"] + tok.eos_token, add_special_tokens=False).input_ids
    return dict(input_ids=(p + r)[:max_len], labels=([-100] * len(p) + r)[:max_len], bucket=bucket)


def collate(exs, pad_id):
    L = max(len(e["input_ids"]) for e in exs)
    ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
    labels = torch.full((len(exs), L), -100, dtype=torch.long)
    mask = torch.zeros((len(exs), L), dtype=torch.long)
    for i, e in enumerate(exs):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        mask[i, :n] = 1
    return dict(input_ids=ids, labels=labels, attention_mask=mask,
                bucket=torch.tensor([e["bucket"] for e in exs]))


class BucketSampler:
    """Draws training examples given the teacher's distribution over buckets.

    Each bucket is its own shuffled queue (sampling without replacement, reshuffled when it runs
    out), so with the `proportional` teacher this matches ordinary shuffled finetuning.
    """

    def __init__(self, buckets, seed=0):
        self.buckets = buckets
        self.rng = np.random.default_rng(seed)
        self.queues = [[] for _ in buckets]
        self.drawn = np.zeros(len(buckets), dtype=np.int64)   # total draws per bucket
        self.seen = [set() for _ in buckets]                   # unique examples per bucket

    def _next(self, i):
        if not self.queues[i]:
            self.queues[i] = self.rng.permutation(len(self.buckets[i])).tolist()
        j = self.queues[i].pop()
        self.seen[i].add(j)
        return self.buckets[i][j]

    def sample(self, probs, n):
        p = np.asarray(probs, dtype=np.float64)
        counts = self.rng.multinomial(n, p / p.sum())
        out = [self._next(i) for i, c in enumerate(counts) for _ in range(c)]
        self.rng.shuffle(out)
        self.drawn += counts
        return out


# ----------------------------------------------------------------------------- answer checking
def last_boxed(s):
    """Return the contents of the last \\boxed{...} (or \\fbox{...}) in s, or None."""
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j]
    return None


def _norm(s):
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\(?:text|mbox|mathrm)\{(.*?)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    return re.sub(r"^[a-zA-Z]=", "", s)


def is_correct(completion, gold):
    pred = last_boxed(completion)
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False
