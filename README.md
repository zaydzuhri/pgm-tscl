# TSCL for LLM finetuning

Teacher-Student Curriculum Learning ([Matiisen et al., 2017](1707.00183v2.pdf)) applied to supervised finetuning of
Qwen3-0.6B on Hendrycks MATH. The **Student** is the LLM. The **tasks** are difficulty buckets of MATH. The
**Teacher** decides what fraction of each training batch comes from each bucket.

## Setup and running

```bash
uv sync
uv run train.py --config configs/sft.json            # ordinary finetuning (baseline)
uv run train.py --config configs/tscl_window.json    # TSCL, Window teacher
uv run train.py --config configs/tscl_window.json --seed 1 --teacher_interval 20   # CLI flags override the JSON
```

Logs go to `runs/<config>_s<seed>/log.jsonl`, with three record types:
- `train`: loss and LR
- `round`: the teacher's distribution, the per-bucket scores, and its Q values
- `eval`: MATH-500 accuracy, overall and per level/subject

## Files

| file | what it holds | how to extend |
|---|---|---|
| `train.py` | config, model loading, evaluation, the Teacher–Student loop | change `model` in the config |
| `data.py` | dataset → buckets, tokenization, `BucketSampler`, answer checking | add a loader to `DATASETS`, or change `bucket_by` |
| `teachers.py` | all teachers (baselines + the paper's Online / Naive / Window / Sampling) | subclass `Teacher` and add it to `TEACHERS` |

## How TSCL maps onto finetuning

We use the paper's **batch POMDP** (Sec 2.2, Appendix B), which is the formulation for supervised learning:

| paper | here |
|---|---|
| task *i* | a MATH bucket (`bucket_by`: `level` → 5 tasks, `subject` → 7, `level_subject` → 35) |
| action *a_t* | a distribution over buckets; the next `teacher_interval` optimizer steps sample batches from it |
| observation *o_t* | per-bucket score on a held-out val split carved out of MATH train (`n_val_per_bucket` per bucket) |
| reward *r_t* | score change *o_t − o_{t−1}* (Online, Sampling) or learning-curve slope (Naive, Window) |
| final task | MATH-500 accuracy (reporting only; the teacher never sees it) |

This makes training a doubly nested loop. The outer loop is the teacher's timesteps. The inner loop is
`teacher_interval` ordinary optimizer steps.

**Score choice (`score`)**: the paper uses validation accuracy. For an LLM that needs generation, which is
slow, so the default is `val_loss` (−NLL of the reference solutions, one forward pass). `val_acc` is faithful
to the paper but much slower. `train_loss` costs nothing extra but is noisier.

## Fair comparison

Ordinary finetuning is just the `proportional` teacher (bucket probability ∝ bucket size, so it matches
shuffled SFT). It therefore runs through exactly the same loop. To compare runs, keep these identical:
`total_steps`, `batch_size`, `lr`, `warmup_ratio`, `eval_every`, `data_seed`.
Every run then trains on the same number of examples, uses the same LR schedule, and is evaluated at the same steps.
`teacher_interval` only affects teachers that observe scores.

Each log record carries `examples`, `tokens` and `teacher_eval_examples`, so you can plot test accuracy against
training data used. Note that equal steps does not mean equal tokens: Level 5 solutions are longer.

## Teachers (`teacher`, `teacher_kwargs`)

| name | kwargs (defaults) |
|---|---|
| `proportional`, `uniform` | — |
| `online` | `alpha=0.1, policy="greedy"\|"boltzmann", eps=0.1, tau=4e-4, use_abs=True` |
| `naive` | same as online + `K=5` (hold one distribution for K rounds) |
| `window` | same as online + `K=10` (FIFO window of scores) |
| `sampling` | `K=10, eps=0.1` |

Boltzmann's `tau` depends on the scale of the scores. The paper's 4e-4 was tuned for accuracy and may not suit
`val_loss`.
