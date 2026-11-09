# TSCL for LLM finetuning

Teacher-Student Curriculum Learning ([Matiisen et al., 2017](1707.00183v2.pdf)) applied to supervised finetuning of
Qwen3-0.6B-Base on Hendrycks MATH. The **Student** is the LLM. The **tasks** are difficulty buckets of MATH. The
**Teacher** decides what fraction of each training batch comes from each bucket.

The model is a base (pretrained-only) model, which starts at about 32% on MATH-500. The prompt is plain text:
`Problem: <problem>\nPlease reason step by step, and put your final answer within \boxed{}.\nSolution: `.
Training uses standard bf16 mixed precision (fp32 weights and AdamW state, bf16 autocast), the same as HF Trainer's
`bf16=True`. Evals generate with a bf16 copy of the weights.

## Setup and running

```bash
uv sync
uv run train.py --config configs/sft.json            # ordinary finetuning (baseline)
uv run train.py --config configs/tscl_window.json    # TSCL, Window teacher
uv run train.py --config configs/tscl_window.json --seed 1 --teacher_interval 20   # CLI flags override the JSON
```

On the Mila cluster, submit through `run_job.sh`, which passes its arguments to `uv run` and keeps the
HuggingFace cache on `$SCRATCH` (SLURM output goes to `slurm_logs/`):

```bash
sbatch run_job.sh train.py --config configs/sft.json
sbatch -J smoke --time=00:30:00 run_job.sh train.py --config configs/smoke.json
```

Runs also log to wandb (project `pgm-tscl`; set `wandb_project`/`wandb_entity`, or `--wandb_project ""` to turn it off).
Log in once with `uv run wandb login`, or put `WANDB_API_KEY` in `.env` (see `.env.example`).

Logs go to `runs/<config>_s<seed>/log.jsonl`, with three record types:
- `train`: loss and LR
- `round`: the teacher's distribution, the per-bucket scores, its Q values, and per-bucket data usage
  (`drawn` = total draws, `unique_seen` = distinct examples used, `refills` = times the bucket ran out and started repeating)
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
| observation *o_t* | per-bucket accuracy (greedy decoding) on a held-out val split carved out of MATH train (`n_val_per_bucket` per bucket) |
| reward *r_t* | score change *o_t − o_{t−1}* (Online, Sampling) or learning-curve slope (Naive, Window) |
| final task | MATH-500 accuracy (reporting only; the teacher never sees it) |

This makes training a doubly nested loop. The outer loop is the teacher's timesteps. The inner loop is
`teacher_interval` ordinary optimizer steps.

**Cost of the observation**: as in the paper, the score is validation accuracy, which needs generation.
Each teacher timestep generates N × `n_val_per_bucket` answers (160 with the defaults). This is the main
overhead of TSCL over plain finetuning. Use `teacher_interval` and `n_val_per_bucket` to trade it off.

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
| `online` | `alpha=0.1, policy="greedy"\|"boltzmann", epsilon=0.1, tau=4e-4, use_abs=True` |
| `naive` | same as online + `K=5` (hold one distribution for K rounds) |
| `window` | same as online + `K=10` (FIFO window of scores) |
| `sampling` | `K=10, epsilon=0.1` |

Boltzmann's `tau` depends on the scale of the scores. The paper's 4e-4 was tuned for accuracy on 4096-example
validation sets. Our val sets are much smaller, so scores are noisier and `tau` probably needs tuning.
