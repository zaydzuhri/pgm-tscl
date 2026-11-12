# Handoff: TSCL for LLM finetuning (for the Claude Code session on the cluster)

This file is context for a new Claude Code session picking up this project. Read it fully, then
`README.md`, then the three Python files (~720 lines total). The paper is `1707.00183v2.pdf`
(Matiisen et al. 2017, *Teacher-Student Curriculum Learning*).

## 1. The user's original request (verbatim)

> Alright, this folder is for a new class project I'm doing for the course probabilistic graphical models. As you can see there is a paper PDF there already. I want you to read it. What we're trying to do is essentially is try to apply TSCL to LLM finetuning. I want to start with a barebones repository that can at least finetune Qwen3-0.6B on Hendryck's MATH. I'm sure you're familiar on how to set that up with huggingface and pytorch. However, since we're trying to implement TSCL, we need to minimally expose the training loop and such so that we can modify the data loader(?) so that the "teacher" can operate on the batch sampling decision across difficulty buckets. It might be a doubly nested training loop? Not sure, read the paper and evaluate. Importantly though, the training script should be structured so that you can change the parameters of "teacher update interval" and "training steps" or whatever necessary, such that you can have a fair comparison between a TSCL training run and a normal finetuning run, since our objective is to have a more data efficient training strategy. Also, I want the repository right now to be as minimal as possible with the least number of python files, BUT it should still be modular in the model and data selection AND more importantly the way we construct the "teacher" (which is just a data loader kindof?) this modularity should also enable switching between normal finetuning and TSCL. Please get started.

Follow-up requirements from the same session, in order:
1. Use **uv** for everything Python.
2. Run training as `train.py --config some_config.json` with the training params in the JSON.
3. Add short, plain-language comments to `train.py` that use **the TSCL paper's terms**, so the user can
   explain the code to their team. Making code more readable is fine; don't change too much.
4. The inner training loop should look **as much like normal finetuning as possible** (gradient
   accumulation / micro-batching was removed on request).
5. **No abbreviated variable names** (`config` not `cfg`, `optimizer` not `opt`, `tokenizer` not `tok`).
   Paper symbols (`Q`, `K`, `alpha`, `tau`, `D`) are fine.
6. **As few evaluation functions as possible**; repeated code is OK. The Teacher's score is **validation
   accuracy** (the val-loss and train-loss alternatives were removed).
7. Comments giving **tensor shapes** for every tensor input/output.
8. Log a running count of **bucket refills** (when a bucket's examples start repeating).

## 2. Working preferences (carry these over)

- Keep the repo minimal (3 Python files) but modular: model, dataset/bucketing, and teacher are each swappable.
- Readable code > clever code. Full variable names, comments tied to the paper, explicit steps.
- Don't run heavy jobs on the user's laptop. That no longer applies on the cluster, but **confirm before
  launching long or multi-GPU jobs**, and check the cluster's scheduler conventions (e.g. SLURM) with the user.
- The user commits themselves. Don't commit or push unless asked.
- Git remote: `https://github.com/zaydzuhri/pgm-tscl.git`. The working tree was clean at handoff (4 commits).

## 3. How the paper maps onto the code

We use the **batch POMDP** formulation (paper Sec 2.2, Appendix B), the one meant for supervised learning:

| paper | code |
|---|---|
| Student | the LLM (`Qwen/Qwen3-0.6B-Base`) |
| task *i* | a MATH difficulty bucket (`bucket_by`: `level` → 5 tasks, `subject` → 7, `level_subject` → 35) |
| action *a_t* | `teacher.distribution()`, an (N,) distribution over buckets |
| observation *o_t* | per-bucket **greedy-decoding accuracy** on a held-out val split (`n_val_per_bucket` MATH train problems per bucket, chosen with a fixed `data_seed`) |
| reward *r_t* | computed inside each teacher: score change (Online, Sampling) or learning-curve slope (Naive, Window) |
| final task | MATH-500 accuracy, for reporting only; the Teacher never sees it |

Training is **one ordinary loop** over optimizer steps (`train.py: main`); the Teacher acts every `teacher_every` steps:

```
o_0 = val accuracy per bucket ; teacher.update(o_0)       # only for teachers that use scores
a = teacher.distribution()
for step in 1..total_train_steps:
    batch ~ a (BucketSampler) ; forward ; backward ; optimizer.step
    test eval every eval_every steps
    if step % teacher_every == 0:                         # one Teacher timestep
        o_t = val accuracy per bucket ; teacher.update(o_t)   # only for teachers that use scores
        a = teacher.distribution()
```

Baselines (`proportional`, `uniform`) never observe scores, so they cost the same as plain finetuning.

**Fair comparison:** normal finetuning is the `proportional` teacher (bucket probability ∝ bucket size,
which matches shuffled SFT) running through the same loop. All configs share `total_train_steps`, `batch_size`,
`lr`, `warmup_ratio`, `eval_every`, `data_seed`, so every run sees the same number of training examples,
the same LR schedule and the same test-eval steps. Logs record `examples`, `tokens` and
`teacher_eval_examples` (the Teacher's overhead).

## 4. Files

- `train.py`
  - `Config` dataclass and `load_config` (defaults ← JSON ← `--key value` CLI overrides)
  - `load_model`
  - `forward_loss`: computes logits only at label positions to save memory with the 150k-token vocab. Verified to equal HF `model(...).loss` exactly.
  - `evaluate_accuracy`: greedy-decoding accuracy (used for the Teacher's observation and the test eval).
  - `evaluate_loss`: teacher-forced loss on the held-out val split, overall and per bucket (no generation).
  - `main`: the loop.
- `data.py`
  - `load_math`: training solutions from the DART-Math pool (`hkust-nlp/dart-math-pool-math`, ~1.5M DeepSeekMath-7B-RL solutions to the 7.5k MATH train problems, level from `query_metadata.level`, subject from `domain`). The val split is held out per problem (one pool solution each, for the val loss). `boxed_solution` rewrites the DART ending `The answer is: X` as `\boxed{X}`. Test is `HuggingFaceH4/MATH-500`. The human-written solutions (`EleutherAI/hendrycks_math`) were used before 2026-11-11; with only 7.5k of them, SFT overfit after about two epochs and never beat the base model.
  - `BucketSampler` takes an `encode` function and tokenizes examples when they are drawn, because the pool is too large to tokenize up front.
  - `split_into_buckets`, `format_prompt`, `encode`, `collate`
  - `BucketSampler`: one shuffled queue per bucket, sampled without replacement and refilled when empty. It tracks `drawn`, `seen` and `refills`.
  - Answer checking: `last_boxed`, then string normalization, then a `math_verify` fallback.
- `teachers.py`
  - Baselines: `proportional`, `uniform`.
  - Paper teachers: `online`, `naive`, `window`, `sampling` (batch versions, Algorithms 5–8).
  - Registry: `TEACHERS`.
- `configs/`: `sft`, `uniform`, `tscl_online`, `tscl_window`, `tscl_sampling` (identical budgets), plus `smoke` (tiny).
- Output: `runs/<config>_s<seed>/{config.json, log.jsonl}`. `log.jsonl` has four record types:
  - `train`: loss, lr, grad_norm, probs (the action in use), drawn, unique_seen, refills
  - `round` (only for teachers that observe scores): scores, Q and other teacher state
  - `val`: loss on the held-out val split (overall and per bucket), every `val_loss_every` steps and once at step 0
  - `eval`: test_acc overall, by level and by subject; val_acc and train_acc (a fixed `n_train_acc_per_bucket` subset of train), overall and per bucket

Prompt format: plain text for a base model, `Problem: <problem>\nPlease reason step by step, and put your
final answer within \boxed{}.\nSolution: `. Only the solution tokens (plus EOS) are trained on. The
instruction is not part of MATH; we keep it so the step-0 eval and the Teacher's o_0 aren't dominated by
formatting failures (Qwen3-0.6B-Base scores ~32% on MATH-500 with it).

## 5. What has been verified and what hasn't

Verified (on CPU, laptop):
- The teachers on a toy gated curriculum: Sampling walks through the buckets in order, and the EMA teachers are slow at alpha=0.1.
- Answer checking, `collate`, `BucketSampler` including the refill counter, and loading every config.
- On a tiny random Qwen3, `forward_loss` gives the right shapes and exactly equals HF's loss.
- Early smoke run on the real model: data loading, model loading, test generation, o_0 scoring and one training step all ran. This was **before** later refactors: micro-batching removed, eval functions merged, val-accuracy scoring.

**Not yet verified:** a full end-to-end run of the current code on GPU.

## 6. Suggested next steps on the cluster

1. Set up the environment and run the smoke test:
   ```bash
   uv sync
   uv run train.py --config configs/smoke.json --out_dir runs/smoke
   cat runs/smoke/log.jsonl
   ```
   Check that `train` records contain `probs` and `refills`, `round` records contain `scores`, and that `eval` records appear.
   Expect ~0 accuracy in the smoke test, because `max_new_tokens` is 32.
2. Measure the costs on the real GPU:
   - seconds per training step at `batch_size=32, max_len=1024`. If it runs out of memory, set `"grad_ckpt": true`.
   - time per MATH-500 eval.
   - time per teacher observation: N × `n_val_per_bucket` = 160 generations with the defaults.

   Teacher observation is the main overhead of TSCL. Tune `teacher_every` and `n_val_per_bucket` with the user.
3. Run `sft` against the TSCL configs with matched budgets and a few seeds. Plot test accuracy against `examples`.

## 7. Open questions / known caveats to raise with the user

- **Budget unit:** equal steps is not equal tokens, because Level 5 solutions are longer. Decide whether the comparison should be token-matched.
- **Noisy scores:** val accuracy on 32 examples per bucket moves in steps of ~3%. The paper used 4096-example val sets. Its `tau=4e-4` (Boltzmann) and `epsilon=0.1` likely need retuning.
- **Generation speed:** HF `generate` is used. vLLM would be much faster for evals if they become the bottleneck, but it adds a dependency. Ask first.
- **More baselines:** a hand-designed curriculum (easy → hard) isn't implemented yet. It's the paper's other baseline and would be a new `Teacher` subclass.
