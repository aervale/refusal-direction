# Refusal is mediated by a single direction: a replication on Qwen2.5-1.5B-Instruct

Replicating [Arditi et al., 2024](https://arxiv.org/abs/2406.11717) ("Refusal in Language Models Is Mediated by a Single Direction") on a small open chat model, using TransformerLens. Original code: [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction).

**The claim, in one paragraph.** Take the mean residual-stream activation on harmful prompts, subtract the mean on harmless prompts. That one vector `r` controls refusal: projecting it out of every activation makes the model comply with harmful requests, adding it makes the model refuse harmless ones, and you can bake the projection into the weights with a rank-1 edit per matrix, with little capability loss.

## Workflow

```
VS Code (laptop)                          Colab (GPU)
-----------------                         -----------
edit src/ exercises                       notebooks/0X_*.ipynb
pytest  (tiny model, CPU, seconds)  --->  git pull + pip install -e .
git push                                  run on the real model
```

The notebooks are thin: all logic lives in `src/refusal_direction/`, so you edit Python files with a real editor and tests, and Colab only runs them.

### Local setup (for tests)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # everything fails with NotImplementedError until you do the exercises
```

### Colab setup

1. Push this repo to GitHub.
2. Open a notebook in Colab (File > Open notebook > GitHub tab), set Runtime to a T4 GPU.
3. Edit `REPO` in the first cell, run it. If Colab asks to restart after pip (numpy downgrade), restart and rerun that cell.
4. Artifacts (the direction, scores, results) are saved to Google Drive, so later notebooks reuse earlier results.

## Exercises

Do them in order. Each has a docstring explaining what and why, and a test file.

| # | File | What you implement | Paper | Test |
|---|------|--------------------|-------|------|
| 1a | `interventions.py` | `project_out`: x - r̂ r̂ᵀ x | §2.4, Eq. 4 | `test_interventions.py` |
| 1b | `interventions.py` | `get_ablation_hooks`: ablate everywhere | §2.4 | `test_interventions.py` |
| 2 | `interventions.py` | `get_addition_hooks`: x + r at one layer | §2.4, Eq. 3 | `test_interventions.py` |
| 3 | `directions.py` | mean activations + difference-in-means | §2.3, Eq. 1 | `test_directions.py` |
| 4 | `selection.py` | `select_direction`: filter + argmin | App. C | `test_selection.py` |
| 5 | `orthogonalize.py` | edit weights so nothing writes along r̂ | §4, Eq. 5 | `test_orthogonalize.py` |
| 6 | `mechanistic.py` | projection by layer, per-head contributions | §5 | `test_mechanistic.py` |

`pytest tests/test_pipeline_smoke.py` checks everything fits together once 1-4 are done.

Stuck? The `solutions` branch has reference implementations: `git diff main solutions -- src/`. Try for real first; the tests tell you a lot.

## Notebooks

| Notebook | Paper | Needs exercises | Rough T4 time |
|----------|-------|-----------------|---------------|
| `01_find_direction` | §2.2-2.3, App. A, C | 1-4 | ~15 min (scoring dominates) |
| `02_ablation_and_addition` | §3, Fig. 1 & 3 | 1-2 | ~15 min |
| `03_weight_orthogonalization` | §4 | 5 | ~5 min |
| `04_capability_evals` | §4.2, Table 1 (lite) | 5 | ~10 min |
| `05_mechanistic_extras` | §5, Fig. 5 | 6 | ~5 min, +15 for GCG |

## Code map

```
src/refusal_direction/
  config.py         all knobs (model, data sizes, selection thresholds)
  data.py           paper's exact data splits, downloaded on first use        [provided]
  model.py          load model, chat template, batched logits, greedy generation [provided]
  refusal.py        logit refusal score + substring refusal metric           [provided]
  directions.py     difference-in-means                                      [Ex 3]
  interventions.py  ablation and addition hooks                              [Ex 1-2]
  selection.py      score candidates [provided], pick one                    [Ex 4]
  orthogonalize.py  weight editing                                           [Ex 5]
  capability.py     WikiText CE, MMLU subset                                 [provided]
  mechanistic.py    probes, head attribution                                 [Ex 6]
  artifacts.py      save/load between notebooks                              [provided]
tests/              tiny random Qwen-architecture model; CPU, no downloads
notebooks/          Colab entry points
```

## Deviations from the paper (worth mentioning in your write-up)

- **Model:** Qwen2.5-1.5B-Instruct instead of the paper's Qwen-1.8B-Chat (a newer model with stronger safety tuning; Qwen 1 also needs `trust_remote_code` and older transformers). Qwen-1.8B-Chat *is* in TransformerLens, so running it as a direct comparison is a good extension: change `Config.model_name` and the chat template in `model.py`.
- **Refusal judging:** substring matching only. The paper also uses Llama Guard 2 for a "safety score"; substring matching overcounts disclaimers-then-comply and undercounts vague non-answers.
- **Capability evals:** WikiText CE + 200 MMLU questions instead of the full benchmark suite.
- **Adversarial suffix:** per-prompt nanoGCG instead of the paper's universal suffix.
- **No system prompt** in the chat template, matching the paper's Qwen setup (`Config.system_prompt` to change).

## Gotchas already handled (so you know why the code looks like it does)

- **TransformerLens is pinned to 2.18.** 3.x+ replaced `HookedTransformer` with `TransformerBridge`.
- **`from_pretrained_no_processing`**, not `from_pretrained`: no LayerNorm folding, so the residual basis matches HF and weight edits are clean.
- **Left padding + explicit attention masks** everywhere, so position `-1` is always the real last token and rotary positions are right. `test_pipeline_smoke.py` checks padding doesn't leak.
- **Float32 accumulation** for means; activations are fp16/bf16 on GPU.

## Results

Fill in as you go (`RESULTS.md`).
