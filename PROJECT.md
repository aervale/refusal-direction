# Project brief (living doc)

_Last updated: 2026-10-18_

## Summary

I'm replicating Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction," on Qwen2.5-1.5B-Instruct. I use TransformerLens in Colab, reuse the authors' data splits from `andyrdt/refusal_direction`, and write my own implementation. The paper claims that a single difference-in-means vector in the residual stream controls refusal. Ablating it makes the model comply with harmful requests. Adding it makes the model refuse harmless ones. A rank-1 edit to the weights makes the jailbreak permanent with little capability loss. I want to find out three things. First, does this hold on a newer, smaller model whose safety tuning differs from any model the paper tested? Second, how much of the effect survives honest controls, such as ablating a random direction and judging refusals with something better than substring matching? Third, where does the "single direction" story start to break? Later work argues that refusal is spread across a multi-dimensional cone of directions (Wollschläger et al., ICML 2025) and that models encode harmfulness and refusal separately (Zhao et al., NeurIPS 2025). This matters for AI safety because the paper is some of the clearest evidence that safety fine-tuning in open-weight models can be shallow. "Abliteration," the weight-editing attack that came out of this paper, is widely used to strip safeguards from released models. So whether refusal really is one removable direction affects decisions about releasing open weights, work on tamper-resistant safety training (for example, the extended-refusal fine-tuning defense in arXiv 2505.19056), and the idea of using the direction as a cheap jailbreak monitor.

## Assumptions

Status: `untested` / `partly checked` / `holds` / `broken`. Update this table as results come in.

| # | Area | Assumption | How I'll check | If it's wrong | Status |
|---|------|-----------|----------------|---------------|--------|
| 1 | Existing work | The finding transfers to Qwen2.5-1.5B. The paper tested 13 models from 1.8B to 72B, including Qwen-1.8B-Chat, but not Qwen2.5. | Notebook 02: bypass refusal rate vs. baseline on JailbreakBench | That's a finding in itself. Rerun on Qwen-1.8B-Chat (TransformerLens supports it) to tell a model difference from a bug in my code | untested |
| 2 | Existing work | "Single direction" is accurate, not just "one direction is enough to move my metrics." Wollschläger et al. find several independent refusal directions | Random-direction control; cosine similarity between the best candidates across layers; ablate the selected direction and look for leftover refusal | Becomes the extension: how many directions does a 1.5B model need? | untested |
| 3 | Existing work | My implementation matches the paper's method: hook sites, selection filters (KL < 0.1, prune the top 20% of layers), and post-instruction positions | Unit tests (pass on a tiny model); check that the selected layer sits at a similar relative depth to the paper's (roughly middle) | Results aren't comparable to the paper. Diff my code against the original repo | partly checked |
| 4 | Data | Base Qwen2.5-1.5B refuses most of the harmful prompts, so filtering still leaves enough training and validation prompts (128 / 32 per class) | `filter_by_refusal` warnings; baseline refusal rate in notebook 02 | Direction estimates get noisy. Pull in more harmful data (e.g. HarmBench) or loosen the filter | untested |
| 5 | Data | Qwen2.5 starts refusals with the tokens "I" and "As", which the logit-based selection score relies on | Sanity cell in notebook 01: look at the first token of 20 refusals | Selection optimizes the wrong thing. Fix the refusal token list | untested |
| 6 | Data | Substring matching measures refusal well enough. It miscounts "disclaimer, then complies" and vague non-answers, which is why the paper also used Llama Guard 2 | Hand-label 50 ablated completions and measure agreement with the substring metric | Below ~85% agreement, add an LLM judge | untested |
| 7 | Data | 100 JailbreakBench prompts are enough. Fine for the headline effect (the paper reports roughly 100% to near 0%), but the standard error is about 5 points, so smaller differences won't be resolved | Report confidence intervals | Use a larger harmful test set for close comparisons | untested |
| 8 | Method | Ablation removes refusal without just breaking the model | KL on harmless prompts, WikiText cross-entropy, and MMLU subset, each compared with a random-direction control | "Bypass" is really model damage. Reframe the result | untested |
| 9 | Method | 200 MMLU questions plus WikiText cross-entropy are enough to claim capabilities are preserved. They only detect big drops (MMLU standard error ~3.5 points) and skip TruthfulQA and GSM8K, where the paper saw changes | Paired bootstrap on MMLU | Weaken the claim, or add TruthfulQA | untested |
| 10 | Method | A per-prompt nanoGCG suffix is a fair stand-in for the paper's universal adversarial suffix in the Section 5 analysis | Compare with a random-suffix control | The Section 5 comparison is weaker. Say so in the write-up | untested |
| 11 | Tools | TransformerLens 2.18 (pinned, since 3.x dropped `HookedTransformer`) loads Qwen2.5-1.5B correctly with `from_pretrained_no_processing` and left padding. Only tested on a tiny random model so far | First thing in Colab: compare TransformerLens logits with Hugging Face logits on 5 prompts | Everything downstream is wrong. Switch to raw Hugging Face hooks or nnsight | partly checked |
| 12 | Tools | A free Colab T4 is enough. The model needs about 3 GB in fp16, and a deepcopy for orthogonalization brings that to about 6 GB. Generation runs without a KV cache, and the 5-15 min per notebook estimates are unverified; GCG may take 20+ min. Sessions can disconnect, so results are saved to Drive | Time each notebook and log the times here | Shrink n_test or max_new_tokens, add a KV cache, or use Colab Pro or RunPod | untested |
| 13 | Impact | A clean replication on a small model is useful beyond my own learning: as a teaching resource for safety reading groups, and as a check that the result holds on current models | Share the repo and write-up; see whether anyone uses it | Still worth doing for learning, but frame it as practice | untested |
| 14 | Impact | The interesting outcome is a deviation. Full confirmation is expected. Partial bypass, capability damage, or evidence of multiple directions would matter to people building tamper-resistant safety training and evals for open-weight releases | n/a, depends on results | n/a | untested |

## Changelog

- 2026-10-18: First version. Repo scaffolded, exercises not started, nothing run on the real model yet.

## References

- Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction. https://arxiv.org/abs/2406.11717
- Wollschläger et al. 2025, The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence. https://arxiv.org/abs/2502.17420
- Zhao et al. 2025, LLMs Encode Harmfulness and Refusal Separately. https://arxiv.org/abs/2507.11878
- An Embarrassingly Simple Defense Against LLM Abliteration Attacks. https://arxiv.org/abs/2505.19056
