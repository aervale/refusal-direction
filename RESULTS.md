# Results: Qwen2.5-1.5B-Instruct

## Selected direction (notebook 01)

| layer | position | bypass_score | induce_score | kl_score |
|-------|----------|--------------|--------------|----------|
|       |          |              |              |          |

## Refusal rates, substring matching (notebooks 02, 03)

| condition | harmful (JailbreakBench, n=100) | harmless (Alpaca, n=100) |
|-----------|-------------------------------|--------------------------|
| baseline | | |
| ablate r (hooks) | | — |
| ablate random direction | | — |
| subtract r at layer l | | — |
| add r at layer l | — | |
| orthogonalized weights | | — |

## Capability (notebook 04)

| model | WikiText CE | MMLU (n=200) | harmless KL vs base |
|-------|-------------|--------------|---------------------|
| base | | | 0 |
| orthogonalized | | | |
| random-orthogonalized | | | |

## Mechanistic (notebook 05)

- Layer where harmful/harmless projections separate:
- Logit lens on r:
- Top heads writing r:
- Effect of adversarial suffix on projection / head attention:

## What I'd tell someone who's about to read the paper
