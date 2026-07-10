# Prompting Strategy Benchmark — Summary
> Remote tokens consumed: **0** for all strategies (local-only inference)

## Sentiment
| Strategy | Accuracy (4 probes) | Avg Latency ms | Remote Tokens |
|---|---|---|---|
| self_consistency_3x | 100.00% | 69 | 0 |
| zero_shot_strict | 100.00% | 104 | 0 |
| few_shot_3 | 100.00% | 124 | 0 |
| baseline | 100.00% | 142 | 0 |
| chain_of_thought | 100.00% | 214 | 0 |

## Summarization
| Strategy | Avg Jaccard Sim | Avg Latency ms | Remote Tokens |
|---|---|---|---|
| self_consistency_3x | 0.3502 | 983 | 0 |
| zero_shot_strict | 0.3305 | 1096 | 0 |
| few_shot_3 | 0.3265 | 1424 | 0 |
| baseline | 0.2973 | 1302 | 0 |
| chain_of_thought | 0.2211 | 2227 | 0 |

---
**Scoring note**: Local tokens cost zero toward the hackathon score. Latency is recorded for the 10-minute wall-clock budget constraint only.
