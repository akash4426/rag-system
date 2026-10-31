# SAH-RAG Experiment Comparison

Retrieval and answer metrics produced by `eval/run_eval.py`.
Answer metrics are averaged over answerable questions unless noted.

| Experiment | R@5 | P@5 | MRR | nDCG@5 | Answer F1 | Answer sim | Faithful | Citation acc | Citation cov | Decision acc | Unans. abstain | False abstain | Retries | Total ms | Ans tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0.800 | 0.400 | 0.800 | 0.764 | 0.277 | 0.771 | 1.000 | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 7583 | 95 |
| E2 | 0.800 | 0.200 | 0.700 | 0.726 | 0.266 | 0.744 | 0.938 | 0.938 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 11409 | 82 |
| E3 | 0.800 | 0.360 | 0.700 | 0.725 | 0.267 | 0.781 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.00 | 10998 | 68 |
| E4 | 0.800 | 0.320 | 0.700 | 0.717 | 0.305 | 0.781 | 1.000 | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.00 | 13506 | 81 |
| E5_structure | 0.800 | 0.320 | 0.700 | 0.717 | 0.294 | 0.775 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 0.000 | 0.40 | 11472 | 70 |
| E5_plain | 0.800 | 0.320 | 0.700 | 0.717 | 0.268 | 0.785 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 0.000 | 0.40 | 10635 | 78 |
