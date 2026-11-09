# SAGDataset 多跳检索实验结果

> 从每个问题的候选段落中检索 supporting facts。指标为文档级 `nDCG@10` 与 `Recall@2`。

| 数据集 | 系统 | 查询数 | nDCG@10 | Recall@2 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | BM25 | 20 | 0.7975 | 0.7000 | 0.8750 | 0.9100 |
| 2WikiMultiHopQA | BGE-M3 Dense | 20 | 0.9189 | 0.8000 | 1.0000 | 1.0000 |
| 2WikiMultiHopQA | OpenJev-9B + SAG2 | 20 | 0.9249 | 0.8500 | 0.9625 | 0.9625 |
| HotpotQA | BM25 | 20 | 0.9562 | 0.8250 | 1.0000 | 0.9750 |
| HotpotQA | BGE-M3 Dense | 20 | 0.9448 | 0.8000 | 1.0000 | 1.0000 |
| HotpotQA | OpenJev-9B + SAG2 | 20 | 0.9481 | 0.8500 | 1.0000 | 0.9750 |
| MuSiQue | BM25 | 20 | 0.5989 | 0.5000 | 0.6375 | 0.7972 |
| MuSiQue | BGE-M3 Dense | 20 | 0.7478 | 0.6000 | 0.8417 | 0.8722 |
| MuSiQue | OpenJev-9B + SAG2 | 20 | 0.8002 | 0.7500 | 0.8375 | 0.9417 |

## 数据集宏平均

| 系统 | nDCG@10 | Recall@2 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7842 | 0.6750 | 0.8375 | 0.8941 |
| BGE-M3 Dense | 0.8705 | 0.7333 | 0.9472 | 0.9574 |
| OpenJev-9B + SAG2 | 0.8910 | 0.8167 | 0.9333 | 0.9597 |

## 运行开销

| 系统 | 模型加载 |
| --- | ---: |
| BM25 | 不适用 |
| BGE-M3 Dense | 5.1 s |
| OpenJev-9B + SAG2 | 26.0 s |

| 数据集 | 系统 | 上下文索引 p50 | 查询均值 | 查询 p50 | 查询 p95 | 总查询时间 | QPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | BM25 | 1.5 ms | 0.3 ms | 0.3 ms | 0.4 ms | 0.01 s | 3746.44 |
| 2WikiMultiHopQA | BGE-M3 Dense | 58.6 ms | 28.7 ms | 22.8 ms | 27.0 ms | 0.57 s | 34.88 |
| 2WikiMultiHopQA | OpenJev-9B + SAG2 | 2.0 ms | 9552.9 ms | 6935.4 ms | 24755.5 ms | 191.06 s | 0.10 |
| HotpotQA | BM25 | 2.0 ms | 0.3 ms | 0.3 ms | 0.5 ms | 0.01 s | 2990.97 |
| HotpotQA | BGE-M3 Dense | 59.8 ms | 22.4 ms | 22.0 ms | 26.3 ms | 0.45 s | 44.66 |
| HotpotQA | OpenJev-9B + SAG2 | 2.0 ms | 6501.7 ms | 6231.7 ms | 9534.0 ms | 130.03 s | 0.15 |
| MuSiQue | BM25 | 3.4 ms | 0.4 ms | 0.4 ms | 0.6 ms | 0.01 s | 2386.07 |
| MuSiQue | BGE-M3 Dense | 102.3 ms | 21.5 ms | 20.8 ms | 25.7 ms | 0.43 s | 46.61 |
| MuSiQue | OpenJev-9B + SAG2 | 3.4 ms | 19045.3 ms | 17289.6 ms | 37518.1 ms | 380.91 s | 0.05 |

## 9B 判断与 Token 用量

| 数据集 | 请求数 | 判断数 | 输入 Token | 输出 Token |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 70 | 230 | 47,484 | 4,295 |
| HotpotQA | 60 | 201 | 44,379 | 3,736 |
| MuSiQue | 120 | 445 | 102,293 | 8,379 |

## 口径

- 2WikiMultiHopQA 使用 `dev.parquet`；MuSiQue 使用 `musique_ans_v1.0_dev.jsonl`；HotpotQA 使用 validation。
- 每个问题的候选段落即该条数据提供的 context。supporting title 作为金标文档。
- HotpotQA 仅保留 supporting title 确实存在于同条 context 的样本；当前 parquet 中可用约 28%。
- 每套数据固定抽样 20 个问题，seed=0；所有系统使用完全相同的样本。
- BGE-M3 与 BM25 的上下文索引和查询时间分开报告；SAG2 在线查询包含 BM25、必要时两轮扩展、OpenJev 判断和 DPP。
- 9B 使用完整 GPU 推理，没有 CPU layer offload。
- 9B 仍是 zero-shot retrieval：动态问题 ID 未命中混训头的任务槽，`slot_head` 按设计回退到保留 A/B 行。
