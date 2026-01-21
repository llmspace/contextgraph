# Scaling Benchmark Report

## Performance by Corpus Size

| Corpus | Decay Acc | Seq Acc | Silhouette | P50 ms | P95 ms | P99 ms |
|--------|-----------|---------|------------|--------|--------|--------|
| 100 | 0.781 | 0.000 | 1.000 | 0.0 | 0.0 | 0.0 |
| 500 | 0.761 | 0.430 | 1.000 | 0.0 | 0.0 | 0.0 |
| 1000 | 0.768 | 0.435 | 1.000 | 0.0 | 0.0 | 0.0 |
| 2000 | 0.774 | 0.781 | 1.000 | 0.0 | 0.0 | 0.0 |
| 5000 | 0.770 | 0.426 | 1.000 | 0.0 | 0.0 | 0.0 |
| 10000 | 0.771 | 0.370 | 1.000 | 0.0 | 0.0 | 0.0 |

## Degradation Analysis

- Decay accuracy rate: 0.0050/10x
- Sequence accuracy rate: -0.1850/10x
- Latency growth rate: 0.0000/10x
