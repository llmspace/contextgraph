# Temporal Benchmark Results

## Dataset

- Memories: 1000
- Queries: 99
- Chains: 38
- Episode boundaries: 58

## E2 Recency Metrics

- Recency-weighted MRR: 0.988
- Decay accuracy: 0.767
- Freshness P@1: 1.000
- Freshness P@20: 1.000
- Freshness P@5: 1.000
- Freshness P@10: 1.000

## E3 Periodic Metrics

- Hourly cluster quality: 1.000
- Daily cluster quality: 1.000
- Periodic R@20: 1.000
- Periodic R@5: 1.000
- Periodic R@10: 1.000
- Periodic R@1: 1.000

## E4 Sequence Metrics

- Sequence accuracy: 0.451
- Temporal ordering (Kendall's tau): 1.000
- Episode boundary F1: 0.792
- Before/after accuracy: 1.000

## Composite Metrics

- Overall temporal score: 0.846
- Improvement over baseline: 181.8%

## Ablation Study

- Baseline: 0.300
- E2 only: 0.925
- E3 only: 0.700
- E4 only: 0.794
- Full: 0.846

### Feature Improvements

- e2_recency: +208.5%
- e3_periodic: +133.3%
- e4_sequence: +164.6%
- full: +181.8%

## Timings

- Total: 1257ms
- Dataset generation: 32ms
- Recency benchmark: 47ms
- Periodic benchmark: 11ms
- Sequence benchmark: 6ms
- Ablation study: 847ms
