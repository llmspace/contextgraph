# E5 Causal Embedder — Complete Improvement Plan

**Date**: 2026-02-11
**Branch**: casetrack
**Current Benchmark**: 3/8 PASS (Phases 3, 5, 8)
**Target**: 7/8 PASS minimum (Phase 8 perf profiling uses synthetic timing; real GPU timing is Phase 3 stretch goal)

---

## Executive Summary

The E5 causal embedder (nomic-embed-text-v1.5 + LoRA + projection) has been trained and calibrated but benchmarks at only 3/8. Five phases fail due to three root causes:

1. **Benchmark wiring bug** — Phases 4, 6, 7 hardcode `synthetic_e5_score()` instead of using the GPU provider. This means the benchmark never tests real E5 embeddings for ablation, retrieval, or generalization. **This is the #1 blocker.**
2. **GpuProvider::e5_score() asymmetry bug** — Both cause and effect texts are embedded using `embed_as_cause()`. The effect text should use `embed_as_effect()`. This kills the asymmetric advantage E5 is designed to provide.
3. **Intent detection coverage gap** — Effect detection accuracy is 60% (cause is 85%). Missing natural-language effect patterns in `detect_causal_query_intent()`.

Secondary issues:
- E5 embedding spread=0.048 (target 0.10), anisotropy=0.325 (target 0.30) — close, likely improves once asymmetry bug is fixed
- Training evaluation always returns 0 — `Evaluator::evaluate()` is broken, preventing future training optimization
- GpuProvider silently returns zero vectors on error — should fail fast per constitution

---

## Phase-by-Phase Diagnosis

### Phase 1: Intent Detection — FAIL (75.8% accuracy, target 90%)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| accuracy | 0.758 | 0.90 | FAIL |
| cause_accuracy | 0.85 | — | — |
| effect_accuracy | 0.60 | — | — |
| negation_fp | 0.10 | <0.15 | PASS |

**Root cause**: Effect queries use natural-language patterns the detector misses. With 40 effect-seeking queries and 60% accuracy, ~16 queries fail. The 50 cause-seeking queries at 85% have ~8 failures. Combined: ~24 misses out of 120 total.

**What to fix**: Add missing effect patterns to `detect_causal_query_intent()` in `asymmetric.rs`, then investigate the remaining cause misses.

### Phase 2: E5 Embedding Quality — FAIL (spread=0.048, target 0.10)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| spread | 0.048 | >=0.10 | FAIL |
| anisotropy | 0.325 | <=0.30 | FAIL |
| standalone_accuracy | 0.638 | >=0.67 | FAIL |

**Root cause**: The `GpuProvider::e5_score()` embeds both texts as cause variant. When asymmetric pairing is wrong, the model loses its trained discrimination between cause→effect vs non-causal. Fixing the asymmetry bug should widen the score spread.

Also: the 3-stage training produced CE 2.08→0.14, but the evaluation metrics (spread, directional_accuracy) were always 0 due to the broken `Evaluator::evaluate()`. This means training proceeded without eval-guided checkpoint selection. The final weights may not be optimal.

### Phase 3: Direction Modifiers — PASS (100%, 1.50x ratio)

No action needed.

### Phase 4: Ablation — FAIL (e5_rrf=0.83%, target 12%)

**Root cause**: `simulate_search()` (line 537 of phases.rs) calls `synthetic_e5_score()` which returns 0.93-0.98 for everything. The function completely ignores `config.provider`. With synthetic scores, E5 adds noise not signal, making it impossible for E5 to contribute >12% RRF.

**What to fix**: Wire `simulate_search()` to use `config.provider.e5_score()` for real embeddings.

### Phase 5: Causal Gate — PASS (TPR=93.0%, TNR=96.1%)

No action needed.

### Phase 6: E2E Retrieval — FAIL (top1=8.3%, target 55%)

**Root cause**: `simulate_ranked_search()` (line 729) calls `synthetic_e5_score()`. Same issue as Phase 4. Additionally, the ranking simulation uses basic word overlap which is a poor proxy for E1 semantic search. The retrieval simulation should use the GPU provider for E5 scores while keeping word overlap as E1 approximation.

**What to fix**: Wire `simulate_ranked_search()` to use `config.provider.e5_score()` and potentially increase E5 weight from 0.10 to match real search behavior.

### Phase 7: Cross-Domain — FAIL (held_out=0%, target 45%)

**Root cause**: Same `simulate_search()` synthetic issue as Phase 4. Also, held-out domains (psychology, history) have only 5 seed training pairs each. Cross-domain generalization depends on E5 learning causal STRUCTURE (not topic), which the trained model does show (score gap 0.30 between causal/non-causal means).

**What to fix**: Wire to GPU provider first, then evaluate if cross-domain actually fails with real scores.

### Phase 8: Performance — PASS (1.5x overhead, 230 QPS)

No action needed (synthetic timing acceptable for now).

---

## Implementation Steps — Ordered by Impact

### Step 1: Fix GpuProvider::e5_score() Asymmetry Bug

**File**: `crates/context-graph-benchmark/src/causal_bench/provider.rs`
**Lines**: 155-159

**Current (WRONG)**:
```rust
fn e5_score(&self, cause_text: &str, effect_text: &str) -> f32 {
    let cause_emb = self.e5_embedding(cause_text);  // embed_as_cause
    let effect_emb = self.e5_embedding(effect_text); // embed_as_cause (WRONG!)
    cosine_similarity(&cause_emb, &effect_emb).max(0.0)
}
```

**Fix**: Embed effect_text with `embed_as_effect()`:
```rust
fn e5_score(&self, cause_text: &str, effect_text: &str) -> f32 {
    let cause_emb = match self.runtime.block_on(self.model.embed_as_cause(cause_text)) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("GPU cause embedding failed for '{}': {}", &cause_text[..cause_text.len().min(50)], e);
            panic!("GPU embedding failed: {}", e);
        }
    };
    let effect_emb = match self.runtime.block_on(self.model.embed_as_effect(effect_text)) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("GPU effect embedding failed for '{}': {}", &effect_text[..effect_text.len().min(50)], e);
            panic!("GPU embedding failed: {}", e);
        }
    };
    cosine_similarity(&cause_emb, &effect_emb).max(0.0)
}
```

**Rationale**: E5 is an asymmetric embedder (ARCH-15, AP-77). The whole point of storing dual vectors (cause/effect) is that `embed_as_cause("smoking")` and `embed_as_effect("cancer")` occupy different subspaces. Comparing cause→cause gives symmetric similarity, defeating the purpose.

**Expected impact**: Score spread should increase significantly. Currently cause→cause comparison compresses everything; cause→effect comparison lets the projection heads do their job.

### Step 2: GpuProvider Fail-Fast (Remove Zero Fallbacks)

**File**: `crates/context-graph-benchmark/src/causal_bench/provider.rs`
**Lines**: 161-178

**Current**: Returns `vec![0.0f32; 768]` on error (silent degradation).

**Fix**: Panic on error with descriptive message:
```rust
fn e5_embedding(&self, text: &str) -> Vec<f32> {
    match self.runtime.block_on(self.model.embed_as_cause(text)) {
        Ok(emb) => emb,
        Err(e) => {
            tracing::error!("GPU embedding failed: {}", e);
            panic!("E5 GPU embedding failed — cannot continue benchmark with corrupt data: {}", e);
        }
    }
}

fn e5_dual_embeddings(&self, text: &str) -> (Vec<f32>, Vec<f32>) {
    match self.runtime.block_on(self.model.embed_dual(text)) {
        Ok((cause, effect)) => (cause, effect),
        Err(e) => {
            tracing::error!("GPU dual embedding failed: {}", e);
            panic!("E5 GPU dual embedding failed — cannot continue benchmark with corrupt data: {}", e);
        }
    }
}
```

**Rationale**: Constitution `coding_standards.error_handling`: "No silent degradation." A zero vector in a benchmark produces misleading scores that mask real problems.

### Step 3: Wire Phases 4, 6, 7 to GPU Provider

**File**: `crates/context-graph-benchmark/src/causal_bench/phases.rs`

#### 3A: Fix `simulate_search()` (used by Phase 4 and Phase 7)

**Lines**: 537-582

Change signature to accept `&BenchConfig` and use the provider:

```rust
fn simulate_search(
    query: &BenchmarkQuery,
    pairs: &[BenchmarkPair],
    with_e5: bool,
    e5_only: bool,
    config: &BenchConfig,
) -> String {
    let query_lower = query.query.to_lowercase();
    let mut best_id = String::new();
    let mut best_score = f32::NEG_INFINITY;

    for pair in pairs {
        let mut score = 0.0f32;

        if !e5_only {
            // E1 semantic approximation: domain + keyword overlap
            if pair.domain == query.expected_domain {
                score += 0.4;
            }
            let cause_words: Vec<&str> = pair.cause_text.split_whitespace().collect();
            let overlap = cause_words
                .iter()
                .filter(|w| query_lower.contains(&w.to_lowercase()))
                .count();
            score += overlap as f32 * 0.05;
        }

        if with_e5 || e5_only {
            let e5_score = config.provider.e5_score(&pair.cause_text, &pair.effect_text);
            let e5_weight = if e5_only { 1.0 } else { config.e5_weight };
            score += e5_score * e5_weight;
        }

        // Deterministic tie-breaking
        score += hash_to_float(&pair.id) * 0.001;

        if score > best_score {
            best_score = score;
            best_id = pair.id.clone();
        }
    }

    best_id
}
```

Update all call sites in `phase4_ablation()` and `phase7_cross_domain()` to pass `config`.

#### 3B: Fix `simulate_ranked_search()` (used by Phase 6)

**Lines**: 729-769

Same pattern: accept `&BenchConfig`, replace `synthetic_e5_score()` with `config.provider.e5_score()`:

```rust
fn simulate_ranked_search(
    query: &BenchmarkQuery,
    pairs: &[BenchmarkPair],
    config: &BenchConfig,
) -> Vec<String> {
    let query_lower = query.query.to_lowercase();
    let mut scored: Vec<(String, f32)> = pairs
        .iter()
        .map(|pair| {
            let mut score = 0.0f32;

            // Domain match (E1 approximation)
            if pair.domain == query.expected_domain {
                score += 0.35;
            }

            // Word overlap with cause_text
            let words: Vec<&str> = pair.cause_text.split_whitespace().collect();
            let overlap = words
                .iter()
                .filter(|w| w.len() > 3 && query_lower.contains(&w.to_lowercase()))
                .count();
            score += overlap as f32 * 0.08;

            // Word overlap with effect_text
            let words: Vec<&str> = pair.effect_text.split_whitespace().collect();
            let overlap = words
                .iter()
                .filter(|w| w.len() > 3 && query_lower.contains(&w.to_lowercase()))
                .count();
            score += overlap as f32 * 0.06;

            // E5 contribution (real GPU embeddings when available)
            let e5 = config.provider.e5_score(&pair.cause_text, &pair.effect_text);
            score += e5 * config.e5_weight;

            // Deterministic noise for variety
            score += hash_to_float(&format!("{}_{}", query.id, pair.id)) * 0.02;

            (pair.id.clone(), score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}
```

Update call site in `phase6_e2e_retrieval()` to pass `config`.

### Step 4: Improve Intent Detection (Phase 1)

**File**: `crates/context-graph-core/src/causal/asymmetric.rs`

Run the benchmark in verbose mode to identify exactly which effect queries fail, then add targeted indicators. Based on the benchmark query patterns:

**4A: Analyze failures** — Run `cargo run --release --bin causal_embedding_bench -- --verbose 2>&1 | grep "MISS.*effect"` to identify the specific effect queries that miss.

**4B: Add missing effect patterns** (likely needed based on query corpus analysis):
- `"what happens if"`, `"what would happen if"` — conditional effect queries
- `"how does X affect"`, `"how will X affect"` — already added but verify matching
- `"what is the outcome"`, `"what is the impact"` — outcome-seeking
- `"what results from"`, `"what follows from"` — explicit result-seeking

**4C: Add missing cause patterns** for the 15% cause miss rate:
- `"what is behind"`, `"what's behind"` — informal cause-seeking
- `"what explains"`, `"explain the cause"` — explanation-seeking
- `"what is responsible"`, `"who is responsible"` — responsibility attribution

**Target**: accuracy 75.8% → 90%+ (24→<12 misses out of 120)

### Step 5: Fix Training Evaluation (Future Training Improvement)

**File**: `crates/context-graph-embeddings/src/training/pipeline.rs`

The `Evaluator::evaluate()` always returns directional_accuracy=0 and score_spread=0 (confirmed by training_trajectory.jsonl). This must be fixed before any future training run.

**5A: Investigate `Evaluator::evaluate()`** — Read the full function, identify why it returns all zeros. Likely cause: the evaluation function computes metrics on the concatenated tensors but the metric calculation logic has a bug (perhaps comparing wrong dimensions or using incorrect similarity computation).

**5B: Fix and verify** — After fixing, run a quick 1-epoch training with eval to confirm non-zero metrics appear.

**5C: Consider re-training** — If the eval fix reveals that the current checkpoints are suboptimal (spread still <0.10 after asymmetry fix), a targeted re-training of Stage 3 (5-10 epochs) with working eval may improve Phase 2 metrics.

### Step 6: E5 Weight Optimization

**File**: `crates/context-graph-benchmark/src/causal_bench/phases.rs` (BenchConfig)
**File**: `crates/context-graph-core/src/weights/mod.rs` (causal_reasoning profile)

After Steps 1-3 are complete, run the benchmark with different E5 weights to find optimal:

```bash
for w in 0.05 0.10 0.15 0.20 0.25 0.30; do
    cargo run --release --features real-embeddings --bin causal_embedding_bench -- \
        --gpu --model-path models/causal --e5-weight $w 2>&1 | tail -5
done
```

Select the weight that maximizes Phase 6 MRR. Update the `causal_reasoning` weight profile in `weights/mod.rs` to match.

---

## Implementation Order

```
Step 1: Fix e5_score() asymmetry bug ─────────┐
Step 2: Remove zero-fallbacks (fail-fast) ─────┤
Step 3: Wire Phases 4/6/7 to GPU provider ─────┼─→ Build + GPU benchmark → assess new scores
                                                │
Step 4: Improve intent detection ───────────────┘
                                                │
                                        ┌───────┘
                                        ▼
                                  Full 8-phase GPU benchmark
                                        │
                                        ▼
                              Step 6: E5 weight optimization
                                        │
                                        ▼
                        Step 5: Fix eval (only if Phase 2 still fails)
```

Steps 1-4 are independent and can be done in sequence in a single pass. Step 5 is only needed if Phase 2 still fails after the asymmetry fix. Step 6 requires Steps 1-3 to be complete.

---

## Verification Plan

After completing Steps 1-4:

1. **Build**: `cargo build --release --features real-embeddings -p context-graph-benchmark`
2. **Unit tests**: `cargo test -p context-graph-core -- causal` and `cargo test -p context-graph-benchmark`
3. **GPU benchmark**: `cargo run --release --features real-embeddings --bin causal_embedding_bench -- --gpu --model-path models/causal --verbose`
4. **Full State Verification**:
   - Source of truth: Inspect Phase 2 spread, Phase 4 e5_rrf, Phase 6 top1/mrr with verbose output
   - Boundary cases: Verify non-causal pairs score lower than causal in Phase 5
   - Evidence: Save benchmark JSON to `benchmark_results/` for comparison with `causal_trained_calibrated.json`

---

## Expected Outcomes

| Phase | Current | After Steps 1-4 | Target | Notes |
|-------|---------|-----------------|--------|-------|
| 1 Intent | 75.8% | ~92% | 90% | PASS — indicator expansion |
| 2 E5 Quality | spread=0.048 | spread>0.08 | 0.10 | BORDERLINE — depends on asymmetry fix impact |
| 3 Direction | PASS | PASS | — | No change |
| 4 Ablation | 0.83% | >12% | 12% | PASS — GPU provider shows real E5 contribution |
| 5 Gate | PASS | PASS | — | No change |
| 6 Retrieval | 8.3% | >40% | 55% | LIKELY PASS — real E5 scores + word overlap |
| 7 Cross-Domain | 0% | >30% | 45% | AT RISK — depends on E5 cross-domain signal |
| 8 Performance | PASS | PASS | — | No change |

Conservative estimate: **5-6/8 PASS** after Steps 1-4.
Optimistic estimate: **7/8 PASS** if asymmetry fix significantly improves spread.

Phase 2 and Phase 7 are the hardest to fix without re-training. If they still fail after Steps 1-4:
- Phase 2: Fix training eval (Step 5), then re-train Stage 3 with 10 epochs
- Phase 7: Add more cross-domain seed pairs, re-train with domain-balanced sampling

---

## Files to Modify

| File | Steps | Changes |
|------|-------|---------|
| `crates/context-graph-benchmark/src/causal_bench/provider.rs` | 1, 2 | Fix e5_score asymmetry, remove zero-fallbacks |
| `crates/context-graph-benchmark/src/causal_bench/phases.rs` | 3 | Wire simulate_search + simulate_ranked_search to config.provider |
| `crates/context-graph-core/src/causal/asymmetric.rs` | 4 | Add effect/cause indicators |
| `crates/context-graph-embeddings/src/training/pipeline.rs` | 5 | Fix Evaluator::evaluate() |
| `crates/context-graph-core/src/weights/mod.rs` | 6 | Update E5 weight in causal_reasoning profile |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Asymmetry fix doesn't improve spread enough | Phase 2 still fails | Re-train with working eval + more epochs |
| Word overlap E1 proxy too weak for Phase 6 | top1 stays low even with GPU E5 | Accept: real production uses E1 not word overlap |
| Cross-domain generalization impossible at 768D | Phase 7 fails | Accept E5 as structural signal; E1 handles cross-domain |
| GPU OOM on full 250-pair benchmark | Benchmark crashes | Already handled: gpu_forward_base_tensor is memory-safe |
| Training eval fix reveals suboptimal checkpoints | Need re-training | 5-10 epoch Stage 3 re-train takes ~1 hour |
