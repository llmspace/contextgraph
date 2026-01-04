---
id: "M05-T05"
title: "Define LifecycleStage Enum (Marblestone)"
description: |
  Implement LifecycleStage enum with Marblestone-inspired dynamic learning rates.
  Variants: Infancy (0-50 interactions), Growth (50-500), Maturity (500+).
layer: "foundation"
status: "COMPLETE"
priority: "critical"
estimated_hours: 2
sequence: 5
depends_on: ["M05-T00"]
spec_refs:
  - "constitution.yaml lines 164-167 (lifecycle definitions)"
  - "contextprd.md Section 2.4 (Lifecycle Marblestone λ Weights)"
  - "learntheory.md (UTL formula: L = f((ΔS × ΔC) · wₑ · cos φ))"
files_implemented:
  - path: "crates/context-graph-utl/src/lifecycle/stage.rs"
    description: "LifecycleStage enum - 597 lines, 18 tests"
    verified: true
  - path: "crates/context-graph-utl/src/lifecycle/lambda.rs"
    description: "LifecycleLambdaWeights struct - 641 lines, 25 tests"
    verified: true
  - path: "crates/context-graph-utl/src/lifecycle/mod.rs"
    description: "Module re-exports"
    verified: true
  - path: "crates/context-graph-utl/src/lib.rs"
    description: "Crate-level re-exports at line 58"
    verified: true
git_reference: "f521803 feat(utl): complete context-graph-utl crate with 453 tests passing"
---

## CRITICAL: Task Status

**THIS TASK IS COMPLETE.** The LifecycleStage enum and LifecycleLambdaWeights are fully implemented and tested.

**Verification Command (MUST PASS):**
```bash
cargo test -p context-graph-utl lifecycle::stage -- --nocapture 2>&1 | grep -E "^test result|passed"
# Expected: test result: ok. 18 passed; 0 failed;
```

---

## What Already Exists

### File: `crates/context-graph-utl/src/lifecycle/stage.rs`

The LifecycleStage enum is implemented with:

**Variants:**
- `Infancy` - Early stage (0-50 interactions), novelty-seeking
- `Growth` - Intermediate stage (50-500 interactions), balanced
- `Maturity` - Mature stage (500+ interactions), coherence-focused

**Constants:**
- `INFANCY_THRESHOLD: u64 = 50`
- `GROWTH_THRESHOLD: u64 = 500`

**Methods Implemented:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `from_interaction_count` | `(count: u64) -> Self` | Determine stage from interaction count |
| `interaction_range` | `(&self) -> (u64, u64)` | Get (min, max) range for stage |
| `description` | `(&self) -> &'static str` | Human-readable description |
| `stance` | `(&self) -> &'static str` | Learning stance name |
| `next_stage` | `(&self) -> Option<Self>` | Get next stage if exists |
| `previous_stage` | `(&self) -> Option<Self>` | Get previous stage if exists |
| `can_transition_to` | `(&self, target: Self) -> bool` | Check if transition valid |
| `index` | `(&self) -> usize` | Numeric index (0, 1, 2) |
| `all` | `() -> [Self; 3]` | Array of all stages |

**Trait Implementations:**
- `Default` (returns `Infancy`)
- `Clone`, `Copy`, `Debug`, `PartialEq`, `Eq`, `Hash`
- `Serialize`, `Deserialize` (serde, lowercase)
- `Display` (stage name)
- `PartialOrd`, `Ord` (by index)

### File: `crates/context-graph-utl/src/lifecycle/lambda.rs`

The LifecycleLambdaWeights struct provides lambda weight computation:

**Marblestone Lambda Weights (from constitution.yaml:164-167):**

| Stage | lambda_s (ΔS) | lambda_c (ΔC) | Stance |
|-------|---------------|---------------|--------|
| Infancy | 0.7 | 0.3 | capture-novelty |
| Growth | 0.5 | 0.5 | balanced |
| Maturity | 0.3 | 0.7 | curation-coherence |

**Key Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `(lambda_s: f32, lambda_c: f32) -> UtlResult<Self>` | Create with validation (sum=1.0) |
| `for_stage` | `(stage: LifecycleStage) -> Self` | Get canonical weights for stage |
| `for_interaction_count` | `(count: u64) -> Self` | Get weights from count |
| `interpolated` | `(count: u64, config: &LifecycleConfig) -> Self` | Smooth transition weights |
| `lambda_s` | `(&self) -> f32` | Surprise weight getter |
| `lambda_c` | `(&self) -> f32` | Coherence weight getter |
| `apply` | `(&self, delta_s: f32, delta_c: f32) -> f32` | Apply weights to values |
| `is_valid` | `(&self) -> bool` | Check sum=1.0 invariant |
| `focus` | `(&self) -> &'static str` | "surprise"/"balanced"/"coherence" |

---

## Source of Truth

**Location:** `crates/context-graph-utl/src/lifecycle/stage.rs` and `lambda.rs`

**How to Verify:**
```bash
# 1. Check file exists and has correct structure
head -70 crates/context-graph-utl/src/lifecycle/stage.rs

# 2. Verify enum variants
grep -n "Infancy\|Growth\|Maturity" crates/context-graph-utl/src/lifecycle/stage.rs | head -10

# 3. Verify constants
grep -n "INFANCY_THRESHOLD\|GROWTH_THRESHOLD" crates/context-graph-utl/src/lifecycle/stage.rs

# 4. Verify lambda weights
grep -n "0.7.*0.3\|0.5.*0.5\|0.3.*0.7" crates/context-graph-utl/src/lifecycle/lambda.rs

# 5. Run all lifecycle tests
cargo test -p context-graph-utl lifecycle -- --nocapture
```

---

## Full State Verification

### Pre-Implementation State
Already implemented in commit `f521803`.

### Post-Implementation Verification

**1. Unit Tests MUST Pass:**
```bash
cargo test -p context-graph-utl lifecycle::stage -- --nocapture 2>&1 | tail -25
# MUST show: test result: ok. 18 passed; 0 failed;

cargo test -p context-graph-utl lifecycle::lambda -- --nocapture 2>&1 | tail -30
# MUST show: test result: ok. 25 passed; 0 failed;
```

**2. Re-exports MUST Work:**
```bash
cargo test -p context-graph-utl test_lifecycle_re_exports -- --nocapture
# MUST show: test test_lifecycle_re_exports ... ok
```

**3. Lambda Weight Invariant:**
```bash
cargo test -p context-graph-utl test_all_stages_weights_sum_to_one -- --nocapture
# MUST show: ok - proves lambda_s + lambda_c = 1.0 for all stages
```

---

## Edge Case Audit (3 Cases)

### Edge Case 1: Boundary Transitions (count = 49 vs 50)
```bash
# Verify Infancy ends at 49, Growth starts at 50
cargo test -p context-graph-utl test_from_interaction_count -- --nocapture
```
**Expected:** count=49 → Infancy, count=50 → Growth

### Edge Case 2: Maximum Interaction Count (u64::MAX)
```bash
# Verify u64::MAX is classified as Maturity
cargo test -p context-graph-utl test_from_interaction_count_maturity -- --nocapture 2>&1 || \
  grep "u64::MAX" crates/context-graph-utl/src/lifecycle/stage.rs
```
**Expected:** u64::MAX → Maturity (line 406 in stage.rs tests)

### Edge Case 3: Invalid Lambda Weight Sum
```bash
# Verify weights that don't sum to 1.0 are rejected
cargo test -p context-graph-utl test_new_invalid_sum -- --nocapture
# MUST show: ok - proves validation rejects 0.6 + 0.6 = 1.2
```

---

## Evidence of Success

Run this comprehensive verification:

```bash
#!/bin/bash
echo "=== M05-T05 VERIFICATION ==="
echo ""

echo "1. Stage Tests:"
cargo test -p context-graph-utl lifecycle::stage --lib 2>&1 | grep -E "^test result|passed"

echo ""
echo "2. Lambda Tests:"
cargo test -p context-graph-utl lifecycle::lambda --lib 2>&1 | grep -E "^test result|passed"

echo ""
echo "3. Manager Tests:"
cargo test -p context-graph-utl lifecycle::manager --lib 2>&1 | grep -E "^test result|passed"

echo ""
echo "4. All Lifecycle Tests Combined:"
cargo test -p context-graph-utl lifecycle --lib 2>&1 | grep -E "^test result"

echo ""
echo "5. File Line Counts:"
wc -l crates/context-graph-utl/src/lifecycle/*.rs

echo ""
echo "6. Re-export Verification:"
grep -n "pub use lifecycle" crates/context-graph-utl/src/lib.rs

echo ""
echo "=== VERIFICATION COMPLETE ==="
```

**Expected Output:**
```
1. Stage Tests:
test result: ok. 18 passed; 0 failed;

2. Lambda Tests:
test result: ok. 25 passed; 0 failed;

3. Manager Tests:
test result: ok. 24 passed; 0 failed;

4. All Lifecycle Tests Combined:
test result: ok. 67 passed; 0 failed;

5. File Line Counts:
  641 crates/context-graph-utl/src/lifecycle/lambda.rs
  529 crates/context-graph-utl/src/lifecycle/manager.rs
   54 crates/context-graph-utl/src/lifecycle/mod.rs
  597 crates/context-graph-utl/src/lifecycle/stage.rs
 1821 total

6. Re-export Verification:
58:pub use lifecycle::{LifecycleLambdaWeights, LifecycleManager, LifecycleStage};
```

---

## Manual Verification Checklist

Run each command and verify output matches expected:

| Check | Command | Expected |
|-------|---------|----------|
| Stage exists | `grep "pub enum LifecycleStage" crates/context-graph-utl/src/lifecycle/stage.rs` | `pub enum LifecycleStage {` |
| Has Infancy | `grep "Infancy," crates/context-graph-utl/src/lifecycle/stage.rs` | `Infancy,` |
| Has Growth | `grep "Growth," crates/context-graph-utl/src/lifecycle/stage.rs` | `Growth,` |
| Has Maturity | `grep "Maturity," crates/context-graph-utl/src/lifecycle/stage.rs` | `Maturity,` |
| Threshold 50 | `grep "INFANCY_THRESHOLD.*50" crates/context-graph-utl/src/lifecycle/stage.rs` | `pub const INFANCY_THRESHOLD: u64 = 50;` |
| Threshold 500 | `grep "GROWTH_THRESHOLD.*500" crates/context-graph-utl/src/lifecycle/stage.rs` | `pub const GROWTH_THRESHOLD: u64 = 500;` |
| Lambda sum test | `cargo test -p context-graph-utl test_all_stages_weights_sum_to_one 2>&1 \| grep ok` | `ok` |
| Serde works | `cargo test -p context-graph-utl lifecycle::stage::tests::test_serialization 2>&1 \| grep ok` | `ok` |

---

## Constitution Alignment

**From constitution.yaml lines 164-167:**
```yaml
lifecycle:  # Marblestone λ weights
  infancy:  { n: "0-50",   ΔS_trig: 0.9, ΔC_trig: 0.2, λ_ΔS: 0.7, λ_ΔC: 0.3, stance: "capture-novelty" }
  growth:   { n: "50-500", ΔS_trig: 0.7, ΔC_trig: 0.4, λ_ΔS: 0.5, λ_ΔC: 0.5, stance: "balanced" }
  maturity: { n: "500+",  ΔS_trig: 0.6, ΔC_trig: 0.5, λ_ΔS: 0.3, λ_ΔC: 0.7, stance: "curation-coherence" }
```

**Implementation matches:**
- Infancy: 0-50 ✓, λ_ΔS=0.7 ✓, λ_ΔC=0.3 ✓
- Growth: 50-500 ✓, λ_ΔS=0.5 ✓, λ_ΔC=0.5 ✓
- Maturity: 500+ ✓, λ_ΔS=0.3 ✓, λ_ΔC=0.7 ✓

---

## SHERLOCK-HOLMES FINAL VERIFICATION

**REQUIRED:** After completing any changes to this task, spawn a sherlock-holmes subagent to forensically verify:

```
Task: Verify M05-T05 LifecycleStage implementation is complete and correct

1. Verify file existence:
   - crates/context-graph-utl/src/lifecycle/stage.rs EXISTS
   - crates/context-graph-utl/src/lifecycle/lambda.rs EXISTS
   - crates/context-graph-utl/src/lifecycle/mod.rs EXISTS

2. Verify enum structure:
   - LifecycleStage has exactly 3 variants: Infancy, Growth, Maturity
   - Constants INFANCY_THRESHOLD=50 and GROWTH_THRESHOLD=500 exist

3. Verify lambda weights:
   - Infancy: lambda_s=0.7, lambda_c=0.3
   - Growth: lambda_s=0.5, lambda_c=0.5
   - Maturity: lambda_s=0.3, lambda_c=0.7
   - All weights sum to 1.0

4. Run ALL lifecycle tests:
   cargo test -p context-graph-utl lifecycle --lib
   MUST show: 67 passed; 0 failed

5. Verify no compilation errors:
   cargo build -p context-graph-utl
   MUST succeed with no errors

6. Cross-check with constitution.yaml lines 164-167
   - All values must match exactly
```

---

## Notes for AI Agents

1. **DO NOT RE-IMPLEMENT** - This task is complete
2. **VERIFY ONLY** - Run the verification commands above
3. **NO MOCKS** - All tests use real LifecycleStage instances
4. **FAIL FAST** - Invalid lambda weights error immediately via UtlError
5. **CONSTITUTION ALIGNMENT** - All values match constitution.yaml exactly

---

*Task Version: 2.0.0*
*Status: COMPLETE*
*Verified: 2026-01-04*
*Git: f521803*
*Tests: 67 passing (stage: 18, lambda: 25, manager: 24)*
