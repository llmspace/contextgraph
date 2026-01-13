# TASK-08: Convert MetaCognitiveProvider to async

## STATUS: COMPLETE (2026-01-13)

**Task ID**: TASK-08 (Original: TASK-PERF-003)
**Layer**: Foundation | **Phase**: 1
**Sequence**: 8
**Estimated Hours**: 1.5
**Depends On**: TASK-06 (COMPLETE - async-trait added)
**Blocks**: None

---

## CRITICAL CONTEXT FOR AI AGENTS

**READ THIS ENTIRE SECTION BEFORE IMPLEMENTING.**

### What This Project Is

This is a Rust project implementing a bio-nervous knowledge graph with Global Workspace Theory (GWT) consciousness, Kuramoto oscillator synchronization, and Unified Theory of Learning (UTL). Key architectural rules:

- **CUDA GPU required** - no CPU fallbacks (Constitution ARCH-08)
- **13 embedding spaces** - E1 through E13, each with dedicated purpose
- **Async-first** - tokio runtime, no blocking I/O (Constitution AP-08)
- **Fail fast** - no stubs, no mock data, no workarounds

### What This Task ACTUALLY Requires

The `MetaCognitiveProvider` trait in `gwt_traits.rs` has 4 methods:
- `evaluate` - ALREADY async (correct)
- `acetylcholine` - SYNC (MUST FIX)
- `monitoring_frequency` - SYNC (MUST FIX)
- `get_recent_scores` - SYNC (MUST FIX)

The 3 sync methods violate **Constitution AP-08**: "No sync I/O in async context"

The implementation in `gwt_providers.rs` uses `futures::executor::block_on()` which causes **DEADLOCKS on single-threaded tokio runtime**.

### Key Files (Verified Paths - 2026-01-13)

| File | Purpose | Line Numbers |
|------|---------|--------------|
| `crates/context-graph-mcp/src/handlers/gwt_traits.rs` | Trait definition - MODIFY THIS | Lines 199-227 |
| `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | Implementation - DO NOT MODIFY | Lines 402-428 |
| `crates/context-graph-core/src/gwt/meta_cognitive.rs` | Underlying MetaCognitiveLoop type | Referenced |

---

## CURRENT STATE (Verified 2026-01-13)

### gwt_traits.rs Lines 199-227 (CURRENT - BROKEN)

```rust
/// Provider trait for meta-cognitive loop operations.
/// TASK-GWT-001: Required for meta_score computation and dream triggering.
#[async_trait]
pub trait MetaCognitiveProvider: Send + Sync {
    /// Evaluate meta-cognitive score.
    ///
    /// MetaScore = sigmoid(2 x (L_predicted - L_actual))
    ///
    /// # Arguments
    /// - predicted_learning: L_predicted in [0, 1]
    /// - actual_learning: L_actual in [0, 1]
    #[allow(dead_code)]
    async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState>;

    /// Get current Acetylcholine level (learning rate modulator).
    fn acetylcholine(&self) -> f32;  // <-- SYNC (MUST FIX)

    /// Get current monitoring frequency (Hz).
    #[allow(dead_code)]
    fn monitoring_frequency(&self) -> f32;  // <-- SYNC (MUST FIX)

    /// Get recent meta-scores for trend analysis.
    #[allow(dead_code)]
    fn get_recent_scores(&self) -> Vec<f32>;  // <-- SYNC (MUST FIX)
}
```

### gwt_providers.rs Lines 414-427 (CURRENT - Uses block_on)

```rust
fn acetylcholine(&self) -> f32 {
    let meta_cognitive = futures::executor::block_on(self.meta_cognitive.read());
    meta_cognitive.acetylcholine()
}

fn monitoring_frequency(&self) -> f32 {
    let meta_cognitive = futures::executor::block_on(self.meta_cognitive.read());
    meta_cognitive.monitoring_frequency()
}

fn get_recent_scores(&self) -> Vec<f32> {
    let meta_cognitive = futures::executor::block_on(self.meta_cognitive.read());
    meta_cognitive.get_recent_scores()
}
```

**THE PROBLEM**: `block_on()` blocks the executor thread while waiting for a `TokioRwLock` that can only be released by that same blocked executor. Result: **DEADLOCK**.

---

## DEFINITION OF DONE

### Target Signature (gwt_traits.rs)

```rust
// crates/context-graph-mcp/src/handlers/gwt_traits.rs

use async_trait::async_trait;
use context_graph_core::error::CoreResult;
use context_graph_core::gwt::MetaCognitiveState;

/// Provider trait for meta-cognitive loop operations.
///
/// All methods are async per Constitution AP-08: "No sync I/O in async context".
/// TASK-GWT-001: Required for meta_score computation and dream triggering.
#[async_trait]
pub trait MetaCognitiveProvider: Send + Sync {
    /// Evaluate meta-cognitive score.
    ///
    /// MetaScore = sigmoid(2 x (L_predicted - L_actual))
    ///
    /// # Arguments
    /// - predicted_learning: L_predicted in [0, 1]
    /// - actual_learning: L_actual in [0, 1]
    #[allow(dead_code)]
    async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState>;

    /// Get current Acetylcholine level (learning rate modulator).
    async fn acetylcholine(&self) -> f32;

    /// Get current monitoring frequency (Hz).
    #[allow(dead_code)]
    async fn monitoring_frequency(&self) -> f32;

    /// Get recent meta-scores for trend analysis.
    #[allow(dead_code)]
    async fn get_recent_scores(&self) -> Vec<f32>;
}
```

### Constraints (MUST ALL PASS)

| Constraint | Requirement | Verification |
|------------|-------------|--------------|
| C1 | Trait has `#[async_trait]` attribute | `grep "#\[async_trait\]" gwt_traits.rs` |
| C2 | ALL 4 methods are `async fn` | `grep -c "async fn" \| grep "4"` |
| C3 | Trait requires `Send + Sync` | `grep "Send + Sync"` |
| C4 | Documentation references AP-08 | `grep "AP-08"` |
| C5 | Return types UNCHANGED | `-> f32`, `-> Vec<f32>` (NOT `Result<T>`) |

---

## IMPLEMENTATION STEPS

### Step 1: Read Current Trait (Verify State)

```bash
# Verify you're modifying the correct file
grep -n "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs
# Expected: Line ~202
```

### Step 2: Modify gwt_traits.rs Lines 199-227

Change ONLY these 3 lines:

```rust
// Line 218: BEFORE
fn acetylcholine(&self) -> f32;
// AFTER
async fn acetylcholine(&self) -> f32;

// Line 222: BEFORE
fn monitoring_frequency(&self) -> f32;
// AFTER
async fn monitoring_frequency(&self) -> f32;

// Line 226: BEFORE
fn get_recent_scores(&self) -> Vec<f32>;
// AFTER
async fn get_recent_scores(&self) -> Vec<f32>;
```

### Step 3: Add AP-08 Documentation

Update the trait docstring (Line 199-200):

```rust
// BEFORE
/// Provider trait for meta-cognitive loop operations.
/// TASK-GWT-001: Required for meta_score computation and dream triggering.

// AFTER
/// Provider trait for meta-cognitive loop operations.
///
/// All methods are async per Constitution AP-08: "No sync I/O in async context".
/// TASK-GWT-001: Required for meta_score computation and dream triggering.
```

### Step 4: DO NOT MODIFY gwt_providers.rs

**CRITICAL**: This task ONLY modifies the trait definition. The implementation fix (removing `block_on()`) is a SEPARATE concern. After this task, compilation WILL FAIL - that is expected and correct.

---

## VERIFICATION COMMANDS

### Primary Verification

```bash
# Step 1: Count async methods in MetaCognitiveProvider trait
grep -A 30 "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "^\s*async fn" | wc -l
# Expected: 4 (all 4 methods async)

# Step 2: Verify no sync fn remains
grep -A 30 "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "^\s*fn [a-z]" | wc -l
# Expected: 0 (no sync methods)

# Step 3: Verify AP-08 documented
grep -B 3 "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -q "AP-08" && echo "PASS: AP-08 documented" || echo "FAIL: Missing AP-08"

# Step 4: Verify async-trait attribute
grep -B 1 "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -q "#\[async_trait\]" && echo "PASS: async_trait present" || echo "FAIL"
```

### Compilation Behavior (Expected Failure)

```bash
# After trait change, this WILL fail - that is CORRECT
cargo check -p context-graph-mcp 2>&1 | head -50

# Expected error pattern:
# error[E0195]: lifetime parameters or bounds on method `acetylcholine` do not match
#    --> crates/context-graph-mcp/src/handlers/gwt_providers.rs:414:23
```

This failure proves the trait change was applied. The implementation fix is a downstream task.

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth

The **gwt_traits.rs file** is the source of truth. The trait definition must have 4 async methods.

### Execute & Inspect Protocol

**1. BEFORE State Capture:**
```bash
echo "=== BEFORE STATE ===" > /tmp/task08-verification.log
echo "File: crates/context-graph-mcp/src/handlers/gwt_traits.rs" >> /tmp/task08-verification.log
grep -n "fn " crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "acetylcholine|monitoring_frequency|get_recent_scores" >> /tmp/task08-verification.log
echo "Sync methods count: $(grep -A 30 'pub trait MetaCognitiveProvider' \
  crates/context-graph-mcp/src/handlers/gwt_traits.rs | grep -E '^\s*fn [a-z]' | wc -l)" >> /tmp/task08-verification.log
```

**2. AFTER Modification:**
```bash
echo "" >> /tmp/task08-verification.log
echo "=== AFTER STATE ===" >> /tmp/task08-verification.log
grep -n "async fn" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "acetylcholine|monitoring_frequency|get_recent_scores" >> /tmp/task08-verification.log
echo "Async methods count: $(grep -A 30 'pub trait MetaCognitiveProvider' \
  crates/context-graph-mcp/src/handlers/gwt_traits.rs | grep -E '^\s*async fn' | wc -l)" >> /tmp/task08-verification.log
```

**3. Evidence of Success:**
```bash
cat /tmp/task08-verification.log
# BEFORE should show: 3 sync methods (fn acetylcholine, fn monitoring_frequency, fn get_recent_scores)
# AFTER should show: 4 async methods total
```

---

## BOUNDARY & EDGE CASE AUDIT

### Edge Case 1: Empty/Zero Return Values

**Input**: Meta-cognitive loop with no history
**Expected**: `acetylcholine()` returns default (1.0), `get_recent_scores()` returns empty Vec
**Verification**: Return types are `f32` and `Vec<f32>`, not `Result<T>` - no error wrapping

### Edge Case 2: Trait Object Compatibility

```rust
// This MUST still compile after changes
fn accepts_dyn_meta(provider: &dyn MetaCognitiveProvider) {
    // async_trait handles dyn compatibility via Box<dyn Future>
}
```

**Verification**:
```bash
grep -r "dyn MetaCognitiveProvider" crates/context-graph-mcp/src/ 2>/dev/null
# If matches exist, verify they compile after trait change
```

### Edge Case 3: Send + Sync Bounds

```rust
// Trait must remain compatible with Arc<dyn MetaCognitiveProvider>
fn spawn_task<P: MetaCognitiveProvider + 'static>(p: Arc<P>) {
    tokio::spawn(async move {
        p.acetylcholine().await;  // Must be Send + Sync
    });
}
```

**Verification**: Trait definition includes `: Send + Sync`

---

## MANUAL TESTING WITH SYNTHETIC DATA

### Test 1: Trait Signature Extraction

```bash
# Extract exact trait signature
grep -A 40 "pub trait MetaCognitiveProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs

# Expected output shows:
# async fn evaluate(...)
# async fn acetylcholine(...)
# async fn monitoring_frequency(...)
# async fn get_recent_scores(...)
```

### Test 2: Compilation Behavior

```bash
# After trait modification, run:
cargo check -p context-graph-mcp 2>&1 | tee /tmp/check-output.log

# Verify error is about async mismatch:
grep -E "lifetime|async|E0195" /tmp/check-output.log && echo "EXPECTED FAILURE" || echo "UNEXPECTED"
```

### Test 3: No Backward Compatibility Hacks

```bash
# Verify no compat shims added
grep -r "compat\|legacy\|deprecated\|_sync\|_blocking" \
  crates/context-graph-mcp/src/handlers/gwt_traits.rs && exit 1 || echo "PASS: No compat hacks"
```

---

## BREAKING CHANGES (INTENTIONAL)

This is a **BREAKING API CHANGE**. The following will fail to compile until implementation is updated:

1. `MetaCognitiveProviderImpl` in `gwt_providers.rs` (lines 402-428)
2. Any tests calling sync methods without `.await`
3. Any code using `dyn MetaCognitiveProvider` with sync calls

**THIS IS EXPECTED AND CORRECT.** Do NOT create workarounds or fallbacks.

---

## COMMON MISTAKES TO AVOID

| Mistake | Why It's Wrong | What To Do Instead |
|---------|----------------|-------------------|
| Modify gwt_providers.rs | Wrong task scope | Only modify gwt_traits.rs |
| Add `-> Result<f32>` | Changes API contract | Keep original return types |
| Add default implementations | Creates sync fallback | No defaults - fail fast |
| Add backward compat shims | Creates technical debt | Clean break only |
| Suppress compiler errors | Hides real problems | Let errors surface |
| Use mock data in tests | Hides integration issues | Use real types |

---

## DEPENDENCY CHAIN

```
TASK-06 (COMPLETE)     TASK-08 (THIS TASK)
async-trait added  -->  Trait made async
                        |
                        v
                   COMPILATION FAILS (expected)
                        |
                        v
                   [Future task: Fix gwt_providers.rs]
```

---

## RELATED CONTEXT

### Constitution Rules (from docs2/constitution.yaml)

- **AP-08**: "No sync I/O in async context" - this task enforces compliance
- **ARCH-08**: "CUDA GPU required for production - no CPU fallbacks"

### Related Issues

- **ISS-004**: `block_on()` deadlock risk (CRITICAL) - this task is step 1 of fix
- **REQ-PERF-003**: MetaCognitiveProviderImpl async methods

### Parallel Tasks

- **TASK-07**: Convert WorkspaceProvider to async (COMPLETE - uncommitted)
- **TASK-16**: Remove block_on from gwt_providers (depends on TASK-07, TASK-08)

---

## ROLLBACK PROCEDURE

If task must be reverted:

```bash
git checkout HEAD -- crates/context-graph-mcp/src/handlers/gwt_traits.rs
cargo check -p context-graph-mcp  # Should pass (original state)
```

---

## SUCCESS CRITERIA CHECKLIST

After completing the task, verify ALL of these:

- [ ] All 4 MetaCognitiveProvider methods are `async fn`
- [ ] Trait still has `#[async_trait]` attribute
- [ ] Trait still requires `Send + Sync`
- [ ] Documentation includes AP-08 reference
- [ ] Return types unchanged (`f32`, `Vec<f32>`)
- [ ] `cargo check -p context-graph-mcp` fails with async mismatch error (expected)
- [ ] No implementation changes in this task
- [ ] No backward compatibility hacks added
- [ ] Verification log captured in /tmp/task08-verification.log
- [ ] Before/after state logged showing the change

---

## NEXT TASK

After TASK-08 is complete, the next tasks in sequence are:
- **TASK-09**: Fix Johari Blind/Unknown action mapping

Tasks that depend on TASK-08 being complete:
- **TASK-16**: Remove block_on from gwt_providers (requires both TASK-07 and TASK-08 complete)

---

*Task Specification v3.0 - Audited 2026-01-13 against actual codebase state*
*No ambiguity. Fail fast. No workarounds.*
