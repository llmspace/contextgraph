# TASK-21: Implement TriggerManager IC Checking

## METADATA
| Field | Value |
|-------|-------|
| Task ID | TASK-21 (Original: TASK-IDENTITY-003) |
| Status | **COMPLETE** |
| Completed | 2026-01-13 |
| Layer | Integration |
| Phase | 3 |
| Sequence | 21 |
| Implements | REQ-IDENTITY-003, REQ-IDENTITY-007 |
| Dependencies | TASK-20 (COMPLETE), TASK-22 (NOT STARTED) |
| Blocks | TASK-24 (DreamEventListener wiring), TASK-38 (MCP tool) |
| Est. Hours | 3 |
| Actual Hours | ~2 |

---

## ⚠️ CRITICAL: WHAT THIS TASK ACTUALLY REQUIRES

**PROBLEM**: The current `TriggerManager` struct does NOT use the `TriggerConfig` struct and does NOT have IC checking capability.

**CURRENT STATE** (verified 2026-01-13):
- `TriggerConfig` EXISTS at `triggers.rs:43-56` ✅
- `TriggerManager` EXISTS at `triggers.rs:180-201` ✅
- `ExtendedTriggerReason::IdentityCritical { ic_value }` EXISTS at `types.rs:564-567` ✅
- **TriggerManager does NOT use TriggerConfig** ❌
- **TriggerManager does NOT have IC checking** ❌
- **TriggerManager does NOT have a `current_ic` field** ❌
- **`check_triggers()` does NOT check for IdentityCritical** ❌

**DEPENDENCY STATUS**:
- TASK-20: `TriggerConfig` struct ✅ COMPLETE
- TASK-22: `GpuMonitor` trait ❌ NOT STARTED (but task spec says TASK-21 can proceed with a placeholder)

---

## CONSTITUTION REQUIREMENTS (MUST COMPLY)

From `docs2/constitution.yaml`:

```yaml
gwt:
  self_ego_node:
    identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t)"
    thresholds:
      healthy: ">0.9"
      warning: "<0.7"
      critical: "<0.5 → dream"  # MUST trigger dream when IC < 0.5

forbidden:
  AP-26: "IC<0.5 MUST trigger dream - no silent failures"
  AP-38: "IC<0.5 MUST auto-trigger dream"

enforcement:
  IDENTITY-007: "IC < 0.5 → auto-trigger dream"
```

**KEY RULE**: When IC drops below 0.5, the system MUST automatically trigger a dream for identity consolidation. No silent failures allowed.

---

## CURRENT CODE ANALYSIS

### File: `crates/context-graph-core/src/dream/triggers.rs`

**Current TriggerManager struct (lines 180-201):**
```rust
pub struct TriggerManager {
    /// Entropy tracking window
    entropy_window: EntropyWindow,
    /// GPU utilization state
    gpu_state: GpuTriggerState,
    /// Whether manual trigger was requested
    manual_trigger: bool,
    /// Last trigger reason (for reporting)
    last_trigger_reason: Option<ExtendedTriggerReason>,
    /// Cooldown after trigger (to prevent rapid re-triggering)
    trigger_cooldown: Duration,
    /// Last trigger time
    last_trigger_time: Option<Instant>,
    /// Whether triggers are enabled
    enabled: bool,
}
```

**MISSING FIELDS THAT MUST BE ADDED**:
- `config: TriggerConfig` - Use the TriggerConfig from TASK-20
- `current_ic: Option<f32>` - Current Identity Continuity value

**Current `check_triggers()` method (lines 304-334):**
```rust
pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
    if !self.enabled { return None; }

    // Check cooldown (manual trigger bypasses cooldown)
    if !self.manual_trigger {
        if let Some(last_time) = self.last_trigger_time {
            if last_time.elapsed() < self.trigger_cooldown { return None; }
        }
    }

    // Priority 1: Manual
    if self.manual_trigger {
        return Some(ExtendedTriggerReason::Manual);
    }

    // Priority 2: GPU (WRONG - should be IdentityCritical)
    if self.gpu_state.should_trigger() {
        return Some(ExtendedTriggerReason::GpuOverload);
    }

    // Priority 3: Entropy
    if self.entropy_window.should_trigger() {
        return Some(ExtendedTriggerReason::HighEntropy);
    }

    None
}
```

**PROBLEM**: The `IdentityCritical` check is completely missing!

---

## EXACT IMPLEMENTATION REQUIREMENTS

### Step 1: Add fields to TriggerManager

**Location**: `crates/context-graph-core/src/dream/triggers.rs:180`

**ADD these fields to the struct:**
```rust
pub struct TriggerManager {
    /// Configuration with thresholds (NEW)
    config: TriggerConfig,

    /// Current Identity Continuity value (NEW)
    /// None = not yet measured, Some(x) = current IC
    current_ic: Option<f32>,

    /// Entropy tracking window
    entropy_window: EntropyWindow,
    // ... rest remains the same
}
```

### Step 2: Add `update_identity_coherence()` method

**NEW METHOD** - add after `update_entropy()`:
```rust
/// Update the current Identity Continuity value.
///
/// # Arguments
///
/// * `ic` - Current IC value, expected in [0.0, 1.0]
///
/// # Clamping Behavior
///
/// - NaN → clamped to 0.0 (worst case) with warning
/// - Infinity → clamped to 1.0 (best case) with warning
/// - Out of range → clamped to [0.0, 1.0] with warning
///
/// # Constitution
///
/// Per AP-10: No NaN/Infinity in UTL values.
pub fn update_identity_coherence(&mut self, ic: f32) {
    if !self.enabled {
        return;
    }

    let ic = if ic.is_nan() {
        tracing::warn!("Invalid IC value NaN, clamping to 0.0 per AP-10");
        0.0
    } else if ic.is_infinite() {
        tracing::warn!("Invalid IC value Infinity, clamping to 1.0 per AP-10");
        1.0
    } else if !(0.0..=1.0).contains(&ic) {
        tracing::warn!("IC value {} out of range, clamping to [0.0, 1.0]", ic);
        ic.clamp(0.0, 1.0)
    } else {
        ic
    };

    self.current_ic = Some(ic);

    if self.config.is_identity_critical(ic) {
        tracing::debug!("IC {} < threshold {} - identity critical state", ic, self.config.ic_threshold);
    }
}
```

### Step 3: Add `check_identity_continuity()` helper

**NEW METHOD**:
```rust
/// Check if identity continuity is in crisis state.
///
/// # Returns
///
/// `true` if `current_ic < config.ic_threshold`
///
/// # Constitution
///
/// Per gwt.self_ego_node.thresholds.critical: IC < 0.5 is critical.
#[inline]
pub fn check_identity_continuity(&self) -> bool {
    match self.current_ic {
        Some(ic) => self.config.is_identity_critical(ic),
        None => false, // No IC measured yet, cannot be critical
    }
}
```

### Step 4: Update `check_triggers()` with correct priority order

**MODIFY** the existing `check_triggers()` method:

```rust
/// Check all trigger conditions and return highest priority trigger.
///
/// # Priority Order (highest first)
///
/// 1. Manual - User-initiated, bypasses cooldown
/// 2. IdentityCritical - IC < 0.5 (AP-26, AP-38, IDENTITY-007)
/// 3. GpuOverload - GPU approaching 30% budget
/// 4. HighEntropy - Entropy > 0.7 for 5 minutes
///
/// # Returns
///
/// * `Some(reason)` - If trigger condition met
/// * `None` - If no trigger condition met or in cooldown
///
/// # Constitution Compliance
///
/// - Manual bypasses cooldown (highest priority)
/// - IdentityCritical MUST trigger when IC < 0.5 (AP-26, AP-38)
/// - GpuOverload when GPU > 30% (Constitution dream.constraints.gpu)
/// - HighEntropy when entropy > 0.7 for 5min (Constitution dream.trigger.entropy)
pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
    if !self.enabled {
        return None;
    }

    // Check cooldown (manual trigger bypasses cooldown)
    if !self.manual_trigger {
        if let Some(last_time) = self.last_trigger_time {
            if last_time.elapsed() < self.trigger_cooldown {
                return None;
            }
        }
    }

    // Priority 1: Manual (highest)
    if self.manual_trigger {
        return Some(ExtendedTriggerReason::Manual);
    }

    // Priority 2: IdentityCritical (CONSTITUTION CRITICAL - AP-26, AP-38)
    if let Some(ic) = self.current_ic {
        if self.config.is_identity_critical(ic) {
            return Some(ExtendedTriggerReason::IdentityCritical { ic_value: ic });
        }
    }

    // Priority 3: GpuOverload
    if self.gpu_state.should_trigger() {
        return Some(ExtendedTriggerReason::GpuOverload);
    }

    // Priority 4: HighEntropy
    if self.entropy_window.should_trigger() {
        return Some(ExtendedTriggerReason::HighEntropy);
    }

    None
}
```

### Step 5: Update constructors to use TriggerConfig

**MODIFY** `new()`:
```rust
/// Create a new trigger manager with constitution defaults.
pub fn new() -> Self {
    let config = TriggerConfig::default();
    Self {
        config,
        current_ic: None,
        entropy_window: EntropyWindow::new(),
        gpu_state: GpuTriggerState::new(),
        manual_trigger: false,
        last_trigger_reason: None,
        trigger_cooldown: config.cooldown, // Use config cooldown
        last_trigger_time: None,
        enabled: true,
    }
}

/// Create with custom config.
pub fn with_config(config: TriggerConfig) -> Self {
    config.validate(); // Fail-fast per AP-26
    Self {
        trigger_cooldown: config.cooldown,
        config,
        current_ic: None,
        entropy_window: EntropyWindow::new(),
        gpu_state: GpuTriggerState::new(),
        manual_trigger: false,
        last_trigger_reason: None,
        last_trigger_time: None,
        enabled: true,
    }
}
```

### Step 6: Add accessor for current_ic

**NEW METHOD**:
```rust
/// Get current Identity Continuity value.
#[inline]
pub fn current_ic(&self) -> Option<f32> {
    self.current_ic
}

/// Get current IC threshold from config.
#[inline]
pub fn ic_threshold(&self) -> f32 {
    self.config.ic_threshold
}
```

---

## REQUIRED TESTS

**Add these tests to the `#[cfg(test)] mod tests` section:**

```rust
// ============ Identity Continuity Trigger Tests ============

#[test]
fn test_trigger_manager_ic_check_triggers_below_threshold() {
    let mut manager = TriggerManager::new();

    // IC = 0.49 < 0.5 threshold → should trigger IdentityCritical
    manager.update_identity_coherence(0.49);

    let trigger = manager.check_triggers();
    assert!(trigger.is_some(), "IC below threshold should trigger");

    match trigger.unwrap() {
        ExtendedTriggerReason::IdentityCritical { ic_value } => {
            assert!((ic_value - 0.49).abs() < 0.001, "IC value should be preserved");
        }
        other => panic!("Expected IdentityCritical, got {:?}", other),
    }
}

#[test]
fn test_trigger_manager_ic_at_threshold_no_trigger() {
    let mut manager = TriggerManager::new();

    // IC = 0.5 (exactly at threshold) → should NOT trigger
    // Constitution: IC < 0.5 is critical (strict less than)
    manager.update_identity_coherence(0.5);

    assert!(!manager.check_identity_continuity(), "IC at threshold should not be critical");
}

#[test]
fn test_trigger_manager_ic_above_threshold_no_trigger() {
    let mut manager = TriggerManager::new();

    // IC = 0.9 (healthy) → should not trigger
    manager.update_identity_coherence(0.9);

    assert!(!manager.check_identity_continuity(), "IC above threshold should not be critical");
    assert!(manager.check_triggers().is_none(), "No trigger expected for healthy IC");
}

#[test]
fn test_trigger_manager_ic_priority_over_gpu() {
    let mut manager = TriggerManager::new();

    // Set up BOTH IC crisis AND GPU overload
    manager.update_identity_coherence(0.3);
    manager.update_gpu_usage(0.35);

    // IdentityCritical should have higher priority than GpuOverload
    let trigger = manager.check_triggers();
    match trigger {
        Some(ExtendedTriggerReason::IdentityCritical { .. }) => {}, // Expected
        other => panic!("Expected IdentityCritical to have priority, got {:?}", other),
    }
}

#[test]
fn test_trigger_manager_manual_priority_over_ic() {
    let mut manager = TriggerManager::new();

    // Set up IC crisis
    manager.update_identity_coherence(0.3);

    // Request manual trigger
    manager.request_manual_trigger();

    // Manual should have highest priority
    assert_eq!(manager.check_triggers(), Some(ExtendedTriggerReason::Manual));
}

#[test]
fn test_trigger_manager_ic_nan_handling() {
    let mut manager = TriggerManager::new();

    // NaN should be clamped to 0.0 per AP-10
    manager.update_identity_coherence(f32::NAN);

    // Should trigger (0.0 < 0.5)
    let trigger = manager.check_triggers();
    match trigger {
        Some(ExtendedTriggerReason::IdentityCritical { ic_value }) => {
            assert_eq!(ic_value, 0.0, "NaN should clamp to 0.0");
        }
        other => panic!("Expected IdentityCritical, got {:?}", other),
    }
}

#[test]
fn test_trigger_manager_ic_infinity_handling() {
    let mut manager = TriggerManager::new();

    // Infinity should be clamped to 1.0 per AP-10
    manager.update_identity_coherence(f32::INFINITY);

    // Should NOT trigger (1.0 >= 0.5)
    assert!(!manager.check_identity_continuity());
}

#[test]
fn test_trigger_manager_with_custom_config() {
    let config = TriggerConfig::default()
        .with_ic_threshold(0.6);  // Higher threshold for more sensitive detection

    let mut manager = TriggerManager::with_config(config);

    // IC = 0.55 < 0.6 (custom threshold) → should trigger
    manager.update_identity_coherence(0.55);

    assert!(manager.check_identity_continuity());

    match manager.check_triggers() {
        Some(ExtendedTriggerReason::IdentityCritical { ic_value }) => {
            assert!((ic_value - 0.55).abs() < 0.001);
        }
        other => panic!("Expected IdentityCritical, got {:?}", other),
    }
}

#[test]
fn test_trigger_manager_no_ic_measured_no_trigger() {
    let manager = TriggerManager::new();

    // No IC has been set → should not be critical
    assert!(!manager.check_identity_continuity());
    assert!(manager.current_ic().is_none());
}
```

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth
The source of truth for this task is:
1. `TriggerManager.check_triggers()` returning `IdentityCritical` when IC < 0.5
2. The `current_ic` field in `TriggerManager`
3. Test results from `cargo test trigger_manager`

### Execute & Inspect Protocol

**After implementation, run these commands:**

```bash
# 1. Verify struct has new fields
grep -A 20 "pub struct TriggerManager" crates/context-graph-core/src/dream/triggers.rs

# 2. Verify check_triggers includes IdentityCritical check
grep -A 30 "Priority 2: IdentityCritical" crates/context-graph-core/src/dream/triggers.rs

# 3. Verify update_identity_coherence exists
grep -A 15 "pub fn update_identity_coherence" crates/context-graph-core/src/dream/triggers.rs

# 4. Run IC-specific tests
cargo test -p context-graph-core test_trigger_manager_ic -- --nocapture

# 5. Run all trigger tests
cargo test -p context-graph-core trigger_manager -- --nocapture
```

### Boundary & Edge Case Audit

**Edge Case 1: IC exactly at 0.5**
```
INPUT: update_identity_coherence(0.5)
EXPECTED: check_identity_continuity() returns false
          check_triggers() does NOT return IdentityCritical
WHY: Constitution says IC < 0.5, not IC <= 0.5
```

**Edge Case 2: IC = 0.0 (minimum)**
```
INPUT: update_identity_coherence(0.0)
EXPECTED: check_identity_continuity() returns true
          check_triggers() returns IdentityCritical { ic_value: 0.0 }
```

**Edge Case 3: IC = NaN**
```
INPUT: update_identity_coherence(f32::NAN)
EXPECTED: Warning logged
          IC clamped to 0.0
          IdentityCritical triggered
WHY: AP-10 prohibits NaN, fail-fast to worst case
```

### Evidence of Success Log

After running tests, you should see:
```
running X tests
test dream::triggers::tests::test_trigger_manager_ic_check_triggers_below_threshold ... ok
test dream::triggers::tests::test_trigger_manager_ic_at_threshold_no_trigger ... ok
test dream::triggers::tests::test_trigger_manager_ic_above_threshold_no_trigger ... ok
test dream::triggers::tests::test_trigger_manager_ic_priority_over_gpu ... ok
test dream::triggers::tests::test_trigger_manager_manual_priority_over_ic ... ok
test dream::triggers::tests::test_trigger_manager_ic_nan_handling ... ok
test dream::triggers::tests::test_trigger_manager_ic_infinity_handling ... ok
test dream::triggers::tests::test_trigger_manager_with_custom_config ... ok
test dream::triggers::tests::test_trigger_manager_no_ic_measured_no_trigger ... ok
```

---

## MANUAL TESTING PROTOCOL

### Test 1: Compile Check
```bash
cargo check -p context-graph-core 2>&1 | head -20
```
**Expected**: No errors

### Test 2: IC Below Threshold Triggers Dream
```bash
cargo test -p context-graph-core test_trigger_manager_ic_check_triggers_below_threshold -- --nocapture
```
**Expected**: Test passes, IdentityCritical with ic_value=0.49 returned

### Test 3: Priority Order Verification
```bash
cargo test -p context-graph-core test_trigger_manager_ic_priority_over_gpu -- --nocapture
```
**Expected**: IdentityCritical takes precedence over GpuOverload

### Test 4: Full Trigger Manager Suite
```bash
cargo test -p context-graph-core trigger_manager -- --test-threads=1 --nocapture 2>&1 | tee /tmp/trigger_tests.log
grep -E "(PASSED|FAILED|ok|FAILED)" /tmp/trigger_tests.log
```
**Expected**: All tests pass

---

## CONSTRAINTS (MUST FOLLOW)

1. **Priority Order MUST be**: Manual > IdentityCritical > GpuOverload > HighEntropy
2. **IC threshold MUST use `<` not `<=`** - per Constitution "IC < 0.5"
3. **NaN/Infinity MUST be handled** - clamp with warning per AP-10
4. **`check_triggers()` MUST return Result or Option** - no silent failures
5. **No backwards compatibility hacks** - clean implementation
6. **No mock data in tests** - use real f32 values
7. **TriggerConfig MUST be validated** - panic on invalid config per AP-26

---

## FILES TO MODIFY

| File | Change |
|------|--------|
| `crates/context-graph-core/src/dream/triggers.rs` | Add `config`, `current_ic` fields to TriggerManager |
| `crates/context-graph-core/src/dream/triggers.rs` | Add `update_identity_coherence()` method |
| `crates/context-graph-core/src/dream/triggers.rs` | Add `check_identity_continuity()` method |
| `crates/context-graph-core/src/dream/triggers.rs` | Modify `check_triggers()` to include IC check |
| `crates/context-graph-core/src/dream/triggers.rs` | Update constructors to use TriggerConfig |
| `crates/context-graph-core/src/dream/triggers.rs` | Add IC-related tests |

---

## OUT OF SCOPE (DO NOT IMPLEMENT)

- GpuMonitor trait refactor (TASK-22)
- NvmlGpuMonitor implementation (TASK-23)
- DreamEventListener wiring (TASK-24)
- MCP tool exposure (TASK-38)
- GpuMonitor generic parameter in TriggerManager (future optimization)

---

## TROUBLESHOOTING

### "Cannot find TriggerConfig"
- Verify `TriggerConfig` is exported from `triggers.rs`
- Verify import: `use super::TriggerConfig;` if needed

### "Method check_triggers has incompatible return type"
- The method returns `Option<ExtendedTriggerReason>` (not Result)
- Do NOT change this signature - it's intentional

### "Unused field current_ic"
- Make sure `check_triggers()` uses `self.current_ic`
- Make sure `update_identity_coherence()` sets `self.current_ic`

### Tests fail with "expected IdentityCritical, got None"
- Verify cooldown is not blocking the trigger
- Verify `enabled` is true
- Verify `update_identity_coherence()` was called before `check_triggers()`

---

## RELATED TASKS

| Task | Relationship |
|------|-------------|
| TASK-19 | COMPLETE - provides IdentityCritical enum variant |
| TASK-20 | COMPLETE - provides TriggerConfig struct |
| TASK-22 | Parallel work - GpuMonitor trait (not a blocker) |
| TASK-24 | Blocked by this - DreamEventListener wiring |
| TASK-38 | Blocked by this - MCP get_identity_continuity tool |

---

## COMPLETION NOTES (2026-01-13)

### Changes Implemented

| File | Change |
|------|--------|
| `triggers.rs:189-217` | Added `config: TriggerConfig` and `current_ic: Option<f32>` fields to `TriggerManager` |
| `triggers.rs:228-241` | Updated `new()` to use `TriggerConfig::default()` |
| `triggers.rs:251-264` | Added `with_config(config: TriggerConfig)` constructor |
| `triggers.rs:342-385` | Added `update_identity_coherence()` and `check_identity_continuity()` methods |
| `triggers.rs:400-457` | Updated `check_triggers()` with IdentityCritical at Priority 2 |
| `triggers.rs:522-535` | Added `current_ic()` and `ic_threshold()` accessor methods |
| `triggers.rs:1083-1295` | Added 15 new IC-specific tests |

### Test Results

```
running 43 tests (dream::triggers module)
test dream::triggers::tests::test_trigger_manager_ic_check_triggers_below_threshold ... ok
test dream::triggers::tests::test_trigger_manager_ic_at_threshold_no_trigger ... ok
test dream::triggers::tests::test_trigger_manager_ic_above_threshold_no_trigger ... ok
test dream::triggers::tests::test_trigger_manager_ic_priority_over_gpu ... ok
test dream::triggers::tests::test_trigger_manager_manual_priority_over_ic ... ok
test dream::triggers::tests::test_trigger_manager_ic_nan_handling ... ok
test dream::triggers::tests::test_trigger_manager_ic_infinity_handling ... ok
test dream::triggers::tests::test_trigger_manager_with_custom_config ... ok
test dream::triggers::tests::test_trigger_manager_no_ic_measured_no_trigger ... ok
test dream::triggers::tests::test_trigger_manager_ic_accessors ... ok
test dream::triggers::tests::test_trigger_manager_ic_negative_clamping ... ok
test dream::triggers::tests::test_trigger_manager_ic_over_one_clamping ... ok
test dream::triggers::tests::test_trigger_manager_ic_minimum_value ... ok
test dream::triggers::tests::test_trigger_manager_ic_disabled_no_update ... ok
... and 29 more tests
test result: ok. 43 passed; 0 failed; 0 ignored
```

### Full State Verification Evidence

**Source of Truth - TriggerManager struct now has:**
- `config: TriggerConfig` at line 191 ✅
- `current_ic: Option<f32>` at line 195 ✅

**`check_triggers()` priority order verified:**
1. Manual (Priority 1) ✅
2. IdentityCritical (Priority 2) ✅
3. GpuOverload (Priority 3) ✅
4. HighEntropy (Priority 4) ✅

**Edge Cases Tested:**
- IC = 0.49 (below threshold) → triggers IdentityCritical ✅
- IC = 0.5 (at threshold) → does NOT trigger ✅
- IC = 0.9 (above threshold) → does NOT trigger ✅
- IC = NaN → clamps to 0.0, triggers IdentityCritical ✅
- IC = Infinity → clamps to 1.0, does NOT trigger ✅
- IC = -0.5 → clamps to 0.0, triggers IdentityCritical ✅
- IC = 1.5 → clamps to 1.0, does NOT trigger ✅

### Code Review Result

Code simplifier review completed and documented in memory `TASK-21_code_review`.

**Result: APPROVED - No changes required**

The implementation is clean, well-documented, and follows existing patterns.

### Constitution Compliance

- AP-26: IC<0.5 triggers dream ✅
- AP-38: IC<0.5 auto-triggers dream ✅
- IDENTITY-007: IC < 0.5 → auto-trigger dream ✅
- AP-10: No NaN/Infinity in UTL values (handled with clamping) ✅
