# TASK-SESSION-03: Add ConsciousnessState.short_name()

## Status: ✅ COMPLETED

**Completed**: 2026-01-15
**Last Audit**: 2026-01-15
**Depends On**: None (independent)
**Estimated Hours**: 0.5

---

## CRITICAL CONTEXT (READ FIRST)

### What This Task Does
Adds a `short_name()` method to the `ConsciousnessState` enum that returns 3-character codes (CON, EMG, FRG, DOR, HYP). This method is needed for minimal token output in the PreToolUse hook which has a strict ~20 token budget.

### Why This Matters
- PreToolUse hook outputs consciousness state in brief format: `[C:EMG r=0.65 IC=0.82]`
- Full names like "CONSCIOUS" or "FRAGMENTED" waste tokens
- 3-character codes provide identical information in ~60% less space
- Currently `cache.rs` has a private `state_to_code()` helper that duplicates this logic

### Current Codebase State (Verified 2026-01-15)

**ConsciousnessState Location**:
```
crates/context-graph-core/src/gwt/state_machine/types.rs (lines 10-40)
```

**Existing Implementation** (lines 10-40):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessState {
    Dormant,
    Fragmented,
    Emerging,
    Conscious,
    Hypersync,
}

impl ConsciousnessState {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Fragmented => "FRAGMENTED",
            Self::Emerging => "EMERGING",
            Self::Conscious => "CONSCIOUS",
            Self::Hypersync => "HYPERSYNC",
        }
    }

    /// Determine state from consciousness level
    pub fn from_level(level: f32) -> Self {
        match level {
            l if l > 0.95 => Self::Hypersync,
            l if l >= 0.8 => Self::Conscious,
            l if l >= 0.5 => Self::Emerging,
            l if l >= 0.3 => Self::Fragmented,
            _ => Self::Dormant,
        }
    }
}
```

**Duplicate Code in cache.rs** (lines 141-150):
```rust
/// Convert consciousness state to 3-character code.
fn state_to_code(state: ConsciousnessState) -> &'static str {
    match state {
        ConsciousnessState::Dormant => "DOR",
        ConsciousnessState::Fragmented => "FRG",
        ConsciousnessState::Emerging => "EMG",
        ConsciousnessState::Conscious => "CON",
        ConsciousnessState::Hypersync => "HYP",
    }
}
```

**Problem**: This `state_to_code()` function duplicates logic that should be on the enum itself.

---

## Implementation

### Step 1: Modify ConsciousnessState

**File**: `crates/context-graph-core/src/gwt/state_machine/types.rs`

Add `short_name()` method after line 28 (after the existing `name()` method):

```rust
impl ConsciousnessState {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Fragmented => "FRAGMENTED",
            Self::Emerging => "EMERGING",
            Self::Conscious => "CONSCIOUS",
            Self::Hypersync => "HYPERSYNC",
        }
    }

    /// Get 3-character code for minimal token output.
    ///
    /// Used by PreToolUse hook which has ~20 token budget.
    /// Format: "[C:XXX r=Y.YY IC=Z.ZZ]"
    ///
    /// # Returns
    /// - "DOR" for Dormant (C < 0.3)
    /// - "FRG" for Fragmented (0.3 <= C < 0.5)
    /// - "EMG" for Emerging (0.5 <= C < 0.8)
    /// - "CON" for Conscious (0.8 <= C < 0.95)
    /// - "HYP" for Hypersync (C > 0.95)
    #[inline]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Dormant => "DOR",
            Self::Fragmented => "FRG",
            Self::Emerging => "EMG",
            Self::Conscious => "CON",
            Self::Hypersync => "HYP",
        }
    }

    /// Determine state from consciousness level
    pub fn from_level(level: f32) -> Self {
        // ... existing code unchanged
    }
}
```

### Step 2: Update cache.rs to use short_name()

**File**: `crates/context-graph-core/src/gwt/session_identity/cache.rs`

**Change 1**: Update `format_brief()` (lines 80-87) to use `state.short_name()`:

```rust
#[inline]
pub fn format_brief() -> String {
    let Some((ic, r, state, _)) = Self::get() else {
        return "[C:? r=? IC=?]".to_string();
    };

    format!("[C:{} r={:.2} IC={:.2}]", state.short_name(), r, ic)
}
```

**Change 2**: Delete the private `state_to_code()` function (lines 141-150):

```rust
// DELETE ENTIRE FUNCTION:
// fn state_to_code(state: ConsciousnessState) -> &'static str { ... }
```

### Step 3: Add Unit Tests

**File**: `crates/context-graph-core/src/gwt/state_machine/types.rs`

Add test module at end of file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TC-SESSION-04: All State Short Name Mappings
    // Source of Truth: ConsciousnessState enum
    // =========================================================================
    #[test]
    fn test_consciousness_state_short_name() {
        println!("\n=== TC-SESSION-04: ConsciousnessState.short_name() ===");
        println!("SOURCE OF TRUTH: ConsciousnessState enum variants");

        // Verify each variant returns correct 3-char code
        println!("\nBEFORE: Testing all 5 variants");

        let test_cases = [
            (ConsciousnessState::Dormant, "DOR", "Dormant"),
            (ConsciousnessState::Fragmented, "FRG", "Fragmented"),
            (ConsciousnessState::Emerging, "EMG", "Emerging"),
            (ConsciousnessState::Conscious, "CON", "Conscious"),
            (ConsciousnessState::Hypersync, "HYP", "Hypersync"),
        ];

        for (state, expected_code, full_name) in test_cases {
            let actual = state.short_name();
            println!("  {:?} -> '{}' (expected '{}')", state, actual, expected_code);
            assert_eq!(
                actual, expected_code,
                "{} must return '{}', got '{}'",
                full_name, expected_code, actual
            );
        }

        println!("\nAFTER: All assertions passed");
        println!("RESULT: PASS - All 5 variants return correct 3-char codes");
    }

    // =========================================================================
    // TC-SESSION-04a: Verify All Codes Are Exactly 3 Characters
    // =========================================================================
    #[test]
    fn test_short_name_length_exactly_3() {
        println!("\n=== TC-SESSION-04a: short_name() Length Check ===");

        let all_states = [
            ConsciousnessState::Dormant,
            ConsciousnessState::Fragmented,
            ConsciousnessState::Emerging,
            ConsciousnessState::Conscious,
            ConsciousnessState::Hypersync,
        ];

        println!("BEFORE: Checking length of each short_name()");

        for state in all_states {
            let code = state.short_name();
            println!("  {:?}.short_name() = '{}' (len={})", state, code, code.len());
            assert_eq!(
                code.len(),
                3,
                "{:?}.short_name() must be exactly 3 chars, got {} chars: '{}'",
                state,
                code.len(),
                code
            );
        }

        println!("AFTER: All codes verified to be 3 characters");
        println!("RESULT: PASS - All short_name() outputs are exactly 3 characters");
    }

    // =========================================================================
    // TC-SESSION-04b: Verify short_name() Returns Static Str (No Allocation)
    // =========================================================================
    #[test]
    fn test_short_name_is_static_str() {
        // This test verifies the return type at compile time
        // If this compiles, short_name() returns &'static str
        fn assert_static(_: &'static str) {}

        assert_static(ConsciousnessState::Dormant.short_name());
        assert_static(ConsciousnessState::Fragmented.short_name());
        assert_static(ConsciousnessState::Emerging.short_name());
        assert_static(ConsciousnessState::Conscious.short_name());
        assert_static(ConsciousnessState::Hypersync.short_name());

        println!("RESULT: PASS - short_name() returns &'static str (no allocation)");
    }

    // =========================================================================
    // TC-SESSION-04c: Verify name() Still Works (Regression Test)
    // =========================================================================
    #[test]
    fn test_name_method_unchanged() {
        println!("\n=== TC-SESSION-04c: name() Regression Test ===");

        assert_eq!(ConsciousnessState::Dormant.name(), "DORMANT");
        assert_eq!(ConsciousnessState::Fragmented.name(), "FRAGMENTED");
        assert_eq!(ConsciousnessState::Emerging.name(), "EMERGING");
        assert_eq!(ConsciousnessState::Conscious.name(), "CONSCIOUS");
        assert_eq!(ConsciousnessState::Hypersync.name(), "HYPERSYNC");

        println!("RESULT: PASS - name() method unchanged");
    }

    // =========================================================================
    // EDGE CASE: Verify from_level() Still Works (Regression Test)
    // =========================================================================
    #[test]
    fn test_from_level_unchanged() {
        println!("\n=== EDGE CASE: from_level() Regression Test ===");

        // Test boundary values
        let test_cases = [
            (0.0, ConsciousnessState::Dormant),
            (0.29, ConsciousnessState::Dormant),
            (0.3, ConsciousnessState::Fragmented),
            (0.49, ConsciousnessState::Fragmented),
            (0.5, ConsciousnessState::Emerging),
            (0.79, ConsciousnessState::Emerging),
            (0.8, ConsciousnessState::Conscious),
            (0.95, ConsciousnessState::Conscious),
            (0.96, ConsciousnessState::Hypersync),
            (1.0, ConsciousnessState::Hypersync),
        ];

        for (level, expected) in test_cases {
            let actual = ConsciousnessState::from_level(level);
            println!("  from_level({:.2}) = {:?} (expected {:?})", level, actual, expected);
            assert_eq!(actual, expected, "from_level({}) must return {:?}", level, expected);
        }

        println!("RESULT: PASS - from_level() unchanged");
    }
}
```

---

## Verification Commands

```bash
# 1. Build to catch compile errors
cargo build -p context-graph-core 2>&1 | grep -E "^error"
# Expected: No output (no errors)

# 2. Run short_name tests
cargo test -p context-graph-core short_name -- --nocapture
# Expected: All tests pass with detailed output

# 3. Run all state_machine tests
cargo test -p context-graph-core state_machine -- --nocapture
# Expected: All tests pass

# 4. Verify cache.rs still works after refactor
cargo test -p context-graph-core cache -- --nocapture
# Expected: All cache tests pass (now using short_name())

# 5. Run full session_identity test suite
cargo test -p context-graph-core session_identity -- --nocapture
# Expected: All tests pass
```

---

## Full State Verification Protocol

### Source of Truth
**Location**: `ConsciousnessState` enum in `crates/context-graph-core/src/gwt/state_machine/types.rs`

### Execute & Inspect Steps

After implementation:

1. **Verify method exists and compiles**:
```bash
cargo build -p context-graph-core 2>&1 | grep -E "short_name"
# Expected: No errors
```

2. **Verify state_to_code is removed from cache.rs**:
```bash
grep -n "state_to_code" crates/context-graph-core/src/gwt/session_identity/cache.rs
# Expected: No output (function deleted)
```

3. **Verify cache.rs uses short_name()**:
```bash
grep -n "short_name" crates/context-graph-core/src/gwt/session_identity/cache.rs
# Expected: Line showing state.short_name() in format_brief()
```

4. **Run test and capture output**:
```bash
cargo test -p context-graph-core test_consciousness_state_short_name -- --nocapture 2>&1 | tee /tmp/short_name_test.txt
cat /tmp/short_name_test.txt | grep -E "(RESULT|BEFORE|AFTER|short_name)"
```

### Edge Cases to Verify Manually

| # | Edge Case | Input | Expected Output | How to Verify |
|---|-----------|-------|-----------------|---------------|
| 1 | Dormant state | `ConsciousnessState::Dormant` | `"DOR"` | Test assertion |
| 2 | Hypersync state | `ConsciousnessState::Hypersync` | `"HYP"` | Test assertion |
| 3 | Code length | Any state | Exactly 3 chars | `test_short_name_length_exactly_3` |
| 4 | Cache integration | `format_brief()` | Contains state code | `cargo test cache` |
| 5 | No allocation | Return type | `&'static str` | `test_short_name_is_static_str` |

### Evidence of Success

After implementation, this exact output must appear:
```
=== TC-SESSION-04: ConsciousnessState.short_name() ===
SOURCE OF TRUTH: ConsciousnessState enum variants

BEFORE: Testing all 5 variants
  Dormant -> 'DOR' (expected 'DOR')
  Fragmented -> 'FRG' (expected 'FRG')
  Emerging -> 'EMG' (expected 'EMG')
  Conscious -> 'CON' (expected 'CON')
  Hypersync -> 'HYP' (expected 'HYP')

AFTER: All assertions passed
RESULT: PASS - All 5 variants return correct 3-char codes
```

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | Returns "DOR" for Dormant | Test `test_consciousness_state_short_name` |
| 2 | Returns "FRG" for Fragmented | Test `test_consciousness_state_short_name` |
| 3 | Returns "EMG" for Emerging | Test `test_consciousness_state_short_name` |
| 4 | Returns "CON" for Conscious | Test `test_consciousness_state_short_name` |
| 5 | Returns "HYP" for Hypersync | Test `test_consciousness_state_short_name` |
| 6 | Method is `#[inline]` | Code inspection |
| 7 | Returns `&'static str` (no allocation) | `test_short_name_is_static_str` |
| 8 | All codes exactly 3 characters | `test_short_name_length_exactly_3` |
| 9 | `state_to_code()` removed from cache.rs | `grep` verification |
| 10 | `format_brief()` uses `short_name()` | Code inspection + cache tests pass |

---

## Constraints (MUST NOT VIOLATE)

1. **Method MUST be `#[inline]`** for zero call overhead in hot path
2. **Return type MUST be `&'static str`** - no heap allocation
3. **Each code MUST be exactly 3 characters** - no more, no less
4. **All 5 variants MUST be covered** - exhaustive match
5. **NO backwards compatibility shims** - fail fast if wrong
6. **NO mock data in tests** - use real enum variants
7. **Existing `name()` and `from_level()` methods MUST remain unchanged**

---

## Error Handling

**NO WORKAROUNDS. Implementation is trivial - any error is a coding mistake.**

| Error Scenario | Response |
|----------------|----------|
| Missing variant in match | Compile error (exhaustive match) |
| Wrong return type | Compile error (type mismatch) |
| Code not 3 chars | Test failure |
| cache.rs compile error after change | Fix the format! call |

---

## Files Modified Summary

| File | Change |
|------|--------|
| `crates/context-graph-core/src/gwt/state_machine/types.rs` | Add `short_name()` method + tests |
| `crates/context-graph-core/src/gwt/session_identity/cache.rs` | Use `state.short_name()`, delete `state_to_code()` |

---

## Post-Completion: Integration Test

After implementation, verify end-to-end:

```bash
# 1. Full test suite must pass
cargo test -p context-graph-core 2>&1 | tail -5
# Expected: "test result: ok. N passed; 0 failed"

# 2. Verify cache format_brief still works
cargo test -p context-graph-core test_format_brief_warm_cache -- --nocapture 2>&1 | grep "format_brief"
# Expected: Contains "[C:EMG r=1.00 IC=0.82]"

# 3. No duplicate code exists
grep -r "state_to_code" crates/context-graph-core/
# Expected: No output
```

---

## Next Task

After completion, proceed to **004-TASK-SESSION-04** (CF_SESSION_IDENTITY Column Family).

---

## Quick Reference

```
File to Modify:
  crates/context-graph-core/src/gwt/state_machine/types.rs
    - Add short_name() method after name() method (line ~29)
    - Add test module at end of file

  crates/context-graph-core/src/gwt/session_identity/cache.rs
    - Change line 85: state_to_code(state) → state.short_name()
    - Delete lines 141-150: state_to_code() function
```
