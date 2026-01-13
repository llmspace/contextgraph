# TASK-20: Implement TriggerConfig with IC threshold

## METADATA
| Field | Value |
|-------|-------|
| Task ID | TASK-20 (Original: TASK-IDENTITY-002) |
| Status | **COMPLETE** |
| Layer | Integration |
| Phase | 3 |
| Sequence | 20 |
| Implements | REQ-IDENTITY-002, REQ-IDENTITY-005 |
| Dependencies | TASK-19 (COMPLETE - IdentityCritical variant exists) |
| Blocks | TASK-21 (TriggerManager IC checking) |
| Est. Hours | 1.5 |

---

## AI AGENT CONTEXT - READ THIS FIRST

### What You Are Implementing

Create a `TriggerConfig` struct in `triggers.rs` that holds configuration for the dream trigger system. This is a **NEW STRUCT** - it does not currently exist.

### Critical Facts About Current State

1. **TASK-19 is COMPLETE**: The `IdentityCritical { ic_value: f32 }` variant already exists in `types.rs:564-567`
2. **TriggerManager EXISTS** at `triggers.rs:32-53` - but it does NOT have a `TriggerConfig` field yet
3. **TriggerConfig does NOT exist** - you must create it
4. **No backwards compatibility needed** - this is new code

### Why This Matters (Constitution)

Per Constitution `gwt.self_ego_node.thresholds`:
- `critical: <0.5` - IC below 0.5 triggers dream consolidation
- `warning: <0.7` - Warning threshold
- `healthy: >0.9` - System is healthy

Per AP-26: Invalid configuration MUST panic immediately (fail-fast).

---

## CURRENT CODEBASE STATE (Verified 2026-01-13)

### File: `crates/context-graph-core/src/dream/triggers.rs`

**Current TriggerManager struct (lines 32-53):**
```rust
#[derive(Debug)]
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

**What's MISSING:**
- `TriggerConfig` struct
- `ic_threshold` field
- `config` field in TriggerManager
- Integration of IC threshold into trigger checking

### File: `crates/context-graph-core/src/dream/types.rs`

**ExtendedTriggerReason (lines 556-583) - ALREADY EXISTS:**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExtendedTriggerReason {
    Manual,
    IdentityCritical { ic_value: f32 },  // <-- EXISTS from TASK-19
    IdleTimeout,
    HighEntropy,
    GpuOverload,
    MemoryPressure,
    Scheduled,
}
```

### File: `crates/context-graph-core/src/dream/mod.rs`

**Current exports (line 98):**
```rust
pub use triggers::{EntropyCalculator, GpuMonitor, TriggerManager};
```

**After this task, add:**
```rust
pub use triggers::{EntropyCalculator, GpuMonitor, TriggerConfig, TriggerManager};
```

---

## EXACT IMPLEMENTATION STEPS

### Step 1: Create TriggerConfig struct

**File**: `crates/context-graph-core/src/dream/triggers.rs`

**Location**: Add BEFORE line 18 (before TriggerManager struct definition)

**Code to add:**
```rust
use std::time::Duration;

/// Configuration for trigger manager.
///
/// Holds thresholds for dream trigger conditions.
///
/// # Constitution Compliance
///
/// - `ic_threshold`: default 0.5 per `gwt.self_ego_node.thresholds.critical`
/// - `entropy_threshold`: default 0.7 per `dream.trigger.entropy`
/// - `cooldown`: default 60s to prevent trigger spam
///
/// # Example
///
/// ```
/// use context_graph_core::dream::TriggerConfig;
///
/// let config = TriggerConfig::default();
/// assert_eq!(config.ic_threshold, 0.5);
///
/// // Custom configuration
/// let custom = TriggerConfig::default()
///     .with_ic_threshold(0.4)
///     .with_entropy_threshold(0.8);
/// custom.validate(); // Panics if invalid
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerConfig {
    /// IC threshold for identity crisis (default: 0.5)
    /// Constitution: `gwt.self_ego_node.thresholds.critical = 0.5`
    /// When IC drops below this, triggers `ExtendedTriggerReason::IdentityCritical`
    pub ic_threshold: f32,

    /// Entropy threshold for high entropy trigger (default: 0.7)
    /// Constitution: `dream.trigger.entropy > 0.7 for 5min`
    pub entropy_threshold: f32,

    /// Cooldown between triggers (default: 60 seconds)
    /// Prevents rapid re-triggering
    pub cooldown: Duration,
}

impl Default for TriggerConfig {
    /// Create config with constitution-mandated defaults.
    fn default() -> Self {
        Self {
            ic_threshold: 0.5,        // Constitution: gwt.self_ego_node.thresholds.critical
            entropy_threshold: 0.7,    // Constitution: dream.trigger.entropy
            cooldown: Duration::from_secs(60),
        }
    }
}

impl TriggerConfig {
    /// Validate configuration against constitution bounds.
    ///
    /// # Panics
    ///
    /// Panics with detailed error message if any value is out of bounds.
    /// Per AP-26: fail-fast on invalid configuration.
    ///
    /// # Constitution Bounds
    ///
    /// - `ic_threshold`: MUST be in [0.0, 1.0]
    /// - `entropy_threshold`: MUST be in [0.0, 1.0]
    /// - `cooldown`: No explicit bound, but Duration::ZERO is unusual
    #[track_caller]
    pub fn validate(&self) {
        assert!(
            (0.0..=1.0).contains(&self.ic_threshold),
            "TriggerConfig: ic_threshold must be in [0.0, 1.0], got {}. \
             Constitution: gwt.self_ego_node.thresholds.critical = 0.5",
            self.ic_threshold
        );
        assert!(
            (0.0..=1.0).contains(&self.entropy_threshold),
            "TriggerConfig: entropy_threshold must be in [0.0, 1.0], got {}. \
             Constitution: dream.trigger.entropy threshold",
            self.entropy_threshold
        );
    }

    /// Create a validated config, panicking if invalid.
    ///
    /// Use this in constructors to fail-fast per AP-26.
    #[track_caller]
    pub fn validated(self) -> Self {
        self.validate();
        self
    }

    /// Builder: set IC threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - IC threshold [0.0, 1.0]. Values < 0.5 are more sensitive.
    pub fn with_ic_threshold(mut self, threshold: f32) -> Self {
        self.ic_threshold = threshold;
        self
    }

    /// Builder: set entropy threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Entropy threshold [0.0, 1.0]. Higher = less sensitive.
    pub fn with_entropy_threshold(mut self, threshold: f32) -> Self {
        self.entropy_threshold = threshold;
        self
    }

    /// Builder: set cooldown duration.
    ///
    /// # Arguments
    ///
    /// * `cooldown` - Duration between allowed triggers.
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Check if IC value indicates identity crisis.
    ///
    /// # Arguments
    ///
    /// * `ic_value` - Current Identity Continuity value [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `true` if `ic_value < ic_threshold` (crisis state)
    #[inline]
    pub fn is_identity_critical(&self, ic_value: f32) -> bool {
        ic_value < self.ic_threshold
    }

    /// Check if entropy value exceeds threshold.
    ///
    /// # Arguments
    ///
    /// * `entropy` - Current entropy value [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `true` if `entropy > entropy_threshold`
    #[inline]
    pub fn is_high_entropy(&self, entropy: f32) -> bool {
        entropy > self.entropy_threshold
    }
}
```

### Step 2: Export TriggerConfig from mod.rs

**File**: `crates/context-graph-core/src/dream/mod.rs`

**Location**: Line 98

**Change from:**
```rust
pub use triggers::{EntropyCalculator, GpuMonitor, TriggerManager};
```

**Change to:**
```rust
pub use triggers::{EntropyCalculator, GpuMonitor, TriggerConfig, TriggerManager};
```

### Step 3: Add tests for TriggerConfig

**File**: `crates/context-graph-core/src/dream/triggers.rs`

**Location**: Add to the `#[cfg(test)] mod tests` section (after line 788)

**Tests to add:**
```rust
    // ============ TriggerConfig Tests ============

    #[test]
    fn test_trigger_config_constitution_defaults() {
        let config = TriggerConfig::default();

        assert_eq!(
            config.ic_threshold, 0.5,
            "ic_threshold must be 0.5 per Constitution gwt.self_ego_node.thresholds.critical"
        );
        assert_eq!(
            config.entropy_threshold, 0.7,
            "entropy_threshold must be 0.7 per Constitution dream.trigger.entropy"
        );
        assert_eq!(
            config.cooldown,
            Duration::from_secs(60),
            "cooldown default is 60 seconds"
        );
    }

    #[test]
    fn test_trigger_config_validate_passes_valid() {
        let config = TriggerConfig::default();
        config.validate(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_negative_ic() {
        let config = TriggerConfig {
            ic_threshold: -0.1,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_ic_over_one() {
        let config = TriggerConfig {
            ic_threshold: 1.5,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "entropy_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_negative_entropy() {
        let config = TriggerConfig {
            entropy_threshold: -0.1,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    fn test_trigger_config_builder_pattern() {
        let config = TriggerConfig::default()
            .with_ic_threshold(0.4)
            .with_entropy_threshold(0.8)
            .with_cooldown(Duration::from_secs(30));

        assert_eq!(config.ic_threshold, 0.4);
        assert_eq!(config.entropy_threshold, 0.8);
        assert_eq!(config.cooldown, Duration::from_secs(30));
    }

    #[test]
    fn test_trigger_config_validated_returns_self() {
        let config = TriggerConfig::default().validated();
        assert_eq!(config.ic_threshold, 0.5);
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validated_panics_invalid() {
        TriggerConfig::default()
            .with_ic_threshold(-1.0)
            .validated();
    }

    #[test]
    fn test_trigger_config_is_identity_critical() {
        let config = TriggerConfig::default(); // ic_threshold = 0.5

        // Below threshold = crisis
        assert!(config.is_identity_critical(0.49), "0.49 < 0.5 should be critical");
        assert!(config.is_identity_critical(0.0), "0.0 < 0.5 should be critical");

        // At or above threshold = not crisis
        assert!(!config.is_identity_critical(0.5), "0.5 >= 0.5 should NOT be critical");
        assert!(!config.is_identity_critical(0.51), "0.51 > 0.5 should NOT be critical");
        assert!(!config.is_identity_critical(1.0), "1.0 > 0.5 should NOT be critical");
    }

    #[test]
    fn test_trigger_config_is_high_entropy() {
        let config = TriggerConfig::default(); // entropy_threshold = 0.7

        // Above threshold = high entropy
        assert!(config.is_high_entropy(0.71), "0.71 > 0.7 should be high entropy");
        assert!(config.is_high_entropy(1.0), "1.0 > 0.7 should be high entropy");

        // At or below threshold = not high entropy
        assert!(!config.is_high_entropy(0.7), "0.7 <= 0.7 should NOT be high entropy");
        assert!(!config.is_high_entropy(0.69), "0.69 < 0.7 should NOT be high entropy");
        assert!(!config.is_high_entropy(0.0), "0.0 < 0.7 should NOT be high entropy");
    }

    #[test]
    fn test_trigger_config_edge_case_boundary_values() {
        // Test exact boundary values
        let config = TriggerConfig {
            ic_threshold: 0.0,
            entropy_threshold: 1.0,
            cooldown: Duration::ZERO,
        };
        config.validate(); // Should pass - 0.0 and 1.0 are valid

        let config_max = TriggerConfig {
            ic_threshold: 1.0,
            entropy_threshold: 0.0,
            cooldown: Duration::from_secs(86400), // 24 hours
        };
        config_max.validate(); // Should pass
    }

    #[test]
    fn test_trigger_config_serialization_roundtrip() {
        // TriggerConfig does not derive Serialize/Deserialize by default
        // but if it did, this test would verify roundtrip
        let config = TriggerConfig::default()
            .with_ic_threshold(0.45)
            .with_entropy_threshold(0.75);

        // Verify config fields survive clone (basic roundtrip)
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }
```

---

## VERIFICATION COMMANDS

### Step 1: Compilation Check
```bash
cargo check -p context-graph-core 2>&1 | head -50
```
**Expected**: No errors related to TriggerConfig

### Step 2: Run TriggerConfig Tests
```bash
cargo test -p context-graph-core trigger_config -- --nocapture 2>&1
```
**Expected**: All `trigger_config` tests pass

### Step 3: Verify Export
```bash
cargo doc -p context-graph-core --no-deps 2>&1 && \
  grep "TriggerConfig" target/doc/context_graph_core/dream/struct.TriggerConfig.html
```
**Expected**: TriggerConfig appears in documentation

### Step 4: Run All Dream Tests
```bash
cargo test -p context-graph-core dream:: -- --test-threads=1 2>&1
```
**Expected**: All dream-related tests pass

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth
The source of truth is the `TriggerConfig` struct in `crates/context-graph-core/src/dream/triggers.rs`.

### Execute & Inspect Protocol

**After implementation, execute:**
```bash
# 1. Verify struct exists and has correct fields
grep -A 15 "pub struct TriggerConfig" crates/context-graph-core/src/dream/triggers.rs

# 2. Verify Default implementation has constitution values
grep -A 10 "impl Default for TriggerConfig" crates/context-graph-core/src/dream/triggers.rs

# 3. Verify validate() method exists with panic assertions
grep -A 20 "pub fn validate" crates/context-graph-core/src/dream/triggers.rs

# 4. Verify export from mod.rs
grep "TriggerConfig" crates/context-graph-core/src/dream/mod.rs

# 5. Run the specific tests
cargo test -p context-graph-core test_trigger_config_constitution_defaults -- --nocapture
cargo test -p context-graph-core test_trigger_config_is_identity_critical -- --nocapture
```

### Boundary & Edge Case Audit

**Edge Case 1: IC at exactly 0.5 (boundary)**
```rust
// Synthetic input
let config = TriggerConfig::default(); // ic_threshold = 0.5
let ic_value = 0.5;

// BEFORE: N/A - struct doesn't exist
// AFTER: is_identity_critical(0.5) returns false (0.5 is NOT < 0.5)

// Expected output:
println!("IC=0.5, threshold=0.5, is_critical={}", config.is_identity_critical(0.5));
// Output: IC=0.5, threshold=0.5, is_critical=false
```

**Edge Case 2: IC at exactly 0.0 (minimum)**
```rust
let config = TriggerConfig::default();
let ic_value = 0.0;

// BEFORE: N/A
// AFTER: is_identity_critical(0.0) returns true (0.0 < 0.5)

println!("IC=0.0, threshold=0.5, is_critical={}", config.is_identity_critical(0.0));
// Output: IC=0.0, threshold=0.5, is_critical=true
```

**Edge Case 3: Invalid ic_threshold (-0.1)**
```rust
let config = TriggerConfig {
    ic_threshold: -0.1,
    ..Default::default()
};

// BEFORE: N/A
// AFTER: validate() panics with message:
//   "TriggerConfig: ic_threshold must be in [0.0, 1.0], got -0.1"

config.validate(); // PANICS per AP-26
```

### Evidence of Success

After running all tests, the output should include:
```
running X tests
test dream::triggers::tests::test_trigger_config_constitution_defaults ... ok
test dream::triggers::tests::test_trigger_config_validate_passes_valid ... ok
test dream::triggers::tests::test_trigger_config_validate_panics_negative_ic ... ok
test dream::triggers::tests::test_trigger_config_validate_panics_ic_over_one ... ok
test dream::triggers::tests::test_trigger_config_validate_panics_negative_entropy ... ok
test dream::triggers::tests::test_trigger_config_builder_pattern ... ok
test dream::triggers::tests::test_trigger_config_validated_returns_self ... ok
test dream::triggers::tests::test_trigger_config_validated_panics_invalid ... ok
test dream::triggers::tests::test_trigger_config_is_identity_critical ... ok
test dream::triggers::tests::test_trigger_config_is_high_entropy ... ok
test dream::triggers::tests::test_trigger_config_edge_case_boundary_values ... ok
test dream::triggers::tests::test_trigger_config_serialization_roundtrip ... ok
```

---

## MANUAL TESTING PROTOCOL

### Test 1: Compile and Doc Generation
```bash
# Clean build to verify no stale artifacts
cargo clean -p context-graph-core
cargo build -p context-graph-core 2>&1 | tee /tmp/build.log
grep -i "error" /tmp/build.log  # Should be empty
```

### Test 2: Verify Default Values Match Constitution
```bash
cargo test -p context-graph-core test_trigger_config_constitution_defaults -- --nocapture 2>&1 | tee /tmp/defaults.log
# Verify output shows assertions passing
```

### Test 3: Verify Fail-Fast Validation (AP-26)
```bash
cargo test -p context-graph-core test_trigger_config_validate_panics -- --nocapture 2>&1 | tee /tmp/panics.log
# Verify all panic tests pass (should_panic attribute works)
```

### Test 4: Verify is_identity_critical() Logic
```bash
cargo test -p context-graph-core test_trigger_config_is_identity_critical -- --nocapture 2>&1 | tee /tmp/critical.log
# Verify: 0.49 is critical, 0.5 is NOT critical
```

---

## CONSTRAINTS (MUST FOLLOW)

1. **`ic_threshold` MUST default to 0.5** - per Constitution `gwt.self_ego_node.thresholds.critical`
2. **`entropy_threshold` MUST default to 0.7** - per Constitution `dream.trigger.entropy`
3. **`validate()` MUST panic on invalid values** - per AP-26 fail-fast
4. **Builder methods MUST return `Self`** - for fluent API
5. **NO backwards compatibility hacks** - this is new code
6. **NO mock data in tests** - use real f32 values
7. **Boundary check: `<` NOT `<=`** - IC crisis is `ic < threshold`, not `ic <= threshold`

---

## FILES TO MODIFY

| File | Lines | Action |
|------|-------|--------|
| `crates/context-graph-core/src/dream/triggers.rs` | Before line 18 | Add TriggerConfig struct and impl blocks |
| `crates/context-graph-core/src/dream/triggers.rs` | End of tests mod | Add TriggerConfig tests |
| `crates/context-graph-core/src/dream/mod.rs` | Line 98 | Add TriggerConfig to exports |

---

## OUT OF SCOPE (DO NOT IMPLEMENT)

- TriggerManager generic refactor (TASK-21)
- GpuMonitor trait (TASK-22)
- DreamEventListener wiring (TASK-24)
- MCP tool exposure (TASK-38)
- Integration of TriggerConfig into TriggerManager (TASK-21 will do this)

---

## CONSTITUTION REFERENCES

- **gwt.self_ego_node.thresholds.critical = 0.5** (lines 233-234): IC below 0.5 is critical
- **dream.trigger.entropy > 0.7 for 5min** (line 255): Entropy threshold
- **AP-26**: "IC<0.5 MUST trigger dream - no silent failures"
- **AP-38**: "IC<0.5 MUST auto-trigger dream"

---

## RELATED TASKS

| Task | Relationship |
|------|-------------|
| TASK-19 | COMPLETE - provides `IdentityCritical` variant |
| TASK-21 | Depends on this task - will integrate TriggerConfig into TriggerManager |
| TASK-24 | Will use TriggerConfig threshold for DreamEventListener |
| TASK-38 | Will expose IC threshold via MCP tool |

---

## TROUBLESHOOTING

### If compilation fails with "unused import Duration":
- Ensure `use std::time::Duration;` is at the top of triggers.rs (it may already be there from existing code)

### If tests fail with "trait bound not satisfied":
- TriggerConfig does NOT need Serialize/Deserialize - don't add them unless explicitly needed
- TriggerConfig does NOT need Copy - it contains Duration which is Copy, but explicit Copy derivation is optional

### If "duplicate definition" error:
- Check that TriggerConfig struct is only defined ONCE in triggers.rs
- Check that the import in mod.rs isn't duplicating an existing import
