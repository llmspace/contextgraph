# TASK-17: Add parking_lot::RwLock to wake_controller

```xml
<task_spec id="TASK-PERF-005" version="3.0">
<metadata>
  <title>Add parking_lot::RwLock to wake_controller</title>
  <status>complete</status>
  <completed_date>2026-01-13</completed_date>
  <layer>logic</layer>
  <sequence>17</sequence>
  <implements><requirement_ref>REQ-PERF-005</requirement_ref></implements>
  <depends_on>NONE</depends_on>
  <estimated_hours>1</estimated_hours>
  <issue_ref>ISS-015</issue_ref>
  <constitution_ref>perf.latency.reflex_cache: "<100μs"</constitution_ref>
</metadata>

<executive_summary>
## What You Need To Do

Replace `std::sync::RwLock` with `parking_lot::RwLock` in wake_controller.rs.

**WHY**: std::sync::RwLock has OS-level mutex overhead (~1-5μs per lock). parking_lot uses
spinlock-based locking for short critical sections (~100-500ns). The WakeController is
in the hot path for dream wake operations which must complete in <100ms per Constitution.

**WHERE**: File `crates/context-graph-core/src/dream/wake_controller.rs`

**LINES TO CHANGE**: 4 field definitions, 1 constructor, ~20 lock access sites

**NO BACKWARDS COMPATIBILITY**: parking_lot::RwLock returns guards directly (not Results).
Remove ALL .expect("Lock poisoned") and .unwrap() calls on lock acquisitions.
</executive_summary>

<current_state_audit date="2026-01-13">
## ACTUAL CURRENT STATE (Audited Against Codebase)

### File: crates/context-graph-core/src/dream/wake_controller.rs

**Lines 12-23 - Current Imports**:
```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use super::constants;
use super::triggers::GpuMonitor;
use super::WakeReason;
```

**NO std::sync::RwLock import** - but it's used with full path `std::sync::RwLock` in struct fields.

**Lines 88-125 - Current Struct Definition**:
```rust
pub struct WakeController {
    /// Current state
    state: Arc<std::sync::RwLock<WakeState>>,          // LINE 90 - CHANGE THIS

    /// Interrupt flag (shared with dream phases)
    interrupt_flag: Arc<AtomicBool>,

    /// Wake reason channel
    wake_sender: watch::Sender<Option<WakeReason>>,
    wake_receiver: watch::Receiver<Option<WakeReason>>,

    /// Wake start time (for latency measurement)
    wake_start: Arc<std::sync::RwLock<Option<Instant>>>,  // LINE 100 - CHANGE THIS

    /// Wake completion time
    wake_complete: Arc<std::sync::RwLock<Option<Instant>>>, // LINE 103 - CHANGE THIS

    /// Maximum allowed latency (Constitution: <100ms)
    max_latency: Duration,

    /// GPU monitor for budget enforcement
    gpu_monitor: Arc<std::sync::RwLock<GpuMonitor>>,    // LINE 109 - CHANGE THIS

    // ... rest of fields are AtomicU64, Duration - no changes needed
}
```

**Lines 127-147 - Current Constructor**:
```rust
impl WakeController {
    pub fn new() -> Self {
        let (wake_sender, wake_receiver) = watch::channel(None);

        Self {
            state: Arc::new(std::sync::RwLock::new(WakeState::Idle)),         // LINE 133
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            wake_sender,
            wake_receiver,
            wake_start: Arc::new(std::sync::RwLock::new(None)),              // LINE 137
            wake_complete: Arc::new(std::sync::RwLock::new(None)),           // LINE 138
            max_latency: constants::MAX_WAKE_LATENCY,
            gpu_monitor: Arc::new(std::sync::RwLock::new(GpuMonitor::new())), // LINE 140
            // ...
        }
    }
}
```

### Cargo.toml: crates/context-graph-core/Cargo.toml
**CURRENT STATE**: NO parking_lot dependency.

```toml
[dependencies]
tokio.workspace = true
async-trait.workspace = true
serde.workspace = true
# ... NO parking_lot
```

### parking_lot Usage Elsewhere
The `context-graph-mcp` crate already uses `parking_lot = "0.12"`:
- `crates/context-graph-mcp/Cargo.toml` line: `parking_lot = "0.12"`
- Used in: kuramoto_stepper.rs, handlers.rs, test files

**PATTERN TO FOLLOW** (from kuramoto_stepper.rs):
```rust
use parking_lot::RwLock;

pub struct KuramotoStepper {
    network: RwLock<KuramotoNetwork>,  // NOT Arc<RwLock<...>> for owned state
    running: AtomicBool,
}

impl KuramotoStepper {
    pub fn get_phases(&self) -> Vec<f32> {
        self.network.read().get_phases()  // NO .unwrap() or .expect()
    }

    pub fn step(&self) {
        self.network.write().step(0.01);  // Direct access, no Result
    }
}
```
</current_state_audit>

<exact_changes_required>
## EXACT CHANGES YOU MUST MAKE

### CHANGE 1: Add parking_lot to Cargo.toml (1 line)

**File**: `crates/context-graph-core/Cargo.toml`
**Location**: After line 23 (after `rayon = "1.10"`)
**Add**:
```toml
parking_lot = "0.12"
```

### CHANGE 2: Update Imports (1 line)

**File**: `crates/context-graph-core/src/dream/wake_controller.rs`
**Location**: After line 12 (after `use std::sync::Arc;`)
**Add**:
```rust
use parking_lot::RwLock;
```

### CHANGE 3: Update Struct Fields (4 lines)

**File**: `crates/context-graph-core/src/dream/wake_controller.rs`
**Line 90**:
- FROM: `state: Arc<std::sync::RwLock<WakeState>>,`
- TO:   `state: Arc<RwLock<WakeState>>,`

**Line 100**:
- FROM: `wake_start: Arc<std::sync::RwLock<Option<Instant>>>,`
- TO:   `wake_start: Arc<RwLock<Option<Instant>>>,`

**Line 103**:
- FROM: `wake_complete: Arc<std::sync::RwLock<Option<Instant>>>,`
- TO:   `wake_complete: Arc<RwLock<Option<Instant>>>,`

**Line 109**:
- FROM: `gpu_monitor: Arc<std::sync::RwLock<GpuMonitor>>,`
- TO:   `gpu_monitor: Arc<RwLock<GpuMonitor>>,`

### CHANGE 4: Update Constructor (4 lines)

**File**: `crates/context-graph-core/src/dream/wake_controller.rs`
**Line 133**:
- FROM: `state: Arc::new(std::sync::RwLock::new(WakeState::Idle)),`
- TO:   `state: Arc::new(RwLock::new(WakeState::Idle)),`

**Line 137**:
- FROM: `wake_start: Arc::new(std::sync::RwLock::new(None)),`
- TO:   `wake_start: Arc::new(RwLock::new(None)),`

**Line 138**:
- FROM: `wake_complete: Arc::new(std::sync::RwLock::new(None)),`
- TO:   `wake_complete: Arc::new(RwLock::new(None)),`

**Line 140**:
- FROM: `gpu_monitor: Arc::new(std::sync::RwLock::new(GpuMonitor::new())),`
- TO:   `gpu_monitor: Arc::new(RwLock::new(GpuMonitor::new())),`

### CHANGE 5: Remove .expect("Lock poisoned") from ALL lock accesses

**CRITICAL**: parking_lot::RwLock does NOT return Result - it returns the guard directly.
Every `.read().expect("Lock poisoned")` becomes `.read()`.
Every `.write().expect("Lock poisoned")` becomes `.write()`.

**Lines to change** (grep for `expect("Lock poisoned")`):
- Line 156: `self.state.write().expect("Lock poisoned")` → `self.state.write()`
- Line 164: `self.wake_start.write().expect("Lock poisoned")` → `self.wake_start.write()`
- Line 165: `self.wake_complete.write().expect("Lock poisoned")` → `self.wake_complete.write()`
- Line 179: `self.state.read().expect("Lock poisoned")` → `self.state.read()`
- Line 188: `self.wake_start.write().expect("Lock poisoned")` → `self.wake_start.write()`
- Line 194: `self.state.write().expect("Lock poisoned")` → `self.state.write()`
- Line 228: `self.wake_complete.write().expect("Lock poisoned")` → `self.wake_complete.write()`
- Line 234: `self.wake_start.read().expect("Lock poisoned")` → `self.wake_start.read()`
- Line 255: `self.state.write().expect("Lock poisoned")` → `self.state.write()`
- Line 268: `self.state.write().expect("Lock poisoned")` → `self.state.write()`
- Line 273: `self.wake_start.write().expect("Lock poisoned")` → `self.wake_start.write()`
- Line 274: `self.wake_complete.write().expect("Lock poisoned")` → `self.wake_complete.write()`
- Line 297: `self.gpu_monitor.read().expect("Lock poisoned")` → `self.gpu_monitor.read()`
- Line 321: `self.gpu_monitor.read().expect("Lock poisoned")` → `self.gpu_monitor.read()`
- Line 335: `self.state.read().expect("Lock poisoned")` → `self.state.read()`
- Line 340: `self.state.read().expect("Lock poisoned")` → `self.state.read()`
- Line 360: `self.gpu_monitor.write().expect("Lock poisoned")` → `self.gpu_monitor.write()`

**Total**: 17 lock access sites to update
</exact_changes_required>

<api_differences>
## API DIFFERENCES: std::sync vs parking_lot

| Feature | std::sync::RwLock | parking_lot::RwLock |
|---------|-------------------|---------------------|
| Lock poisoning | Yes (returns Result) | No (returns guard) |
| Read lock | `.read().unwrap()` | `.read()` |
| Write lock | `.write().unwrap()` | `.write()` |
| Size | 40-56 bytes (OS mutex) | 8 bytes (word-sized) |
| Performance | OS syscall (~1-5μs) | Spinlock (~100-500ns) |
| Fairness | None (OS dependent) | Fair queuing (FIFO) |

### CRITICAL: Error Handling Change

**std::sync**:
```rust
// Returns Result, can poison on panic
let guard = lock.read().expect("Lock poisoned");  // or .unwrap()
let guard = lock.write().expect("Lock poisoned");
```

**parking_lot**:
```rust
// Returns guard directly, no poisoning
let guard = lock.read();   // Direct access
let guard = lock.write();  // Direct access
```

### When NOT to Use parking_lot

From Constitution (rust_standards.async_patterns):
- **Held across await points**: Use `tokio::sync::RwLock` instead
- **Requires poisoning semantics**: Keep `std::sync::RwLock`

The WakeController does NOT hold locks across await points - all lock scopes are
short synchronous operations. parking_lot is the correct choice.
</api_differences>

<definition_of_done>
## DEFINITION OF DONE

### Required Signatures After Implementation

```rust
// crates/context-graph-core/src/dream/wake_controller.rs

use parking_lot::RwLock;  // NOT std::sync::RwLock

pub struct WakeController {
    state: Arc<RwLock<WakeState>>,              // parking_lot
    wake_start: Arc<RwLock<Option<Instant>>>,   // parking_lot
    wake_complete: Arc<RwLock<Option<Instant>>>, // parking_lot
    gpu_monitor: Arc<RwLock<GpuMonitor>>,       // parking_lot
    // ... other fields unchanged
}

impl WakeController {
    pub fn state(&self) -> WakeState {
        *self.state.read()  // NO .unwrap(), NO .expect()
    }

    pub fn prepare_for_dream(&self) {
        let mut state = self.state.write();  // NO .expect()
        *state = WakeState::Dreaming;
        // ...
    }
}
```

### Constraints (MUST ALL PASS)

- [ ] `parking_lot = "0.12"` in context-graph-core/Cargo.toml
- [ ] `use parking_lot::RwLock;` in wake_controller.rs imports
- [ ] ZERO occurrences of `std::sync::RwLock` in wake_controller.rs
- [ ] ZERO occurrences of `.expect("Lock poisoned")` in wake_controller.rs
- [ ] ZERO occurrences of `.unwrap()` on lock acquisitions
- [ ] All 17 lock access sites use direct `.read()` or `.write()` without Result handling
</definition_of_done>

<verification_commands>
## VERIFICATION COMMANDS

### 1. Compilation Check
```bash
cargo check -p context-graph-core
# EXPECTED: Compiles successfully with no errors
```

### 2. Dependency Verification
```bash
cargo tree -p context-graph-core | grep parking_lot
# EXPECTED: parking_lot v0.12.x
```

### 3. Static Analysis - No std::sync::RwLock
```bash
grep "std::sync::RwLock" crates/context-graph-core/src/dream/wake_controller.rs
# EXPECTED: (empty - no matches)
```

### 4. Static Analysis - No Lock Poisoning Handling
```bash
grep -E "(expect|unwrap).*Lock" crates/context-graph-core/src/dream/wake_controller.rs
# EXPECTED: (empty - no matches)
```

### 5. Unit Tests
```bash
cargo test -p context-graph-core wake_controller -- --nocapture
# EXPECTED: All tests pass (13+ tests)
```

### 6. FSV Integration Test
```bash
cargo test -p context-graph-core --test wake_controller_fsv_test -- --nocapture
# EXPECTED: All FSV tests pass (12+ tests)
```

### 7. Full Test Suite
```bash
cargo test -p context-graph-core
# EXPECTED: All tests pass
```
</verification_commands>

<test_file_info>
## EXISTING TEST FILE

**File**: `crates/context-graph-core/tests/wake_controller_fsv_test.rs`
**Status**: Already exists with comprehensive FSV tests
**Test Count**: 12 tests covering:
- Source of truth verification
- Execute & inspect protocol
- Edge cases (3 required by FSV)
- Integration tests

**IMPORTANT**: These tests do NOT use mock data. They use:
- Real WakeController instances
- Real state machine transitions
- Real timing measurements
- Real UUID generation for session IDs

The tests MUST continue to pass after parking_lot migration. NO changes to test
logic are required - the API surface remains identical (method names, return types).
Only internal locking implementation changes.
</test_file_info>

<full_state_verification_protocol>
## FULL STATE VERIFICATION PROTOCOL (MANDATORY)

After completing the implementation, you MUST perform these verification steps:

### 1. Define Source of Truth
**Source**: `crates/context-graph-core/src/dream/wake_controller.rs`
**Verification**: File must compile and contain only parking_lot::RwLock

### 2. Execute & Inspect
```bash
# Step A: Compile
cargo build -p context-graph-core 2>&1 | tee /tmp/task17_compile.log

# Step B: Run tests and capture output
cargo test -p context-graph-core wake_controller 2>&1 | tee /tmp/task17_test.log

# Step C: Verify source of truth
echo "=== SOURCE OF TRUTH INSPECTION ===" | tee /tmp/task17_verify.log
grep -n "use parking_lot" crates/context-graph-core/src/dream/wake_controller.rs >> /tmp/task17_verify.log
grep -c "std::sync::RwLock" crates/context-graph-core/src/dream/wake_controller.rs >> /tmp/task17_verify.log
grep -c "expect.*Lock" crates/context-graph-core/src/dream/wake_controller.rs >> /tmp/task17_verify.log
```

### 3. Boundary & Edge Case Audit (3 Required)

**Edge Case 1: Wake during Idle state**
- BEFORE: `controller.state() == Idle`
- ACTION: `controller.signal_wake(ExternalQuery)`
- AFTER: `controller.state() == Idle` (unchanged, ignored)
- VERIFY: Lock acquisition works without poisoning

**Edge Case 2: GPU budget exactly at 30%**
- BEFORE: `gpu_usage = 0.30`
- ACTION: `controller.check_gpu_budget()`
- AFTER: No wake triggered (0.30 > 0.30 is false)
- VERIFY: gpu_monitor lock read works correctly

**Edge Case 3: Double wake signal**
- BEFORE: `state = Dreaming`, first wake sent
- ACTION: Second `signal_wake()` call
- AFTER: Second wake ignored, no error
- VERIFY: state lock handles rapid read/write correctly

### 4. Evidence of Success
After all tests pass, create this log:
```
╔═══════════════════════════════════════════════════════════════════╗
║          TASK-17 VERIFICATION - EVIDENCE OF SUCCESS               ║
╠═══════════════════════════════════════════════════════════════════╣
║ Timestamp: YYYY-MM-DDTHH:MM:SS UTC                                ║
╠═══════════════════════════════════════════════════════════════════╣
║ Dependency Check:                                                  ║
║   $ cargo tree -p context-graph-core | grep parking_lot           ║
║   OUTPUT: parking_lot v0.12.x                                     ║
║   RESULT: PASS                                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║ Static Analysis:                                                   ║
║   std::sync::RwLock occurrences: 0   PASS                         ║
║   .expect("Lock poisoned"): 0        PASS                         ║
║   parking_lot::RwLock import: 1      PASS                         ║
╠═══════════════════════════════════════════════════════════════════╣
║ Compilation: cargo check -p context-graph-core                    ║
║   RESULT: SUCCESS (0 errors)                                      ║
╠═══════════════════════════════════════════════════════════════════╣
║ Unit Tests: cargo test -p context-graph-core wake_controller      ║
║   RESULT: XX passed, 0 failed                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║ Edge Case 1 (Idle wake): PASS                                     ║
║ Edge Case 2 (30% GPU): PASS                                       ║
║ Edge Case 3 (Double wake): PASS                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║                    ALL VERIFICATIONS PASSED                        ║
╚═══════════════════════════════════════════════════════════════════╝
```
</full_state_verification_protocol>

<manual_testing_protocol>
## MANUAL TESTING PROTOCOL

Run each test individually and verify output:

### Test 1: State Machine Transitions
```bash
cargo test -p context-graph-core test_state_transitions_complete_cycle -- --nocapture
```
**Expected Output**: Shows Idle → Dreaming → Waking → Completing → Idle
**Verify**: Each state printed, no panics from lock poisoning

### Test 2: Wake Latency Measurement
```bash
cargo test -p context-graph-core test_wake_latency_success -- --nocapture
```
**Expected Output**: Latency < 100ms
**Verify**: Timing measurement works with parking_lot locks

### Test 3: GPU Budget Enforcement
```bash
cargo test -p context-graph-core test_gpu_budget_exceeded -- --nocapture
```
**Expected Output**: GpuBudgetExceeded error at 50% usage
**Verify**: gpu_monitor read/write locks work correctly

### Test 4: FSV Integration
```bash
cargo test -p context-graph-core --test wake_controller_fsv_test fsv_integration_full_wake_cycle -- --nocapture
```
**Expected Output**: All 5 steps pass, shows timing and state transitions
**Verify**: Full cycle works with new locking

### Test 5: Edge Cases
```bash
cargo test -p context-graph-core --test wake_controller_fsv_test fsv_edge_case -- --nocapture
```
**Expected Output**: All 5 edge cases pass
**Verify**: Edge cases work with parking_lot semantics
</manual_testing_protocol>

<synthetic_test_data>
## SYNTHETIC TEST DATA & EXPECTED OUTPUTS

### Test Data 1: GPU Budget Threshold
**Input**: `controller.set_gpu_usage(0.30)` + `check_gpu_budget()`
**Expected**: Ok(()) - exactly 30% does NOT trigger (strict > comparison)

**Input**: `controller.set_gpu_usage(0.31)` + `check_gpu_budget()`
**Expected**: Err(GpuBudgetExceeded { usage: 31.0, max: 30.0 })

### Test Data 2: Wake Reason Channel
**Input**: `controller.signal_wake(WakeReason::ExternalQuery)`
**Expected**: Receiver contains Some(WakeReason::ExternalQuery)

### Test Data 3: Latency Tracking
**Input**: prepare_for_dream → signal_wake → complete_wake
**Expected**: Duration < 100ms (typically < 1ms in tests)

### Test Data 4: Stats Tracking
**Input**: Full wake cycle
**Expected**: stats.wake_count increments by 1, violations = 0
</synthetic_test_data>

<files_to_modify>
## FILES TO MODIFY

1. **crates/context-graph-core/Cargo.toml**
   - Add: `parking_lot = "0.12"`

2. **crates/context-graph-core/src/dream/wake_controller.rs**
   - Add import: `use parking_lot::RwLock;`
   - Update 4 struct field types
   - Update 4 constructor initializations
   - Remove 17 `.expect("Lock poisoned")` calls
</files_to_modify>

<files_unchanged>
## FILES THAT SHOULD NOT CHANGE

- `crates/context-graph-core/tests/wake_controller_fsv_test.rs` - Test logic unchanged
- `crates/context-graph-core/src/dream/mod.rs` - Exports unchanged
- Any other files in the project
</files_unchanged>

<error_handling_requirements>
## ERROR HANDLING REQUIREMENTS (NO FALLBACKS)

**CRITICAL**: This task implements parking_lot with NO backwards compatibility.

1. **Compilation Errors**: If the code doesn't compile, the task is NOT complete.
   Fix the error. Do not add workarounds.

2. **Test Failures**: If tests fail, the parking_lot migration broke something.
   Debug and fix. Do not skip tests or add mock data.

3. **Lock Acquisition**: parking_lot locks WILL block if contended. This is
   expected behavior. They do NOT panic on acquisition (no poison).

4. **Error Logging**: All existing tracing macros (debug!, error!, warn!, info!)
   remain unchanged. They already provide robust logging for:
   - Wake latency violations (line 243-250)
   - GPU budget exceeded (line 300-304)
   - Signal failures (line 202-206)
</error_handling_requirements>

<constitution_compliance>
## CONSTITUTION COMPLIANCE

### Rules Satisfied By This Task

- **perf.latency.reflex_cache: <100μs** - parking_lot achieves ~100-500ns lock times
- **rust_standards.type_safety** - parking_lot maintains same type safety guarantees
- **AP-14: "No .unwrap() in library code"** - parking_lot removes need for unwrap on locks

### Rules NOT Affected

- **AP-08: "No sync I/O in async context"** - WakeController uses sync locks, not async
- **ARCH-06: "All memory ops through MCP tools"** - WakeController is internal state

### Note on Arc Usage

The struct uses `Arc<RwLock<T>>` pattern (not just `RwLock<T>`) because:
1. WakeHandle needs to clone the interrupt_flag
2. Multiple references to WakeController exist
3. wake_sender is shared with subscribers

This is correct and unchanged by parking_lot migration.
</constitution_compliance>

<related_tasks>
## RELATED TASKS

### Dependencies (None)
This task has NO dependencies. It can be executed at any time.

### Related But Independent
- **TASK-16**: Remove block_on from gwt_providers - COMPLETE (different file, different crate)
- **TASK-18**: Pre-allocate HashMap capacity - INDEPENDENT (different optimization)

### Blocked By This Task
None - TASK-17 is a leaf node in the dependency graph.
</related_tasks>

<ai_agent_checklist>
## AI AGENT IMPLEMENTATION CHECKLIST

### Pre-Implementation
- [ ] Read this entire task document
- [ ] Verify crates/context-graph-core/Cargo.toml exists
- [ ] Verify crates/context-graph-core/src/dream/wake_controller.rs exists
- [ ] Confirm no parking_lot dependency currently exists in context-graph-core

### Implementation Steps
1. [ ] Add `parking_lot = "0.12"` to Cargo.toml
2. [ ] Add `use parking_lot::RwLock;` import
3. [ ] Change 4 struct field types from `std::sync::RwLock` to `RwLock`
4. [ ] Change 4 constructor lines from `std::sync::RwLock::new()` to `RwLock::new()`
5. [ ] Remove ALL 17 `.expect("Lock poisoned")` calls
6. [ ] Run `cargo check -p context-graph-core` - MUST pass
7. [ ] Run `cargo test -p context-graph-core wake_controller` - ALL MUST pass

### Post-Implementation Verification
- [ ] `grep "std::sync::RwLock" wake_controller.rs` returns empty
- [ ] `grep "expect.*Lock" wake_controller.rs` returns empty
- [ ] `cargo tree -p context-graph-core | grep parking_lot` shows v0.12.x
- [ ] All unit tests pass
- [ ] All FSV tests pass
- [ ] Create Evidence of Success log

### Completion Criteria
The task is COMPLETE when:
1. All 17 lock sites use parking_lot directly without Result handling
2. All tests pass without modification
3. Evidence of Success log is created
4. No std::sync::RwLock remains in wake_controller.rs
</ai_agent_checklist>
</task_spec>
```
