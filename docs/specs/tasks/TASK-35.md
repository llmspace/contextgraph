# TASK-35: Wire trigger_dream MCP Tool to TriggerManager.request_manual_trigger()

```yaml
id: TASK-35
title: Wire trigger_dream to TriggerManager Manual Trigger
status: done
layer: surface
sequence: 35
implements: REQ-MCP-009
depends_on: [TASK-24, TASK-21]
estimated_hours: 2
actual_hours: 1.5
constitution_refs: [AP-26, dream.trigger, dream.constraints.gpu, perf.latency.dream_wake]
completed_date: 2026-01-13
```

## CRITICAL CONTEXT

**The `trigger_dream` tool handler ALREADY EXISTS** at:
- `crates/context-graph-mcp/src/handlers/dream.rs:44` (`call_trigger_dream`)
- Dispatch entry at `crates/context-graph-mcp/src/handlers/tools/dispatch.rs:99`
- Tool definition at `crates/context-graph-mcp/src/tools/definitions/dream.rs:10`

**WHAT IS MISSING**: The existing `call_trigger_dream` uses `DreamScheduler.should_trigger_dream()` but does **NOT** call `TriggerManager.request_manual_trigger()`. This task wires the two together.

---

## CURRENT STATE ANALYSIS (2026-01-13)

### Files That ALREADY EXIST

| File | Purpose | Status |
|------|---------|--------|
| `crates/context-graph-mcp/src/handlers/dream.rs` | Contains `call_trigger_dream` handler | EXISTS |
| `crates/context-graph-mcp/src/tools/definitions/dream.rs` | Tool schema definition | EXISTS |
| `crates/context-graph-mcp/src/tools/names.rs:44` | `TRIGGER_DREAM` constant | EXISTS |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs:99` | Dispatch entry | EXISTS |
| `crates/context-graph-core/src/dream/triggers.rs` | `TriggerManager` with `request_manual_trigger()` | EXISTS |

### Current `call_trigger_dream` Implementation (Lines 44-167)

```rust
// Current flow (INCOMPLETE):
// 1. Checks dream_controller exists (FAIL FAST) - GOOD
// 2. Checks dream_scheduler exists (FAIL FAST) - GOOD
// 3. Parses "force" parameter - GOOD
// 4. Gets activity_level from scheduler - GOOD
// 5. Calls scheduler.should_trigger_dream() - WRONG (should use TriggerManager)
// 6. Returns result - GOOD
```

**THE BUG**: Line 89 calls `scheduler.should_trigger_dream()` instead of `TriggerManager.request_manual_trigger()`.

### TriggerManager API (triggers.rs:491-494)

```rust
/// Request a manual dream trigger.
/// Manual triggers have highest priority and bypass cooldown.
pub fn request_manual_trigger(&mut self) {
    info!("Manual dream trigger requested");
    self.manual_trigger = true;
}
```

**Priority Order** (from `check_triggers()` at line 521):
1. **Manual** - Highest priority, bypasses cooldown
2. **IdentityCritical** - IC < 0.5 (AP-26, AP-38, IDENTITY-007)
3. **GpuOverload** - GPU > 30%
4. **HighEntropy** - Entropy > 0.7 for 5min

---

## SCOPE

### In Scope
1. Add `trigger_manager: Option<Arc<RwLock<TriggerManager>>>` field to `Handlers` struct
2. Modify `call_trigger_dream` to call `trigger_manager.request_manual_trigger()`
3. Check GPU eligibility via `GpuMonitor::is_eligible_for_dream()` before triggering
4. Add comprehensive tests with real data (NO MOCKS)

### Out of Scope
- TriggerManager implementation (TASK-21 - DONE)
- DreamEventListener TriggerManager wiring (TASK-24 - DONE)
- DreamController/DreamScheduler implementation
- Creating new files (handlers already exist)

---

## REQUIRED CHANGES

### 1. Add TriggerManager to Handlers (handlers/core/handlers.rs)

**Location**: `crates/context-graph-mcp/src/handlers/core/handlers.rs` after line 129

```rust
// ========== DREAM TRIGGER MANAGER (TASK-35) ==========
/// TriggerManager for manual dream triggering via MCP.
/// TASK-35: Required for trigger_dream to call request_manual_trigger().
pub(in crate::handlers) trigger_manager:
    Option<Arc<RwLock<context_graph_core::dream::TriggerManager>>>,
```

**Also add import** at top:
```rust
use context_graph_core::dream::TriggerManager;
```

### 2. Update call_trigger_dream (handlers/dream.rs)

**Location**: `crates/context-graph-mcp/src/handlers/dream.rs`

**REPLACE EXISTING IMPLEMENTATION** (lines 44-167) with:

```rust
/// trigger_dream tool implementation.
///
/// TASK-35: Wires MCP trigger_dream to TriggerManager.request_manual_trigger().
/// FAIL FAST if TriggerManager not initialized.
///
/// Arguments:
/// - force: bool - Force trigger even if GPU busy (not recommended, violates constitution)
/// - rationale: string - REQUIRED reason for manual trigger (for audit logging)
///
/// Returns:
/// - triggered: bool - Whether manual trigger was accepted
/// - trigger_reason: string - "Manual" if accepted
/// - gpu_utilization: Option<f32> - Current GPU usage if available
/// - gpu_eligible: bool - Whether GPU < 80% (constitution: dream.trigger.gpu)
/// - error: Option<string> - Error message if failed
///
/// # Constitution Compliance
/// - GPU eligibility: dream.trigger.gpu = "<80%"
/// - GPU budget during dream: dream.constraints.gpu = "<30%"
/// - AP-26: No silent failures - FAIL FAST on missing components
pub(super) async fn call_trigger_dream(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    debug!("Handling trigger_dream tool call");

    // FAIL FAST: TriggerManager is REQUIRED
    let trigger_manager = match &self.trigger_manager {
        Some(tm) => tm,
        None => {
            error!("trigger_dream: TriggerManager not initialized - FAIL FAST per AP-26");
            return JsonRpcResponse::error(
                id,
                error_codes::DREAM_NOT_INITIALIZED,
                "TriggerManager not initialized. Configure with with_trigger_manager().",
            );
        }
    };

    // Parse rationale - REQUIRED for audit logging
    let rationale = match args.get("rationale").and_then(|v| v.as_str()) {
        Some(r) if !r.trim().is_empty() => r.to_string(),
        _ => {
            warn!("trigger_dream: rationale is required for audit compliance");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "rationale is required for manual dream trigger (audit compliance)",
            );
        }
    };

    // Parse force parameter
    let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

    // Check GPU eligibility (unless forced)
    // Constitution: dream.trigger.gpu = "<80%"
    let (gpu_utilization, gpu_eligible) = {
        let manager = trigger_manager.read();
        let gpu_usage = manager.current_gpu_usage();
        // Eligibility threshold is 80% per constitution
        let eligible = gpu_usage < 0.80;
        (gpu_usage, eligible)
    };

    if !gpu_eligible && !force {
        info!(
            "trigger_dream: GPU not eligible ({}% >= 80%), rationale: {}",
            gpu_utilization * 100.0,
            rationale
        );
        return self.tool_result_with_pulse(
            id,
            json!({
                "triggered": false,
                "trigger_reason": null,
                "gpu_utilization": gpu_utilization,
                "gpu_eligible": false,
                "error": format!(
                    "GPU usage {}% >= 80% eligibility threshold. Use force=true to override (not recommended).",
                    (gpu_utilization * 100.0).round()
                ),
                "rationale_received": rationale,
                "constitution_ref": "dream.trigger.gpu = '<80%'"
            }),
        );
    }

    // Check cooldown
    let cooldown_remaining = {
        let manager = trigger_manager.read();
        manager.cooldown_remaining()
    };

    if let Some(remaining) = cooldown_remaining {
        if !force {
            info!(
                "trigger_dream: In cooldown ({}s remaining), rationale: {}",
                remaining.as_secs(),
                rationale
            );
            return self.tool_result_with_pulse(
                id,
                json!({
                    "triggered": false,
                    "trigger_reason": null,
                    "gpu_utilization": gpu_utilization,
                    "gpu_eligible": gpu_eligible,
                    "cooldown_remaining_secs": remaining.as_secs(),
                    "error": format!(
                        "Trigger cooldown active, {}s remaining. Use force=true to override.",
                        remaining.as_secs()
                    ),
                    "rationale_received": rationale
                }),
            );
        }
    }

    // REQUEST MANUAL TRIGGER - This is the key operation
    {
        let mut manager = trigger_manager.write();
        manager.request_manual_trigger();
    }

    info!(
        "trigger_dream: Manual trigger ACCEPTED, rationale: '{}', GPU: {}%, forced: {}",
        rationale,
        (gpu_utilization * 100.0).round(),
        force
    );

    // Verify trigger was set (Full State Verification)
    let trigger_set = {
        let manager = trigger_manager.read();
        matches!(
            manager.check_triggers(),
            Some(context_graph_core::dream::types::ExtendedTriggerReason::Manual)
        )
    };

    if !trigger_set {
        error!("trigger_dream: Manual trigger was NOT set after request_manual_trigger()");
        return JsonRpcResponse::error(
            id,
            error_codes::DREAM_TRIGGER_FAILED,
            "Manual trigger request failed - check_triggers() did not return Manual",
        );
    }

    self.tool_result_with_pulse(
        id,
        json!({
            "triggered": true,
            "trigger_reason": "Manual",
            "gpu_utilization": gpu_utilization,
            "gpu_eligible": gpu_eligible,
            "rationale_logged": rationale,
            "forced": force,
            "note": "Dream cycle will be executed by background scheduler when check_triggers() is called"
        }),
    )
}
```

### 3. Add Error Code (protocol.rs)

**Location**: `crates/context-graph-mcp/src/protocol.rs` in `error_codes` module

```rust
/// Manual dream trigger request failed (Full State Verification failed)
pub const DREAM_TRIGGER_FAILED: i32 = -32055;
```

### 4. Update Tool Schema (tools/definitions/dream.rs)

**Location**: `crates/context-graph-mcp/src/tools/definitions/dream.rs`

**REPLACE** the `trigger_dream` definition (lines 10-27) with:

```rust
// trigger_dream - Manually trigger a dream consolidation cycle
ToolDefinition::new(
    "trigger_dream",
    "Manually trigger a dream consolidation cycle via TriggerManager. \
     Requires rationale for audit logging. GPU must be < 80% eligible. \
     Manual triggers have highest priority and bypass cooldown. \
     Returns immediately; actual dream executes in background scheduler.",
    json!({
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1024,
                "description": "REQUIRED: Reason for manual trigger (for audit logging)"
            },
            "force": {
                "type": "boolean",
                "default": false,
                "description": "Force trigger even if GPU busy or in cooldown (not recommended)"
            }
        },
        "required": ["rationale"]
    }),
),
```

### 5. Add Builder Method (handlers/core/handlers.rs)

Add builder method after other `with_*` methods:

```rust
/// Configure TriggerManager for manual dream triggering.
/// TASK-35: Required for trigger_dream MCP tool.
pub fn with_trigger_manager(mut self, trigger_manager: Arc<RwLock<TriggerManager>>) -> Self {
    self.trigger_manager = Some(trigger_manager);
    self
}
```

---

## TESTS TO ADD

**Location**: `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/dream_tools.rs`

### Test 1: Happy Path - Manual Trigger Accepted

```rust
#[tokio::test]
async fn test_trigger_dream_manual_happy_path_fsv() {
    // SETUP: TriggerManager with low GPU usage
    let trigger_manager = Arc::new(RwLock::new(TriggerManager::new()));
    {
        let mut tm = trigger_manager.write();
        tm.update_gpu_usage(0.25); // 25% < 80% eligibility
    }

    // BEFORE STATE
    let before = {
        let tm = trigger_manager.read();
        (tm.check_triggers(), tm.current_gpu_usage())
    };
    assert!(before.0.is_none(), "BEFORE: No triggers should be set");
    println!("BEFORE: check_triggers()={:?}, gpu={}", before.0, before.1);

    // EXECUTE: Build handlers with trigger manager and call tool
    let handlers = create_test_handlers()
        .with_trigger_manager(trigger_manager.clone());

    let args = json!({
        "rationale": "Test manual trigger for FSV verification"
    });
    let response = handlers.call_trigger_dream(Some(JsonRpcId::Number(1)), args).await;

    // AFTER STATE
    let after = {
        let tm = trigger_manager.read();
        tm.check_triggers()
    };
    println!("AFTER: check_triggers()={:?}", after);

    // EVIDENCE
    assert!(response.result.is_some(), "Response should have result");
    let result = response.result.unwrap();
    assert_eq!(result["triggered"], true, "triggered should be true");
    assert_eq!(result["trigger_reason"], "Manual", "trigger_reason should be Manual");
    assert!(matches!(
        after,
        Some(ExtendedTriggerReason::Manual)
    ), "TriggerManager should return Manual trigger");
}
```

### Test 2: GPU Not Eligible (Blocked)

```rust
#[tokio::test]
async fn test_trigger_dream_gpu_not_eligible_fsv() {
    // SETUP: TriggerManager with HIGH GPU usage
    let trigger_manager = Arc::new(RwLock::new(TriggerManager::new()));
    {
        let mut tm = trigger_manager.write();
        tm.update_gpu_usage(0.85); // 85% > 80% eligibility threshold
    }

    // BEFORE
    let before_gpu = {
        let tm = trigger_manager.read();
        tm.current_gpu_usage()
    };
    println!("BEFORE: GPU usage = {}%", before_gpu * 100.0);

    // EXECUTE
    let handlers = create_test_handlers()
        .with_trigger_manager(trigger_manager.clone());

    let args = json!({ "rationale": "Test GPU eligibility check" });
    let response = handlers.call_trigger_dream(Some(JsonRpcId::Number(2)), args).await;

    // AFTER
    let after_trigger = {
        let tm = trigger_manager.read();
        tm.check_triggers()
    };
    println!("AFTER: check_triggers()={:?}", after_trigger);

    // EVIDENCE
    let result = response.result.unwrap();
    assert_eq!(result["triggered"], false, "triggered should be false");
    assert_eq!(result["gpu_eligible"], false, "gpu_eligible should be false");
    assert!(result["error"].as_str().unwrap().contains("80%"));
    assert!(after_trigger.is_none(), "No trigger should be set");
}
```

### Test 3: Missing Rationale (Validation Error)

```rust
#[tokio::test]
async fn test_trigger_dream_missing_rationale_fails_fast() {
    let trigger_manager = Arc::new(RwLock::new(TriggerManager::new()));
    let handlers = create_test_handlers()
        .with_trigger_manager(trigger_manager);

    // Missing rationale
    let args = json!({});
    let response = handlers.call_trigger_dream(Some(JsonRpcId::Number(3)), args).await;

    // EVIDENCE
    assert!(response.error.is_some(), "Should return error for missing rationale");
    let error = response.error.unwrap();
    assert!(error.message.contains("rationale"));
}
```

### Test 4: Force Bypasses GPU Check

```rust
#[tokio::test]
async fn test_trigger_dream_force_bypasses_gpu_check_fsv() {
    let trigger_manager = Arc::new(RwLock::new(TriggerManager::new()));
    {
        let mut tm = trigger_manager.write();
        tm.update_gpu_usage(0.90); // 90% > 80% - would normally block
    }

    let handlers = create_test_handlers()
        .with_trigger_manager(trigger_manager.clone());

    // Force=true should bypass GPU check
    let args = json!({
        "rationale": "Emergency consolidation",
        "force": true
    });
    let response = handlers.call_trigger_dream(Some(JsonRpcId::Number(4)), args).await;

    // EVIDENCE
    let result = response.result.unwrap();
    assert_eq!(result["triggered"], true, "force=true should allow trigger");
    assert_eq!(result["forced"], true);

    // Verify trigger was set
    let trigger = {
        let tm = trigger_manager.read();
        tm.check_triggers()
    };
    assert!(matches!(trigger, Some(ExtendedTriggerReason::Manual)));
}
```

### Test 5: TriggerManager Not Initialized (Fail Fast)

```rust
#[tokio::test]
async fn test_trigger_dream_no_trigger_manager_fails_fast() {
    // Handlers WITHOUT trigger_manager
    let handlers = create_test_handlers(); // trigger_manager is None

    let args = json!({ "rationale": "Test" });
    let response = handlers.call_trigger_dream(Some(JsonRpcId::Number(5)), args).await;

    // EVIDENCE: Should fail fast with specific error
    assert!(response.error.is_some(), "Should fail fast when TriggerManager missing");
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::DREAM_NOT_INITIALIZED);
    assert!(error.message.contains("TriggerManager"));
}
```

---

## EDGE CASES TO TEST

| Case | Input | Expected Output | Source of Truth Check |
|------|-------|-----------------|----------------------|
| Empty rationale | `{ "rationale": "" }` | Error: rationale required | Response error |
| Whitespace rationale | `{ "rationale": "   " }` | Error: rationale required | Response error |
| Max length rationale | 1024 chars | Accepted | `trigger_manager.check_triggers() == Manual` |
| GPU exactly 80% | 0.80 usage | Blocked (>= not <) | `triggered=false, gpu_eligible=false` |
| GPU exactly 79.99% | 0.7999 usage | Accepted | `trigger_manager.check_triggers() == Manual` |
| In cooldown | After previous trigger | Blocked | `cooldown_remaining_secs > 0` |
| Force during cooldown | force=true + cooldown | Accepted | `trigger_manager.check_triggers() == Manual` |

---

## FULL STATE VERIFICATION CHECKLIST

After completing implementation:

### 1. Source of Truth
- **Where**: `TriggerManager` struct's internal `manual_trigger: bool` field
- **How to verify**: Call `trigger_manager.check_triggers()` - should return `Some(ExtendedTriggerReason::Manual)`

### 2. Execute & Inspect
```rust
// BEFORE
let before = trigger_manager.read().check_triggers(); // Should be None

// EXECUTE
handlers.call_trigger_dream(id, json!({"rationale": "test"})).await;

// AFTER - READ SOURCE OF TRUTH
let after = trigger_manager.read().check_triggers();
assert!(matches!(after, Some(ExtendedTriggerReason::Manual)));
```

### 3. Boundary & Edge Cases (3 Required)

**Edge Case 1: GPU at threshold boundary**
```rust
// BEFORE
trigger_manager.write().update_gpu_usage(0.80); // Exactly 80%
let before = trigger_manager.read().check_triggers();
println!("BEFORE: GPU=80%, triggers={:?}", before);

// EXECUTE
let result = handlers.call_trigger_dream(...);

// AFTER
let after = trigger_manager.read().check_triggers();
println!("AFTER: triggers={:?}", after);
// EVIDENCE: after should be None because 80% is NOT < 80%
```

**Edge Case 2: Empty input**
```rust
// BEFORE
let before = trigger_manager.read().check_triggers();
println!("BEFORE: triggers={:?}", before);

// EXECUTE
let result = handlers.call_trigger_dream(id, json!({})).await;

// AFTER
let after = trigger_manager.read().check_triggers();
println!("AFTER: triggers={:?}", after);
// EVIDENCE: after should equal before (no change, validation failed)
```

**Edge Case 3: Force during IC crisis**
```rust
// BEFORE - IC in crisis state
trigger_manager.write().update_identity_coherence(0.3);
let before = trigger_manager.read().check_triggers();
println!("BEFORE: IC=0.3, triggers={:?}", before);
// before should be Some(IdentityCritical{ic_value: 0.3})

// EXECUTE
let result = handlers.call_trigger_dream(id, json!({"rationale": "test", "force": true})).await;

// AFTER
let after = trigger_manager.read().check_triggers();
println!("AFTER: triggers={:?}", after);
// EVIDENCE: after should be Some(Manual) because Manual > IdentityCritical priority
```

### 4. Evidence Log Template
```
=== TASK-35 FSV Evidence Log ===
Test: [test name]
Timestamp: [ISO timestamp]

BEFORE STATE:
  - check_triggers(): [value]
  - current_gpu_usage(): [value]
  - cooldown_remaining(): [value]

EXECUTE:
  - call_trigger_dream(rationale="[value]", force=[value])

AFTER STATE:
  - check_triggers(): [value]
  - manual_trigger field: [value via check_triggers]

VERIFICATION:
  - Expected: [what should happen]
  - Actual: [what happened]
  - PASS/FAIL: [result]

PHYSICAL PROOF:
  - TriggerManager internal state shows manual_trigger=true
  - check_triggers() returns Some(Manual)
===
```

---

## MANUAL TESTING PROTOCOL

### Test 1: Basic Manual Trigger via MCP

```bash
# 1. Start MCP server
cargo run -p context-graph-mcp

# 2. Send trigger_dream request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"trigger_dream","arguments":{"rationale":"Manual test"}}}' | nc localhost 3000

# 3. EXPECTED RESPONSE:
# {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\"triggered\":true,\"trigger_reason\":\"Manual\",...}"}]}}

# 4. VERIFY: Call get_dream_status to confirm trigger state
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_dream_status","arguments":{}}}' | nc localhost 3000
```

### Test 2: GPU Eligibility Block

```bash
# With GPU mocked at 85%:
# Response should show triggered=false, gpu_eligible=false
```

---

## DEPENDENCIES

| Dependency | Status | What It Provides |
|------------|--------|------------------|
| TASK-21 | DONE | `TriggerManager` struct with `request_manual_trigger()` |
| TASK-24 | DONE | `DreamEventListener` wired to `TriggerManager` for IC triggers |
| TASK-23 | DONE | `NvmlGpuMonitor` for real GPU monitoring (feature-gated) |

---

## FILES TO MODIFY

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/core/handlers.rs` | Add `trigger_manager` field + builder |
| `crates/context-graph-mcp/src/handlers/dream.rs` | Replace `call_trigger_dream` implementation |
| `crates/context-graph-mcp/src/tools/definitions/dream.rs` | Update schema to require rationale |
| `crates/context-graph-mcp/src/protocol.rs` | Add `DREAM_TRIGGER_FAILED` error code |
| `crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools/dream_tools.rs` | Add FSV tests |

---

## VERIFICATION COMMANDS

```bash
# Build to check compilation
cargo check -p context-graph-mcp

# Run specific tests
cargo test -p context-graph-mcp trigger_dream -- --nocapture

# Run all dream tests
cargo test -p context-graph-mcp dream -- --nocapture

# Verify tool appears in tools/list
cargo run -p context-graph-mcp -- --list-tools | grep trigger_dream
```

---

## SUCCESS CRITERIA

1. [x] `call_trigger_dream` calls `TriggerManager.request_manual_trigger()`
2. [x] `rationale` parameter is REQUIRED (validation error if missing)
3. [x] GPU eligibility checked (< 80%) unless `force=true`
4. [x] Cooldown respected unless `force=true`
5. [x] Full State Verification: `check_triggers()` returns `Some(Manual)` after call
6. [x] All 6 tests pass with real data (NO MOCKS for trigger logic)
7. [x] `cargo test -p context-graph-mcp dream` passes (14 tests)
8. [ ] Manual MCP test via netcat/curl succeeds (requires server startup)

---

## ANTI-PATTERNS TO AVOID

- **AP-26**: No silent failures - FAIL FAST if TriggerManager missing
- **AP-12**: No magic numbers - use named constants for thresholds
- **AP-14**: No `.unwrap()` - use proper error handling
- **NO BACKWARDS COMPATIBILITY**: Old code paths removed, new ones must work
- **NO MOCK DATA IN TESTS**: Use real `TriggerManager` instances

---

## CONSTITUTION REFERENCES

| Ref | Location | Value |
|-----|----------|-------|
| `dream.trigger.gpu` | Line 255 | `"<80%"` - Eligibility to START dream |
| `dream.constraints.gpu` | Line 273 | `"<30%"` - Budget DURING dream |
| `perf.latency.dream_wake` | Line 130 | `"<100ms"` - Wake latency |
| `AP-26` | Line 88 | `"IC<0.5 MUST trigger dream - no silent failures"` |
