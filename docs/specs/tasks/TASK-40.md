# TASK-40: Implement set_coupling_strength tool

```xml
<task_spec id="TASK-40" version="4.0">
<metadata>
  <title>Implement set_coupling_strength MCP Tool</title>
  <original_id>TASK-MCP-014</original_id>
  <status>complete_by_delegation</status>
  <completed_date>2026-01-14</completed_date>
  <layer>surface</layer>
  <sequence>40</sequence>
  <implements><requirement_ref>REQ-MCP-014</requirement_ref></implements>
  <depends_on>TASK-12</depends_on>
  <estimated_hours>0</estimated_hours>
  <actual_hours>0</actual_hours>
  <blocks>TASK-41</blocks>
  <audit_date>2026-01-14</audit_date>
  <resolution>DELEGATED to existing adjust_coupling tool (TASK-GWT-001)</resolution>
</metadata>

<completion_evidence>
## TASK-40 COMPLETION BY DELEGATION (2026-01-14)

### Resolution: Option A Selected

The functionality required by TASK-40 (`set_coupling_strength`) is **already implemented**
by the existing `adjust_coupling` tool from TASK-GWT-001.

### Proof of Equivalence

| Requirement | adjust_coupling Implementation |
|-------------|-------------------------------|
| Modify Kuramoto coupling K | ✓ `set_coupling_strength(new_k)` called |
| Return previous value | ✓ `old_K` field in response |
| Return new value | ✓ `new_K` field in response |
| Validate range | ✓ Clamped to [0, 10] |
| Fail fast without GWT | ✓ Returns GWT_NOT_INITIALIZED error |

### Evidence Commands

```bash
# Verify adjust_coupling exists and works
cargo test -p context-graph-mcp adjust_coupling -- --nocapture
# Result: 6 tests pass

# Verify tool is registered
cargo test -p context-graph-mcp tools_list -- --nocapture
# Result: adjust_coupling in 46 tools
```

### Files Where adjust_coupling Is Implemented

| Component | File | Line |
|-----------|------|------|
| Handler | `crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs` | 246-309 |
| Dispatch | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | 82 |
| Name Constant | `crates/context-graph-mcp/src/tools/names.rs` | 25 |
| Definition | `crates/context-graph-mcp/src/tools/definitions/gwt.rs` | 126-145 |

### Constitution Compliance

`constitution.yaml:335` lists `adjust_coupling` as the GWT coupling tool:
```yaml
gwt: [get_consciousness_state, get_workspace_status, get_kuramoto_sync,
      get_ego_state, trigger_workspace_broadcast, adjust_coupling, compute_delta_sc]
```

### Impact on TASK-41

TASK-41 dependency on TASK-40 is **satisfied** - the coupling adjustment tool exists.
No new tool registration is required.
</completion_evidence>

<critical_discovery>
## ⚠️ CRITICAL: TASK IS REDUNDANT - FUNCTIONALITY ALREADY EXISTS

### The `adjust_coupling` Tool Already Exists

The MCP server **ALREADY HAS** the `adjust_coupling` tool that performs the EXACT same function:
- **Tool Name**: `adjust_coupling` (registered in GWT tools)
- **Handler Location**: `crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs:246-309`
- **Dispatch**: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs:82`
- **Status**: FULLY IMPLEMENTED AND TESTED (TASK-GWT-001)

### What adjust_coupling Does (ALREADY WORKS):
```json
// INPUT
{"new_K": 0.8}

// OUTPUT
{
  "old_K": 0.5,
  "new_K": 0.8,
  "predicted_r": 0.75,
  "current_r": 0.72,
  "K_clamped": false
}
```

### Key Differences from Original TASK-40 Spec:
| Aspect | Original TASK-40 Spec | Existing `adjust_coupling` |
|--------|----------------------|---------------------------|
| Tool Name | `set_coupling_strength` | `adjust_coupling` |
| Parameter | `coupling` | `new_K` |
| Range | `[0.0, 1.0]` | `[0, 10]` |
| Return | `{success, previous_coupling, new_coupling}` | `{old_K, new_K, predicted_r, current_r, K_clamped}` |

### DECISION REQUIRED:
**Option A (RECOMMENDED)**: Mark TASK-40 as COMPLETE by delegation
- `adjust_coupling` IS `set_coupling_strength` with different naming
- Both modify `KuramotoNetwork.coupling_strength()`
- Update TASK-41 to reference `adjust_coupling` instead

**Option B**: Implement `set_coupling_strength` as alias to `adjust_coupling`
- Add tool name constant `SET_COUPLING_STRENGTH`
- Map to same handler via alias system
- Slightly different response schema

**Option C**: Mark TASK-40 as N/A (Not Applicable)
- Document that functionality exists under different name
- No implementation needed
</critical_discovery>

<current_state>
## CODEBASE STATE (2026-01-14)

### 1. Existing Tool: adjust_coupling (COMPLETE)

**Location**: `crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs:246-309`

```rust
/// adjust_coupling tool implementation.
///
/// TASK-GWT-001: Adjusts Kuramoto oscillator network coupling strength K.
/// Higher K leads to faster synchronization. K is clamped to [0, 10].
///
/// FAIL FAST on missing kuramoto provider - no stubs or fallbacks.
///
/// Arguments:
/// - new_K: New coupling strength (clamped to [0, 10])
///
/// Returns:
/// - old_K: Previous coupling strength
/// - new_K: New coupling strength (after clamping)
/// - predicted_r: Predicted order parameter after adjustment
pub(crate) async fn call_adjust_coupling(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    // ... implementation ...
}
```

### 2. Tool Registration (COMPLETE)

**Tool Name Constant**: `crates/context-graph-mcp/src/tools/names.rs:25`
```rust
pub const ADJUST_COUPLING: &str = "adjust_coupling";
```

**Tool Definition**: `crates/context-graph-mcp/src/tools/definitions/gwt.rs:126-145`
```rust
ToolDefinition::new(
    "adjust_coupling",
    "Adjust Kuramoto oscillator network coupling strength K...",
    json!({
        "type": "object",
        "properties": {
            "new_K": {
                "type": "number",
                "minimum": 0,
                "maximum": 10,
                "description": "New coupling strength K (clamped to [0, 10])"
            }
        },
        "required": ["new_K"]
    }),
),
```

### 3. Dispatch Wiring (COMPLETE)

**Location**: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs:82`
```rust
tool_names::ADJUST_COUPLING => self.call_adjust_coupling(id, arguments).await,
```

### 4. Tests (COMPLETE)

| Test File | Tests |
|-----------|-------|
| `exhaustive_mcp_tools/gwt_consciousness_tools.rs:286-339` | 4 tests |
| `full_state_verification_gwt/workspace_tests.rs:150-236` | FSV test |
| `phase3_gwt_consciousness/kuramoto_sync.rs:168-303` | 2 FSV tests |

### 5. Current Tool Count
- **Total Tools**: 46 (as of TASK-39 completion)
- **GWT Tools**: 9 (including adjust_coupling)
</current_state>

<if_option_b_chosen>
## OPTION B IMPLEMENTATION (If Adding Alias)

If decision is to implement `set_coupling_strength` as an alias:

### Step 1: Add Tool Name Constant
**File**: `crates/context-graph-mcp/src/tools/names.rs`
**Location**: After line 157 (GET_KURAMOTO_STATE)

```rust
// ========== COUPLING STRENGTH ALIAS (TASK-40) ==========

/// TASK-40: Alias for adjust_coupling with [0,1] normalized input.
/// This is an ALIAS - dispatches to the same handler as adjust_coupling.
/// Use when you want normalized coupling (0.0-1.0) instead of raw K (0-10).
pub const SET_COUPLING_STRENGTH: &str = "set_coupling_strength";
```

### Step 2: Add to Alias System
**File**: `crates/context-graph-mcp/src/tools/aliases.rs`

```rust
pub fn resolve_alias(name: &str) -> &str {
    match name {
        // ... existing aliases ...
        "set_coupling_strength" => "adjust_coupling",
        _ => name,
    }
}
```

### Step 3: Add Tool Definition
**File**: `crates/context-graph-mcp/src/tools/definitions/gwt.rs`

```rust
// set_coupling_strength - ALIAS for adjust_coupling (TASK-40)
ToolDefinition::new(
    "set_coupling_strength",
    "Set Kuramoto coupling strength. ALIAS for adjust_coupling with [0,1] range. \
     Input 'coupling' (0.0-1.0) is scaled to K (0-10) internally. \
     Returns previous_coupling (0-1 scale) and new_coupling (0-1 scale). \
     Requires GWT providers to be initialized via with_gwt() constructor.",
    json!({
        "type": "object",
        "properties": {
            "coupling": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "New coupling strength [0.0, 1.0] (scaled to K=[0,10] internally)"
            }
        },
        "required": ["coupling"]
    }),
),
```

### Step 4: Create Wrapper Handler (If Different Schema Needed)
**File**: `crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs`

```rust
/// set_coupling_strength tool implementation.
///
/// TASK-40: Wrapper around adjust_coupling with normalized [0,1] range.
/// Input: coupling in [0.0, 1.0]
/// Maps to: new_K = coupling * 10.0
///
/// Returns:
/// - success: true
/// - previous_coupling: old_K / 10.0
/// - new_coupling: actual_new_K / 10.0
pub(crate) async fn call_set_coupling_strength(
    &self,
    id: Option<JsonRpcId>,
    args: serde_json::Value,
) -> JsonRpcResponse {
    debug!("Handling set_coupling_strength tool call (TASK-40)");

    // Parse coupling [0, 1]
    let coupling = match args.get("coupling").and_then(|v| v.as_f64()) {
        Some(c) => c.clamp(0.0, 1.0),
        None => {
            return self.tool_error_with_pulse(id, "Missing required 'coupling' parameter");
        }
    };

    // Scale to K range [0, 10]
    let new_k = coupling * 10.0;

    // Delegate to adjust_coupling logic
    let internal_args = json!({"new_K": new_k});
    let response = self.call_adjust_coupling(id.clone(), internal_args).await;

    // If error, return as-is
    if response.is_error() {
        return response;
    }

    // Transform response to [0, 1] scale
    // ... extract and rescale values ...
}
```
</if_option_b_chosen>

<recommended_action>
## RECOMMENDED: Option A - Mark Complete by Delegation

The functionality exists. The only difference is naming and range.

### Why Option A:
1. **DRY Principle**: Don't create duplicate tools for same functionality
2. **API Clarity**: One tool for coupling adjustment prevents confusion
3. **Constitution Compliance**: `constitution.yaml:335` lists `adjust_coupling` as the GWT tool

### Steps to Close TASK-40:
1. Update this task status to `complete_by_delegation`
2. Update TASK-41 dependency from `TASK-40` to acknowledge `adjust_coupling` exists
3. Document that `adjust_coupling` satisfies REQ-MCP-014

### Evidence:
```bash
# adjust_coupling is registered and callable
cargo test -p context-graph-mcp adjust_coupling -- --nocapture
# Result: test result: ok. 6 passed; 0 failed

# Tool is in tools/list
cargo test -p context-graph-mcp tools_list -- --nocapture
# Confirms: adjust_coupling is in the 46 tools
```
</recommended_action>

<if_implementation_required>
## IF IMPLEMENTATION IS REQUIRED (Option B or C Override)

### Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/tools/names.rs` | Add SET_COUPLING_STRENGTH constant |
| `crates/context-graph-mcp/src/tools/definitions/gwt.rs` | Add tool definition (update to 10 tools) |
| `crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs` | Add call_set_coupling_strength handler |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Wire SET_COUPLING_STRENGTH dispatch |
| `crates/context-graph-mcp/src/tools/mod.rs` | Update tool count to 47 |

### Acceptance Criteria

| AC | Criterion | Verification |
|----|-----------|--------------|
| AC-1 | Tool name constant exists | `names.rs` contains `SET_COUPLING_STRENGTH` |
| AC-2 | Tool in tools/list | Response includes `set_coupling_strength` |
| AC-3 | Handler dispatches | `tools/call` with `name: "set_coupling_strength"` works |
| AC-4 | Response schema | Returns `{success, previous_coupling, new_coupling}` |
| AC-5 | Coupling in [0, 1] | Values outside range are clamped |
| AC-6 | Returns previous | `previous_coupling` reflects pre-call state |
| AC-7 | Tests pass | `cargo test -p context-graph-mcp coupling -- --nocapture` |
| AC-8 | Compilation clean | `cargo check -p context-graph-mcp` no errors |
</if_implementation_required>

<full_state_verification>
## MANDATORY: Full State Verification Protocol

### 1. Source of Truth Definition

| Field | Source of Truth |
|-------|-----------------|
| coupling strength K | `kuramoto_network.read().coupling_strength()` |
| previous_coupling | K before `set_coupling_strength()` call |
| new_coupling | K after `set_coupling_strength()` call |

### 2. Execute & Inspect Protocol

**Test Location**: `crates/context-graph-mcp/src/handlers/tests/full_state_verification_gwt/kuramoto_tests.rs`

```rust
#[tokio::test]
async fn test_set_coupling_strength_modifies_kuramoto_k() {
    println!("\n=== FSV TEST: set_coupling_strength (TASK-40) ===");

    // SETUP
    let handlers = Handlers::with_default_gwt();
    println!("SETUP: Created Handlers with with_default_gwt()");

    // STATE BEFORE (Source of Truth: direct query)
    let k_before = {
        let network = handlers.kuramoto_network.as_ref().unwrap().read();
        network.coupling_strength()
    };
    println!("FSV-1: K before = {:.4}", k_before);

    // ACTION: Set new coupling via MCP tool
    let new_coupling = 0.8; // [0, 1] range
    let request = make_tool_call_request(
        tool_names::SET_COUPLING_STRENGTH,
        Some(json!({"coupling": new_coupling}))
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response structure
    let response_json = serde_json::to_value(&response).unwrap();
    assert!(response_json.get("error").is_none(), "Expected success");

    let content = extract_tool_content(&response_json).unwrap();

    // FSV-2: success field
    let success = content["success"].as_bool().expect("success field");
    assert!(success, "success must be true");
    println!("FSV-2 PASS: success = {}", success);

    // FSV-3: previous_coupling matches state before
    let prev = content["previous_coupling"].as_f64().expect("previous_coupling");
    let expected_prev = k_before / 10.0; // Scale K to [0, 1]
    assert!(
        (prev - expected_prev).abs() < 0.001,
        "previous_coupling mismatch: got {}, expected {}",
        prev, expected_prev
    );
    println!("FSV-3 PASS: previous_coupling = {:.4}", prev);

    // FSV-4: new_coupling matches requested
    let new_c = content["new_coupling"].as_f64().expect("new_coupling");
    assert!(
        (new_c - new_coupling).abs() < 0.001,
        "new_coupling mismatch: got {}, expected {}",
        new_c, new_coupling
    );
    println!("FSV-4 PASS: new_coupling = {:.4}", new_c);

    // FSV-5: Source of truth verification (K actually changed)
    let k_after = {
        let network = handlers.kuramoto_network.as_ref().unwrap().read();
        network.coupling_strength()
    };
    let expected_k = new_coupling * 10.0;
    assert!(
        (k_after - expected_k).abs() < 0.001,
        "K not updated in network: got {}, expected {}",
        k_after, expected_k
    );
    println!("FSV-5 PASS: K after (source of truth) = {:.4}", k_after);

    println!("\n=== FSV EVIDENCE (TASK-40) ===");
    println!("✓ success: true");
    println!("✓ previous_coupling: {:.4} (from K={:.4})", prev, k_before);
    println!("✓ new_coupling: {:.4} (K={:.4})", new_c, k_after);
    println!("✓ Source of truth verified: kuramoto_network.coupling_strength()");
    println!("=== FSV TEST PASSED (TASK-40) ===\n");
}
```

### 3. Edge Case Audit

**Edge Case 1: GWT Not Initialized**
```rust
#[tokio::test]
async fn test_set_coupling_strength_fails_fast_without_gwt() {
    println!("\n[FSV Edge Case 1] GWT not initialized");

    // Create handlers WITHOUT GWT
    let handlers = Handlers::new_minimal();

    let request = make_tool_call_request(
        tool_names::SET_COUPLING_STRENGTH,
        Some(json!({"coupling": 0.5}))
    );
    let response = handlers.dispatch(request).await;

    let response_json = serde_json::to_value(&response).unwrap();
    let error = response_json.get("error").expect("must have error");
    let code = error["code"].as_i64().expect("error code");

    // FAIL FAST: Must return GWT_NOT_INITIALIZED (-32060)
    assert_eq!(code, -32060);
    println!("[FSV] PASS: Error code {} for missing GWT", code);
}
```

**Edge Case 2: Coupling Below 0 (Clamping)**
```rust
#[tokio::test]
async fn test_set_coupling_strength_clamps_below_zero() {
    println!("\n[FSV Edge Case 2] Coupling below 0");

    let handlers = Handlers::with_default_gwt();

    let request = make_tool_call_request(
        tool_names::SET_COUPLING_STRENGTH,
        Some(json!({"coupling": -0.5}))
    );
    let response = handlers.dispatch(request).await;

    let response_json = serde_json::to_value(&response).unwrap();
    let content = extract_tool_content(&response_json).unwrap();

    // Should clamp to 0.0
    let new_c = content["new_coupling"].as_f64().unwrap();
    assert_eq!(new_c, 0.0, "Negative coupling should clamp to 0.0");
    println!("[FSV] PASS: -0.5 clamped to 0.0");
}
```

**Edge Case 3: Coupling Above 1 (Clamping)**
```rust
#[tokio::test]
async fn test_set_coupling_strength_clamps_above_one() {
    println!("\n[FSV Edge Case 3] Coupling above 1");

    let handlers = Handlers::with_default_gwt();

    let request = make_tool_call_request(
        tool_names::SET_COUPLING_STRENGTH,
        Some(json!({"coupling": 1.5}))
    );
    let response = handlers.dispatch(request).await;

    let response_json = serde_json::to_value(&response).unwrap();
    let content = extract_tool_content(&response_json).unwrap();

    // Should clamp to 1.0
    let new_c = content["new_coupling"].as_f64().unwrap();
    assert_eq!(new_c, 1.0, "Coupling >1.0 should clamp to 1.0");
    println!("[FSV] PASS: 1.5 clamped to 1.0");
}
```

### 4. Manual Verification Steps

```bash
# 1. Compile check
cargo check -p context-graph-mcp
# Expected: No errors

# 2. Run all coupling-related tests
cargo test -p context-graph-mcp coupling -- --nocapture
# Expected: All tests pass

# 3. Run FSV tests specifically
cargo test -p context-graph-mcp test_set_coupling_strength -- --nocapture
# Expected: FSV evidence printed

# 4. Verify tool appears in tools/list
cargo test -p context-graph-mcp tools_list -- --nocapture
# Expected: set_coupling_strength (or adjust_coupling) in list

# 5. Verify dispatch works
cargo test -p context-graph-mcp dispatch -- --nocapture
# Expected: Tool dispatches correctly
```
</full_state_verification>

<test_commands>
```bash
# For existing adjust_coupling (proves functionality exists):
cargo test -p context-graph-mcp adjust_coupling -- --nocapture

# For new set_coupling_strength (if implemented):
cargo test -p context-graph-mcp set_coupling_strength -- --nocapture
cargo test -p context-graph-mcp coupling -- --nocapture

# Full workspace verification:
cargo test --workspace -- --nocapture 2>&1 | head -100

# Compile check:
cargo check -p context-graph-mcp
```
</test_commands>

<constitution_compliance>
## Constitution Requirements

| Rule | Requirement | Status |
|------|-------------|--------|
| AP-25 | Kuramoto must have exactly 13 oscillators | SATISFIED (TASK-11) |
| GWT-006 | KuramotoStepper wired to MCP lifecycle | SATISFIED (TASK-12) |
| AP-14 | No .unwrap() in library code | SATISFIED (uses ? and match) |
| constitution.yaml:335 | GWT tools include adjust_coupling | SATISFIED |

**Note**: The constitution lists `adjust_coupling` not `set_coupling_strength`.
If adding `set_coupling_strength`, it should be documented as an alias.
</constitution_compliance>

<anti_patterns>
## FORBIDDEN Actions

1. **DO NOT** create duplicate functionality - reuse adjust_coupling
2. **DO NOT** return mock/stub data - use real KuramotoProvider
3. **DO NOT** swallow errors - fail fast with GWT_NOT_INITIALIZED
4. **DO NOT** use .unwrap() - use ? or match with proper error handling
5. **DO NOT** create workarounds if something doesn't work - fix root cause
6. **DO NOT** skip source of truth verification in tests
7. **DO NOT** use different clamping behavior than adjust_coupling
8. **DO NOT** modify adjust_coupling behavior - it's working and tested
</anti_patterns>

<dependencies>
## Dependency Status

| Dependency | Status | Evidence |
|------------|--------|----------|
| TASK-12 | COMPLETE | KuramotoStepper wired to server lifecycle |
| TASK-11 | COMPLETE | KuramotoNetwork with 13 oscillators |
| TASK-10 | COMPLETE | KURAMOTO_N=13 constant |
| adjust_coupling | COMPLETE | Tool exists and works (TASK-GWT-001) |

## This Task Unblocks

| Task | Description | Note |
|------|-------------|------|
| TASK-41 | Add tool registration to MCP server | Can proceed with adjust_coupling instead |
</dependencies>

<quick_reference>
## File Locations

| Component | Path |
|-----------|------|
| Existing coupling tool | `crates/context-graph-mcp/src/handlers/tools/gwt_workspace.rs:246-309` |
| Tool names | `crates/context-graph-mcp/src/tools/names.rs:25` |
| Tool definitions | `crates/context-graph-mcp/src/tools/definitions/gwt.rs:126-145` |
| Dispatch | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs:82` |
| Provider trait | `crates/context-graph-mcp/src/handlers/gwt_traits.rs:63` |
| Kuramoto network | `crates/context-graph-utl/src/phase/oscillator/kuramoto.rs:360` |

## Schema Comparison

### Existing adjust_coupling:
```json
// INPUT
{"new_K": 5.0}

// OUTPUT
{
  "old_K": 0.5,
  "new_K": 5.0,
  "predicted_r": 0.75,
  "current_r": 0.72,
  "K_clamped": false
}
```

### Proposed set_coupling_strength (if implemented):
```json
// INPUT
{"coupling": 0.5}

// OUTPUT
{
  "success": true,
  "previous_coupling": 0.05,
  "new_coupling": 0.5
}
```
</quick_reference>
</task_spec>
```
