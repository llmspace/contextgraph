# TASK-09: Fix Johari Blind/Unknown action mapping

```xml
<task_spec id="TASK-09" version="2.0">
<metadata>
  <title>Fix Johari Blind/Unknown action mapping in functions.rs</title>
  <original_id>TASK-UTL-001</original_id>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>9</sequence>
  <implements><requirement_ref>REQ-UTL-001</requirement_ref></implements>
  <depends_on>NONE</depends_on>
  <blocks>TASK-31</blocks>
  <estimated_hours>1</estimated_hours>
</metadata>

<context>
## Problem Statement

The file `crates/context-graph-utl/src/johari/retrieval/functions.rs` has Blind and Unknown action mappings SWAPPED compared to the constitution.yaml specification.

### Constitution Specification (docs2/constitution.yaml lines 154-157)
```yaml
utl:
  johari:
    Open: "ΔS<0.5, ΔC>0.5 → DirectRecall"
    Blind: "ΔS>0.5, ΔC<0.5 → TriggerDream"      # <-- CORRECT
    Hidden: "ΔS<0.5, ΔC<0.5 → GetNeighborhood"
    Unknown: "ΔS>0.5, ΔC>0.5 → EpistemicAction"  # <-- CORRECT
```

### Current BUGGY Implementation (functions.rs lines 34-41)
```rust
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        JohariQuadrant::Open => SuggestedAction::DirectRecall,
        JohariQuadrant::Blind => SuggestedAction::EpistemicAction,  // ❌ WRONG
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
        JohariQuadrant::Unknown => SuggestedAction::TriggerDream,   // ❌ WRONG
    }
}
```

### Impact
This bug affects any code calling `get_suggested_action()` from `context_graph_utl::johari::retrieval::functions`. The incorrect mapping would:
- For Blind quadrant (high surprise, low confidence): Trigger EpistemicAction (updating beliefs we don't trust) instead of TriggerDream (consolidation)
- For Unknown quadrant (high surprise, high confidence): Trigger TriggerDream (sleeping on info) instead of EpistemicAction (acting on trusted info)

### Why This Is Isolated
Other parts of the codebase have the CORRECT mapping:
- `crates/context-graph-utl/src/processor/utl_processor.rs` (lines 236-243) - CORRECT
- `crates/context-graph-mcp/src/middleware/cognitive_pulse.rs` (lines 11-13) - CORRECT
- `crates/context-graph-core/src/types/fingerprint/johari/thresholds.rs` (lines 145-146) - CORRECT

The bug is ONLY in `functions.rs` and the tests that verify the buggy behavior.
</context>

<input_context_files>
- crates/context-graph-utl/src/johari/retrieval/functions.rs (MODIFY - swap Blind/Unknown)
- crates/context-graph-utl/src/johari/retrieval/tests.rs (MODIFY - fix test assertions)
- crates/context-graph-utl/src/johari/retrieval/action.rs (READ ONLY - understand SuggestedAction)
- crates/context-graph-utl/src/johari/retrieval/quadrant_retrieval.rs (READ ONLY - doc examples need fixing)
- docs2/constitution.yaml (READ ONLY - source of truth for mapping, lines 154-157)
</input_context_files>

<scope>
<in_scope>
1. Fix `get_suggested_action()` in functions.rs - swap Blind↔TriggerDream, Unknown↔EpistemicAction
2. Fix ALL tests in tests.rs that verify the wrong mapping
3. Fix doc comments/examples in functions.rs that show wrong mapping (lines 29-31)
4. Fix doc comments/examples in quadrant_retrieval.rs that show wrong mapping (line 33)
5. Add constitution reference comment to the function
6. Verify fix with cargo test
</in_scope>
<out_of_scope>
- Other Johari functions (already correct)
- MCP tool integration (TASK-31 depends on this task)
- utl_processor.rs (already correct)
- cognitive_pulse.rs (already correct)
- thresholds.rs (already correct)
</out_of_scope>
</scope>

<definition_of_done>
<signatures>
```rust
// FILE: crates/context-graph-utl/src/johari/retrieval/functions.rs

use context_graph_core::types::JohariQuadrant;
use super::action::SuggestedAction;

/// Returns the suggested action for a given Johari quadrant.
///
/// # Constitution Compliance (constitution.yaml utl.johari lines 154-157)
/// - Open (ΔS<0.5, ΔC>0.5) → DirectRecall
/// - Hidden (ΔS<0.5, ΔC<0.5) → GetNeighborhood
/// - Blind (ΔS>0.5, ΔC<0.5) → TriggerDream (ISS-011 FIX)
/// - Unknown (ΔS>0.5, ΔC>0.5) → EpistemicAction (ISS-011 FIX)
///
/// # Example
/// ```
/// use context_graph_utl::johari::{get_suggested_action, SuggestedAction, JohariQuadrant};
///
/// assert_eq!(get_suggested_action(JohariQuadrant::Open), SuggestedAction::DirectRecall);
/// assert_eq!(get_suggested_action(JohariQuadrant::Blind), SuggestedAction::TriggerDream);
/// assert_eq!(get_suggested_action(JohariQuadrant::Hidden), SuggestedAction::GetNeighborhood);
/// assert_eq!(get_suggested_action(JohariQuadrant::Unknown), SuggestedAction::EpistemicAction);
/// ```
#[inline]
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        // Low surprise, high confidence → Direct retrieval works
        JohariQuadrant::Open => SuggestedAction::DirectRecall,

        // Low surprise, low confidence → Explore neighborhood for context
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,

        // High surprise, low confidence → Need dream consolidation to integrate
        // FIXED ISS-011: Was incorrectly EpistemicAction
        JohariQuadrant::Blind => SuggestedAction::TriggerDream,

        // High surprise, high confidence → Epistemic action to update beliefs
        // FIXED ISS-011: Was incorrectly TriggerDream
        JohariQuadrant::Unknown => SuggestedAction::EpistemicAction,
    }
}
```
</signatures>

<test_assertions>
```rust
// FILE: crates/context-graph-utl/src/johari/retrieval/tests.rs

#[test]
fn test_suggested_action_mapping() {
    // Constitution compliance: utl.johari (lines 154-157)
    assert_eq!(
        get_suggested_action(JohariQuadrant::Open),
        SuggestedAction::DirectRecall,
        "Open (low surprise, high confidence) → DirectRecall"
    );
    assert_eq!(
        get_suggested_action(JohariQuadrant::Blind),
        SuggestedAction::TriggerDream,  // FIXED: Was EpistemicAction
        "Blind (high surprise, low confidence) → TriggerDream"
    );
    assert_eq!(
        get_suggested_action(JohariQuadrant::Hidden),
        SuggestedAction::GetNeighborhood,
        "Hidden (low surprise, low confidence) → GetNeighborhood"
    );
    assert_eq!(
        get_suggested_action(JohariQuadrant::Unknown),
        SuggestedAction::EpistemicAction,  // FIXED: Was TriggerDream
        "Unknown (high surprise, high confidence) → EpistemicAction"
    );
}

#[test]
fn test_constitution_compliance() {
    let retrieval = QuadrantRetrieval::with_default_weights();

    // Constitution: utl.johari.Open = "ΔS<0.5, ΔC>0.5 → DirectRecall"
    assert_eq!(
        retrieval.get_action(JohariQuadrant::Open),
        SuggestedAction::DirectRecall
    );

    // Constitution: utl.johari.Blind = "ΔS>0.5, ΔC<0.5 → TriggerDream"
    assert_eq!(
        retrieval.get_action(JohariQuadrant::Blind),
        SuggestedAction::TriggerDream  // FIXED
    );

    // Constitution: utl.johari.Hidden = "ΔS<0.5, ΔC<0.5 → GetNeighborhood"
    assert_eq!(
        retrieval.get_action(JohariQuadrant::Hidden),
        SuggestedAction::GetNeighborhood
    );

    // Constitution: utl.johari.Unknown = "ΔS>0.5, ΔC>0.5 → EpistemicAction"
    assert_eq!(
        retrieval.get_action(JohariQuadrant::Unknown),
        SuggestedAction::EpistemicAction  // FIXED
    );
}

#[test]
fn test_quadrant_retrieval_get_action() {
    let retrieval = QuadrantRetrieval::with_default_weights();

    assert_eq!(
        retrieval.get_action(JohariQuadrant::Open),
        SuggestedAction::DirectRecall
    );
    assert_eq!(
        retrieval.get_action(JohariQuadrant::Blind),
        SuggestedAction::TriggerDream  // FIXED
    );
}
```
</test_assertions>

<constraints>
- Blind MUST map to TriggerDream (not EpistemicAction)
- Unknown MUST map to EpistemicAction (not TriggerDream)
- All four quadrants MUST be tested
- Documentation MUST reference constitution.yaml utl.johari section
- NO BACKWARDS COMPATIBILITY - wrong behavior must error/fail
</constraints>

<verification>
```bash
# Unit tests for the specific module
cargo test -p context-graph-utl johari::retrieval

# All Johari tests across the crate
cargo test -p context-graph-utl johari

# Doc tests to verify examples compile and pass
cargo test -p context-graph-utl --doc

# Integration tests
cargo test -p context-graph-utl --test integration_tests johari

# Full crate test
cargo test -p context-graph-utl
```
</verification>
</definition_of_done>

<files_to_modify>
1. `crates/context-graph-utl/src/johari/retrieval/functions.rs`
   - Line 29-31: Fix doc example assertions
   - Line 34-41: Swap Blind↔TriggerDream, Unknown↔EpistemicAction
   - Add constitution reference comment

2. `crates/context-graph-utl/src/johari/retrieval/tests.rs`
   - Line 19-30: Fix test_suggested_action_mapping assertions
   - Line 133-140: Fix test_quadrant_retrieval_get_action assertions
   - Line 197-224: Fix test_constitution_compliance assertions

3. `crates/context-graph-utl/src/johari/retrieval/quadrant_retrieval.rs`
   - Line 33: Fix doc example (Blind should map to TriggerDream)
</files_to_modify>

<files_to_create>
NONE - this is a bug fix, not new functionality
</files_to_create>

<test_commands>
```bash
# MANDATORY: Run before claiming task complete
cargo test -p context-graph-utl johari
cargo test -p context-graph-utl --doc
cargo test --workspace
```
</test_commands>
</task_spec>
```

---

## Semantic Reasoning

### Why the Constitution Mapping Makes Sense

**Blind (high surprise ΔS>0.5, low confidence ΔC<0.5) → TriggerDream**
- High surprise = unexpected/novel information
- Low confidence = don't trust it yet
- Action: TriggerDream to consolidate and integrate the surprise through sleep-like processing
- Semantic: "I see something surprising but don't trust it - let me sleep on it"

**Unknown (high surprise ΔS>0.5, high confidence ΔC>0.5) → EpistemicAction**
- High surprise = unexpected/novel information
- High confidence = trust the source
- Action: EpistemicAction to explicitly update beliefs based on trusted new info
- Semantic: "I see something surprising and trust it - I should update my beliefs"

### Why the Bug Was Wrong

The swapped version created semantically incorrect behavior:
- Blind→EpistemicAction would update beliefs based on info we DON'T trust
- Unknown→TriggerDream would sleep on info we SHOULD act on immediately

---

## Full State Verification Requirements

After implementing the fix, you MUST perform these verification steps:

### Source of Truth Identification
- **Primary**: `functions.rs::get_suggested_action()` return values
- **Secondary**: Test assertions in `tests.rs`
- **Validation**: Constitution.yaml lines 154-157

### Execute & Inspect Protocol
1. Run `cargo test -p context-graph-utl johari::retrieval::tests::test_suggested_action_mapping`
2. Verify test output shows PASS (not just no errors - check the assertion messages)
3. Run `cargo test -p context-graph-utl --doc` to verify doc examples pass

### Manual Test Cases

#### Test Case 1: Blind Quadrant
```rust
// Input: JohariQuadrant::Blind
// Expected Output: SuggestedAction::TriggerDream
// Before state: get_suggested_action(Blind) returns EpistemicAction (WRONG)
// After state: get_suggested_action(Blind) returns TriggerDream (CORRECT)
```

#### Test Case 2: Unknown Quadrant
```rust
// Input: JohariQuadrant::Unknown
// Expected Output: SuggestedAction::EpistemicAction
// Before state: get_suggested_action(Unknown) returns TriggerDream (WRONG)
// After state: get_suggested_action(Unknown) returns EpistemicAction (CORRECT)
```

#### Test Case 3: Open/Hidden Unchanged
```rust
// Input: JohariQuadrant::Open, JohariQuadrant::Hidden
// Expected: DirectRecall, GetNeighborhood (unchanged)
// Verify no regression in these mappings
```

### Edge Case Audit

1. **QuadrantRetrieval delegation**: Verify `QuadrantRetrieval::get_action()` calls `get_suggested_action()` and inherits the fix
2. **Doc examples**: Verify all `/// # Example` blocks compile and assert correct values
3. **Re-exports**: Verify `context_graph_utl::johari::get_suggested_action` re-export works

### Evidence of Success Log
After implementation, provide output of:
```bash
cargo test -p context-graph-utl johari 2>&1 | grep -E "(test_suggested_action_mapping|test_constitution_compliance|PASSED|FAILED)"
```

This MUST show all relevant tests PASSED.

---

## Regression Prevention

The test explicitly documents the constitution reference in assertion messages:
- `"Blind (high surprise, low confidence) → TriggerDream"`
- `"Unknown (high surprise, high confidence) → EpistemicAction"`

Any future change to this mapping will fail with a descriptive message pointing to the constitution spec.

---

## Related Files Reference

| File | Status | Action |
|------|--------|--------|
| `functions.rs` | BUGGY | FIX |
| `tests.rs` | BUGGY (tests wrong behavior) | FIX |
| `quadrant_retrieval.rs` | Doc example wrong | FIX doc |
| `action.rs` | Correct | READ ONLY |
| `utl_processor.rs` | Correct | DO NOT MODIFY |
| `cognitive_pulse.rs` | Correct | DO NOT MODIFY |
| `thresholds.rs` | Correct | DO NOT MODIFY |
| `johari_tests.rs` | Correct (integration test) | DO NOT MODIFY |
