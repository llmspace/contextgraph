# TASK-P0-002: Remove North Star Naming Remnants from MCP Crate

## CRITICAL: TASK STATUS RE-EVALUATION (2026-01-16)

**IMPORTANT**: This task document has been COMPLETELY REWRITTEN after auditing the codebase against the original (outdated) spec. The original document was WRONG about what needed to be done.

### What TASK-P0-001 Actually Accomplished

TASK-P0-001 (commit 5f6dfc7, 2026-01-16) successfully:
1. **Removed `GoalLevel::NorthStar`** - Strategic is now the top-level (3-level hierarchy)
2. **Renamed methods**:
   - `north_star()` → `top_level_goals()`
   - `has_north_star()` → `has_top_level_goals()`
   - `path_to_north_star()` → `path_to_root()`
3. **Renamed fields**: `theta_to_north_star` → `alignment_score`
4. **Removed 1 MCP tool**: `auto_bootstrap_north_star`
5. **Kept 5 tools** that were originally marked for removal in outdated spec (they serve valid purposes)

### Tools KEPT (Valid for New Architecture)

| Tool | Reason Kept | New Purpose |
|------|-------------|-------------|
| `get_alignment_drift` | Monitors drift from **Strategic** goals | Detects divergence from emergent topics |
| `get_drift_history` | Historical tracking still useful | Topic stability analysis |
| `trigger_drift_correction` | Manual correction still needed | Realign with strategic goals |
| `get_ego_state` | GWT identity system (NOT North Star) | Purpose vector, identity tracking |
| `get_identity_continuity` | IC monitoring (AP-26, AP-37, AP-38) | Crisis detection thresholds |

### Current Tool Count: 58 (NOT 53)

The functional spec was wrong. Tool count went from 59 → 58 (only `auto_bootstrap_north_star` removed).

---

## Revised Task Scope

TASK-P0-002 must now focus on **cleaning up naming remnants** and **updating documentation**, NOT removing functional tools.

### Actual Remaining Work

#### 1. Update MCP Tool Descriptions (Documentation Only)

**Files to Modify**:
- `crates/context-graph-mcp/src/tools/definitions/autonomous.rs`

**Changes Required**:
- Update tool descriptions that still reference "North Star" to say "Strategic goal" or "top-level goals"
- Example: Line 22-25 says "Drift measures how far the system has deviated from the North Star goal alignment"
- Should say: "Drift measures how far the system has deviated from Strategic goal alignment"

#### 2. Update Handler Comments and Logging

**Files to Modify**:
- `crates/context-graph-mcp/src/handlers/autonomous/status.rs` (lines 54, 114, 119, 136, 143, 177, 204)
- `crates/context-graph-mcp/src/handlers/autonomous/drift.rs` (line 152, 398)
- `crates/context-graph-mcp/src/handlers/purpose/drift.rs` (line 76, 226)
- `crates/context-graph-mcp/src/handlers/memory/search.rs` (line 157)
- `crates/context-graph-mcp/src/handlers/memory/retrieve.rs` (line 80)

**Changes Required**:
- Rename variables like `north_star_status` → `top_level_status`
- Update log messages mentioning "North Star"
- Update JSON response field names (potential API change - evaluate carefully)

#### 3. Update Test File Names

**Files to Rename/Delete**:
- `crates/context-graph-mcp/src/handlers/tests/north_star.rs` - Already marked as deprecated, can be deleted
- `crates/context-graph-mcp/src/handlers/tests/purpose/north_star_alignment.rs` - Delete (tests deprecated tool)
- `crates/context-graph-mcp/src/handlers/tests/purpose/north_star_update.rs` - Delete (tests deprecated tool)

**Files to Update mod.rs**:
- `crates/context-graph-mcp/src/handlers/tests/mod.rs` (line 73)
- `crates/context-graph-mcp/src/handlers/tests/purpose/mod.rs` (lines 34, 38)

#### 4. Update Helper Function Names

**File**: `crates/context-graph-mcp/src/handlers/tests/mod.rs`
- Rename `create_test_handlers_no_north_star()` → `create_test_handlers_no_goals()`
- Rename `create_test_handlers_with_rocksdb_no_north_star()` → `create_test_handlers_with_rocksdb_no_goals()`

---

## Full State Verification Protocol

### Source of Truth
1. **MCP Tool Registry**: `crates/context-graph-mcp/src/tools/registry.rs` - Must have exactly 58 tools
2. **Tool Definitions**: `crates/context-graph-mcp/src/tools/definitions/*.rs` - No "North Star" in user-facing descriptions
3. **Test Results**: `cargo test --package context-graph-mcp` - All tests must pass

### Pre-Execution Verification

```bash
# Count "north_star" references in MCP crate (current baseline)
grep -rn "north_star\|NorthStar" crates/context-graph-mcp/src/ --include="*.rs" | wc -l
# Expected: ~100+ references (comments, variable names, function names)

# Verify tool count is 58
grep -n "assert_eq.*58" crates/context-graph-mcp/src/tools/registry.rs
# Expected: Line ~207-211 shows assertion for 58 tools

# Verify cargo check passes
cargo check --package context-graph-mcp
# Expected: Success (with warnings, no errors)
```

### Post-Execution Verification

```bash
# Count "north_star" references - should be reduced significantly
grep -rn "north_star\|NorthStar" crates/context-graph-mcp/src/ --include="*.rs" | grep -v "// " | wc -l
# Expected: Significantly reduced (only comments mentioning TASK-P0-001)

# Verify no "North Star" in tool descriptions (user-facing)
grep -rn "North Star" crates/context-graph-mcp/src/tools/definitions/ --include="*.rs" | grep -v "//"
# Expected: 0 matches

# All tests must pass
cargo test --package context-graph-mcp
# Expected: All tests pass

# Tool registry count unchanged
cargo test --package context-graph-mcp test_register_all_tools_returns_58
# Expected: PASS
```

### Physical Evidence Verification

After completion, manually verify:
1. **Tool List**: Call `tools/list` MCP method, verify no tool description mentions "North Star"
2. **Status Response**: Call `get_autonomous_status`, verify JSON response uses "strategic_goal" not "north_star"
3. **Test Files**: Verify `north_star.rs`, `north_star_alignment.rs`, `north_star_update.rs` are deleted

---

## Edge Case Testing

### Edge Case 1: API Backwards Compatibility
**Scenario**: External client relies on JSON field name `north_star_configured`
**Action**: Evaluate if this is a breaking change. If clients exist, consider:
  - Keeping field name for now
  - Adding deprecation notice
  - Supporting both field names
**Verification**: Check if any integration tests use this field name

### Edge Case 2: Logging Format Changes
**Scenario**: Log analysis tools parse "north_star" from logs
**Action**: Document log format changes in CHANGELOG
**Verification**: Grep log messages before/after change

### Edge Case 3: Test Helper Rename Cascade
**Scenario**: Other test files import renamed helper functions
**Action**: Update all imports in test files
**Verification**: `cargo test --package context-graph-mcp` compiles and passes

---

## Synthetic Test Data

### Test Input: get_autonomous_status Call
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_autonomous_status",
    "arguments": {
      "include_metrics": true
    }
  },
  "id": 1
}
```

### Expected Output (After Changes)
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"status\":\"healthy\",\"strategic_goal_configured\":true,\"health_score\":0.85}"
    }]
  },
  "id": 1
}
```

Note: Field name should be `strategic_goal_configured` NOT `north_star_configured`.

---

## Definition of Done

### Mandatory Criteria
- [ ] All tool descriptions reference "Strategic goal" not "North Star"
- [ ] Variable/function names updated from `north_star` to `strategic` or `top_level`
- [ ] Deprecated test files deleted (`north_star.rs`, `north_star_alignment.rs`, `north_star_update.rs`)
- [ ] `cargo check --package context-graph-mcp` passes
- [ ] `cargo test --package context-graph-mcp` passes (all tests)
- [ ] Tool count remains 58

### Verification Commands (Must All Pass)
```bash
# 1. No "North Star" in tool definitions (user-facing)
test $(grep -rn "North Star" crates/context-graph-mcp/src/tools/definitions/ --include="*.rs" | grep -v "//" | wc -l) -eq 0

# 2. Deprecated test files deleted
test ! -f crates/context-graph-mcp/src/handlers/tests/north_star.rs
test ! -f crates/context-graph-mcp/src/handlers/tests/purpose/north_star_alignment.rs
test ! -f crates/context-graph-mcp/src/handlers/tests/purpose/north_star_update.rs

# 3. Tool registry count
cargo test --package context-graph-mcp test_register_all_tools_returns_58 --quiet

# 4. All MCP tests pass
cargo test --package context-graph-mcp --quiet
```

---

## Execution Checklist

### Phase 1: Documentation Updates
- [ ] Update tool descriptions in `autonomous.rs`
- [ ] Update tool descriptions in `gwt.rs` (if any mention North Star)
- [ ] Update comments in handler files

### Phase 2: Code Renaming
- [ ] Rename `north_star_status` → `strategic_status` in `status.rs`
- [ ] Rename `north_star_configured` → `strategic_goal_configured` in JSON responses
- [ ] Rename helper functions in `tests/mod.rs`
- [ ] Update all call sites of renamed functions

### Phase 3: Test Cleanup
- [ ] Delete `north_star.rs` test file
- [ ] Delete `north_star_alignment.rs` test file
- [ ] Delete `north_star_update.rs` test file
- [ ] Update `tests/mod.rs` to remove module declarations
- [ ] Update `tests/purpose/mod.rs` to remove module declarations

### Phase 4: Verification
- [ ] Run all pre-verification commands
- [ ] Run all post-verification commands
- [ ] Run edge case tests
- [ ] Manual API response verification

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| API Breaking Change | MEDIUM | Document field name changes, consider dual support |
| Test Failures | LOW | Rename helpers cascade to all usages |
| Missing References | LOW | Comprehensive grep verification |

---

## Dependencies

- **Depends On**: TASK-P0-001 (COMPLETED)
- **Blocks**: TASK-P0-003 (Constitution update)
- **Estimated Complexity**: LOW (naming/documentation, no logic changes)

---

## IMPORTANT NOTES FOR AI AGENT

1. **DO NOT DELETE FUNCTIONAL TOOLS** - The 5 tools listed above are KEPT, they work correctly
2. **FOCUS ON NAMING ONLY** - This is a documentation/naming cleanup task
3. **VERIFY TOOL COUNT REMAINS 58** - Never reduce below 58 tools
4. **TEST AFTER EACH CHANGE** - Run `cargo test --package context-graph-mcp` frequently
5. **BACKWARDS COMPATIBILITY** - Consider JSON field name changes carefully
6. **NO MOCK DATA IN TESTS** - Use real MCP infrastructure for verification
