# TASK-HOOKS-015: Configure .claude/settings.json Hook Registrations

```xml
<task_spec id="TASK-HOOKS-015" version="2.0">
<metadata>
  <title>Configure .claude/settings.json Hook Registrations</title>
  <status>complete</status>
  <layer>surface</layer>
  <sequence>15</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-05</requirement_ref>
    <requirement_ref>REQ-HOOKS-06</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-014</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
Claude Code discovers hooks through .claude/settings.json configuration.
This task creates the settings file that registers all Context Graph hooks
with appropriate timeouts and matchers.

Per constitution AP-50: Only native Claude Code hooks via settings.json.
No custom hook infrastructure.

## CRITICAL: NO BACKWARDS COMPATIBILITY
The hook system MUST fail fast with explicit error messages. Unknown hook types,
malformed JSON, or missing required fields MUST result in immediate failure with
exit code 4 (ERR_INVALID_INPUT). No silent fallbacks, no default values for
required fields, no graceful degradation.
</context>

<current_state>
## Existing Files (VERIFIED 2026-01-15)

### Shell Scripts (.claude/hooks/)
All 5 shell scripts exist and are executable:
- session_start.sh (5000ms timeout)
- session_end.sh (30000ms timeout, no wrapper)
- pre_tool_use.sh (100ms FAST PATH timeout)
- post_tool_use.sh (3000ms timeout)
- user_prompt_submit.sh (2000ms timeout)

### CLI Implementation
Binary: ./target/release/context-graph-cli
Commands: hooks session-start, pre-tool, post-tool, prompt-submit, session-end, generate-config

### Current settings.json (DISCREPANCY)
The current .claude/settings.json uses claude-flow/cli npm package, NOT the local CLI.
This task must update it to use the local context-graph-cli binary via shell scripts.

### Type System (crates/context-graph-cli/src/commands/hooks/types.rs)
- HookEventType: SessionStart, PreToolUse, PostToolUse, UserPromptSubmit, SessionEnd
- HookInput: hook_type, session_id, timestamp_ms, payload (ALL REQUIRED)
- HookOutput: success, error?, consciousness_state?, ic_classification?, context_injection?, drift_metrics?, execution_time_ms
- HookPayload: Typed variants for each hook type (internally tagged enum)
- ICLevel: Healthy (>=0.9), Normal (>=0.7), Warning (>=0.5), Critical (<0.5)
- SessionEndStatus: Normal, Timeout, Error, UserAbort, Clear
</current_state>

<input_context_files>
  <file purpose="hooks_reference">docs2/claudehooks.md</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md</file>
  <file purpose="shell_scripts">.claude/hooks/</file>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="cli_args">crates/context-graph-cli/src/commands/hooks/args.rs</file>
  <file purpose="cli_handlers">crates/context-graph-cli/src/commands/hooks/mod.rs</file>
  <file purpose="constitution">docs/specs/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check>Shell scripts exist in .claude/hooks/ (VERIFIED)</check>
  <check>Scripts are executable (chmod +x)</check>
  <check>CLI binary exists at ./target/release/context-graph-cli</check>
  <check>Type definitions match between shell scripts and CLI</check>
</prerequisites>

<scope>
  <in_scope>
    - Update .claude/settings.json to use shell script hooks
    - Configure PreToolUse hook with tool matchers (100ms fast path)
    - Configure PostToolUse hook (3000ms)
    - Configure SessionStart hook (5000ms)
    - Configure SessionEnd hook (30000ms)
    - Configure UserPromptSubmit hook (2000ms)
    - Set timeouts per constitution requirements
    - Ensure HookInput JSON format compliance
  </in_scope>
  <out_of_scope>
    - Hook script implementation (TASK-HOOKS-014 - completed)
    - Custom hook types not in Claude Code spec
    - MCP server configuration (separate task)
    - claude-flow npm package hooks (REMOVE - use local CLI only)
  </out_of_scope>
</scope>

<hook_input_contract>
## HookInput JSON Format (REQUIRED FIELDS)
All fields are REQUIRED. Missing fields MUST cause exit code 4.

```json
{
  "hook_type": "session_start|pre_tool_use|post_tool_use|user_prompt_submit|session_end",
  "session_id": "string (non-empty)",
  "timestamp_ms": 1705312345678,
  "payload": { ... }
}
```

## Payload Variants (internally tagged with "type" field)

### SessionStart Payload
```json
{
  "type": "session_start",
  "cwd": "/home/user/project",
  "source": "cli|ide|resume",
  "previous_session_id": "optional-string"
}
```

### PreToolUse Payload (FAST PATH - 100ms max)
```json
{
  "type": "pre_tool_use",
  "tool_name": "Write|Edit|Bash|etc",
  "tool_input": {},
  "tool_use_id": "unique-id"
}
```

### PostToolUse Payload
```json
{
  "type": "post_tool_use",
  "tool_name": "Write",
  "tool_input": {},
  "tool_response": "string result",
  "tool_use_id": "unique-id"
}
```

### UserPromptSubmit Payload
```json
{
  "type": "user_prompt_submit",
  "prompt": "user's input text",
  "context": []
}
```

### SessionEnd Payload
```json
{
  "type": "session_end",
  "duration_ms": 3600000,
  "status": "normal|timeout|error|user_abort|clear",
  "reason": "optional-string"
}
```
</hook_input_contract>

<hook_output_contract>
## HookOutput JSON Format

```json
{
  "success": true,
  "error": "optional error message",
  "consciousness_state": {
    "consciousness": 0.85,
    "integration": 0.80,
    "reflection": 0.75,
    "differentiation": 0.82,
    "identity_continuity": 0.88
  },
  "ic_classification": {
    "level": "healthy|normal|warning|critical",
    "score": 0.88,
    "quadrant": "open|blind|hidden|unknown",
    "timestamp": "2026-01-15T00:00:00Z"
  },
  "context_injection": "optional string to inject into context",
  "drift_metrics": {
    "session_id": "current-session",
    "previous_session_id": "previous-session",
    "identity_distance": 0.15,
    "restoration_confidence": 0.92
  },
  "execution_time_ms": 42
}
```
</hook_output_contract>

<exit_codes>
## Exit Code Specification (AP-26)
- 0: Success
- 1: General error (catch-all)
- 2: Timeout exceeded
- 3: Database error (connection, query failure)
- 4: Invalid input (malformed JSON, missing required fields)

## FAIL FAST REQUIREMENTS
- Missing hook_type: exit 4 immediately
- Empty session_id: exit 4 immediately
- timestamp_ms <= 0: exit 4 immediately
- Unknown hook_type value: exit 4 immediately
- Malformed JSON: exit 4 immediately
- NO SILENT FAILURES - all errors must be logged to stderr
</exit_codes>

<timeout_budgets>
## Timeout Requirements (Constitution + TECH-HOOKS.md)
| Hook | Timeout | Notes |
|------|---------|-------|
| PreToolUse | 100ms | FAST PATH - no DB access |
| PostToolUse | 3000ms | Async learning allowed |
| UserPromptSubmit | 2000ms | Context injection |
| SessionStart | 5000ms | Identity restoration |
| SessionEnd | 30000ms | Full persistence |
</timeout_budgets>

<definition_of_done>
  <signatures>
    <signature file=".claude/settings.json">
      {
        "hooks": {
          "SessionStart": [
            {
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/hooks/session_start.sh",
                  "timeout": 5000
                }
              ]
            }
          ],
          "SessionEnd": [
            {
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/hooks/session_end.sh",
                  "timeout": 30000
                }
              ]
            }
          ],
          "PreToolUse": [
            {
              "matcher": ".*",
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/hooks/pre_tool_use.sh",
                  "timeout": 100
                }
              ]
            }
          ],
          "PostToolUse": [
            {
              "matcher": ".*",
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/hooks/post_tool_use.sh",
                  "timeout": 3000
                }
              ]
            }
          ],
          "UserPromptSubmit": [
            {
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/hooks/user_prompt_submit.sh",
                  "timeout": 2000
                }
              ]
            }
          ]
        }
      }
    </signature>
  </signatures>

  <constraints>
    - File must be valid JSON (parseable by jq)
    - Hook commands must use relative paths from project root
    - Timeouts must match constitution requirements EXACTLY
    - MUST NOT include unsupported hook types
    - NO absolute paths (portability)
    - NO backwards compatibility fallbacks
    - ALL errors must produce structured JSON on stderr
  </constraints>

  <verification>
    - jq . .claude/settings.json (valid JSON)
    - Claude Code recognizes hooks on startup
    - Hook commands execute when triggered
    - Exit codes match specification
    - Timeout enforcement works
  </verification>
</definition_of_done>

<full_state_verification>
## Source of Truth
- Constitution Reference: AP-50, AP-26, IDENTITY-002, GWT-003
- Type Definitions: crates/context-graph-cli/src/commands/hooks/types.rs
- Shell Scripts: .claude/hooks/*.sh

## Execute & Inspect Pattern
1. Run each hook with valid input, verify output matches HookOutput contract
2. Run each hook with invalid input, verify exit code 4 and stderr message
3. Run each hook past timeout, verify exit code 2

## Boundary & Edge Case Audit (3 minimum)

### Edge Case 1: Empty session_id
INPUT:
```json
{"hook_type":"session_start","session_id":"","timestamp_ms":1705312345678,"payload":{"type":"session_start","cwd":"/tmp","source":"cli"}}
```
EXPECTED: Exit code 4, stderr contains "session_id cannot be empty"
BEFORE STATE: No session exists
AFTER STATE: No session created, error logged

### Edge Case 2: PreToolUse exceeds 100ms timeout
INPUT: Valid PreToolUse with artificially slow handler
EXPECTED: Exit code 2, stderr contains "Timeout after 100ms"
BEFORE STATE: Tool pending
AFTER STATE: Tool allowed to proceed (timeout is advisory, not blocking)

### Edge Case 3: Unknown hook_type
INPUT:
```json
{"hook_type":"unknown_hook","session_id":"test-123","timestamp_ms":1705312345678,"payload":{}}
```
EXPECTED: Exit code 4, stderr contains deserialization error
BEFORE STATE: N/A
AFTER STATE: Hook rejected, no state change

## Evidence of Success Logging
Each hook execution MUST log to structured format:
```json
{"event":"hook_executed","hook_type":"session_start","session_id":"xxx","execution_time_ms":42,"success":true}
```
</full_state_verification>

<manual_verification>
## Database Verification
After SessionStart with previous_session_id:
```bash
sqlite3 ./data/context-graph.db "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1;"
```
Verify: session record exists, previous_session_id linked correctly

## Output Verification
After PostToolUse:
```bash
cat ./logs/hooks.log | jq 'select(.hook_type=="post_tool_use")'
```
Verify: tool_name, tool_use_id, success status recorded

## Graph Verification
After UserPromptSubmit:
```bash
sqlite3 ./data/context-graph.db "SELECT COUNT(*) FROM graph_nodes WHERE session_id='current-session';"
```
Verify: nodes created for prompt context
</manual_verification>

<test_commands>
  <command desc="Validate JSON syntax">jq . .claude/settings.json</command>
  <command desc="Test SessionStart with valid input">
    echo '{"hook_type":"session_start","session_id":"test-001","timestamp_ms":1705312345678,"payload":{"type":"session_start","cwd":"/tmp","source":"cli"}}' | .claude/hooks/session_start.sh
  </command>
  <command desc="Test PreToolUse fast path">
    echo '{"hook_type":"pre_tool_use","session_id":"test-001","timestamp_ms":1705312345678,"payload":{"type":"pre_tool_use","tool_name":"Write","tool_input":{},"tool_use_id":"tool-001"}}' | timeout 0.2s .claude/hooks/pre_tool_use.sh
  </command>
  <command desc="Test invalid input rejection">
    echo '{"session_id":"test-001"}' | .claude/hooks/session_start.sh; echo "Exit code: $?"
  </command>
  <command desc="Verify exit code 4 on missing hook_type">
    echo '{"session_id":"test","timestamp_ms":123,"payload":{}}' | .claude/hooks/session_start.sh 2>&1; echo "Exit: $?"
  </command>
</test_commands>

<implementation_notes>
## Key Changes from v1.0
1. REMOVED: claude-flow npm package references
2. ADDED: Local CLI binary via shell scripts
3. UPDATED: Timeouts to match constitution
4. ADDED: Full HookInput/HookOutput contracts
5. ADDED: Fail-fast requirements
6. ADDED: Edge case verification

## Shell Script Contract
Each shell script:
1. Reads JSON from stdin
2. Validates input exists (exit 4 if empty)
3. Extracts session_id from env or JSON
4. Calls CLI with appropriate subcommand
5. Returns CLI exit code

## CLI Binary Location
Primary: ./target/release/context-graph-cli
Fallback search order:
1. ./target/release/context-graph-cli
2. ./target/debug/context-graph-cli
3. $HOME/.cargo/bin/context-graph-cli
</implementation_notes>

<files_to_modify>
  <file path=".claude/settings.json">
    Replace claude-flow npm hooks with shell script hooks
    Update timeouts to constitution requirements
    Remove Stop/Notification hooks (not in Claude Code spec)
  </file>
</files_to_modify>

<no_mock_data>
## Real Data Requirements
- session_id: Use actual CLAUDE_SESSION_ID from environment
- timestamp_ms: Use actual Unix timestamp (Date.now())
- tool_name: Use actual tool names from Claude Code
- tool_input: Use actual tool parameters
- NO synthetic/mock data in production or tests
- Tests MUST use real CLI binary, not mocks
</no_mock_data>
</task_spec>
```
