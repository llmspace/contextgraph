# TASK-HOOKS-014: Create Shell Scripts for Claude Code Hooks

```xml
<task_spec id="TASK-HOOKS-014" version="2.0">
<metadata>
  <title>Create Shell Scripts for Claude Code Hooks</title>
  <status>complete</status>
  <layer>surface</layer>
  <sequence>14</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-02</requirement_ref>
    <requirement_ref>REQ-HOOKS-03</requirement_ref>
    <requirement_ref>REQ-HOOKS-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="complete">TASK-HOOKS-006</task_ref>
    <task_ref status="complete">TASK-HOOKS-007</task_ref>
    <task_ref status="complete">TASK-HOOKS-008</task_ref>
    <task_ref status="complete">TASK-HOOKS-009</task_ref>
    <task_ref status="complete">TASK-HOOKS-010</task_ref>
    <task_ref status="complete">TASK-HOOKS-011</task_ref>
    <task_ref status="complete">TASK-HOOKS-012</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>3.0</estimated_hours>
</metadata>

<critical_rules>
  <!-- NO BACKWARDS COMPATIBILITY - FAIL FAST -->
  <rule id="CR-01">FAIL IMMEDIATELY on any error - no fallbacks, no graceful degradation</rule>
  <rule id="CR-02">All errors MUST be logged to stderr with structured JSON</rule>
  <rule id="CR-03">Exit codes MUST match AP-26: 0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input</rule>
  <rule id="CR-04">NO MOCK DATA in tests - use real CLI binary with real database</rule>
  <rule id="CR-05">Scripts MUST call context-graph-cli hooks subcommands - NOT non-existent commands</rule>
</critical_rules>

<current_state>
  <!-- VERIFIED 2026-01-15 via filesystem inspection -->
  <existing_scripts>
    <script path=".claude/hooks/session_start.sh" status="exists" executable="true"/>
    <script path=".claude/hooks/pre_tool_use.sh" status="exists" executable="true"/>
    <script path=".claude/hooks/session_end.sh" status="MISSING"/>
    <script path=".claude/hooks/post_tool_use.sh" status="MISSING"/>
    <script path=".claude/hooks/user_prompt_submit.sh" status="MISSING"/>
  </existing_scripts>

  <cli_implementation location="crates/context-graph-cli/src/commands/hooks/">
    <file>mod.rs</file> <!-- Hook dispatcher -->
    <file>args.rs</file> <!-- CLI arguments with clap -->
    <file>types.rs</file> <!-- HookInput, HookOutput, HookPayload types -->
    <file>error.rs</file> <!-- HookError, exit codes -->
    <file>session_start.rs</file> <!-- SessionStart handler -->
    <file>session_end.rs</file> <!-- SessionEnd handler -->
    <file>pre_tool_use.rs</file> <!-- PreToolUse handler -->
    <file>post_tool_use.rs</file> <!-- PostToolUse handler -->
    <file>user_prompt_submit.rs</file> <!-- UserPromptSubmit handler -->
  </cli_implementation>

  <verified_cli_commands>
    <!-- These are the ACTUAL commands that exist -->
    <command>context-graph-cli hooks session-start --stdin --format json</command>
    <command>context-graph-cli hooks session-end --stdin --format json</command>
    <command>context-graph-cli hooks pre-tool --session-id ID --stdin --fast-path true</command>
    <command>context-graph-cli hooks post-tool --stdin --format json</command>
    <command>context-graph-cli hooks prompt-submit --stdin --format json</command>
    <command>context-graph-cli hooks generate-config</command> <!-- NOT IMPLEMENTED - returns exit 1 -->
  </verified_cli_commands>

  <settings_json_current location=".claude/settings.json">
    <!-- Currently uses npx @claude-flow/cli, NOT context-graph-cli -->
    <discrepancy>Shell scripts use context-graph-cli but settings.json uses @claude-flow/cli</discrepancy>
  </settings_json_current>
</current_state>

<context>
Claude Code hooks are configured in .claude/settings.json and execute shell commands.
This task creates shell scripts that bridge Claude Code hooks to context-graph-cli.

IMPORTANT: The CLI commands are in the `hooks` subcommand namespace:
- `context-graph-cli hooks session-start` (NOT `consciousness brief`)
- `context-graph-cli hooks session-end` (NOT `identity snapshot`)
- `context-graph-cli hooks pre-tool` (NOT `consciousness inject`)
- `context-graph-cli hooks post-tool` (NOT `tool record`)
- `context-graph-cli hooks prompt-submit`

Constitution References:
- AP-50: Shell scripts call CLI (NOT library functions)
- AP-53: Native hooks only (via .claude/settings.json)
- AP-26: Exit codes (0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input)
- IDENTITY-002: IC thresholds (healthy >=0.9, warning <0.7, critical <0.5)
</context>

<input_context_files>
  <file purpose="constitution" path="docs2/constitution.yaml">
    Read sections: claude_code.hooks, forbidden (AP-50,51,52,53), perf
  </file>
  <file purpose="cli_args" path="crates/context-graph-cli/src/commands/hooks/args.rs">
    Contains HooksCommands enum with actual subcommand names
  </file>
  <file purpose="cli_types" path="crates/context-graph-cli/src/commands/hooks/types.rs">
    Contains HookInput, HookOutput, HookPayload, ConsciousnessState
  </file>
  <file purpose="existing_session_start" path=".claude/hooks/session_start.sh">
    Reference implementation - COPY THIS PATTERN
  </file>
  <file purpose="existing_pre_tool" path=".claude/hooks/pre_tool_use.sh">
    Reference implementation for FAST PATH (100ms timeout)
  </file>
</input_context_files>

<prerequisites>
  <check cmd="test -x ./target/release/context-graph-cli || test -x ./target/debug/context-graph-cli">
    CLI binary must be built
  </check>
  <check cmd="test -d .claude/hooks">
    .claude/hooks/ directory must exist
  </check>
  <check cmd="./target/release/context-graph-cli hooks --help 2>/dev/null | grep -q session-start">
    CLI hooks subcommand must work
  </check>
</prerequisites>

<scope>
  <in_scope>
    - Create .claude/hooks/session_end.sh (MISSING)
    - Create .claude/hooks/post_tool_use.sh (MISSING)
    - Create .claude/hooks/user_prompt_submit.sh (MISSING)
    - Verify existing session_start.sh and pre_tool_use.sh work correctly
    - All scripts MUST call actual CLI commands (hooks subcommand)
    - All scripts MUST handle timeouts per constitution
    - All scripts MUST output JSON to stdout, errors to stderr
  </in_scope>
  <out_of_scope>
    - .claude/settings.json configuration (TASK-HOOKS-015)
    - Windows batch files
    - Non-existent CLI commands (consciousness brief, identity restore, etc.)
  </out_of_scope>
</scope>

<timeout_budgets>
  <!-- FROM constitution.yaml claude_code.performance.hooks -->
  <hook name="SessionStart" timeout_ms="5000" async="false"/>
  <hook name="PreToolUse" timeout_ms="100" async="false" fast_path="true">
    CRITICAL: No database access allowed - must complete in 100ms
  </hook>
  <hook name="PostToolUse" timeout_ms="3000" async="true"/>
  <hook name="UserPromptSubmit" timeout_ms="2000" async="false"/>
  <hook name="SessionEnd" timeout_ms="30000" async="false"/>
</timeout_budgets>

<definition_of_done>
  <scripts>
    <script path=".claude/hooks/session_start.sh" exists="true">
      #!/bin/bash
      # VERIFIED EXISTING - calls: context-graph-cli hooks session-start --stdin
      # Timeout: 5000ms
      # Exit codes: 0=success, 2=timeout, 3=db_error, 4=invalid_input
    </script>

    <script path=".claude/hooks/session_end.sh" exists="false">
      #!/bin/bash
      # TO CREATE - calls: context-graph-cli hooks session-end --stdin
      # Timeout: 30000ms
      # Input: { "session_id": "...", "stats": {...}, "reason": "..." }
      # Output: JSON with snapshot_id, IC metrics
    </script>

    <script path=".claude/hooks/pre_tool_use.sh" exists="true">
      #!/bin/bash
      # VERIFIED EXISTING - calls: context-graph-cli hooks pre-tool --stdin --fast-path true
      # Timeout: 100ms (FAST PATH - NO DB ACCESS)
      # Exit codes: 0=success, 4=invalid_input
    </script>

    <script path=".claude/hooks/post_tool_use.sh" exists="false">
      #!/bin/bash
      # TO CREATE - calls: context-graph-cli hooks post-tool --stdin
      # Timeout: 3000ms (async OK)
      # Input: { "tool_name": "...", "tool_result": {...}, "success": bool }
      # Output: JSON with IC update, trajectory recorded
    </script>

    <script path=".claude/hooks/user_prompt_submit.sh" exists="false">
      #!/bin/bash
      # TO CREATE - calls: context-graph-cli hooks prompt-submit --stdin
      # Timeout: 2000ms
      # Input: { "prompt": "...", "session_id": "..." }
      # Output: JSON with context injection, Johari guidance
    </script>
  </scripts>

  <constraints>
    <constraint id="C-01">Scripts MUST use set -euo pipefail</constraint>
    <constraint id="C-02">Scripts MUST output valid JSON to stdout on success</constraint>
    <constraint id="C-03">Scripts MUST output error JSON to stderr on failure</constraint>
    <constraint id="C-04">Scripts MUST complete within their timeout budget</constraint>
    <constraint id="C-05">Scripts MUST NOT use fallback/graceful degradation</constraint>
    <constraint id="C-06">Scripts MUST find CLI via CONTEXT_GRAPH_CLI env or known paths</constraint>
    <constraint id="C-07">PreToolUse MUST NOT access database (100ms fast path)</constraint>
    <constraint id="C-08">All scripts MUST be chmod +x</constraint>
  </constraints>
</definition_of_done>

<implementation_templates>
  <!-- CORRECT templates based on actual CLI commands -->

  <template name="session_end.sh">
#!/bin/bash
# Claude Code Hook: SessionEnd
# Timeout: 30000ms (30 seconds for full persistence)
#
# Constitution: AP-50, AP-26
# Exit Codes: 0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input

set -euo pipefail

INPUT=$(cat)
if [ -z "$INPUT" ]; then
    echo '{"success":false,"error":"Empty stdin","exit_code":4}' >&2
    exit 4
fi

CONTEXT_GRAPH_CLI="${CONTEXT_GRAPH_CLI:-context-graph-cli}"
if ! command -v "$CONTEXT_GRAPH_CLI" &amp;>/dev/null; then
    for candidate in \
        "./target/release/context-graph-cli" \
        "./target/debug/context-graph-cli" \
        "$HOME/.cargo/bin/context-graph-cli" \
    ; do
        if [ -x "$candidate" ]; then
            CONTEXT_GRAPH_CLI="$candidate"
            break
        fi
    done
fi

# Execute CLI - NO TIMEOUT WRAPPER (30s is already long)
echo "$INPUT" | "$CONTEXT_GRAPH_CLI" hooks session-end --stdin --format json
  </template>

  <template name="post_tool_use.sh">
#!/bin/bash
# Claude Code Hook: PostToolUse
# Timeout: 3000ms (async allowed)
#
# Constitution: AP-50, AP-26
# Exit Codes: 0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input

set -euo pipefail

INPUT=$(cat)
if [ -z "$INPUT" ]; then
    echo '{"success":false,"error":"Empty stdin","exit_code":4}' >&2
    exit 4
fi

CONTEXT_GRAPH_CLI="${CONTEXT_GRAPH_CLI:-context-graph-cli}"
if ! command -v "$CONTEXT_GRAPH_CLI" &amp;>/dev/null; then
    for candidate in \
        "./target/release/context-graph-cli" \
        "./target/debug/context-graph-cli" \
        "$HOME/.cargo/bin/context-graph-cli" \
    ; do
        if [ -x "$candidate" ]; then
            CONTEXT_GRAPH_CLI="$candidate"
            break
        fi
    done
fi

# Execute CLI with 3s timeout
echo "$INPUT" | timeout 3s "$CONTEXT_GRAPH_CLI" hooks post-tool --stdin --format json
exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo '{"success":false,"error":"Timeout after 3000ms","exit_code":2}' >&2
    exit 2
fi
exit $exit_code
  </template>

  <template name="user_prompt_submit.sh">
#!/bin/bash
# Claude Code Hook: UserPromptSubmit
# Timeout: 2000ms
#
# Constitution: AP-50, AP-26
# Exit Codes: 0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input

set -euo pipefail

INPUT=$(cat)
if [ -z "$INPUT" ]; then
    echo '{"success":false,"error":"Empty stdin","exit_code":4}' >&2
    exit 4
fi

CONTEXT_GRAPH_CLI="${CONTEXT_GRAPH_CLI:-context-graph-cli}"
if ! command -v "$CONTEXT_GRAPH_CLI" &amp;>/dev/null; then
    for candidate in \
        "./target/release/context-graph-cli" \
        "./target/debug/context-graph-cli" \
        "$HOME/.cargo/bin/context-graph-cli" \
    ; do
        if [ -x "$candidate" ]; then
            CONTEXT_GRAPH_CLI="$candidate"
            break
        fi
    done
fi

# Execute CLI with 2s timeout
echo "$INPUT" | timeout 2s "$CONTEXT_GRAPH_CLI" hooks prompt-submit --stdin --format json
exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo '{"success":false,"error":"Timeout after 2000ms","exit_code":2}' >&2
    exit 2
fi
exit $exit_code
  </template>
</implementation_templates>

<verification>
  <source_of_truth>
    <!-- The SINGLE source of truth for each component -->
    <truth component="CLI commands">crates/context-graph-cli/src/commands/hooks/args.rs:292 (HooksCommands enum)</truth>
    <truth component="Hook types">crates/context-graph-cli/src/commands/hooks/types.rs</truth>
    <truth component="Exit codes">crates/context-graph-cli/src/commands/hooks/error.rs (HookError.exit_code())</truth>
    <truth component="Timeouts">docs2/constitution.yaml:claude_code.performance.hooks</truth>
    <truth component="Anti-patterns">docs2/constitution.yaml:forbidden (AP-50 through AP-53)</truth>
  </source_of_truth>

  <full_state_verification>
    <step id="FSV-01" name="Build CLI">
      <command>cargo build --release -p context-graph-cli</command>
      <success_criteria>Exit code 0, binary at ./target/release/context-graph-cli</success_criteria>
    </step>

    <step id="FSV-02" name="Verify CLI hooks subcommand">
      <command>./target/release/context-graph-cli hooks --help</command>
      <success_criteria>Shows session-start, session-end, pre-tool, post-tool, prompt-submit</success_criteria>
    </step>

    <step id="FSV-03" name="Test session_start.sh with real input">
      <command>echo '{"session_id":"test-verify-001","timestamp":"2026-01-15T00:00:00Z"}' | .claude/hooks/session_start.sh</command>
      <success_criteria>Valid JSON output with "success":true</success_criteria>
    </step>

    <step id="FSV-04" name="Test pre_tool_use.sh FAST PATH">
      <command>time (echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/test"}}' | timeout 0.15s .claude/hooks/pre_tool_use.sh)</command>
      <success_criteria>Completes in &lt;100ms, valid JSON output</success_criteria>
    </step>

    <step id="FSV-05" name="Test session_end.sh with real input">
      <command>echo '{"session_id":"test-verify-001","reason":"normal","stats":{"tool_calls":5}}' | .claude/hooks/session_end.sh</command>
      <success_criteria>Valid JSON output with snapshot_id</success_criteria>
    </step>

    <step id="FSV-06" name="Test post_tool_use.sh">
      <command>echo '{"tool_name":"Read","tool_result":"content","success":true}' | .claude/hooks/post_tool_use.sh</command>
      <success_criteria>Valid JSON output, IC metrics present</success_criteria>
    </step>

    <step id="FSV-07" name="Test user_prompt_submit.sh">
      <command>echo '{"prompt":"What is X?","session_id":"test-verify-001"}' | .claude/hooks/user_prompt_submit.sh</command>
      <success_criteria>Valid JSON with context injection</success_criteria>
    </step>
  </full_state_verification>

  <boundary_edge_cases>
    <case id="EC-01" name="Empty stdin">
      <test>echo -n '' | .claude/hooks/session_start.sh</test>
      <expected>Exit code 4, error JSON to stderr</expected>
    </case>

    <case id="EC-02" name="Invalid JSON">
      <test>echo 'not-json' | .claude/hooks/session_start.sh</test>
      <expected>Exit code 4, parse error to stderr</expected>
    </case>

    <case id="EC-03" name="Missing session_id">
      <test>echo '{"timestamp":"2026-01-15T00:00:00Z"}' | .claude/hooks/session_start.sh</test>
      <expected>Exit code 4 OR generates session_id internally</expected>
    </case>

    <case id="EC-04" name="PreToolUse exceeds 100ms">
      <test>SLOW_MODE=1 timeout 0.15s .claude/hooks/pre_tool_use.sh</test>
      <expected>Exit code 124 (timeout), error JSON to stderr</expected>
    </case>

    <case id="EC-05" name="CLI binary not found">
      <test>CONTEXT_GRAPH_CLI=/nonexistent .claude/hooks/session_start.sh</test>
      <expected>Exit code 1, error about missing binary</expected>
    </case>
  </boundary_edge_cases>

  <evidence_of_success>
    <!-- Log outputs that MUST be verified manually -->
    <evidence id="EV-01">
      All 5 scripts exist in .claude/hooks/ with chmod +x
      Command: ls -la .claude/hooks/*.sh
    </evidence>

    <evidence id="EV-02">
      session_start.sh outputs valid JSON with IC metrics
      Log: {"success":true,"ic":{"value":X,"status":"healthy|warning|critical"}}
    </evidence>

    <evidence id="EV-03">
      pre_tool_use.sh completes under 100ms (FAST PATH verified)
      Log: real 0m0.0XXs (where XX &lt; 10)
    </evidence>

    <evidence id="EV-04">
      Error cases produce structured JSON to stderr
      Log: {"success":false,"error":"...","exit_code":N}
    </evidence>
  </evidence_of_success>
</verification>

<files_to_create>
  <file path=".claude/hooks/session_end.sh" executable="true">
    Session end hook - calls context-graph-cli hooks session-end
    Timeout: 30000ms
  </file>
  <file path=".claude/hooks/post_tool_use.sh" executable="true">
    Post-tool hook - calls context-graph-cli hooks post-tool
    Timeout: 3000ms
  </file>
  <file path=".claude/hooks/user_prompt_submit.sh" executable="true">
    User prompt hook - calls context-graph-cli hooks prompt-submit
    Timeout: 2000ms
  </file>
</files_to_create>

<files_to_verify>
  <file path=".claude/hooks/session_start.sh">
    Verify: Calls correct CLI command, handles errors properly
  </file>
  <file path=".claude/hooks/pre_tool_use.sh">
    Verify: FAST PATH (100ms), no database access
  </file>
</files_to_verify>

<anti_patterns_to_avoid>
  <!-- THESE WILL CAUSE IMMEDIATE FAILURE -->
  <anti_pattern id="BAD-01">
    Using non-existent commands like `context-graph-cli consciousness brief`
    CORRECT: `context-graph-cli hooks session-start`
  </anti_pattern>

  <anti_pattern id="BAD-02">
    Adding fallback logic like `|| true` or `2>/dev/null || echo default`
    CORRECT: Fail immediately with proper exit code
  </anti_pattern>

  <anti_pattern id="BAD-03">
    Using mock data in tests like `echo '{"mock":true}'`
    CORRECT: Use real CLI with real database operations
  </anti_pattern>

  <anti_pattern id="BAD-04">
    Exceeding timeout budget (especially 100ms for PreToolUse)
    CORRECT: Use `timeout` command and verify timing
  </anti_pattern>

  <anti_pattern id="BAD-05">
    Outputting non-JSON or mixing stdout/stderr incorrectly
    CORRECT: JSON to stdout on success, error JSON to stderr on failure
  </anti_pattern>
</anti_patterns_to_avoid>

<test_commands>
  <!-- NO MOCK DATA - Real CLI execution required -->
  <command desc="Build CLI first">cargo build --release -p context-graph-cli</command>
  <command desc="Make all scripts executable">chmod +x .claude/hooks/*.sh</command>
  <command desc="Test session_start with real data">
    echo '{"session_id":"integration-test-001","timestamp":"'$(date -Iseconds)'"}' | .claude/hooks/session_start.sh
  </command>
  <command desc="Test pre_tool FAST PATH timing">
    time (echo '{"tool_name":"Read","tool_input":{"file_path":"./README.md"}}' | .claude/hooks/pre_tool_use.sh)
  </command>
  <command desc="Test session_end">
    echo '{"session_id":"integration-test-001","reason":"test_complete","stats":{"tool_calls":3,"duration_ms":5000}}' | .claude/hooks/session_end.sh
  </command>
  <command desc="Verify exit codes">
    echo '' | .claude/hooks/session_start.sh; echo "Exit: $?"
  </command>
</test_commands>
</task_spec>
```
