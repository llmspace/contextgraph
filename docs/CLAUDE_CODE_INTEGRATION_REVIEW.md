# Context Graph + Claude Code Integration Review

**Date**: 2026-02-16
**Branch**: ccintegration
**Scope**: Hooks, Skills, Memory System, Context Injection Pipeline

---

## 1. Architecture Overview

### Data Flow

```
User Prompt → Claude Code
                ↓
        Hook Event Dispatch
                ↓
  .claude/hooks/<event>.sh  (shell wrapper)
                ↓
  context-graph-cli hooks <subcommand>  (Rust binary)
                ↓
  TCP JSON-RPC → MCP Server :3100  (warm GPU models)
                ↓
  RocksDB (51 CFs) + 13 Embedders + HNSW Indexes
                ↓
  Context Injection String → Claude Code system-reminder
```

### Hook Lifecycle (per session)

| Phase | Hook | Timeout | DB Access | Purpose |
|-------|------|---------|-----------|---------|
| 1 | `SessionStart` | 5s | No | Initialize session, link to previous, compute drift metrics |
| 2 | `UserPromptSubmit` | 2s (3s shell) | Yes (MCP) | E1 semantic search + E11 entity extraction + E4 sequence + divergence alerts → context injection |
| 3 | `PreToolUse` | 500ms (1s shell) | No | Return cached memories + tool-specific guidance |
| 4 | `PostToolUse` | 3s (5s shell) | Yes (MCP) | Capture tool responses as HookDescription memories, update coherence |
| 5 | `PostToolUseFailure` | 20s | Yes | Log failures for learning |
| 6 | `SubagentStart` | 3s | No | Track subagent spawning |
| 7 | `SubagentStop` | 20s | Yes | Capture subagent results |
| 8 | `PreCompact` | 20s | Yes | Save state before context window compression |
| 9 | `TaskCompleted` | 20s | Yes | Capture task results |
| 10 | `Stop` | 3s | No | Graceful shutdown |
| 11 | `Notification` | 3s | No | Log notifications |
| 12 | `SessionEnd` | 30s | No | Persist state, clean up filesystem cache |

### Memory Cache Architecture

```
user_prompt_submit (2s budget)
    → MCP search_graph (E1 semantic, 5 results, minSimilarity=0.3)
    → MCP extract_entities (E11 KEPLER)
    → MCP search_by_entities (E11, 3 results)
    → MCP get_conversation_context (E4, 5 turns)
    → MCP get_divergence_alerts (2-hour lookback)
    → Write results to /tmp/cg-memory-cache/{session_id}.json
    → Generate context injection markdown

pre_tool_use (500ms budget, 100ms CLI)
    → Read /tmp/cg-memory-cache/{session_id}.json (no MCP)
    → Filter by tool relevance + similarity >= 0.5
    → Return top 3 cached memories + tool guidance

session_end (30s budget)
    → Delete /tmp/cg-memory-cache/{session_id}.json
```

### Skills Coverage

46 skill directories under `.claude/skills/`, with 7 Context Graph-specific skills:
- `semantic-search` / `cg-memory-search` — E1 multi-space search
- `memory-inject` / `context-inject` — Store and retrieve memories
- `cg-code-search` — E7 code pattern search
- `cg-causal` — E5 cause-effect exploration
- `cg-entity` — E11 KEPLER entity search
- `cg-session` / `session-context` — E4 timeline navigation
- `cg-provenance` — Audit trails and merge history
- `cg-blind-spot` — E9 HDC typo-robust search, embedder disagreement
- `cg-topics` — Topic clusters and divergence alerts
- `cg-curator` — Memory lifecycle management
- `cg-graph` — Graph traversal and neighbor discovery

---

## 2. What Works Well

### 2.1 Multi-Embedder Context Injection Pipeline
The `user_prompt_submit` handler uses 4 complementary search strategies in a single 2-second budget:
- **E1 semantic search** finds topically similar memories
- **E11 entity extraction** catches what E1 misses (e.g., "Diesel" → knows it's an ORM)
- **E4 sequential context** provides conversation continuity
- **Divergence alerts** flag potential contradictions

This is the system's strongest feature — no single embedder can match this coverage.

### 2.2 Filesystem Cache for Cross-Process Communication
The filesystem-based cache (`memory_cache.rs`) correctly solves the fundamental architectural constraint that each hook invocation is a separate process (AP-50). The atomic write-then-rename pattern prevents corruption, and the 300s TTL + session cleanup prevent stale data accumulation.

### 2.3 Fail-Fast Error Philosophy
Every handler logs errors prominently but never blocks the hook pipeline. MCP unavailability, entity extraction failure, and divergence fetch timeout all degrade gracefully with specific tracing spans for debugging.

### 2.4 Tool-Specific Guidance in PreToolUse
The `get_tool_guidance()` function provides contextual hints for each tool type (Read, Write, Bash, etc.) without any I/O, staying well within the 100ms CLI budget.

### 2.5 Comprehensive Settings Configuration
All 12 hook events are wired in `settings.json` with appropriate timeouts and matchers. The `PreToolUse` matcher (`Edit|Write|Read|Bash|Task`) correctly targets the most impactful tools.

### 2.6 Identity Marker Detection
The 6-category identity marker system (SelfReference, Goal, Value, Capability, Challenge, Confirmation) with priority ordering (Challenge > SelfReference > others) provides behavioral guidance that adapts to user intent.

---

## 3. Improvement Recommendations

### 3.1 HIGH PRIORITY — Context Injection Quality

#### R1: PostToolUse is not capturing Read tool responses to cache
**Current**: `post_tool_use.sh` matcher is `Write|Edit|Bash|Task` — it does NOT fire for `Read` tool.
**Impact**: The `TOOLS_TO_CAPTURE` list in `post_tool_use.rs` includes `Read`, but the shell matcher never triggers PostToolUse for Read operations. This means file content Claude reads is never captured as memory.
**Fix**: Add `Read` to the PostToolUse matcher in `settings.json`:
```json
"matcher": "Write|Edit|Bash|Task|Read"
```
**Risk**: Increases hook invocations. Read is the most frequent tool — could add latency. Consider adding a response-length filter in the shell script to skip trivial reads.

#### R2: PreToolUse matcher should include Grep and Glob
**Current**: PreToolUse fires for `Edit|Write|Read|Bash|Task` but not `Grep` or `Glob`.
**Impact**: Search operations (Grep/Glob) don't receive cached memory context, missing opportunities to guide search strategy with prior knowledge.
**Fix**: Add `Grep|Glob` to the PreToolUse matcher:
```json
"matcher": "Edit|Write|Read|Bash|Task|Grep|Glob"
```
**Risk**: Low. PreToolUse is fast path (<100ms CLI logic), no DB access. Adding more tools just means more cache reads.

#### R3: Context injection content is too verbose for Claude Code
**Current**: The `generate_context_injection()` output includes section headers like `## Coherence State`, `## Identity Marker Detected`, `## Relevant Memories`, each with sub-headers and metadata.
**Impact**: Claude Code has limited `system-reminder` budget. Verbose context injection competes with other system instructions.
**Recommendation**:
- Remove the Coherence State section entirely (integration/reflection/differentiation metrics are internal bookkeeping, not useful for task execution)
- Remove Identity Marker guidance (Claude already handles conversational tone natively)
- Keep only: Retrieved Memories, E11 Entity Discoveries, Divergence Alerts (actionable information)
- Truncate memory content to 300 chars (currently 500 for E1, 400 for E11)
- Use bullet points instead of markdown headers to reduce overhead

**Estimated reduction**: ~40-60% of injected context is non-actionable metadata.

#### R4: Memory similarity threshold too low
**Current**: `MIN_MEMORY_SIMILARITY = 0.3` for E1 search results.
**Impact**: At 0.3, many marginally relevant memories get injected, diluting the signal. The pre_tool_use cache then filters at 0.5, but the user_prompt_submit injection has already included them.
**Fix**: Raise to 0.4 in `user_prompt_submit.rs`. The E11 entity search provides a separate discovery channel for things E1 misses, so E1's threshold can be higher.

### 3.2 MEDIUM PRIORITY — Missing Hooks and Capabilities

#### R5: PreCompact hook should snapshot critical context before compression
**Current**: `pre_compact.sh` exists but its Rust handler is minimal.
**Impact**: When Claude Code compresses context, the most important memories from the current session could be lost. PreCompact has 20s budget — enough to store a session summary.
**Fix**: In the PreCompact handler:
1. Extract the last 5 user prompts from conversation context
2. Store a `SessionSummary` memory via MCP with importance=0.8
3. This memory will be retrievable in the continued conversation even after compaction

#### R6: TaskCompleted hook should extract learnings
**Current**: `task_completed.sh` exists but the Rust handler is not implemented.
**Impact**: When complex multi-step tasks complete, the patterns and decisions made are valuable for future reference but currently not captured.
**Fix**: Implement the TaskCompleted handler to:
1. Parse the task result
2. If successful, store a concise summary as a memory with `source_type: TaskResult`
3. Extract any file paths modified during the task for code awareness

#### R7: PostToolUse should store tool failure patterns
**Current**: Failed tools are logged but not stored as memories.
**Impact**: If Claude repeatedly hits the same error (e.g., a build failure pattern), the knowledge graph can't surface the previous failure context.
**Fix**: In `capture_tool_memory()`, also capture failed tool responses (currently skipped with `if !tool_success { return; }`), but with importance=0.3 and a `FailurePattern` source type.

### 3.3 MEDIUM PRIORITY — Search Strategy Improvements

#### R8: UserPromptSubmit should use `pipeline` strategy for code-heavy prompts
**Current**: Always uses `search_graph` with default multi-space strategy.
**Impact**: When the user prompt contains code patterns (function names, error messages with stack traces), the pipeline strategy (E13→E1→E12) would provide better precision.
**Fix**: Add code-pattern detection before the search call:
```rust
let strategy = if looks_like_code(&prompt) {
    "pipeline"
} else {
    "multi_space"
};
```
Simple heuristic: count backticks, brackets, `::`, `fn `, `def `, `function `, `class `.

#### R9: E11 entity search should deduplicate against E1 results
**Current**: E11 discovered memories are shown separately under "E11 Entity Discoveries" even if they overlap with E1 results.
**Impact**: Duplicate content wastes the context injection budget.
**Fix**: After both searches complete, deduplicate by `fingerprintId` before generating the injection string.

#### R10: UserPromptSubmit should use the causal search for "why" and "because" prompts
**Current**: The prompt is always searched with E1 semantic + E11 entity.
**Impact**: Prompts like "Why did the build fail?" or "What caused the regression?" would benefit from E5 causal search.
**Fix**: Detect causal intent keywords (`why`, `because`, `caused by`, `root cause`, `leads to`) and add a parallel E5 causal search with `CausalDirection::Effect` (searching for effects of the mentioned cause).

### 3.4 LOW PRIORITY — Operational Improvements

#### R11: SessionStart should inject project-level context
**Current**: SessionStart only sets up the session snapshot and drift metrics.
**Impact**: At session start, Claude has no memory of previous sessions' work. The previous session's key decisions and patterns could be injected.
**Fix**: If `previous_session_id` is provided and drift_metrics show near-zero drift, inject a 1-paragraph summary of the previous session's key outcomes as context injection from SessionStart.

#### R12: Shell scripts duplicate CLI binary discovery logic
**Current**: Every `.sh` file has identical 15-line CLI binary discovery code.
**Impact**: Maintenance burden — any path change must be updated in 12 files.
**Fix**: Extract to `.claude/hooks/lib/find-cli.sh` and source it:
```bash
source "$(dirname "$0")/lib/find-cli.sh"
```

#### R13: Timeout budget mismatch between settings.json and Rust constants
**Current**: `settings.json` sets UserPromptSubmit timeout to 3000ms, but `user_prompt_submit.rs` has `USER_PROMPT_SUBMIT_TIMEOUT_MS = 2000`. Similarly PreToolUse: 1000ms in settings vs 500ms constant.
**Impact**: The settings give more headroom than the Rust code expects, which is actually correct (shell overhead). But this should be documented to prevent confusion.
**Note**: No code change needed — just documenting the intentional design.

#### R14: Memory context budget is static
**Current**: `MEMORY_CONTEXT_BUDGET_CHARS = 2000` (~500 tokens) is hardcoded.
**Impact**: This budget doesn't adapt to the complexity of the user's prompt. A simple "fix the typo" needs less context than "redesign the authentication system".
**Fix**: Scale the budget based on prompt length:
```rust
let budget = if prompt.len() > 500 { 3000 } else { 2000 };
```

#### R15: Skills don't reference hook-injected context
**Current**: Skills like `cg-code-search` and `cg-entity` operate independently of hooks. When a user invokes `/cg-code-search`, it makes fresh MCP calls without leveraging already-cached memories from the hook pipeline.
**Impact**: Redundant MCP calls when hook-cached context is already available.
**Fix**: Skills could check the filesystem cache first (`get_cached_memories`) before making MCP calls, using cached results as a relevance hint.

---

## 4. Settings Configuration Audit

### Current Matchers vs Recommended

| Hook | Current Matcher | Recommended | Rationale |
|------|----------------|-------------|-----------|
| PreToolUse | `Edit\|Write\|Read\|Bash\|Task` | `Edit\|Write\|Read\|Bash\|Task\|Grep\|Glob` | Search operations should get context |
| PostToolUse | `Write\|Edit\|Bash\|Task` | `Write\|Edit\|Bash\|Task\|Read` | Read content should be capturable |
| PostToolUseFailure | `Write\|Edit\|Bash` | `Write\|Edit\|Bash\|Task` | Task failures are high-value learning |
| Others | No matcher (all events) | No change | Correct |

### Timeout Analysis

| Hook | Shell Timeout | Rust Timeout | Margin | Status |
|------|-------------|--------------|--------|--------|
| SessionStart | 5000ms | N/A (cache only) | OK | Healthy |
| UserPromptSubmit | 3000ms (2500ms internal) | 2000ms | 500ms shell overhead | Healthy |
| PreToolUse | 1000ms (500ms internal) | 500ms | 500ms shell overhead | Healthy |
| PostToolUse | 5000ms (3000ms internal) | 3000ms | 2000ms margin | Generous |
| SessionEnd | 30000ms | N/A (cache only) | OK | Healthy |

---

## 5. Skills Quality Audit

All 7 CG-specific skills verified against Rust source-of-truth (handler code):

| Skill | Param Names | Required/Optional | Edge Cases | Status |
|-------|------------|-------------------|------------|--------|
| `cg-code-search` | Correct (fixed this session) | Correct | Documented | PASS |
| `cg-blind-spot` | Correct (fixed this session) | Correct | Documented | PASS |
| `semantic-search` | Correct | Correct | Documented | PASS |
| `memory-inject` | Correct | Correct | Documented | PASS |
| `cg-causal` | Correct | Correct | Documented | PASS |
| `cg-entity` | Correct | Correct | Documented | PASS |
| `cg-session` | Correct | Correct | Documented | PASS |
| `cg-provenance` | Correct | Correct | Documented | PASS |
| `cg-topics` | Correct | Correct | Documented | PASS |
| `cg-curator` | Correct | Correct | Documented | PASS |
| `cg-graph` | Correct | Correct | Documented | PASS |

---

## 6. Priority Implementation Roadmap

### Phase 1 — Quick Wins (< 1 hour each)

1. **R3**: Slim down context injection — remove Coherence State and Identity Marker sections
2. **R1**: Add `Read` to PostToolUse matcher in settings.json
3. **R2**: Add `Grep|Glob` to PreToolUse matcher in settings.json
4. **R4**: Raise MIN_MEMORY_SIMILARITY from 0.3 to 0.4
5. **R9**: Deduplicate E11 results against E1 by fingerprintId
6. **R12**: Extract CLI binary discovery to shared lib

### Phase 2 — Medium Effort (2-4 hours each)

7. **R8**: Add code-pattern detection for pipeline strategy selection
8. **R10**: Add causal intent detection for E5 search
9. **R5**: Implement PreCompact session summary capture
10. **R7**: Capture failure patterns in PostToolUse

### Phase 3 — Larger Scope (4-8 hours each)

11. **R6**: Implement TaskCompleted learning extraction
12. **R11**: SessionStart previous-session summary injection
13. **R14**: Adaptive memory context budget
14. **R15**: Skill-hook cache integration

---

## 7. Test Coverage

Current test suite: **8,069 tests pass**, zero failures across all 10 crates.

Hook-specific tests:
- `user_prompt_submit.rs`: 7 async tests + 3 sync tests (identity markers, context evaluation, coherence mapping)
- `pre_tool_use.rs`: 7 tests (timing, output structure, tool guidance, classification, no-DB verification)
- `post_tool_use.rs`: 8 tests (tool processing, impact analysis, success inference, edge cases)
- `session_start.rs`: 11 tests (creation, linking, drift metrics, cosine distance, state restoration)
- `session_end.rs`: 7 tests (warm/cold cache, persistence, schema compliance, timing)
- `memory_cache.rs`: 6 tests (cache/retrieve, expiry, limits, path sanitization, overwrite)

All tests use real SessionCache state (no mocks) per user's testing philosophy.

---

## 8. Summary

The Context Graph integration with Claude Code is architecturally sound. The multi-embedder pipeline (E1+E11+E4+divergence) in UserPromptSubmit is the system's strongest differentiator. The filesystem cache correctly solves the cross-process communication constraint.

The highest-impact improvement is **R3 (slim down context injection)** — currently ~40-60% of the injected content is metadata that doesn't help Claude execute tasks better. Removing the Coherence State and Identity Marker sections would let the actually useful content (memories, entity discoveries, divergence alerts) occupy more of the limited system-reminder budget.

The second highest-impact improvement is **R1+R2 (expand hook matchers)** — Read responses and search operations are currently not receiving the full benefit of the hook pipeline, which is a configuration-only fix.
