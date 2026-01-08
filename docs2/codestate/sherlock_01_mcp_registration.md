# SHERLOCK HOLMES CASE FILE: MCP Server Registration Investigation

## Case ID: SHERLOCK-01-MCP-REG
## Date: 2026-01-08
## Investigator: Sherlock Holmes (Agent 1)
## Subject: Why context-graph MCP server doesn't appear in Claude Code's `/mcp` list

---

## 1. EXECUTIVE SUMMARY

**VERDICT: IMPLEMENTED BUT NOT REGISTERED**

The context-graph MCP server is fully implemented as a Rust binary (`context-graph-mcp`) with 20+ MCP tools, uses stdio transport as required by Claude Code, and has been successfully compiled (449MB debug, 26MB release). However, it is **NOT REGISTERED** in any Claude Code configuration file - the `.mcp.json` file in `/home/cabdru/contextgraph/` only registers `claude-flow`, and the user's global Claude configuration (`~/.claude.json`) for the contextgraph project likewise only contains third-party MCP servers (claude-flow, ruv-swarm, flow-nexus, serena). The native context-graph MCP server has never been added to any configuration.

---

## 2. EVIDENCE GATHERED

### 2.1 Binary Existence (VERIFIED)

| Location | Size | Status |
|----------|------|--------|
| `/home/cabdru/contextgraph/target/debug/context-graph-mcp` | 449,631,320 bytes | EXISTS |
| `/home/cabdru/contextgraph/target/release/context-graph-mcp` | 26,437,016 bytes | EXISTS |

**Evidence Source**: `ls -la /home/cabdru/contextgraph/target/debug/` (lines 1-2)

### 2.2 Cargo.toml Binary Definition (VERIFIED)

File: `/home/cabdru/contextgraph/crates/context-graph-mcp/Cargo.toml` (lines 15-17)

```toml
[[bin]]
name = "context-graph-mcp"
path = "src/main.rs"
```

### 2.3 Main Entry Point Implementation (VERIFIED)

File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/main.rs` (lines 1-95)

Key evidence:
- Line 8-9: Transport documented as stdio (default)
- Line 42-43: `#[tokio::main] async fn main()`
- Line 48-52: Logging configured to write to **stderr** (correct for MCP - stdout must be JSON-RPC only)
- Line 88-91: Server created with `McpServer::new(config).await` and `server.run().await`

### 2.4 Server Implementation (VERIFIED)

File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/server.rs`

Key evidence:
- Line 189-234: `run()` method reads from stdin, writes JSON-RPC to stdout
- Line 196-203: Line-by-line stdin reading
- Line 219-224: Newline-delimited JSON output to stdout

**Transport**: stdio (correct for Claude Code)

### 2.5 Tool Definitions (VERIFIED - 20+ tools)

File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools.rs`

Implemented MCP tools (from tools.rs and handlers/tools.rs):
1. `inject_context` - Context injection with UTL processing
2. `store_memory` - Direct memory storage
3. `get_memetic_status` - UTL metrics and system state
4. `get_graph_manifest` - 5-layer architecture description
5. `search_graph` - Semantic search
6. `utl_status` - UTL system state query
7. `get_consciousness_state` - GWT consciousness (C, r, meta_score)
8. `get_kuramoto_sync` - Kuramoto oscillator network state
9. `get_workspace_status` - Global Workspace status
10. `get_ego_state` - Self-Ego Node state
11. `trigger_workspace_broadcast` - Force memory into workspace
12. `adjust_coupling` - Modify Kuramoto coupling K
13. `get_threshold_status` - Adaptive threshold calibration
14. `get_calibration_metrics` - ECE, MCE, Brier scores
15. `trigger_recalibration` - Force threshold recalibration
16. `trigger_dream` - Dream consolidation
17. `get_dream_status` - Dream phase status
18. `abort_dream` - Abort active dream
19. `get_amortized_shortcuts` - Amortized learning shortcuts
20. `get_neuromodulation_state` - Neurotransmitter levels
21. `adjust_neuromodulator` - Modify NT levels
22. `get_steering_feedback` - Steering subsystem feedback
23. `omni_infer` - Omnidirectional inference

### 2.6 Current .mcp.json Configuration (THE PROBLEM)

File: `/home/cabdru/contextgraph/.mcp.json` (lines 1-20)

```json
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@3.0.0-alpha.24", "mcp", "start"],
      "env": { ... },
      "autoStart": true
    }
  }
}
```

**OBSERVATION**: Only `claude-flow` is registered. The native `context-graph-mcp` is NOT registered.

### 2.7 User's Global Claude Configuration (CONFIRMS THE PROBLEM)

File: `/home/cabdru/.claude.json` (lines 767-816)

For the `/home/cabdru/contextgraph` project:
```json
"mcpServers": {
  "claude-flow": { ... },
  "ruv-swarm": { ... },
  "flow-nexus": { ... },
  "serena": { ... }
}
```

**OBSERVATION**: Four third-party MCP servers registered, but NOT `context-graph-mcp`.

### 2.8 Binary Startup Test (PARTIAL FAILURE - External cause)

```
$ timeout 5 ./target/release/context-graph-mcp 2>&1

ERROR: Failed to open RocksDB at '/home/cabdru/contextgraph/contextgraph_data':
IO error: While lock file: .../LOCK: Resource temporarily unavailable
```

**OBSERVATION**: The binary attempts to start but fails because RocksDB is locked by another process. This is NOT a fundamental issue with the MCP server - it would work if the database were not locked.

---

## 3. ROOT CAUSE ANALYSIS

### Primary Root Cause: NO REGISTRATION

The context-graph MCP server binary exists and is correctly implemented with stdio transport, but it has **NEVER BEEN REGISTERED** with Claude Code.

Claude Code discovers MCP servers through:
1. **Project-level `.mcp.json`** - Only contains `claude-flow`
2. **User-level `~/.claude.json` projects[path].mcpServers`** - Only contains third-party servers

### Secondary Issue: RocksDB Lock Conflict

When attempting to start the server, it fails because the RocksDB database at `contextgraph_data/` is locked by another process. This is a runtime issue, not an implementation issue.

### Tertiary Issue: GPU/Model Requirements

The server requires:
1. NVIDIA CUDA GPU with 8GB+ VRAM
2. Model files in `./models` directory
3. No other process holding the RocksDB lock

---

## 4. WHAT'S ACTUALLY PRESENT vs. WHAT'S MISSING

### PRESENT (Implemented and Compiled)

| Component | Status | Location |
|-----------|--------|----------|
| Rust binary | COMPILED | `target/release/context-graph-mcp` |
| stdio transport | IMPLEMENTED | `server.rs:189-234` |
| JSON-RPC 2.0 | IMPLEMENTED | `protocol.rs` |
| 20+ MCP tools | IMPLEMENTED | `tools.rs`, `handlers/tools.rs` |
| Tool schemas | IMPLEMENTED | `tools.rs:get_tool_definitions()` |
| Error handling | IMPLEMENTED | Fail-fast pattern |
| Logging to stderr | IMPLEMENTED | `main.rs:48-52` |

### MISSING (Not Configured)

| Component | Status | Required Action |
|-----------|--------|-----------------|
| Registration in `.mcp.json` | MISSING | Add server entry |
| Registration in `~/.claude.json` | MISSING | Add server entry |
| RocksDB lock management | MISSING | Ensure single instance |
| Model files | UNKNOWN | Verify `./models` directory |

---

## 5. RECOMMENDED FIX

### Step 1: Add to Project .mcp.json

Edit `/home/cabdru/contextgraph/.mcp.json`:

```json
{
  "mcpServers": {
    "context-graph": {
      "command": "/home/cabdru/contextgraph/target/release/context-graph-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "warn",
        "CONTEXT_GRAPH_STORAGE_PATH": "/home/cabdru/contextgraph/contextgraph_data"
      },
      "autoStart": true
    },
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@3.0.0-alpha.24", "mcp", "start"],
      "env": {
        "CLAUDE_FLOW_MODE": "v3",
        "CLAUDE_FLOW_HOOKS_ENABLED": "true",
        "CLAUDE_FLOW_TOPOLOGY": "hierarchical-mesh",
        "CLAUDE_FLOW_MAX_AGENTS": "15",
        "CLAUDE_FLOW_MEMORY_BACKEND": "hybrid"
      },
      "autoStart": true
    }
  }
}
```

### Step 2: Alternative - Use claude mcp add command

```bash
claude mcp add context-graph \
  /home/cabdru/contextgraph/target/release/context-graph-mcp \
  --env RUST_LOG=warn \
  --env CONTEXT_GRAPH_STORAGE_PATH=/home/cabdru/contextgraph/contextgraph_data
```

### Step 3: Release RocksDB Lock

Before starting the MCP server, ensure no other process holds the database lock:

```bash
# Find processes using the database
lsof /home/cabdru/contextgraph/contextgraph_data/LOCK

# Or use fuser
fuser -v /home/cabdru/contextgraph/contextgraph_data/LOCK
```

### Step 4: Verify Model Files (If Needed)

Ensure model files exist at the expected location:
```bash
ls -la /home/cabdru/contextgraph/models/
```

Or set the environment variable:
```bash
export CONTEXT_GRAPH_MODELS_PATH=/path/to/your/models
```

---

## 6. VERIFICATION CHECKLIST

After applying the fix, verify:

- [ ] Run `claude /mcp` and confirm `context-graph` appears in the list
- [ ] Run `claude /mcp context-graph` to see available tools (should show 20+)
- [ ] Test a simple tool call: `inject_context` with sample content
- [ ] Verify RocksDB database is accessible (no lock errors)
- [ ] Check logs at stderr for any initialization errors

---

## 7. CHAIN OF CUSTODY

| Timestamp | Action | Evidence |
|-----------|--------|----------|
| 2026-01-08 18:58 | Examined `.mcp.json` | Only claude-flow registered |
| 2026-01-08 18:58 | Examined `~/.claude.json` | No context-graph entry |
| 2026-01-08 18:58 | Verified binary exists | 26MB release binary present |
| 2026-01-08 18:58 | Tested binary startup | Failed due to RocksDB lock |
| 2026-01-08 18:58 | Verified tool definitions | 20+ tools implemented |

---

## 8. VERDICT

**CASE CLOSED: GUILTY (Configuration Error)**

The context-graph MCP server is fully implemented but has never been registered with Claude Code. The `.mcp.json` file and `~/.claude.json` project configuration only contain third-party MCP servers (claude-flow, ruv-swarm, flow-nexus, serena), with no entry for the native `context-graph-mcp` binary.

The fix is straightforward: add the server registration to `.mcp.json` with the absolute path to the compiled binary.

---

## 9. SUPPLEMENTARY EVIDENCE: Complete Tool List from tools.rs

The PRD at `/home/cabdru/contextgraph/docs2/contextprd.md` (Section 5) describes the expected MCP tools. The implementation in `tools.rs` and `handlers/tools.rs` covers:

**Core Tools (PRD Section 5.2):**
- `inject_context` - IMPLEMENTED
- `store_memory` - IMPLEMENTED
- `search_graph` - IMPLEMENTED
- `get_memetic_status` - IMPLEMENTED
- `utl_status` - IMPLEMENTED

**GWT Tools (PRD Section 5.10):**
- `get_consciousness_state` - IMPLEMENTED
- `get_workspace_status` - IMPLEMENTED
- `get_kuramoto_sync` - IMPLEMENTED
- `get_ego_state` - IMPLEMENTED
- `trigger_workspace_broadcast` - IMPLEMENTED
- `adjust_coupling` - IMPLEMENTED

**Additional Tools:**
- `get_graph_manifest` - IMPLEMENTED
- `trigger_dream` - IMPLEMENTED
- `get_dream_status` - IMPLEMENTED
- `abort_dream` - IMPLEMENTED
- `get_amortized_shortcuts` - IMPLEMENTED
- `get_neuromodulation_state` - IMPLEMENTED
- `adjust_neuromodulator` - IMPLEMENTED
- `get_threshold_status` - IMPLEMENTED
- `get_calibration_metrics` - IMPLEMENTED
- `trigger_recalibration` - IMPLEMENTED
- `get_steering_feedback` - IMPLEMENTED
- `omni_infer` - IMPLEMENTED

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The impossible: A bug in Claude Code preventing MCP discovery.
The improbable but true: The server was simply never registered.

**INVESTIGATION COMPLETE.**
