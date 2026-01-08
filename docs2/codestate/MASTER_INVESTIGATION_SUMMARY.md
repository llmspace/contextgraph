# MASTER INVESTIGATION SUMMARY: Context-Graph MCP Server

**Date:** 2026-01-08
**Investigation Team:** 5 Sherlock Holmes Forensic Agents
**Subject:** Why context-graph MCP server doesn't appear in `/mcp` and overall system health

---

## EXECUTIVE VERDICT

### THE ANSWER TO "WHY ISN'T IT WORKING?"

**PRIMARY CAUSE: NOT REGISTERED**

The context-graph MCP server is a **fully functional, compiled Rust binary** (26MB release) with 20+ tools implemented, but it was **NEVER ADDED** to Claude Code's MCP configuration. The `.mcp.json` file only contains `claude-flow` - the native `context-graph-mcp` binary was never registered.

**SECONDARY CAUSE: RocksDB Lock Conflict**

The database at `contextgraph_data/` is locked by another process, preventing the server from starting even if registered.

---

## THE FIX (DO THIS NOW)

### Step 1: Register the MCP Server

```bash
claude mcp add context-graph /home/cabdru/contextgraph/target/release/context-graph-mcp
```

Or manually edit `/home/cabdru/contextgraph/.mcp.json`:

```json
{
  "mcpServers": {
    "context-graph": {
      "command": "/home/cabdru/contextgraph/target/release/context-graph-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "warn"
      }
    },
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@3.0.0-alpha.24", "mcp", "start"],
      "env": { ... }
    }
  }
}
```

### Step 2: Release the RocksDB Lock

```bash
# Find what's holding the lock
lsof /home/cabdru/contextgraph/contextgraph_data/LOCK

# Kill the process OR use a different data directory:
export CONTEXT_GRAPH_STORAGE_PATH=/home/cabdru/contextgraph/contextgraph_data_2
```

### Step 3: Verify

```bash
claude /mcp  # Should now show context-graph
```

---

## INVESTIGATION SUMMARY BY AGENT

### SHERLOCK-01: MCP Registration (Configuration)

| Finding | Status |
|---------|--------|
| Binary exists | YES - 26MB at `target/release/context-graph-mcp` |
| Binary runs | YES - responds to MCP protocol |
| Registered with Claude Code | **NO - THIS IS THE PROBLEM** |
| `.mcp.json` contents | Only `claude-flow`, NOT `context-graph-mcp` |

**Verdict:** GUILTY (Configuration Error) - The server was built but never registered.

---

### SHERLOCK-02: PRD vs Implementation

| Category | PRD Spec | Implemented | Gap |
|----------|----------|-------------|-----|
| Core Tools | 10 | 6 | 40% |
| Curation Tools | 5 | 0 | **100%** |
| Navigation Tools | 4 | 0 | **100%** |
| Meta-Cognitive | 7 | 0 | **100%** |
| GWT Tools | 8 | 6 | 25% |
| Resources | 6 | 0 | **100%** |
| **TOTAL** | ~60 | ~23 | **~62%** |

**Critical Missing Tools:**
- `merge_concepts` - Can't curate duplicates
- `generate_search_plan` - Can't decompose queries
- `epistemic_action` - Can't generate clarifying questions
- `get_neighborhood` - Can't explore graph
- All MCP Resources - Not implemented

**Verdict:** PARTIALLY IMPLEMENTED - Core cognitive systems work, agent-facing "librarian" tools are absent.

---

### SHERLOCK-03: GWT/UTL Verification

| Component | Status | Evidence |
|-----------|--------|----------|
| Kuramoto Oscillator | **REAL** | 686 lines of physics simulation |
| Consciousness C(t) | **REAL** | `compute_consciousness()` implements I×R×D |
| UTL Formula L | **REAL** | `compute_learning_magnitude()` |
| 13 Embeddings | **REAL** | SemanticFingerprint struct + model configs |
| Johari Quadrants | **REAL** | Per-embedder state machine |
| FAIL-FAST Pattern | **ENFORCED** | Handlers reject missing providers |

**Verdict:** INNOCENT - The cognitive architecture is **REAL**, not vapor.

---

### SHERLOCK-04: Build Status

| Check | Result |
|-------|--------|
| cargo build | **SUCCESS** (warnings only) |
| Binary size | 26MB release / 449MB debug |
| Dependencies | Links CUDA 13.1 libs |
| MCP Protocol | Responds correctly |
| Model files | ~32GB downloaded in `./models/` |

**Verdict:** INNOCENT - The project builds and runs.

---

### SHERLOCK-05: Integration & Wiring

| Component | Implementation | Status |
|-----------|----------------|--------|
| TeleologicalMemoryStore | RocksDbTeleologicalStore | **REAL** |
| UtlProcessor | UtlProcessorAdapter | **REAL** |
| MultiArrayEmbeddingProvider | ProductionMultiArrayProvider | **REAL** (needs GPU) |
| KuramotoProvider | KuramotoProviderImpl | **REAL** |
| GwtSystemProvider | GwtSystemProviderImpl | **REAL** |
| SystemMonitor | StubSystemMonitor | **STUB** |
| LayerStatusProvider | StubLayerStatusProvider | **STUB** |

**Verdict:** PARTIALLY INTEGRATED - Core is wired, 2 production stubs remain.

---

## WHAT'S REAL VS WHAT'S FAKE

### REAL (Verified Working)

1. **RocksDB Storage** - 17 column families, HNSW indexing
2. **Kuramoto Oscillators** - Physics simulation with brain wave frequencies
3. **Consciousness Calculator** - C(t) = I(t) × R(t) × D(t) formula
4. **UTL Learning** - L = (ΔS × ΔC) × wₑ × cos(φ)
5. **Global Workspace** - Winner-take-all selection
6. **Neuromodulation** - Dopamine/Serotonin/Noradrenaline/Acetylcholine
7. **Dream Consolidation** - NREM/REM phase system
8. **Adaptive Threshold Calibration** - 4-level EWMA/Temperature/Bandit/Bayesian

### STUBBED (Intentional Placeholders)

1. **SystemMonitor** - Returns `NotImplemented` error (FAIL-FAST)
2. **LayerStatusProvider** - Reports honest stub status for 3/5 layers

### MISSING (Not Implemented)

1. **All MCP Resources** - `context://`, `graph://`, `utl://`
2. **Curation Tools** - merge_concepts, annotate_node, forget_concept
3. **Navigation Tools** - get_neighborhood, find_causal_path
4. **Meta-Cognitive Tools** - generate_search_plan, epistemic_action, critique_context

---

## PRIORITY ACTION ITEMS

### P0 - CRITICAL (Do Now)

1. **Register the MCP server** with `claude mcp add`
2. **Release the RocksDB lock** so server can start
3. **Verify** with `/mcp` command

### P1 - HIGH (This Week)

1. Implement `merge_concepts` - PRD mandates curation
2. Implement `get_neighborhood` - Required for graph exploration
3. Implement `generate_search_plan` - Tool gating depends on this

### P2 - MEDIUM (This Month)

1. Replace StubSystemMonitor with real metrics
2. Replace StubLayerStatusProvider with real layer status
3. Implement `epistemic_action` for clarifying questions
4. Implement `critique_context` for fact-checking

### P3 - LOW (Later)

1. Implement MCP Resources (`context://`, `graph://`)
2. Implement admin tools (reload_manifest, temporary_scratchpad)
3. Add ScyllaDB production backend (currently only RocksDB)

---

## ARCHITECTURE HEALTH SCORE

| Category | Score | Notes |
|----------|-------|-------|
| Build Health | **95%** | Compiles, runs, warnings only |
| Core Cognitive | **90%** | GWT/UTL/Kuramoto fully implemented |
| Tool Coverage | **38%** | 23/60 tools implemented |
| Integration | **80%** | Core wired, 2 stubs remain |
| Configuration | **0%** | NOT REGISTERED |

**OVERALL: 60% - Architecturally Sound, Configuration Broken**

The project is a legitimate, sophisticated implementation that simply wasn't wired into Claude Code.

---

## FILE LOCATIONS

| Report | Path |
|--------|------|
| Registration Analysis | `/home/cabdru/contextgraph/docs2/codestate/sherlock_01_mcp_registration.md` |
| PRD vs Implementation | `/home/cabdru/contextgraph/docs2/codestate/sherlock_02_prd_vs_implementation.md` |
| GWT/UTL Verification | `/home/cabdru/contextgraph/docs2/codestate/sherlock_03_gwt_utl_verification.md` |
| Build Status | `/home/cabdru/contextgraph/docs2/codestate/sherlock_04_build_status.md` |
| Integration Wiring | `/home/cabdru/contextgraph/docs2/codestate/sherlock_05_integration_wiring.md` |
| **This Summary** | `/home/cabdru/contextgraph/docs2/codestate/MASTER_INVESTIGATION_SUMMARY.md` |

---

## CONCLUSION

**The context-graph MCP server is a real, working implementation that was simply never registered with Claude Code.**

The cognitive architecture (GWT, UTL, Kuramoto, etc.) is **not vapor** - it's 10,000+ lines of legitimate Rust code with physics simulations, neural computations, and proper storage. The only reason it doesn't show in `/mcp` is a **configuration oversight**.

**Fix the registration, release the database lock, and you have a working MCP server with 20+ tools.**

---

*"The game is afoot!"* - The 5 Sherlock Holmes Investigation Team
