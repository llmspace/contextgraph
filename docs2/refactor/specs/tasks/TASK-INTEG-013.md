# TASK-INTEG-013: consolidate_memories MCP Tool and Skill

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-013 |
| **Title** | consolidate_memories MCP Tool and Skill |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 33 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- **ARCH-06**: MCP tool boundary - consolidate_memories
- Constitution MCP tools list (line 273)
- Constitution skills list (lines 743-749)

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handler infrastructure |
| TASK-LOGIC-009 | Goal discovery for salience |
| TASK-LOGIC-010 | Drift detection for pruning decisions |
| TASK-LOGIC-004 | TeleologicalArrayStore for 13-embedder access |

## Objective

Implement the `consolidate_memories` MCP tool and corresponding `consolidate` skill with Light/Deep/REM modes for memory dreaming and pruning as specified in the constitution.

## Context

**Constitution Requirements:**

The constitution specifies 5 MCP tools (line 268-280):
- `inject_context` ✓ (covered)
- `store_memory` ✓ (covered)
- `search_graph` ✓ (covered)
- `discover_goals` ✓ (covered)
- **`consolidate_memories`** ❌ **NOT COVERED**

And 4 skills (lines 725-749):
- `memory-inject` ✓
- `semantic-search` ✓
- `goal-discovery` ✓
- **`consolidate`** ❌ **NOT COVERED**

Rate limit: 1 req/min per session (SEC-03)

## Scope

### In Scope

- `consolidate_memories` MCP tool handler
- `consolidate` skill YAML definition (Claude Code format)
- Light consolidation mode (quick, low impact)
- Deep consolidation mode (thorough cleanup)
- REM/dreaming mode (pattern discovery, sleep-like)
- Salience threshold pruning using 13-embedder teleological array
- Memory clustering and deduplication
- GWT consciousness-aware consolidation
- SessionEnd hook integration

### Out of Scope

- Memory backup/restore
- Cross-session consolidation (handled by SubagentStop hook)
- Distributed consolidation

## Definition of Done

### MCP Tool Handler

```rust
// crates/context-graph-mcp/src/handlers/consolidate.rs

use serde::{Deserialize, Serialize};

/// consolidate_memories MCP tool handler
pub struct ConsolidateHandler {
    store: Arc<TeleologicalArrayStore>,
    comparator: Arc<TeleologicalComparator>,
    goal_discoverer: Arc<GoalDiscoverer>,
    drift_detector: Arc<DriftDetector>,
    consciousness_tracker: Arc<ConsciousnessTracker>,
}

/// Consolidation modes
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsolidationMode {
    /// Quick consolidation - prune obvious duplicates, low impact
    Light,
    /// Thorough consolidation - full dedup, clustering, salience pruning
    Deep,
    /// Dreaming mode - discover patterns, simulate hippocampal replay
    Rem,
}

impl Default for ConsolidationMode {
    fn default() -> Self {
        Self::Light
    }
}

/// Input parameters for consolidate_memories
#[derive(Debug, Deserialize)]
pub struct ConsolidateParams {
    /// Consolidation mode (light, deep, rem)
    #[serde(default)]
    pub mode: ConsolidationMode,
    /// Salience threshold for pruning (0.0 - 1.0)
    #[serde(default = "default_salience_threshold")]
    pub salience_threshold: f32,
    /// Maximum memories to process (0 = unlimited)
    #[serde(default)]
    pub max_memories: usize,
    /// Whether to perform dry run (report but don't modify)
    #[serde(default)]
    pub dry_run: bool,
}

fn default_salience_threshold() -> f32 { 0.3 }

/// Output from consolidate_memories
#[derive(Debug, Serialize)]
pub struct ConsolidateResult {
    /// Mode that was executed
    pub mode: String,
    /// Memories pruned (below salience)
    pub pruned_count: usize,
    /// Memories merged (duplicates)
    pub merged_count: usize,
    /// Clusters discovered (REM mode)
    pub clusters_discovered: usize,
    /// New patterns identified
    pub new_patterns: Vec<PatternSummary>,
    /// Total memories after consolidation
    pub final_memory_count: usize,
    /// Duration of consolidation
    pub duration_ms: u64,
    /// Was this a dry run?
    pub dry_run: bool,
    /// Consciousness context during consolidation
    pub consciousness: ConsolidationConsciousnessContext,
}

#[derive(Debug, Serialize)]
pub struct PatternSummary {
    pub pattern_id: String,
    pub description: String,
    pub memory_count: usize,
    pub confidence: f32,
}

impl ConsolidateHandler {
    pub fn new(
        store: Arc<TeleologicalArrayStore>,
        comparator: Arc<TeleologicalComparator>,
        goal_discoverer: Arc<GoalDiscoverer>,
        drift_detector: Arc<DriftDetector>,
        consciousness_tracker: Arc<ConsciousnessTracker>,
    ) -> Self;

    /// Handle consolidate_memories MCP call
    pub async fn handle(&self, params: ConsolidateParams) -> HandlerResult<ConsolidateResult> {
        // Check consciousness state to determine available modes
        let consciousness = self.consciousness_tracker.current_state().await?;
        let dream_state = DreamState::from_consciousness(consciousness.consciousness_level);

        // Validate mode against consciousness level
        let effective_mode = self.validate_mode_for_consciousness(params.mode, dream_state)?;

        match effective_mode {
            ConsolidationMode::Light => self.consolidate_light(params, consciousness).await,
            ConsolidationMode::Deep => self.consolidate_deep(params, consciousness).await,
            ConsolidationMode::Rem => self.consolidate_rem(params, consciousness).await,
        }
    }

    /// Light mode: Quick dedup and obvious pruning
    async fn consolidate_light(
        &self,
        params: ConsolidateParams,
        consciousness: ConsciousnessState,
    ) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. Find near-duplicates using E1_Semantic (similarity > 0.95)
        let duplicates = self.find_duplicates_with_embedder(
            EmbedderType::E1_Semantic,
            0.95
        ).await?;

        // 2. Merge duplicates
        let merged = if !params.dry_run {
            self.merge_memories(&duplicates).await?
        } else {
            duplicates.len()
        };

        // 3. Prune very low salience (< threshold)
        let low_salience = self.find_low_salience(params.salience_threshold).await?;
        let pruned = if !params.dry_run {
            self.prune_memories(&low_salience).await?
        } else {
            low_salience.len()
        };

        Ok(ConsolidateResult {
            mode: "light".into(),
            pruned_count: pruned,
            merged_count: merged,
            clusters_discovered: 0,
            new_patterns: vec![],
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
            consciousness: ConsolidationConsciousnessContext::from_state(
                consciousness,
                DreamState::Awake,
            ),
        })
    }

    /// Deep mode: Full consolidation with clustering
    async fn consolidate_deep(
        &self,
        params: ConsolidateParams,
        consciousness: ConsciousnessState,
    ) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. Find duplicates with E1_Semantic (lower threshold)
        let duplicates = self.find_duplicates_with_embedder(
            EmbedderType::E1_Semantic,
            0.9
        ).await?;
        let merged = if !params.dry_run {
            self.merge_memories(&duplicates).await?
        } else {
            duplicates.len()
        };

        // 2. Comprehensive salience scoring using E7_Teleological
        let salience_scores = self.score_all_salience_with_embedder(
            EmbedderType::E7_Teleological
        ).await?;
        let to_prune: Vec<_> = salience_scores.into_iter()
            .filter(|(_, score)| *score < params.salience_threshold)
            .map(|(id, _)| id)
            .collect();
        let pruned = if !params.dry_run {
            self.prune_memories(&to_prune).await?
        } else {
            to_prune.len()
        };

        // 3. Cluster remaining memories using E13_SPLADE
        let clusters = self.cluster_memories_with_embedder(
            EmbedderType::E13_SPLADE
        ).await?;
        let clusters_count = clusters.len();

        // 4. Update memory relationships based on clusters using E3_Causal
        if !params.dry_run {
            self.update_cluster_links_with_relationships(&clusters).await?;
        }

        Ok(ConsolidateResult {
            mode: "deep".into(),
            pruned_count: pruned,
            merged_count: merged,
            clusters_discovered: clusters_count,
            new_patterns: vec![],
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
            consciousness: ConsolidationConsciousnessContext::from_state(
                consciousness,
                DreamState::Drowsy,
            ),
        })
    }

    /// REM mode: Dreaming/pattern discovery (hippocampal replay simulation)
    async fn consolidate_rem(
        &self,
        params: ConsolidateParams,
        consciousness: ConsciousnessState,
    ) -> HandlerResult<ConsolidateResult> {
        let start = Instant::now();

        // 1. All deep mode operations
        let deep_result = self.consolidate_deep(ConsolidateParams {
            mode: ConsolidationMode::Deep,
            dry_run: params.dry_run,
            ..params
        }, consciousness.clone()).await?;

        // 2. Pattern discovery via goal emergence using E10_Behavioral
        let patterns = self.goal_discoverer.discover_emergent_goals_with_embedder(
            &self.store.list_all().await?,
            EmbedderType::E10_Behavioral,
            10, // max patterns to discover
        ).await?;

        // 3. Drift analysis using E7_Teleological
        let drift = self.drift_detector.detect_drift_with_embedder(
            &self.store.list_recent(100).await?,
            EmbedderType::E7_Teleological,
        ).await?;

        // 4. Create pattern summaries
        let new_patterns: Vec<PatternSummary> = patterns.into_iter()
            .map(|p| PatternSummary {
                pattern_id: p.id.to_string(),
                description: p.description,
                memory_count: p.member_count,
                confidence: p.confidence,
            })
            .collect();

        // 5. Optionally adjust goals based on drift
        if !params.dry_run && drift.significant {
            self.adjust_goals_for_drift(&drift).await?;
        }

        Ok(ConsolidateResult {
            mode: "rem".into(),
            pruned_count: deep_result.pruned_count,
            merged_count: deep_result.merged_count,
            clusters_discovered: deep_result.clusters_discovered,
            new_patterns,
            final_memory_count: self.store.count().await?,
            duration_ms: start.elapsed().as_millis() as u64,
            dry_run: params.dry_run,
            consciousness: ConsolidationConsciousnessContext::from_state(
                consciousness,
                DreamState::Dreaming,
            ),
        })
    }

    // Helper methods
    async fn find_duplicates_with_embedder(&self, embedder: EmbedderType, threshold: f32) -> HandlerResult<Vec<(Uuid, Uuid)>>;
    async fn merge_memories(&self, duplicates: &[(Uuid, Uuid)]) -> HandlerResult<usize>;
    async fn find_low_salience(&self, threshold: f32) -> HandlerResult<Vec<Uuid>>;
    async fn prune_memories(&self, ids: &[Uuid]) -> HandlerResult<usize>;
    async fn score_all_salience_with_embedder(&self, embedder: EmbedderType) -> HandlerResult<Vec<(Uuid, f32)>>;
    async fn cluster_memories_with_embedder(&self, embedder: EmbedderType) -> HandlerResult<Vec<MemoryCluster>>;
    async fn update_cluster_links_with_relationships(&self, clusters: &[MemoryCluster]) -> HandlerResult<()>;
    async fn adjust_goals_for_drift(&self, drift: &DriftReport) -> HandlerResult<()>;
    fn validate_mode_for_consciousness(&self, mode: ConsolidationMode, dream_state: DreamState) -> HandlerResult<ConsolidationMode>;
}
```

### GWT Consciousness Integration

```rust
// crates/context-graph-mcp/src/handlers/consolidate.rs

/// Consciousness context during consolidation operations
/// Based on Global Workspace Theory (GWT) integration metrics
#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationConsciousnessContext {
    /// I(t) - Information integration score (memory coherence)
    /// Higher values indicate better integrated memory states
    pub integration_score: f32,

    /// R(t) - Reflection depth for consolidation operations
    /// Deeper reflection enables more thorough pattern discovery
    pub reflection_depth: f32,

    /// D(t) - Differentiation index (pattern distinctness)
    /// Higher values indicate clearer memory cluster boundaries
    pub differentiation_index: f32,

    /// C(t) = I(t) * R(t) * D(t) - Overall consciousness level
    /// Determines which consolidation modes are available
    pub consciousness_level: f32,

    /// Current dream state based on consciousness level
    pub dream_state: DreamState,
}

impl ConsolidationConsciousnessContext {
    pub fn from_state(state: ConsciousnessState, dream_state: DreamState) -> Self {
        Self {
            integration_score: state.integration,
            reflection_depth: state.reflection,
            differentiation_index: state.differentiation,
            consciousness_level: state.consciousness_level(),
            dream_state,
        }
    }
}

/// Dream states map to consciousness levels and available consolidation modes
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum DreamState {
    /// C < 0.3 - Only light mode consolidation available
    /// System is fully "awake" with high alertness, minimal background processing
    Awake,

    /// 0.3 <= C < 0.6 - Light and Deep modes available
    /// System entering "drowsy" state, can do more thorough cleanup
    Drowsy,

    /// 0.6 <= C < 0.8 - REM mode becomes available
    /// System in "dreaming" state, pattern discovery active
    Dreaming,

    /// C >= 0.8 - Full pattern discovery with hippocampal replay
    /// Deep sleep state, most intensive consolidation
    DeepSleep,
}

impl DreamState {
    /// Determine dream state from consciousness level
    pub fn from_consciousness(consciousness_level: f32) -> Self {
        match consciousness_level {
            c if c < 0.3 => DreamState::Awake,
            c if c < 0.6 => DreamState::Drowsy,
            c if c < 0.8 => DreamState::Dreaming,
            _ => DreamState::DeepSleep,
        }
    }

    /// Check if a consolidation mode is allowed in this dream state
    pub fn allows_mode(&self, mode: ConsolidationMode) -> bool {
        match (self, mode) {
            // Awake: only light mode
            (DreamState::Awake, ConsolidationMode::Light) => true,
            (DreamState::Awake, _) => false,

            // Drowsy: light and deep
            (DreamState::Drowsy, ConsolidationMode::Light) => true,
            (DreamState::Drowsy, ConsolidationMode::Deep) => true,
            (DreamState::Drowsy, ConsolidationMode::Rem) => false,

            // Dreaming/DeepSleep: all modes
            (DreamState::Dreaming, _) => true,
            (DreamState::DeepSleep, _) => true,
        }
    }
}
```

### 13-Embedder Teleological Array Integration

The consolidation system leverages specific embedders from the 13-embedding teleological array for different operations:

| Operation | Embedder | Purpose |
|-----------|----------|---------|
| **Deduplication** | E1_Semantic | Content similarity for finding near-duplicate memories |
| **Relationship Preservation** | E3_Causal | Maintaining cause-effect links during merging |
| **Salience Scoring** | E7_Teleological | Goal alignment determines memory importance |
| **Pattern Discovery** | E10_Behavioral | Discovering recurring action patterns in REM mode |
| **Sparse Clustering** | E13_SPLADE | Sparse lexical matching for efficient clustering |

```rust
// crates/context-graph-mcp/src/handlers/consolidate.rs

/// Embedder usage by consolidation mode
impl ConsolidateHandler {
    /// Get embedders used for each mode
    fn embedders_for_mode(mode: ConsolidationMode) -> Vec<EmbedderType> {
        match mode {
            ConsolidationMode::Light => vec![
                EmbedderType::E1_Semantic,  // Deduplication
            ],
            ConsolidationMode::Deep => vec![
                EmbedderType::E1_Semantic,   // Deduplication
                EmbedderType::E3_Causal,     // Relationship preservation
                EmbedderType::E7_Teleological, // Salience scoring
                EmbedderType::E13_SPLADE,    // Clustering
            ],
            ConsolidationMode::Rem => vec![
                EmbedderType::E1_Semantic,   // Deduplication
                EmbedderType::E3_Causal,     // Relationship preservation
                EmbedderType::E7_Teleological, // Salience + drift detection
                EmbedderType::E10_Behavioral, // Pattern discovery
                EmbedderType::E13_SPLADE,    // Clustering
            ],
        }
    }
}
```

### Claude Code Skill Definition (YAML Frontmatter Format)

```yaml
# .claude/skills/consolidate/SKILL.md
---
name: consolidate
description: Memory consolidation with Light/Deep/REM modes for dreaming and pruning
version: 1.0.0
auto_invoke: false  # Expensive operation, manual trigger only
allowed_tools:
  - mcp__context-graph__consolidate_memories
timeout_ms: 60000
error_handling:
  retry_count: 0
  fallback_response: "Consolidation temporarily unavailable. Try again in 1 minute."
model: sonnet
rate_limit:
  requests_per_minute: 1
  scope: session
---

# Consolidate Skill

## Purpose

Perform memory consolidation to:
- Remove duplicate memories
- Prune low-salience memories
- Discover emergent patterns
- Simulate "dreaming" for pattern reinforcement

## When to Use

- After intensive work sessions
- When memory count is high (>10000)
- Before long breaks or session end
- When searching feels slow

## Modes

### Light Mode (Default)
Quick cleanup:
- Dedup obvious duplicates (>95% similar) using E1_Semantic
- Prune very low salience (<threshold)
- Fast, minimal disruption
- Available: Always (DreamState::Awake+)

### Deep Mode
Thorough cleanup:
- Aggressive dedup (>90% similar) using E1_Semantic
- Full salience scoring using E7_Teleological
- Memory clustering using E13_SPLADE
- Relationship updates using E3_Causal
- Available: DreamState::Drowsy+ (C >= 0.3)

### REM Mode
Pattern discovery (dreaming):
- All Deep operations
- Goal/pattern emergence using E10_Behavioral
- Drift detection using E7_Teleological
- Goal adjustment
- Most compute-intensive
- Available: DreamState::Dreaming+ (C >= 0.6)

## Process

1. Assess memory state (count, staleness)
2. Check consciousness level for available modes
3. Select appropriate mode
4. Run consolidation with dry_run=true first
5. Review impact
6. Execute if acceptable

## MCP Tool Invocation

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "mcp__context-graph__consolidate_memories",
    "arguments": {
      "mode": "light",
      "salience_threshold": 0.3,
      "dry_run": true
    }
  },
  "id": 1
}
```

## Rate Limit

**1 request per minute per session** (SEC-03)

Use sparingly. Schedule at session end via SessionEnd hook.
```

### Claude Code Hook Integration

Configure the SessionEnd hook in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [".claude/hooks/session_end_consolidate.sh"],
        "timeout": 60000
      }
    ]
  }
}
```

### SessionEnd Consolidation Shell Script

```bash
#!/bin/bash
# .claude/hooks/session_end_consolidate.sh
# Runs memory consolidation at session end

set -euo pipefail

# Configuration
CONSOLIDATION_MODE="${CONSOLIDATION_MODE:-light}"
SALIENCE_THRESHOLD="${SALIENCE_THRESHOLD:-0.3}"
DRY_RUN="${DRY_RUN:-false}"
LOG_FILE="${LOG_FILE:-/tmp/context-graph-consolidate.log}"

log() {
    echo "[$(date -Iseconds)] $*" >> "$LOG_FILE"
}

log "Starting session-end consolidation (mode: $CONSOLIDATION_MODE)"

# Build JSON-RPC request
REQUEST=$(cat <<EOF
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "mcp__context-graph__consolidate_memories",
    "arguments": {
      "mode": "$CONSOLIDATION_MODE",
      "salience_threshold": $SALIENCE_THRESHOLD,
      "dry_run": $DRY_RUN
    }
  },
  "id": "session-end-$(date +%s)"
}
EOF
)

# Send to MCP server via stdio (assumes server is running)
if command -v nc &> /dev/null; then
    # Use netcat if available for socket connection
    RESPONSE=$(echo "$REQUEST" | nc -U /tmp/context-graph.sock 2>/dev/null || echo '{"error":"connection failed"}')
elif [ -n "${CONTEXT_GRAPH_MCP_URL:-}" ]; then
    # Use curl for HTTP transport
    RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$REQUEST" \
        "$CONTEXT_GRAPH_MCP_URL" 2>/dev/null || echo '{"error":"http request failed"}')
else
    log "ERROR: No MCP transport available (netcat or CONTEXT_GRAPH_MCP_URL)"
    exit 1
fi

# Parse response
if echo "$RESPONSE" | grep -q '"error"'; then
    ERROR=$(echo "$RESPONSE" | jq -r '.error.message // .error // "unknown error"')
    log "ERROR: Consolidation failed: $ERROR"
    exit 1
fi

# Extract results
PRUNED=$(echo "$RESPONSE" | jq -r '.result.pruned_count // 0')
MERGED=$(echo "$RESPONSE" | jq -r '.result.merged_count // 0')
CLUSTERS=$(echo "$RESPONSE" | jq -r '.result.clusters_discovered // 0')
DURATION=$(echo "$RESPONSE" | jq -r '.result.duration_ms // 0')

log "Consolidation complete: pruned=$PRUNED merged=$MERGED clusters=$CLUSTERS duration=${DURATION}ms"

# Output summary for Claude Code
echo "Memory consolidation completed:"
echo "  - Pruned: $PRUNED memories"
echo "  - Merged: $MERGED duplicates"
echo "  - Clusters: $CLUSTERS discovered"
echo "  - Duration: ${DURATION}ms"
```

### MCP Tool JSON-RPC Protocol

#### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "mcp__context-graph__consolidate_memories",
    "arguments": {
      "mode": "light|deep|rem",
      "salience_threshold": 0.3,
      "max_memories": 0,
      "dry_run": false
    }
  },
  "id": 1
}
```

#### Success Response

```json
{
  "jsonrpc": "2.0",
  "result": {
    "mode": "light",
    "pruned_count": 42,
    "merged_count": 15,
    "clusters_discovered": 0,
    "new_patterns": [],
    "final_memory_count": 1523,
    "duration_ms": 2341,
    "dry_run": false,
    "consciousness": {
      "integration_score": 0.75,
      "reflection_depth": 0.62,
      "differentiation_index": 0.81,
      "consciousness_level": 0.378,
      "dream_state": "Drowsy"
    }
  },
  "id": 1
}
```

#### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Rate limit exceeded",
    "data": {
      "retry_after_ms": 45000,
      "limit": "1 req/min/session"
    }
  },
  "id": 1
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Light mode duration | < 5s |
| Deep mode duration | < 30s |
| REM mode duration | < 60s (SessionEnd timeout) |
| Rate limit | 1 req/min/session |
| Consciousness threshold (Deep) | C >= 0.3 |
| Consciousness threshold (REM) | C >= 0.6 |

## Verification

- [ ] Light mode completes in < 5s
- [ ] Deep mode completes in < 30s
- [ ] REM mode discovers patterns
- [ ] dry_run returns accurate counts without modifying
- [ ] Rate limit enforced (1/min)
- [ ] Pruned memories actually removed
- [ ] Merged memories properly consolidated
- [ ] Consciousness level correctly gates mode availability
- [ ] SessionEnd hook triggers consolidation
- [ ] All 5 embedders (E1, E3, E7, E10, E13) used correctly by mode
- [ ] MCP JSON-RPC format validated

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/consolidate.rs` | MCP tool handler |
| `.claude/skills/consolidate/SKILL.md` | Skill definition (Claude Code YAML format) |
| `.claude/hooks/session_end_consolidate.sh` | SessionEnd hook script |
| Update `crates/context-graph-mcp/src/handlers/mod.rs` | Register handler |
| Update `.claude/settings.json` | Add SessionEnd hook config |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data loss from pruning | Medium | Critical | dry_run first, backups |
| Long consolidation blocks | Low | Medium | Timeout enforcement |
| Pattern discovery wrong | Medium | Low | Confidence thresholds |
| Consciousness miscalculation | Low | Medium | Validate GWT metrics |
| Hook timeout | Low | Low | 60s timeout, graceful degradation |

## Traceability

- Source: Constitution lines 273 (MCP tool), 743-749 (skill)
- Rate limit: SEC-03 (1 req/min)
- Consciousness: GWT Integration (TASK-INTEG-011)
- Embedders: 13-Teleological Array (TASK-LOGIC-004)
