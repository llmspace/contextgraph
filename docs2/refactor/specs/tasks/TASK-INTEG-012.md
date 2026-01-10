# TASK-INTEG-012: PreCompact and SubagentStop Hooks

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-012 |
| **Title** | PreCompact and SubagentStop Hooks |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 32 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- **ARCH-07**: Hook-driven lifecycle (PreCompact, SubagentStop)
- Constitution lines 713-719

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-004 | Hook protocol infrastructure |
| TASK-LOGIC-009 | Goal discovery for salience scoring |
| TASK-LOGIC-010 | Teleological drift detection for 13-embedder array |
| TASK-INTEG-003 | GWT consciousness MCP integration |

## Objective

Implement two missing hooks required by the constitution:
1. **PreCompact** - Extract salient memories before context compaction (10000ms timeout)
2. **SubagentStop** - Merge subagent learnings into main memory (5000ms timeout)

## Context

**Constitution Requirements (lines 713-719):**

```xml
<hook event="PreCompact" required="false">
  Extract salient memories before context compaction.
  Timeout: 10000ms.
</hook>

<hook event="SubagentStop" required="false">
  Merge subagent learnings into main memory.
  Timeout: 5000ms.
</hook>
```

These hooks are currently **MISSING** from the task specifications despite being required by the constitution.

## Claude Code Hook Configuration

PreCompact and SubagentStop are direct Claude Code hook events. Configure in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [".claude/hooks/pre_compact.sh"],
        "timeout": 10000
      }
    ],
    "SubagentStop": [
      {
        "hooks": [".claude/hooks/subagent_stop.sh"],
        "timeout": 5000
      }
    ]
  }
}
```

### Hook Event Payloads

Claude Code passes specific JSON payloads for these events:

**PreCompact Payload:**
```json
{
  "type": "PreCompact",
  "session_id": "session-uuid",
  "conversation_length": 45000,
  "compaction_target": 25000,
  "compaction_ratio": 0.44,
  "timestamp": "2025-01-10T12:00:00Z"
}
```

**SubagentStop Payload:**
```json
{
  "type": "SubagentStop",
  "subagent_id": "subagent-uuid",
  "parent_session_id": "parent-session-uuid",
  "task_summary": "Implemented authentication module",
  "success": true,
  "duration_ms": 45000,
  "memories_created": ["mem-uuid-1", "mem-uuid-2"],
  "timestamp": "2025-01-10T12:05:00Z"
}
```

## Scope

### In Scope

- `PreCompactHandler` implementation
- `SubagentStopHandler` implementation
- Salient memory extraction logic using 13-embedder teleological array
- Subagent learning merge logic with memory consolidation
- GWT consciousness-aware salience scoring
- Shell hook scripts for Claude Code integration
- Timeout enforcement per constitution (PreCompact: 10000ms, SubagentStop: 5000ms)
- Integration with existing hook protocol
- Integration with `consolidate_memories` MCP tool

### Out of Scope

- Context compaction logic (handled by Claude Code)
- Subagent spawning logic (see TASK-INTEG-014)

## 13-Embedder Teleological Array Integration

Salience scoring and learning extraction leverages all 13 embedders for comprehensive memory analysis:

| Embedder | Purpose | Weight |
|----------|---------|--------|
| E1_Semantic | Core meaning extraction | 0.15 |
| E2_Temporal | Time-based relevance | 0.08 |
| E3_Causal | Cause-effect relationships | 0.10 |
| E4_Counterfactual | Alternative reasoning paths | 0.06 |
| E5_Moral | Ethical considerations | 0.05 |
| E6_Aesthetic | Quality and elegance patterns | 0.04 |
| E7_Teleological | Goal alignment scoring | 0.12 |
| E8_Contextual | Situational relevance | 0.10 |
| E9_Structural | Code/document structure | 0.08 |
| E10_Behavioral | Action pattern matching | 0.06 |
| E11_Emotional | Sentiment and emphasis | 0.04 |
| E12_Code | Code-specific semantics | 0.07 |
| E13_SPLADE | Sparse lexical matching | 0.05 |

### RRF Fusion for Salience Scores

```rust
/// Compute final salience using Reciprocal Rank Fusion across all 13 embedders
pub fn compute_rrf_salience(
    memory: &TeleologicalArray,
    context: &CompactionContext,
    k: f32,  // Typically 60.0
) -> f32 {
    let embedder_scores = [
        (E1_Semantic, 0.15, memory.semantic_similarity(context)),
        (E2_Temporal, 0.08, memory.temporal_relevance(context.current_time)),
        (E3_Causal, 0.10, memory.causal_chain_score(context)),
        (E4_Counterfactual, 0.06, memory.counterfactual_value(context)),
        (E5_Moral, 0.05, memory.moral_salience(context)),
        (E6_Aesthetic, 0.04, memory.aesthetic_quality()),
        (E7_Teleological, 0.12, memory.goal_alignment(context.active_goals)),
        (E8_Contextual, 0.10, memory.contextual_fit(context)),
        (E9_Structural, 0.08, memory.structural_importance()),
        (E10_Behavioral, 0.06, memory.behavioral_pattern_match(context)),
        (E11_Emotional, 0.04, memory.emotional_weight()),
        (E12_Code, 0.07, memory.code_relevance(context)),
        (E13_SPLADE, 0.05, memory.lexical_overlap(context)),
    ];

    // RRF: sum of weight * 1/(k + rank) for each embedder
    embedder_scores.iter()
        .map(|(_, weight, score)| weight * (1.0 / (k + (1.0 - score) * 100.0)))
        .sum::<f32>()
}
```

## GWT Consciousness Integration

Consciousness-aware memory extraction adjusts salience thresholds based on integration state:

```rust
/// Consciousness context for compaction decisions
pub struct CompactionConsciousnessContext {
    /// I(t) - how integrated are memories in global workspace
    pub integration_score: f32,
    /// R(t) - depth of reflective understanding
    pub reflection_depth: f32,
    /// D(t) - memory uniqueness/differentiation
    pub differentiation_index: f32,
    /// C(t) = I(t) * R(t) * D(t) - overall consciousness level
    pub consciousness_level: f32,
    /// Derived salience mode based on consciousness level
    pub salience_mode: SalienceMode,
}

impl CompactionConsciousnessContext {
    pub fn compute(gwt_state: &GwtState) -> Self {
        let integration_score = gwt_state.global_workspace_integration();
        let reflection_depth = gwt_state.metacognitive_depth();
        let differentiation_index = gwt_state.memory_differentiation();
        let consciousness_level = integration_score * reflection_depth * differentiation_index;

        let salience_mode = match consciousness_level {
            c if c < 0.3 => SalienceMode::Minimal,
            c if c < 0.6 => SalienceMode::Standard,
            c if c < 0.8 => SalienceMode::Enhanced,
            _ => SalienceMode::Maximum,
        };

        Self {
            integration_score,
            reflection_depth,
            differentiation_index,
            consciousness_level,
            salience_mode,
        }
    }
}

/// Salience mode determines extraction aggressiveness
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SalienceMode {
    /// C < 0.3: Extract only critical memories (threshold: 0.9)
    Minimal,
    /// 0.3 <= C < 0.6: Normal salience threshold (0.7)
    Standard,
    /// 0.6 <= C < 0.8: Lower threshold, more preservation (0.5)
    Enhanced,
    /// C >= 0.8: Preserve maximum context (threshold: 0.3)
    Maximum,
}

impl SalienceMode {
    pub fn threshold(&self) -> f32 {
        match self {
            SalienceMode::Minimal => 0.9,
            SalienceMode::Standard => 0.7,
            SalienceMode::Enhanced => 0.5,
            SalienceMode::Maximum => 0.3,
        }
    }

    pub fn max_memories(&self) -> usize {
        match self {
            SalienceMode::Minimal => 25,
            SalienceMode::Standard => 50,
            SalienceMode::Enhanced => 100,
            SalienceMode::Maximum => 200,
        }
    }
}
```

## Definition of Done

### Shell Hook Scripts

Claude Code executes shell scripts that invoke the MCP tools:

**`.claude/hooks/pre_compact.sh`:**
```bash
#!/bin/bash
# Pre-compact hook - extracts salient memories before context compaction
# Receives JSON payload on stdin from Claude Code

set -e

# Read the hook payload
PAYLOAD=$(cat)

# Extract fields from payload
SESSION_ID=$(echo "$PAYLOAD" | jq -r '.session_id')
COMPACTION_RATIO=$(echo "$PAYLOAD" | jq -r '.compaction_ratio')

# Invoke MCP tool for memory extraction
# The MCP server handles the actual salience scoring with 13-embedder array
echo "$PAYLOAD" | npx context-graph-mcp extract-salient-memories \
    --session-id "$SESSION_ID" \
    --compaction-ratio "$COMPACTION_RATIO" \
    --use-gwt-consciousness \
    --output json

# Exit code 0 allows compaction to proceed
# Non-zero exit would abort compaction
exit 0
```

**`.claude/hooks/subagent_stop.sh`:**
```bash
#!/bin/bash
# Subagent stop hook - merges learnings and consolidates memories
# Receives JSON payload on stdin from Claude Code

set -e

# Read the hook payload
PAYLOAD=$(cat)

# Extract fields from payload
SUBAGENT_ID=$(echo "$PAYLOAD" | jq -r '.subagent_id')
PARENT_SESSION_ID=$(echo "$PAYLOAD" | jq -r '.parent_session_id')
TASK_SUMMARY=$(echo "$PAYLOAD" | jq -r '.task_summary')
SUCCESS=$(echo "$PAYLOAD" | jq -r '.success')
MEMORIES=$(echo "$PAYLOAD" | jq -r '.memories_created | join(",")')

# Step 1: Extract and score learnings
LEARNINGS=$(npx context-graph-mcp extract-learnings \
    --subagent-id "$SUBAGENT_ID" \
    --memories "$MEMORIES" \
    --output json)

# Step 2: Merge into parent session
npx context-graph-mcp merge-learnings \
    --parent-session "$PARENT_SESSION_ID" \
    --learnings "$LEARNINGS" \
    --strategy selective

# Step 3: Consolidate memories for permanent storage
# Integrates with consolidate_memories MCP tool
npx context-graph-mcp consolidate-memories \
    --session-id "$PARENT_SESSION_ID" \
    --source-subagent "$SUBAGENT_ID" \
    --task-summary "$TASK_SUMMARY" \
    --success "$SUCCESS"

exit 0
```

### PreCompact Hook

```rust
// crates/context-graph-mcp/src/hooks/pre_compact.rs

use std::time::Duration;

/// Handler for PreCompact hook
///
/// Triggered before Claude Code compacts context window.
/// Extracts salient memories to preserve important context.
/// Uses 13-embedder teleological array with GWT consciousness integration.
pub struct PreCompactHandler {
    timeout: Duration,
    gwt_client: GwtClient,
    embedder_array: TeleologicalEmbedderArray,
}

impl PreCompactHandler {
    pub fn new(gwt_client: GwtClient, embedder_array: TeleologicalEmbedderArray) -> Self {
        Self {
            timeout: Duration::from_millis(10_000), // Constitution: 10000ms
            gwt_client,
            embedder_array,
        }
    }

    /// Handle PreCompact event from Claude Code
    ///
    /// # Arguments
    /// * `payload` - JSON payload from Claude Code hook event
    ///
    /// # Returns
    /// Salient memories to preserve with consciousness-adjusted thresholds
    pub async fn handle(&self, payload: PreCompactPayload) -> HookResult<PreCompactOutput> {
        tokio::time::timeout(self.timeout, async {
            // 1. Get current consciousness state for salience mode
            let gwt_state = self.gwt_client.get_consciousness_state(&payload.session_id).await?;
            let consciousness_ctx = CompactionConsciousnessContext::compute(&gwt_state);

            // 2. Retrieve active memories for the session
            let active_memories = self.retrieve_active_memories(&payload.session_id).await?;

            // 3. Score each memory using 13-embedder RRF fusion
            let compaction_context = CompactionContext {
                current_time: payload.timestamp,
                compaction_ratio: payload.compaction_ratio,
                active_goals: gwt_state.active_goals.clone(),
            };

            let scored_memories: Vec<(Uuid, f32)> = active_memories.iter()
                .map(|memory| {
                    let score = compute_rrf_salience(memory, &compaction_context, 60.0);
                    (memory.id, score)
                })
                .collect();

            // 4. Apply consciousness-aware threshold
            let threshold = consciousness_ctx.salience_mode.threshold();
            let max_memories = consciousness_ctx.salience_mode.max_memories();

            let salient: Vec<(Uuid, f32)> = scored_memories.into_iter()
                .filter(|(_, score)| *score >= threshold)
                .take(max_memories)
                .collect();

            // 5. Analyze context for summary
            let context_analysis = self.analyze_context(&active_memories, &consciousness_ctx).await?;

            // 6. Create extraction output
            let output = PreCompactOutput {
                extracted_memories: salient,
                context_summary: context_analysis.summary,
                recommended_retention: context_analysis.key_entities,
                consciousness_level: consciousness_ctx.consciousness_level,
                salience_mode: consciousness_ctx.salience_mode,
            };

            Ok(output)
        }).await
        .map_err(|_| HookError::Timeout("PreCompact".into(), 10_000))?
    }

    async fn retrieve_active_memories(&self, session_id: &str) -> HookResult<Vec<TeleologicalArray>>;

    async fn analyze_context(
        &self,
        memories: &[TeleologicalArray],
        consciousness: &CompactionConsciousnessContext,
    ) -> HookResult<ContextAnalysis>;
}

/// Payload received from Claude Code PreCompact hook event
#[derive(Debug, Deserialize)]
pub struct PreCompactPayload {
    /// Event type (always "PreCompact")
    #[serde(rename = "type")]
    pub event_type: String,
    /// Current session ID
    pub session_id: String,
    /// Current conversation length in tokens
    pub conversation_length: usize,
    /// Target length after compaction
    pub compaction_target: usize,
    /// Ratio of context being removed (0.0-1.0)
    pub compaction_ratio: f32,
    /// When the compaction is occurring
    pub timestamp: DateTime<Utc>,
}

/// Context for salience scoring during compaction
#[derive(Debug)]
pub struct CompactionContext {
    /// Current timestamp for temporal scoring
    pub current_time: DateTime<Utc>,
    /// How much context is being removed
    pub compaction_ratio: f32,
    /// Active goals for teleological scoring
    pub active_goals: Vec<Goal>,
}

/// Output from PreCompact handler
#[derive(Debug, Serialize)]
pub struct PreCompactOutput {
    /// Memories to preserve (id -> salience score)
    pub extracted_memories: Vec<(Uuid, f32)>,
    /// Summary of what was important in compacted context
    pub context_summary: String,
    /// Key entities to remember
    pub recommended_retention: Vec<String>,
    /// Current consciousness level C(t)
    pub consciousness_level: f32,
    /// Salience mode derived from consciousness
    pub salience_mode: SalienceMode,
}
```

### SubagentStop Hook

```rust
// crates/context-graph-mcp/src/hooks/subagent_stop.rs

use std::time::Duration;

/// Handler for SubagentStop hook
///
/// Triggered when a subagent completes its work.
/// Merges subagent learnings into main session memory.
/// Integrates with consolidate_memories MCP tool for permanent storage.
pub struct SubagentStopHandler {
    timeout: Duration,
    merge_strategy: MergeStrategy,
    memory_consolidator: MemoryConsolidator,
    embedder_array: TeleologicalEmbedderArray,
}

impl SubagentStopHandler {
    pub fn new(memory_consolidator: MemoryConsolidator, embedder_array: TeleologicalEmbedderArray) -> Self {
        Self {
            timeout: Duration::from_millis(5_000), // Constitution: 5000ms
            merge_strategy: MergeStrategy::Selective,
            memory_consolidator,
            embedder_array,
        }
    }

    /// Handle SubagentStop event from Claude Code
    ///
    /// # Arguments
    /// * `payload` - JSON payload from Claude Code hook event
    ///
    /// # Returns
    /// Merge and consolidation result
    pub async fn handle(&self, payload: SubagentStopPayload) -> HookResult<SubagentStopOutput> {
        tokio::time::timeout(self.timeout, async {
            // 1. Extract learnings from subagent memories
            let learnings = self.extract_learnings(&payload).await?;

            // 2. Score learnings using 13-embedder array for relevance to parent session
            let scored = self.score_learnings_with_embedders(&learnings, &payload.parent_session_id).await?;

            // 3. Filter and prepare for merge based on strategy
            let to_merge = self.select_for_merge(&scored);

            // 4. Perform merge into main session memory
            let merged = self.merge_into_main(&to_merge, &payload.parent_session_id).await?;

            // 5. Consolidate memories for permanent storage via MCP tool
            let consolidation_result = self.memory_consolidator.consolidate(ConsolidationRequest {
                session_id: payload.parent_session_id.clone(),
                source_subagent: payload.subagent_id.clone(),
                memory_ids: merged.clone(),
                task_summary: payload.task_summary.clone(),
                success: payload.success,
            }).await?;

            // 6. Extract goal updates from consolidated learnings
            let updated_goals = self.extract_goal_updates(&merged, &payload).await?;

            let output = SubagentStopOutput {
                merged_count: merged.len(),
                discarded_count: learnings.len() - merged.len(),
                consolidated_count: consolidation_result.consolidated_count,
                summary: self.summarize_learnings(&merged, &payload.task_summary),
                updated_goals,
                consolidation_status: consolidation_result.status,
            };

            Ok(output)
        }).await
        .map_err(|_| HookError::Timeout("SubagentStop".into(), 5_000))?
    }

    async fn extract_learnings(&self, payload: &SubagentStopPayload) -> HookResult<Vec<Learning>>;

    async fn score_learnings_with_embedders(
        &self,
        learnings: &[Learning],
        parent_session: &str,
    ) -> HookResult<Vec<(Learning, f32)>> {
        // Use 13-embedder array for comprehensive relevance scoring
        let parent_context = self.get_parent_context(parent_session).await?;

        let scored: Vec<(Learning, f32)> = learnings.iter()
            .map(|learning| {
                let memory = self.get_memory(&learning.memory_id)?;
                let score = compute_rrf_salience(&memory, &parent_context, 60.0);
                (learning.clone(), score)
            })
            .collect();

        Ok(scored)
    }

    fn select_for_merge(&self, scored: &[(Learning, f32)]) -> Vec<Learning>;

    async fn merge_into_main(
        &self,
        learnings: &[Learning],
        parent_session: &str,
    ) -> HookResult<Vec<Uuid>>;

    fn summarize_learnings(&self, merged: &[Uuid], task_summary: &str) -> String;

    async fn extract_goal_updates(&self, merged: &[Uuid], payload: &SubagentStopPayload) -> HookResult<Vec<GoalUpdate>>;
}

/// Merge strategy options
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Merge all learnings
    All,
    /// Merge only high-relevance learnings (score > 0.7)
    Selective,
    /// Merge only if subagent was successful
    OnSuccess,
    /// No automatic merge (manual consolidation required)
    Manual,
}

/// Payload received from Claude Code SubagentStop hook event
#[derive(Debug, Deserialize)]
pub struct SubagentStopPayload {
    /// Event type (always "SubagentStop")
    #[serde(rename = "type")]
    pub event_type: String,
    /// ID of the stopping subagent
    pub subagent_id: String,
    /// Parent session to merge into
    pub parent_session_id: String,
    /// Summary of what the subagent was working on
    pub task_summary: String,
    /// Whether subagent completed successfully
    pub success: bool,
    /// Duration of subagent execution in milliseconds
    pub duration_ms: u64,
    /// UUIDs of memories created during subagent execution
    pub memories_created: Vec<String>,
    /// When the subagent stopped
    pub timestamp: DateTime<Utc>,
}

/// Output from SubagentStop handler
#[derive(Debug, Serialize)]
pub struct SubagentStopOutput {
    /// How many memories were merged
    pub merged_count: usize,
    /// How many were discarded
    pub discarded_count: usize,
    /// How many memories were consolidated for permanent storage
    pub consolidated_count: usize,
    /// Human-readable summary of what was learned
    pub summary: String,
    /// Goal updates derived from learnings
    pub updated_goals: Vec<GoalUpdate>,
    /// Status of memory consolidation
    pub consolidation_status: ConsolidationStatus,
}

/// Request to consolidate memories via MCP tool
#[derive(Debug)]
pub struct ConsolidationRequest {
    pub session_id: String,
    pub source_subagent: String,
    pub memory_ids: Vec<Uuid>,
    pub task_summary: String,
    pub success: bool,
}

/// Result from memory consolidation
#[derive(Debug)]
pub struct ConsolidationResult {
    pub consolidated_count: usize,
    pub status: ConsolidationStatus,
}

#[derive(Debug, Clone, Serialize)]
pub enum ConsolidationStatus {
    Success,
    PartialSuccess { failed_ids: Vec<Uuid> },
    Failed { error: String },
    Skipped { reason: String },
}

/// A learning from subagent to potentially merge
#[derive(Debug)]
pub struct Learning {
    pub memory_id: Uuid,
    pub learning_type: LearningType,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum LearningType {
    NewPattern,
    ConfirmedKnowledge,
    CorrectedMistake,
    NewEntity,
    CausalLink,
}

#[derive(Debug)]
pub struct GoalUpdate {
    pub goal_id: Uuid,
    pub update_type: GoalUpdateType,
    pub delta: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum GoalUpdateType {
    Progress,
    Completion,
    Refinement,
    Abandonment,
}
```

### Hook Registration

```rust
// Update crates/context-graph-mcp/src/hooks/mod.rs

pub mod pre_compact;
pub mod subagent_stop;
pub mod consciousness;

pub use pre_compact::{
    PreCompactHandler, PreCompactPayload, PreCompactOutput,
    CompactionContext, CompactionConsciousnessContext, SalienceMode,
};
pub use subagent_stop::{
    SubagentStopHandler, SubagentStopPayload, SubagentStopOutput,
    ConsolidationRequest, ConsolidationResult, ConsolidationStatus,
};

/// Extended hook dispatcher with new hooks and consciousness integration
impl HookDispatcher {
    pub async fn dispatch_pre_compact(
        &self,
        payload: PreCompactPayload,
    ) -> HookResult<PreCompactOutput> {
        self.pre_compact_handler.handle(payload).await
    }

    pub async fn dispatch_subagent_stop(
        &self,
        payload: SubagentStopPayload,
    ) -> HookResult<SubagentStopOutput> {
        self.subagent_stop_handler.handle(payload).await
    }
}
```

### Memory Consolidation MCP Integration

The SubagentStop hook integrates with the `consolidate_memories` MCP tool for permanent storage:

```rust
/// Memory consolidator wraps the MCP consolidate_memories tool
pub struct MemoryConsolidator {
    mcp_client: McpClient,
}

impl MemoryConsolidator {
    pub async fn consolidate(&self, request: ConsolidationRequest) -> HookResult<ConsolidationResult> {
        // Call MCP tool: consolidate_memories
        let result = self.mcp_client.call_tool(
            "consolidate_memories",
            json!({
                "session_id": request.session_id,
                "memory_ids": request.memory_ids,
                "source": {
                    "type": "subagent",
                    "subagent_id": request.source_subagent,
                    "task_summary": request.task_summary,
                    "success": request.success,
                },
                "consolidation_type": "subagent_merge",
            }),
        ).await?;

        // Parse result
        Ok(ConsolidationResult {
            consolidated_count: result.consolidated_count,
            status: match result.status.as_str() {
                "success" => ConsolidationStatus::Success,
                "partial" => ConsolidationStatus::PartialSuccess {
                    failed_ids: result.failed_ids,
                },
                "failed" => ConsolidationStatus::Failed {
                    error: result.error,
                },
                _ => ConsolidationStatus::Skipped {
                    reason: result.reason,
                },
            },
        })
    }
}
```

### Constraints

| Constraint | Target | Source |
|------------|--------|--------|
| PreCompact timeout | 10000ms | Constitution line 715 |
| SubagentStop timeout | 5000ms | Constitution line 718 |
| PreCompact max memories (Minimal) | 25 | GWT consciousness C < 0.3 |
| PreCompact max memories (Standard) | 50 | GWT consciousness 0.3 <= C < 0.6 |
| PreCompact max memories (Enhanced) | 100 | GWT consciousness 0.6 <= C < 0.8 |
| PreCompact max memories (Maximum) | 200 | GWT consciousness C >= 0.8 |
| Merge relevance threshold | 0.7 | Selective merge strategy |
| Embedder count | 13 | Teleological array (E1-E13) |
| RRF k parameter | 60.0 | Standard RRF fusion |

## Verification

### PreCompact Hook Verification

- [ ] PreCompact hook triggers before context compaction
- [ ] PreCompact completes within 10000ms timeout
- [ ] Shell script `.claude/hooks/pre_compact.sh` is executable
- [ ] Claude Code settings.json correctly configures PreCompact hook
- [ ] JSON payload received correctly from Claude Code
- [ ] GWT consciousness state retrieved successfully
- [ ] Consciousness level C(t) computed correctly as I(t) * R(t) * D(t)
- [ ] SalienceMode correctly derived from consciousness level
- [ ] All 13 embedders contribute to RRF salience scoring
- [ ] RRF fusion produces normalized salience scores
- [ ] Salient memories extracted above consciousness-adjusted threshold
- [ ] Output correctly serialized as JSON response

### SubagentStop Hook Verification

- [ ] SubagentStop hook triggers on subagent completion
- [ ] SubagentStop completes within 5000ms timeout
- [ ] Shell script `.claude/hooks/subagent_stop.sh` is executable
- [ ] Claude Code settings.json correctly configures SubagentStop hook
- [ ] JSON payload received correctly including subagent_id and task_summary
- [ ] Learnings extracted from subagent memories
- [ ] 13-embedder scoring applied to learnings
- [ ] Selective merge strategy filters low-relevance learnings
- [ ] Learnings merged into parent session correctly
- [ ] `consolidate_memories` MCP tool invoked for permanent storage
- [ ] Consolidation result correctly captured
- [ ] Goal updates extracted from merged learnings
- [ ] Failed merge doesn't block subagent cleanup
- [ ] Output correctly serialized as JSON response

### Integration Verification

- [ ] Both hooks integrate with existing hook protocol (TASK-INTEG-004)
- [ ] Goal discovery (TASK-LOGIC-009) provides active goals for teleological scoring
- [ ] Teleological drift detection (TASK-LOGIC-010) provides 13-embedder infrastructure
- [ ] GWT consciousness integration (TASK-INTEG-003) provides consciousness state

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/hooks/pre_compact.rs` | PreCompact handler |
| `crates/context-graph-mcp/src/hooks/subagent_stop.rs` | SubagentStop handler |
| `crates/context-graph-mcp/src/hooks/consciousness.rs` | GWT consciousness context |
| `.claude/hooks/pre_compact.sh` | Shell script for Claude Code PreCompact hook |
| `.claude/hooks/subagent_stop.sh` | Shell script for Claude Code SubagentStop hook |
| Update `crates/context-graph-mcp/src/hooks/mod.rs` | Export new handlers |
| Update `.claude/settings.json` | Add hook configuration |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Timeout too short for large context | Medium | Medium | GWT consciousness adjusts memory limits; log slow operations |
| Merge conflicts | Low | Low | Idempotent merge operations with conflict resolution |
| Memory bloat from merge | Low | Medium | Selective merge strategy with consciousness-aware limits |
| GWT state unavailable | Low | High | Fallback to Standard salience mode |
| Embedder failure | Low | Medium | Graceful degradation: use subset of available embedders |
| Shell script execution failure | Low | High | Validate scripts on startup; provide clear error messages |
| MCP consolidate_memories timeout | Medium | Medium | Retry logic with exponential backoff |

## Traceability

- Source: Constitution claude_code_integration/hooks (lines 713-719)
- Depends on: TASK-INTEG-004 (Hook protocol infrastructure)
- Depends on: TASK-LOGIC-009 (Goal discovery for salience scoring)
- Depends on: TASK-LOGIC-010 (Teleological drift detection for 13-embedder array)
- Depends on: TASK-INTEG-003 (GWT consciousness MCP integration)
- Related: consolidate_memories MCP tool for permanent memory storage
