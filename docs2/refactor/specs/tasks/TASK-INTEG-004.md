# TASK-INTEG-004: Hook Protocol and Core Handlers

```xml
<task_spec id="TASK-INTEG-004" version="2.0">
<metadata>
  <title>Implement Hook Protocol and Core Handlers (Claude Code Optimized)</title>
  <status>todo</status>
  <layer>integration</layer>
  <sequence>24</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-PROTOCOL-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-PRETASK-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-POSTTASK-01</requirement_ref>
    <requirement_ref>ARCH-07</requirement_ref><!-- Hooks Control Memory Lifecycle -->
  </implements>
  <depends_on>
    <task_ref>TASK-INTEG-001</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_days>4</estimated_days>
</metadata>

<context>
Hook handlers integrate with Claude Code's official hooks system to enable automatic memory
injection, context loading, and learning from operations. This implements core handlers for
all Claude Code hook events while maintaining compliance with the 13-embedding teleological
array and GWT consciousness model.

**Claude Code Hook System (2025-01+):**
- Hooks are configured in `settings.json` or executed via `.claude/hooks/` shell scripts
- Hook events: PreToolUse, PostToolUse, Notification, Stop, SubagentStop, UserPromptSubmit,
  SessionStart, SessionEnd, PreCompact
- Hooks receive JSON input via stdin and can return JSON output via stdout
- Tool-specific hooks use `tool_name` matcher patterns

**Constitution Alignment:**
- ARCH-07: Hooks MUST drive memory operations (SessionStart→init, PreToolUse→inject,
  PostToolUse→store, SessionEnd→consolidate)
- GWT consciousness model: C(t) = I(t) × R(t) × D(t)
- 13-embedding teleological array integration
</context>

<objective>
Implement the hook protocol and core handlers for all Claude Code hook events that trigger
teleological array operations, with proper timeout handling and GWT consciousness integration.
</objective>

<rationale>
Hooks enable automatic learning per ARCH-07:
1. SessionStart: Initialize workspace, load SELF_EGO_NODE, warm caches (5000ms timeout)
2. PreToolUse: Inject relevant context before tool execution (3000ms timeout)
3. PostToolUse: Store outcomes and train patterns (3000ms timeout)
4. SessionEnd: Memory consolidation and goal discovery (60000ms timeout)
5. PreCompact: Extract salient memories before context compaction
6. SubagentStop: Merge subagent learnings into main memory
</rationale>

<input_context_files>
  <file purpose="hook_spec">docs2/constitution.yaml#claude_code_integration.hooks</file>
  <file purpose="gwt_spec">docs2/constitution.yaml#gwt</file>
  <file purpose="task_breakdown">docs2/refactor/07-TASK-BREAKDOWN.md#phase-4</file>
</input_context_files>

<prerequisites>
  <check>TASK-INTEG-001 complete (MemoryHandler exists)</check>
  <check>TASK-INTEG-003 complete (GWT ConsciousnessHandler exists)</check>
</prerequisites>

<scope>
  <in_scope>
    <item>Define ClaudeCodeHookEvent enum matching official Claude Code events</item>
    <item>Define HookPayload and HookResponse types</item>
    <item>Implement hooks/session-start handler with SELF_EGO_NODE loading</item>
    <item>Implement hooks/session-end handler with consolidation</item>
    <item>Implement hooks/pre-tool-use handler with context injection</item>
    <item>Implement hooks/post-tool-use handler with pattern learning</item>
    <item>Implement hooks/pre-compact handler for memory extraction</item>
    <item>Implement hooks/subagent-stop handler for learning merge</item>
    <item>Trajectory tracking for learning across 13 embedders (E1-E13)</item>
    <item>GWT consciousness state integration C(t) = I(t) × R(t) × D(t)</item>
    <item>Shell script templates in .claude/hooks/</item>
    <item>settings.json hook configuration examples</item>
  </in_scope>
  <out_of_scope>
    <item>Edit hooks (TASK-INTEG-005)</item>
    <item>File operation hooks (TASK-INTEG-006)</item>
    <item>Background workers (TASK-INTEG-007)</item>
  </out_of_scope>
</scope>

<claude_code_hook_configuration>
  <!-- settings.json hook configuration format -->
  <settings_json_example>
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": ".claude/hooks/session-start.sh"
      }
    ],
    "PreToolUse": [
      {
        "type": "command",
        "command": ".claude/hooks/pre-tool-use.sh",
        "tool_name": "Read|Grep|Glob|Bash"
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": ".claude/hooks/post-tool-use.sh",
        "tool_name": "Edit|Write|Bash"
      }
    ],
    "SessionEnd": [
      {
        "type": "command",
        "command": ".claude/hooks/session-end.sh"
      }
    ],
    "PreCompact": [
      {
        "type": "command",
        "command": ".claude/hooks/pre-compact.sh"
      }
    ],
    "SubagentStop": [
      {
        "type": "command",
        "command": ".claude/hooks/subagent-stop.sh"
      }
    ]
  }
}
  </settings_json_example>

  <!-- Timeout requirements per constitution -->
  <timeouts>
    <timeout event="SessionStart" ms="5000" on_failure="log warning, continue session"/>
    <timeout event="PreToolUse" ms="3000" on_failure="continue without injection"/>
    <timeout event="PostToolUse" ms="3000" async_allowed="true"/>
    <timeout event="SessionEnd" ms="60000" mode="deep dreaming"/>
    <timeout event="PreCompact" ms="10000"/>
    <timeout event="SubagentStop" ms="5000"/>
  </timeouts>
</claude_code_hook_configuration>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-mcp/src/hooks/protocol.rs">
      use serde::{Deserialize, Serialize};

      /// Claude Code hook event types matching official specification.
      /// Reference: Claude Code documentation 2025-01+
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
      #[serde(rename_all = "PascalCase")]
      pub enum ClaudeCodeHookEvent {
          /// Fired before a tool is executed.
          /// Can inject context into tool parameters.
          PreToolUse,
          /// Fired after a tool completes execution.
          /// Used for pattern learning and outcome storage.
          PostToolUse,
          /// Fired when Claude produces a notification.
          Notification,
          /// Fired when the main agent stops.
          Stop,
          /// Fired when a subagent stops.
          /// Used to merge subagent learnings.
          SubagentStop,
          /// Fired when user submits a prompt.
          UserPromptSubmit,
          /// Fired at session start.
          /// Initializes workspace, loads SELF_EGO_NODE.
          SessionStart,
          /// Fired at session end.
          /// Triggers memory consolidation and goal discovery.
          SessionEnd,
          /// Fired before context compaction.
          /// Extracts salient memories before truncation.
          PreCompact,
      }

      impl ClaudeCodeHookEvent {
          /// Get the timeout in milliseconds per constitution.yaml
          pub fn timeout_ms(&self) -> u64 {
              match self {
                  Self::SessionStart => 5000,
                  Self::PreToolUse => 3000,
                  Self::PostToolUse => 3000,
                  Self::SessionEnd => 60000,
                  Self::PreCompact => 10000,
                  Self::SubagentStop => 5000,
                  Self::Notification => 1000,
                  Self::Stop => 5000,
                  Self::UserPromptSubmit => 1000,
              }
          }
      }

      /// Payload sent with hook invocations (from Claude Code via stdin).
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct HookPayload {
          pub event: ClaudeCodeHookEvent,
          pub session_id: String,
          pub timestamp: DateTime<Utc>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub tool_name: Option<String>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub tool_input: Option<serde_json::Value>,
          pub context: HookContext,
      }

      /// Context data specific to each hook type.
      #[derive(Debug, Clone, Serialize, Deserialize)]
      #[serde(tag = "type", rename_all = "snake_case")]
      pub enum HookContext {
          Session(SessionContext),
          ToolUse(ToolUseContext),
          Notification(NotificationContext),
          Compact(CompactContext),
          Subagent(SubagentContext),
      }

      /// Context for SessionStart/SessionEnd hooks.
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct SessionContext {
          pub session_id: String,
          pub namespace: Option<String>,
          pub previous_session_id: Option<String>,
          pub metadata: Option<serde_json::Value>,
          /// GWT consciousness state if available
          pub consciousness_state: Option<ConsciousnessState>,
      }

      /// Context for PreToolUse/PostToolUse hooks.
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct ToolUseContext {
          pub tool_name: String,
          pub tool_input: serde_json::Value,
          pub result: Option<ToolResult>,
          /// Teleological alignment per embedder E1-E13
          pub embedder_alignments: Option<[f32; 13]>,
      }

      /// Context for PreCompact hooks.
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct CompactContext {
          pub session_id: String,
          pub context_tokens: usize,
          pub max_tokens: usize,
          pub memories_to_preserve: Vec<String>,
      }

      /// Context for SubagentStop hooks.
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct SubagentContext {
          pub subagent_id: String,
          pub parent_session_id: String,
          pub learnings: Vec<SubagentLearning>,
      }

      /// Response returned from hook handlers (to Claude Code via stdout).
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct HookResponse {
          pub success: bool,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub context_recommendations: Option<Vec<ContextRecommendation>>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub trajectory_id: Option<String>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub consciousness: Option<ConsciousnessMetrics>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub metrics: Option<HookMetrics>,
          #[serde(skip_serializing_if = "Option::is_none")]
          pub error: Option<String>,
      }

      /// GWT consciousness metrics: C(t) = I(t) × R(t) × D(t)
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct ConsciousnessMetrics {
          /// C(t): Overall consciousness score [0,1]
          pub consciousness: f32,
          /// I(t): Integration (Kuramoto order parameter r)
          pub integration: f32,
          /// R(t): Self-Reflection (MetaUTL predict accuracy)
          pub reflection: f32,
          /// D(t): Differentiation (13D entropy H(PV))
          pub differentiation: f32,
          /// Current consciousness state
          pub state: ConsciousnessState,
      }

      /// GWT consciousness states based on Kuramoto order parameter r.
      #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
      #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
      pub enum ConsciousnessState {
          /// r < 0.3: Minimal integration
          Dormant,
          /// 0.3 ≤ r < 0.5: Partial integration
          Fragmented,
          /// 0.5 ≤ r < 0.8: Building coherence
          Emerging,
          /// r ≥ 0.8: Full consciousness
          Conscious,
          /// r > 0.95: Pathological over-synchronization
          Hypersync,
      }
    </signature>

    <signature file="crates/context-graph-mcp/src/hooks/session.rs">
      use std::sync::Arc;

      /// Session lifecycle handler per ARCH-07.
      /// Handles SessionStart and SessionEnd hooks.
      pub struct SessionHandler {
          memory_handler: Arc<MemoryHandler>,
          consciousness_handler: Arc<ConsciousnessHandler>,
          session_store: Arc<SessionStore>,
          consolidator: Arc<MemoryConsolidator>,
          ego_node_id: Option<Uuid>,
      }

      impl SessionHandler {
          pub fn new(
              memory_handler: Arc<MemoryHandler>,
              consciousness_handler: Arc<ConsciousnessHandler>,
              session_store: Arc<SessionStore>,
              consolidator: Arc<MemoryConsolidator>,
          ) -> Self;

          /// Handle SessionStart hook (5000ms timeout).
          /// Actions per constitution:
          /// 1. Initialize workspace
          /// 2. Load SELF_EGO_NODE
          /// 3. Warm embedding caches
          pub async fn handle_start(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Handle SessionEnd hook (60000ms timeout).
          /// Actions per constitution:
          /// 1. Run memory consolidation
          /// 2. Goal discovery
          pub async fn handle_end(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Load SELF_EGO_NODE from memory.
          async fn load_ego_node(&self) -> Result<Option<EgoNode>, HandlerError>;

          /// Compute consciousness metrics C(t) = I(t) × R(t) × D(t).
          async fn compute_consciousness(&self) -> ConsciousnessMetrics;
      }
    </signature>

    <signature file="crates/context-graph-mcp/src/hooks/tool_use.rs">
      /// Tool use hook handler.
      /// Handles PreToolUse and PostToolUse hooks.
      pub struct ToolUseHandler {
          memory_handler: Arc<MemoryHandler>,
          trajectory_tracker: Arc<TrajectoryTracker>,
          pattern_store: Arc<PatternStore>,
          consciousness_handler: Arc<ConsciousnessHandler>,
      }

      impl ToolUseHandler {
          pub fn new(
              memory_handler: Arc<MemoryHandler>,
              trajectory_tracker: Arc<TrajectoryTracker>,
              pattern_store: Arc<PatternStore>,
              consciousness_handler: Arc<ConsciousnessHandler>,
          ) -> Self;

          /// Handle PreToolUse hook (3000ms timeout).
          /// Actions: Inject relevant context before tool execution.
          /// Matcher pattern: "Read|Grep|Glob|Bash"
          pub async fn handle_pre_tool_use(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Handle PostToolUse hook (3000ms timeout, async allowed).
          /// Actions: Store learned patterns from tool output.
          /// Matcher pattern: "Edit|Write|Bash"
          pub async fn handle_post_tool_use(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Search for context relevant to the tool operation
          /// using all 13 embedders (E1-E13).
          async fn search_relevant_context(
              &self,
              tool_name: &str,
              tool_input: &serde_json::Value,
          ) -> Vec<ContextRecommendation>;

          /// Store pattern with per-embedder alignment scores.
          async fn store_tool_pattern(
              &self,
              context: &ToolUseContext,
              verdict: Verdict,
          ) -> Result<usize, HandlerError>;
      }
    </signature>

    <signature file="crates/context-graph-mcp/src/hooks/compact.rs">
      /// PreCompact hook handler.
      /// Extracts salient memories before context compaction.
      pub struct CompactHandler {
          memory_handler: Arc<MemoryHandler>,
          consciousness_handler: Arc<ConsciousnessHandler>,
      }

      impl CompactHandler {
          pub fn new(
              memory_handler: Arc<MemoryHandler>,
              consciousness_handler: Arc<ConsciousnessHandler>,
          ) -> Self;

          /// Handle PreCompact hook (10000ms timeout).
          /// Actions: Extract salient memories before context compaction.
          pub async fn handle_pre_compact(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Identify memories that should be preserved based on:
          /// - Teleological alignment
          /// - Consciousness relevance
          /// - Recency and importance
          async fn identify_salient_memories(
              &self,
              context: &CompactContext,
          ) -> Vec<MemoryId>;
      }
    </signature>

    <signature file="crates/context-graph-mcp/src/hooks/subagent.rs">
      /// SubagentStop hook handler.
      /// Merges subagent learnings into main memory.
      pub struct SubagentHandler {
          memory_handler: Arc<MemoryHandler>,
          trajectory_tracker: Arc<TrajectoryTracker>,
      }

      impl SubagentHandler {
          pub fn new(
              memory_handler: Arc<MemoryHandler>,
              trajectory_tracker: Arc<TrajectoryTracker>,
          ) -> Self;

          /// Handle SubagentStop hook (5000ms timeout).
          /// Actions: Merge subagent learnings into main memory.
          pub async fn handle_subagent_stop(
              &self,
              payload: HookPayload,
          ) -> Result<HookResponse, HandlerError>;

          /// Merge learnings from subagent trajectory into parent session.
          async fn merge_learnings(
              &self,
              context: &SubagentContext,
          ) -> Result<usize, HandlerError>;
      }
    </signature>

    <signature file="crates/context-graph-mcp/src/hooks/trajectory.rs">
      /// Trajectory tracking for learning across sessions.
      /// Integrates with 13-embedder teleological system.
      pub struct TrajectoryTracker {
          trajectories: RwLock<HashMap<String, Trajectory>>,
          embedder_weights: [f32; 13],
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Trajectory {
          pub id: String,
          pub session_id: String,
          pub task_id: String,
          pub steps: Vec<TrajectoryStep>,
          pub started_at: DateTime<Utc>,
          pub completed_at: Option<DateTime<Utc>>,
          pub verdict: Option<Verdict>,
          /// Per-embedder alignment scores [E1..E13]
          pub embedder_alignments: [f32; 13],
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct TrajectoryStep {
          pub step_id: String,
          pub tool_name: String,
          pub timestamp: DateTime<Utc>,
          pub input_hash: String,
          pub output_summary: Option<String>,
          pub success: bool,
          /// Consciousness state at this step
          pub consciousness_state: ConsciousnessState,
      }

      impl TrajectoryTracker {
          pub fn new() -> Self;

          pub async fn start_trajectory(
              &self,
              session_id: &str,
              task_id: &str,
          ) -> String;

          pub async fn add_step(
              &self,
              trajectory_id: &str,
              step: TrajectoryStep,
          );

          pub async fn complete_trajectory(
              &self,
              trajectory_id: &str,
              verdict: Verdict,
              embedder_alignments: [f32; 13],
          );

          /// Get per-embedder alignment for current trajectory.
          pub async fn get_embedder_alignments(
              &self,
              trajectory_id: &str,
          ) -> Option<[f32; 13]>;
      }
    </signature>
  </signatures>

  <constraints>
    <constraint>SessionStart hook completes within 5000ms</constraint>
    <constraint>PreToolUse hook completes within 3000ms (MUST NOT block >100ms critical path)</constraint>
    <constraint>PostToolUse hook completes within 3000ms (async allowed)</constraint>
    <constraint>SessionEnd hook completes within 60000ms (deep dreaming mode)</constraint>
    <constraint>PreCompact hook completes within 10000ms</constraint>
    <constraint>SubagentStop hook completes within 5000ms</constraint>
    <constraint>All hooks gracefully handle failures (don't break Claude Code)</constraint>
    <constraint>Trajectory tracking persists across sessions</constraint>
    <constraint>Session state serializable for restoration</constraint>
    <constraint>GWT consciousness C(t) computed on every hook invocation</constraint>
    <constraint>13-embedder alignment tracked for pattern learning</constraint>
  </constraints>

  <verification>
    <command>cargo test -p context-graph-mcp hooks::protocol</command>
    <command>cargo test -p context-graph-mcp hooks::session</command>
    <command>cargo test -p context-graph-mcp hooks::tool_use</command>
    <command>cargo test -p context-graph-mcp hooks::compact</command>
    <command>cargo test -p context-graph-mcp hooks::subagent</command>
    <command>cargo test -p context-graph-mcp hooks::trajectory</command>
  </verification>
</definition_of_done>

<pseudo_code>
// ═══════════════════════════════════════════════════════════════════════
// crates/context-graph-mcp/src/hooks/protocol.rs
// ═══════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Claude Code hook event types matching official specification.
/// These MUST match the events defined in Claude Code's hook system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ClaudeCodeHookEvent {
    PreToolUse,
    PostToolUse,
    Notification,
    Stop,
    SubagentStop,
    UserPromptSubmit,
    SessionStart,
    SessionEnd,
    PreCompact,
}

impl ClaudeCodeHookEvent {
    /// Get the timeout in milliseconds per constitution.yaml
    pub fn timeout_ms(&self) -> u64 {
        match self {
            Self::SessionStart => 5000,
            Self::PreToolUse => 3000,
            Self::PostToolUse => 3000,
            Self::SessionEnd => 60000,
            Self::PreCompact => 10000,
            Self::SubagentStop => 5000,
            Self::Notification => 1000,
            Self::Stop => 5000,
            Self::UserPromptSubmit => 1000,
        }
    }

    /// Check if this event allows async processing
    pub fn async_allowed(&self) -> bool {
        matches!(self, Self::PostToolUse | Self::SessionEnd)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookPayload {
    pub event: ClaudeCodeHookEvent,
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<serde_json::Value>,
    pub context: HookContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HookContext {
    Session(SessionContext),
    ToolUse(ToolUseContext),
    Notification(NotificationContext),
    Compact(CompactContext),
    Subagent(SubagentContext),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub session_id: String,
    pub namespace: Option<String>,
    pub previous_session_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub consciousness_state: Option<ConsciousnessState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseContext {
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub result: Option<ToolResult>,
    /// Teleological alignment per embedder E1-E13
    pub embedder_alignments: Option<[f32; 13]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationContext {
    pub message: String,
    pub level: NotificationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactContext {
    pub session_id: String,
    pub context_tokens: usize,
    pub max_tokens: usize,
    pub memories_to_preserve: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentContext {
    pub subagent_id: String,
    pub parent_session_id: String,
    pub learnings: Vec<SubagentLearning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentLearning {
    pub pattern: String,
    pub embedder_alignments: [f32; 13],
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_recommendations: Option<Vec<ContextRecommendation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consciousness: Option<ConsciousnessMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<HookMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// GWT consciousness metrics: C(t) = I(t) × R(t) × D(t)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// C(t): Overall consciousness score [0,1]
    pub consciousness: f32,
    /// I(t): Integration (Kuramoto order parameter r)
    pub integration: f32,
    /// R(t): Self-Reflection (MetaUTL predict accuracy)
    pub reflection: f32,
    /// D(t): Differentiation (13D entropy H(PV))
    pub differentiation: f32,
    /// Current consciousness state
    pub state: ConsciousnessState,
}

impl ConsciousnessMetrics {
    /// Compute consciousness: C(t) = I(t) × R(t) × D(t)
    pub fn compute(integration: f32, reflection: f32, differentiation: f32) -> Self {
        let consciousness = integration * reflection * differentiation;
        let state = ConsciousnessState::from_order_parameter(integration);
        Self {
            consciousness,
            integration,
            reflection,
            differentiation,
            state,
        }
    }
}

/// GWT consciousness states based on Kuramoto order parameter r.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConsciousnessState {
    Dormant,
    Fragmented,
    Emerging,
    Conscious,
    Hypersync,
}

impl ConsciousnessState {
    /// Determine state from Kuramoto order parameter r.
    pub fn from_order_parameter(r: f32) -> Self {
        match r {
            r if r > 0.95 => Self::Hypersync,
            r if r >= 0.8 => Self::Conscious,
            r if r >= 0.5 => Self::Emerging,
            r if r >= 0.3 => Self::Fragmented,
            _ => Self::Dormant,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRecommendation {
    pub memory_id: String,
    pub content_preview: String,
    pub relevance_score: f32,
    /// Which embedders (E1-E13) matched best
    pub matching_embedders: Vec<EmbedderMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderMatch {
    /// Embedder index (0-12 for E1-E13)
    pub embedder_index: usize,
    /// Embedder name (E1_Semantic, E2_Temporal_Recent, etc.)
    pub embedder_name: String,
    /// Similarity score for this embedder
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookMetrics {
    pub processing_time_ms: u64,
    pub memories_accessed: usize,
    pub patterns_matched: usize,
    /// Per-embedder processing times
    pub embedder_times_ms: Option<[u64; 13]>,
}

// ═══════════════════════════════════════════════════════════════════════
// crates/context-graph-mcp/src/hooks/session.rs
// ═══════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use tokio::time::{timeout, Duration};

use crate::protocol::HandlerError;
use crate::handlers::memory::MemoryHandler;
use crate::handlers::consciousness::ConsciousnessHandler;
use super::protocol::*;

/// SELF_EGO_NODE ID constant per constitution
pub const SELF_EGO_NODE_ID: &str = "SELF_EGO_NODE";

pub struct SessionHandler {
    memory_handler: Arc<MemoryHandler>,
    consciousness_handler: Arc<ConsciousnessHandler>,
    session_store: Arc<SessionStore>,
    consolidator: Arc<MemoryConsolidator>,
    ego_node_id: Option<Uuid>,
}

impl SessionHandler {
    pub fn new(
        memory_handler: Arc<MemoryHandler>,
        consciousness_handler: Arc<ConsciousnessHandler>,
        session_store: Arc<SessionStore>,
        consolidator: Arc<MemoryConsolidator>,
    ) -> Self {
        Self {
            memory_handler,
            consciousness_handler,
            session_store,
            consolidator,
            ego_node_id: None,
        }
    }

    /// Handle SessionStart hook (5000ms timeout).
    /// Per constitution.yaml claude_code_integration.hooks.SessionStart:
    /// 1. Initialize workspace
    /// 2. Load SELF_EGO_NODE
    /// 3. Warm embedding caches
    pub async fn handle_start(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::SessionStart.timeout_ms()
        );

        // Wrap in timeout
        let result = timeout(timeout_duration, async {
            self.handle_start_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                // Timeout - log warning, continue session per constitution
                tracing::warn!(
                    session_id = %payload.session_id,
                    "SessionStart hook timed out after 5000ms, continuing session"
                );
                Ok(HookResponse {
                    success: false,
                    error: Some("Hook timed out".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_start_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let session_ctx = match &payload.context {
            HookContext::Session(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected session context",
            )),
        };

        // 1. Initialize session state
        let session = Session {
            id: session_ctx.session_id.clone(),
            namespace: session_ctx.namespace.clone(),
            started_at: Utc::now(),
            previous_session_id: session_ctx.previous_session_id.clone(),
            metadata: session_ctx.metadata.clone(),
        };
        self.session_store.store(session).await
            .map_err(|e| HandlerError::new(ErrorCode::InternalError, e.to_string()))?;

        // 2. Load SELF_EGO_NODE
        let ego_node = self.load_ego_node().await?;

        // 3. Warm embedding caches (background, don't block)
        let memory_handler = self.memory_handler.clone();
        tokio::spawn(async move {
            let _ = memory_handler.warm_caches().await;
        });

        // 4. Compute consciousness metrics
        let consciousness = self.compute_consciousness().await;

        // 5. Load relevant prior context if previous session exists
        let recommendations = if let Some(prev_id) = &session_ctx.previous_session_id {
            self.load_prior_context(prev_id).await.ok()
        } else {
            None
        };

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: recommendations.as_ref().map(|r| r.len()).unwrap_or(0),
            patterns_matched: 0,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: recommendations,
            trajectory_id: None,
            consciousness: Some(consciousness),
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Handle SessionEnd hook (60000ms timeout, deep dreaming mode).
    /// Per constitution.yaml claude_code_integration.hooks.SessionEnd:
    /// 1. Run memory consolidation
    /// 2. Goal discovery
    pub async fn handle_end(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::SessionEnd.timeout_ms()
        );

        let result = timeout(timeout_duration, async {
            self.handle_end_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                tracing::warn!(
                    session_id = %payload.session_id,
                    "SessionEnd hook timed out after 60000ms"
                );
                Ok(HookResponse {
                    success: false,
                    error: Some("Hook timed out during consolidation".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_end_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let session_ctx = match &payload.context {
            HookContext::Session(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected session context",
            )),
        };

        // 1. Run memory consolidation (deep dreaming mode)
        let consolidation_result = self.consolidator
            .consolidate_session(&session_ctx.session_id)
            .await
            .map_err(|e| HandlerError::new(ErrorCode::InternalError, e.to_string()))?;

        // 2. Goal discovery from consolidated patterns
        let discovered_goals = self.discover_goals(&session_ctx.session_id).await?;

        // 3. Update session state
        self.session_store
            .mark_completed(&session_ctx.session_id)
            .await
            .map_err(|e| HandlerError::new(ErrorCode::InternalError, e.to_string()))?;

        // 4. Compute final consciousness metrics
        let consciousness = self.compute_consciousness().await;

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: consolidation_result.original_count,
            patterns_matched: consolidation_result.patterns_extracted + discovered_goals,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: None,
            trajectory_id: None,
            consciousness: Some(consciousness),
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Load SELF_EGO_NODE per constitution gwt.self_ego_node.
    async fn load_ego_node(&self) -> Result<Option<EgoNode>, HandlerError> {
        let search_params = SearchParams {
            query: SELF_EGO_NODE_ID.to_string(),
            strategy: SearchStrategy::ExactMatch,
            limit: Some(1),
            threshold: None,
            options: None,
        };

        let result = self.memory_handler
            .handle_search(search_params)
            .await
            .ok();

        // Parse ego node from result if found
        Ok(result.and_then(|r| r.memories.first().map(|m| {
            EgoNode {
                id: m.memory_id,
                fingerprint: m.fingerprint.clone(),
                purpose_vector: m.purpose_vector.clone(),
            }
        })))
    }

    /// Compute consciousness: C(t) = I(t) × R(t) × D(t)
    async fn compute_consciousness(&self) -> ConsciousnessMetrics {
        let state = self.consciousness_handler.get_state().await
            .unwrap_or_default();

        ConsciousnessMetrics {
            consciousness: state.consciousness,
            integration: state.kuramoto_r,
            reflection: state.meta_score,
            differentiation: state.differentiation,
            state: ConsciousnessState::from_order_parameter(state.kuramoto_r),
        }
    }

    async fn load_prior_context(
        &self,
        prev_session_id: &str,
    ) -> Result<Vec<ContextRecommendation>, HandlerError> {
        let search_params = SearchParams {
            query: "recent session context".to_string(),
            strategy: SearchStrategy::EmbedderGroup {
                group: EmbedderGroup::Temporal,
                embedders: None,
            },
            limit: Some(10),
            threshold: None,
            options: Some(SearchOptions {
                namespace: Some(prev_session_id.to_string()),
                ..Default::default()
            }),
        };

        let result = self.memory_handler
            .handle_search(search_params)
            .await?;

        Ok(result.memories.into_iter()
            .map(|m| ContextRecommendation {
                memory_id: m.memory_id.to_string(),
                content_preview: m.content.chars().take(200).collect(),
                relevance_score: m.overall_similarity,
                matching_embedders: Vec::new(),
            })
            .collect())
    }

    async fn discover_goals(&self, session_id: &str) -> Result<usize, HandlerError> {
        // Goal discovery through clustering on purpose vectors
        // Returns count of discovered goals
        Ok(0) // Placeholder - actual implementation in NORTH-015
    }
}

// ═══════════════════════════════════════════════════════════════════════
// crates/context-graph-mcp/src/hooks/tool_use.rs
// ═══════════════════════════════════════════════════════════════════════

pub struct ToolUseHandler {
    memory_handler: Arc<MemoryHandler>,
    trajectory_tracker: Arc<TrajectoryTracker>,
    pattern_store: Arc<PatternStore>,
    consciousness_handler: Arc<ConsciousnessHandler>,
}

impl ToolUseHandler {
    pub fn new(
        memory_handler: Arc<MemoryHandler>,
        trajectory_tracker: Arc<TrajectoryTracker>,
        pattern_store: Arc<PatternStore>,
        consciousness_handler: Arc<ConsciousnessHandler>,
    ) -> Self {
        Self {
            memory_handler,
            trajectory_tracker,
            pattern_store,
            consciousness_handler,
        }
    }

    /// Handle PreToolUse hook (3000ms timeout).
    /// Matcher: "Read|Grep|Glob|Bash"
    /// Constraint: MUST NOT block for more than 100ms in critical path
    pub async fn handle_pre_tool_use(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::PreToolUse.timeout_ms()
        );

        let result = timeout(timeout_duration, async {
            self.handle_pre_tool_use_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                // Timeout - continue without injection per constitution
                tracing::warn!(
                    session_id = %payload.session_id,
                    tool_name = ?payload.tool_name,
                    "PreToolUse hook timed out, continuing without injection"
                );
                Ok(HookResponse {
                    success: false,
                    error: Some("Hook timed out".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_pre_tool_use_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let tool_ctx = match &payload.context {
            HookContext::ToolUse(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected tool use context",
            )),
        };

        // Start trajectory tracking
        let trajectory_id = self.trajectory_tracker
            .start_trajectory(&payload.session_id, &tool_ctx.tool_name)
            .await;

        // Search for relevant context using all 13 embedders
        let recommendations = self.search_relevant_context(
            &tool_ctx.tool_name,
            &tool_ctx.tool_input,
        ).await;

        // Get consciousness state
        let consciousness = self.compute_consciousness().await;

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: recommendations.len(),
            patterns_matched: 0,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: Some(recommendations),
            trajectory_id: Some(trajectory_id),
            consciousness: Some(consciousness),
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Handle PostToolUse hook (3000ms timeout, async allowed).
    /// Matcher: "Edit|Write|Bash"
    pub async fn handle_post_tool_use(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::PostToolUse.timeout_ms()
        );

        let result = timeout(timeout_duration, async {
            self.handle_post_tool_use_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                // Async allowed - spawn background task
                let handler = self.clone();
                let payload_clone = payload.clone();
                tokio::spawn(async move {
                    let _ = handler.handle_post_tool_use_inner(&payload_clone).await;
                });

                Ok(HookResponse {
                    success: true,
                    error: Some("Processing in background".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_post_tool_use_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let tool_ctx = match &payload.context {
            HookContext::ToolUse(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected tool use context",
            )),
        };

        // Determine verdict from result
        let verdict = match &tool_ctx.result {
            Some(result) if result.success => Verdict::Success,
            Some(_) => Verdict::Failure,
            None => Verdict::Partial,
        };

        // Complete trajectory
        let trajectory_id = format!("{}_{}", payload.session_id, tool_ctx.tool_name);
        let embedder_alignments = tool_ctx.embedder_alignments.unwrap_or([0.0; 13]);
        self.trajectory_tracker
            .complete_trajectory(&trajectory_id, verdict.clone(), embedder_alignments)
            .await;

        // Store pattern with per-embedder alignment scores
        let patterns_matched = self.store_tool_pattern(tool_ctx, verdict).await?;

        // Get consciousness state
        let consciousness = self.compute_consciousness().await;

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: 1,
            patterns_matched,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: None,
            trajectory_id: Some(trajectory_id),
            consciousness: Some(consciousness),
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Search for context using all 13 embedders (E1-E13).
    async fn search_relevant_context(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> Vec<ContextRecommendation> {
        // Build query from tool context
        let query = format!(
            "tool:{} input:{}",
            tool_name,
            serde_json::to_string(tool_input).unwrap_or_default()
        );

        let search_params = SearchParams {
            query,
            strategy: SearchStrategy::AutoDiscover {
                max_entry_points: Some(3),
                min_confidence: Some(0.6),
            },
            limit: Some(5),
            threshold: Some(0.5),
            options: None,
        };

        let result = self.memory_handler
            .handle_search(search_params)
            .await
            .unwrap_or_default();

        result.memories.into_iter()
            .map(|m| ContextRecommendation {
                memory_id: m.memory_id.to_string(),
                content_preview: m.content.chars().take(200).collect(),
                relevance_score: m.overall_similarity,
                matching_embedders: m.entry_point_hits.as_ref()
                    .map(|h| h.iter()
                        .enumerate()
                        .filter(|(_, score)| **score > 0.5)
                        .map(|(i, score)| EmbedderMatch {
                            embedder_index: i,
                            embedder_name: embedder_name(i),
                            score: *score,
                        })
                        .collect())
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// Store pattern with per-embedder alignment scores.
    async fn store_tool_pattern(
        &self,
        tool_ctx: &ToolUseContext,
        verdict: Verdict,
    ) -> Result<usize, HandlerError> {
        if verdict != Verdict::Success {
            return Ok(0);
        }

        let Some(result) = &tool_ctx.result else {
            return Ok(0);
        };

        let Some(output) = &result.output else {
            return Ok(0);
        };

        let pattern = Pattern {
            tool: tool_ctx.tool_name.clone(),
            description: format!("{:?}", tool_ctx.tool_input),
            output_summary: output.chars().take(500).collect(),
            success: true,
            embedder_alignments: tool_ctx.embedder_alignments.unwrap_or([0.0; 13]),
            created_at: Utc::now(),
        };

        self.pattern_store.store(pattern).await
            .map(|_| 1)
            .map_err(|e| HandlerError::new(ErrorCode::InternalError, e.to_string()))
    }

    async fn compute_consciousness(&self) -> ConsciousnessMetrics {
        let state = self.consciousness_handler.get_state().await
            .unwrap_or_default();

        ConsciousnessMetrics {
            consciousness: state.consciousness,
            integration: state.kuramoto_r,
            reflection: state.meta_score,
            differentiation: state.differentiation,
            state: ConsciousnessState::from_order_parameter(state.kuramoto_r),
        }
    }
}

/// Map embedder index (0-12) to name (E1-E13).
fn embedder_name(index: usize) -> String {
    const NAMES: [&str; 13] = [
        "E1_Semantic",
        "E2_Temporal_Recent",
        "E3_Temporal_Periodic",
        "E4_Temporal_Positional",
        "E5_Causal",
        "E6_Sparse",
        "E7_Code",
        "E8_Graph_MiniLM",
        "E9_HDC",
        "E10_Multimodal",
        "E11_Entity_MiniLM",
        "E12_LateInteraction",
        "E13_SPLADE",
    ];
    NAMES.get(index).unwrap_or(&"Unknown").to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// crates/context-graph-mcp/src/hooks/compact.rs
// ═══════════════════════════════════════════════════════════════════════

pub struct CompactHandler {
    memory_handler: Arc<MemoryHandler>,
    consciousness_handler: Arc<ConsciousnessHandler>,
}

impl CompactHandler {
    pub fn new(
        memory_handler: Arc<MemoryHandler>,
        consciousness_handler: Arc<ConsciousnessHandler>,
    ) -> Self {
        Self {
            memory_handler,
            consciousness_handler,
        }
    }

    /// Handle PreCompact hook (10000ms timeout).
    /// Actions: Extract salient memories before context compaction.
    pub async fn handle_pre_compact(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::PreCompact.timeout_ms()
        );

        let result = timeout(timeout_duration, async {
            self.handle_pre_compact_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                tracing::warn!(
                    session_id = %payload.session_id,
                    "PreCompact hook timed out after 10000ms"
                );
                Ok(HookResponse {
                    success: false,
                    error: Some("Hook timed out".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_pre_compact_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let compact_ctx = match &payload.context {
            HookContext::Compact(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected compact context",
            )),
        };

        // Identify salient memories to preserve
        let salient_ids = self.identify_salient_memories(compact_ctx).await;

        // Build recommendations for memories to preserve
        let recommendations: Vec<ContextRecommendation> = salient_ids.iter()
            .filter_map(|id| {
                // Lookup memory content
                Some(ContextRecommendation {
                    memory_id: id.to_string(),
                    content_preview: String::new(), // Filled by memory lookup
                    relevance_score: 1.0,
                    matching_embedders: Vec::new(),
                })
            })
            .collect();

        // Get consciousness state
        let consciousness = self.compute_consciousness().await;

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: salient_ids.len(),
            patterns_matched: 0,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: Some(recommendations),
            trajectory_id: None,
            consciousness: Some(consciousness),
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Identify memories that should be preserved based on:
    /// - Teleological alignment (purpose vector scores)
    /// - Consciousness relevance (Kuramoto r threshold)
    /// - Recency and importance
    async fn identify_salient_memories(
        &self,
        context: &CompactContext,
    ) -> Vec<Uuid> {
        // Start with explicitly requested memories
        let mut salient: Vec<Uuid> = context.memories_to_preserve.iter()
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        // Add memories with high purpose vector alignment
        let search_params = SearchParams {
            query: "high alignment salient memories".to_string(),
            strategy: SearchStrategy::EmbedderGroup {
                group: EmbedderGroup::All,
                embedders: None,
            },
            limit: Some(20),
            threshold: Some(0.75), // OPTIMAL threshold per constitution
            options: None,
        };

        if let Ok(result) = self.memory_handler.handle_search(search_params).await {
            for memory in result.memories {
                if !salient.contains(&memory.memory_id) {
                    salient.push(memory.memory_id);
                }
            }
        }

        salient
    }

    async fn compute_consciousness(&self) -> ConsciousnessMetrics {
        let state = self.consciousness_handler.get_state().await
            .unwrap_or_default();

        ConsciousnessMetrics {
            consciousness: state.consciousness,
            integration: state.kuramoto_r,
            reflection: state.meta_score,
            differentiation: state.differentiation,
            state: ConsciousnessState::from_order_parameter(state.kuramoto_r),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// crates/context-graph-mcp/src/hooks/subagent.rs
// ═══════════════════════════════════════════════════════════════════════

pub struct SubagentHandler {
    memory_handler: Arc<MemoryHandler>,
    trajectory_tracker: Arc<TrajectoryTracker>,
}

impl SubagentHandler {
    pub fn new(
        memory_handler: Arc<MemoryHandler>,
        trajectory_tracker: Arc<TrajectoryTracker>,
    ) -> Self {
        Self {
            memory_handler,
            trajectory_tracker,
        }
    }

    /// Handle SubagentStop hook (5000ms timeout).
    /// Actions: Merge subagent learnings into main memory.
    pub async fn handle_subagent_stop(
        &self,
        payload: HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(
            ClaudeCodeHookEvent::SubagentStop.timeout_ms()
        );

        let result = timeout(timeout_duration, async {
            self.handle_subagent_stop_inner(&payload).await
        }).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                tracing::warn!(
                    session_id = %payload.session_id,
                    "SubagentStop hook timed out after 5000ms"
                );
                Ok(HookResponse {
                    success: false,
                    error: Some("Hook timed out".to_string()),
                    ..Default::default()
                })
            }
        }
    }

    async fn handle_subagent_stop_inner(
        &self,
        payload: &HookPayload,
    ) -> Result<HookResponse, HandlerError> {
        let start = std::time::Instant::now();

        let subagent_ctx = match &payload.context {
            HookContext::Subagent(ctx) => ctx,
            _ => return Err(HandlerError::new(
                ErrorCode::InvalidParams,
                "Expected subagent context",
            )),
        };

        // Merge learnings from subagent into parent session
        let merged_count = self.merge_learnings(subagent_ctx).await?;

        let metrics = HookMetrics {
            processing_time_ms: start.elapsed().as_millis() as u64,
            memories_accessed: subagent_ctx.learnings.len(),
            patterns_matched: merged_count,
            embedder_times_ms: None,
        };

        Ok(HookResponse {
            success: true,
            context_recommendations: None,
            trajectory_id: None,
            consciousness: None,
            metrics: Some(metrics),
            error: None,
        })
    }

    /// Merge learnings from subagent trajectory into parent session.
    async fn merge_learnings(
        &self,
        context: &SubagentContext,
    ) -> Result<usize, HandlerError> {
        let mut merged = 0;

        for learning in &context.learnings {
            // Store each learning with its embedder alignments
            let inject_params = InjectParams {
                content: learning.pattern.clone(),
                memory_type: Some(MemoryType::SubagentLearning),
                namespace: Some(context.parent_session_id.clone()),
                metadata: Some(serde_json::json!({
                    "subagent_id": context.subagent_id,
                    "embedder_alignments": learning.embedder_alignments,
                    "success": learning.success,
                })),
                options: None,
            };

            if self.memory_handler.handle_inject(inject_params).await.is_ok() {
                merged += 1;
            }
        }

        Ok(merged)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Shell Script Templates: .claude/hooks/
// ═══════════════════════════════════════════════════════════════════════

/*
.claude/hooks/session-start.sh:
#!/bin/bash
# SessionStart hook - Initialize workspace, load SELF_EGO_NODE
# Timeout: 5000ms

# Read JSON payload from stdin
PAYLOAD=$(cat)

# Call MCP server hook endpoint
echo "$PAYLOAD" | context-graph-mcp hooks session-start

# Exit code determines hook success
exit $?


.claude/hooks/pre-tool-use.sh:
#!/bin/bash
# PreToolUse hook - Inject relevant context before tool execution
# Timeout: 3000ms
# Matcher: Read|Grep|Glob|Bash

PAYLOAD=$(cat)
echo "$PAYLOAD" | context-graph-mcp hooks pre-tool-use
exit $?


.claude/hooks/post-tool-use.sh:
#!/bin/bash
# PostToolUse hook - Store learned patterns from tool output
# Timeout: 3000ms (async allowed)
# Matcher: Edit|Write|Bash

PAYLOAD=$(cat)
echo "$PAYLOAD" | context-graph-mcp hooks post-tool-use
exit $?


.claude/hooks/session-end.sh:
#!/bin/bash
# SessionEnd hook - Memory consolidation and goal discovery
# Timeout: 60000ms (deep dreaming mode)

PAYLOAD=$(cat)
echo "$PAYLOAD" | context-graph-mcp hooks session-end
exit $?


.claude/hooks/pre-compact.sh:
#!/bin/bash
# PreCompact hook - Extract salient memories before context compaction
# Timeout: 10000ms

PAYLOAD=$(cat)
echo "$PAYLOAD" | context-graph-mcp hooks pre-compact
exit $?


.claude/hooks/subagent-stop.sh:
#!/bin/bash
# SubagentStop hook - Merge subagent learnings into main memory
# Timeout: 5000ms

PAYLOAD=$(cat)
echo "$PAYLOAD" | context-graph-mcp hooks subagent-stop
exit $?
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hook_event_timeouts() {
        assert_eq!(ClaudeCodeHookEvent::SessionStart.timeout_ms(), 5000);
        assert_eq!(ClaudeCodeHookEvent::PreToolUse.timeout_ms(), 3000);
        assert_eq!(ClaudeCodeHookEvent::PostToolUse.timeout_ms(), 3000);
        assert_eq!(ClaudeCodeHookEvent::SessionEnd.timeout_ms(), 60000);
        assert_eq!(ClaudeCodeHookEvent::PreCompact.timeout_ms(), 10000);
        assert_eq!(ClaudeCodeHookEvent::SubagentStop.timeout_ms(), 5000);
    }

    #[tokio::test]
    async fn test_consciousness_state_from_r() {
        assert_eq!(ConsciousnessState::from_order_parameter(0.1), ConsciousnessState::Dormant);
        assert_eq!(ConsciousnessState::from_order_parameter(0.4), ConsciousnessState::Fragmented);
        assert_eq!(ConsciousnessState::from_order_parameter(0.6), ConsciousnessState::Emerging);
        assert_eq!(ConsciousnessState::from_order_parameter(0.85), ConsciousnessState::Conscious);
        assert_eq!(ConsciousnessState::from_order_parameter(0.98), ConsciousnessState::Hypersync);
    }

    #[tokio::test]
    async fn test_consciousness_metrics_compute() {
        let metrics = ConsciousnessMetrics::compute(0.8, 0.9, 0.7);
        assert!((metrics.consciousness - 0.504).abs() < 0.001);
        assert_eq!(metrics.state, ConsciousnessState::Conscious);
    }

    #[tokio::test]
    async fn test_embedder_name_mapping() {
        assert_eq!(embedder_name(0), "E1_Semantic");
        assert_eq!(embedder_name(6), "E7_Code");
        assert_eq!(embedder_name(12), "E13_SPLADE");
        assert_eq!(embedder_name(13), "Unknown");
    }

    #[tokio::test]
    async fn test_session_start_hook() {
        // Test SessionStart hook with proper context
    }

    #[tokio::test]
    async fn test_session_end_hook() {
        // Test SessionEnd hook with consolidation
    }

    #[tokio::test]
    async fn test_pre_tool_use_hook() {
        // Test PreToolUse context injection
    }

    #[tokio::test]
    async fn test_post_tool_use_hook() {
        // Test PostToolUse pattern learning
    }

    #[tokio::test]
    async fn test_pre_compact_hook() {
        // Test PreCompact memory extraction
    }

    #[tokio::test]
    async fn test_subagent_stop_hook() {
        // Test SubagentStop learning merge
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-mcp/src/hooks/protocol.rs">
    Claude Code hook protocol types matching official specification
  </file>
  <file path="crates/context-graph-mcp/src/hooks/session.rs">
    SessionStart and SessionEnd hook handlers with GWT integration
  </file>
  <file path="crates/context-graph-mcp/src/hooks/tool_use.rs">
    PreToolUse and PostToolUse hook handlers with 13-embedder search
  </file>
  <file path="crates/context-graph-mcp/src/hooks/compact.rs">
    PreCompact hook handler for memory extraction
  </file>
  <file path="crates/context-graph-mcp/src/hooks/subagent.rs">
    SubagentStop hook handler for learning merge
  </file>
  <file path="crates/context-graph-mcp/src/hooks/trajectory.rs">
    Trajectory tracking with per-embedder alignment
  </file>
  <file path="crates/context-graph-mcp/src/hooks/mod.rs">
    Hooks module definition
  </file>
  <file path=".claude/hooks/session-start.sh">
    Shell script for SessionStart hook
  </file>
  <file path=".claude/hooks/pre-tool-use.sh">
    Shell script for PreToolUse hook
  </file>
  <file path=".claude/hooks/post-tool-use.sh">
    Shell script for PostToolUse hook
  </file>
  <file path=".claude/hooks/session-end.sh">
    Shell script for SessionEnd hook
  </file>
  <file path=".claude/hooks/pre-compact.sh">
    Shell script for PreCompact hook
  </file>
  <file path=".claude/hooks/subagent-stop.sh">
    Shell script for SubagentStop hook
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-mcp/src/handlers/core.rs">
    Add dispatch routes for hooks/* endpoints
  </file>
  <file path=".claude/settings.json">
    Add hook configuration entries
  </file>
</files_to_modify>

<validation_criteria>
  <criterion>SessionStart loads SELF_EGO_NODE and computes consciousness C(t)</criterion>
  <criterion>SessionEnd triggers consolidation with 60s deep dreaming</criterion>
  <criterion>PreToolUse injects context within 3000ms (100ms critical path)</criterion>
  <criterion>PostToolUse stores patterns with per-embedder alignments [E1-E13]</criterion>
  <criterion>PreCompact identifies salient memories for preservation</criterion>
  <criterion>SubagentStop merges learnings into parent session</criterion>
  <criterion>All hooks respect constitution timeout requirements</criterion>
  <criterion>GWT consciousness C(t) = I(t) × R(t) × D(t) computed on each hook</criterion>
  <criterion>Shell scripts work with Claude Code's hook execution model</criterion>
</validation_criteria>

<test_commands>
  <command>cargo test -p context-graph-mcp hooks -- --nocapture</command>
  <command>cargo test -p context-graph-mcp hooks::protocol</command>
  <command>cargo test -p context-graph-mcp hooks::session</command>
  <command>cargo test -p context-graph-mcp hooks::tool_use</command>
  <command>cargo test -p context-graph-mcp hooks::compact</command>
  <command>cargo test -p context-graph-mcp hooks::subagent</command>
</test_commands>
</task_spec>
```
