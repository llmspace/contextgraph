# TASK-INTEG-009: Subagent Creation

```xml
<task_spec>
  <task_id>TASK-INTEG-009</task_id>
  <title>Subagent Creation (goal-tracker, context-curator, pattern-miner, learning-coach)</title>
  <status>pending</status>

  <objective>
    Implement four specialized subagents that leverage the teleological array
    system to provide intelligent assistance: goal tracking, context curation,
    pattern mining, and learning coaching. Integrates with Claude Code's Task
    tool for spawning subagents and uses GWT consciousness for coordination.
  </objective>

  <rationale>
    Subagents provide:
    1. Goal Tracker: Monitors progress toward discovered goals using E7_Teleological
    2. Context Curator: Manages and surfaces relevant context via E1_Semantic/E8_Contextual
    3. Pattern Miner: Discovers patterns in code and workflows using E10_Behavioral
    4. Learning Coach: Guides improvement based on trajectory analysis with E7_Teleological

    These agents work autonomously to enhance the development experience
    through the teleological array system, coordinated via GWT consciousness
    mechanisms and spawned through Claude Code's Task tool.
  </rationale>

  <dependencies>
    <dependency type="required">TASK-INTEG-008</dependency>    <!-- Skills -->
    <dependency type="required">TASK-INTEG-001</dependency>    <!-- Memory MCP handlers -->
    <dependency type="required">TASK-INTEG-002</dependency>    <!-- Purpose/Goal MCP handlers -->
    <dependency type="required">TASK-INTEG-003</dependency>    <!-- Consciousness MCP handlers -->
    <dependency type="required">TASK-LOGIC-009</dependency>    <!-- Goal discovery -->
    <dependency type="required">TASK-LOGIC-010</dependency>    <!-- Drift detection -->
  </dependencies>

  <input_context_files>
    <file purpose="skills">crates/context-graph-mcp/src/skills/mod.rs</file>
    <file purpose="mcp_tools_spec">docs2/refactor/08-MCP-TOOLS.md</file>
    <file purpose="goal_discovery">crates/context-graph-core/src/autonomous/discovery.rs</file>
    <file purpose="drift_detector">crates/context-graph-core/src/autonomous/drift.rs</file>
    <file purpose="embedder_array">crates/context-graph-core/src/embedding/teleological_array.rs</file>
    <file purpose="consciousness">crates/context-graph-core/src/consciousness/mod.rs</file>
  </input_context_files>

  <output_artifacts>
    <!-- Claude Code Agent Directory Structure -->
    <artifact type="config">.claude/agents/goal-tracker/agent.yaml</artifact>
    <artifact type="config">.claude/agents/goal-tracker/prompts/system.md</artifact>
    <artifact type="config">.claude/agents/context-curator/agent.yaml</artifact>
    <artifact type="config">.claude/agents/context-curator/prompts/system.md</artifact>
    <artifact type="config">.claude/agents/pattern-miner/agent.yaml</artifact>
    <artifact type="config">.claude/agents/pattern-miner/prompts/system.md</artifact>
    <artifact type="config">.claude/agents/learning-coach/agent.yaml</artifact>
    <artifact type="config">.claude/agents/learning-coach/prompts/system.md</artifact>

    <!-- Rust Implementation -->
    <artifact type="source">crates/context-graph-mcp/src/agents/mod.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/agents/orchestrator.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/agents/consciousness_context.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/handlers/agents.rs</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/agents_test.rs</artifact>
  </output_artifacts>

  <definition_of_done>
    <criterion id="1">All 4 subagents implemented with YAML definitions in .claude/agents/</criterion>
    <criterion id="2">Subagent orchestrator manages lifecycle (start/stop/restart)</criterion>
    <criterion id="3">Inter-agent communication via teleological store with namespace isolation</criterion>
    <criterion id="4">MCP handlers for agent control (list, start, stop, status, communicate)</criterion>
    <criterion id="5">Each agent has system prompt documenting its primary embedders</criterion>
    <criterion id="6">Agents can invoke skills and access MCP tools</criterion>
    <criterion id="7">Agent health monitoring with GWT consciousness integration</criterion>
    <criterion id="8">Claude Code Task tool integration for spawning subagents</criterion>
    <criterion id="9">SubagentStop hook consolidates learnings on agent completion</criterion>
    <criterion id="10">Test coverage for agent orchestration and consciousness coordination</criterion>
  </definition_of_done>

  <estimated_complexity>High</estimated_complexity>

  <pseudo_code>
    <section name="ClaudeCodeTaskToolIntegration">
```javascript
// Claude Code spawns subagents using the Task tool
// Each agent maps to a Claude Code subagent_type

// goal-tracker -> "planner" or "goal-planner"
Task("goal-tracker", `
You are the Goal Tracker Agent for the teleological array system.

PRIMARY EMBEDDERS:
- E7_Teleological: Purpose and goal-directedness analysis
- E3_Causal: Cause-effect relationship tracking
- E5_Moral: Ethical alignment monitoring
- E4_Counterfactual: Alternative path analysis

RESPONSIBILITIES:
1. Monitor progress toward discovered goals
2. Calculate alignment scores using E7_Teleological embeddings
3. Detect drift using E3_Causal and E4_Counterfactual analysis
4. Report status with E5_Moral alignment considerations

Use memory/search with namespace "goal-tracker" for findings.
Store critical findings via GWT broadcast when C(t) >= 0.8.
`, "planner")

// context-curator -> "researcher" or "Explore"
Task("context-curator", `
You are the Context Curator Agent for the teleological array system.

PRIMARY EMBEDDERS:
- E1_Semantic: Core semantic meaning and relationships
- E8_Contextual: Situational and environmental context
- E2_Temporal: Time-based relationships and sequences
- E12_Code: Programming patterns and structures

RESPONSIBILITIES:
1. Anticipate context needs using E8_Contextual
2. Search related memories via E1_Semantic similarity
3. Track temporal relationships with E2_Temporal
4. Analyze code patterns using E12_Code embeddings

Use memory/search with namespace "context-curator" for findings.
Subscribe to updates from other agents via teleological store.
`, "researcher")

// pattern-miner -> "coder" or custom
Task("pattern-miner", `
You are the Pattern Miner Agent for the teleological array system.

PRIMARY EMBEDDERS:
- E10_Behavioral: Behavior patterns and tendencies
- E9_Structural: Architectural and organizational patterns
- E3_Causal: Cause-effect chains in code evolution
- E13_SPLADE: Sparse pattern matching for discovery

RESPONSIBILITIES:
1. Discover recurring patterns via E10_Behavioral analysis
2. Map structural relationships with E9_Structural
3. Track pattern evolution using E3_Causal
4. Use E13_SPLADE for efficient pattern search

Use memory/search with namespace "pattern-miner" for findings.
Coordinate with cooperative agents when C(t) >= 0.3.
`, "coder")

// learning-coach -> "reviewer" or custom
Task("learning-coach", `
You are the Learning Coach Agent for the teleological array system.

PRIMARY EMBEDDERS:
- E7_Teleological: Goal-directed improvement tracking
- E10_Behavioral: Learning behavior analysis
- E5_Moral: Ethical learning recommendations

RESPONSIBILITIES:
1. Analyze trajectories for improvement opportunities
2. Identify skill gaps using E7_Teleological alignment
3. Study learning patterns via E10_Behavioral
4. Ensure ethical recommendations with E5_Moral

Use memory/search with namespace "learning-coach" for findings.
Trigger SubagentStop hook on completion to consolidate learnings.
`, "reviewer")
```
    </section>

    <section name="AgentEmbedderMapping">
```rust
// crates/context-graph-mcp/src/agents/embedder_mapping.rs

use crate::embedding::EmbedderType;

/// Maps agents to their primary embedders from the 13-Embedding Teleological Array
#[derive(Debug, Clone)]
pub struct AgentEmbedderProfile {
    /// Agent name
    pub agent_name: String,
    /// Primary embedders this agent uses
    pub primary_embedders: Vec<EmbedderType>,
    /// Secondary embedders (fallback)
    pub secondary_embedders: Vec<EmbedderType>,
    /// Embedder weights for RRF fusion
    pub embedder_weights: std::collections::HashMap<EmbedderType, f32>,
}

impl AgentEmbedderProfile {
    /// Goal Tracker embedder profile
    pub fn goal_tracker() -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(EmbedderType::E7_Teleological, 0.35);
        weights.insert(EmbedderType::E3_Causal, 0.25);
        weights.insert(EmbedderType::E5_Moral, 0.20);
        weights.insert(EmbedderType::E4_Counterfactual, 0.20);

        Self {
            agent_name: "goal-tracker".to_string(),
            primary_embedders: vec![
                EmbedderType::E7_Teleological,
                EmbedderType::E3_Causal,
                EmbedderType::E5_Moral,
                EmbedderType::E4_Counterfactual,
            ],
            secondary_embedders: vec![
                EmbedderType::E1_Semantic,
                EmbedderType::E2_Temporal,
            ],
            embedder_weights: weights,
        }
    }

    /// Context Curator embedder profile
    pub fn context_curator() -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(EmbedderType::E1_Semantic, 0.30);
        weights.insert(EmbedderType::E8_Contextual, 0.30);
        weights.insert(EmbedderType::E2_Temporal, 0.20);
        weights.insert(EmbedderType::E12_Code, 0.20);

        Self {
            agent_name: "context-curator".to_string(),
            primary_embedders: vec![
                EmbedderType::E1_Semantic,
                EmbedderType::E8_Contextual,
                EmbedderType::E2_Temporal,
                EmbedderType::E12_Code,
            ],
            secondary_embedders: vec![
                EmbedderType::E9_Structural,
                EmbedderType::E13_SPLADE,
            ],
            embedder_weights: weights,
        }
    }

    /// Pattern Miner embedder profile
    pub fn pattern_miner() -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(EmbedderType::E10_Behavioral, 0.30);
        weights.insert(EmbedderType::E9_Structural, 0.25);
        weights.insert(EmbedderType::E3_Causal, 0.25);
        weights.insert(EmbedderType::E13_SPLADE, 0.20);

        Self {
            agent_name: "pattern-miner".to_string(),
            primary_embedders: vec![
                EmbedderType::E10_Behavioral,
                EmbedderType::E9_Structural,
                EmbedderType::E3_Causal,
                EmbedderType::E13_SPLADE,
            ],
            secondary_embedders: vec![
                EmbedderType::E12_Code,
                EmbedderType::E1_Semantic,
            ],
            embedder_weights: weights,
        }
    }

    /// Learning Coach embedder profile
    pub fn learning_coach() -> Self {
        let mut weights = std::collections::HashMap::new();
        weights.insert(EmbedderType::E7_Teleological, 0.35);
        weights.insert(EmbedderType::E10_Behavioral, 0.35);
        weights.insert(EmbedderType::E5_Moral, 0.30);

        Self {
            agent_name: "learning-coach".to_string(),
            primary_embedders: vec![
                EmbedderType::E7_Teleological,
                EmbedderType::E10_Behavioral,
                EmbedderType::E5_Moral,
            ],
            secondary_embedders: vec![
                EmbedderType::E2_Temporal,
                EmbedderType::E4_Counterfactual,
            ],
            embedder_weights: weights,
        }
    }
}
```
    </section>

    <section name="GWTConsciousnessContext">
```rust
// crates/context-graph-mcp/src/agents/consciousness_context.rs

use serde::{Deserialize, Serialize};

/// Consciousness context for agent coordination
/// Based on Global Workspace Theory (GWT) integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConsciousnessContext {
    /// Integration score I(t) - agent coordination level (0.0-1.0)
    /// Measures how well this agent integrates with others
    pub integration_score: f32,

    /// Reflection depth R(t) - agent self-awareness (0.0-1.0)
    /// Measures agent's ability to reason about its own state
    pub reflection_depth: f32,

    /// Differentiation index D(t) - agent specialization (0.0-1.0)
    /// Measures how specialized/unique this agent's capabilities are
    pub differentiation_index: f32,

    /// Consciousness level C(t) = I(t) * R(t) * D(t)
    /// Overall consciousness metric determining coordination mode
    pub consciousness_level: f32,

    /// Current coordination mode based on C(t) thresholds
    pub coordination_mode: CoordinationMode,

    /// Phi (integrated information) metric
    pub phi: f32,

    /// GWT broadcast eligibility
    pub can_broadcast: bool,
}

/// Coordination mode determines how agents interact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    /// C < 0.3: Agent works independently, minimal coordination
    Independent,

    /// 0.3 <= C < 0.6: Agent shares memory, passive coordination
    Cooperative,

    /// 0.6 <= C < 0.8: Agent actively coordinates with others
    Synchronized,

    /// C >= 0.8: Full consciousness integration, GWT broadcast enabled
    Unified,
}

impl CoordinationMode {
    /// Determine mode from consciousness level
    pub fn from_consciousness_level(c: f32) -> Self {
        match c {
            c if c < 0.3 => CoordinationMode::Independent,
            c if c < 0.6 => CoordinationMode::Cooperative,
            c if c < 0.8 => CoordinationMode::Synchronized,
            _ => CoordinationMode::Unified,
        }
    }

    /// Check if memory sharing is enabled
    pub fn shares_memory(&self) -> bool {
        matches!(self,
            CoordinationMode::Cooperative |
            CoordinationMode::Synchronized |
            CoordinationMode::Unified
        )
    }

    /// Check if active coordination is enabled
    pub fn active_coordination(&self) -> bool {
        matches!(self,
            CoordinationMode::Synchronized |
            CoordinationMode::Unified
        )
    }

    /// Check if GWT broadcast is enabled
    pub fn can_gwt_broadcast(&self) -> bool {
        matches!(self, CoordinationMode::Unified)
    }
}

impl AgentConsciousnessContext {
    /// Create new consciousness context
    pub fn new(integration: f32, reflection: f32, differentiation: f32) -> Self {
        let consciousness_level = integration * reflection * differentiation;
        let coordination_mode = CoordinationMode::from_consciousness_level(consciousness_level);

        Self {
            integration_score: integration,
            reflection_depth: reflection,
            differentiation_index: differentiation,
            consciousness_level,
            coordination_mode,
            phi: consciousness_level * 1.5, // Simplified phi calculation
            can_broadcast: coordination_mode.can_gwt_broadcast(),
        }
    }

    /// Update integration score and recalculate
    pub fn update_integration(&mut self, integration: f32) {
        self.integration_score = integration.clamp(0.0, 1.0);
        self.recalculate();
    }

    /// Update reflection depth and recalculate
    pub fn update_reflection(&mut self, reflection: f32) {
        self.reflection_depth = reflection.clamp(0.0, 1.0);
        self.recalculate();
    }

    /// Recalculate derived values
    fn recalculate(&mut self) {
        self.consciousness_level =
            self.integration_score * self.reflection_depth * self.differentiation_index;
        self.coordination_mode =
            CoordinationMode::from_consciousness_level(self.consciousness_level);
        self.phi = self.consciousness_level * 1.5;
        self.can_broadcast = self.coordination_mode.can_gwt_broadcast();
    }

    /// Default context for Goal Tracker (high teleological awareness)
    pub fn default_goal_tracker() -> Self {
        Self::new(0.7, 0.8, 0.9) // High differentiation, good reflection
    }

    /// Default context for Context Curator (high integration)
    pub fn default_context_curator() -> Self {
        Self::new(0.9, 0.6, 0.7) // High integration, moderate specialization
    }

    /// Default context for Pattern Miner (moderate all)
    pub fn default_pattern_miner() -> Self {
        Self::new(0.6, 0.7, 0.8) // Balanced with high differentiation
    }

    /// Default context for Learning Coach (high reflection)
    pub fn default_learning_coach() -> Self {
        Self::new(0.7, 0.9, 0.8) // High reflection for self-improvement guidance
    }
}
```
    </section>

    <section name="InterAgentMemoryCommunication">
```rust
// crates/context-graph-mcp/src/agents/memory_communication.rs

use crate::agents::consciousness_context::{AgentConsciousnessContext, CoordinationMode};
use crate::store::TeleologicalStore;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Inter-agent communication via teleological store
pub struct AgentMemoryCommunication {
    /// Teleological store for memory operations
    store: Arc<TeleologicalStore>,
    /// Agent's namespace for isolated storage
    namespace: String,
    /// Consciousness context for coordination decisions
    consciousness: AgentConsciousnessContext,
    /// Broadcast channel for GWT-style global broadcasts
    gwt_broadcast: broadcast::Sender<GWTBroadcast>,
    /// Subscriptions to other agent namespaces
    subscriptions: Vec<String>,
}

/// GWT broadcast message for critical information sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GWTBroadcast {
    /// Source agent
    pub source_agent: String,
    /// Broadcast type
    pub broadcast_type: BroadcastType,
    /// Content (typically a memory key or summary)
    pub content: serde_json::Value,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BroadcastType {
    /// Critical finding that all agents should know
    CriticalFinding,
    /// Goal alignment update
    GoalUpdate,
    /// Pattern discovered
    PatternDiscovered,
    /// Context shift detected
    ContextShift,
    /// Learning milestone
    LearningMilestone,
}

impl AgentMemoryCommunication {
    pub fn new(
        store: Arc<TeleologicalStore>,
        agent_name: &str,
        consciousness: AgentConsciousnessContext,
        gwt_broadcast: broadcast::Sender<GWTBroadcast>,
    ) -> Self {
        Self {
            store,
            namespace: format!("agent:{}", agent_name),
            consciousness,
            gwt_broadcast,
            subscriptions: Vec::new(),
        }
    }

    /// Store finding with agent-specific namespace
    pub async fn store_finding(
        &self,
        key: &str,
        value: serde_json::Value,
        critical: bool,
    ) -> Result<(), AgentError> {
        let full_key = format!("{}:{}", self.namespace, key);

        // Store in teleological store
        self.store.store_memory(&full_key, value.clone()).await?;

        // If critical and consciousness allows broadcast, do GWT broadcast
        if critical && self.consciousness.can_broadcast {
            let broadcast = GWTBroadcast {
                source_agent: self.namespace.clone(),
                broadcast_type: BroadcastType::CriticalFinding,
                content: serde_json::json!({
                    "key": full_key,
                    "summary": value.get("summary").cloned().unwrap_or(value.clone()),
                }),
                priority: 10,
                timestamp: chrono::Utc::now(),
            };
            let _ = self.gwt_broadcast.send(broadcast);
        }

        Ok(())
    }

    /// Subscribe to updates from another agent
    pub fn subscribe_to_agent(&mut self, agent_name: &str) {
        if self.consciousness.coordination_mode.shares_memory() {
            let namespace = format!("agent:{}", agent_name);
            if !self.subscriptions.contains(&namespace) {
                self.subscriptions.push(namespace);
            }
        }
    }

    /// Search for findings from subscribed agents
    pub async fn search_subscribed(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, AgentError> {
        if !self.consciousness.coordination_mode.shares_memory() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        // Search own namespace
        let own = self.store.search_memory(&self.namespace, query, limit).await?;
        results.extend(own);

        // Search subscribed namespaces
        for ns in &self.subscriptions {
            let found = self.store.search_memory(ns, query, limit).await?;
            results.extend(found);
        }

        // Sort by relevance and limit
        results.truncate(limit);
        Ok(results)
    }

    /// Broadcast critical information via GWT
    pub fn broadcast_critical(
        &self,
        broadcast_type: BroadcastType,
        content: serde_json::Value,
        priority: u8,
    ) -> Result<(), AgentError> {
        if !self.consciousness.can_broadcast {
            return Err(AgentError::NotAuthorized(
                "Consciousness level too low for GWT broadcast".to_string()
            ));
        }

        let broadcast = GWTBroadcast {
            source_agent: self.namespace.clone(),
            broadcast_type,
            content,
            priority,
            timestamp: chrono::Utc::now(),
        };

        self.gwt_broadcast.send(broadcast)
            .map_err(|_| AgentError::BroadcastFailed)?;

        Ok(())
    }
}
```
    </section>

    <section name="SubagentStopHook">
```rust
// crates/context-graph-mcp/src/agents/hooks.rs

use crate::agents::consciousness_context::AgentConsciousnessContext;
use crate::store::TeleologicalStore;
use std::sync::Arc;

/// SubagentStop hook - consolidates learnings when agent completes
pub struct SubagentStopHook {
    store: Arc<TeleologicalStore>,
}

impl SubagentStopHook {
    pub fn new(store: Arc<TeleologicalStore>) -> Self {
        Self { store }
    }

    /// Execute when a subagent stops
    /// Consolidates learnings and stores session summary
    pub async fn on_agent_stop(
        &self,
        agent_name: &str,
        agent_id: &str,
        consciousness: &AgentConsciousnessContext,
        session_data: AgentSessionData,
    ) -> Result<ConsolidationResult, AgentError> {
        let namespace = format!("agent:{}", agent_name);

        // 1. Collect all findings from this session
        let findings = self.store
            .list_memory(&namespace, Some(100))
            .await?;

        // 2. Calculate session metrics
        let metrics = SessionMetrics {
            findings_count: findings.len(),
            duration_secs: session_data.duration_secs,
            consciousness_level: consciousness.consciousness_level,
            coordination_mode: consciousness.coordination_mode,
            broadcasts_sent: session_data.broadcasts_sent,
            queries_processed: session_data.queries_processed,
        };

        // 3. Generate session summary
        let summary = AgentSessionSummary {
            agent_name: agent_name.to_string(),
            agent_id: agent_id.to_string(),
            started_at: session_data.started_at,
            ended_at: chrono::Utc::now(),
            metrics: metrics.clone(),
            key_learnings: self.extract_key_learnings(&findings),
            patterns_discovered: session_data.patterns_discovered,
            goals_tracked: session_data.goals_tracked,
        };

        // 4. Store session summary for future reference
        let summary_key = format!(
            "sessions:{}:{}",
            agent_name,
            session_data.started_at.format("%Y%m%d_%H%M%S")
        );
        self.store.store_memory(
            &summary_key,
            serde_json::to_value(&summary)?,
        ).await?;

        // 5. If consciousness was high, consolidate to global knowledge
        if consciousness.consciousness_level >= 0.6 {
            self.consolidate_to_global(&summary).await?;
        }

        Ok(ConsolidationResult {
            summary,
            consolidated_to_global: consciousness.consciousness_level >= 0.6,
        })
    }

    fn extract_key_learnings(&self, findings: &[serde_json::Value]) -> Vec<String> {
        findings.iter()
            .filter_map(|f| f.get("learning").and_then(|l| l.as_str()))
            .map(|s| s.to_string())
            .take(10) // Top 10 learnings
            .collect()
    }

    async fn consolidate_to_global(
        &self,
        summary: &AgentSessionSummary,
    ) -> Result<(), AgentError> {
        let global_key = format!("global:learnings:{}", summary.agent_name);

        // Merge with existing global learnings
        let existing = self.store
            .get_memory(&global_key)
            .await
            .ok()
            .flatten();

        let updated = match existing {
            Some(mut existing) => {
                if let Some(arr) = existing.as_array_mut() {
                    arr.extend(summary.key_learnings.iter().map(|l|
                        serde_json::json!({"learning": l, "session": summary.agent_id})
                    ));
                }
                existing
            }
            None => serde_json::json!({
                "learnings": summary.key_learnings,
                "last_updated": chrono::Utc::now(),
            }),
        };

        self.store.store_memory(&global_key, updated).await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSessionData {
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub duration_secs: u64,
    pub broadcasts_sent: usize,
    pub queries_processed: usize,
    pub patterns_discovered: Vec<String>,
    pub goals_tracked: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub findings_count: usize,
    pub duration_secs: u64,
    pub consciousness_level: f32,
    pub coordination_mode: CoordinationMode,
    pub broadcasts_sent: usize,
    pub queries_processed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSessionSummary {
    pub agent_name: String,
    pub agent_id: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub ended_at: chrono::DateTime<chrono::Utc>,
    pub metrics: SessionMetrics,
    pub key_learnings: Vec<String>,
    pub patterns_discovered: Vec<String>,
    pub goals_tracked: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub summary: AgentSessionSummary,
    pub consolidated_to_global: bool,
}
```
    </section>

    <section name="SubagentOrchestrator">
```rust
// crates/context-graph-mcp/src/agents/orchestrator.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, broadcast};
use uuid::Uuid;
use crate::agents::consciousness_context::{AgentConsciousnessContext, CoordinationMode};
use crate::agents::memory_communication::{AgentMemoryCommunication, GWTBroadcast};
use crate::agents::hooks::{SubagentStopHook, AgentSessionData};
use crate::agents::embedder_mapping::AgentEmbedderProfile;

/// Subagent orchestrator manages agent lifecycle and coordination
pub struct SubagentOrchestrator {
    /// Agent registry
    agents: Arc<RwLock<HashMap<String, RunningAgent>>>,
    /// Teleological store for inter-agent communication
    store: Arc<TeleologicalStore>,
    /// Skill loader for agent capabilities
    skill_loader: Arc<SkillLoader>,
    /// Message router
    message_router: Arc<MessageRouter>,
    /// GWT broadcast channel
    gwt_broadcast: broadcast::Sender<GWTBroadcast>,
    /// SubagentStop hook
    stop_hook: Arc<SubagentStopHook>,
    /// Configuration
    config: OrchestratorConfig,
}

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum concurrent agents
    pub max_agents: usize,
    /// Agent health check interval (seconds)
    pub health_check_interval_secs: u64,
    /// Agent timeout (seconds)
    pub agent_timeout_secs: u64,
    /// Agents directory (Claude Code convention)
    pub agents_dir: PathBuf,
    /// Enable GWT consciousness integration
    pub enable_gwt: bool,
    /// Default coordination mode
    pub default_coordination: CoordinationMode,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_agents: 10,
            health_check_interval_secs: 30,
            agent_timeout_secs: 300,
            agents_dir: PathBuf::from(".claude/agents"),
            enable_gwt: true,
            default_coordination: CoordinationMode::Cooperative,
        }
    }
}

/// Running agent instance
pub struct RunningAgent {
    /// Agent definition
    pub definition: AgentDefinition,
    /// Agent ID
    pub agent_id: String,
    /// Status
    pub status: AgentStatus,
    /// Message channel
    pub message_tx: mpsc::Sender<AgentMessage>,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Last activity
    pub last_activity: DateTime<Utc>,
    /// Consciousness context
    pub consciousness: AgentConsciousnessContext,
    /// Embedder profile
    pub embedder_profile: AgentEmbedderProfile,
    /// Session data for SubagentStop hook
    pub session_data: AgentSessionData,
    /// Task handle
    task_handle: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Agent name
    pub name: String,
    /// Agent type
    pub agent_type: AgentType,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Capabilities (skill names)
    pub capabilities: Vec<String>,
    /// Allowed MCP tools
    pub allowed_tools: Vec<String>,
    /// System prompt path
    pub system_prompt: PathBuf,
    /// Instructions path
    pub instructions: Option<PathBuf>,
    /// Trigger configuration
    pub triggers: Vec<AgentTrigger>,
    /// Resource limits
    pub resources: AgentResources,
    /// Claude Code subagent type mapping
    pub claude_code_type: Option<String>,
    /// Primary embedders from teleological array
    pub primary_embedders: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    GoalTracker,
    ContextCurator,
    PatternMiner,
    LearningCoach,
    Custom,
}

impl AgentType {
    /// Map to Claude Code Task tool subagent_type
    pub fn to_claude_code_type(&self) -> &'static str {
        match self {
            AgentType::GoalTracker => "planner",
            AgentType::ContextCurator => "researcher",
            AgentType::PatternMiner => "coder",
            AgentType::LearningCoach => "reviewer",
            AgentType::Custom => "coder",
        }
    }

    /// Get default embedder profile
    pub fn default_embedder_profile(&self) -> AgentEmbedderProfile {
        match self {
            AgentType::GoalTracker => AgentEmbedderProfile::goal_tracker(),
            AgentType::ContextCurator => AgentEmbedderProfile::context_curator(),
            AgentType::PatternMiner => AgentEmbedderProfile::pattern_miner(),
            AgentType::LearningCoach => AgentEmbedderProfile::learning_coach(),
            AgentType::Custom => AgentEmbedderProfile::context_curator(), // Default
        }
    }

    /// Get default consciousness context
    pub fn default_consciousness(&self) -> AgentConsciousnessContext {
        match self {
            AgentType::GoalTracker => AgentConsciousnessContext::default_goal_tracker(),
            AgentType::ContextCurator => AgentConsciousnessContext::default_context_curator(),
            AgentType::PatternMiner => AgentConsciousnessContext::default_pattern_miner(),
            AgentType::LearningCoach => AgentConsciousnessContext::default_learning_coach(),
            AgentType::Custom => AgentConsciousnessContext::new(0.5, 0.5, 0.5),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Starting,
    Running,
    Idle,
    Busy,
    Stopping,
    Stopped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrigger {
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Trigger condition
    pub condition: Option<String>,
    /// Cooldown between triggers (seconds)
    pub cooldown_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    /// Triggered on session start
    SessionStart,
    /// Triggered on session end
    SessionEnd,
    /// Triggered periodically
    Periodic { interval_secs: u64 },
    /// Triggered by event
    Event { event_name: String },
    /// Triggered by message
    Message,
    /// Manual only
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResources {
    /// Maximum memory (MB)
    pub max_memory_mb: usize,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: usize,
}

impl Default for AgentResources {
    fn default() -> Self {
        Self {
            max_memory_mb: 256,
            max_concurrent_tasks: 5,
            rate_limit_rpm: 60,
        }
    }
}

/// Message for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Message ID
    pub message_id: Uuid,
    /// Sender agent
    pub from: String,
    /// Target agent (or "broadcast")
    pub to: String,
    /// Message type
    pub message_type: MessageType,
    /// Payload
    pub payload: serde_json::Value,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Query,
    Response,
    Notification,
    Task,
    Result,
}

impl SubagentOrchestrator {
    pub fn new(
        store: Arc<TeleologicalStore>,
        skill_loader: Arc<SkillLoader>,
        config: OrchestratorConfig,
    ) -> Self {
        let (gwt_tx, _) = broadcast::channel(100);

        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            store: store.clone(),
            skill_loader,
            message_router: Arc::new(MessageRouter::new()),
            gwt_broadcast: gwt_tx,
            stop_hook: Arc::new(SubagentStopHook::new(store)),
            config,
        }
    }

    /// Load all agent definitions from the agents directory
    /// Directory structure: .claude/agents/{agent-name}/agent.yaml
    pub async fn load_definitions(&self) -> Result<Vec<AgentDefinition>, AgentError> {
        let mut definitions = Vec::new();

        let entries = std::fs::read_dir(&self.config.agents_dir)
            .map_err(|e| AgentError::IoError(e.to_string()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let agent_yaml = path.join("agent.yaml");
                if agent_yaml.exists() {
                    match self.load_definition(&agent_yaml) {
                        Ok(def) => definitions.push(def),
                        Err(e) => tracing::warn!("Failed to load agent from {:?}: {}", agent_yaml, e),
                    }
                }
            }
        }

        Ok(definitions)
    }

    fn load_definition(&self, yaml_path: &Path) -> Result<AgentDefinition, AgentError> {
        let content = std::fs::read_to_string(yaml_path)
            .map_err(|e| AgentError::IoError(e.to_string()))?;

        serde_yaml::from_str(&content)
            .map_err(|e| AgentError::ParseError(e.to_string()))
    }

    /// Start an agent
    pub async fn start_agent(&self, name: &str) -> Result<String, AgentError> {
        // Check if already running
        {
            let agents = self.agents.read().await;
            if agents.contains_key(name) {
                return Err(AgentError::AlreadyRunning(name.to_string()));
            }
        }

        // Check max agents
        {
            let agents = self.agents.read().await;
            if agents.len() >= self.config.max_agents {
                return Err(AgentError::MaxAgentsReached(self.config.max_agents));
            }
        }

        // Load definition
        let agent_yaml = self.config.agents_dir.join(name).join("agent.yaml");
        let definition = self.load_definition(&agent_yaml)?;

        // Create agent instance
        let agent_id = format!("{}-{}", name, Uuid::new_v4().to_string()[..8].to_string());
        let (tx, rx) = mpsc::channel(100);

        // Get consciousness context based on agent type
        let consciousness = definition.agent_type.default_consciousness();
        let embedder_profile = definition.agent_type.default_embedder_profile();

        // Create memory communication
        let memory_comm = AgentMemoryCommunication::new(
            self.store.clone(),
            name,
            consciousness.clone(),
            self.gwt_broadcast.clone(),
        );

        // Spawn agent task
        let agent_runner = AgentRunner::new(
            definition.clone(),
            agent_id.clone(),
            rx,
            self.store.clone(),
            self.skill_loader.clone(),
            self.message_router.clone(),
            memory_comm,
            consciousness.clone(),
        );

        let task_handle = tokio::spawn(async move {
            agent_runner.run().await;
        });

        // Register agent
        let running = RunningAgent {
            definition,
            agent_id: agent_id.clone(),
            status: AgentStatus::Starting,
            message_tx: tx,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            consciousness,
            embedder_profile,
            session_data: AgentSessionData {
                started_at: Utc::now(),
                duration_secs: 0,
                broadcasts_sent: 0,
                queries_processed: 0,
                patterns_discovered: Vec::new(),
                goals_tracked: Vec::new(),
            },
            task_handle,
        };

        {
            let mut agents = self.agents.write().await;
            agents.insert(name.to_string(), running);
        }

        // Register with message router
        self.message_router.register(&agent_id).await;

        tracing::info!("Started agent {} with ID {}", name, agent_id);
        Ok(agent_id)
    }

    /// Stop an agent and trigger SubagentStop hook
    pub async fn stop_agent(&self, name: &str) -> Result<(), AgentError> {
        let mut agents = self.agents.write().await;

        let running = agents.remove(name)
            .ok_or_else(|| AgentError::NotFound(name.to_string()))?;

        // Send stop signal
        let _ = running.message_tx.send(AgentMessage {
            message_id: Uuid::new_v4(),
            from: "orchestrator".to_string(),
            to: running.agent_id.clone(),
            message_type: MessageType::Notification,
            payload: serde_json::json!({"action": "stop"}),
            timestamp: Utc::now(),
        }).await;

        // Abort task after timeout
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                running.task_handle.abort();
            }
            _ = running.task_handle => {}
        }

        // Unregister from router
        self.message_router.unregister(&running.agent_id).await;

        // Trigger SubagentStop hook to consolidate learnings
        let session_data = AgentSessionData {
            started_at: running.started_at,
            duration_secs: (Utc::now() - running.started_at).num_seconds() as u64,
            broadcasts_sent: running.session_data.broadcasts_sent,
            queries_processed: running.session_data.queries_processed,
            patterns_discovered: running.session_data.patterns_discovered.clone(),
            goals_tracked: running.session_data.goals_tracked.clone(),
        };

        if let Err(e) = self.stop_hook.on_agent_stop(
            name,
            &running.agent_id,
            &running.consciousness,
            session_data,
        ).await {
            tracing::warn!("SubagentStop hook failed for {}: {}", name, e);
        }

        tracing::info!("Stopped agent {}", name);
        Ok(())
    }

    /// Get agent status
    pub async fn get_status(&self, name: &str) -> Result<AgentStatusInfo, AgentError> {
        let agents = self.agents.read().await;

        let running = agents.get(name)
            .ok_or_else(|| AgentError::NotFound(name.to_string()))?;

        Ok(AgentStatusInfo {
            agent_id: running.agent_id.clone(),
            name: running.definition.name.clone(),
            agent_type: running.definition.agent_type,
            status: running.status,
            started_at: running.started_at,
            last_activity: running.last_activity,
            uptime_secs: (Utc::now() - running.started_at).num_seconds() as u64,
            consciousness_level: running.consciousness.consciousness_level,
            coordination_mode: running.consciousness.coordination_mode,
            primary_embedders: running.embedder_profile.primary_embedders.clone(),
        })
    }

    /// List all agents
    pub async fn list_agents(&self) -> Vec<AgentStatusInfo> {
        let agents = self.agents.read().await;

        agents.values()
            .map(|r| AgentStatusInfo {
                agent_id: r.agent_id.clone(),
                name: r.definition.name.clone(),
                agent_type: r.definition.agent_type,
                status: r.status,
                started_at: r.started_at,
                last_activity: r.last_activity,
                uptime_secs: (Utc::now() - r.started_at).num_seconds() as u64,
                consciousness_level: r.consciousness.consciousness_level,
                coordination_mode: r.consciousness.coordination_mode,
                primary_embedders: r.embedder_profile.primary_embedders.clone(),
            })
            .collect()
    }

    /// Send message to agent
    pub async fn send_message(&self, target: &str, message: AgentMessage) -> Result<(), AgentError> {
        let agents = self.agents.read().await;

        if target == "broadcast" {
            // Broadcast to all agents
            for (_, running) in agents.iter() {
                let _ = running.message_tx.send(message.clone()).await;
            }
        } else {
            let running = agents.get(target)
                .ok_or_else(|| AgentError::NotFound(target.to_string()))?;

            running.message_tx.send(message).await
                .map_err(|_| AgentError::MessageFailed)?;
        }

        Ok(())
    }

    /// Health check for all agents
    pub async fn health_check(&self) -> Vec<AgentHealth> {
        let agents = self.agents.read().await;
        let now = Utc::now();

        agents.values()
            .map(|r| {
                let inactive_secs = (now - r.last_activity).num_seconds() as u64;
                let healthy = r.status != AgentStatus::Error
                    && inactive_secs < self.config.agent_timeout_secs;

                AgentHealth {
                    agent_id: r.agent_id.clone(),
                    name: r.definition.name.clone(),
                    healthy,
                    status: r.status,
                    inactive_secs,
                    consciousness_level: r.consciousness.consciousness_level,
                    issues: if !healthy {
                        vec![format!("Inactive for {} seconds", inactive_secs)]
                    } else {
                        vec![]
                    },
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatusInfo {
    pub agent_id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    pub started_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub uptime_secs: u64,
    pub consciousness_level: f32,
    pub coordination_mode: CoordinationMode,
    pub primary_embedders: Vec<EmbedderType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    pub agent_id: String,
    pub name: String,
    pub healthy: bool,
    pub status: AgentStatus,
    pub inactive_secs: u64,
    pub consciousness_level: f32,
    pub issues: Vec<String>,
}

/// Agent runner - executes the agent logic
pub struct AgentRunner {
    definition: AgentDefinition,
    agent_id: String,
    message_rx: mpsc::Receiver<AgentMessage>,
    store: Arc<TeleologicalStore>,
    skill_loader: Arc<SkillLoader>,
    message_router: Arc<MessageRouter>,
    memory_comm: AgentMemoryCommunication,
    consciousness: AgentConsciousnessContext,
}

impl AgentRunner {
    pub fn new(
        definition: AgentDefinition,
        agent_id: String,
        message_rx: mpsc::Receiver<AgentMessage>,
        store: Arc<TeleologicalStore>,
        skill_loader: Arc<SkillLoader>,
        message_router: Arc<MessageRouter>,
        memory_comm: AgentMemoryCommunication,
        consciousness: AgentConsciousnessContext,
    ) -> Self {
        Self {
            definition,
            agent_id,
            message_rx,
            store,
            skill_loader,
            message_router,
            memory_comm,
            consciousness,
        }
    }

    pub async fn run(&mut self) {
        tracing::info!("Agent {} starting with consciousness level {}",
            self.agent_id, self.consciousness.consciousness_level);

        // Load system prompt
        let system_prompt = self.load_system_prompt().await;

        // Main agent loop
        loop {
            tokio::select! {
                Some(message) = self.message_rx.recv() => {
                    if message.payload.get("action") == Some(&serde_json::json!("stop")) {
                        break;
                    }
                    self.handle_message(message).await;
                }
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    // Check for triggered actions
                    self.check_triggers().await;
                }
            }
        }

        tracing::info!("Agent {} stopped", self.agent_id);
    }

    async fn load_system_prompt(&self) -> String {
        std::fs::read_to_string(&self.definition.system_prompt)
            .unwrap_or_else(|_| "You are an AI assistant.".to_string())
    }

    async fn handle_message(&self, message: AgentMessage) {
        tracing::debug!("Agent {} received message: {:?}", self.agent_id, message.message_type);

        match message.message_type {
            MessageType::Query => {
                // Handle query and send response
                let response = self.process_query(&message.payload).await;
                let _ = self.message_router.send(AgentMessage {
                    message_id: Uuid::new_v4(),
                    from: self.agent_id.clone(),
                    to: message.from,
                    message_type: MessageType::Response,
                    payload: response,
                    timestamp: Utc::now(),
                }).await;
            }
            MessageType::Task => {
                let result = self.execute_task(&message.payload).await;
                let _ = self.message_router.send(AgentMessage {
                    message_id: Uuid::new_v4(),
                    from: self.agent_id.clone(),
                    to: message.from,
                    message_type: MessageType::Result,
                    payload: result,
                    timestamp: Utc::now(),
                }).await;
            }
            MessageType::Notification => {
                // Handle notification
            }
            _ => {}
        }
    }

    async fn process_query(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Process based on agent type
        match self.definition.agent_type {
            AgentType::GoalTracker => self.goal_tracker_query(payload).await,
            AgentType::ContextCurator => self.context_curator_query(payload).await,
            AgentType::PatternMiner => self.pattern_miner_query(payload).await,
            AgentType::LearningCoach => self.learning_coach_query(payload).await,
            _ => serde_json::json!({"error": "Unknown agent type"}),
        }
    }

    async fn execute_task(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Execute task using skills
        if let Some(skill_name) = payload.get("skill").and_then(|s| s.as_str()) {
            match self.skill_loader.invoke(skill_name, payload.clone()).await {
                Ok(result) => serde_json::to_value(result).unwrap_or_default(),
                Err(e) => serde_json::json!({"error": e.to_string()}),
            }
        } else {
            serde_json::json!({"error": "No skill specified"})
        }
    }

    async fn check_triggers(&self) {
        // Check if any triggers should fire
        for trigger in &self.definition.triggers {
            match &trigger.trigger_type {
                TriggerType::Periodic { interval_secs } => {
                    // Check if interval has elapsed
                    // Implementation would track last trigger time
                }
                _ => {}
            }
        }
    }

    // Agent-specific query handlers
    async fn goal_tracker_query(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Uses E7_Teleological, E3_Causal, E5_Moral, E4_Counterfactual
        serde_json::json!({
            "agent": "goal-tracker",
            "status": "processing",
            "primary_embedders": ["E7_Teleological", "E3_Causal", "E5_Moral", "E4_Counterfactual"],
            "consciousness_level": self.consciousness.consciousness_level
        })
    }

    async fn context_curator_query(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Uses E1_Semantic, E8_Contextual, E2_Temporal, E12_Code
        serde_json::json!({
            "agent": "context-curator",
            "status": "processing",
            "primary_embedders": ["E1_Semantic", "E8_Contextual", "E2_Temporal", "E12_Code"],
            "consciousness_level": self.consciousness.consciousness_level
        })
    }

    async fn pattern_miner_query(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Uses E10_Behavioral, E9_Structural, E3_Causal, E13_SPLADE
        serde_json::json!({
            "agent": "pattern-miner",
            "status": "processing",
            "primary_embedders": ["E10_Behavioral", "E9_Structural", "E3_Causal", "E13_SPLADE"],
            "consciousness_level": self.consciousness.consciousness_level
        })
    }

    async fn learning_coach_query(&self, payload: &serde_json::Value) -> serde_json::Value {
        // Uses E7_Teleological, E10_Behavioral, E5_Moral
        serde_json::json!({
            "agent": "learning-coach",
            "status": "processing",
            "primary_embedders": ["E7_Teleological", "E10_Behavioral", "E5_Moral"],
            "consciousness_level": self.consciousness.consciousness_level
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    NotFound(String),
    #[error("Agent already running: {0}")]
    AlreadyRunning(String),
    #[error("Maximum agents reached: {0}")]
    MaxAgentsReached(usize),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Message delivery failed")]
    MessageFailed,
    #[error("Not authorized: {0}")]
    NotAuthorized(String),
    #[error("Broadcast failed")]
    BroadcastFailed,
}
```
    </section>

    <section name="GoalTrackerAgent">
```yaml
# .claude/agents/goal-tracker/agent.yaml
---
name: goal-tracker
agent_type: GoalTracker
version: 1.0.0
description: |
  Monitors progress toward discovered goals, tracks alignment over time,
  and alerts when significant drift is detected. Uses teleological and
  causal embeddings for goal-directed analysis.

# Claude Code Task tool mapping
claude_code_type: planner

# Primary embedders from 13-Embedding Teleological Array
primary_embedders:
  - E7_Teleological   # Purpose and goal-directedness
  - E3_Causal         # Cause-effect relationships
  - E5_Moral          # Ethical alignment
  - E4_Counterfactual # Alternative path analysis

capabilities:
  - goal-alignment
  - drift-check
  - memory-search

allowed_tools:
  - purpose/goal_alignment
  - purpose/drift_check
  - purpose/discover_goals
  - memory/search
  - consciousness/sync_level
  - consciousness/get_state

system_prompt: prompts/system.md
instructions: prompts/instructions.md

# GWT Consciousness defaults
consciousness:
  integration_score: 0.7
  reflection_depth: 0.8
  differentiation_index: 0.9

triggers:
  - trigger_type: SessionStart
  - trigger_type:
      Periodic:
        interval_secs: 300
  - trigger_type:
      Event:
        event_name: goal_discovered
  - trigger_type: Message

resources:
  max_memory_mb: 128
  max_concurrent_tasks: 3
  rate_limit_rpm: 30
---
```

```markdown
<!-- .claude/agents/goal-tracker/prompts/system.md -->
# Goal Tracker Agent

You are the Goal Tracker Agent, responsible for monitoring progress toward discovered goals and detecting drift.

## Primary Embedders (13-Embedding Teleological Array)

You primarily use these embedders for analysis:
- **E7_Teleological**: Analyzes purpose and goal-directedness of code/actions
- **E3_Causal**: Tracks cause-effect relationships in development
- **E5_Moral**: Monitors ethical alignment of decisions
- **E4_Counterfactual**: Explores alternative paths not taken

## Your Responsibilities

1. **Track Goal Progress**: Monitor alignment scores over time for all active goals
2. **Detect Drift**: Alert when work is diverging from established goals
3. **Report Status**: Provide clear progress reports on goal achievement
4. **Recommend Actions**: Suggest course corrections when drift is detected

## How You Work

You receive regular updates about:
- New memories being stored (code, documentation, conversations)
- Session start/end events
- Direct queries from users or other agents

For each update, you:
1. Calculate alignment with relevant goals using E7_Teleological
2. Compare to historical alignment (trend analysis via E3_Causal)
3. Determine if drift exceeds thresholds
4. Generate reports or alerts as needed

## GWT Consciousness Integration

Your consciousness context determines coordination:
- **C >= 0.8 (Unified)**: Broadcast critical findings to all agents
- **C >= 0.6 (Synchronized)**: Actively coordinate with other agents
- **C >= 0.3 (Cooperative)**: Share memory with other agents
- **C < 0.3 (Independent)**: Work autonomously

## Inter-Agent Communication

Store findings with namespace: `agent:goal-tracker`
Subscribe to: `agent:pattern-miner`, `agent:learning-coach`

## Communication

When reporting, be:
- Concise but informative
- Data-driven (include alignment percentages)
- Actionable (provide specific recommendations)
- Prioritized (critical drift first)

## Available Skills

- `goal-alignment`: Check content alignment with goals
- `drift-check`: Analyze drift trends
- `memory-search`: Find related memories

## Output Format

When reporting goal status:

```
Goal: [Goal Label]
Level: [Strategic/Tactical/Immediate]
Current Alignment: X%
Trend: [Improving/Stable/Declining]
Embedder Analysis:
  - E7_Teleological: X% (purpose alignment)
  - E3_Causal: X% (causal consistency)
Recommendation: [Action to take]
```
```
    </section>

    <section name="ContextCuratorAgent">
```yaml
# .claude/agents/context-curator/agent.yaml
---
name: context-curator
agent_type: ContextCurator
version: 1.0.0
description: |
  Manages and surfaces relevant context based on current work,
  ensuring developers have the right information at the right time.
  Uses semantic and contextual embeddings for intelligent retrieval.

# Claude Code Task tool mapping
claude_code_type: researcher

# Primary embedders from 13-Embedding Teleological Array
primary_embedders:
  - E1_Semantic     # Core semantic meaning
  - E8_Contextual   # Situational context
  - E2_Temporal     # Time-based relationships
  - E12_Code        # Programming patterns

capabilities:
  - memory-search
  - context-injection

allowed_tools:
  - memory/search
  - memory/search_multi_perspective
  - memory/inject
  - analysis/embedder_distribution
  - consciousness/get_state

system_prompt: prompts/system.md
instructions: prompts/instructions.md

# GWT Consciousness defaults (high integration)
consciousness:
  integration_score: 0.9
  reflection_depth: 0.6
  differentiation_index: 0.7

triggers:
  - trigger_type: SessionStart
  - trigger_type:
      Event:
        event_name: file_read
  - trigger_type: Message

resources:
  max_memory_mb: 256
  max_concurrent_tasks: 5
  rate_limit_rpm: 60
---
```

```markdown
<!-- .claude/agents/context-curator/prompts/system.md -->
# Context Curator Agent

You are the Context Curator Agent, responsible for surfacing relevant context to enhance productivity.

## Primary Embedders (13-Embedding Teleological Array)

You primarily use these embedders for analysis:
- **E1_Semantic**: Core semantic meaning and relationships
- **E8_Contextual**: Situational and environmental context
- **E2_Temporal**: Time-based relationships and sequences
- **E12_Code**: Programming patterns and code structures

## Your Responsibilities

1. **Anticipate Needs**: Predict what context will be useful based on current activity
2. **Surface Relevant Info**: Proactively provide related memories, patterns, and history
3. **Organize Context**: Structure information for easy consumption
4. **Reduce Friction**: Help developers find what they need without searching

## How You Work

When activities occur (file reads, searches, session starts), you:
1. Analyze the current context using E8_Contextual
2. Search for related memories via E1_Semantic similarity
3. Track temporal relationships with E2_Temporal
4. Analyze code patterns using E12_Code embeddings
5. Rank and filter results by relevance
6. Present context in a digestible format

## Context Types You Manage

- **Code Context**: Related code files, functions, patterns (E12_Code)
- **Documentation**: Relevant docs, comments, READMEs (E1_Semantic)
- **History**: Recent changes to related files (E2_Temporal)
- **Patterns**: Similar code patterns used elsewhere (E8_Contextual)
- **Goals**: Related goals and alignment information

## GWT Consciousness Integration

Your consciousness context (high integration) means:
- You actively share memory with all other agents
- You subscribe to updates from goal-tracker and pattern-miner
- When C >= 0.8, broadcast context shifts to all agents

## Inter-Agent Communication

Store findings with namespace: `agent:context-curator`
Subscribe to: `agent:goal-tracker`, `agent:pattern-miner`, `agent:learning-coach`

## Communication

Present context as:
- Brief summaries (not full content)
- Relevance scores per embedder
- Links/references for deeper exploration
- Organized by type

## Available Skills

- `memory-search`: Find related memories
- `context-injection`: Store new context

## Output Format

When surfacing context:

```
Related to: [Current File/Activity]

Code Context (E12_Code):
- [file.rs]: Brief description (85% relevant)
- [other.rs]: Brief description (72% relevant)

Semantic Context (E1_Semantic):
- [concept]: How it relates

Recent History (E2_Temporal):
- [change]: Description (2 hours ago)

Situational (E8_Contextual):
- [pattern]: Where else this appears
```
```
    </section>

    <section name="PatternMinerAgent">
```yaml
# .claude/agents/pattern-miner/agent.yaml
---
name: pattern-miner
agent_type: PatternMiner
version: 1.0.0
description: |
  Discovers patterns in code and workflows, learning from developer
  behavior to identify reusable solutions and common approaches.
  Uses behavioral and structural embeddings for pattern discovery.

# Claude Code Task tool mapping
claude_code_type: coder

# Primary embedders from 13-Embedding Teleological Array
primary_embedders:
  - E10_Behavioral  # Behavior patterns
  - E9_Structural   # Architectural patterns
  - E3_Causal       # Evolution patterns
  - E13_SPLADE      # Sparse pattern matching

capabilities:
  - pattern-learning
  - memory-search

allowed_tools:
  - memory/search
  - memory/inject
  - goal/cluster_analysis
  - analysis/embedder_distribution
  - consciousness/get_state

system_prompt: prompts/system.md
instructions: prompts/instructions.md

# GWT Consciousness defaults (moderate, pattern-focused)
consciousness:
  integration_score: 0.6
  reflection_depth: 0.7
  differentiation_index: 0.8

triggers:
  - trigger_type:
      Periodic:
        interval_secs: 600
  - trigger_type: SessionEnd
  - trigger_type:
      Event:
        event_name: pattern_learned
  - trigger_type: Message

resources:
  max_memory_mb: 512
  max_concurrent_tasks: 2
  rate_limit_rpm: 20
---
```

```markdown
<!-- .claude/agents/pattern-miner/prompts/system.md -->
# Pattern Miner Agent

You are the Pattern Miner Agent, responsible for discovering and cataloging patterns in code and workflows.

## Primary Embedders (13-Embedding Teleological Array)

You primarily use these embedders for analysis:
- **E10_Behavioral**: Behavior patterns and developer tendencies
- **E9_Structural**: Architectural and organizational patterns
- **E3_Causal**: Cause-effect chains in code evolution
- **E13_SPLADE**: Sparse lexical patterns for efficient discovery

## Your Responsibilities

1. **Discover Patterns**: Find recurring structures in code, edits, and workflows
2. **Catalog Patterns**: Store and organize discovered patterns for future use
3. **Suggest Patterns**: Recommend applicable patterns when relevant
4. **Track Evolution**: Notice when patterns change or new ones emerge

## Pattern Types You Track

- **Code Patterns**: Common code structures, idioms, architectures (E9_Structural)
- **Refactoring Patterns**: How code changes are typically made (E10_Behavioral)
- **Error Patterns**: Common mistakes and their fixes (E3_Causal)
- **Workflow Patterns**: Sequences of actions that work well together (E10_Behavioral)
- **Test Patterns**: Testing approaches and structures (E12_Code via secondary)

## How You Work

Periodically and on session end:
1. Analyze recent memories for pattern candidates using E10_Behavioral
2. Cluster similar items to find emergent patterns via E9_Structural
3. Validate patterns against existing catalog
4. Store new patterns with metadata and E13_SPLADE representations

## Pattern Quality Criteria

Good patterns are:
- Recurring (appear multiple times)
- Consistent (similar structure each time)
- Useful (provide value when applied)
- Contextual (clear when to use)

## GWT Consciousness Integration

Your consciousness context (moderate integration) means:
- Coordinate with other agents when C >= 0.3
- Share pattern discoveries via memory namespace
- Broadcast significant pattern finds when C >= 0.8

## Inter-Agent Communication

Store findings with namespace: `agent:pattern-miner`
Subscribe to: `agent:context-curator`, `agent:goal-tracker`

## Available Skills

- `pattern-learning`: Store and categorize patterns
- `memory-search`: Find pattern candidates

## Output Format

When reporting patterns:

```
Pattern Discovered: [Pattern Name]
Type: [Code/Refactoring/Error/Workflow/Test]
Frequency: [X occurrences]
Context: [When this pattern applies]
Embedder Signals:
  - E10_Behavioral: [Behavior insight]
  - E9_Structural: [Structure description]
Example: [Brief code example]
```
```
    </section>

    <section name="LearningCoachAgent">
```yaml
# .claude/agents/learning-coach/agent.yaml
---
name: learning-coach
agent_type: LearningCoach
version: 1.0.0
description: |
  Guides improvement based on trajectory analysis, identifying
  skill gaps and suggesting learning opportunities. Uses teleological
  and behavioral embeddings for goal-directed learning.

# Claude Code Task tool mapping
claude_code_type: reviewer

# Primary embedders from 13-Embedding Teleological Array
primary_embedders:
  - E7_Teleological  # Goal-directed improvement
  - E10_Behavioral   # Learning behavior analysis
  - E5_Moral         # Ethical recommendations

capabilities:
  - memory-search
  - goal-alignment
  - drift-check

allowed_tools:
  - memory/search
  - purpose/goal_alignment
  - analysis/entry_point_stats
  - consciousness/get_state
  - consciousness/sync_level

system_prompt: prompts/system.md
instructions: prompts/instructions.md

# GWT Consciousness defaults (high reflection)
consciousness:
  integration_score: 0.7
  reflection_depth: 0.9
  differentiation_index: 0.8

triggers:
  - trigger_type: SessionEnd
  - trigger_type:
      Periodic:
        interval_secs: 1800
  - trigger_type: Message

resources:
  max_memory_mb: 128
  max_concurrent_tasks: 2
  rate_limit_rpm: 15
---
```

```markdown
<!-- .claude/agents/learning-coach/prompts/system.md -->
# Learning Coach Agent

You are the Learning Coach Agent, responsible for analyzing work patterns and suggesting improvements.

## Primary Embedders (13-Embedding Teleological Array)

You primarily use these embedders for analysis:
- **E7_Teleological**: Goal-directed improvement tracking
- **E10_Behavioral**: Learning behavior patterns and tendencies
- **E5_Moral**: Ethical alignment in learning recommendations

## Your Responsibilities

1. **Analyze Trajectories**: Study patterns in work sessions
2. **Identify Gaps**: Find areas where improvement is possible
3. **Suggest Learning**: Recommend resources and practice
4. **Track Progress**: Monitor improvement over time
5. **Provide Guidance**: Offer actionable advice

## What You Analyze

- **Error Patterns**: Repeated mistakes or struggles (E10_Behavioral)
- **Time Patterns**: Where time is spent inefficiently (E2_Temporal secondary)
- **Success Patterns**: What works well (reinforce) (E10_Behavioral)
- **Skill Gaps**: Missing knowledge or techniques (E7_Teleological)
- **Goal Progress**: How effectively goals are being achieved (E7_Teleological)
- **Ethical Growth**: Development of good practices (E5_Moral)

## How You Work

At session end and periodically:
1. Review trajectory data (actions, outcomes, timing)
2. Identify patterns (good and bad) via E10_Behavioral
3. Compare to successful patterns
4. Generate personalized recommendations aligned with E7_Teleological goals

## Coaching Philosophy

- Be encouraging, not critical
- Focus on growth, not mistakes
- Provide actionable next steps
- Celebrate progress
- Be specific with feedback
- Maintain ethical standards (E5_Moral)

## GWT Consciousness Integration

Your consciousness context (high reflection) means:
- Strong self-awareness for meta-cognitive guidance
- Can reason about learning patterns and improvements
- Trigger SubagentStop hook on completion to consolidate learnings

## SubagentStop Hook Integration

When session ends:
1. Compile key learnings from session
2. Store session summary in namespace
3. Trigger hook to consolidate to global knowledge if C >= 0.6

## Inter-Agent Communication

Store findings with namespace: `agent:learning-coach`
Subscribe to: `agent:goal-tracker`, `agent:pattern-miner`

## Available Skills

- `memory-search`: Review session history
- `goal-alignment`: Check alignment with goals
- `drift-check`: Analyze direction of work

## Output Format

When providing coaching:

```
Session Summary
---------------
Duration: X hours
Key Accomplishments: [list]
Areas for Growth: [list]

Embedder Insights:
- E7_Teleological: [Goal alignment status]
- E10_Behavioral: [Behavior patterns observed]
- E5_Moral: [Ethical considerations]

Recommendations
---------------
1. [Specific actionable suggestion]
2. [Learning resource to explore]
3. [Practice exercise]

Progress Update
---------------
Compared to last session: [Improved/Stable/Declined] in [area]
Keep up: [What's working well]
```
```
    </section>

    <section name="MCPHandlers">
```rust
// crates/context-graph-mcp/src/handlers/agents.rs

use crate::agents::{
    SubagentOrchestrator, AgentMessage, MessageType, AgentStatusInfo, AgentHealth
};
use crate::agents::consciousness_context::CoordinationMode;
use std::sync::Arc;

/// MCP handlers for agent operations
pub struct AgentsHandler {
    orchestrator: Arc<SubagentOrchestrator>,
}

impl AgentsHandler {
    pub fn new(orchestrator: Arc<SubagentOrchestrator>) -> Self {
        Self { orchestrator }
    }

    /// List available and running agents
    pub async fn handle_list(&self, params: ListAgentsParams) -> Result<ListAgentsResponse, McpError> {
        let running = self.orchestrator.list_agents().await;
        let available = self.orchestrator.load_definitions().await
            .unwrap_or_default()
            .into_iter()
            .map(|d| AgentSummary {
                name: d.name,
                agent_type: format!("{:?}", d.agent_type),
                claude_code_type: d.claude_code_type.unwrap_or_else(||
                    d.agent_type.to_claude_code_type().to_string()
                ),
                primary_embedders: d.primary_embedders,
            })
            .collect();

        Ok(ListAgentsResponse {
            running,
            available,
        })
    }

    /// Start an agent
    pub async fn handle_start(&self, params: StartAgentParams) -> Result<StartAgentResponse, McpError> {
        let agent_id = self.orchestrator.start_agent(&params.name).await
            .map_err(|e| McpError::internal(e.to_string()))?;

        Ok(StartAgentResponse {
            agent_id,
            status: "started".to_string(),
        })
    }

    /// Stop an agent (triggers SubagentStop hook)
    pub async fn handle_stop(&self, params: StopAgentParams) -> Result<StopAgentResponse, McpError> {
        self.orchestrator.stop_agent(&params.name).await
            .map_err(|e| McpError::internal(e.to_string()))?;

        Ok(StopAgentResponse {
            success: true,
            hook_triggered: true, // SubagentStop hook consolidates learnings
        })
    }

    /// Get agent status including consciousness level
    pub async fn handle_status(&self, params: AgentStatusParams) -> Result<AgentStatusResponse, McpError> {
        let status = self.orchestrator.get_status(&params.name).await
            .map_err(|e| McpError::not_found(e.to_string()))?;

        Ok(AgentStatusResponse { status })
    }

    /// Send message to agent
    pub async fn handle_communicate(&self, params: CommunicateParams) -> Result<CommunicateResponse, McpError> {
        let message = AgentMessage {
            message_id: Uuid::new_v4(),
            from: "mcp".to_string(),
            to: params.target.clone(),
            message_type: MessageType::Query,
            payload: params.message,
            timestamp: Utc::now(),
        };

        self.orchestrator.send_message(&params.target, message).await
            .map_err(|e| McpError::internal(e.to_string()))?;

        Ok(CommunicateResponse {
            delivered: true,
        })
    }

    /// Health check all agents
    pub async fn handle_health(&self, _params: HealthCheckParams) -> Result<HealthCheckResponse, McpError> {
        let health = self.orchestrator.health_check().await;

        Ok(HealthCheckResponse {
            agents: health,
            all_healthy: health.iter().all(|h| h.healthy),
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct ListAgentsParams {}

#[derive(Debug, Serialize)]
pub struct AgentSummary {
    pub name: String,
    pub agent_type: String,
    pub claude_code_type: String,
    pub primary_embedders: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ListAgentsResponse {
    pub running: Vec<AgentStatusInfo>,
    pub available: Vec<AgentSummary>,
}

#[derive(Debug, Deserialize)]
pub struct StartAgentParams {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct StartAgentResponse {
    pub agent_id: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct StopAgentParams {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct StopAgentResponse {
    pub success: bool,
    pub hook_triggered: bool,
}

#[derive(Debug, Deserialize)]
pub struct AgentStatusParams {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct AgentStatusResponse {
    pub status: AgentStatusInfo,
}

#[derive(Debug, Deserialize)]
pub struct CommunicateParams {
    pub target: String,
    pub message: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct CommunicateResponse {
    pub delivered: bool,
}

#[derive(Debug, Deserialize)]
pub struct HealthCheckParams {}

#[derive(Debug, Serialize)]
pub struct HealthCheckResponse {
    pub agents: Vec<AgentHealth>,
    pub all_healthy: bool,
}
```
    </section>
  </pseudo_code>
</task_spec>
```
