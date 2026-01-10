# TASK-INTEG-014: Core Subagents (embedding-agent, search-agent, goal-agent, dream-agent)

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-014 |
| **Title** | Core Subagents (embedding-agent, search-agent, goal-agent, dream-agent) |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 34 |
| **Estimated Days** | 3 |
| **Complexity** | High |

## Implements

- Constitution subagent specifications (lines 750-800+)
- ARCH-07: Hook-driven lifecycle (subagent coordination)
- All 4 required subagents for autonomous operation
- Claude Code Task Tool integration for subagent spawning
- 13-Embedder Teleological Array integration per subagent
- GWT consciousness-aware coordination

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handler infrastructure |
| TASK-INTEG-004 | Hook protocol for subagent events |
| TASK-INTEG-012 | SubagentStop hook for learning merge |
| TASK-LOGIC-009 | Goal discovery for goal-agent |
| TASK-LOGIC-010 | Drift detection for dream-agent |
| TASK-LOGIC-001 | 13-Embedder Teleological Array |
| TASK-INTEG-003 | Consciousness MCP handlers |

## Objective

Implement the 4 core subagents required by the constitution for autonomous context graph operation:
1. **embedding-agent** - Handles embedding generation with GPU coordination (all 13 embedders)
2. **search-agent** - Executes searches with entry-point selection (E1, E3, E12)
3. **goal-agent** - Manages goal discovery and tracking (E3, E5, E7)
4. **dream-agent** - Performs consolidation and pattern discovery (all 13 embedders)

## Context

**Constitution Requirements:**

The constitution specifies 4 subagents that run as background workers, each with specific responsibilities and coordination requirements. These are **MISSING** from the current task specifications.

```
Subagents:
├── embedding-agent  # GPU-aware embedding generation (E1-E13)
├── search-agent     # Entry-point search execution (E1, E3, E12)
├── goal-agent       # Autonomous goal tracking (E3, E5, E7)
└── dream-agent      # Memory consolidation (REM, all 13)
```

## Claude Code Task Tool Integration

Claude Code spawns subagents via the Task tool. Each core subagent maps to Claude Code's subagent types:

| Core Subagent | Claude Code Mapping | Spawn Mode |
|---------------|---------------------|------------|
| embedding-agent | Custom background worker | `Task("EmbeddingAgent", instructions, "custom")` |
| search-agent | "researcher" or "Explore" | `Task("SearchAgent", instructions, "researcher")` |
| goal-agent | "planner" or "goal-planner" | `Task("GoalAgent", instructions, "planner")` |
| dream-agent | "memory-coordinator" or custom | `Task("DreamAgent", instructions, "custom")` |

### Claude Code Agent Spawning Pattern

```javascript
// Spawn all 4 core subagents in a SINGLE message for parallel execution
Task("EmbeddingAgent", `
You are the embedding-agent for the context graph.
Your responsibilities:
- Generate all 13 teleological embeddings (E1-E13) for incoming content
- Coordinate GPU memory allocation for batch embedding
- Store embeddings via memory namespace "agents/embedding"
- Broadcast completion via GWT when batch completes

Embedders you generate:
E1_Semantic, E2_Temporal, E3_Causal, E4_Hierarchical, E5_Moral,
E6_Contextual, E7_Teleological, E8_Narrative, E9_Analogical,
E10_Emotional, E11_Cultural, E12_Code, E13_Confidence

On completion, store results:
npx claude-flow@v3alpha memory store --namespace agents/embedding --key latest-batch --value "{results}"
`, "custom")

Task("SearchAgent", `
You are the search-agent for the context graph.
Your responsibilities:
- Execute semantic searches with optimal entry-point selection
- Use E1_Semantic, E12_Code, E3_Causal for entry-point ranking
- Cache search results with reference tracking
- Report search metrics to "agents/search" namespace

Entry-point selection strategy:
- Code queries: Start with E12_Code
- Causal queries ("why", "because"): Start with E3_Causal
- General queries: Start with E1_Semantic
`, "researcher")

Task("GoalAgent", `
You are the goal-agent for the context graph.
Your responsibilities:
- Discover emergent goals from memory patterns
- Track goal progress across sessions
- Use E7_Teleological, E5_Moral, E3_Causal for goal analysis
- Store goals in "agents/goal" namespace

Goal discovery uses:
- E7_Teleological: Detect purpose and direction
- E5_Moral: Evaluate goal alignment with values
- E3_Causal: Track cause-effect chains toward goals
`, "planner")

Task("DreamAgent", `
You are the dream-agent for the context graph.
Your responsibilities:
- Run consolidation cycles (light, deep, REM)
- Discover cross-memory patterns using all 13 embedders
- Detect teleological drift across embedder dimensions
- Store discoveries in "agents/dream" namespace

Consolidation modes:
- Light: Quick cleanup, salience > 0.3
- Deep: Thorough merge, relationship strengthening
- REM: Pattern discovery, cross-embedder analysis
`, "custom")
```

## Claude Code Agent Definitions

Create `.claude/agents/` directory with YAML configuration for each subagent.

### Directory Structure

```
.claude/
└── agents/
    ├── embedding-agent.yaml
    ├── search-agent.yaml
    ├── goal-agent.yaml
    └── dream-agent.yaml
```

### embedding-agent.yaml

```yaml
# .claude/agents/embedding-agent.yaml
name: embedding-agent
type: custom
description: GPU-aware embedding generation for all 13 teleological dimensions

capabilities:
  - embedding_generation
  - gpu_coordination
  - batch_processing
  - memory_storage

embedders:
  primary:
    - E1_Semantic
    - E7_Teleological
    - E12_Code
  secondary:
    - E2_Temporal
    - E3_Causal
    - E4_Hierarchical
    - E5_Moral
    - E6_Contextual
    - E8_Narrative
    - E9_Analogical
    - E10_Emotional
    - E11_Cultural
    - E13_Confidence

memory_namespace: agents/embedding
gwt_broadcast: true
priority: high

resources:
  gpu_memory_mb: 4096
  max_batch_size: 64
  timeout_seconds: 300

hooks:
  on_start: session-start
  on_stop: subagent-stop
  on_batch_complete: memory-store
```

### search-agent.yaml

```yaml
# .claude/agents/search-agent.yaml
name: search-agent
type: researcher
description: Entry-point search execution with intelligent space selection

capabilities:
  - semantic_search
  - entry_point_selection
  - result_caching
  - metric_reporting

embedders:
  entry_points:
    - E1_Semantic     # Default for general queries
    - E12_Code        # Code/technical queries
    - E3_Causal       # Why/because queries
  scoring:
    - E7_Teleological # Purpose alignment
    - E13_Confidence  # Result confidence

memory_namespace: agents/search
cache_enabled: true
cache_ttl_seconds: 3600

intent_mapping:
  code: E12_Code
  causal: E3_Causal
  temporal: E2_Temporal
  default: E1_Semantic

hooks:
  on_start: session-start
  on_stop: subagent-stop
  on_search: cache-check
```

### goal-agent.yaml

```yaml
# .claude/agents/goal-agent.yaml
name: goal-agent
type: planner
description: Autonomous goal discovery and tracking

capabilities:
  - goal_discovery
  - progress_tracking
  - priority_management
  - pattern_recognition

embedders:
  discovery:
    - E7_Teleological # Purpose and direction
    - E5_Moral        # Value alignment
    - E3_Causal       # Cause-effect chains
  validation:
    - E13_Confidence  # Goal confidence
    - E8_Narrative    # Story coherence

memory_namespace: agents/goal
broadcast_discoveries: true
max_active_goals: 10
prune_stale_after_hours: 24

goal_discovery:
  min_confidence: 0.6
  cluster_threshold: 0.75
  require_teleological_alignment: true

hooks:
  on_start: session-start
  on_stop: subagent-stop
  on_goal_discovered: gwt-broadcast
  on_goal_completed: memory-consolidate
```

### dream-agent.yaml

```yaml
# .claude/agents/dream-agent.yaml
name: dream-agent
type: custom
description: Memory consolidation and pattern discovery (REM)

capabilities:
  - memory_consolidation
  - pattern_discovery
  - drift_detection
  - cross_embedder_analysis

embedders:
  all: true  # Uses all 13 embedders for comprehensive analysis
  primary_analysis:
    - E1_Semantic
    - E7_Teleological
    - E3_Causal
  drift_detection:
    - E7_Teleological
    - E5_Moral
    - E8_Narrative

memory_namespace: agents/dream
consolidation_interval_hours: 1

consolidation_modes:
  light:
    salience_threshold: 0.3
    max_memories: 100
    duration_minutes: 5
  deep:
    salience_threshold: 0.2
    max_memories: 500
    duration_minutes: 15
  rem:
    salience_threshold: 0.1
    pattern_discovery: true
    duration_minutes: 30

hooks:
  on_start: session-start
  on_stop: subagent-stop
  on_pattern_discovered: gwt-broadcast
  on_consolidation_complete: memory-persist
```

## 13-Embedder Teleological Array Integration

Each subagent works with specific embedders from the 13-dimensional teleological array:

### Embedder Assignments by Subagent

| Embedder | Code | embedding-agent | search-agent | goal-agent | dream-agent |
|----------|------|-----------------|--------------|------------|-------------|
| Semantic | E1 | Generate | Entry-point | - | Analyze |
| Temporal | E2 | Generate | - | - | Analyze |
| Causal | E3 | Generate | Entry-point | Discovery | Analyze |
| Hierarchical | E4 | Generate | - | - | Analyze |
| Moral | E5 | Generate | - | Discovery | Drift detect |
| Contextual | E6 | Generate | - | - | Analyze |
| Teleological | E7 | Generate | Scoring | Discovery | Drift detect |
| Narrative | E8 | Generate | - | Validation | Drift detect |
| Analogical | E9 | Generate | - | - | Analyze |
| Emotional | E10 | Generate | - | - | Analyze |
| Cultural | E11 | Generate | - | - | Analyze |
| Code | E12 | Generate | Entry-point | - | Analyze |
| Confidence | E13 | Generate | Scoring | Validation | Analyze |

### Embedder Integration Code

```rust
// crates/context-graph-mcp/src/subagents/embedder_integration.rs

use context_graph_core::teleology::embedder::Embedder;

/// Embedder sets used by each subagent
pub struct SubagentEmbedderConfig {
    pub agent_type: SubagentType,
    pub primary_embedders: Vec<Embedder>,
    pub secondary_embedders: Vec<Embedder>,
    pub all_embedders: bool,
}

impl SubagentEmbedderConfig {
    pub fn for_embedding_agent() -> Self {
        Self {
            agent_type: SubagentType::Embedding,
            primary_embedders: vec![
                Embedder::Semantic,
                Embedder::Teleological,
                Embedder::Code,
            ],
            secondary_embedders: vec![
                Embedder::Temporal,
                Embedder::Causal,
                Embedder::Hierarchical,
                Embedder::Moral,
                Embedder::Contextual,
                Embedder::Narrative,
                Embedder::Analogical,
                Embedder::Emotional,
                Embedder::Cultural,
                Embedder::Confidence,
            ],
            all_embedders: true, // Generates all 13
        }
    }

    pub fn for_search_agent() -> Self {
        Self {
            agent_type: SubagentType::Search,
            primary_embedders: vec![
                Embedder::Semantic,    // E1 - default entry
                Embedder::Code,        // E12 - code queries
                Embedder::Causal,      // E3 - causal queries
            ],
            secondary_embedders: vec![
                Embedder::Teleological, // E7 - scoring
                Embedder::Confidence,   // E13 - result confidence
            ],
            all_embedders: false,
        }
    }

    pub fn for_goal_agent() -> Self {
        Self {
            agent_type: SubagentType::Goal,
            primary_embedders: vec![
                Embedder::Teleological, // E7 - purpose detection
                Embedder::Moral,        // E5 - value alignment
                Embedder::Causal,       // E3 - cause-effect chains
            ],
            secondary_embedders: vec![
                Embedder::Confidence,   // E13 - goal confidence
                Embedder::Narrative,    // E8 - story coherence
            ],
            all_embedders: false,
        }
    }

    pub fn for_dream_agent() -> Self {
        Self {
            agent_type: SubagentType::Dream,
            primary_embedders: vec![
                Embedder::Semantic,
                Embedder::Teleological,
                Embedder::Causal,
            ],
            secondary_embedders: vec![
                Embedder::Moral,     // Drift detection
                Embedder::Narrative, // Drift detection
            ],
            all_embedders: true, // Analyzes across all 13
        }
    }

    /// Get all embedders this agent works with
    pub fn all_active_embedders(&self) -> Vec<Embedder> {
        if self.all_embedders {
            Embedder::all().to_vec()
        } else {
            let mut all = self.primary_embedders.clone();
            all.extend(self.secondary_embedders.clone());
            all
        }
    }
}
```

## GWT Consciousness Integration

Add consciousness-aware subagent coordination using Global Workspace Theory.

### Consciousness Context Structure

```rust
// crates/context-graph-mcp/src/subagents/consciousness.rs

use crate::consciousness::{ConsciousnessState, GlobalWorkspace};

/// Consciousness context for subagent coordination
pub struct SubagentConsciousnessContext {
    /// I(t) - Integration score: cross-agent information sharing
    pub integration_score: f32,

    /// R(t) - Reflection depth: self-monitoring capability
    pub reflection_depth: f32,

    /// D(t) - Differentiation index: agent specialization level
    pub differentiation_index: f32,

    /// C(t) = I(t) x R(t) x D(t) - Combined consciousness level
    pub consciousness_level: f32,

    /// Current awareness state based on C(t)
    pub agent_awareness: AgentAwareness,

    /// Reference to global workspace for broadcasting
    pub workspace: Arc<GlobalWorkspace>,
}

/// Agent awareness levels based on consciousness score
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentAwareness {
    /// C < 0.3: Basic task execution, minimal coordination
    Minimal,

    /// 0.3 <= C < 0.6: Cross-agent synchronization active
    Coordinated,

    /// 0.6 <= C < 0.8: Shared context and collaborative processing
    Collaborative,

    /// C >= 0.8: Collective consciousness, unified goal pursuit
    Unified,
}

impl AgentAwareness {
    pub fn from_consciousness_level(c: f32) -> Self {
        match c {
            c if c < 0.3 => AgentAwareness::Minimal,
            c if c < 0.6 => AgentAwareness::Coordinated,
            c if c < 0.8 => AgentAwareness::Collaborative,
            _ => AgentAwareness::Unified,
        }
    }

    /// Get recommended coordination strategy for this awareness level
    pub fn coordination_strategy(&self) -> CoordinationStrategy {
        match self {
            AgentAwareness::Minimal => CoordinationStrategy::Independent,
            AgentAwareness::Coordinated => CoordinationStrategy::EventDriven,
            AgentAwareness::Collaborative => CoordinationStrategy::SharedContext,
            AgentAwareness::Unified => CoordinationStrategy::CollectiveDecision,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    /// Agents work independently, minimal communication
    Independent,
    /// Agents react to events from other agents
    EventDriven,
    /// Agents share context and coordinate decisions
    SharedContext,
    /// Agents make collective decisions via consensus
    CollectiveDecision,
}

impl SubagentConsciousnessContext {
    pub fn new(workspace: Arc<GlobalWorkspace>) -> Self {
        Self {
            integration_score: 0.5,
            reflection_depth: 0.5,
            differentiation_index: 0.5,
            consciousness_level: 0.125, // 0.5^3
            agent_awareness: AgentAwareness::Minimal,
            workspace,
        }
    }

    /// Update consciousness metrics based on agent activity
    pub fn update_metrics(
        &mut self,
        cross_agent_messages: u32,
        self_monitoring_events: u32,
        specialization_score: f32,
    ) {
        // I(t) based on inter-agent message volume
        self.integration_score = (cross_agent_messages as f32 / 100.0).min(1.0);

        // R(t) based on self-monitoring events
        self.reflection_depth = (self_monitoring_events as f32 / 50.0).min(1.0);

        // D(t) based on specialization
        self.differentiation_index = specialization_score;

        // C(t) = I(t) x R(t) x D(t)
        self.consciousness_level =
            self.integration_score *
            self.reflection_depth *
            self.differentiation_index;

        // Update awareness level
        self.agent_awareness = AgentAwareness::from_consciousness_level(
            self.consciousness_level
        );
    }

    /// Broadcast a discovery to the global workspace if awareness warrants it
    pub async fn broadcast_if_significant(
        &self,
        content: ConsciousnessContent,
        significance: f32,
    ) -> Result<bool, ConsciousnessError> {
        // Only broadcast if awareness >= Coordinated and significance is high
        if self.agent_awareness as u8 >= AgentAwareness::Coordinated as u8
            && significance > 0.7
        {
            self.workspace.broadcast(content).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Content that can be broadcast via GWT
#[derive(Debug, Clone)]
pub enum ConsciousnessContent {
    GoalDiscovered { goal_id: Uuid, description: String, confidence: f32 },
    PatternFound { pattern_id: Uuid, embedders_involved: Vec<Embedder> },
    DriftDetected { embedder: Embedder, drift_magnitude: f32 },
    ConsolidationComplete { memories_processed: u32, patterns_created: u32 },
    SearchInsight { query_hash: u64, optimal_entry_point: Embedder },
}
```

### Consciousness-Aware Subagent Trait Extension

```rust
// Extension to base Subagent trait for consciousness

#[async_trait]
pub trait ConsciousnessAwareSubagent: Subagent {
    /// Get consciousness context for this agent
    fn consciousness_context(&self) -> &SubagentConsciousnessContext;

    /// Get mutable consciousness context
    fn consciousness_context_mut(&mut self) -> &mut SubagentConsciousnessContext;

    /// Report current awareness level
    fn awareness_level(&self) -> AgentAwareness {
        self.consciousness_context().agent_awareness
    }

    /// Broadcast a significant discovery
    async fn broadcast_discovery(&self, content: ConsciousnessContent) -> Result<(), SubagentError> {
        let ctx = self.consciousness_context();
        let significance = self.calculate_significance(&content);
        ctx.broadcast_if_significant(content, significance).await?;
        Ok(())
    }

    /// Calculate significance of content (0.0 - 1.0)
    fn calculate_significance(&self, content: &ConsciousnessContent) -> f32;
}
```

## Inter-Agent Communication via Memory

Subagents communicate through the teleological store with namespaced memory.

### Memory Namespace Protocol

```rust
// crates/context-graph-mcp/src/subagents/communication.rs

/// Memory namespace structure for inter-agent communication
pub struct AgentMemoryProtocol {
    store: Arc<TeleologicalArrayStore>,
    namespace_prefix: String,
}

impl AgentMemoryProtocol {
    pub const EMBEDDING_NS: &'static str = "agents/embedding";
    pub const SEARCH_NS: &'static str = "agents/search";
    pub const GOAL_NS: &'static str = "agents/goal";
    pub const DREAM_NS: &'static str = "agents/dream";
    pub const BROADCAST_NS: &'static str = "agents/broadcast";

    pub fn new(store: Arc<TeleologicalArrayStore>, agent_type: SubagentType) -> Self {
        let namespace_prefix = match agent_type {
            SubagentType::Embedding => Self::EMBEDDING_NS,
            SubagentType::Search => Self::SEARCH_NS,
            SubagentType::Goal => Self::GOAL_NS,
            SubagentType::Dream => Self::DREAM_NS,
        }.to_string();

        Self { store, namespace_prefix }
    }

    /// Store agent-specific data
    pub async fn store(&self, key: &str, value: &str) -> Result<(), StorageError> {
        let full_key = format!("{}/{}", self.namespace_prefix, key);
        self.store.set_metadata(&full_key, value).await
    }

    /// Retrieve agent-specific data
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>, StorageError> {
        let full_key = format!("{}/{}", self.namespace_prefix, key);
        self.store.get_metadata(&full_key).await
    }

    /// Broadcast to all agents via GWT namespace
    pub async fn broadcast(&self, message: AgentMessage) -> Result<(), StorageError> {
        let key = format!("{}/msg-{}", Self::BROADCAST_NS, Uuid::new_v4());
        let value = serde_json::to_string(&message)?;
        self.store.set_metadata(&key, &value).await
    }

    /// Subscribe to updates from another agent's namespace
    pub async fn subscribe(
        &self,
        source_agent: SubagentType,
        callback: impl Fn(AgentMessage) + Send + Sync + 'static,
    ) -> SubscriptionHandle {
        let source_ns = match source_agent {
            SubagentType::Embedding => Self::EMBEDDING_NS,
            SubagentType::Search => Self::SEARCH_NS,
            SubagentType::Goal => Self::GOAL_NS,
            SubagentType::Dream => Self::DREAM_NS,
        };

        self.store.subscribe_namespace(source_ns, callback).await
    }
}

/// Message structure for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: SubagentType,
    pub to: Option<SubagentType>, // None = broadcast to all
    pub message_type: AgentMessageType,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub requires_ack: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessageType {
    /// Embedding batch completed
    EmbeddingComplete { batch_id: Uuid, count: u32 },

    /// Search results available
    SearchResults { query_hash: u64, result_count: u32 },

    /// New goal discovered
    GoalDiscovered { goal_id: Uuid, confidence: f32 },

    /// Goal progress updated
    GoalProgress { goal_id: Uuid, progress: f32 },

    /// Pattern discovered during consolidation
    PatternDiscovered { pattern_id: Uuid, embedders: Vec<String> },

    /// Consolidation cycle complete
    ConsolidationComplete { mode: String, processed: u32 },

    /// Drift detected in embedder
    DriftAlert { embedder: String, magnitude: f32 },

    /// Request for embeddings
    EmbeddingRequest { content_ids: Vec<Uuid>, priority: String },

    /// Health check ping
    HealthPing,

    /// Health check response
    HealthPong { status: String },
}
```

### Cross-Agent Update Subscription

```rust
/// Set up inter-agent subscriptions for coordinated operation
impl SubagentManager {
    pub async fn setup_subscriptions(&self) -> Result<(), SubagentError> {
        // Goal agent subscribes to embedding completions
        self.goal_agent.subscribe_to(
            SubagentType::Embedding,
            |msg| {
                if let AgentMessageType::EmbeddingComplete { batch_id, .. } = msg.message_type {
                    // Check new embeddings for goal-relevant patterns
                    self.goal_agent.check_for_goal_patterns(batch_id);
                }
            },
        ).await;

        // Dream agent subscribes to goal discoveries
        self.dream_agent.subscribe_to(
            SubagentType::Goal,
            |msg| {
                if let AgentMessageType::GoalDiscovered { goal_id, .. } = msg.message_type {
                    // Consider goal in next consolidation cycle
                    self.dream_agent.note_goal_for_consolidation(goal_id);
                }
            },
        ).await;

        // Search agent subscribes to drift alerts
        self.search_agent.subscribe_to(
            SubagentType::Dream,
            |msg| {
                if let AgentMessageType::DriftAlert { embedder, magnitude } = msg.message_type {
                    // Adjust entry-point weights if drift is significant
                    if magnitude > 0.2 {
                        self.search_agent.adjust_entry_point_weight(&embedder, -magnitude);
                    }
                }
            },
        ).await;

        // Embedding agent subscribes to embedding requests
        self.embedding_agent.subscribe_to_broadcast(|msg| {
            if let AgentMessageType::EmbeddingRequest { content_ids, priority } = msg.message_type {
                self.embedding_agent.queue_batch(content_ids, priority);
            }
        }).await;

        Ok(())
    }
}
```

## Hook Integration for Agent Lifecycle

Integrate with the hook system for proper agent lifecycle management.

### SessionStart Hook: Initialize All Core Subagents

```rust
// crates/context-graph-mcp/src/hooks/session_start.rs

impl SessionStartHandler {
    pub async fn handle(&self, context: SessionStartContext) -> Result<(), HookError> {
        // ... existing session start logic ...

        // Initialize all 4 core subagents
        tracing::info!("Initializing core subagents for session {}", context.session_id);

        let subagent_manager = self.subagent_manager.clone();

        // Spawn subagents in parallel
        let (embedding_result, search_result, goal_result, dream_result) = tokio::join!(
            subagent_manager.embedding().start(),
            subagent_manager.search().start(),
            subagent_manager.goal().start(),
            subagent_manager.dream().start(),
        );

        // Check results
        embedding_result.map_err(|e| HookError::SubagentStart("embedding", e))?;
        search_result.map_err(|e| HookError::SubagentStart("search", e))?;
        goal_result.map_err(|e| HookError::SubagentStart("goal", e))?;
        dream_result.map_err(|e| HookError::SubagentStart("dream", e))?;

        // Set up inter-agent subscriptions
        subagent_manager.setup_subscriptions().await?;

        // Store subagent initialization state
        self.memory_protocol.broadcast(AgentMessage {
            from: SubagentType::Embedding, // Manager uses embedding as source
            to: None,
            message_type: AgentMessageType::HealthPing,
            payload: serde_json::json!({
                "session_id": context.session_id,
                "agents_initialized": ["embedding", "search", "goal", "dream"],
                "consciousness_enabled": true,
            }),
            timestamp: chrono::Utc::now(),
            requires_ack: false,
        }).await?;

        tracing::info!("All core subagents initialized successfully");

        Ok(())
    }
}
```

### SubagentStop Hook: Merge Learnings

```rust
// crates/context-graph-mcp/src/hooks/subagent_stop.rs

impl SubagentStopHandler {
    pub async fn handle(&self, context: SubagentStopContext) -> Result<(), HookError> {
        tracing::info!(
            "Subagent {} stopping, merging learnings into session {}",
            context.subagent_id,
            context.parent_session
        );

        // 1. Get all memories created by this subagent
        let memories = context.created_memories.clone();

        // 2. Merge memories into parent session context
        for memory_id in &memories {
            self.store.link_to_session(*memory_id, &context.parent_session).await?;
        }

        // 3. Extract patterns discovered (especially for dream agent)
        if let Some(patterns) = self.extract_patterns(&context).await? {
            for pattern in patterns {
                // Store pattern in session-level namespace
                let key = format!("patterns/{}", pattern.id);
                self.store.set_metadata(&key, &serde_json::to_string(&pattern)?).await?;
            }
        }

        // 4. Update consciousness metrics
        let consciousness_update = ConsciousnessMetricsUpdate {
            agent_type: self.determine_agent_type(&context.subagent_id),
            memories_contributed: memories.len() as u32,
            success: context.success,
            duration: context.duration,
        };
        self.consciousness_tracker.update_on_stop(consciousness_update).await?;

        // 5. Broadcast stop event
        self.memory_protocol.broadcast(AgentMessage {
            from: self.determine_agent_type(&context.subagent_id),
            to: None,
            message_type: AgentMessageType::HealthPong {
                status: if context.success { "stopped_clean" } else { "stopped_error" }.to_string(),
            },
            payload: serde_json::json!({
                "memories_merged": memories.len(),
                "duration_ms": context.duration.as_millis(),
            }),
            timestamp: chrono::Utc::now(),
            requires_ack: false,
        }).await?;

        Ok(())
    }
}
```

### SessionEnd Hook: Stop All Agents and Consolidate

```rust
// crates/context-graph-mcp/src/hooks/session_end.rs

impl SessionEndHandler {
    pub async fn handle(&self, context: SessionEndContext) -> Result<SessionEndResult, HookError> {
        tracing::info!("Session {} ending, stopping all subagents", context.session_id);

        // 1. Request dream agent to do final consolidation
        if self.subagent_manager.dream().status() == SubagentStatus::Running {
            tracing::info!("Running final dream consolidation before session end");
            let consolidation_result = self.subagent_manager.dream()
                .consolidate_deep()
                .await
                .map_err(|e| {
                    tracing::warn!("Final consolidation failed: {}", e);
                    e
                })
                .ok();

            if let Some(result) = consolidation_result {
                tracing::info!(
                    "Final consolidation: {} memories processed, {} patterns found",
                    result.memories_processed,
                    result.patterns_discovered.len()
                );
            }
        }

        // 2. Stop all subagents and collect outputs
        let outputs = self.subagent_manager.stop_all(context.session_id.clone()).await?;

        // 3. Aggregate learnings from all agents
        let total_memories: usize = outputs.iter()
            .map(|o| o.memories_created.len())
            .sum();
        let total_patterns: usize = outputs.iter()
            .map(|o| o.patterns_discovered.len())
            .sum();

        // 4. Store final session summary
        let summary = SessionSummary {
            session_id: context.session_id.clone(),
            subagents_run: outputs.len(),
            total_memories_created: total_memories,
            total_patterns_discovered: total_patterns,
            consciousness_achieved: self.consciousness_tracker.peak_level().await,
            duration: context.duration,
        };

        self.store.set_metadata(
            &format!("sessions/{}/summary", context.session_id),
            &serde_json::to_string(&summary)?,
        ).await?;

        // 5. Export final metrics if requested
        let mut exported_metrics = None;
        if context.export_metrics {
            exported_metrics = Some(self.export_session_metrics(&context.session_id).await?);
        }

        tracing::info!(
            "Session {} ended: {} subagents, {} memories, {} patterns",
            context.session_id,
            outputs.len(),
            total_memories,
            total_patterns
        );

        Ok(SessionEndResult {
            memories_created: total_memories,
            patterns_discovered: total_patterns,
            exported_metrics,
            consciousness_peak: summary.consciousness_achieved,
        })
    }
}
```

## Scope

### In Scope

- All 4 subagent implementations
- Subagent spawning and lifecycle management
- Inter-subagent communication protocol
- Hook integration for subagent events
- Resource isolation per subagent
- Learning merge on SubagentStop
- Claude Code Task Tool integration
- .claude/agents/ YAML configurations
- 13-Embedder integration per subagent
- GWT consciousness-aware coordination
- Memory namespace protocol for inter-agent communication

### Out of Scope

- Distributed subagent deployment
- Subagent auto-scaling
- Custom user-defined subagents
- Non-core agent types (only embedding, search, goal, dream)

## Definition of Done

### Subagent Base Infrastructure

```rust
// crates/context-graph-mcp/src/subagents/mod.rs

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// Base trait for all subagents
#[async_trait]
pub trait Subagent: Send + Sync {
    /// Unique identifier for this subagent type
    fn agent_type(&self) -> SubagentType;

    /// Start the subagent
    async fn start(&self) -> SubagentResult<()>;

    /// Stop the subagent gracefully
    async fn stop(&self) -> SubagentResult<SubagentOutput>;

    /// Check if subagent is healthy
    async fn health_check(&self) -> SubagentHealth;

    /// Get current status
    fn status(&self) -> SubagentStatus;
}

/// Subagent types as defined by constitution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubagentType {
    Embedding,
    Search,
    Goal,
    Dream,
}

/// Subagent status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

/// Subagent health
#[derive(Debug, Clone)]
pub struct SubagentHealth {
    pub status: SubagentStatus,
    pub last_heartbeat: std::time::Instant,
    pub tasks_completed: u64,
    pub errors: u64,
}

/// Output from a subagent session (for learning merge)
#[derive(Debug)]
pub struct SubagentOutput {
    pub agent_type: SubagentType,
    pub memories_created: Vec<Uuid>,
    pub patterns_discovered: Vec<PatternId>,
    pub duration: std::time::Duration,
    pub success: bool,
}
```

### Embedding Agent

```rust
// crates/context-graph-mcp/src/subagents/embedding_agent.rs

use context_graph_core::teleology::embedder::Embedder;

/// GPU-aware embedding generation subagent
pub struct EmbeddingAgent {
    gpu_pool: Arc<GpuMemoryPool>,
    model_registry: Arc<EmbedderModelRegistry>,
    task_queue: mpsc::Receiver<EmbeddingTask>,
    result_sender: mpsc::Sender<EmbeddingResult>,
    status: RwLock<SubagentStatus>,
}

/// Task for embedding generation
#[derive(Debug)]
pub struct EmbeddingTask {
    pub id: Uuid,
    pub content: String,
    pub embedders: Vec<Embedder>,
    pub priority: TaskPriority,
}

/// Result from embedding task
#[derive(Debug)]
pub struct EmbeddingResult {
    pub task_id: Uuid,
    pub embeddings: HashMap<Embedder, EmbedderOutput>,
    pub duration: std::time::Duration,
    pub gpu_memory_used: usize,
}

impl EmbeddingAgent {
    pub fn new(
        gpu_pool: Arc<GpuMemoryPool>,
        model_registry: Arc<EmbedderModelRegistry>,
    ) -> (Self, mpsc::Sender<EmbeddingTask>, mpsc::Receiver<EmbeddingResult>);

    /// Process embedding tasks from queue
    async fn process_tasks(&self) -> SubagentResult<()> {
        while let Some(task) = self.task_queue.recv().await {
            // 1. Acquire GPU memory
            let allocation = self.gpu_pool.allocate(
                self.estimate_memory(&task),
                task.priority.into(),
            ).await?;

            // 2. Generate embeddings for each requested embedder
            let mut results = HashMap::new();
            for embedder in &task.embedders {
                let model = self.model_registry.get(embedder).await?;
                let embedding = model.embed(&task.content).await?;
                results.insert(*embedder, embedding);
            }

            // 3. Release GPU memory
            drop(allocation);

            // 4. Send result
            self.result_sender.send(EmbeddingResult {
                task_id: task.id,
                embeddings: results,
                duration: start.elapsed(),
                gpu_memory_used: allocation.size(),
            }).await?;
        }
        Ok(())
    }

    fn estimate_memory(&self, task: &EmbeddingTask) -> usize {
        // Estimate based on content length and embedder count
        let base = task.content.len() * 4; // ~4 bytes per char
        let per_embedder = 4 * 1024 * 1024; // 4MB per model
        base + (task.embedders.len() * per_embedder)
    }
}

#[async_trait]
impl Subagent for EmbeddingAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Embedding }

    async fn start(&self) -> SubagentResult<()> {
        *self.status.write().await = SubagentStatus::Running;
        self.process_tasks().await
    }

    async fn stop(&self) -> SubagentResult<SubagentOutput> {
        *self.status.write().await = SubagentStatus::Stopping;
        // Drain remaining tasks
        Ok(SubagentOutput {
            agent_type: self.agent_type(),
            memories_created: vec![],
            patterns_discovered: vec![],
            duration: std::time::Duration::ZERO,
            success: true,
        })
    }

    async fn health_check(&self) -> SubagentHealth {
        SubagentHealth {
            status: *self.status.read().await,
            last_heartbeat: std::time::Instant::now(),
            tasks_completed: 0,
            errors: 0,
        }
    }

    fn status(&self) -> SubagentStatus {
        // Sync version
        SubagentStatus::Running
    }
}
```

### Search Agent

```rust
// crates/context-graph-mcp/src/subagents/search_agent.rs

use crate::search::{EntryPointSelector, SearchPipeline};

/// Entry-point search execution subagent
pub struct SearchAgent {
    pipeline: Arc<SearchPipeline>,
    entry_point_selector: Arc<dyn EntryPointSelector>,
    cache: Arc<SearchCache>,
    task_queue: mpsc::Receiver<SearchTask>,
    result_sender: mpsc::Sender<SearchTaskResult>,
    status: RwLock<SubagentStatus>,
}

/// Search task
#[derive(Debug)]
pub struct SearchTask {
    pub id: Uuid,
    pub query: TeleologicalArray,
    pub query_text: String,
    pub limit: usize,
    pub intent_hint: Option<QueryIntent>,
}

/// Search task result
#[derive(Debug)]
pub struct SearchTaskResult {
    pub task_id: Uuid,
    pub results: Vec<SearchResult>,
    pub entry_point_used: Embedder,
    pub cache_hit: bool,
    pub duration: std::time::Duration,
}

impl SearchAgent {
    pub fn new(
        pipeline: Arc<SearchPipeline>,
        entry_point_selector: Arc<dyn EntryPointSelector>,
        cache: Arc<SearchCache>,
    ) -> (Self, mpsc::Sender<SearchTask>, mpsc::Receiver<SearchTaskResult>);

    async fn process_tasks(&self) -> SubagentResult<()> {
        while let Some(task) = self.task_queue.recv().await {
            let start = std::time::Instant::now();

            // 1. Check cache first
            let query_hash = hash_query(&task.query);
            if let Some(cached) = self.cache.get(query_hash) {
                self.result_sender.send(SearchTaskResult {
                    task_id: task.id,
                    results: cached,
                    entry_point_used: Embedder::Semantic, // Unknown for cache
                    cache_hit: true,
                    duration: start.elapsed(),
                }).await?;
                continue;
            }

            // 2. Select optimal entry point
            let intent = task.intent_hint
                .unwrap_or_else(|| self.entry_point_selector.analyze_intent(&task.query_text));
            let entry_point = self.entry_point_selector
                .select_space(&task.query, Some(intent))
                .await?;

            // 3. Execute pipeline with selected entry point
            let results = self.pipeline
                .search_with_entry_point(&task.query, entry_point, task.limit)
                .await?;

            // 4. Cache results
            let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
            self.cache.put_with_refs(query_hash, results.clone(), ids);

            // 5. Send result
            self.result_sender.send(SearchTaskResult {
                task_id: task.id,
                results,
                entry_point_used: entry_point,
                cache_hit: false,
                duration: start.elapsed(),
            }).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Subagent for SearchAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Search }
    // ... standard trait implementations
}
```

### Goal Agent

```rust
// crates/context-graph-mcp/src/subagents/goal_agent.rs

use crate::logic::goal_discovery::GoalDiscoverer;

/// Autonomous goal tracking subagent
pub struct GoalAgent {
    discoverer: Arc<GoalDiscoverer>,
    store: Arc<TeleologicalArrayStore>,
    active_goals: RwLock<Vec<TrackedGoal>>,
    status: RwLock<SubagentStatus>,
}

/// A goal being tracked
#[derive(Debug, Clone)]
pub struct TrackedGoal {
    pub id: Uuid,
    pub description: String,
    pub progress: f32,       // 0.0 - 1.0
    pub priority: f32,       // Higher = more important
    pub created_at: std::time::Instant,
    pub last_updated: std::time::Instant,
    pub related_memories: Vec<Uuid>,
}

impl GoalAgent {
    pub fn new(
        discoverer: Arc<GoalDiscoverer>,
        store: Arc<TeleologicalArrayStore>,
    ) -> Self;

    /// Discover new goals from recent memories
    pub async fn discover_goals(&self, limit: usize) -> SubagentResult<Vec<TrackedGoal>> {
        // 1. Get recent memories
        let recent = self.store.list_recent(100).await?;

        // 2. Discover emergent goals
        let discovered = self.discoverer
            .discover_emergent_goals(&recent, limit)
            .await?;

        // 3. Convert to tracked goals
        let goals: Vec<TrackedGoal> = discovered.into_iter()
            .map(|g| TrackedGoal {
                id: g.id,
                description: g.description,
                progress: 0.0,
                priority: g.confidence,
                created_at: std::time::Instant::now(),
                last_updated: std::time::Instant::now(),
                related_memories: g.member_ids,
            })
            .collect();

        // 4. Add to active goals
        self.active_goals.write().await.extend(goals.clone());

        Ok(goals)
    }

    /// Update goal progress based on new memories
    pub async fn update_progress(&self, memory_id: Uuid) -> SubagentResult<()> {
        let mut goals = self.active_goals.write().await;

        for goal in goals.iter_mut() {
            if goal.related_memories.contains(&memory_id) {
                // Simple progress increment (real impl would be smarter)
                goal.progress = (goal.progress + 0.1).min(1.0);
                goal.last_updated = std::time::Instant::now();
            }
        }

        // Remove completed goals
        goals.retain(|g| g.progress < 1.0);

        Ok(())
    }

    /// Get current active goals
    pub async fn get_active_goals(&self) -> Vec<TrackedGoal> {
        self.active_goals.read().await.clone()
    }

    /// Prune stale goals (not updated recently)
    pub async fn prune_stale(&self, max_age: std::time::Duration) -> usize {
        let mut goals = self.active_goals.write().await;
        let before = goals.len();
        goals.retain(|g| g.last_updated.elapsed() < max_age);
        before - goals.len()
    }
}

#[async_trait]
impl Subagent for GoalAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Goal }
    // ... standard trait implementations
}
```

### Dream Agent

```rust
// crates/context-graph-mcp/src/subagents/dream_agent.rs

use crate::handlers::consolidate::{ConsolidateHandler, ConsolidationMode};

/// Memory consolidation (REM/dreaming) subagent
pub struct DreamAgent {
    consolidator: Arc<ConsolidateHandler>,
    drift_detector: Arc<DriftDetector>,
    status: RwLock<SubagentStatus>,
    last_consolidation: RwLock<Option<std::time::Instant>>,
    patterns_discovered: RwLock<Vec<PatternSummary>>,
}

impl DreamAgent {
    pub fn new(
        consolidator: Arc<ConsolidateHandler>,
        drift_detector: Arc<DriftDetector>,
    ) -> Self;

    /// Run light consolidation (quick cleanup)
    pub async fn consolidate_light(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Light,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };
        self.consolidator.handle(params).await
    }

    /// Run deep consolidation (thorough cleanup)
    pub async fn consolidate_deep(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Deep,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };
        self.consolidator.handle(params).await
    }

    /// Run REM/dreaming consolidation (pattern discovery)
    pub async fn dream(&self) -> SubagentResult<ConsolidateResult> {
        let params = ConsolidateParams {
            mode: ConsolidationMode::Rem,
            salience_threshold: 0.3,
            max_memories: 0,
            dry_run: false,
        };

        let result = self.consolidator.handle(params).await?;

        // Store discovered patterns
        let mut patterns = self.patterns_discovered.write().await;
        patterns.extend(result.new_patterns.clone());

        // Update last consolidation time
        *self.last_consolidation.write().await = Some(std::time::Instant::now());

        Ok(result)
    }

    /// Check if consolidation is due
    pub async fn should_consolidate(&self) -> bool {
        let last = self.last_consolidation.read().await;
        match *last {
            None => true,
            Some(t) => t.elapsed() > std::time::Duration::from_secs(3600), // 1 hour
        }
    }

    /// Get all patterns discovered across dream sessions
    pub async fn get_patterns(&self) -> Vec<PatternSummary> {
        self.patterns_discovered.read().await.clone()
    }
}

#[async_trait]
impl Subagent for DreamAgent {
    fn agent_type(&self) -> SubagentType { SubagentType::Dream }
    // ... standard trait implementations
}
```

### Subagent Manager

```rust
// crates/context-graph-mcp/src/subagents/manager.rs

/// Manages all subagents lifecycle
pub struct SubagentManager {
    embedding_agent: Arc<EmbeddingAgent>,
    search_agent: Arc<SearchAgent>,
    goal_agent: Arc<GoalAgent>,
    dream_agent: Arc<DreamAgent>,
    hook_dispatcher: Arc<HookDispatcher>,
}

impl SubagentManager {
    pub fn new(
        embedding_agent: Arc<EmbeddingAgent>,
        search_agent: Arc<SearchAgent>,
        goal_agent: Arc<GoalAgent>,
        dream_agent: Arc<DreamAgent>,
        hook_dispatcher: Arc<HookDispatcher>,
    ) -> Self;

    /// Start all subagents
    pub async fn start_all(&self) -> SubagentResult<()> {
        tokio::try_join!(
            self.embedding_agent.start(),
            self.search_agent.start(),
            self.goal_agent.start(),
            self.dream_agent.start(),
        )?;
        Ok(())
    }

    /// Stop all subagents and merge learnings
    pub async fn stop_all(&self, parent_session: SessionId) -> SubagentResult<()> {
        // Stop each and collect outputs
        let outputs = vec![
            self.embedding_agent.stop().await?,
            self.search_agent.stop().await?,
            self.goal_agent.stop().await?,
            self.dream_agent.stop().await?,
        ];

        // Dispatch SubagentStop hook for each
        for output in outputs {
            let context = SubagentStopContext {
                subagent_id: SubagentId::new(),
                parent_session: parent_session.clone(),
                task_description: format!("{:?} agent", output.agent_type),
                success: output.success,
                created_memories: output.memories_created,
                duration: output.duration,
            };
            self.hook_dispatcher.dispatch_subagent_stop(context).await?;
        }

        Ok(())
    }

    /// Get health status of all subagents
    pub async fn health_check_all(&self) -> HashMap<SubagentType, SubagentHealth> {
        let mut health = HashMap::new();
        health.insert(SubagentType::Embedding, self.embedding_agent.health_check().await);
        health.insert(SubagentType::Search, self.search_agent.health_check().await);
        health.insert(SubagentType::Goal, self.goal_agent.health_check().await);
        health.insert(SubagentType::Dream, self.dream_agent.health_check().await);
        health
    }

    /// Get specific subagent
    pub fn embedding(&self) -> Arc<EmbeddingAgent> { self.embedding_agent.clone() }
    pub fn search(&self) -> Arc<SearchAgent> { self.search_agent.clone() }
    pub fn goal(&self) -> Arc<GoalAgent> { self.goal_agent.clone() }
    pub fn dream(&self) -> Arc<DreamAgent> { self.dream_agent.clone() }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Subagent startup | < 500ms |
| Health check | < 10ms |
| Stop with merge | < 5s (SubagentStop timeout) |
| Max concurrent tasks | 100 per agent |

## Verification

### Core Subagent Implementation
- [ ] All 4 subagents implement Subagent trait
- [ ] EmbeddingAgent handles GPU memory coordination
- [ ] SearchAgent uses entry-point selection
- [ ] GoalAgent discovers and tracks goals
- [ ] DreamAgent performs consolidation
- [ ] SubagentStop hook triggers on stop
- [ ] Learnings merged into parent session
- [ ] Health checks return accurate status

### Claude Code Integration
- [ ] Task tool spawning works for all 4 agents
- [ ] .claude/agents/ YAML configs are valid
- [ ] Agents map correctly to Claude Code subagent_types
- [ ] Parallel agent spawning in single message works

### 13-Embedder Integration
- [ ] EmbeddingAgent generates all 13 embedders (E1-E13)
- [ ] SearchAgent uses E1, E3, E12 for entry-point selection
- [ ] GoalAgent uses E3, E5, E7 for goal discovery
- [ ] DreamAgent analyzes all 13 for pattern discovery
- [ ] SubagentEmbedderConfig correctly assigns embedders

### GWT Consciousness Integration
- [ ] SubagentConsciousnessContext tracks I(t), R(t), D(t)
- [ ] C(t) = I(t) x R(t) x D(t) calculation correct
- [ ] AgentAwareness levels map to correct thresholds
- [ ] CoordinationStrategy matches awareness level
- [ ] ConsciousnessAwareSubagent trait implemented

### Inter-Agent Communication
- [ ] AgentMemoryProtocol namespaces are correct
- [ ] Broadcast messages reach all agents
- [ ] Subscription callbacks fire correctly
- [ ] AgentMessageType covers all communication needs

### Hook Integration
- [ ] SessionStart initializes all 4 subagents
- [ ] SubagentStop merges learnings correctly
- [ ] SessionEnd stops all agents and consolidates
- [ ] Final dream consolidation runs before session end

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/subagents/mod.rs` | Subagent traits and types |
| `crates/context-graph-mcp/src/subagents/embedding_agent.rs` | Embedding subagent |
| `crates/context-graph-mcp/src/subagents/search_agent.rs` | Search subagent |
| `crates/context-graph-mcp/src/subagents/goal_agent.rs` | Goal subagent |
| `crates/context-graph-mcp/src/subagents/dream_agent.rs` | Dream subagent |
| `crates/context-graph-mcp/src/subagents/manager.rs` | Subagent lifecycle manager |
| `crates/context-graph-mcp/src/subagents/embedder_integration.rs` | Embedder config per agent |
| `crates/context-graph-mcp/src/subagents/consciousness.rs` | GWT consciousness context |
| `crates/context-graph-mcp/src/subagents/communication.rs` | Inter-agent memory protocol |
| `.claude/agents/embedding-agent.yaml` | Claude Code embedding agent config |
| `.claude/agents/search-agent.yaml` | Claude Code search agent config |
| `.claude/agents/goal-agent.yaml` | Claude Code goal agent config |
| `.claude/agents/dream-agent.yaml` | Claude Code dream agent config |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subagent deadlock | Medium | High | Timeouts, health checks |
| Memory leak from stopped agents | Low | Medium | RAII, explicit cleanup |
| Learning loss on crash | Medium | Medium | Periodic checkpoints |
| Consciousness calculation divergence | Low | Medium | Bounded metrics (0-1) |
| Inter-agent message flood | Medium | Medium | Rate limiting, priority queues |
| Embedder model load time | Medium | Low | Lazy loading, model caching |
| GWT broadcast overhead | Low | Low | Significance threshold filtering |

## Traceability

- Source: Constitution subagent specifications
- Depends on: TASK-INTEG-012 (SubagentStop hook)
- Depends on: TASK-LOGIC-001 (13-Embedder Teleological Array)
- Depends on: TASK-INTEG-003 (Consciousness MCP handlers)
- Integrates with: Claude Code Task Tool for agent spawning
- Coordinates via: GWT Global Workspace for consciousness broadcasts
