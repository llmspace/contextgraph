# TASK-INTEG-015: SELF_EGO_NODE Management

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-015 |
| **Title** | SELF_EGO_NODE Management |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 35 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- Constitution line 154: `SELF_EGO_NODE: "00000000-0000-0000-0000-000000000000"` (reserved root)
- ARCH-01: Atomic storage (special handling for ego node)
- Autonomous self-model for context graph

## Dependencies

| Task | Reason |
|------|--------|
| TASK-CORE-003 | TeleologicalArrayStore for storage |
| TASK-INTEG-001 | MCP handlers for access |
| TASK-INTEG-004 | Hooks for ego updates |

## Objective

Implement the `SELF_EGO_NODE` as the reserved root node representing the AI's self-model within the context graph. This node anchors the autonomous agent's identity, goals, and meta-cognition.

## Context

**Constitution Requirement:**

```
SELF_EGO_NODE: "00000000-0000-0000-0000-000000000000" (reserved root)
```

The SELF_EGO_NODE is a special TeleologicalArray that:
1. Always exists (created on graph initialization)
2. Cannot be deleted
3. Represents the agent's self-model
4. Anchors goal hierarchies and meta-cognition
5. Stores persistent identity across sessions

## Claude Code Integration

### SessionStart Hook Integration

The SELF_EGO_NODE is initialized/loaded during SessionStart hook:

**settings.json Configuration:**
```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": ".claude/hooks/ego-init.sh \"$SESSION_ID\"",
        "timeout": 5000
      }
    ]
  }
}
```

**Shell Script (.claude/hooks/ego-init.sh):**
```bash
#!/bin/bash
SESSION_ID="$1"
# Initialize or restore ego node for session
npx claude-flow@v3alpha hooks session-start --session-id "$SESSION_ID" --init-ego
```

### GWT Consciousness Integration

The SELF_EGO_NODE represents the core of agent consciousness in the GWT model:

```rust
/// Consciousness-aware ego context
pub struct EgoConsciousnessContext {
    /// Self-model from ego node
    pub self_model: SelfModel,
    /// Current consciousness state
    pub consciousness_state: ConsciousnessState,
    /// GWT integration score C(t) = I(t) × R(t) × D(t)
    pub gwt_score: f32,
    /// Self-awareness level (meta-cognition depth)
    pub self_awareness: SelfAwareness,
}

#[derive(Debug, Clone, Copy)]
pub enum SelfAwareness {
    /// No self-reflection
    Unreflective,
    /// Basic self-monitoring
    Monitoring,
    /// Active introspection
    Introspective,
    /// Deep self-understanding
    Enlightened,
}

impl EgoConsciousnessContext {
    /// Calculate self-awareness from meta-cognition
    pub fn calculate_awareness(&self) -> SelfAwareness {
        let meta = &self.self_model.meta;
        let depth = meta.introspection.len() as f32 / 10.0;
        let confidence = meta.confidence;
        let awareness_score = (depth + confidence) / 2.0;

        match awareness_score {
            s if s < 0.3 => SelfAwareness::Unreflective,
            s if s < 0.5 => SelfAwareness::Monitoring,
            s if s < 0.8 => SelfAwareness::Introspective,
            _ => SelfAwareness::Enlightened,
        }
    }
}
```

### 13-Embedder Integration for Self-Model

The ego node content is embedded across all 13 dimensions for rich self-representation:

```rust
impl EgoNodeManager {
    /// Generate embeddings for self-model components
    pub async fn embed_self_model(&self, embedder: &MultiEmbedder) -> StorageResult<EgoEmbeddings> {
        let model = self.get_self_model().await?;

        // E1: Semantic - identity and traits
        let identity_text = format!("{} - {}", model.identity.name,
            model.traits.expertise.join(", "));
        let e1 = embedder.embed_semantic(&identity_text).await?;

        // E7: Teleological - active goals
        let goals_text = model.active_goals.iter()
            .map(|g| g.description.clone())
            .collect::<Vec<_>>()
            .join("; ");
        let e7 = embedder.embed_teleological(&goals_text).await?;

        // E11: Emotional - meta-cognitive state
        let meta_text = format!("confidence: {}, load: {}",
            model.meta.confidence, model.meta.cognitive_load);
        let e11 = embedder.embed_emotional(&meta_text).await?;

        Ok(EgoEmbeddings { e1, e7, e11, /* ... other embeddings */ })
    }
}
```

### Memory Integration for Persistent Identity

The ego node integrates with Claude Flow memory for cross-session persistence:

```rust
impl EgoNodeManager {
    /// Persist ego state to memory system
    pub async fn persist_to_memory(&self, memory: &MemoryCoordinator) -> StorageResult<()> {
        let model = self.get_self_model().await?;

        memory.store(
            "ego",  // namespace
            "self_model",
            &serde_json::to_string(&model)?,
            None,  // no TTL - permanent
        ).await?;

        Ok(())
    }

    /// Restore ego state from memory system
    pub async fn restore_from_memory(&self, memory: &MemoryCoordinator) -> StorageResult<Option<SelfModel>> {
        if let Some(value) = memory.retrieve("ego", "self_model").await? {
            let model: SelfModel = serde_json::from_str(&value)?;
            // Update cache
            *self.cache.write().await = Some(model.clone());
            return Ok(Some(model));
        }
        Ok(None)
    }
}
```

## Scope

### In Scope

- SELF_EGO_NODE constant definition
- Automatic creation on store initialization
- Protection against deletion/corruption
- Self-model fields (identity, goals, traits, meta)
- Ego node update API
- Session-to-session persistence

### Out of Scope

- Multi-agent ego models
- Ego node versioning/history
- Distributed ego consensus

## Definition of Done

### Signatures

```rust
// crates/context-graph-core/src/teleology/ego.rs

use uuid::Uuid;

/// Reserved UUID for the self-ego node
pub const SELF_EGO_NODE: Uuid = Uuid::from_bytes([
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00,
    0x00, 0x00,
    0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]);

/// Check if a UUID is the reserved ego node
pub fn is_ego_node(id: Uuid) -> bool {
    id == SELF_EGO_NODE
}

/// Self-model stored in the ego node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Agent identity
    pub identity: AgentIdentity,
    /// Current active goals
    pub active_goals: Vec<GoalRef>,
    /// Learned traits and preferences
    pub traits: AgentTraits,
    /// Meta-cognitive state
    pub meta: MetaCognition,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Agent identity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Name/identifier for this agent instance
    pub name: String,
    /// Version of the context graph
    pub version: String,
    /// When this agent was first created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Session count across lifetime
    pub session_count: u64,
}

/// Reference to a goal (not full goal to avoid cycles)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalRef {
    pub id: Uuid,
    pub description: String,
    pub priority: f32,
}

/// Agent traits learned over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTraits {
    /// Preferred communication style
    pub communication_style: CommunicationStyle,
    /// Domain expertise areas
    pub expertise: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Learned preferences
    pub preferences: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Concise,
    Detailed,
    Technical,
    Casual,
    Formal,
}

/// Meta-cognitive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognition {
    /// Current cognitive load estimate (0.0 - 1.0)
    pub cognitive_load: f32,
    /// Confidence in recent responses
    pub confidence: f32,
    /// Areas needing improvement
    pub growth_areas: Vec<String>,
    /// Recent introspection notes
    pub introspection: Vec<String>,
}

impl Default for SelfModel {
    fn default() -> Self {
        Self {
            identity: AgentIdentity {
                name: "ContextGraph Agent".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                created_at: chrono::Utc::now(),
                session_count: 0,
            },
            active_goals: vec![],
            traits: AgentTraits {
                communication_style: CommunicationStyle::Detailed,
                expertise: vec![],
                limitations: vec![],
                preferences: HashMap::new(),
            },
            meta: MetaCognition {
                cognitive_load: 0.0,
                confidence: 0.5,
                growth_areas: vec![],
                introspection: vec![],
            },
            updated_at: chrono::Utc::now(),
        }
    }
}
```

### Ego Node Manager

```rust
// crates/context-graph-storage/src/teleological/ego_manager.rs

use context_graph_core::teleology::ego::{SELF_EGO_NODE, SelfModel, is_ego_node};

/// Manages the SELF_EGO_NODE lifecycle and access
pub struct EgoNodeManager {
    store: Arc<TeleologicalArrayStore>,
    cache: RwLock<Option<SelfModel>>,
}

impl EgoNodeManager {
    pub fn new(store: Arc<TeleologicalArrayStore>) -> Self;

    /// Initialize ego node if it doesn't exist
    ///
    /// Called during store initialization
    pub async fn ensure_exists(&self) -> StorageResult<()> {
        if self.store.get(SELF_EGO_NODE).await?.is_none() {
            let model = SelfModel::default();
            let array = self.model_to_array(model)?;
            self.store.store(array).await?;
        }
        Ok(())
    }

    /// Get the current self-model
    pub async fn get_self_model(&self) -> StorageResult<SelfModel> {
        // Check cache first
        if let Some(model) = self.cache.read().await.as_ref() {
            return Ok(model.clone());
        }

        // Load from store
        let array = self.store.get(SELF_EGO_NODE).await?
            .ok_or(StorageError::EgoNodeMissing)?;
        let model = self.array_to_model(&array)?;

        // Update cache
        *self.cache.write().await = Some(model.clone());

        Ok(model)
    }

    /// Update the self-model
    ///
    /// # Note
    /// This is the ONLY way to modify the ego node
    pub async fn update_self_model<F>(&self, updater: F) -> StorageResult<SelfModel>
    where
        F: FnOnce(&mut SelfModel),
    {
        let mut model = self.get_self_model().await?;
        updater(&mut model);
        model.updated_at = chrono::Utc::now();

        let array = self.model_to_array(model.clone())?;
        self.store.update(array).await?;

        // Update cache
        *self.cache.write().await = Some(model.clone());

        Ok(model)
    }

    /// Add a goal reference to the ego node
    pub async fn add_goal(&self, goal: GoalRef) -> StorageResult<()> {
        self.update_self_model(|model| {
            // Check for duplicate
            if !model.active_goals.iter().any(|g| g.id == goal.id) {
                model.active_goals.push(goal);
            }
        }).await?;
        Ok(())
    }

    /// Remove a goal reference from the ego node
    pub async fn remove_goal(&self, goal_id: Uuid) -> StorageResult<()> {
        self.update_self_model(|model| {
            model.active_goals.retain(|g| g.id != goal_id);
        }).await?;
        Ok(())
    }

    /// Update cognitive load
    pub async fn set_cognitive_load(&self, load: f32) -> StorageResult<()> {
        self.update_self_model(|model| {
            model.meta.cognitive_load = load.clamp(0.0, 1.0);
        }).await?;
        Ok(())
    }

    /// Increment session count (called on SessionStart)
    pub async fn new_session(&self) -> StorageResult<u64> {
        let model = self.update_self_model(|model| {
            model.identity.session_count += 1;
        }).await?;
        Ok(model.identity.session_count)
    }

    /// Protect ego node from deletion
    ///
    /// Called before any delete operation
    pub fn guard_delete(&self, id: Uuid) -> StorageResult<()> {
        if is_ego_node(id) {
            return Err(StorageError::CannotDeleteEgoNode);
        }
        Ok(())
    }

    // Internal helpers

    fn model_to_array(&self, model: SelfModel) -> StorageResult<TeleologicalArray> {
        let json = serde_json::to_string(&model)?;

        // Create array with special ego node ID
        let mut array = TeleologicalArray::new_with_id(SELF_EGO_NODE);
        array.content = json;
        array.metadata.insert("type".into(), "ego_node".into());
        array.metadata.insert("protected".into(), "true".into());

        Ok(array)
    }

    fn array_to_model(&self, array: &TeleologicalArray) -> StorageResult<SelfModel> {
        let model: SelfModel = serde_json::from_str(&array.content)?;
        Ok(model)
    }
}

/// Error types specific to ego node operations
#[derive(Debug, thiserror::Error)]
pub enum EgoError {
    #[error("Cannot delete the SELF_EGO_NODE")]
    CannotDelete,
    #[error("Ego node is missing from store")]
    Missing,
    #[error("Ego node is corrupted: {0}")]
    Corrupted(String),
}
```

### Integration with Store

```rust
// Update crates/context-graph-storage/src/teleological/store.rs

impl TeleologicalArrayStore {
    /// Initialize store with ego node
    pub async fn initialize(&self) -> StorageResult<()> {
        // Ensure ego node exists
        self.ego_manager.ensure_exists().await?;
        Ok(())
    }

    /// Delete with ego node protection
    pub async fn delete(&self, id: Uuid) -> StorageResult<()> {
        // Guard against ego node deletion
        self.ego_manager.guard_delete(id)?;

        // Proceed with normal deletion
        self.inner_delete(id).await
    }
}
```

### Hook Integration

```rust
// crates/context-graph-mcp/src/hooks/session_start.rs

impl SessionStartHandler {
    pub async fn handle(&self, context: SessionStartContext) -> HookResult<SessionStartOutput> {
        // ... existing logic ...

        // Increment session count in ego node
        let session_count = self.ego_manager.new_session().await?;

        // Include in output
        output.session_count = session_count;

        Ok(output)
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Ego node read | < 1ms (cached) |
| Ego node update | < 10ms |
| Storage overhead | < 10KB |
| Always present | Guaranteed after init |

## Verification

- [ ] SELF_EGO_NODE has reserved UUID 00000000-0000-0000-0000-000000000000
- [ ] Ego node created automatically on store init
- [ ] Ego node cannot be deleted (returns error)
- [ ] SelfModel serializes/deserializes correctly
- [ ] Session count increments on SessionStart
- [ ] Goals can be added/removed from ego
- [ ] Cognitive load updates persist
- [ ] Cache invalidates on update
- [ ] SessionStart hook initializes ego node
- [ ] GWT consciousness score calculated from self-model
- [ ] Self-awareness level derived from meta-cognition
- [ ] Ego embeddings generated across 13 dimensions
- [ ] Memory persistence works cross-session

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/teleology/ego.rs` | Ego types and constants |
| `crates/context-graph-storage/src/teleological/ego_manager.rs` | Ego node management |
| Update `crates/context-graph-storage/src/teleological/store.rs` | Ego integration |
| `.claude/hooks/ego-init.sh` | Ego initialization hook |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ego node corruption | Low | Critical | Validation on read |
| Cache staleness | Low | Low | Invalidate on write |
| UUID collision | Impossible | N/A | Reserved UUID |

## Traceability

- Source: Constitution line 154 (SELF_EGO_NODE)
- Related: Autonomous agent self-model requirement
