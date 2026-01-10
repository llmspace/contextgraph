# TASK-INTEG-008: Skills Development

```xml
<task_spec>
  <task_id>TASK-INTEG-008</task_id>
  <title>Skills Development (memory-search, goal-alignment, pattern-learning, context-injection, drift-check)</title>
  <status>pending</status>

  <objective>
    Implement Claude Code skills that wrap MCP tools to provide higher-level
    abstractions for memory search, goal alignment checking, pattern learning,
    context injection, and drift detection using the teleological array system.
    Skills integrate with GWT consciousness model for adaptive invocation modes.
  </objective>

  <rationale>
    Skills provide:
    1. Higher-level abstractions over raw MCP tools
    2. Auto-invocation based on context patterns (regex matching)
    3. Pre/post processing of inputs/outputs
    4. Error handling with retry logic and fallback responses
    5. User-friendly interfaces for Claude Code users via /slash commands
    6. Consciousness-aware invocation (Background, Standard, Priority, Critical)
    7. Integration with 13-embedding teleological array for semantic precision

    Skills make the teleological array system accessible without requiring
    direct MCP tool knowledge. Each skill maps to specific embedders for
    optimal search and alignment operations.
  </rationale>

  <dependencies>
    <dependency type="required">TASK-INTEG-001</dependency>    <!-- Memory MCP handlers -->
    <dependency type="required">TASK-INTEG-002</dependency>    <!-- Purpose/Goal MCP handlers -->
    <dependency type="required">TASK-INTEG-003</dependency>    <!-- Consciousness MCP handlers -->
    <dependency type="required">TASK-LOGIC-001</dependency>    <!-- Search engine -->
    <dependency type="required">TASK-LOGIC-008</dependency>    <!-- Alignment calculator -->
    <dependency type="required">TASK-LOGIC-010</dependency>    <!-- Teleological drift detection -->
  </dependencies>

  <input_context_files>
    <file purpose="mcp_tools_spec">docs2/refactor/08-MCP-TOOLS.md</file>
    <file purpose="memory_handler">crates/context-graph-mcp/src/handlers/memory.rs</file>
    <file purpose="purpose_handler">crates/context-graph-mcp/src/handlers/purpose.rs</file>
    <file purpose="consciousness_handler">crates/context-graph-mcp/src/handlers/consciousness.rs</file>
    <file purpose="search_engine">crates/context-graph-storage/src/teleological/search/engine.rs</file>
    <file purpose="drift_detector">crates/context-graph-storage/src/teleological/drift/detector.rs</file>
  </input_context_files>

  <output_artifacts>
    <!-- Rust skill loader and invocation system -->
    <artifact type="source">crates/context-graph-mcp/src/skills/mod.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/skills/loader.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/skills/consciousness.rs</artifact>

    <!-- Claude Code Skills (SKILL.md format with YAML frontmatter) -->
    <artifact type="skill">.claude/skills/memory-search/SKILL.md</artifact>
    <artifact type="skill">.claude/skills/goal-alignment/SKILL.md</artifact>
    <artifact type="skill">.claude/skills/pattern-learning/SKILL.md</artifact>
    <artifact type="skill">.claude/skills/context-injection/SKILL.md</artifact>
    <artifact type="skill">.claude/skills/drift-check/SKILL.md</artifact>

    <!-- Tests -->
    <artifact type="test">crates/context-graph-mcp/tests/skills_test.rs</artifact>
  </output_artifacts>

  <definition_of_done>
    <criterion id="1">All 5 core skills implemented with SKILL.md files and YAML frontmatter</criterion>
    <criterion id="2">Skills auto-invoke appropriate MCP tools based on context_patterns</criterion>
    <criterion id="3">Skill loader discovers and loads skills from .claude/skills/</criterion>
    <criterion id="4">Error handling with retry_count and fallback_response per skill</criterion>
    <criterion id="5">Each skill documents primary embedders from 13-embedding array</criterion>
    <criterion id="6">Skills transform results into user-friendly formats</criterion>
    <criterion id="7">Skills integrate with GWT consciousness for adaptive invocation</criterion>
    <criterion id="8">Skills invoke correct MCP tools (search_memories, calculate_alignment, etc.)</criterion>
    <criterion id="9">Skills respect timeout limits per Claude Code constitution</criterion>
    <criterion id="10">Test coverage for skill loading, invocation, and consciousness routing</criterion>
  </definition_of_done>

  <estimated_complexity>Medium</estimated_complexity>

  <claude_code_skill_format>
    Claude Code uses `.claude/skills/` directory for skills. Each skill is a directory
    containing a `SKILL.md` file with YAML frontmatter defining the skill configuration.
    Skills are invoked via `/skill-name` slash commands.

    Directory Structure:
    ```
    .claude/skills/
    ├── memory-search/
    │   └── SKILL.md
    ├── goal-alignment/
    │   └── SKILL.md
    ├── pattern-learning/
    │   └── SKILL.md
    ├── context-injection/
    │   └── SKILL.md
    └── drift-check/
        └── SKILL.md
    ```

    YAML Frontmatter Format:
    ```yaml
    ---
    name: skill-name
    description: Human-readable description of what the skill does
    version: 1.0.0
    auto_invoke: true|false
    context_patterns:
      - "regex pattern 1"
      - "regex pattern 2"
    error_handling:
      retry_count: 2
      fallback_response: "Message shown when skill fails"
    timeout_ms: 30000  # Per Claude Code constitution
    ---
    ```
  </claude_code_skill_format>

  <embedder_mapping>
    Each skill maps to specific embedders from the 13-embedding teleological array:

    | Skill | Primary Embedders | Purpose |
    |-------|-------------------|---------|
    | memory-search | E1_Semantic, E12_Code, E3_Causal, E13_SPLADE | Semantic search with code and keyword expansion |
    | goal-alignment | E7_Teleological, E5_Moral, E4_Counterfactual | Purpose-driven alignment checking |
    | pattern-learning | E10_Behavioral, E9_Structural, E3_Causal | Pattern extraction and storage |
    | context-injection | E8_Contextual, E1_Semantic, E2_Temporal | Context-aware memory injection |
    | drift-check | E7_Teleological, E4_Counterfactual, E5_Moral | Purpose drift detection |

    Embedder Reference:
    - E1_Semantic: General semantic meaning (384-dim)
    - E2_Temporal: Time-aware recency (64-dim)
    - E3_Causal: Cause-effect relationships (256-dim)
    - E4_Counterfactual: Alternative scenarios (256-dim)
    - E5_Moral: Ethical alignment (128-dim)
    - E6_Cultural: Cross-cultural context (128-dim)
    - E7_Teleological: Purpose/goal orientation (256-dim)
    - E8_Contextual: Situational awareness (256-dim)
    - E9_Structural: Code/document structure (192-dim)
    - E10_Behavioral: Action patterns (192-dim)
    - E11_Relational: Entity relationships (256-dim)
    - E12_Code: Programming semantics (512-dim)
    - E13_SPLADE: Sparse keyword expansion (30522-dim)
  </embedder_mapping>

  <mcp_tool_mapping>
    Each skill invokes specific MCP tools from context-graph server:

    | Skill | MCP Tools | Operations |
    |-------|-----------|------------|
    | memory-search | `search_memories`, `search_by_embedder` | Multi-embedder semantic search |
    | goal-alignment | `calculate_alignment`, `check_goal_drift` | Purpose alignment calculation |
    | pattern-learning | `extract_patterns`, `store_pattern`, `memory/inject` | Pattern extraction and storage |
    | context-injection | `inject_context`, `preload_context`, `memory/inject` | Context injection and preloading |
    | drift-check | `detect_drift`, `get_drift_report`, `purpose/drift_check` | Drift detection and reporting |
  </mcp_tool_mapping>

  <pseudo_code>
    <section name="ConsciousnessAwareSkillInvocation">
```rust
// crates/context-graph-mcp/src/skills/consciousness.rs

use serde::{Deserialize, Serialize};

/// Consciousness context for skill invocation
/// Integrates with GWT (Global Workspace Theory) consciousness model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillConsciousnessContext {
    /// I(t) - Integration score indicating skill relevance to current task
    /// Range: 0.0 to 1.0
    pub integration_score: f32,

    /// R(t) - Reflection depth indicating processing depth required
    /// Range: 0.0 to 1.0
    pub reflection_depth: f32,

    /// D(t) - Differentiation index indicating skill uniqueness for task
    /// Range: 0.0 to 1.0
    pub differentiation_index: f32,

    /// C(t) = I(t) * R(t) * D(t) - Overall consciousness level
    /// Determines invocation mode
    pub consciousness_level: f32,

    /// Invocation mode based on consciousness level
    pub invocation_mode: InvocationMode,

    /// Primary embedders for this skill invocation
    pub active_embedders: Vec<String>,

    /// Workspace broadcast metadata
    pub workspace_context: Option<WorkspaceContext>,
}

/// Invocation mode determines execution priority and resource allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvocationMode {
    /// C < 0.3: Async non-blocking, background processing
    Background,

    /// 0.3 <= C < 0.6: Standard priority execution
    Standard,

    /// 0.6 <= C < 0.8: Immediate execution with elevated resources
    Priority,

    /// C >= 0.8: Full consciousness engagement, synchronous critical path
    Critical,
}

impl InvocationMode {
    /// Calculate invocation mode from consciousness level
    pub fn from_consciousness_level(c: f32) -> Self {
        match c {
            c if c < 0.3 => InvocationMode::Background,
            c if c < 0.6 => InvocationMode::Standard,
            c if c < 0.8 => InvocationMode::Priority,
            _ => InvocationMode::Critical,
        }
    }

    /// Get timeout multiplier for this mode
    pub fn timeout_multiplier(&self) -> f32 {
        match self {
            InvocationMode::Background => 2.0,   // Longer timeout, async
            InvocationMode::Standard => 1.0,     // Default timeout
            InvocationMode::Priority => 0.75,    // Faster timeout expected
            InvocationMode::Critical => 0.5,     // Must complete quickly
        }
    }

    /// Get retry count for this mode
    pub fn retry_count(&self) -> u32 {
        match self {
            InvocationMode::Background => 3,
            InvocationMode::Standard => 2,
            InvocationMode::Priority => 1,
            InvocationMode::Critical => 0,  // No retries, must succeed first time
        }
    }
}

/// Workspace context for GWT broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceContext {
    /// Current task description
    pub task: String,

    /// Active goal ID being pursued
    pub active_goal_id: Option<String>,

    /// Recent memory IDs in workspace
    pub workspace_memories: Vec<String>,

    /// Attention weights per embedder
    pub attention_weights: std::collections::HashMap<String, f32>,
}

impl SkillConsciousnessContext {
    /// Create new consciousness context
    pub fn new(
        integration: f32,
        reflection: f32,
        differentiation: f32,
    ) -> Self {
        let consciousness_level = integration * reflection * differentiation;
        let invocation_mode = InvocationMode::from_consciousness_level(consciousness_level);

        Self {
            integration_score: integration,
            reflection_depth: reflection,
            differentiation_index: differentiation,
            consciousness_level,
            invocation_mode,
            active_embedders: Vec::new(),
            workspace_context: None,
        }
    }

    /// Set active embedders for this invocation
    pub fn with_embedders(mut self, embedders: Vec<String>) -> Self {
        self.active_embedders = embedders;
        self
    }

    /// Set workspace context
    pub fn with_workspace(mut self, context: WorkspaceContext) -> Self {
        self.workspace_context = Some(context);
        self
    }

    /// Calculate effective timeout for skill
    pub fn effective_timeout_ms(&self, base_timeout_ms: u64) -> u64 {
        (base_timeout_ms as f32 * self.invocation_mode.timeout_multiplier()) as u64
    }
}

/// Trait for consciousness-aware skill execution
pub trait ConsciousnessAwareSkill {
    /// Calculate consciousness context for given input
    fn calculate_consciousness(
        &self,
        query: &str,
        workspace: Option<&WorkspaceContext>,
    ) -> SkillConsciousnessContext;

    /// Get primary embedders for this skill
    fn primary_embedders(&self) -> Vec<&'static str>;

    /// Execute with consciousness-aware routing
    async fn execute_conscious(
        &self,
        params: serde_json::Value,
        consciousness: SkillConsciousnessContext,
    ) -> Result<serde_json::Value, SkillError>;
}
```
    </section>

    <section name="SkillLoader">
```rust
// crates/context-graph-mcp/src/skills/mod.rs

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::time::{Duration, Instant};

pub mod consciousness;
pub mod loader;

use consciousness::{SkillConsciousnessContext, InvocationMode, ConsciousnessAwareSkill};

/// Skill definition from SKILL.md YAML frontmatter
/// Compatible with Claude Code's skill system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    /// Skill name (used for /skill-name invocation)
    pub name: String,

    /// Skill version
    pub version: String,

    /// Human-readable description
    pub description: String,

    /// Whether to auto-invoke based on context patterns
    #[serde(default)]
    pub auto_invoke: bool,

    /// Regex patterns for auto-invocation
    #[serde(default)]
    pub context_patterns: Vec<String>,

    /// Error handling configuration
    #[serde(default)]
    pub error_handling: ErrorHandlingConfig,

    /// Timeout in milliseconds (per Claude Code constitution)
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,

    /// MCP configuration
    pub mcp: McpConfig,

    /// Parameters schema
    #[serde(default)]
    pub parameters: HashMap<String, ParameterDef>,

    /// Primary embedders from 13-embedding array
    #[serde(default)]
    pub primary_embedders: Vec<String>,

    /// Success handlers
    #[serde(default)]
    pub on_success: Vec<SuccessAction>,

    /// Error handlers (legacy, prefer error_handling)
    #[serde(default)]
    pub on_error: Vec<ErrorAction>,

    /// Result transformations
    #[serde(default)]
    pub transforms: Option<TransformConfig>,
}

fn default_timeout() -> u64 {
    30000  // 30 seconds default per Claude Code constitution
}

/// Error handling configuration per Claude Code spec
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorHandlingConfig {
    /// Number of retry attempts
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,

    /// Fallback response when skill fails
    #[serde(default)]
    pub fallback_response: Option<String>,

    /// Backoff strategy
    #[serde(default = "default_backoff")]
    pub backoff: String,
}

fn default_retry_count() -> u32 { 2 }
fn default_backoff() -> String { "exponential".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// MCP server name
    pub server: String,
    /// Primary tool to invoke
    pub tool: String,
    /// Alternative tools (for fallback)
    #[serde(default)]
    pub alternative_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDef {
    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: String,
    /// Whether required
    #[serde(default)]
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Description
    pub description: Option<String>,
    /// Allowed values (for enums)
    pub values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SuccessAction {
    #[serde(rename = "log")]
    Log { message: String },
    #[serde(rename = "notify_hook")]
    NotifyHook { hook: String },
    #[serde(rename = "set_context")]
    SetContext { key: String },
    #[serde(rename = "store_goals")]
    StoreGoals { path: String },
    #[serde(rename = "emit_event")]
    EmitEvent { event: String },
    #[serde(rename = "broadcast_workspace")]
    BroadcastWorkspace { data: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ErrorAction {
    #[serde(rename = "log_error")]
    LogError { message: String },
    #[serde(rename = "retry")]
    Retry { max_attempts: u32, backoff: String },
    #[serde(rename = "fallback")]
    Fallback { skill: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    /// Result transformation expression
    pub result: Option<String>,
}

/// Skill registry and loader
pub struct SkillLoader {
    /// Base directory for skills (.claude/skills/)
    skills_dir: PathBuf,
    /// Loaded skills
    skills: Arc<RwLock<HashMap<String, LoadedSkill>>>,
    /// MCP client for invoking tools
    mcp_client: Arc<dyn McpClient>,
    /// Compiled context patterns for auto-invocation
    patterns: Arc<RwLock<HashMap<String, Vec<regex::Regex>>>>,
}

/// MCP client trait for tool invocation
#[async_trait::async_trait]
pub trait McpClient: Send + Sync {
    async fn invoke(
        &self,
        server: &str,
        tool: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// Skill definition from YAML frontmatter
    pub definition: SkillDefinition,
    /// Full SKILL.md content (for documentation)
    pub markdown_content: String,
    /// Path to SKILL.md file
    pub skill_path: PathBuf,
    /// Load timestamp
    pub loaded_at: DateTime<Utc>,
}

impl SkillLoader {
    pub fn new(skills_dir: PathBuf, mcp_client: Arc<dyn McpClient>) -> Self {
        Self {
            skills_dir,
            skills: Arc::new(RwLock::new(HashMap::new())),
            mcp_client,
            patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load all skills from the .claude/skills/ directory
    pub async fn load_all(&self) -> Result<usize, SkillError> {
        let mut count = 0;
        let mut skills = self.skills.write().await;
        let mut patterns = self.patterns.write().await;

        // Scan skills directory
        let entries = std::fs::read_dir(&self.skills_dir)
            .map_err(|e| SkillError::IoError(e.to_string()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Look for SKILL.md (Claude Code format)
                let skill_md = path.join("SKILL.md");
                if skill_md.exists() {
                    match self.load_skill(&skill_md).await {
                        Ok(skill) => {
                            let name = skill.definition.name.clone();

                            // Compile context patterns
                            if skill.definition.auto_invoke {
                                let compiled: Vec<regex::Regex> = skill.definition.context_patterns
                                    .iter()
                                    .filter_map(|p| regex::Regex::new(p).ok())
                                    .collect();
                                patterns.insert(name.clone(), compiled);
                            }

                            skills.insert(name, skill);
                            count += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load skill from {:?}: {}", skill_md, e);
                        }
                    }
                }
            }
        }

        tracing::info!("Loaded {} skills from {:?}", count, self.skills_dir);
        Ok(count)
    }

    /// Load a single skill from SKILL.md file
    async fn load_skill(&self, md_path: &Path) -> Result<LoadedSkill, SkillError> {
        let content = std::fs::read_to_string(md_path)
            .map_err(|e| SkillError::IoError(e.to_string()))?;

        // Extract YAML frontmatter between --- markers
        let definition = self.parse_frontmatter(&content)?;

        Ok(LoadedSkill {
            definition,
            markdown_content: content.clone(),
            skill_path: md_path.to_path_buf(),
            loaded_at: Utc::now(),
        })
    }

    /// Parse YAML frontmatter from SKILL.md content
    fn parse_frontmatter(&self, content: &str) -> Result<SkillDefinition, SkillError> {
        // Find frontmatter between --- markers
        if !content.starts_with("---") {
            return Err(SkillError::ParseError("Missing YAML frontmatter".to_string()));
        }

        let end_marker = content[3..].find("---")
            .ok_or_else(|| SkillError::ParseError("Missing closing ---".to_string()))?;

        let yaml_content = &content[3..end_marker + 3];

        serde_yaml::from_str(yaml_content)
            .map_err(|e| SkillError::ParseError(e.to_string()))
    }

    /// Check if any skill should auto-invoke for given context
    pub async fn check_auto_invoke(&self, context: &str) -> Vec<String> {
        let patterns = self.patterns.read().await;
        let mut matching = Vec::new();

        for (skill_name, skill_patterns) in patterns.iter() {
            for pattern in skill_patterns {
                if pattern.is_match(context) {
                    matching.push(skill_name.clone());
                    break;
                }
            }
        }

        matching
    }

    /// Get a loaded skill by name
    pub async fn get(&self, name: &str) -> Option<LoadedSkill> {
        self.skills.read().await.get(name).cloned()
    }

    /// List all loaded skills
    pub async fn list(&self) -> Vec<String> {
        self.skills.read().await.keys().cloned().collect()
    }

    /// Invoke a skill with consciousness-aware routing
    pub async fn invoke(
        &self,
        name: &str,
        params: serde_json::Value,
        consciousness: Option<SkillConsciousnessContext>,
    ) -> Result<SkillResult, SkillError> {
        let skill = self.get(name).await
            .ok_or_else(|| SkillError::NotFound(name.to_string()))?;

        let invoker = SkillInvoker::new(
            skill,
            self.mcp_client.clone(),
        );

        invoker.invoke(params, consciousness).await
    }
}

/// Skill invocation handler
pub struct SkillInvoker {
    skill: LoadedSkill,
    mcp_client: Arc<dyn McpClient>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResult {
    /// Whether invocation succeeded
    pub success: bool,
    /// Result data
    pub data: serde_json::Value,
    /// Messages/logs
    pub messages: Vec<String>,
    /// Execution time (ms)
    pub execution_time_ms: u64,
    /// Consciousness context used
    pub consciousness: Option<SkillConsciousnessContext>,
    /// Embedders activated
    pub embedders_used: Vec<String>,
}

impl SkillInvoker {
    pub fn new(skill: LoadedSkill, mcp_client: Arc<dyn McpClient>) -> Self {
        Self { skill, mcp_client }
    }

    /// Invoke the skill with given parameters and optional consciousness context
    pub async fn invoke(
        &self,
        params: serde_json::Value,
        consciousness: Option<SkillConsciousnessContext>,
    ) -> Result<SkillResult, SkillError> {
        let start = Instant::now();
        let mut messages = Vec::new();

        // Calculate effective timeout based on consciousness level
        let timeout_ms = consciousness.as_ref()
            .map(|c| c.effective_timeout_ms(self.skill.definition.timeout_ms))
            .unwrap_or(self.skill.definition.timeout_ms);

        // Calculate retry count based on consciousness or error_handling config
        let retry_count = consciousness.as_ref()
            .map(|c| c.invocation_mode.retry_count())
            .unwrap_or(self.skill.definition.error_handling.retry_count);

        // 1. Validate parameters
        let validated_params = self.validate_and_apply_defaults(params)?;

        // 2. Invoke MCP tool with timeout
        let mcp_result = tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            self.invoke_mcp(&validated_params)
        ).await;

        // 3. Handle result
        match mcp_result {
            Ok(Ok(result)) => {
                // Run success actions
                for action in &self.skill.definition.on_success {
                    let msg = self.run_success_action(action, &result).await?;
                    if let Some(m) = msg {
                        messages.push(m);
                    }
                }

                // Transform result if configured
                let transformed = self.transform_result(result)?;

                Ok(SkillResult {
                    success: true,
                    data: transformed,
                    messages,
                    execution_time_ms: start.elapsed().as_millis() as u64,
                    consciousness,
                    embedders_used: self.skill.definition.primary_embedders.clone(),
                })
            }
            Ok(Err(e)) | Err(_) => {
                let error = match mcp_result {
                    Ok(Err(e)) => e,
                    Err(_) => SkillError::Timeout(timeout_ms),
                    _ => unreachable!(),
                };

                // Attempt retries
                for attempt in 1..=retry_count {
                    let delay = self.calculate_backoff(&self.skill.definition.error_handling.backoff, attempt);
                    tokio::time::sleep(delay).await;

                    if let Ok(result) = self.invoke_mcp(&validated_params).await {
                        let transformed = self.transform_result(result)?;
                        return Ok(SkillResult {
                            success: true,
                            data: transformed,
                            messages,
                            execution_time_ms: start.elapsed().as_millis() as u64,
                            consciousness,
                            embedders_used: self.skill.definition.primary_embedders.clone(),
                        });
                    }
                }

                // Return fallback response if configured
                if let Some(fallback) = &self.skill.definition.error_handling.fallback_response {
                    return Ok(SkillResult {
                        success: false,
                        data: serde_json::json!({"fallback": fallback}),
                        messages: vec![fallback.clone()],
                        execution_time_ms: start.elapsed().as_millis() as u64,
                        consciousness,
                        embedders_used: vec![],
                    });
                }

                Err(error)
            }
        }
    }

    fn validate_and_apply_defaults(
        &self,
        mut params: serde_json::Value,
    ) -> Result<serde_json::Value, SkillError> {
        let params_obj = params.as_object_mut()
            .ok_or_else(|| SkillError::InvalidParams("Expected object".to_string()))?;

        for (name, def) in &self.skill.definition.parameters {
            if !params_obj.contains_key(name) {
                if def.required {
                    return Err(SkillError::MissingParam(name.clone()));
                }
                if let Some(default) = &def.default {
                    params_obj.insert(name.clone(), default.clone());
                }
            }
        }

        Ok(params)
    }

    async fn invoke_mcp(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SkillError> {
        let config = &self.skill.definition.mcp;

        self.mcp_client.invoke(
            &config.server,
            &config.tool,
            params.clone(),
        )
        .await
        .map_err(|e| SkillError::McpError(e.to_string()))
    }

    async fn run_success_action(
        &self,
        action: &SuccessAction,
        result: &serde_json::Value,
    ) -> Result<Option<String>, SkillError> {
        match action {
            SuccessAction::Log { message } => {
                let msg = self.interpolate(message, result)?;
                tracing::info!("{}", msg);
                Ok(Some(msg))
            }
            SuccessAction::NotifyHook { hook } => {
                tracing::debug!("Notifying hook: {}", hook);
                Ok(None)
            }
            SuccessAction::SetContext { key } => {
                tracing::debug!("Setting context key: {}", key);
                Ok(None)
            }
            SuccessAction::StoreGoals { path } => {
                tracing::debug!("Storing goals from: {}", path);
                Ok(None)
            }
            SuccessAction::EmitEvent { event } => {
                tracing::debug!("Emitting event: {}", event);
                Ok(None)
            }
            SuccessAction::BroadcastWorkspace { data } => {
                tracing::debug!("Broadcasting to workspace: {}", data);
                Ok(None)
            }
        }
    }

    fn transform_result(&self, result: serde_json::Value) -> Result<serde_json::Value, SkillError> {
        if let Some(config) = &self.skill.definition.transforms {
            if let Some(_expr) = &config.result {
                // Apply transformation expression
                // Full implementation would use expression engine
                return Ok(result);
            }
        }
        Ok(result)
    }

    fn interpolate(&self, template: &str, context: &serde_json::Value) -> Result<String, SkillError> {
        let mut result = template.to_string();

        let re = regex::Regex::new(r"\{\{([^}]+)\}\}").unwrap();
        for cap in re.captures_iter(template) {
            let key = &cap[1];
            let value = self.extract_value(context, key)
                .unwrap_or_else(|| serde_json::Value::String(format!("{{{{{}}}}}",key)));
            let replacement = match value {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            result = result.replace(&cap[0], &replacement);
        }

        Ok(result)
    }

    fn extract_value(&self, obj: &serde_json::Value, path: &str) -> Option<serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = obj;

        for part in parts {
            current = current.get(part)?;
        }

        Some(current.clone())
    }

    fn calculate_backoff(&self, backoff_type: &str, attempt: u32) -> Duration {
        match backoff_type {
            "exponential" => Duration::from_millis(100 * 2u64.pow(attempt - 1)),
            "linear" => Duration::from_millis(100 * attempt as u64),
            _ => Duration::from_millis(100),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("Skill not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Missing required parameter: {0}")]
    MissingParam(String),
    #[error("MCP error: {0}")]
    McpError(String),
    #[error("Timeout after {0}ms")]
    Timeout(u64),
}
```
    </section>

    <section name="MemorySearchSkill">
```markdown
<!-- .claude/skills/memory-search/SKILL.md -->
---
name: memory-search
description: Search teleological memory with consciousness-aware context injection using 13-embedding array
version: 1.0.0
auto_invoke: true
context_patterns:
  - "search.*memory"
  - "find.*context"
  - "recall.*information"
  - "look up.*memories"
  - "retrieve.*from memory"
error_handling:
  retry_count: 2
  fallback_response: "Memory search temporarily unavailable. Please try again."
  backoff: exponential
timeout_ms: 30000

mcp:
  server: context-graph
  tool: search_memories
  alternative_tools:
    - search_by_embedder
    - memory/search

primary_embedders:
  - E1_Semantic      # General semantic meaning (384-dim)
  - E12_Code         # Programming semantics (512-dim)
  - E3_Causal        # Cause-effect relationships (256-dim)
  - E13_SPLADE       # Sparse keyword expansion (30522-dim)

parameters:
  query:
    type: string
    required: true
    description: Search query text
  strategy:
    type: object
    required: false
    default:
      type: auto_discover
      max_entry_points: 5
      min_confidence: 0.6
    description: Search strategy configuration
  limit:
    type: integer
    required: false
    default: 10
    description: Maximum results to return
  threshold:
    type: number
    required: false
    default: 0.5
    description: Minimum similarity threshold
  namespace:
    type: string
    required: false
    description: Namespace to search within
  embedders:
    type: array
    required: false
    description: Specific embedders to use (overrides auto-discovery)

on_success:
  - type: log
    message: "Found {{result.memories.length}} memories for query"
  - type: set_context
    key: last_search_results
  - type: broadcast_workspace
    data: search_complete

on_error:
  - type: log_error
    message: "Memory search failed: {{error.message}}"
---

# Memory Search Skill

Search teleological memory using entry-point discovery across the 13-embedding array.
Automatically selects optimal embedding spaces for the query based on content analysis.

## Invocation

Via slash command: `/memory-search`

Auto-invokes when context matches patterns like:
- "search memory for..."
- "find context about..."
- "recall information on..."

## Primary Embedders

| Embedder | Dimension | Purpose |
|----------|-----------|---------|
| E1_Semantic | 384 | General semantic meaning |
| E12_Code | 512 | Programming language semantics |
| E3_Causal | 256 | Cause-effect relationships |
| E13_SPLADE | 30522 | Sparse keyword expansion |

## MCP Tools Invoked

- `search_memories` - Primary multi-embedder search
- `search_by_embedder` - Single embedder fallback

## Consciousness Integration

Invocation mode affects search depth:
- **Background (C<0.3)**: Quick semantic-only search
- **Standard (0.3-0.6)**: Multi-embedder with top-5 entry points
- **Priority (0.6-0.8)**: Full embedder sweep with RRF fusion
- **Critical (C>=0.8)**: Exhaustive search with consciousness broadcast

## Example Usage

```json
{
  "query": "authentication flow implementation",
  "strategy": {
    "type": "auto_discover",
    "max_entry_points": 5
  },
  "limit": 10
}
```
```
    </section>

    <section name="GoalAlignmentSkill">
```markdown
<!-- .claude/skills/goal-alignment/SKILL.md -->
---
name: goal-alignment
description: Check alignment between content and discovered goals using teleological embedders
version: 1.0.0
auto_invoke: true
context_patterns:
  - "check.*alignment"
  - "align.*with.*goal"
  - "verify.*purpose"
  - "goal.*match"
error_handling:
  retry_count: 2
  fallback_response: "Goal alignment check unavailable. Manual review recommended."
  backoff: exponential
timeout_ms: 30000

mcp:
  server: context-graph
  tool: calculate_alignment
  alternative_tools:
    - check_goal_drift
    - purpose/goal_alignment

primary_embedders:
  - E7_Teleological      # Purpose/goal orientation (256-dim)
  - E5_Moral             # Ethical alignment (128-dim)
  - E4_Counterfactual    # Alternative scenarios (256-dim)

parameters:
  content:
    type: string
    required: true
    description: Content to check alignment for
  goal_id:
    type: string
    required: false
    description: Specific goal ID to check against
  comparison_type:
    type: object
    required: false
    default:
      type: auto_discover
      max_entry_points: 5
    description: Comparison strategy

on_success:
  - type: log
    message: "Goal alignment: {{result.overall_alignment}}% with {{result.matched_goal.label}}"
  - type: notify_hook
    hook: post_alignment_check

on_error:
  - type: log_error
    message: "Goal alignment check failed: {{error.message}}"
---

# Goal Alignment Skill

Check alignment between content and discovered goals using the teleological
array comparison for accurate apples-to-apples measurement.

## Invocation

Via slash command: `/goal-alignment`

Auto-invokes when context matches patterns like:
- "check alignment with..."
- "verify purpose of..."
- "does this align with goal..."

## Primary Embedders

| Embedder | Dimension | Purpose |
|----------|-----------|---------|
| E7_Teleological | 256 | Purpose and goal orientation |
| E5_Moral | 128 | Ethical alignment assessment |
| E4_Counterfactual | 256 | Alternative scenario analysis |

## MCP Tools Invoked

- `calculate_alignment` - Primary alignment calculation
- `check_goal_drift` - Drift-aware alignment check

## Consciousness Integration

Higher consciousness levels enable:
- Per-embedder breakdown analysis
- Counterfactual scenario generation
- Moral implication assessment
- Recommendation generation

## Example Usage

```json
{
  "content": "Implementing user data export feature",
  "goal_id": "goal_privacy_compliance",
  "comparison_type": {
    "type": "weighted_full",
    "weights": {
      "E7_Teleological": 0.5,
      "E5_Moral": 0.3,
      "E4_Counterfactual": 0.2
    }
  }
}
```
```
    </section>

    <section name="PatternLearningSkill">
```markdown
<!-- .claude/skills/pattern-learning/SKILL.md -->
---
name: pattern-learning
description: Learn and store patterns from observations using behavioral and structural embedders
version: 1.0.0
auto_invoke: false
context_patterns:
  - "learn.*pattern"
  - "store.*pattern"
  - "remember.*approach"
error_handling:
  retry_count: 3
  fallback_response: "Pattern storage failed. Pattern noted for manual review."
  backoff: exponential
timeout_ms: 45000

mcp:
  server: context-graph
  tool: store_pattern
  alternative_tools:
    - extract_patterns
    - memory/inject

primary_embedders:
  - E10_Behavioral    # Action patterns (192-dim)
  - E9_Structural     # Code/document structure (192-dim)
  - E3_Causal         # Cause-effect relationships (256-dim)

parameters:
  pattern_type:
    type: string
    required: true
    values: [code, workflow, error, refactoring, test, architecture]
    description: Type of pattern to learn
  content:
    type: string
    required: true
    description: Pattern content to learn
  context:
    type: object
    required: false
    description: Additional context (file_path, language, framework)
  tags:
    type: array
    required: false
    description: Tags for categorization

on_success:
  - type: log
    message: "Learned {{params.pattern_type}} pattern: {{result.pattern_id}}"
  - type: emit_event
    event: pattern_learned

on_error:
  - type: log_error
    message: "Pattern learning failed: {{error.message}}"
---

# Pattern Learning Skill

Learn and store patterns from observations. Supports code patterns,
workflow patterns, error patterns, and architectural patterns for future recall.

## Invocation

Via slash command: `/pattern-learning`

This skill does NOT auto-invoke to prevent accidental pattern storage.
Explicit invocation required.

## Primary Embedders

| Embedder | Dimension | Purpose |
|----------|-----------|---------|
| E10_Behavioral | 192 | Action and workflow patterns |
| E9_Structural | 192 | Code and document structure |
| E3_Causal | 256 | Cause-effect relationships |

## MCP Tools Invoked

- `store_pattern` - Store pattern with embeddings
- `extract_patterns` - Extract patterns from content
- `memory/inject` - Low-level memory injection

## Supported Pattern Types

| Type | Description | Use Case |
|------|-------------|----------|
| code | Code snippets and idioms | Reusable code patterns |
| workflow | Process sequences | Development workflows |
| error | Error handling patterns | Common error resolutions |
| refactoring | Code transformations | Refactoring templates |
| test | Test patterns | Testing approaches |
| architecture | System design | Architectural decisions |

## Example Usage

```json
{
  "pattern_type": "code",
  "content": "async fn handle_error<T, E>(result: Result<T, E>) -> Option<T> { ... }",
  "context": {
    "language": "rust",
    "framework": "tokio",
    "file_path": "src/utils/error.rs"
  },
  "tags": ["error-handling", "async", "rust"]
}
```
```
    </section>

    <section name="ContextInjectionSkill">
```markdown
<!-- .claude/skills/context-injection/SKILL.md -->
---
name: context-injection
description: Inject content into teleological memory with consciousness-aware context selection
version: 1.0.0
auto_invoke: true
context_patterns:
  - "inject.*context"
  - "add.*to.*memory"
  - "store.*context"
  - "remember.*this"
error_handling:
  retry_count: 3
  fallback_response: "Context injection failed. Content queued for retry."
  backoff: exponential
timeout_ms: 30000

mcp:
  server: context-graph
  tool: inject_context
  alternative_tools:
    - preload_context
    - memory/inject

primary_embedders:
  - E8_Contextual    # Situational awareness (256-dim)
  - E1_Semantic      # General semantic meaning (384-dim)
  - E2_Temporal      # Time-aware recency (64-dim)

parameters:
  content:
    type: string
    required: true
    description: Content to inject into memory
  memory_type:
    type: string
    required: false
    default: general
    values: [code_context, documentation, code_snippet, conversation, general, pattern]
    description: Type of memory being stored
  namespace:
    type: string
    required: false
    default: default
    description: Namespace for organization
  metadata:
    type: object
    required: false
    default: {}
    description: Additional metadata to store
  priority:
    type: string
    required: false
    default: normal
    values: [low, normal, high, critical]
    description: Memory priority level

on_success:
  - type: log
    message: "Injected memory {{result.memory_id}} with {{result.embedders_generated}} embedders"
  - type: notify_hook
    hook: post_memory_inject
  - type: broadcast_workspace
    data: memory_injected

on_error:
  - type: log_error
    message: "Memory injection failed: {{error.message}}"
---

# Context Injection Skill

Inject content into teleological memory with autonomous embedding
across all 13 embedding dimensions. Supports priority-based storage
and consciousness-aware context selection.

## Invocation

Via slash command: `/context-injection`

Auto-invokes when context matches patterns like:
- "inject context about..."
- "add this to memory..."
- "remember this for later..."

## Primary Embedders

| Embedder | Dimension | Purpose |
|----------|-----------|---------|
| E8_Contextual | 256 | Situational awareness |
| E1_Semantic | 384 | General semantic meaning |
| E2_Temporal | 64 | Time-aware recency decay |

## MCP Tools Invoked

- `inject_context` - Primary context injection
- `preload_context` - Preload for anticipated needs
- `memory/inject` - Low-level memory storage

## Memory Types

| Type | Embedder Focus | Retention |
|------|----------------|-----------|
| code_context | E12_Code, E9_Structural | Long |
| documentation | E1_Semantic, E8_Contextual | Medium |
| code_snippet | E12_Code, E3_Causal | Long |
| conversation | E2_Temporal, E8_Contextual | Short |
| general | E1_Semantic | Medium |
| pattern | E10_Behavioral, E9_Structural | Long |

## Consciousness Integration

Injection mode based on consciousness level:
- **Background**: Async injection, minimal embedders
- **Standard**: Full 13-embedder generation
- **Priority**: Immediate injection with broadcast
- **Critical**: Synchronous with workspace notification

## Example Usage

```json
{
  "content": "The authentication system uses JWT tokens with 24h expiry...",
  "memory_type": "documentation",
  "namespace": "auth-system",
  "metadata": {
    "source": "architecture-doc",
    "version": "2.0"
  },
  "priority": "high"
}
```
```
    </section>

    <section name="DriftCheckSkill">
```markdown
<!-- .claude/skills/drift-check/SKILL.md -->
---
name: drift-check
description: Detect purpose drift between current work and established goals using per-embedder analysis
version: 1.0.0
auto_invoke: true
context_patterns:
  - "check.*drift"
  - "detect.*drift"
  - "purpose.*deviation"
  - "goal.*divergence"
error_handling:
  retry_count: 2
  fallback_response: "Drift detection unavailable. Manual goal review recommended."
  backoff: linear
timeout_ms: 45000

mcp:
  server: context-graph
  tool: detect_drift
  alternative_tools:
    - get_drift_report
    - purpose/drift_check

primary_embedders:
  - E7_Teleological      # Purpose/goal orientation (256-dim)
  - E4_Counterfactual    # Alternative scenarios (256-dim)
  - E5_Moral             # Ethical alignment (128-dim)

parameters:
  memory_ids:
    type: array
    required: true
    description: Memory IDs representing recent work
  goal_id:
    type: string
    required: true
    description: Goal to check drift against
  comparison_type:
    type: object
    required: false
    default:
      type: matrix_strategy
      matrix: semantic_focused
  include_trend:
    type: boolean
    required: false
    default: true
    description: Include drift trend analysis
  embedder_breakdown:
    type: boolean
    required: false
    default: true
    description: Include per-embedder drift analysis

on_success:
  - type: log
    message: "Drift check: {{result.overall_drift.drift_level}} drift detected"
  - type: emit_event
    event: drift_check_complete

on_error:
  - type: log_error
    message: "Drift check failed: {{error.message}}"
---

# Drift Check Skill

Check for purpose drift between current work and established goals.
Detects when work is diverging from intended direction using per-embedder
analysis from TASK-LOGIC-010.

## Invocation

Via slash command: `/drift-check`

Auto-invokes when context matches patterns like:
- "check for drift..."
- "detect purpose drift..."
- "has the work diverged..."

## Primary Embedders

| Embedder | Dimension | Purpose |
|----------|-----------|---------|
| E7_Teleological | 256 | Purpose and goal orientation |
| E4_Counterfactual | 256 | Alternative scenario analysis |
| E5_Moral | 128 | Ethical alignment assessment |

## MCP Tools Invoked

- `detect_drift` - Primary drift detection
- `get_drift_report` - Detailed drift report generation
- `purpose/drift_check` - Legacy drift check endpoint

## Drift Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| none | < 0.2 | Continue as planned |
| low | 0.2 - 0.4 | Monitor closely |
| medium | 0.4 - 0.6 | Review direction |
| high | 0.6 - 0.8 | Immediate attention |
| critical | >= 0.8 | Stop and realign |

## Per-Embedder Analysis

From TASK-LOGIC-010, drift is analyzed per embedder:
- Teleological drift: Direct purpose misalignment
- Counterfactual drift: Divergent alternative paths
- Moral drift: Ethical concern accumulation

## Consciousness Integration

Higher consciousness levels enable:
- Trend prediction (improving/stable/worsening)
- Projected time to critical drift
- Automatic course correction recommendations
- Workspace broadcast for team awareness

## Example Usage

```json
{
  "memory_ids": ["mem_abc123", "mem_def456", "mem_ghi789"],
  "goal_id": "goal_feature_privacy",
  "comparison_type": {
    "type": "matrix_strategy",
    "matrix": "semantic_focused"
  },
  "include_trend": true,
  "embedder_breakdown": true
}
```

## Response Format

```json
{
  "status": "MEDIUM drift",
  "hasDrifted": true,
  "driftScore": "45.2%",
  "trend": {
    "direction": "worsening",
    "velocity": "0.023",
    "warning": "Projected critical in 5 iterations"
  },
  "driftedAreas": [
    {"area": "Purpose alignment", "level": "high"},
    {"area": "Ethical consideration", "level": "medium"}
  ],
  "actions": [
    {
      "area": "Purpose alignment",
      "problem": "Work diverging from core privacy goal",
      "solution": "Refocus on data minimization principle"
    }
  ]
}
```
```
    </section>
  </pseudo_code>

  <timeout_configuration>
    Per Claude Code constitution, skills must respect timeout limits:

    | Invocation Mode | Base Timeout | Multiplier | Effective Timeout |
    |-----------------|--------------|------------|-------------------|
    | Background | 30000ms | 2.0x | 60000ms |
    | Standard | 30000ms | 1.0x | 30000ms |
    | Priority | 30000ms | 0.75x | 22500ms |
    | Critical | 30000ms | 0.5x | 15000ms |

    Skills should:
    1. Set reasonable `timeout_ms` in YAML frontmatter (default: 30000)
    2. Respect consciousness-based timeout multipliers
    3. Provide fallback responses when timeout occurs
    4. Log timeout events for monitoring
  </timeout_configuration>
</task_spec>
```
