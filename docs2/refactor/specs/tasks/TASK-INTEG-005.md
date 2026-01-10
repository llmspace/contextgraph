# TASK-INTEG-005: Edit Hooks Implementation

```xml
<task_spec>
  <task_id>TASK-INTEG-005</task_id>
  <title>Edit Hooks Implementation (PreToolUse/PostToolUse for Write|Edit with Pattern Training)</title>
  <status>pending</status>

  <objective>
    Implement PreToolUse and PostToolUse hook handlers for Write|Edit tool events that
    integrate with the 13-embedding teleological array system for change tracking,
    alignment validation, GWT consciousness metrics, and pattern learning from code edits.
  </objective>

  <rationale>
    Edit hooks are critical for:
    1. PreToolUse (Write|Edit): Validate changes against goal alignment before writing
    2. PostToolUse (Write|Edit): Track changes for learning and drift detection
    3. Pattern training: Learn from edit patterns with consciousness-aware prioritization
    4. Change embedding: Store edit contexts as 13-dimension teleological arrays for retrieval
    5. GWT Consciousness: Track integration, reflection, and differentiation metrics

    These hooks enable continuous learning from developer actions within Claude Code.

    Claude Code Hook Model:
    - Uses PreToolUse/PostToolUse events (NOT PreFileWrite/PostFileWrite)
    - Hook receives tool_name, tool_input via JSON stdin
    - Matcher pattern "Write|Edit" captures all file modification tools
    - Constitution mandates 3000ms maximum timeout for hooks
  </rationale>

  <dependencies>
    <dependency type="required">TASK-INTEG-004</dependency>    <!-- Hook protocol -->
    <dependency type="required">TASK-CORE-003</dependency>     <!-- TeleologicalArray type -->
    <dependency type="required">TASK-LOGIC-006</dependency>    <!-- Trajectory tracking -->
    <dependency type="required">TASK-LOGIC-010</dependency>    <!-- Drift detection -->
  </dependencies>

  <input_context_files>
    <file purpose="hook_protocol">crates/context-graph-mcp/src/hooks/protocol.rs</file>
    <file purpose="trajectory_tracking">crates/context-graph-core/src/autonomous/trajectory.rs</file>
    <file purpose="alignment_calculator">crates/context-graph-core/src/alignment/calculator.rs</file>
    <file purpose="drift_detector">crates/context-graph-core/src/autonomous/drift.rs</file>
    <file purpose="embedder_pipeline">crates/context-graph-core/src/teleology/embedder.rs</file>
  </input_context_files>

  <output_artifacts>
    <artifact type="source">crates/context-graph-mcp/src/hooks/edit.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/hooks/file_ops.rs</artifact>
    <artifact type="source">crates/context-graph-mcp/src/hooks/consciousness.rs</artifact>
    <artifact type="config">.claude/hooks/pre_tool_use_file.sh</artifact>
    <artifact type="config">.claude/hooks/post_tool_use_file.sh</artifact>
    <artifact type="config">.claude/settings.json</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/edit_hooks_test.rs</artifact>
  </output_artifacts>

  <definition_of_done>
    <criterion id="1">PreToolUse handler validates alignment before allowing writes</criterion>
    <criterion id="2">PostToolUse handler stores edits as 13-dimension teleological arrays</criterion>
    <criterion id="3">Pattern training extracts learnable patterns with consciousness prioritization</criterion>
    <criterion id="4">Edit contexts include before/after content for diff analysis</criterion>
    <criterion id="5">Alignment warnings emitted when writes may cause drift</criterion>
    <criterion id="6">Hook timeout respects 3000ms constitution limit</criterion>
    <criterion id="7">Pattern training runs asynchronously to avoid blocking</criterion>
    <criterion id="8">Shell scripts parse Claude Code's PreToolUse/PostToolUse JSON format</criterion>
    <criterion id="9">settings.json configures hooks with Write|Edit matcher pattern</criterion>
    <criterion id="10">GWT consciousness metrics computed for each edit (I, R, D, C)</criterion>
    <criterion id="11">All 13 embedders integrated (E1-E13 including SPLADE)</criterion>
  </definition_of_done>

  <estimated_complexity>High</estimated_complexity>

  <pseudo_code>
    <section name="ClaudeCodeConfiguration">
```json
// .claude/settings.json - Configure hooks for Claude Code
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [".claude/hooks/pre_tool_use_file.sh"],
        "timeout": 3000
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [".claude/hooks/post_tool_use_file.sh"],
        "timeout": 3000
      }
    ]
  }
}
```
    </section>

    <section name="13EmbedderTypes">
```rust
// crates/context-graph-core/src/teleology/embedder_types.rs

/// The complete 13-embedding teleological array system
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EmbedderType {
    /// E1: Semantic meaning and concepts
    E1_Semantic,
    /// E2: Temporal ordering and time relationships
    E2_Temporal,
    /// E3: Cause-effect relationships
    E3_Causal,
    /// E4: Alternative outcomes and what-if scenarios
    E4_Counterfactual,
    /// E5: Ethical and value judgments
    E5_Moral,
    /// E6: Style, elegance, and beauty
    E6_Aesthetic,
    /// E7: Purpose, goals, and intentions
    E7_Teleological,
    /// E8: Environmental and situational context
    E8_Contextual,
    /// E9: Structural patterns and organization
    E9_Structural,
    /// E10: Actions and behaviors
    E10_Behavioral,
    /// E11: Emotional tone and affect
    E11_Emotional,
    /// E12: Code-specific semantics and patterns
    E12_Code,
    /// E13: Sparse Lexical and Expansion for keyword matching
    E13_SPLADE,
}

impl EmbedderType {
    /// All 13 embedders in canonical order
    pub const ALL: [EmbedderType; 13] = [
        Self::E1_Semantic,
        Self::E2_Temporal,
        Self::E3_Causal,
        Self::E4_Counterfactual,
        Self::E5_Moral,
        Self::E6_Aesthetic,
        Self::E7_Teleological,
        Self::E8_Contextual,
        Self::E9_Structural,
        Self::E10_Behavioral,
        Self::E11_Emotional,
        Self::E12_Code,
        Self::E13_SPLADE,
    ];

    /// Fast embedders for pre-write validation (must complete within timeout)
    pub const FAST_VALIDATION: [EmbedderType; 4] = [
        Self::E1_Semantic,
        Self::E12_Code,
        Self::E3_Causal,
        Self::E7_Teleological,
    ];

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::E1_Semantic => "semantic",
            Self::E2_Temporal => "temporal",
            Self::E3_Causal => "causal",
            Self::E4_Counterfactual => "counterfactual",
            Self::E5_Moral => "moral",
            Self::E6_Aesthetic => "aesthetic",
            Self::E7_Teleological => "teleological",
            Self::E8_Contextual => "contextual",
            Self::E9_Structural => "structural",
            Self::E10_Behavioral => "behavioral",
            Self::E11_Emotional => "emotional",
            Self::E12_Code => "code",
            Self::E13_SPLADE => "splade",
        }
    }
}
```
    </section>

    <section name="ConsciousnessMetrics">
```rust
// crates/context-graph-mcp/src/hooks/consciousness.rs

use serde::{Deserialize, Serialize};

/// GWT (Global Workspace Theory) consciousness metrics for edits
/// Based on constitution consciousness framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditConsciousnessMetrics {
    /// Integration score I(t) - how well edit integrates with codebase
    pub integration_score: f32,
    /// Reflection depth R(t) - self-referential awareness of changes
    pub reflection_depth: f32,
    /// Differentiation index D(t) - uniqueness of information added
    pub differentiation_index: f32,
    /// Overall consciousness level C(t) = I(t) * R(t) * D(t)
    pub consciousness_level: f32,
    /// Derived significance classification
    pub edit_significance: EditSignificance,
}

/// Classification of edit significance based on consciousness level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EditSignificance {
    /// C < 0.3: Low consciousness involvement, routine changes
    Routine,
    /// 0.3 <= C < 0.6: Moderate consciousness, standard development
    Moderate,
    /// 0.6 <= C < 0.8: Significant consciousness, important changes
    Significant,
    /// C >= 0.8: Full conscious awareness required, critical changes
    Critical,
}

impl EditConsciousnessMetrics {
    /// Compute consciousness metrics for an edit
    pub fn compute(
        edit_context: &EditContext,
        related_memories: &[MemoryNode],
        teleological_array: &TeleologicalArray,
    ) -> Self {
        // I(t): Integration - how well does this connect to existing code?
        let integration_score = Self::compute_integration(
            edit_context,
            related_memories,
        );

        // R(t): Reflection - does the edit show self-awareness?
        let reflection_depth = Self::compute_reflection(
            edit_context,
            teleological_array,
        );

        // D(t): Differentiation - how unique/novel is this change?
        let differentiation_index = Self::compute_differentiation(
            edit_context,
            related_memories,
            teleological_array,
        );

        // C(t) = I(t) * R(t) * D(t)
        let consciousness_level = integration_score * reflection_depth * differentiation_index;

        let edit_significance = match consciousness_level {
            c if c < 0.3 => EditSignificance::Routine,
            c if c < 0.6 => EditSignificance::Moderate,
            c if c < 0.8 => EditSignificance::Significant,
            _ => EditSignificance::Critical,
        };

        Self {
            integration_score,
            reflection_depth,
            differentiation_index,
            consciousness_level,
            edit_significance,
        }
    }

    fn compute_integration(
        edit_context: &EditContext,
        related_memories: &[MemoryNode],
    ) -> f32 {
        // Higher integration = more connections to existing code
        let connection_count = related_memories.len() as f32;
        let max_connections = 20.0;
        (connection_count / max_connections).min(1.0)
    }

    fn compute_reflection(
        edit_context: &EditContext,
        teleological_array: &TeleologicalArray,
    ) -> f32 {
        // Reflection based on teleological (purposeful) embeddings
        // Higher if edit has clear goals and self-referential patterns
        let teleological_strength = teleological_array
            .get_embedding(EmbedderType::E7_Teleological)
            .map(|e| e.magnitude())
            .unwrap_or(0.0);

        let behavioral_strength = teleological_array
            .get_embedding(EmbedderType::E10_Behavioral)
            .map(|e| e.magnitude())
            .unwrap_or(0.0);

        (teleological_strength + behavioral_strength) / 2.0
    }

    fn compute_differentiation(
        edit_context: &EditContext,
        related_memories: &[MemoryNode],
        teleological_array: &TeleologicalArray,
    ) -> f32 {
        // Differentiation = how unique compared to related memories
        if related_memories.is_empty() {
            return 1.0; // Completely novel
        }

        let avg_similarity: f32 = related_memories
            .iter()
            .map(|m| m.teleological_array.cosine_similarity(teleological_array))
            .sum::<f32>() / related_memories.len() as f32;

        // Invert: lower similarity = higher differentiation
        1.0 - avg_similarity
    }
}

impl EditSignificance {
    /// Get storage priority based on consciousness significance
    pub fn storage_priority(&self) -> StoragePriority {
        match self {
            Self::Routine => StoragePriority::Low,
            Self::Moderate => StoragePriority::Normal,
            Self::Significant => StoragePriority::High,
            Self::Critical => StoragePriority::Critical,
        }
    }

    /// Whether to trigger immediate pattern learning
    pub fn requires_immediate_learning(&self) -> bool {
        matches!(self, Self::Significant | Self::Critical)
    }
}
```
    </section>

    <section name="PreToolUseHandler">
```rust
// crates/context-graph-mcp/src/hooks/edit.rs

use crate::hooks::protocol::{HookEvent, HookPayload, HookResponse, HookStatus};
use crate::hooks::consciousness::{EditConsciousnessMetrics, EditSignificance};
use context_graph_core::teleology::{TeleologicalArray, TeleologicalComparator, EmbedderType};
use context_graph_core::autonomous::drift::TeleologicalDriftDetector;
use context_graph_core::alignment::TeleologicalAlignmentCalculator;

/// Claude Code PreToolUse input format for Write|Edit tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodePreToolUseInput {
    /// Tool name: "Write" or "Edit"
    pub tool_name: String,
    /// Tool-specific input parameters
    pub tool_input: ToolInputParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputParams {
    /// File path being written (Write tool)
    pub file_path: Option<String>,
    /// Content to write (Write tool)
    pub content: Option<String>,
    /// For Edit tool - the old string to replace
    pub old_string: Option<String>,
    /// For Edit tool - the new string replacement
    pub new_string: Option<String>,
}

/// Pre-tool-use hook handler for alignment validation (Write|Edit matcher)
pub struct PreToolUseFileHandler {
    /// Teleological store for retrieving related content
    store: Arc<TeleologicalStore>,
    /// Alignment calculator for goal checking
    alignment_calculator: Arc<TeleologicalAlignmentCalculator>,
    /// Drift detector for change impact analysis
    drift_detector: Arc<TeleologicalDriftDetector>,
    /// Embedding pipeline for generating preview arrays
    embedder_pipeline: Arc<EmbedderPipeline>,
    /// Configuration
    config: PreToolUseFileConfig,
}

#[derive(Debug, Clone)]
pub struct PreToolUseFileConfig {
    /// Minimum alignment score to allow write (0.0 to 1.0)
    pub min_alignment_threshold: f32,
    /// Whether to block on low alignment (vs just warn)
    pub block_on_low_alignment: bool,
    /// Maximum latency before skipping validation (ms)
    /// CONSTITUTION MANDATE: Must not exceed 3000ms
    pub max_validation_latency_ms: u64,
    /// Embedders to use for quick validation (subset of 13)
    pub validation_embedders: Vec<EmbedderType>,
}

impl Default for PreToolUseFileConfig {
    fn default() -> Self {
        Self {
            min_alignment_threshold: 0.5,
            block_on_low_alignment: false,
            // Constitution mandates 3000ms max for hooks
            max_validation_latency_ms: 3000,
            // Use fast embedders for pre-write validation (FAST_VALIDATION subset)
            validation_embedders: vec![
                EmbedderType::E1_Semantic,
                EmbedderType::E12_Code,
                EmbedderType::E3_Causal,
                EmbedderType::E7_Teleological,
            ],
        }
    }
}

/// Payload for file write hooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWritePayload {
    /// File path being written
    pub file_path: String,
    /// New content to be written
    pub new_content: String,
    /// Original content (if file exists)
    pub original_content: Option<String>,
    /// Operation type (create, update, append)
    pub operation: FileOperation,
    /// Tool that initiated the write (Write, Edit, MultiEdit)
    pub source_tool: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileOperation {
    Create,
    Update,
    Append,
    Delete,
}

impl PreToolUseFileHandler {
    pub fn new(
        store: Arc<TeleologicalStore>,
        alignment_calculator: Arc<TeleologicalAlignmentCalculator>,
        drift_detector: Arc<TeleologicalDriftDetector>,
        embedder_pipeline: Arc<EmbedderPipeline>,
        config: PreToolUseFileConfig,
    ) -> Self {
        Self {
            store,
            alignment_calculator,
            drift_detector,
            embedder_pipeline,
            config,
        }
    }

    /// Parse Claude Code's PreToolUse JSON input
    pub fn parse_claude_code_input(input: &str) -> Result<FileWritePayload, HookError> {
        let claude_input: ClaudeCodePreToolUseInput = serde_json::from_str(input)
            .map_err(|e| HookError::ParseError(format!("Invalid Claude Code input: {}", e)))?;

        // Extract file path and content based on tool type
        let (file_path, new_content, old_content) = match claude_input.tool_name.as_str() {
            "Write" => {
                let path = claude_input.tool_input.file_path
                    .ok_or_else(|| HookError::ParseError("Write tool missing file_path".into()))?;
                let content = claude_input.tool_input.content
                    .ok_or_else(|| HookError::ParseError("Write tool missing content".into()))?;
                (path, content, None)
            }
            "Edit" => {
                let path = claude_input.tool_input.file_path
                    .ok_or_else(|| HookError::ParseError("Edit tool missing file_path".into()))?;
                let new_str = claude_input.tool_input.new_string
                    .ok_or_else(|| HookError::ParseError("Edit tool missing new_string".into()))?;
                let old_str = claude_input.tool_input.old_string;
                (path, new_str, old_str)
            }
            other => {
                return Err(HookError::UnsupportedTool(other.to_string()));
            }
        };

        Ok(FileWritePayload {
            file_path,
            new_content,
            original_content: old_content,
            operation: if old_content.is_some() { FileOperation::Update } else { FileOperation::Create },
            source_tool: claude_input.tool_name,
        })
    }

    /// Handle PreToolUse event for Write|Edit tools
    pub async fn handle(&self, payload: FileWritePayload) -> Result<HookResponse, HookError> {
        let start = Instant::now();

        // 1. Generate partial teleological array for new content (fast embedders only)
        let preview_array = self.generate_preview_array(&payload.new_content).await?;

        // 2. Check if we're running out of time
        if start.elapsed().as_millis() > (self.config.max_validation_latency_ms / 2) as u128 {
            // Skip validation if taking too long
            return Ok(HookResponse {
                status: HookStatus::Proceed,
                message: Some("Validation skipped due to time constraints".to_string()),
                warnings: vec![],
                injected_context: None,
                metrics: HookMetrics::timeout(),
            });
        }

        // 3. Find active goals related to this file
        let related_goals = self.find_related_goals(&payload.file_path).await?;

        // 4. Calculate alignment against active goals
        let mut alignment_scores = Vec::new();
        let mut warnings = Vec::new();

        for goal in &related_goals {
            let alignment = self.alignment_calculator.calculate_alignment(
                &preview_array,
                &goal.teleological_array,
                ComparisonStrategy::fast_validation(),
            )?;

            alignment_scores.push((goal.goal_id.clone(), alignment.overall_score));

            if alignment.overall_score < self.config.min_alignment_threshold {
                warnings.push(AlignmentWarning {
                    goal_id: goal.goal_id.clone(),
                    goal_label: goal.label.clone(),
                    alignment_score: alignment.overall_score,
                    threshold: self.config.min_alignment_threshold,
                    drifted_embedders: alignment.low_scoring_embedders(),
                    recommendation: self.generate_recommendation(&alignment, &goal),
                });
            }
        }

        // 5. Check for potential drift if original content exists
        let drift_warnings = if let Some(ref original) = payload.original_content {
            self.check_edit_drift(&original, &payload.new_content, &preview_array).await?
        } else {
            vec![]
        };

        warnings.extend(drift_warnings.into_iter().map(|d| AlignmentWarning {
            goal_id: "drift".to_string(),
            goal_label: format!("Drift detected in {}", d.embedder),
            alignment_score: 1.0 - d.drift_magnitude,
            threshold: self.config.min_alignment_threshold,
            drifted_embedders: vec![d.embedder],
            recommendation: d.recommendation,
        }));

        // 6. Determine response status
        let status = if warnings.is_empty() {
            HookStatus::Proceed
        } else if self.config.block_on_low_alignment &&
                  warnings.iter().any(|w| w.alignment_score < 0.3) {
            HookStatus::Block
        } else {
            HookStatus::ProceedWithWarning
        };

        // 7. Build response
        Ok(HookResponse {
            status,
            message: self.build_status_message(&warnings),
            warnings: warnings.iter().map(|w| w.to_string()).collect(),
            injected_context: Some(serde_json::json!({
                "alignment_scores": alignment_scores,
                "preview_embedders": self.config.validation_embedders,
                "validation_latency_ms": start.elapsed().as_millis(),
            })),
            metrics: HookMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embedders_used: self.config.validation_embedders.len(),
                goals_checked: related_goals.len(),
            },
        })
    }

    /// Generate preview array using fast embedders only
    async fn generate_preview_array(&self, content: &str) -> Result<TeleologicalArray, HookError> {
        self.embedder_pipeline.generate_partial(
            content,
            &self.config.validation_embedders,
        ).await.map_err(HookError::EmbeddingFailed)
    }

    /// Find goals related to the file path
    async fn find_related_goals(&self, file_path: &str) -> Result<Vec<GoalNode>, HookError> {
        // Search for goals by file path metadata
        self.store.search_goals_by_file(file_path, 5).await
            .map_err(HookError::StorageError)
    }

    /// Check for drift between original and new content
    async fn check_edit_drift(
        &self,
        original: &str,
        new_content: &str,
        new_array: &TeleologicalArray,
    ) -> Result<Vec<DriftWarning>, HookError> {
        // Generate array for original content
        let original_array = self.generate_preview_array(original).await?;

        // Analyze drift
        let drift_report = self.drift_detector.analyze_drift(
            &original_array,
            new_array,
            DriftAnalysisConfig::quick(),
        )?;

        // Return significant drifts
        Ok(drift_report.per_embedder
            .into_iter()
            .filter(|(_, d)| d.drift_level >= DriftLevel::Medium)
            .map(|(embedder, d)| DriftWarning {
                embedder,
                drift_magnitude: d.similarity,
                drift_level: d.drift_level,
                recommendation: d.recommendation,
            })
            .collect())
    }

    fn generate_recommendation(&self, alignment: &AlignmentResult, goal: &GoalNode) -> String {
        let weak_embedders: Vec<_> = alignment.per_embedder
            .iter()
            .filter(|(_, score)| *score < 0.5)
            .map(|(e, _)| e.short_name())
            .collect();

        if weak_embedders.is_empty() {
            "Changes are well-aligned with the goal.".to_string()
        } else {
            format!(
                "Consider reviewing changes for {} alignment with goal '{}'. Weak areas: {}",
                goal.level.display_name(),
                goal.label,
                weak_embedders.join(", ")
            )
        }
    }

    fn build_status_message(&self, warnings: &[AlignmentWarning]) -> Option<String> {
        if warnings.is_empty() {
            None
        } else {
            Some(format!(
                "{} alignment warning(s) detected. Review before proceeding.",
                warnings.len()
            ))
        }
    }
}

#[derive(Debug)]
struct AlignmentWarning {
    goal_id: String,
    goal_label: String,
    alignment_score: f32,
    threshold: f32,
    drifted_embedders: Vec<EmbedderType>,
    recommendation: String,
}

impl std::fmt::Display for AlignmentWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Goal '{}' alignment: {:.2} (threshold: {:.2}) - {}",
            self.goal_label, self.alignment_score, self.threshold, self.recommendation
        )
    }
}
```
    </section>

    <section name="PostToolUseHandler">
```rust
/// Claude Code PostToolUse input format for Write|Edit tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodePostToolUseInput {
    /// Tool name: "Write" or "Edit"
    pub tool_name: String,
    /// Tool-specific input parameters
    pub tool_input: ToolInputParams,
    /// Tool response (success/failure info)
    pub tool_response: Option<ToolResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse {
    /// Whether the tool succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Any output from the tool
    pub output: Option<String>,
}

/// Post-tool-use hook handler for tracking and learning (Write|Edit matcher)
pub struct PostToolUseFileHandler {
    /// Teleological store for persisting edit memories
    store: Arc<TeleologicalStore>,
    /// Full embedding pipeline (all 13 embedders)
    embedder_pipeline: Arc<EmbedderPipeline>,
    /// Trajectory tracker for learning
    trajectory_tracker: Arc<TrajectoryTracker>,
    /// Pattern extractor for edit patterns
    pattern_extractor: Arc<EditPatternExtractor>,
    /// Background task queue
    task_queue: Arc<BackgroundTaskQueue>,
    /// Consciousness metrics calculator
    consciousness_calculator: ConsciousnessCalculator,
    /// Configuration
    config: PostToolUseFileConfig,
}

#[derive(Debug, Clone)]
pub struct PostToolUseFileConfig {
    /// Whether to store all edits or only significant ones
    pub store_all_edits: bool,
    /// Minimum content length to trigger storage
    pub min_content_length: usize,
    /// File extensions to process
    pub process_extensions: HashSet<String>,
    /// Whether to run pattern training synchronously
    pub sync_pattern_training: bool,
    /// Maximum edits before triggering batch pattern analysis
    pub pattern_batch_size: usize,
    /// Consciousness-aware storage: prioritize high-consciousness edits
    pub consciousness_aware_storage: bool,
    /// Minimum consciousness level to trigger immediate pattern learning
    pub immediate_learning_threshold: f32,
}

impl Default for PostToolUseFileConfig {
    fn default() -> Self {
        let mut extensions = HashSet::new();
        extensions.extend([
            "rs", "ts", "tsx", "js", "jsx", "py", "go", "java",
            "md", "json", "yaml", "yml", "toml"
        ].iter().map(|s| s.to_string()));

        Self {
            store_all_edits: true,
            min_content_length: 50,
            process_extensions: extensions,
            sync_pattern_training: false,
            pattern_batch_size: 100,
            consciousness_aware_storage: true,
            // Significant or Critical edits get immediate learning
            immediate_learning_threshold: 0.6,
        }
    }
}

/// Edit context for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditContext {
    /// Unique ID for this edit
    pub edit_id: Uuid,
    /// File path that was edited
    pub file_path: String,
    /// Content before edit (if available)
    pub before_content: Option<String>,
    /// Content after edit
    pub after_content: String,
    /// Diff representation
    pub diff: Option<String>,
    /// Edit operation type
    pub operation: FileOperation,
    /// Source tool (Write, Edit from Claude Code)
    pub source_tool: String,
    /// Session ID
    pub session_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Related memories discovered during edit
    pub related_memories: Vec<Uuid>,
    /// Alignment score at time of edit
    pub alignment_snapshot: Option<f32>,
    /// GWT consciousness metrics for this edit
    pub consciousness_metrics: Option<EditConsciousnessMetrics>,
    /// Storage priority based on consciousness level
    pub storage_priority: StoragePriority,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StoragePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl PostToolUseFileHandler {
    pub fn new(
        store: Arc<TeleologicalStore>,
        embedder_pipeline: Arc<EmbedderPipeline>,
        trajectory_tracker: Arc<TrajectoryTracker>,
        pattern_extractor: Arc<EditPatternExtractor>,
        task_queue: Arc<BackgroundTaskQueue>,
        consciousness_calculator: ConsciousnessCalculator,
        config: PostToolUseFileConfig,
    ) -> Self {
        Self {
            store,
            embedder_pipeline,
            trajectory_tracker,
            pattern_extractor,
            task_queue,
            consciousness_calculator,
            config,
        }
    }

    /// Parse Claude Code's PostToolUse JSON input
    pub fn parse_claude_code_input(input: &str) -> Result<(FileWritePayload, Option<ToolResponse>), HookError> {
        let claude_input: ClaudeCodePostToolUseInput = serde_json::from_str(input)
            .map_err(|e| HookError::ParseError(format!("Invalid Claude Code input: {}", e)))?;

        // Extract file path and content based on tool type
        let (file_path, new_content, old_content) = match claude_input.tool_name.as_str() {
            "Write" => {
                let path = claude_input.tool_input.file_path
                    .ok_or_else(|| HookError::ParseError("Write tool missing file_path".into()))?;
                let content = claude_input.tool_input.content
                    .ok_or_else(|| HookError::ParseError("Write tool missing content".into()))?;
                (path, content, None)
            }
            "Edit" => {
                let path = claude_input.tool_input.file_path
                    .ok_or_else(|| HookError::ParseError("Edit tool missing file_path".into()))?;
                let new_str = claude_input.tool_input.new_string
                    .ok_or_else(|| HookError::ParseError("Edit tool missing new_string".into()))?;
                let old_str = claude_input.tool_input.old_string;
                (path, new_str, old_str)
            }
            other => {
                return Err(HookError::UnsupportedTool(other.to_string()));
            }
        };

        let payload = FileWritePayload {
            file_path,
            new_content,
            original_content: old_content,
            operation: if old_content.is_some() { FileOperation::Update } else { FileOperation::Create },
            source_tool: claude_input.tool_name,
        };

        Ok((payload, claude_input.tool_response))
    }

    /// Handle PostToolUse event for Write|Edit tools
    pub async fn handle(&self, payload: FileWritePayload) -> Result<HookResponse, HookError> {
        let start = Instant::now();

        // 1. Check if we should process this file
        if !self.should_process(&payload) {
            return Ok(HookResponse::skip("File type not configured for processing"));
        }

        // 2. Generate full 13-dimension teleological array for new content
        let teleological_array = self.embedder_pipeline
            .generate_full(&payload.new_content, &EmbedderType::ALL)
            .await
            .map_err(HookError::EmbeddingFailed)?;

        // 3. Find related memories for consciousness calculation
        let related_memories = self.store
            .search_similar(&teleological_array, 10)
            .await
            .unwrap_or_default();

        // 4. Compute GWT consciousness metrics
        let consciousness_metrics = EditConsciousnessMetrics::compute(
            &payload,
            &related_memories,
            &teleological_array,
        );

        // 5. Determine storage priority based on consciousness level
        let storage_priority = consciousness_metrics.edit_significance.storage_priority();

        // 6. Create edit context with consciousness metrics
        let edit_context = EditContext {
            edit_id: Uuid::new_v4(),
            file_path: payload.file_path.clone(),
            before_content: payload.original_content.clone(),
            after_content: payload.new_content.clone(),
            diff: self.compute_diff(&payload.original_content, &payload.new_content),
            operation: payload.operation.clone(),
            source_tool: payload.source_tool.clone(),
            session_id: self.get_current_session_id(),
            timestamp: Utc::now(),
            related_memories: related_memories.iter().map(|m| m.id).collect(),
            alignment_snapshot: None,
            consciousness_metrics: Some(consciousness_metrics.clone()),
            storage_priority,
        };

        // 7. Store in teleological store with priority
        let memory_id = self.store.inject(InjectionRequest {
            content: payload.new_content.clone(),
            teleological_array: teleological_array.clone(),
            memory_type: MemoryType::CodeEdit,
            namespace: self.derive_namespace(&payload.file_path),
            metadata: serde_json::to_value(&edit_context)?,
            priority: storage_priority,
        }).await.map_err(HookError::StorageError)?;

        // 8. Record in trajectory tracker
        let trajectory_step = TrajectoryStep {
            action: TrajectoryAction::FileWrite {
                file_path: payload.file_path.clone(),
                memory_id,
            },
            timestamp: Utc::now(),
            context_size: payload.new_content.len(),
            outcome: None, // Will be filled by verdict system
            consciousness_level: Some(consciousness_metrics.consciousness_level),
        };

        self.trajectory_tracker.record_step(trajectory_step).await?;

        // 9. Queue pattern training - consciousness-aware prioritization
        let pattern_task = PatternTrainingTask {
            edit_id: edit_context.edit_id,
            memory_id,
            before_content: payload.original_content,
            after_content: payload.new_content,
            file_path: payload.file_path.clone(),
            consciousness_level: consciousness_metrics.consciousness_level,
        };

        // Use consciousness level to decide sync vs async training
        let requires_immediate = consciousness_metrics.edit_significance.requires_immediate_learning()
            || consciousness_metrics.consciousness_level >= self.config.immediate_learning_threshold;

        if self.config.sync_pattern_training || requires_immediate {
            self.train_patterns(pattern_task).await?;
        } else {
            self.task_queue.enqueue(BackgroundTask::PatternTraining(pattern_task));
        }

        // 10. Check if we should trigger batch pattern analysis
        let pending_edits = self.trajectory_tracker.pending_edit_count().await;
        if pending_edits >= self.config.pattern_batch_size {
            self.task_queue.enqueue(BackgroundTask::BatchPatternAnalysis {
                session_id: self.get_current_session_id(),
            });
        }

        Ok(HookResponse {
            status: HookStatus::Success,
            message: Some(format!("Edit stored as memory {} (consciousness: {:.2})",
                memory_id, consciousness_metrics.consciousness_level)),
            warnings: vec![],
            injected_context: Some(serde_json::json!({
                "memory_id": memory_id,
                "edit_id": edit_context.edit_id,
                "embedders_generated": 13,
                "embedder_types": ["E1_Semantic", "E2_Temporal", "E3_Causal", "E4_Counterfactual",
                                   "E5_Moral", "E6_Aesthetic", "E7_Teleological", "E8_Contextual",
                                   "E9_Structural", "E10_Behavioral", "E11_Emotional", "E12_Code", "E13_SPLADE"],
                "pattern_training_queued": !requires_immediate,
                "consciousness": {
                    "integration": consciousness_metrics.integration_score,
                    "reflection": consciousness_metrics.reflection_depth,
                    "differentiation": consciousness_metrics.differentiation_index,
                    "level": consciousness_metrics.consciousness_level,
                    "significance": format!("{:?}", consciousness_metrics.edit_significance),
                },
                "storage_priority": format!("{:?}", storage_priority),
            })),
            metrics: HookMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embedders_used: 13,
                goals_checked: 0,
            },
        })
    }

    fn should_process(&self, payload: &FileWritePayload) -> bool {
        // Check content length
        if payload.new_content.len() < self.config.min_content_length {
            return false;
        }

        // Check file extension
        let extension = std::path::Path::new(&payload.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        self.config.process_extensions.contains(extension)
    }

    fn compute_diff(&self, before: &Option<String>, after: &str) -> Option<String> {
        before.as_ref().map(|b| {
            // Generate unified diff
            let diff = similar::TextDiff::from_lines(b, after);
            diff.unified_diff()
                .header("before", "after")
                .to_string()
        })
    }

    fn derive_namespace(&self, file_path: &str) -> String {
        // Extract namespace from file path structure
        let path = std::path::Path::new(file_path);

        // Use parent directory as namespace
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string()
    }

    fn get_current_session_id(&self) -> String {
        // Get from session context
        SESSION_CONTEXT.with(|ctx| ctx.borrow().session_id.clone())
            .unwrap_or_else(|| format!("session-{}", Utc::now().format("%Y%m%d")))
    }

    /// Train patterns from edit
    async fn train_patterns(&self, task: PatternTrainingTask) -> Result<(), HookError> {
        // Extract patterns from the edit
        let patterns = self.pattern_extractor.extract_patterns(
            task.before_content.as_deref(),
            &task.after_content,
            &task.file_path,
        )?;

        // Store patterns for learning
        for pattern in patterns {
            self.store.store_pattern(pattern).await?;
        }

        Ok(())
    }
}

/// Pattern extractor for learning from edits
pub struct EditPatternExtractor {
    /// AST parser for code analysis
    ast_parser: Arc<AstParser>,
    /// Pattern matcher for common edit types
    pattern_matcher: PatternMatcher,
}

impl EditPatternExtractor {
    /// Extract learnable patterns from an edit
    pub fn extract_patterns(
        &self,
        before: Option<&str>,
        after: &str,
        file_path: &str,
    ) -> Result<Vec<EditPattern>, PatternError> {
        let mut patterns = Vec::new();

        // 1. Detect edit type
        let edit_type = self.classify_edit(before, after);

        // 2. Extract structural patterns (for code files)
        if let Some(lang) = self.detect_language(file_path) {
            let structural = self.extract_structural_patterns(before, after, lang)?;
            patterns.extend(structural);
        }

        // 3. Extract semantic patterns
        let semantic = self.extract_semantic_patterns(before, after)?;
        patterns.extend(semantic);

        // 4. Extract refactoring patterns
        if let Some(before_content) = before {
            let refactoring = self.extract_refactoring_patterns(before_content, after)?;
            patterns.extend(refactoring);
        }

        Ok(patterns)
    }

    fn classify_edit(&self, before: Option<&str>, after: &str) -> EditType {
        match before {
            None => EditType::Creation,
            Some(b) if b.is_empty() => EditType::Creation,
            Some(b) if after.is_empty() => EditType::Deletion,
            Some(b) => {
                let similarity = strsim::jaro_winkler(b, after) as f32;
                if similarity > 0.9 {
                    EditType::MinorModification
                } else if similarity > 0.5 {
                    EditType::MajorModification
                } else {
                    EditType::Rewrite
                }
            }
        }
    }

    fn detect_language(&self, file_path: &str) -> Option<Language> {
        let ext = std::path::Path::new(file_path)
            .extension()
            .and_then(|e| e.to_str())?;

        match ext {
            "rs" => Some(Language::Rust),
            "ts" | "tsx" => Some(Language::TypeScript),
            "js" | "jsx" => Some(Language::JavaScript),
            "py" => Some(Language::Python),
            "go" => Some(Language::Go),
            _ => None,
        }
    }

    fn extract_structural_patterns(
        &self,
        before: Option<&str>,
        after: &str,
        lang: Language,
    ) -> Result<Vec<EditPattern>, PatternError> {
        // Parse ASTs
        let after_ast = self.ast_parser.parse(after, lang)?;

        let mut patterns = Vec::new();

        // Detect function/method changes
        if let Some(functions) = after_ast.find_functions() {
            for func in functions {
                patterns.push(EditPattern::FunctionDefinition {
                    name: func.name.clone(),
                    signature_hash: func.signature_hash(),
                    language: lang,
                });
            }
        }

        // Detect import changes
        if let Some(imports) = after_ast.find_imports() {
            for import in imports {
                patterns.push(EditPattern::ImportAddition {
                    module: import.module.clone(),
                    items: import.items.clone(),
                    language: lang,
                });
            }
        }

        Ok(patterns)
    }

    fn extract_semantic_patterns(
        &self,
        before: Option<&str>,
        after: &str,
    ) -> Result<Vec<EditPattern>, PatternError> {
        let mut patterns = Vec::new();

        // Detect common patterns
        if self.pattern_matcher.is_error_handling_addition(before, after) {
            patterns.push(EditPattern::ErrorHandling);
        }

        if self.pattern_matcher.is_logging_addition(before, after) {
            patterns.push(EditPattern::LoggingAddition);
        }

        if self.pattern_matcher.is_test_addition(before, after) {
            patterns.push(EditPattern::TestAddition);
        }

        if self.pattern_matcher.is_documentation_addition(before, after) {
            patterns.push(EditPattern::DocumentationAddition);
        }

        Ok(patterns)
    }

    fn extract_refactoring_patterns(
        &self,
        before: &str,
        after: &str,
    ) -> Result<Vec<EditPattern>, PatternError> {
        let mut patterns = Vec::new();

        // Detect rename refactoring
        if let Some(rename) = self.pattern_matcher.detect_rename(before, after) {
            patterns.push(EditPattern::Rename {
                old_name: rename.old_name,
                new_name: rename.new_name,
            });
        }

        // Detect extract refactoring
        if let Some(extract) = self.pattern_matcher.detect_extraction(before, after) {
            patterns.push(EditPattern::ExtractFunction {
                extracted_name: extract.new_function_name,
                source_lines: extract.line_count,
            });
        }

        Ok(patterns)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditPattern {
    FunctionDefinition {
        name: String,
        signature_hash: String,
        language: Language,
    },
    ImportAddition {
        module: String,
        items: Vec<String>,
        language: Language,
    },
    ErrorHandling,
    LoggingAddition,
    TestAddition,
    DocumentationAddition,
    Rename {
        old_name: String,
        new_name: String,
    },
    ExtractFunction {
        extracted_name: String,
        source_lines: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EditType {
    Creation,
    Deletion,
    MinorModification,
    MajorModification,
    Rewrite,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
}
```
    </section>

    <section name="ShellScripts">
```bash
#!/bin/bash
# .claude/hooks/pre_tool_use_file.sh
# PreToolUse hook for Write|Edit tools - validates alignment before file writes
# Configured via settings.json: { "matcher": "Write|Edit", "timeout": 3000 }

set -e

# Read Claude Code's PreToolUse JSON from stdin
# Format: { "tool_name": "Write"|"Edit", "tool_input": { "file_path": "...", ... } }
INPUT=$(cat)

# Extract tool name and input
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name')

# Parse based on tool type
case "$TOOL_NAME" in
  "Write")
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path')
    NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.content')
    ORIGINAL_CONTENT=""
    ;;
  "Edit")
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path')
    NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string')
    ORIGINAL_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.old_string // empty')
    ;;
  *)
    # Unknown tool, allow by default
    exit 0
    ;;
esac

# Call MCP tool for validation (must complete within 3000ms per constitution)
RESPONSE=$(timeout 2.5 echo '{
  "jsonrpc": "2.0",
  "id": "pre-tool-use-file-'$(date +%s)'",
  "method": "hooks/pre_tool_use_file",
  "params": {
    "tool_name": "'"$TOOL_NAME"'",
    "file_path": "'"$FILE_PATH"'",
    "new_content": '"$(echo "$NEW_CONTENT" | jq -Rs .)"',
    "original_content": '"$(echo "$ORIGINAL_CONTENT" | jq -Rs .)"'
  }
}' | nc -U /tmp/contextgraph.sock 2>/dev/null || echo '{"result":{"status":"proceed"}}')

# Parse response
STATUS=$(echo "$RESPONSE" | jq -r '.result.status // "proceed"')
WARNINGS=$(echo "$RESPONSE" | jq -r '.result.warnings[]? // empty')

# Output warnings if any
if [ -n "$WARNINGS" ]; then
  echo "Alignment warnings:" >&2
  echo "$WARNINGS" >&2
fi

# Exit based on status
case "$STATUS" in
  "proceed")
    exit 0
    ;;
  "proceed_with_warning")
    exit 0
    ;;
  "block")
    echo "Write blocked due to low goal alignment" >&2
    exit 1
    ;;
  *)
    # Unknown status, allow by default
    exit 0
    ;;
esac
```

```bash
#!/bin/bash
# .claude/hooks/post_tool_use_file.sh
# PostToolUse hook for Write|Edit tools - stores edits and triggers pattern learning
# Configured via settings.json: { "matcher": "Write|Edit", "timeout": 3000 }

set -e

# Read Claude Code's PostToolUse JSON from stdin
# Format: { "tool_name": "Write"|"Edit", "tool_input": {...}, "tool_response": {...} }
INPUT=$(cat)

# Extract tool name and input
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name')
TOOL_SUCCESS=$(echo "$INPUT" | jq -r '.tool_response.success // true')

# Only process successful writes
if [ "$TOOL_SUCCESS" != "true" ]; then
  exit 0
fi

# Parse based on tool type
case "$TOOL_NAME" in
  "Write")
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path')
    NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.content')
    ORIGINAL_CONTENT=""
    ;;
  "Edit")
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path')
    NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string')
    ORIGINAL_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.old_string // empty')
    ;;
  *)
    # Unknown tool, skip
    exit 0
    ;;
esac

# Call MCP tool to store and learn (must complete within 3000ms per constitution)
RESPONSE=$(timeout 2.5 echo '{
  "jsonrpc": "2.0",
  "id": "post-tool-use-file-'$(date +%s)'",
  "method": "hooks/post_tool_use_file",
  "params": {
    "tool_name": "'"$TOOL_NAME"'",
    "file_path": "'"$FILE_PATH"'",
    "new_content": '"$(echo "$NEW_CONTENT" | jq -Rs .)"',
    "original_content": '"$(echo "$ORIGINAL_CONTENT" | jq -Rs .)"',
    "train_patterns": true,
    "compute_consciousness": true
  }
}' | nc -U /tmp/contextgraph.sock 2>/dev/null || echo '{"result":{}}')

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error.message // empty')
if [ -n "$ERROR" ]; then
  echo "Warning: PostToolUse hook failed: $ERROR" >&2
  # Don't block the operation, just log
fi

# Output memory ID and consciousness level if successful
MEMORY_ID=$(echo "$RESPONSE" | jq -r '.result.memory_id // empty')
CONSCIOUSNESS=$(echo "$RESPONSE" | jq -r '.result.consciousness.level // empty')

if [ -n "$MEMORY_ID" ]; then
  if [ -n "$CONSCIOUSNESS" ]; then
    echo "Edit stored as memory: $MEMORY_ID (consciousness: $CONSCIOUSNESS)" >&2
  else
    echo "Edit stored as memory: $MEMORY_ID" >&2
  fi
fi

exit 0
```
    </section>
  </pseudo_code>
</task_spec>
```
