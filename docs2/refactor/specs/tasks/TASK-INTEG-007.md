# TASK-INTEG-007: Bash Execution Hooks Implementation

```xml
<task_spec>
  <task_id>TASK-INTEG-007</task_id>
  <title>PreToolUse/PostToolUse for Bash (Claude Code Integration)</title>
  <status>pending</status>

  <objective>
    Implement PreToolUse and PostToolUse hook handlers for the Bash tool that provide
    safety checks before command execution and consciousness-aware learning from
    command outputs for the 13-embedder teleological array system.
  </objective>

  <rationale>
    Claude Code uses PreToolUse/PostToolUse events for ALL tool interactions, including Bash.
    These hooks enable:
    1. PreToolUse (Bash): Safety validation and context injection before commands
    2. PostToolUse (Bash): Learning from command outputs and error patterns
    3. Command pattern recognition: Build understanding of useful command sequences
    4. Risk mitigation: Warn or block potentially dangerous commands
    5. GWT Consciousness Integration: Adaptive safety based on consciousness level
    6. 13-Embedder Learning: Full teleological array storage for command knowledge

    These hooks protect the system while enabling learning from shell interactions,
    fully integrated with Claude Code's hook architecture.
  </rationale>

  <claude_code_hook_model>
    Claude Code passes JSON payloads to hooks with this structure:

    PreToolUse event (matcher: "Bash"):
    {
      "tool_name": "Bash",
      "tool_input": {
        "command": "git status",
        "description": "Check git status",
        "timeout": 120000,
        "run_in_background": false
      }
    }

    PostToolUse event (matcher: "Bash"):
    {
      "tool_name": "Bash",
      "tool_input": {
        "command": "git status",
        "description": "Check git status",
        "timeout": 120000
      },
      "tool_response": {
        "stdout": "On branch main...",
        "stderr": "",
        "exit_code": 0,
        "interrupted": false
      }
    }
  </claude_code_hook_model>

  <dependencies>
    <dependency type="required">TASK-INTEG-004</dependency>    <!-- Hook protocol -->
    <dependency type="required">TASK-CORE-003</dependency>     <!-- TeleologicalArray type -->
    <dependency type="required">TASK-LOGIC-006</dependency>    <!-- Trajectory tracking -->
    <dependency type="required">TASK-LOGIC-010</dependency>    <!-- Teleological drift (13 embedders) -->
  </dependencies>

  <input_context_files>
    <file purpose="hook_protocol">crates/context-graph-mcp/src/hooks/protocol.rs</file>
    <file purpose="trajectory_tracking">crates/context-graph-core/src/autonomous/trajectory.rs</file>
    <file purpose="embedder_pipeline">crates/context-graph-core/src/teleology/embedder.rs</file>
    <file purpose="store">crates/context-graph-storage/src/teleological/store.rs</file>
    <file purpose="consciousness">crates/context-graph-core/src/consciousness/gwt.rs</file>
  </input_context_files>

  <output_artifacts>
    <artifact type="source">crates/context-graph-mcp/src/hooks/bash_tool.rs</artifact>
    <artifact type="config">.claude/hooks/pre_tool_use_bash.sh</artifact>
    <artifact type="config">.claude/hooks/post_tool_use_bash.sh</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/bash_hooks_test.rs</artifact>
  </output_artifacts>

  <claude_code_settings>
    Add this configuration to .claude/settings.json:
    ```json
    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [".claude/hooks/pre_tool_use_bash.sh"],
            "timeout": 3000
          }
        ],
        "PostToolUse": [
          {
            "matcher": "Bash",
            "hooks": [".claude/hooks/post_tool_use_bash.sh"],
            "timeout": 3000
          }
        ]
      }
    }
    ```

    The matcher "Bash" will match tool_name exactly.
    Timeout is 3000ms per constitution requirements.
  </claude_code_settings>

  <definition_of_done>
    <criterion id="1">PreToolUse handler validates commands against safety rules</criterion>
    <criterion id="2">PostToolUse handler stores command outputs using 13-embedder array</criterion>
    <criterion id="3">Dangerous command patterns detected and warned/blocked</criterion>
    <criterion id="4">Command sequences tracked for pattern learning</criterion>
    <criterion id="5">Error outputs analyzed for troubleshooting context</criterion>
    <criterion id="6">Hook latency under 3000ms (constitution timeout budget)</criterion>
    <criterion id="7">Configurable safety levels (permissive, standard, strict)</criterion>
    <criterion id="8">Shell scripts parse Claude Code's tool_input/tool_response format</criterion>
    <criterion id="9">GWT consciousness integration with adaptive safety awareness</criterion>
    <criterion id="10">All 13 embedders used for command/output storage</criterion>
  </definition_of_done>

  <estimated_complexity>Medium-High</estimated_complexity>

  <pseudo_code>
    <section name="13-Embedder Integration">
```rust
// crates/context-graph-mcp/src/hooks/bash_tool.rs

use crate::hooks::protocol::{HookEvent, HookPayload, HookResponse, HookStatus};
use std::sync::Arc;
use regex::Regex;

/// 13-Embedder references for teleological array generation
/// Used in PostToolUse to create full embeddings of command knowledge
pub const EMBEDDER_MANIFEST: [&str; 13] = [
    "E1_Semantic",       // Semantic meaning of command/output
    "E2_Temporal",       // When command was run, sequence timing
    "E3_Causal",         // What caused this command, what it causes
    "E4_Counterfactual", // What would happen with different args
    "E5_Moral",          // Ethical implications (data deletion, etc.)
    "E6_Aesthetic",      // Code style, command formatting
    "E7_Teleological",   // Purpose/goal this command serves
    "E8_Contextual",     // Working directory, environment context
    "E9_Structural",     // Command syntax structure, piping
    "E10_Behavioral",    // Success/failure patterns, exit codes
    "E11_Emotional",     // User frustration indicators from errors
    "E12_Code",          // Code-specific patterns in output
    "E13_SPLADE",        // Sparse lexical for command search
];

/// Embedder weights for bash command processing
/// Emphasizes behavioral, causal, and contextual understanding
pub struct BashEmbedderWeights {
    pub e1_semantic: f32,       // 0.8 - command meaning
    pub e2_temporal: f32,       // 0.5 - timing context
    pub e3_causal: f32,         // 0.9 - cause-effect chains
    pub e4_counterfactual: f32, // 0.6 - alternative outcomes
    pub e5_moral: f32,          // 0.7 - safety implications
    pub e6_aesthetic: f32,      // 0.3 - formatting (less important)
    pub e7_teleological: f32,   // 0.9 - goal alignment
    pub e8_contextual: f32,     // 0.95 - environment critical
    pub e9_structural: f32,     // 0.7 - command structure
    pub e10_behavioral: f32,    // 0.95 - success/failure patterns
    pub e11_emotional: f32,     // 0.4 - error frustration
    pub e12_code: f32,          // 0.6 - code in output
    pub e13_splade: f32,        // 0.8 - searchability
}

impl Default for BashEmbedderWeights {
    fn default() -> Self {
        Self {
            e1_semantic: 0.8,
            e2_temporal: 0.5,
            e3_causal: 0.9,
            e4_counterfactual: 0.6,
            e5_moral: 0.7,
            e6_aesthetic: 0.3,
            e7_teleological: 0.9,
            e8_contextual: 0.95,
            e9_structural: 0.7,
            e10_behavioral: 0.95,
            e11_emotional: 0.4,
            e12_code: 0.6,
            e13_splade: 0.8,
        }
    }
}
```
    </section>

    <section name="GWT Consciousness Integration">
```rust
/// GWT-based consciousness context for command processing
/// Higher consciousness = more thorough safety checks and learning
pub struct BashConsciousnessContext {
    /// I(t) - Integration: command relevance to session goals
    pub integration_score: f32,
    /// R(t) - Reflection: depth of command understanding
    pub reflection_depth: f32,
    /// D(t) - Differentiation: command uniqueness/novelty
    pub differentiation_index: f32,
    /// C(t) = I(t) * R(t) * D(t) - Overall consciousness level
    pub consciousness_level: f32,
    /// Derived safety awareness level
    pub safety_awareness: SafetyAwareness,
}

/// Safety awareness levels derived from consciousness score
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyAwareness {
    /// C < 0.3: Standard safety checks, minimal logging
    Routine,
    /// 0.3 <= C < 0.6: Enhanced monitoring, basic pattern extraction
    Attentive,
    /// 0.6 <= C < 0.8: Full pattern analysis, detailed trajectory tracking
    Vigilant,
    /// C >= 0.8: All safety systems active, full 13-embedder storage
    FullAwareness,
}

impl BashConsciousnessContext {
    /// Calculate consciousness context for a command
    pub fn calculate(
        command: &str,
        session_goals: &[String],
        command_history: &[CommandRecord],
    ) -> Self {
        // I(t): How relevant is this command to session goals?
        let integration_score = Self::calculate_integration(command, session_goals);

        // R(t): How well do we understand this command?
        let reflection_depth = Self::calculate_reflection(command, command_history);

        // D(t): How novel/unique is this command?
        let differentiation_index = Self::calculate_differentiation(command, command_history);

        // C(t) = I(t) * R(t) * D(t)
        let consciousness_level = integration_score * reflection_depth * differentiation_index;

        let safety_awareness = if consciousness_level >= 0.8 {
            SafetyAwareness::FullAwareness
        } else if consciousness_level >= 0.6 {
            SafetyAwareness::Vigilant
        } else if consciousness_level >= 0.3 {
            SafetyAwareness::Attentive
        } else {
            SafetyAwareness::Routine
        };

        Self {
            integration_score,
            reflection_depth,
            differentiation_index,
            consciousness_level,
            safety_awareness,
        }
    }

    fn calculate_integration(command: &str, session_goals: &[String]) -> f32 {
        // Calculate semantic similarity between command and goals
        // Higher if command directly serves stated goals
        let mut max_relevance = 0.3_f32; // Base relevance

        for goal in session_goals {
            let goal_lower = goal.to_lowercase();
            let cmd_lower = command.to_lowercase();

            // Simple keyword matching (real impl uses embeddings)
            if cmd_lower.contains("test") && goal_lower.contains("test") {
                max_relevance = max_relevance.max(0.9);
            }
            if cmd_lower.contains("build") && goal_lower.contains("build") {
                max_relevance = max_relevance.max(0.85);
            }
            if cmd_lower.contains("git") && goal_lower.contains("commit") {
                max_relevance = max_relevance.max(0.8);
            }
        }

        max_relevance
    }

    fn calculate_reflection(command: &str, history: &[CommandRecord]) -> f32 {
        // Higher if we have seen similar commands before
        let similar_count = history.iter()
            .filter(|r| Self::commands_similar(&r.command, command))
            .count();

        match similar_count {
            0 => 0.4,      // Novel command - lower reflection
            1..=2 => 0.6,  // Some familiarity
            3..=5 => 0.8,  // Well understood
            _ => 0.95,     // Very familiar pattern
        }
    }

    fn calculate_differentiation(command: &str, history: &[CommandRecord]) -> f32 {
        // Higher if command is unique/different from recent commands
        if history.is_empty() {
            return 0.9; // First command is novel
        }

        let recent = history.iter().rev().take(5);
        let similar_count = recent
            .filter(|r| Self::commands_similar(&r.command, command))
            .count();

        match similar_count {
            0 => 0.95,     // Completely different
            1 => 0.7,      // Somewhat novel
            2..=3 => 0.5,  // Repetitive
            _ => 0.3,      // Very repetitive
        }
    }

    fn commands_similar(cmd1: &str, cmd2: &str) -> bool {
        // Extract base command (first word)
        let base1 = cmd1.split_whitespace().next().unwrap_or("");
        let base2 = cmd2.split_whitespace().next().unwrap_or("");
        base1 == base2
    }
}

impl SafetyAwareness {
    /// Get learning settings based on awareness level
    pub fn learning_config(&self) -> ConsciousnessLearningConfig {
        match self {
            SafetyAwareness::Routine => ConsciousnessLearningConfig {
                store_output: false,
                extract_patterns: false,
                track_trajectory: false,
                embedder_count: 0,
                log_level: "minimal",
            },
            SafetyAwareness::Attentive => ConsciousnessLearningConfig {
                store_output: true,
                extract_patterns: false,
                track_trajectory: true,
                embedder_count: 5, // Core embedders only
                log_level: "basic",
            },
            SafetyAwareness::Vigilant => ConsciousnessLearningConfig {
                store_output: true,
                extract_patterns: true,
                track_trajectory: true,
                embedder_count: 9, // Most embedders
                log_level: "detailed",
            },
            SafetyAwareness::FullAwareness => ConsciousnessLearningConfig {
                store_output: true,
                extract_patterns: true,
                track_trajectory: true,
                embedder_count: 13, // All 13 embedders
                log_level: "comprehensive",
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsciousnessLearningConfig {
    pub store_output: bool,
    pub extract_patterns: bool,
    pub track_trajectory: bool,
    pub embedder_count: usize,
    pub log_level: &'static str,
}
```
    </section>

    <section name="PreToolUseHandler for Bash">
```rust
/// Claude Code tool input structure for Bash tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeBashInput {
    /// Command to execute
    pub command: String,
    /// Description of what the command does
    pub description: Option<String>,
    /// Timeout in milliseconds
    pub timeout: Option<u64>,
    /// Whether to run in background
    pub run_in_background: Option<bool>,
}

/// PreToolUse payload from Claude Code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreToolUsePayload {
    /// Tool name (will be "Bash")
    pub tool_name: String,
    /// Tool input
    pub tool_input: ClaudeCodeBashInput,
}

/// Pre-tool-use hook handler for Bash safety checks
pub struct PreToolUseBashHandler {
    /// Safety rule engine
    safety_engine: Arc<CommandSafetyEngine>,
    /// Teleological store for context
    store: Arc<TeleologicalStore>,
    /// Consciousness evaluator
    consciousness: Arc<ConsciousnessEvaluator>,
    /// Configuration
    config: PreToolUseBashConfig,
}

#[derive(Debug, Clone)]
pub struct PreToolUseBashConfig {
    /// Safety level (permissive, standard, strict)
    pub safety_level: SafetyLevel,
    /// Maximum command length to analyze
    pub max_command_length: usize,
    /// Whether to inject context suggestions
    pub inject_context: bool,
    /// Maximum latency for safety checks (ms) - must fit in 3000ms budget
    pub max_latency_ms: u64,
    /// Custom blocked patterns
    pub blocked_patterns: Vec<String>,
    /// Custom allowed patterns (override blocks)
    pub allowed_patterns: Vec<String>,
    /// Enable consciousness-aware safety
    pub consciousness_aware: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    /// Minimal checks, mostly informational
    Permissive,
    /// Standard checks, warn on risky commands
    Standard,
    /// Strict checks, block risky commands
    Strict,
}

impl Default for PreToolUseBashConfig {
    fn default() -> Self {
        Self {
            safety_level: SafetyLevel::Standard,
            max_command_length: 10_000,
            inject_context: true,
            // Constitution requires 3000ms max, budget 2500ms for processing
            max_latency_ms: 2500,
            blocked_patterns: vec![],
            allowed_patterns: vec![],
            consciousness_aware: true,
        }
    }
}

/// Command analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandAnalysis {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Detected command type
    pub command_type: CommandType,
    /// Safety warnings
    pub warnings: Vec<SafetyWarning>,
    /// Suggested alternatives (if risky)
    pub alternatives: Vec<String>,
    /// Related context from previous commands
    pub related_context: Vec<CommandContext>,
    /// Consciousness context
    pub consciousness: Option<BashConsciousnessContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    FileSystem,
    Git,
    Package,
    Build,
    Test,
    Network,
    System,
    Docker,
    Database,
    Custom,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyWarning {
    pub level: RiskLevel,
    pub message: String,
    pub pattern_matched: Option<String>,
    pub mitigation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandContext {
    pub previous_command: String,
    pub output_summary: String,
    pub timestamp: DateTime<Utc>,
    pub relevance_score: f32,
}

impl PreToolUseBashHandler {
    pub fn new(
        safety_engine: Arc<CommandSafetyEngine>,
        store: Arc<TeleologicalStore>,
        consciousness: Arc<ConsciousnessEvaluator>,
        config: PreToolUseBashConfig,
    ) -> Self {
        Self {
            safety_engine,
            store,
            consciousness,
            config,
        }
    }

    /// Handle PreToolUse event for Bash tool
    pub async fn handle(&self, payload: PreToolUsePayload) -> Result<HookResponse, HookError> {
        let start = Instant::now();
        let command = &payload.tool_input.command;

        // 1. Quick command length check
        if command.len() > self.config.max_command_length {
            return Ok(HookResponse {
                status: HookStatus::ProceedWithWarning,
                message: Some("Command exceeds maximum analyzable length".to_string()),
                warnings: vec!["Command too long for full safety analysis".to_string()],
                injected_context: None,
                metrics: HookMetrics::quick(start.elapsed().as_millis() as u64),
            });
        }

        // 2. Calculate consciousness context if enabled
        let consciousness_ctx = if self.config.consciousness_aware {
            let session_goals = self.get_session_goals().await?;
            let command_history = self.get_recent_commands(10).await?;
            Some(BashConsciousnessContext::calculate(command, &session_goals, &command_history))
        } else {
            None
        };

        // 3. Adjust safety level based on consciousness
        let effective_safety = if let Some(ref ctx) = consciousness_ctx {
            match ctx.safety_awareness {
                SafetyAwareness::FullAwareness => SafetyLevel::Strict,
                SafetyAwareness::Vigilant => self.config.safety_level.max(SafetyLevel::Standard),
                _ => self.config.safety_level,
            }
        } else {
            self.config.safety_level
        };

        // 4. Analyze command safety
        let analysis = self.safety_engine.analyze(command, effective_safety, &self.config)?;

        // 5. Determine action based on safety level and risk
        let (status, should_block) = self.determine_action(effective_safety, &analysis);

        // 6. Fetch related context if enabled and time permits
        let budget_remaining = self.config.max_latency_ms.saturating_sub(
            start.elapsed().as_millis() as u64
        );
        let related_context = if self.config.inject_context && budget_remaining > 500 {
            self.fetch_related_context(command).await.unwrap_or_default()
        } else {
            vec![]
        };

        // 7. Build response
        let warnings: Vec<String> = analysis.warnings
            .iter()
            .map(|w| format!("[{}] {}", w.level.display(), w.message))
            .collect();

        let mut response = HookResponse {
            status,
            message: self.build_status_message(&analysis, should_block),
            warnings,
            injected_context: Some(serde_json::json!({
                "command_type": analysis.command_type,
                "risk_level": analysis.risk_level,
                "alternatives": analysis.alternatives,
                "related_context": related_context,
                "consciousness": consciousness_ctx,
                "description": payload.tool_input.description,
            })),
            metrics: HookMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embedders_used: 0,
                goals_checked: consciousness_ctx.map(|c| c.integration_score as u32).unwrap_or(0),
            },
        };

        // 8. Add alternatives if command is risky
        if !analysis.alternatives.is_empty() && analysis.risk_level >= RiskLevel::Medium {
            response.message = Some(format!(
                "{}. Consider: {}",
                response.message.unwrap_or_default(),
                analysis.alternatives.join(" or ")
            ));
        }

        Ok(response)
    }

    fn determine_action(&self, safety_level: SafetyLevel, analysis: &CommandAnalysis) -> (HookStatus, bool) {
        match (safety_level, analysis.risk_level) {
            // Permissive: Only block critical
            (SafetyLevel::Permissive, RiskLevel::Critical) => (HookStatus::Block, true),
            (SafetyLevel::Permissive, _) => (HookStatus::Proceed, false),

            // Standard: Block critical, warn high
            (SafetyLevel::Standard, RiskLevel::Critical) => (HookStatus::Block, true),
            (SafetyLevel::Standard, RiskLevel::High) => (HookStatus::ProceedWithWarning, false),
            (SafetyLevel::Standard, RiskLevel::Medium) => (HookStatus::ProceedWithWarning, false),
            (SafetyLevel::Standard, _) => (HookStatus::Proceed, false),

            // Strict: Block high+, warn medium
            (SafetyLevel::Strict, RiskLevel::Critical) => (HookStatus::Block, true),
            (SafetyLevel::Strict, RiskLevel::High) => (HookStatus::Block, true),
            (SafetyLevel::Strict, RiskLevel::Medium) => (HookStatus::ProceedWithWarning, false),
            (SafetyLevel::Strict, RiskLevel::Low) => (HookStatus::ProceedWithWarning, false),
            (SafetyLevel::Strict, RiskLevel::Safe) => (HookStatus::Proceed, false),
        }
    }

    fn build_status_message(&self, analysis: &CommandAnalysis, blocked: bool) -> Option<String> {
        if blocked {
            Some(format!(
                "Command blocked due to {} risk: {}",
                analysis.risk_level.display(),
                analysis.warnings.first()
                    .map(|w| w.message.as_str())
                    .unwrap_or("Safety policy violation")
            ))
        } else if analysis.risk_level >= RiskLevel::Medium {
            Some(format!(
                "{} risk detected for {:?} command",
                analysis.risk_level.display(),
                analysis.command_type
            ))
        } else {
            None
        }
    }

    async fn fetch_related_context(&self, command: &str) -> Result<Vec<CommandContext>, HookError> {
        let results = self.store.search_commands(command, 3).await?;
        Ok(results
            .into_iter()
            .map(|r| CommandContext {
                previous_command: r.command,
                output_summary: truncate_content(&r.output, 100),
                timestamp: r.timestamp,
                relevance_score: r.similarity,
            })
            .collect())
    }

    async fn get_session_goals(&self) -> Result<Vec<String>, HookError> {
        // Fetch from session state
        self.store.get_session_goals().await.map_err(HookError::StorageError)
    }

    async fn get_recent_commands(&self, limit: usize) -> Result<Vec<CommandRecord>, HookError> {
        self.store.get_recent_commands(limit).await.map_err(HookError::StorageError)
    }
}

impl SafetyLevel {
    fn max(self, other: SafetyLevel) -> SafetyLevel {
        if self as u8 > other as u8 { self } else { other }
    }
}

impl RiskLevel {
    pub fn display(&self) -> &'static str {
        match self {
            RiskLevel::Safe => "Safe",
            RiskLevel::Low => "Low",
            RiskLevel::Medium => "Medium",
            RiskLevel::High => "High",
            RiskLevel::Critical => "CRITICAL",
        }
    }
}
```
    </section>

    <section name="CommandSafetyEngine">
```rust
/// Command safety analysis engine
pub struct CommandSafetyEngine {
    /// Built-in dangerous patterns
    dangerous_patterns: Vec<DangerousPattern>,
    /// Command type classifiers
    classifiers: Vec<CommandClassifier>,
}

#[derive(Debug, Clone)]
pub struct DangerousPattern {
    pub pattern: Regex,
    pub risk_level: RiskLevel,
    pub description: String,
    pub mitigation: Option<String>,
    pub alternatives: Vec<String>,
}

impl CommandSafetyEngine {
    pub fn new() -> Self {
        let dangerous_patterns = vec![
            // Critical - data destruction
            DangerousPattern {
                pattern: Regex::new(r"rm\s+(-rf?|--recursive)\s+(/|~|\$HOME|\*|\.\./)").unwrap(),
                risk_level: RiskLevel::Critical,
                description: "Recursive deletion of critical directories".to_string(),
                mitigation: Some("Use trash-cli or move to backup first".to_string()),
                alternatives: vec!["trash-put".to_string(), "mv to backup dir".to_string()],
            },
            DangerousPattern {
                pattern: Regex::new(r">\s*/dev/sd[a-z]").unwrap(),
                risk_level: RiskLevel::Critical,
                description: "Direct write to block device".to_string(),
                mitigation: None,
                alternatives: vec![],
            },
            DangerousPattern {
                pattern: Regex::new(r"mkfs\.|dd\s+if=.+of=/dev").unwrap(),
                risk_level: RiskLevel::Critical,
                description: "Filesystem formatting or raw disk write".to_string(),
                mitigation: Some("Verify device path carefully".to_string()),
                alternatives: vec![],
            },

            // High - system modification
            DangerousPattern {
                pattern: Regex::new(r"chmod\s+(-R\s+)?[0-7]*777").unwrap(),
                risk_level: RiskLevel::High,
                description: "Setting world-writable permissions".to_string(),
                mitigation: Some("Use more restrictive permissions like 755 or 644".to_string()),
                alternatives: vec!["chmod 755".to_string(), "chmod 644".to_string()],
            },
            DangerousPattern {
                pattern: Regex::new(r"curl\s+.*\|\s*(sudo\s+)?(bash|sh)").unwrap(),
                risk_level: RiskLevel::High,
                description: "Piping remote script to shell".to_string(),
                mitigation: Some("Download and review script before execution".to_string()),
                alternatives: vec!["curl -O then review".to_string()],
            },
            DangerousPattern {
                pattern: Regex::new(r"sudo\s+.*rm|rm\s+.*sudo").unwrap(),
                risk_level: RiskLevel::High,
                description: "Sudo with file deletion".to_string(),
                mitigation: Some("Verify paths before sudo rm".to_string()),
                alternatives: vec![],
            },

            // Medium - potentially risky
            DangerousPattern {
                pattern: Regex::new(r"git\s+(push|reset|rebase)\s+(-f|--force)").unwrap(),
                risk_level: RiskLevel::Medium,
                description: "Force git operation".to_string(),
                mitigation: Some("Use --force-with-lease for safer push".to_string()),
                alternatives: vec!["git push --force-with-lease".to_string()],
            },
            DangerousPattern {
                pattern: Regex::new(r"npm\s+publish|cargo\s+publish").unwrap(),
                risk_level: RiskLevel::Medium,
                description: "Publishing package".to_string(),
                mitigation: Some("Verify version and content before publish".to_string()),
                alternatives: vec!["--dry-run first".to_string()],
            },
            DangerousPattern {
                pattern: Regex::new(r"docker\s+system\s+prune").unwrap(),
                risk_level: RiskLevel::Medium,
                description: "Docker system cleanup".to_string(),
                mitigation: Some("May remove needed images/containers".to_string()),
                alternatives: vec!["docker image prune".to_string()],
            },

            // Low - informational
            DangerousPattern {
                pattern: Regex::new(r"sudo\s+").unwrap(),
                risk_level: RiskLevel::Low,
                description: "Elevated privileges".to_string(),
                mitigation: None,
                alternatives: vec![],
            },
        ];

        Self {
            dangerous_patterns,
            classifiers: Self::build_classifiers(),
        }
    }

    fn build_classifiers() -> Vec<CommandClassifier> {
        vec![
            CommandClassifier {
                pattern: Regex::new(r"^(ls|cat|head|tail|less|more|find|grep|wc)").unwrap(),
                command_type: CommandType::FileSystem,
            },
            CommandClassifier {
                pattern: Regex::new(r"^git\s").unwrap(),
                command_type: CommandType::Git,
            },
            CommandClassifier {
                pattern: Regex::new(r"^(npm|yarn|pnpm|cargo|pip|poetry)").unwrap(),
                command_type: CommandType::Package,
            },
            CommandClassifier {
                pattern: Regex::new(r"^(make|cmake|cargo\s+build|npm\s+run\s+build)").unwrap(),
                command_type: CommandType::Build,
            },
            CommandClassifier {
                pattern: Regex::new(r"(test|spec|pytest|jest|cargo\s+test)").unwrap(),
                command_type: CommandType::Test,
            },
            CommandClassifier {
                pattern: Regex::new(r"^(curl|wget|ssh|scp|rsync)").unwrap(),
                command_type: CommandType::Network,
            },
            CommandClassifier {
                pattern: Regex::new(r"^(systemctl|service|launchctl)").unwrap(),
                command_type: CommandType::System,
            },
            CommandClassifier {
                pattern: Regex::new(r"^docker").unwrap(),
                command_type: CommandType::Docker,
            },
            CommandClassifier {
                pattern: Regex::new(r"(psql|mysql|mongo|redis-cli|sqlite)").unwrap(),
                command_type: CommandType::Database,
            },
        ]
    }

    /// Analyze command for safety
    pub fn analyze(
        &self,
        command: &str,
        safety_level: SafetyLevel,
        config: &PreToolUseBashConfig,
    ) -> Result<CommandAnalysis, HookError> {
        let mut warnings = Vec::new();
        let mut max_risk = RiskLevel::Safe;
        let mut alternatives = Vec::new();

        // Check against dangerous patterns
        for pattern in &self.dangerous_patterns {
            if pattern.pattern.is_match(command) {
                // Check if allowed by config override
                if config.allowed_patterns.iter().any(|p| command.contains(p)) {
                    continue;
                }

                warnings.push(SafetyWarning {
                    level: pattern.risk_level,
                    message: pattern.description.clone(),
                    pattern_matched: Some(pattern.pattern.as_str().to_string()),
                    mitigation: pattern.mitigation.clone(),
                });

                alternatives.extend(pattern.alternatives.clone());

                if pattern.risk_level > max_risk {
                    max_risk = pattern.risk_level;
                }
            }
        }

        // Check custom blocked patterns
        for blocked in &config.blocked_patterns {
            if command.contains(blocked) {
                warnings.push(SafetyWarning {
                    level: RiskLevel::High,
                    message: format!("Matches custom blocked pattern: {}", blocked),
                    pattern_matched: Some(blocked.clone()),
                    mitigation: None,
                });
                if max_risk < RiskLevel::High {
                    max_risk = RiskLevel::High;
                }
            }
        }

        // Classify command type
        let command_type = self.classify_command(command);

        Ok(CommandAnalysis {
            risk_level: max_risk,
            command_type,
            warnings,
            alternatives,
            related_context: vec![],
            consciousness: None,
        })
    }

    fn classify_command(&self, command: &str) -> CommandType {
        for classifier in &self.classifiers {
            if classifier.pattern.is_match(command) {
                return classifier.command_type.clone();
            }
        }
        CommandType::Unknown
    }
}

#[derive(Debug, Clone)]
struct CommandClassifier {
    pattern: Regex,
    command_type: CommandType,
}
```
    </section>

    <section name="PostToolUseHandler for Bash">
```rust
/// PostToolUse payload from Claude Code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostToolUsePayload {
    /// Tool name (will be "Bash")
    pub tool_name: String,
    /// Original tool input
    pub tool_input: ClaudeCodeBashInput,
    /// Tool response/output
    pub tool_response: ClaudeCodeBashResponse,
}

/// Claude Code Bash tool response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeBashResponse {
    /// Standard output
    pub stdout: Option<String>,
    /// Standard error
    pub stderr: Option<String>,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Whether command was interrupted
    pub interrupted: Option<bool>,
    /// Execution duration (not always provided)
    pub duration_ms: Option<u64>,
}

/// Post-tool-use hook handler for Bash learning
pub struct PostToolUseBashHandler {
    /// Teleological store
    store: Arc<TeleologicalStore>,
    /// Embedding pipeline (13 embedders)
    embedder_pipeline: Arc<EmbedderPipeline>,
    /// Trajectory tracker
    trajectory_tracker: Arc<TrajectoryTracker>,
    /// Command pattern learner
    pattern_learner: Arc<CommandPatternLearner>,
    /// Consciousness evaluator
    consciousness: Arc<ConsciousnessEvaluator>,
    /// Configuration
    config: PostToolUseBashConfig,
}

#[derive(Debug, Clone)]
pub struct PostToolUseBashConfig {
    /// Store successful command outputs
    pub store_success_outputs: bool,
    /// Store error outputs (for troubleshooting learning)
    pub store_error_outputs: bool,
    /// Maximum output length to store
    pub max_output_length: usize,
    /// Minimum output length to store
    pub min_output_length: usize,
    /// Learn command sequences
    pub learn_sequences: bool,
    /// Error pattern extraction
    pub extract_error_patterns: bool,
    /// Enable consciousness-aware learning
    pub consciousness_aware: bool,
    /// Maximum latency budget (must fit in 3000ms)
    pub max_latency_ms: u64,
}

impl Default for PostToolUseBashConfig {
    fn default() -> Self {
        Self {
            store_success_outputs: true,
            store_error_outputs: true,
            max_output_length: 50_000,
            min_output_length: 10,
            learn_sequences: true,
            extract_error_patterns: true,
            consciousness_aware: true,
            // Constitution requires 3000ms max, budget 2800ms for processing
            max_latency_ms: 2800,
        }
    }
}

/// Command execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRecord {
    /// Unique ID
    pub record_id: Uuid,
    /// Command executed
    pub command: String,
    /// Exit code
    pub exit_code: i32,
    /// Output (truncated)
    pub output: String,
    /// Whether command succeeded
    pub success: bool,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Session ID
    pub session_id: String,
    /// Working directory (from description or inferred)
    pub working_dir: Option<String>,
    /// Duration
    pub duration_ms: u64,
    /// Extracted patterns
    pub patterns: Vec<String>,
    /// Consciousness level when recorded
    pub consciousness_level: f32,
    /// Number of embedders used
    pub embedders_used: usize,
}

impl PostToolUseBashHandler {
    pub fn new(
        store: Arc<TeleologicalStore>,
        embedder_pipeline: Arc<EmbedderPipeline>,
        trajectory_tracker: Arc<TrajectoryTracker>,
        pattern_learner: Arc<CommandPatternLearner>,
        consciousness: Arc<ConsciousnessEvaluator>,
        config: PostToolUseBashConfig,
    ) -> Self {
        Self {
            store,
            embedder_pipeline,
            trajectory_tracker,
            pattern_learner,
            consciousness,
            config,
        }
    }

    /// Handle PostToolUse event for Bash tool
    pub async fn handle(&self, payload: PostToolUsePayload) -> Result<HookResponse, HookError> {
        let start = Instant::now();

        let command = &payload.tool_input.command;
        let exit_code = payload.tool_response.exit_code.unwrap_or(0);
        let stdout = payload.tool_response.stdout.as_deref().unwrap_or("");
        let stderr = payload.tool_response.stderr.as_deref().unwrap_or("");
        let duration_ms = payload.tool_response.duration_ms.unwrap_or(0);
        let success = exit_code == 0;

        // 1. Calculate consciousness context
        let consciousness_ctx = if self.config.consciousness_aware {
            let session_goals = self.get_session_goals().await?;
            let command_history = self.get_recent_commands(10).await?;
            BashConsciousnessContext::calculate(command, &session_goals, &command_history)
        } else {
            BashConsciousnessContext {
                integration_score: 0.5,
                reflection_depth: 0.5,
                differentiation_index: 0.5,
                consciousness_level: 0.125,
                safety_awareness: SafetyAwareness::Routine,
            }
        };

        // 2. Get learning config based on consciousness level
        let learning_config = consciousness_ctx.safety_awareness.learning_config();

        // 3. Determine if we should store this output
        let should_store = self.should_store(command, stdout, stderr, success, &learning_config);

        if !should_store {
            return Ok(HookResponse {
                status: HookStatus::Success,
                message: Some(format!(
                    "Output not stored (consciousness: {:.2}, awareness: {:?})",
                    consciousness_ctx.consciousness_level,
                    consciousness_ctx.safety_awareness
                )),
                warnings: vec![],
                injected_context: None,
                metrics: HookMetrics::quick(start.elapsed().as_millis() as u64),
            });
        }

        // 4. Prepare output content
        let output = if success {
            stdout.to_string()
        } else {
            format!("{}\n{}", stderr, stdout)
        };

        let truncated_output = truncate_content(&output, self.config.max_output_length);

        // 5. Generate teleological array (using consciousness-aware embedder count)
        let content_for_embedding = format!(
            "Command: {}\nDescription: {}\nOutput: {}",
            command,
            payload.tool_input.description.as_deref().unwrap_or(""),
            truncated_output
        );

        let teleological_array = if learning_config.embedder_count > 0 {
            // Check time budget
            let elapsed = start.elapsed().as_millis() as u64;
            if elapsed < self.config.max_latency_ms - 1000 {
                self.embedder_pipeline
                    .generate_partial(&content_for_embedding, learning_config.embedder_count)
                    .await
                    .map_err(HookError::EmbeddingFailed)?
            } else {
                // Time budget exceeded, skip embeddings
                TeleologicalArray::empty()
            }
        } else {
            TeleologicalArray::empty()
        };

        // 6. Extract error patterns if configured and failed
        let patterns = if learning_config.extract_patterns && !success {
            self.pattern_learner.extract_error_patterns(stderr).await?
        } else {
            vec![]
        };

        // 7. Create command record
        let record = CommandRecord {
            record_id: Uuid::new_v4(),
            command: command.clone(),
            exit_code,
            output: truncated_output.clone(),
            success,
            timestamp: Utc::now(),
            session_id: get_current_session_id(),
            working_dir: None, // Could be extracted from description
            duration_ms,
            patterns: patterns.clone(),
            consciousness_level: consciousness_ctx.consciousness_level,
            embedders_used: learning_config.embedder_count,
        };

        // 8. Store in teleological store if we have embeddings
        let memory_id = if !teleological_array.is_empty() {
            Some(self.store.inject(InjectionRequest {
                content: content_for_embedding,
                teleological_array,
                memory_type: if success { MemoryType::CommandOutput } else { MemoryType::ErrorLog },
                namespace: "commands".to_string(),
                metadata: serde_json::to_value(&record)?,
            }).await.map_err(HookError::StorageError)?)
        } else {
            None
        };

        // 9. Record in trajectory if enabled
        if learning_config.track_trajectory {
            let trajectory_step = TrajectoryStep {
                action: TrajectoryAction::BashExec {
                    command: command.clone(),
                    exit_code,
                    memory_id,
                },
                timestamp: Utc::now(),
                context_size: output.len(),
                outcome: Some(if success {
                    TrajectoryOutcome::Success
                } else {
                    TrajectoryOutcome::Failure {
                        error: stderr.lines().next().unwrap_or("Unknown error").to_string(),
                    }
                }),
            };
            self.trajectory_tracker.record_step(trajectory_step).await?;
        }

        // 10. Learn command sequences if enabled
        if self.config.learn_sequences {
            self.pattern_learner.record_command(command, success, duration_ms).await?;
        }

        // 11. Build response
        let mut response_context = serde_json::json!({
            "memory_id": memory_id,
            "record_id": record.record_id,
            "success": success,
            "duration_ms": duration_ms,
            "consciousness_level": consciousness_ctx.consciousness_level,
            "safety_awareness": format!("{:?}", consciousness_ctx.safety_awareness),
            "embedders_used": learning_config.embedder_count,
        });

        // Add error analysis if failed
        if !success && !patterns.is_empty() {
            response_context["error_patterns"] = serde_json::json!(patterns);

            // Only fetch troubleshooting if we have time budget
            if start.elapsed().as_millis() < (self.config.max_latency_ms - 500) as u128 {
                response_context["troubleshooting"] = serde_json::json!(
                    self.pattern_learner.suggest_fixes(&patterns).await?
                );
            }
        }

        Ok(HookResponse {
            status: HookStatus::Success,
            message: Some(format!(
                "Command {} stored (C={:.2}, embedders={}/13)",
                if success { "output" } else { "error" },
                consciousness_ctx.consciousness_level,
                learning_config.embedder_count
            )),
            warnings: if !success {
                vec![format!("Command failed with exit code {}", exit_code)]
            } else {
                vec![]
            },
            injected_context: Some(response_context),
            metrics: HookMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embedders_used: learning_config.embedder_count as u32,
                goals_checked: 0,
            },
        })
    }

    fn should_store(
        &self,
        command: &str,
        stdout: &str,
        stderr: &str,
        success: bool,
        learning_config: &ConsciousnessLearningConfig,
    ) -> bool {
        // Check learning config first
        if !learning_config.store_output {
            return false;
        }

        // Check config settings
        if success && !self.config.store_success_outputs {
            return false;
        }
        if !success && !self.config.store_error_outputs {
            return false;
        }

        // Check output length
        let output_len = if success {
            stdout.len()
        } else {
            stderr.len() + stdout.len()
        };

        if output_len < self.config.min_output_length {
            return false;
        }

        // Filter out noisy commands
        let noisy_commands = ["ls", "pwd", "echo", "cd", "clear"];
        let cmd_start = command.split_whitespace().next().unwrap_or("");
        if noisy_commands.contains(&cmd_start) && success {
            return false;
        }

        true
    }

    async fn get_session_goals(&self) -> Result<Vec<String>, HookError> {
        self.store.get_session_goals().await.map_err(HookError::StorageError)
    }

    async fn get_recent_commands(&self, limit: usize) -> Result<Vec<CommandRecord>, HookError> {
        self.store.get_recent_commands(limit).await.map_err(HookError::StorageError)
    }
}
```
    </section>

    <section name="CommandPatternLearner">
```rust
/// Command pattern learner for sequence detection and error analysis
pub struct CommandPatternLearner {
    /// Store for pattern persistence
    store: Arc<TeleologicalStore>,
    /// Recent command buffer
    command_buffer: Arc<RwLock<VecDeque<CommandBufferEntry>>>,
    /// Error pattern database
    error_patterns: Arc<ErrorPatternDatabase>,
}

#[derive(Debug, Clone)]
struct CommandBufferEntry {
    command: String,
    success: bool,
    duration_ms: u64,
    timestamp: DateTime<Utc>,
}

impl CommandPatternLearner {
    /// Extract error patterns from stderr
    pub async fn extract_error_patterns(&self, stderr: &str) -> Result<Vec<String>, HookError> {
        let mut patterns = Vec::new();

        // Common error patterns
        let error_regexes = [
            (r"error\[E\d+\]", "Rust compiler error"),
            (r"npm ERR!", "npm error"),
            (r"TypeError:|ReferenceError:|SyntaxError:", "JavaScript error"),
            (r"ModuleNotFoundError:|ImportError:", "Python import error"),
            (r"Permission denied", "Permission error"),
            (r"No such file or directory", "File not found"),
            (r"command not found", "Missing command"),
            (r"Connection refused", "Network error"),
            (r"ENOENT:", "Node.js file error"),
            (r"EACCES:", "Node.js permission error"),
            (r"cargo:.*error", "Cargo build error"),
        ];

        for (regex, description) in &error_regexes {
            if Regex::new(regex).unwrap().is_match(stderr) {
                patterns.push(description.to_string());
            }
        }

        Ok(patterns)
    }

    /// Record command for sequence learning
    pub async fn record_command(
        &self,
        command: &str,
        success: bool,
        duration_ms: u64,
    ) -> Result<(), HookError> {
        let entry = CommandBufferEntry {
            command: command.to_string(),
            success,
            duration_ms,
            timestamp: Utc::now(),
        };

        let mut buffer = self.command_buffer.write().await;
        buffer.push_back(entry);

        // Keep last 100 commands
        while buffer.len() > 100 {
            buffer.pop_front();
        }

        // Check for repeated sequences
        if buffer.len() >= 3 {
            self.detect_sequences(&buffer).await?;
        }

        Ok(())
    }

    /// Suggest fixes based on error patterns
    pub async fn suggest_fixes(&self, patterns: &[String]) -> Result<Vec<String>, HookError> {
        let mut suggestions = Vec::new();

        for pattern in patterns {
            if let Some(fixes) = self.error_patterns.get_fixes(pattern).await {
                suggestions.extend(fixes);
            }
        }

        Ok(suggestions)
    }

    async fn detect_sequences(&self, buffer: &VecDeque<CommandBufferEntry>) -> Result<(), HookError> {
        // Look for repeating patterns
        let commands: Vec<_> = buffer.iter()
            .map(|e| e.command.split_whitespace().next().unwrap_or(""))
            .collect();

        // Find 2-3 command patterns that repeat
        // Store as useful sequences for suggesting "you usually run X after Y"
        Ok(())
    }
}
```
    </section>

    <section name="ShellScripts">
```bash
#!/bin/bash
# .claude/hooks/pre_tool_use_bash.sh
# PreToolUse hook for Bash tool - safety checks before command execution
# Configured in .claude/settings.json with matcher: "Bash"
# Timeout: 3000ms (per constitution)

set -e

# Read input from stdin (JSON payload from Claude Code)
# Format: { "tool_name": "Bash", "tool_input": { "command": "...", "description": "...", "timeout": ... } }
INPUT=$(cat)

# Extract tool_input fields (Claude Code format)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
DESCRIPTION=$(echo "$INPUT" | jq -r '.tool_input.description // empty')
TIMEOUT=$(echo "$INPUT" | jq -r '.tool_input.timeout // 120000')
BACKGROUND=$(echo "$INPUT" | jq -r '.tool_input.run_in_background // false')

# Validate we have a command
if [ -z "$COMMAND" ]; then
  echo '{"error": "No command provided in tool_input"}' >&2
  exit 1
fi

# Call MCP tool for safety check
RESPONSE=$(echo '{
  "jsonrpc": "2.0",
  "id": "pre-tool-use-bash-'$(date +%s%N)'",
  "method": "hooks/pre_tool_use_bash",
  "params": {
    "tool_name": "Bash",
    "tool_input": {
      "command": '"$(echo "$COMMAND" | jq -Rs .)"',
      "description": '"$(echo "$DESCRIPTION" | jq -Rs .)"',
      "timeout": '"$TIMEOUT"',
      "run_in_background": '"$BACKGROUND"'
    }
  }
}' | nc -U /tmp/contextgraph.sock 2>/dev/null || echo '{"result":{"status":"proceed"}}')

# Parse response
STATUS=$(echo "$RESPONSE" | jq -r '.result.status // "proceed"')
MESSAGE=$(echo "$RESPONSE" | jq -r '.result.message // empty')
WARNINGS=$(echo "$RESPONSE" | jq -r '.result.warnings[]? // empty')
CONSCIOUSNESS=$(echo "$RESPONSE" | jq -r '.result.injected_context.consciousness.consciousness_level // 0')

# Output warnings
if [ -n "$WARNINGS" ]; then
  echo "Safety warnings (consciousness=${CONSCIOUSNESS}):" >&2
  echo "$WARNINGS" | while read -r warning; do
    [ -n "$warning" ] && echo "  - $warning" >&2
  done
fi

# Output message
if [ -n "$MESSAGE" ]; then
  echo "$MESSAGE" >&2
fi

# Output suggested alternatives if any
ALTERNATIVES=$(echo "$RESPONSE" | jq -r '.result.injected_context.alternatives[]? // empty')
if [ -n "$ALTERNATIVES" ]; then
  echo "Suggested alternatives:" >&2
  echo "$ALTERNATIVES" | while read -r alt; do
    [ -n "$alt" ] && echo "  - $alt" >&2
  done
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
    echo "Command blocked for safety reasons" >&2
    exit 1
    ;;
  *)
    exit 0
    ;;
esac
```

```bash
#!/bin/bash
# .claude/hooks/post_tool_use_bash.sh
# PostToolUse hook for Bash tool - stores command outputs for learning
# Configured in .claude/settings.json with matcher: "Bash"
# Timeout: 3000ms (per constitution)

set -e

# Read input from stdin (JSON payload from Claude Code)
# Format: {
#   "tool_name": "Bash",
#   "tool_input": { "command": "...", "description": "...", "timeout": ... },
#   "tool_response": { "stdout": "...", "stderr": "...", "exit_code": 0, "interrupted": false }
# }
INPUT=$(cat)

# Extract tool_input fields
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
DESCRIPTION=$(echo "$INPUT" | jq -r '.tool_input.description // empty')
TIMEOUT=$(echo "$INPUT" | jq -r '.tool_input.timeout // 120000')

# Extract tool_response fields
STDOUT=$(echo "$INPUT" | jq -r '.tool_response.stdout // ""')
STDERR=$(echo "$INPUT" | jq -r '.tool_response.stderr // ""')
EXIT_CODE=$(echo "$INPUT" | jq -r '.tool_response.exit_code // 0')
INTERRUPTED=$(echo "$INPUT" | jq -r '.tool_response.interrupted // false')
DURATION_MS=$(echo "$INPUT" | jq -r '.tool_response.duration_ms // 0')

# Skip if no command
if [ -z "$COMMAND" ]; then
  exit 0
fi

# Skip noisy commands (unless they failed)
CMD_START=$(echo "$COMMAND" | awk '{print $1}')
case "$CMD_START" in
  ls|pwd|echo|cd|clear)
    if [ "$EXIT_CODE" -eq 0 ]; then
      exit 0
    fi
    ;;
esac

# Check output length
TOTAL_LEN=$((${#STDOUT} + ${#STDERR}))
if [ "$TOTAL_LEN" -lt 10 ]; then
  exit 0
fi

# Call MCP tool to store and learn
RESPONSE=$(echo '{
  "jsonrpc": "2.0",
  "id": "post-tool-use-bash-'$(date +%s%N)'",
  "method": "hooks/post_tool_use_bash",
  "params": {
    "tool_name": "Bash",
    "tool_input": {
      "command": '"$(echo "$COMMAND" | jq -Rs .)"',
      "description": '"$(echo "$DESCRIPTION" | jq -Rs .)"',
      "timeout": '"$TIMEOUT"'
    },
    "tool_response": {
      "stdout": '"$(echo "$STDOUT" | head -c 50000 | jq -Rs .)"',
      "stderr": '"$(echo "$STDERR" | head -c 10000 | jq -Rs .)"',
      "exit_code": '"$EXIT_CODE"',
      "interrupted": '"$INTERRUPTED"',
      "duration_ms": '"$DURATION_MS"'
    }
  }
}' | nc -U /tmp/contextgraph.sock 2>/dev/null || echo '{"error":{"message":"Connection failed"}}')

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error.message // empty')
if [ -n "$ERROR" ]; then
  echo "Warning: PostToolUse hook failed: $ERROR" >&2
fi

# Extract learning info
MEMORY_ID=$(echo "$RESPONSE" | jq -r '.result.injected_context.memory_id // empty')
CONSCIOUSNESS=$(echo "$RESPONSE" | jq -r '.result.injected_context.consciousness_level // 0')
EMBEDDERS=$(echo "$RESPONSE" | jq -r '.result.injected_context.embedders_used // 0')

# Log learning info if verbose
if [ -n "$MEMORY_ID" ]; then
  echo "Stored as memory $MEMORY_ID (C=$CONSCIOUSNESS, embedders=$EMBEDDERS/13)" >&2
fi

# If command failed, output troubleshooting suggestions
if [ "$EXIT_CODE" -ne 0 ]; then
  TROUBLESHOOTING=$(echo "$RESPONSE" | jq -r '.result.injected_context.troubleshooting[]? // empty')
  if [ -n "$TROUBLESHOOTING" ]; then
    echo "Troubleshooting suggestions:" >&2
    echo "$TROUBLESHOOTING" | while read -r suggestion; do
      [ -n "$suggestion" ] && echo "  - $suggestion" >&2
    done
  fi

  # Output error patterns
  ERROR_PATTERNS=$(echo "$RESPONSE" | jq -r '.result.injected_context.error_patterns[]? // empty')
  if [ -n "$ERROR_PATTERNS" ]; then
    echo "Detected error patterns:" >&2
    echo "$ERROR_PATTERNS" | while read -r pattern; do
      [ -n "$pattern" ] && echo "  - $pattern" >&2
    done
  fi
fi

exit 0
```
    </section>
  </pseudo_code>

  <integration_notes>
    <note id="1">
      Claude Code uses PreToolUse/PostToolUse events for ALL tool interactions.
      The hook receives tool_name ("Bash"), tool_input, and (for post) tool_response.
    </note>
    <note id="2">
      The "matcher" field in settings.json determines which tools trigger the hook.
      Use "Bash" to match the Bash tool specifically.
    </note>
    <note id="3">
      Constitution requires 3000ms timeout for hooks. Budget internal processing
      to complete within this limit (2500ms for pre, 2800ms for post).
    </note>
    <note id="4">
      GWT consciousness integration enables adaptive learning:
      - Low consciousness: Minimal logging, no embeddings
      - High consciousness: Full 13-embedder storage, trajectory tracking
    </note>
    <note id="5">
      The 13 embedders (E1-E13) capture different aspects of command knowledge:
      semantic, temporal, causal, counterfactual, moral, aesthetic, teleological,
      contextual, structural, behavioral, emotional, code, and sparse lexical.
    </note>
  </integration_notes>
</task_spec>
```
