# TASK-INTEG-010: Integration Testing (Hooks, Skills, Subagents)

```xml
<task_spec>
  <task_id>TASK-INTEG-010</task_id>
  <title>Integration Testing for Hooks, Skills, and Subagents</title>
  <status>pending</status>

  <objective>
    Implement comprehensive integration tests for the Claude Code integration
    components (hooks, skills, subagents) working together with the teleological
    array system. Tests must validate Claude Code's exact event format, all 13
    embedders, GWT consciousness calculations, timeout compliance, and agent
    lifecycle management.
  </objective>

  <rationale>
    Integration testing ensures:
    1. All components work together correctly
    2. Hook-to-skill-to-store data flows are correct
    3. Subagent orchestration functions properly
    4. Performance meets targets under realistic conditions
    5. Error handling and recovery work across components
    6. Claude Code event formats are correctly parsed and processed
    7. All 13 embedders (E1-E13) generate valid arrays with RRF fusion
    8. GWT consciousness C(t) = I(t) x R(t) x D(t) is accurately computed
    9. Constitutional timeout requirements are enforced
    10. Skill YAML parsing and auto-invoke patterns work correctly
    11. Agent spawning via Claude Code Task tool is properly tracked

    This testing phase validates the complete Claude Code integration.
  </rationale>

  <dependencies>
    <dependency type="required">TASK-INTEG-004</dependency>    <!-- Hook protocol -->
    <dependency type="required">TASK-INTEG-005</dependency>    <!-- Edit hooks -->
    <dependency type="required">TASK-INTEG-006</dependency>    <!-- File read hooks -->
    <dependency type="required">TASK-INTEG-007</dependency>    <!-- Bash hooks -->
    <dependency type="required">TASK-INTEG-008</dependency>    <!-- Skills -->
    <dependency type="required">TASK-INTEG-009</dependency>    <!-- Subagents -->
    <dependency type="required">TASK-LOGIC-010</dependency>    <!-- Teleological drift detection -->
    <dependency type="required">TASK-INTEG-001</dependency>    <!-- Memory MCP handlers -->
    <dependency type="required">TASK-INTEG-003</dependency>    <!-- Consciousness MCP dispatch -->
  </dependencies>

  <input_context_files>
    <file purpose="hooks">crates/context-graph-mcp/src/hooks/mod.rs</file>
    <file purpose="skills">crates/context-graph-mcp/src/skills/mod.rs</file>
    <file purpose="agents">crates/context-graph-mcp/src/agents/mod.rs</file>
    <file purpose="store">crates/context-graph-storage/src/teleological/store.rs</file>
    <file purpose="embedders">crates/context-graph-storage/src/teleological/embedders/mod.rs</file>
    <file purpose="consciousness">crates/context-graph-storage/src/teleological/consciousness/gwt.rs</file>
    <file purpose="constitution">docs2/CONSTITUTION.md</file>
    <file purpose="claude_settings">.claude/settings.json</file>
  </input_context_files>

  <output_artifacts>
    <artifact type="test">crates/context-graph-mcp/tests/hooks_integration.rs</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/skills_integration.rs</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/agents_integration.rs</artifact>
    <artifact type="test">crates/context-graph-mcp/tests/full_integration.rs</artifact>
    <artifact type="test">tests/claude_code/compatibility_test.rs</artifact>
    <artifact type="test">tests/integration/claude_code_hooks_test.rs</artifact>
    <artifact type="test">tests/integration/consciousness_test.rs</artifact>
    <artifact type="test">tests/integration/embedder_test.rs</artifact>
    <artifact type="bench">crates/context-graph-mcp/benches/integration_bench.rs</artifact>
    <artifact type="bench">tests/benchmarks/timeout_compliance_bench.rs</artifact>
  </output_artifacts>

  <definition_of_done>
    <criterion id="1">Hook integration tests cover all 10 hook types E2E</criterion>
    <criterion id="2">Skill integration tests verify all 5 skills function correctly</criterion>
    <criterion id="3">Subagent integration tests verify lifecycle and communication</criterion>
    <criterion id="4">Full system tests exercise complete workflows</criterion>
    <criterion id="5">Claude Code compatibility verified with settings.json</criterion>
    <criterion id="6">Performance benchmarks pass all targets</criterion>
    <criterion id="7">Error recovery scenarios tested</criterion>
    <criterion id="8">Test coverage exceeds 80% for integration code</criterion>
    <criterion id="9">Claude Code event format tests validate exact JSON structure</criterion>
    <criterion id="10">All 13 embedders (E1-E13) tested with RRF fusion verification</criterion>
    <criterion id="11">GWT consciousness tests verify C(t) calculation and state transitions</criterion>
    <criterion id="12">Timeout compliance tests enforce constitutional limits</criterion>
    <criterion id="13">Skill YAML parsing and auto-invoke pattern tests pass</criterion>
    <criterion id="14">Agent lifecycle tests verify Task tool spawning and SubagentStop consolidation</criterion>
  </definition_of_done>

  <estimated_complexity>High</estimated_complexity>

  <claude_code_integration>
    <event_format>
      Claude Code sends events in this exact JSON format:
      {
        "tool_name": "Write" | "Edit" | "Read" | "Bash" | ...,
        "tool_input": { /* tool-specific parameters */ },
        "session_id": "session-uuid-here",
        "environment": {
          "cwd": "/path/to/working/dir",
          "user": "username",
          "platform": "linux" | "darwin" | "win32"
        }
      }
    </event_format>

    <hook_matchers>
      PreToolUse and PostToolUse hooks match on tool_name:
      - Write: matches file creation/overwrite operations
      - Edit: matches in-place file modifications
      - Read: matches file read operations
      - Bash: matches command execution
      - Glob: matches file pattern searches
      - Grep: matches content searches
      - Task: matches subagent spawning
      - TodoWrite: matches task list operations
      - WebFetch: matches web content retrieval
      - WebSearch: matches web searches
    </hook_matchers>

    <hook_types>
      SessionStart, SessionEnd, PreCompact hooks for session lifecycle
      PreToolUse, PostToolUse hooks for tool interception
    </hook_types>
  </claude_code_integration>

  <timeout_requirements>
    Per constitution requirements:
    - SessionStart: must complete within 5000ms
    - PreToolUse: must complete within 3000ms
    - PostToolUse: must complete within 3000ms
    - SessionEnd: must complete within 60000ms
    - PreCompact: must complete within 10000ms

    Tests MUST verify these limits are never exceeded.
  </timeout_requirements>

  <pseudo_code>
    <section name="ClaudeCodeE2ETestingFramework">
```rust
// tests/integration/claude_code_hooks_test.rs

//! Claude Code E2E Testing Framework
//!
//! Tests hooks with Claude Code's actual event format and validates
//! PreToolUse/PostToolUse handlers for Write, Edit, Read, Bash matchers.

use context_graph_mcp::hooks::*;
use context_graph_mcp::claude_code::*;
use serde_json::json;
use std::time::{Duration, Instant};

/// Claude Code event format as sent by the actual Claude Code client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeEvent {
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub session_id: String,
    pub environment: ClaudeCodeEnvironment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeEnvironment {
    pub cwd: String,
    pub user: String,
    pub platform: String,  // "linux" | "darwin" | "win32"
}

/// Test harness for Claude Code integration tests
struct ClaudeCodeTestHarness {
    hook_dispatcher: Arc<HookDispatcher>,
    store: Arc<TeleologicalStore>,
    settings_validator: SettingsValidator,
}

impl ClaudeCodeTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let hook_dispatcher = Arc::new(HookDispatcher::new(store.clone()).await);
        let settings_validator = SettingsValidator::load(".claude/settings.json").unwrap();

        Self {
            hook_dispatcher,
            store,
            settings_validator,
        }
    }

    fn create_event(&self, tool_name: &str, tool_input: serde_json::Value) -> ClaudeCodeEvent {
        ClaudeCodeEvent {
            tool_name: tool_name.to_string(),
            tool_input,
            session_id: format!("test-session-{}", uuid::Uuid::new_v4()),
            environment: ClaudeCodeEnvironment {
                cwd: "/home/user/project".to_string(),
                user: "testuser".to_string(),
                platform: "linux".to_string(),
            },
        }
    }
}

mod event_format_tests {
    use super::*;

    #[tokio::test]
    async fn test_write_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Exact format Claude Code sends for Write tool
        let event = harness.create_event("Write", json!({
            "file_path": "/home/user/project/src/main.rs",
            "content": "fn main() { println!(\"Hello\"); }"
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(matches!(response.status, HookStatus::Proceed | HookStatus::ProceedWithWarning));
    }

    #[tokio::test]
    async fn test_edit_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Exact format Claude Code sends for Edit tool
        let event = harness.create_event("Edit", json!({
            "file_path": "/home/user/project/src/lib.rs",
            "old_string": "fn old_function()",
            "new_string": "fn new_function()",
            "replace_all": false
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_read_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Exact format Claude Code sends for Read tool
        let event = harness.create_event("Read", json!({
            "file_path": "/home/user/project/src/config.rs",
            "offset": 0,
            "limit": 500
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bash_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Exact format Claude Code sends for Bash tool
        let event = harness.create_event("Bash", json!({
            "command": "cargo test --lib",
            "description": "Run library tests",
            "timeout": 120000
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_task_event_format_for_subagent() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Exact format Claude Code sends for Task tool (subagent spawning)
        let event = harness.create_event("Task", json!({
            "description": "Analyze authentication patterns",
            "prompt": "You are a security researcher...",
            "mode": "researcher"
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        // Task events should trigger agent tracking
        assert!(response.injected_context.as_ref()
            .map(|c| c.get("agent_tracking_enabled").is_some())
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn test_glob_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        let event = harness.create_event("Glob", json!({
            "pattern": "**/*.rs",
            "path": "/home/user/project"
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_grep_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        let event = harness.create_event("Grep", json!({
            "pattern": "TODO|FIXME",
            "path": "/home/user/project/src",
            "output_mode": "content"
        }));

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&event)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_todowrite_event_format() {
        let harness = ClaudeCodeTestHarness::new().await;

        let event = harness.create_event("TodoWrite", json!({
            "todos": [
                {"content": "Implement auth", "status": "in_progress", "activeForm": "Implementing auth"},
                {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"}
            ]
        }));

        let result = harness.hook_dispatcher
            .dispatch_post_tool_use(&event, json!({"success": true}))
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_event_lifecycle() {
        let harness = ClaudeCodeTestHarness::new().await;

        // SessionStart event
        let session_id = "test-session-lifecycle";
        let start_result = harness.hook_dispatcher
            .dispatch_session_start(session_id, json!({
                "project": "contextgraph",
                "git_branch": "main"
            }))
            .await;

        assert!(start_result.is_ok());
        let start_response = start_result.unwrap();
        assert!(start_response.injected_context.is_some());

        // SessionEnd event
        let end_result = harness.hook_dispatcher
            .dispatch_session_end(session_id)
            .await;

        assert!(end_result.is_ok());
        let end_response = end_result.unwrap();
        assert!(end_response.injected_context.as_ref()
            .map(|c| c.get("consolidated_count").is_some())
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn test_pre_compact_event() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Inject some memories first
        for i in 0..20 {
            harness.store.inject(InjectionRequest {
                content: format!("Memory content {}", i),
                memory_type: MemoryType::General,
                namespace: "test".to_string(),
                ..Default::default()
            }).await.unwrap();
        }

        let result = harness.hook_dispatcher
            .dispatch_pre_compact("test-session")
            .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        // PreCompact should prepare context summary
        assert!(response.injected_context.is_some());
    }
}

mod settings_json_tests {
    use super::*;

    #[tokio::test]
    async fn test_settings_json_hook_configuration() {
        let harness = ClaudeCodeTestHarness::new().await;

        // Validate that settings.json has correct hook configurations
        let hooks = harness.settings_validator.get_hooks();

        // Must have at least the core hooks
        assert!(hooks.iter().any(|h| h.hook_type == "PreToolUse"));
        assert!(hooks.iter().any(|h| h.hook_type == "PostToolUse"));
        assert!(hooks.iter().any(|h| h.hook_type == "SessionStart"));
        assert!(hooks.iter().any(|h| h.hook_type == "SessionEnd"));
    }

    #[tokio::test]
    async fn test_settings_json_tool_matchers() {
        let harness = ClaudeCodeTestHarness::new().await;

        let pre_tool_hooks = harness.settings_validator.get_pre_tool_use_hooks();

        // Verify matchers are configured for key tools
        let matchers: Vec<&str> = pre_tool_hooks.iter()
            .flat_map(|h| h.matchers.iter().map(|m| m.as_str()))
            .collect();

        assert!(matchers.contains(&"Write") || matchers.contains(&"*"));
        assert!(matchers.contains(&"Edit") || matchers.contains(&"*"));
        assert!(matchers.contains(&"Read") || matchers.contains(&"*"));
        assert!(matchers.contains(&"Bash") || matchers.contains(&"*"));
    }

    #[tokio::test]
    async fn test_settings_json_mcp_server_config() {
        let harness = ClaudeCodeTestHarness::new().await;

        let mcp_servers = harness.settings_validator.get_mcp_servers();

        // Verify context-graph MCP server is configured
        assert!(mcp_servers.iter().any(|s| s.name.contains("context-graph")));
    }
}
```
    </section>

    <section name="EmbedderIntegrationTests">
```rust
// tests/integration/embedder_test.rs

//! 13-Embedder Test Coverage
//!
//! Tests each embedder (E1-E13) and verifies RRF fusion across all embedders.

use context_graph_storage::teleological::embedders::*;
use context_graph_storage::teleological::array::TeleologicalArray;
use ndarray::Array1;

/// Test harness for embedder integration tests
struct EmbedderTestHarness {
    pipeline: EmbedderPipeline,
}

impl EmbedderTestHarness {
    async fn new() -> Self {
        let pipeline = EmbedderPipeline::new().await.unwrap();
        Self { pipeline }
    }
}

mod individual_embedder_tests {
    use super::*;

    #[tokio::test]
    async fn test_e1_semantic_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let content = "Implement user authentication with JWT tokens";
        let embedding = harness.pipeline.embed_with(EmbedderType::E1_Semantic, content).await.unwrap();

        assert_eq!(embedding.len(), 768, "E1_Semantic should produce 768-dim vector");
        assert!(embedding.iter().all(|&v| v.is_finite()), "All values must be finite");
        assert!((embedding.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 0.01, "Should be normalized");
    }

    #[tokio::test]
    async fn test_e2_temporal_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let content = "Yesterday we deployed the feature, today we are testing it";
        let embedding = harness.pipeline.embed_with(EmbedderType::E2_Temporal, content).await.unwrap();

        assert_eq!(embedding.len(), 768);
        // Temporal embedder should capture time-related semantics

        let content2 = "In the future we will optimize this code";
        let embedding2 = harness.pipeline.embed_with(EmbedderType::E2_Temporal, content2).await.unwrap();

        // Past and future content should have different temporal encodings
        let similarity = cosine_similarity(&embedding, &embedding2);
        assert!(similarity < 0.95, "Temporal difference should be encoded");
    }

    #[tokio::test]
    async fn test_e3_causal_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let content = "If the user clicks the button, then the form submits";
        let embedding = harness.pipeline.embed_with(EmbedderType::E3_Causal, content).await.unwrap();

        assert_eq!(embedding.len(), 768);

        // Causal embedder should capture cause-effect relationships
        let cause = "User clicked button";
        let effect = "Form submitted";
        let cause_emb = harness.pipeline.embed_with(EmbedderType::E3_Causal, cause).await.unwrap();
        let effect_emb = harness.pipeline.embed_with(EmbedderType::E3_Causal, effect).await.unwrap();

        // The causal chain should link these
    }

    #[tokio::test]
    async fn test_e4_counterfactual_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let actual = "We used REST API for the implementation";
        let counterfactual = "If we had used GraphQL instead, the queries would be more efficient";

        let actual_emb = harness.pipeline.embed_with(EmbedderType::E4_Counterfactual, actual).await.unwrap();
        let counter_emb = harness.pipeline.embed_with(EmbedderType::E4_Counterfactual, counterfactual).await.unwrap();

        assert_eq!(actual_emb.len(), 768);
        assert_eq!(counter_emb.len(), 768);

        // Counterfactual should be related but distinct
        let similarity = cosine_similarity(&actual_emb, &counter_emb);
        assert!(similarity > 0.3 && similarity < 0.9, "Counterfactual should be related but distinct");
    }

    #[tokio::test]
    async fn test_e5_moral_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let ethical = "We must protect user privacy and never share data without consent";
        let unethical = "We can sell user data to maximize revenue";

        let ethical_emb = harness.pipeline.embed_with(EmbedderType::E5_Moral, ethical).await.unwrap();
        let unethical_emb = harness.pipeline.embed_with(EmbedderType::E5_Moral, unethical).await.unwrap();

        assert_eq!(ethical_emb.len(), 768);

        // Moral embedder should distinguish ethical from unethical
        let similarity = cosine_similarity(&ethical_emb, &unethical_emb);
        assert!(similarity < 0.7, "Ethical and unethical should be distinct");
    }

    #[tokio::test]
    async fn test_e6_aesthetic_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let elegant = "The solution uses a single recursive function with O(n) complexity";
        let ugly = "We have 20 nested if statements and global variables everywhere";

        let elegant_emb = harness.pipeline.embed_with(EmbedderType::E6_Aesthetic, elegant).await.unwrap();
        let ugly_emb = harness.pipeline.embed_with(EmbedderType::E6_Aesthetic, ugly).await.unwrap();

        assert_eq!(elegant_emb.len(), 768);
        // Aesthetic embedder should capture code quality aspects
    }

    #[tokio::test]
    async fn test_e7_teleological_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let purposeful = "This function validates user input to prevent SQL injection attacks";
        let aimless = "This function does some stuff with strings";

        let purposeful_emb = harness.pipeline.embed_with(EmbedderType::E7_Teleological, purposeful).await.unwrap();
        let aimless_emb = harness.pipeline.embed_with(EmbedderType::E7_Teleological, aimless).await.unwrap();

        assert_eq!(purposeful_emb.len(), 768);
        // Teleological embedder should encode purpose/intent
    }

    #[tokio::test]
    async fn test_e8_contextual_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let with_context = "In the context of our microservices architecture, this service handles auth";
        let embedding = harness.pipeline.embed_with(EmbedderType::E8_Contextual, with_context).await.unwrap();

        assert_eq!(embedding.len(), 768);
        // Contextual embedder should capture situational context
    }

    #[tokio::test]
    async fn test_e9_structural_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let hierarchical = "The UserService depends on UserRepository which uses DatabaseConnection";
        let flat = "Functions a, b, c are independent utilities";

        let hier_emb = harness.pipeline.embed_with(EmbedderType::E9_Structural, hierarchical).await.unwrap();
        let flat_emb = harness.pipeline.embed_with(EmbedderType::E9_Structural, flat).await.unwrap();

        assert_eq!(hier_emb.len(), 768);
        // Structural embedder should capture architectural patterns
    }

    #[tokio::test]
    async fn test_e10_behavioral_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let behavior = "When user logs in, system creates session, sets cookie, and redirects to dashboard";
        let embedding = harness.pipeline.embed_with(EmbedderType::E10_Behavioral, behavior).await.unwrap();

        assert_eq!(embedding.len(), 768);
        // Behavioral embedder should capture action sequences
    }

    #[tokio::test]
    async fn test_e11_emotional_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let frustrated = "This legacy code is a nightmare and nobody understands it anymore";
        let satisfied = "The refactoring went smoothly and the code is now clean and maintainable";

        let frust_emb = harness.pipeline.embed_with(EmbedderType::E11_Emotional, frustrated).await.unwrap();
        let sat_emb = harness.pipeline.embed_with(EmbedderType::E11_Emotional, satisfied).await.unwrap();

        assert_eq!(frust_emb.len(), 768);
        // Emotional embedder should capture sentiment
        let similarity = cosine_similarity(&frust_emb, &sat_emb);
        assert!(similarity < 0.8, "Different emotions should be distinguishable");
    }

    #[tokio::test]
    async fn test_e12_code_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let rust_code = r#"
            pub fn authenticate(token: &str) -> Result<User, AuthError> {
                let claims = decode_jwt(token)?;
                Ok(User::from_claims(claims))
            }
        "#;
        let embedding = harness.pipeline.embed_with(EmbedderType::E12_Code, rust_code).await.unwrap();

        assert_eq!(embedding.len(), 768);

        // Similar code should have similar embeddings
        let similar_code = r#"
            pub fn verify_auth(jwt: &str) -> Result<User, AuthError> {
                let parsed = parse_jwt(jwt)?;
                Ok(User::from_parsed(parsed))
            }
        "#;
        let similar_emb = harness.pipeline.embed_with(EmbedderType::E12_Code, similar_code).await.unwrap();

        let similarity = cosine_similarity(&embedding, &similar_emb);
        assert!(similarity > 0.7, "Similar code should have high similarity");
    }

    #[tokio::test]
    async fn test_e13_splade_embedder() {
        let harness = EmbedderTestHarness::new().await;

        let content = "authentication login JWT token verification security";
        let embedding = harness.pipeline.embed_with(EmbedderType::E13_SPLADE, content).await.unwrap();

        // SPLADE produces sparse vectors - check sparsity
        let non_zero = embedding.iter().filter(|&&v| v.abs() > 1e-6).count();
        let sparsity = 1.0 - (non_zero as f32 / embedding.len() as f32);
        assert!(sparsity > 0.7, "SPLADE should produce sparse vectors, got sparsity {}", sparsity);
    }
}

mod rrf_fusion_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_pipeline_generates_all_13_embeddings() {
        let harness = EmbedderTestHarness::new().await;

        let content = "Implement secure user authentication with role-based access control";
        let array = harness.pipeline.embed_all(content).await.unwrap();

        assert_eq!(array.embeddings.len(), 13, "Should generate 13 embeddings");

        // Verify each embedder type is present
        assert!(array.embeddings.contains_key(&EmbedderType::E1_Semantic));
        assert!(array.embeddings.contains_key(&EmbedderType::E2_Temporal));
        assert!(array.embeddings.contains_key(&EmbedderType::E3_Causal));
        assert!(array.embeddings.contains_key(&EmbedderType::E4_Counterfactual));
        assert!(array.embeddings.contains_key(&EmbedderType::E5_Moral));
        assert!(array.embeddings.contains_key(&EmbedderType::E6_Aesthetic));
        assert!(array.embeddings.contains_key(&EmbedderType::E7_Teleological));
        assert!(array.embeddings.contains_key(&EmbedderType::E8_Contextual));
        assert!(array.embeddings.contains_key(&EmbedderType::E9_Structural));
        assert!(array.embeddings.contains_key(&EmbedderType::E10_Behavioral));
        assert!(array.embeddings.contains_key(&EmbedderType::E11_Emotional));
        assert!(array.embeddings.contains_key(&EmbedderType::E12_Code));
        assert!(array.embeddings.contains_key(&EmbedderType::E13_SPLADE));
    }

    #[tokio::test]
    async fn test_rrf_fusion_ranking() {
        let harness = EmbedderTestHarness::new().await;

        // Create query and documents
        let query = "authentication security tokens";
        let doc1 = "JWT token authentication for secure login";
        let doc2 = "Database connection pooling for performance";
        let doc3 = "OAuth2 security tokens for API authentication";

        let query_array = harness.pipeline.embed_all(query).await.unwrap();
        let doc1_array = harness.pipeline.embed_all(doc1).await.unwrap();
        let doc2_array = harness.pipeline.embed_all(doc2).await.unwrap();
        let doc3_array = harness.pipeline.embed_all(doc3).await.unwrap();

        // Compute RRF scores
        let rrf = RRFFusion::new(60); // k=60 is typical
        let scores = rrf.compute_scores(
            &query_array,
            &[doc1_array.clone(), doc2_array.clone(), doc3_array.clone()],
        );

        // doc1 and doc3 should rank higher than doc2 (more relevant to query)
        assert!(scores[0] > scores[1], "doc1 should score higher than doc2");
        assert!(scores[2] > scores[1], "doc3 should score higher than doc2");
    }

    #[tokio::test]
    async fn test_rrf_per_embedder_contribution() {
        let harness = EmbedderTestHarness::new().await;

        let query = "authentication";
        let doc = "secure login with JWT";

        let query_array = harness.pipeline.embed_all(query).await.unwrap();
        let doc_array = harness.pipeline.embed_all(doc).await.unwrap();

        let rrf = RRFFusion::new(60);
        let contribution = rrf.per_embedder_scores(&query_array, &doc_array);

        // Each embedder should contribute a score
        assert_eq!(contribution.len(), 13);

        // E1_Semantic and E12_Code should contribute most for this query
        assert!(contribution[&EmbedderType::E1_Semantic] > 0.0);
    }

    #[tokio::test]
    async fn test_embedder_weight_configuration() {
        let harness = EmbedderTestHarness::new().await;

        // Custom weights for different use cases
        let code_heavy_weights = EmbedderWeights::new()
            .with_weight(EmbedderType::E12_Code, 2.0)
            .with_weight(EmbedderType::E9_Structural, 1.5)
            .with_weight(EmbedderType::E1_Semantic, 0.5);

        let rrf = RRFFusion::with_weights(60, code_heavy_weights);

        let query = "function implementation";
        let code_doc = "pub fn process(data: &[u8]) -> Result<Output, Error>";
        let prose_doc = "The process function handles data processing";

        let q = harness.pipeline.embed_all(query).await.unwrap();
        let code = harness.pipeline.embed_all(code_doc).await.unwrap();
        let prose = harness.pipeline.embed_all(prose_doc).await.unwrap();

        let scores = rrf.compute_scores(&q, &[code.clone(), prose.clone()]);

        // With code-heavy weights, code doc should rank higher
        assert!(scores[0] > scores[1], "Code doc should rank higher with code-heavy weights");
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```
    </section>

    <section name="ConsciousnessIntegrationTests">
```rust
// tests/integration/consciousness_test.rs

//! GWT Consciousness Tests
//!
//! Tests consciousness calculations:
//! - C(t) = I(t) x R(t) x D(t) calculation accuracy
//! - Consciousness state transitions (Dormant -> Fragmented -> Emerging -> Conscious -> Hypersync)
//! - GWT broadcast mechanism
//! - Workspace integration tests

use context_graph_storage::teleological::consciousness::*;
use context_graph_storage::teleological::TeleologicalStore;
use std::sync::Arc;

/// Test harness for consciousness integration tests
struct ConsciousnessTestHarness {
    gwt: Arc<GlobalWorkspaceTheory>,
    store: Arc<TeleologicalStore>,
}

impl ConsciousnessTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let gwt = Arc::new(GlobalWorkspaceTheory::new(store.clone()));

        Self { gwt, store }
    }
}

mod consciousness_calculation_tests {
    use super::*;

    #[tokio::test]
    async fn test_consciousness_formula_accuracy() {
        let harness = ConsciousnessTestHarness::new().await;

        // C(t) = I(t) x R(t) x D(t)
        // Where:
        //   I(t) = Integration (how well information is bound)
        //   R(t) = Relevance (how relevant to current context)
        //   D(t) = Differentiation (how distinct/novel)

        let metrics = ConsciousnessMetrics {
            integration: 0.8,
            relevance: 0.9,
            differentiation: 0.7,
        };

        let consciousness = harness.gwt.calculate_consciousness(&metrics);

        // C(t) = 0.8 * 0.9 * 0.7 = 0.504
        let expected = 0.8 * 0.9 * 0.7;
        assert!((consciousness - expected).abs() < 0.001,
            "Expected C(t) = {}, got {}", expected, consciousness);
    }

    #[tokio::test]
    async fn test_consciousness_bounds() {
        let harness = ConsciousnessTestHarness::new().await;

        // Maximum consciousness
        let max_metrics = ConsciousnessMetrics {
            integration: 1.0,
            relevance: 1.0,
            differentiation: 1.0,
        };
        let max_c = harness.gwt.calculate_consciousness(&max_metrics);
        assert!((max_c - 1.0).abs() < 0.001, "Max consciousness should be 1.0");

        // Minimum consciousness
        let min_metrics = ConsciousnessMetrics {
            integration: 0.0,
            relevance: 0.0,
            differentiation: 0.0,
        };
        let min_c = harness.gwt.calculate_consciousness(&min_metrics);
        assert!((min_c - 0.0).abs() < 0.001, "Min consciousness should be 0.0");

        // Any zero factor should zero consciousness
        let zero_int = ConsciousnessMetrics {
            integration: 0.0,
            relevance: 0.9,
            differentiation: 0.8,
        };
        let zero_c = harness.gwt.calculate_consciousness(&zero_int);
        assert!((zero_c - 0.0).abs() < 0.001, "Zero integration should zero consciousness");
    }

    #[tokio::test]
    async fn test_integration_factor_calculation() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store multiple related memories
        for i in 0..10 {
            harness.store.inject(InjectionRequest {
                content: format!("Authentication module component {}", i),
                memory_type: MemoryType::CodeContext,
                namespace: "auth".to_string(),
                ..Default::default()
            }).await.unwrap();
        }

        // Integration should be higher when memories are related
        let integration = harness.gwt.calculate_integration("auth").await.unwrap();
        assert!(integration > 0.5, "Related memories should show high integration");
    }

    #[tokio::test]
    async fn test_relevance_factor_calculation() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store a memory
        harness.store.inject(InjectionRequest {
            content: "JWT token validation logic".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "auth".to_string(),
            ..Default::default()
        }).await.unwrap();

        // Relevance to a related query
        let relevant_query = "authentication tokens";
        let relevance = harness.gwt.calculate_relevance(relevant_query).await.unwrap();
        assert!(relevance > 0.5, "Related query should show high relevance");

        // Relevance to an unrelated query
        let irrelevant_query = "database migrations";
        let irrelevance = harness.gwt.calculate_relevance(irrelevant_query).await.unwrap();
        assert!(irrelevance < relevance, "Unrelated query should show lower relevance");
    }

    #[tokio::test]
    async fn test_differentiation_factor_calculation() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store similar memories
        for i in 0..5 {
            harness.store.inject(InjectionRequest {
                content: format!("User login function version {}", i),
                memory_type: MemoryType::CodeContext,
                namespace: "auth".to_string(),
                ..Default::default()
            }).await.unwrap();
        }

        // Differentiation should be lower for similar content
        let diff_similar = harness.gwt.calculate_differentiation("User login function").await.unwrap();

        // Store novel content
        harness.store.inject(InjectionRequest {
            content: "Quantum-resistant encryption algorithm using lattice cryptography".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "crypto".to_string(),
            ..Default::default()
        }).await.unwrap();

        let diff_novel = harness.gwt.calculate_differentiation("Quantum encryption").await.unwrap();

        assert!(diff_novel > diff_similar, "Novel content should show higher differentiation");
    }
}

mod state_transition_tests {
    use super::*;

    #[tokio::test]
    async fn test_consciousness_state_transitions() {
        let harness = ConsciousnessTestHarness::new().await;

        // State thresholds:
        // Dormant: C(t) < 0.1
        // Fragmented: 0.1 <= C(t) < 0.3
        // Emerging: 0.3 <= C(t) < 0.6
        // Conscious: 0.6 <= C(t) < 0.9
        // Hypersync: C(t) >= 0.9

        // Test Dormant state
        let dormant_metrics = ConsciousnessMetrics {
            integration: 0.3,
            relevance: 0.3,
            differentiation: 0.1, // C = 0.009
        };
        let state = harness.gwt.determine_state(&dormant_metrics);
        assert_eq!(state, ConsciousnessState::Dormant);

        // Test Fragmented state
        let fragmented_metrics = ConsciousnessMetrics {
            integration: 0.5,
            relevance: 0.5,
            differentiation: 0.5, // C = 0.125
        };
        let state = harness.gwt.determine_state(&fragmented_metrics);
        assert_eq!(state, ConsciousnessState::Fragmented);

        // Test Emerging state
        let emerging_metrics = ConsciousnessMetrics {
            integration: 0.7,
            relevance: 0.7,
            differentiation: 0.7, // C = 0.343
        };
        let state = harness.gwt.determine_state(&emerging_metrics);
        assert_eq!(state, ConsciousnessState::Emerging);

        // Test Conscious state
        let conscious_metrics = ConsciousnessMetrics {
            integration: 0.85,
            relevance: 0.85,
            differentiation: 0.85, // C = 0.614
        };
        let state = harness.gwt.determine_state(&conscious_metrics);
        assert_eq!(state, ConsciousnessState::Conscious);

        // Test Hypersync state
        let hypersync_metrics = ConsciousnessMetrics {
            integration: 0.97,
            relevance: 0.97,
            differentiation: 0.97, // C = 0.913
        };
        let state = harness.gwt.determine_state(&hypersync_metrics);
        assert_eq!(state, ConsciousnessState::Hypersync);
    }

    #[tokio::test]
    async fn test_state_transition_history() {
        let harness = ConsciousnessTestHarness::new().await;

        // Simulate gradual consciousness increase
        let states_over_time = vec![
            (0.05, ConsciousnessState::Dormant),
            (0.15, ConsciousnessState::Fragmented),
            (0.45, ConsciousnessState::Emerging),
            (0.75, ConsciousnessState::Conscious),
            (0.92, ConsciousnessState::Hypersync),
        ];

        let mut history = Vec::new();
        for (c_value, expected_state) in states_over_time {
            let metrics = ConsciousnessMetrics::from_consciousness_value(c_value);
            let state = harness.gwt.determine_state(&metrics);
            assert_eq!(state, expected_state);
            history.push((c_value, state));
        }

        // Verify monotonic state progression
        for i in 1..history.len() {
            assert!(history[i].1 >= history[i-1].1,
                "States should progress monotonically with increasing consciousness");
        }
    }
}

mod gwt_broadcast_tests {
    use super::*;

    #[tokio::test]
    async fn test_workspace_broadcast() {
        let harness = ConsciousnessTestHarness::new().await;

        // Add content to workspace
        let memory_id = harness.store.inject(InjectionRequest {
            content: "Critical security vulnerability in auth module".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "security".to_string(),
            priority: Some(Priority::Critical),
            ..Default::default()
        }).await.unwrap();

        // Broadcast should make high-consciousness content available globally
        let broadcast_result = harness.gwt.broadcast(memory_id).await.unwrap();

        assert!(broadcast_result.reached_workspaces > 0, "Should broadcast to workspaces");
        assert!(broadcast_result.consciousness_level > 0.5, "Critical content should have high consciousness");
    }

    #[tokio::test]
    async fn test_workspace_competition() {
        let harness = ConsciousnessTestHarness::new().await;

        // Add multiple competing contents
        let ids: Vec<_> = (0..5).map(|i| {
            harness.store.inject(InjectionRequest {
                content: format!("Content competing for attention {}", i),
                memory_type: MemoryType::General,
                namespace: "test".to_string(),
                ..Default::default()
            })
        }).collect();

        let memory_ids: Vec<_> = futures::future::join_all(ids)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Only highest consciousness content should win broadcast
        let winner = harness.gwt.compete_for_broadcast(&memory_ids).await.unwrap();

        assert!(winner.is_some(), "Competition should produce a winner");
    }

    #[tokio::test]
    async fn test_attention_modulation() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store content
        let memory_id = harness.store.inject(InjectionRequest {
            content: "Authentication implementation details".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "auth".to_string(),
            ..Default::default()
        }).await.unwrap();

        // Modulate attention (simulate focusing on auth)
        harness.gwt.modulate_attention("authentication").await.unwrap();

        // Related content should now have higher consciousness
        let consciousness = harness.gwt.get_consciousness_level(memory_id).await.unwrap();
        assert!(consciousness > 0.3, "Attended content should have elevated consciousness");
    }
}

mod workspace_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_workspace_with_teleological_store() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store a goal
        let goal_array = create_test_teleological_array("secure authentication").await;
        harness.store.store_goal(GoalNode {
            goal_id: "auth-goal".to_string(),
            label: "Implement secure auth".to_string(),
            level: GoalLevel::Strategic,
            teleological_array: goal_array,
            ..Default::default()
        }).await.unwrap();

        // Store related memory
        let memory_id = harness.store.inject(InjectionRequest {
            content: "JWT validation with RS256 signing".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "auth".to_string(),
            ..Default::default()
        }).await.unwrap();

        // Workspace should integrate goal and memory
        let workspace_state = harness.gwt.get_workspace_state().await.unwrap();

        assert!(workspace_state.active_goals.contains(&"auth-goal".to_string()));
        assert!(workspace_state.consciousness_level > 0.0);
    }

    #[tokio::test]
    async fn test_consciousness_affects_search_ranking() {
        let harness = ConsciousnessTestHarness::new().await;

        // Store memories with different consciousness levels
        let high_c_id = harness.store.inject(InjectionRequest {
            content: "Critical authentication bypass vulnerability".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "security".to_string(),
            priority: Some(Priority::Critical),
            ..Default::default()
        }).await.unwrap();

        let low_c_id = harness.store.inject(InjectionRequest {
            content: "Minor code formatting changes".to_string(),
            memory_type: MemoryType::General,
            namespace: "misc".to_string(),
            ..Default::default()
        }).await.unwrap();

        // Search with consciousness-weighted ranking
        let results = harness.gwt.consciousness_weighted_search("code changes").await.unwrap();

        // High consciousness content should rank higher
        let high_c_rank = results.iter().position(|r| r.id == high_c_id);
        let low_c_rank = results.iter().position(|r| r.id == low_c_id);

        if let (Some(high_r), Some(low_r)) = (high_c_rank, low_c_rank) {
            assert!(high_r < low_r, "High consciousness content should rank higher");
        }
    }
}
```
    </section>

    <section name="TimeoutComplianceTests">
```rust
// tests/benchmarks/timeout_compliance_bench.rs

//! Hook Timeout Compliance Tests
//!
//! Per constitution requirements:
//! - SessionStart: must complete within 5000ms
//! - PreToolUse: must complete within 3000ms
//! - PostToolUse: must complete within 3000ms
//! - SessionEnd: must complete within 60000ms
//! - PreCompact: must complete within 10000ms

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use context_graph_mcp::hooks::*;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Constitutional timeout limits in milliseconds
const TIMEOUT_SESSION_START: u64 = 5000;
const TIMEOUT_PRE_TOOL_USE: u64 = 3000;
const TIMEOUT_POST_TOOL_USE: u64 = 3000;
const TIMEOUT_SESSION_END: u64 = 60000;
const TIMEOUT_PRE_COMPACT: u64 = 10000;

struct TimeoutTestHarness {
    hook_dispatcher: Arc<HookDispatcher>,
    store: Arc<TeleologicalStore>,
}

impl TimeoutTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let hook_dispatcher = Arc::new(HookDispatcher::new(store.clone()).await);
        Self { hook_dispatcher, store }
    }

    async fn with_data(memory_count: usize) -> Self {
        let harness = Self::new().await;

        // Pre-populate with test data
        for i in 0..memory_count {
            harness.store.inject(InjectionRequest {
                content: format!("Test memory content {}", i),
                memory_type: MemoryType::General,
                namespace: "test".to_string(),
                ..Default::default()
            }).await.unwrap();
        }

        harness
    }
}

mod constitutional_timeout_tests {
    use super::*;

    #[tokio::test]
    async fn test_session_start_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(1000).await;

        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_SESSION_START),
            harness.hook_dispatcher.dispatch_session_start("test-session", json!({}))
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "SessionStart must complete within {}ms, took {:?}",
            TIMEOUT_SESSION_START, elapsed);
        assert!(elapsed.as_millis() < TIMEOUT_SESSION_START as u128,
            "SessionStart took {}ms, limit is {}ms", elapsed.as_millis(), TIMEOUT_SESSION_START);
    }

    #[tokio::test]
    async fn test_pre_tool_use_timeout_compliance_write() {
        let harness = TimeoutTestHarness::with_data(1000).await;

        let event = ClaudeCodeEvent {
            tool_name: "Write".to_string(),
            tool_input: json!({
                "file_path": "/test/file.rs",
                "content": "fn test() {}"
            }),
            session_id: "test".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/test".to_string(),
                user: "test".to_string(),
                platform: "linux".to_string(),
            },
        };

        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_PRE_TOOL_USE),
            harness.hook_dispatcher.dispatch_pre_tool_use(&event)
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "PreToolUse (Write) must complete within {}ms, took {:?}",
            TIMEOUT_PRE_TOOL_USE, elapsed);
        assert!(elapsed.as_millis() < TIMEOUT_PRE_TOOL_USE as u128,
            "PreToolUse took {}ms, limit is {}ms", elapsed.as_millis(), TIMEOUT_PRE_TOOL_USE);
    }

    #[tokio::test]
    async fn test_pre_tool_use_timeout_compliance_bash() {
        let harness = TimeoutTestHarness::with_data(1000).await;

        let event = ClaudeCodeEvent {
            tool_name: "Bash".to_string(),
            tool_input: json!({
                "command": "rm -rf /important",
                "description": "Dangerous command"
            }),
            session_id: "test".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/test".to_string(),
                user: "test".to_string(),
                platform: "linux".to_string(),
            },
        };

        // Even command safety analysis must complete within timeout
        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_PRE_TOOL_USE),
            harness.hook_dispatcher.dispatch_pre_tool_use(&event)
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "PreToolUse (Bash) must complete within {}ms", TIMEOUT_PRE_TOOL_USE);
        assert!(elapsed.as_millis() < TIMEOUT_PRE_TOOL_USE as u128);
    }

    #[tokio::test]
    async fn test_post_tool_use_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(1000).await;

        let event = ClaudeCodeEvent {
            tool_name: "Write".to_string(),
            tool_input: json!({
                "file_path": "/test/large_file.rs",
                "content": "x".repeat(100_000) // 100KB file
            }),
            session_id: "test".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/test".to_string(),
                user: "test".to_string(),
                platform: "linux".to_string(),
            },
        };

        let tool_result = json!({"success": true});

        // Even with large content, embedding must complete within timeout
        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_POST_TOOL_USE),
            harness.hook_dispatcher.dispatch_post_tool_use(&event, tool_result)
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "PostToolUse must complete within {}ms, took {:?}",
            TIMEOUT_POST_TOOL_USE, elapsed);
        assert!(elapsed.as_millis() < TIMEOUT_POST_TOOL_USE as u128);
    }

    #[tokio::test]
    async fn test_session_end_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(5000).await; // 5000 memories to consolidate

        // Start session first
        harness.hook_dispatcher.dispatch_session_start("end-test-session", json!({})).await.unwrap();

        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_SESSION_END),
            harness.hook_dispatcher.dispatch_session_end("end-test-session")
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "SessionEnd must complete within {}ms, took {:?}",
            TIMEOUT_SESSION_END, elapsed);
        assert!(elapsed.as_millis() < TIMEOUT_SESSION_END as u128);
    }

    #[tokio::test]
    async fn test_pre_compact_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(10000).await; // 10000 memories

        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_PRE_COMPACT),
            harness.hook_dispatcher.dispatch_pre_compact("compact-test")
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "PreCompact must complete within {}ms, took {:?}",
            TIMEOUT_PRE_COMPACT, elapsed);
        assert!(elapsed.as_millis() < TIMEOUT_PRE_COMPACT as u128);
    }
}

mod stress_timeout_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_hooks_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(1000).await;

        // Simulate 10 concurrent hook invocations
        let futures: Vec<_> = (0..10).map(|i| {
            let dispatcher = harness.hook_dispatcher.clone();
            async move {
                let event = ClaudeCodeEvent {
                    tool_name: "Read".to_string(),
                    tool_input: json!({ "file_path": format!("/test/file{}.rs", i) }),
                    session_id: format!("session-{}", i),
                    environment: ClaudeCodeEnvironment {
                        cwd: "/test".to_string(),
                        user: "test".to_string(),
                        platform: "linux".to_string(),
                    },
                };

                let start = Instant::now();
                let result = timeout(
                    Duration::from_millis(TIMEOUT_PRE_TOOL_USE),
                    dispatcher.dispatch_pre_tool_use(&event)
                ).await;
                (i, start.elapsed(), result.is_ok())
            }
        }).collect();

        let results = futures::future::join_all(futures).await;

        for (i, elapsed, ok) in results {
            assert!(ok, "Concurrent hook {} must complete within timeout", i);
            assert!(elapsed.as_millis() < TIMEOUT_PRE_TOOL_USE as u128,
                "Concurrent hook {} took {}ms", i, elapsed.as_millis());
        }
    }

    #[tokio::test]
    async fn test_large_context_timeout_compliance() {
        let harness = TimeoutTestHarness::with_data(50000).await; // 50K memories

        let event = ClaudeCodeEvent {
            tool_name: "Read".to_string(),
            tool_input: json!({ "file_path": "/test/file.rs" }),
            session_id: "large-context-test".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/test".to_string(),
                user: "test".to_string(),
                platform: "linux".to_string(),
            },
        };

        // Even with large context, must complete within timeout
        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_PRE_TOOL_USE),
            harness.hook_dispatcher.dispatch_pre_tool_use(&event)
        ).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Large context hook must complete within timeout");
        assert!(elapsed.as_millis() < TIMEOUT_PRE_TOOL_USE as u128);
    }
}

fn timeout_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("Timeout Compliance");
    group.sample_size(50);

    for memory_count in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("SessionStart", memory_count),
            &memory_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let harness = TimeoutTestHarness::with_data(count).await;
                        harness.hook_dispatcher
                            .dispatch_session_start("bench-session", json!({}))
                            .await
                            .unwrap()
                    })
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PreToolUse", memory_count),
            &memory_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let harness = TimeoutTestHarness::with_data(count).await;
                        let event = ClaudeCodeEvent {
                            tool_name: "Write".to_string(),
                            tool_input: json!({"file_path": "/test.rs", "content": "fn t() {}"}),
                            session_id: "bench".to_string(),
                            environment: ClaudeCodeEnvironment {
                                cwd: "/".to_string(),
                                user: "u".to_string(),
                                platform: "linux".to_string(),
                            },
                        };
                        harness.hook_dispatcher
                            .dispatch_pre_tool_use(&event)
                            .await
                            .unwrap()
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, timeout_benchmarks);
criterion_main!(benches);
```
    </section>

    <section name="SkillInvocationTests">
```rust
// tests/integration/skills_test.rs

//! Skill Invocation Tests
//!
//! Tests skill system:
//! - Skill YAML parsing from .claude/skills/
//! - Auto-invoke pattern matching
//! - MCP tool invocation from skills
//! - Error handling and fallback responses

use context_graph_mcp::skills::*;
use std::path::PathBuf;

struct SkillInvocationTestHarness {
    skill_loader: SkillLoader,
    skill_registry: SkillRegistry,
}

impl SkillInvocationTestHarness {
    async fn new() -> Self {
        let skills_dir = PathBuf::from(".claude/skills");
        let skill_loader = SkillLoader::new(skills_dir.clone());
        let skill_registry = SkillRegistry::new(skills_dir);

        Self { skill_loader, skill_registry }
    }
}

mod yaml_parsing_tests {
    use super::*;

    #[tokio::test]
    async fn test_skill_yaml_parsing() {
        let harness = SkillInvocationTestHarness::new().await;

        // Load skills from .claude/skills/
        let skills = harness.skill_loader.load_all().await.unwrap();

        // Should find the core skills
        assert!(skills.len() >= 5, "Should have at least 5 core skills");
    }

    #[tokio::test]
    async fn test_skill_yaml_frontmatter() {
        let harness = SkillInvocationTestHarness::new().await;

        // Parse a skill with full frontmatter
        let skill = harness.skill_loader.load_skill("memory-search").await.unwrap();

        // Verify frontmatter fields
        assert!(!skill.name.is_empty());
        assert!(!skill.description.is_empty());
        assert!(skill.auto_invoke.is_some() || skill.allowed_tools.is_some());
    }

    #[tokio::test]
    async fn test_skill_yaml_allowed_tools() {
        let harness = SkillInvocationTestHarness::new().await;

        let skill = harness.skill_loader.load_skill("memory-search").await.unwrap();

        // Verify allowed_tools are MCP tools
        if let Some(tools) = &skill.allowed_tools {
            for tool in tools {
                // Should be valid MCP tool references
                assert!(tool.contains("mcp__") || tool.starts_with("context_graph__"),
                    "Tool {} should be MCP tool reference", tool);
            }
        }
    }

    #[tokio::test]
    async fn test_skill_yaml_parameters() {
        let harness = SkillInvocationTestHarness::new().await;

        let skill = harness.skill_loader.load_skill("goal-alignment").await.unwrap();

        // Verify parameters are defined
        assert!(!skill.parameters.is_empty(), "Skill should have parameters");

        // Check for required vs optional parameters
        let has_required = skill.parameters.iter().any(|p| p.required);
        assert!(has_required, "Should have at least one required parameter");
    }
}

mod auto_invoke_tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_invoke_pattern_matching() {
        let harness = SkillInvocationTestHarness::new().await;

        // Test patterns that should match memory-search skill
        let should_match = [
            "search memory for authentication",
            "find memories about JWT",
            "what do we know about the auth system",
            "recall patterns for error handling",
        ];

        for prompt in should_match {
            let matched = harness.skill_registry.match_auto_invoke(prompt).await;
            assert!(matched.iter().any(|s| s.name == "memory-search"),
                "Prompt '{}' should match memory-search skill", prompt);
        }
    }

    #[tokio::test]
    async fn test_auto_invoke_negative_patterns() {
        let harness = SkillInvocationTestHarness::new().await;

        // Test patterns that should NOT match memory-search skill
        let should_not_match = [
            "write a function to add numbers",
            "create a new file",
            "run the tests",
        ];

        for prompt in should_not_match {
            let matched = harness.skill_registry.match_auto_invoke(prompt).await;
            let matches_memory = matched.iter().any(|s| s.name == "memory-search");
            // May or may not match depending on implementation
            // Main assertion is it doesn't crash
            assert!(matched.len() <= 5, "Should not match too many skills");
        }
    }

    #[tokio::test]
    async fn test_auto_invoke_regex_patterns() {
        let harness = SkillInvocationTestHarness::new().await;

        // Load a skill with regex auto_invoke
        let skill = harness.skill_loader.load_skill("drift-check").await.unwrap();

        if let Some(auto_invoke) = &skill.auto_invoke {
            // Verify the pattern compiles
            let regex = regex::Regex::new(auto_invoke);
            assert!(regex.is_ok(), "Auto-invoke pattern should be valid regex");
        }
    }

    #[tokio::test]
    async fn test_skill_priority_ordering() {
        let harness = SkillInvocationTestHarness::new().await;

        // Test prompt that matches multiple skills
        let prompt = "check goal alignment for authentication memory";
        let matched = harness.skill_registry.match_auto_invoke(prompt).await;

        // Skills should be ordered by match quality/priority
        if matched.len() > 1 {
            // First skill should be most relevant
            let first_skill = &matched[0];
            assert!(first_skill.match_score >= matched[1].match_score,
                "Skills should be ordered by match score");
        }
    }
}

mod mcp_invocation_tests {
    use super::*;

    #[tokio::test]
    async fn test_skill_invokes_mcp_tools() {
        let harness = SkillInvocationTestHarness::new().await;

        // Mock MCP client
        let mcp_client = MockMcpClient::new();

        // Invoke skill that uses MCP tools
        let result = harness.skill_registry.invoke_skill(
            "memory-search",
            json!({ "query": "authentication" }),
            &mcp_client,
        ).await;

        assert!(result.is_ok());

        // Verify MCP tool was called
        let calls = mcp_client.get_calls();
        assert!(!calls.is_empty(), "Should have invoked MCP tools");
        assert!(calls.iter().any(|c| c.tool_name.contains("search")));
    }

    #[tokio::test]
    async fn test_skill_mcp_tool_chaining() {
        let harness = SkillInvocationTestHarness::new().await;
        let mcp_client = MockMcpClient::new();

        // Invoke skill that chains multiple MCP tools
        let result = harness.skill_registry.invoke_skill(
            "context-injection",
            json!({
                "content": "Test content",
                "memory_type": "documentation"
            }),
            &mcp_client,
        ).await;

        assert!(result.is_ok());

        // Verify multiple MCP tools were called in sequence
        let calls = mcp_client.get_calls();
        assert!(calls.len() >= 2, "Should chain multiple MCP tools");
    }

    #[tokio::test]
    async fn test_skill_mcp_error_propagation() {
        let harness = SkillInvocationTestHarness::new().await;

        // Configure mock to fail
        let mcp_client = MockMcpClient::new();
        mcp_client.set_failure_mode(true);

        let result = harness.skill_registry.invoke_skill(
            "memory-search",
            json!({ "query": "test" }),
            &mcp_client,
        ).await;

        // Should propagate MCP error
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, SkillError::McpError(_)));
    }
}

mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_missing_skill_error() {
        let harness = SkillInvocationTestHarness::new().await;

        let result = harness.skill_loader.load_skill("nonexistent-skill").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SkillError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_invalid_yaml_error() {
        let harness = SkillInvocationTestHarness::new().await;

        // Attempt to load skill with invalid YAML
        let result = harness.skill_loader.load_skill_from_content(
            "invalid: yaml: content: [[["
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SkillError::ParseError(_)));
    }

    #[tokio::test]
    async fn test_missing_required_parameter() {
        let harness = SkillInvocationTestHarness::new().await;
        let mcp_client = MockMcpClient::new();

        // Invoke without required parameter
        let result = harness.skill_registry.invoke_skill(
            "memory-search",
            json!({}), // Missing "query" parameter
            &mcp_client,
        ).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SkillError::MissingParameter(_)));
    }

    #[tokio::test]
    async fn test_fallback_response() {
        let harness = SkillInvocationTestHarness::new().await;
        let mcp_client = MockMcpClient::new();
        mcp_client.set_failure_mode(true);

        // Some skills should have fallback responses
        let skill = harness.skill_loader.load_skill("memory-search").await.unwrap();

        if let Some(fallback) = &skill.fallback_response {
            // Fallback should be valid JSON or string
            assert!(!fallback.is_empty());
        }
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let harness = SkillInvocationTestHarness::new().await;
        let mcp_client = MockMcpClient::new();
        mcp_client.set_delay(Duration::from_secs(10)); // Slow response

        let result = timeout(
            Duration::from_secs(2),
            harness.skill_registry.invoke_skill(
                "memory-search",
                json!({ "query": "test" }),
                &mcp_client,
            )
        ).await;

        // Should timeout
        assert!(result.is_err() || result.unwrap().is_err());
    }
}
```
    </section>

    <section name="AgentLifecycleTests">
```rust
// tests/integration/agent_lifecycle_test.rs

//! Agent Lifecycle Tests
//!
//! Tests agent system:
//! - Agent spawning via Claude Code Task tool
//! - Inter-agent memory communication
//! - SubagentStop hook consolidation
//! - Agent coordination modes

use context_graph_mcp::agents::*;
use context_graph_mcp::hooks::*;
use std::sync::Arc;

struct AgentLifecycleTestHarness {
    orchestrator: Arc<SubagentOrchestrator>,
    hook_dispatcher: Arc<HookDispatcher>,
    store: Arc<TeleologicalStore>,
}

impl AgentLifecycleTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let hook_dispatcher = Arc::new(HookDispatcher::new(store.clone()).await);
        let orchestrator = Arc::new(SubagentOrchestrator::new(
            store.clone(),
            hook_dispatcher.clone(),
        ));

        Self { orchestrator, hook_dispatcher, store }
    }
}

mod task_tool_spawning_tests {
    use super::*;

    #[tokio::test]
    async fn test_task_tool_triggers_agent_spawn() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Simulate Task tool event from Claude Code
        let task_event = ClaudeCodeEvent {
            tool_name: "Task".to_string(),
            tool_input: json!({
                "description": "Analyze authentication patterns",
                "prompt": "You are a security researcher analyzing auth patterns...",
                "mode": "researcher"
            }),
            session_id: "test-session".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/project".to_string(),
                user: "user".to_string(),
                platform: "linux".to_string(),
            },
        };

        // Pre-hook should register pending agent
        let pre_result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&task_event)
            .await
            .unwrap();

        assert!(pre_result.injected_context.as_ref()
            .map(|c| c.get("agent_pending").is_some())
            .unwrap_or(false));

        // Post-hook should confirm agent spawned
        let post_result = harness.hook_dispatcher
            .dispatch_post_tool_use(&task_event, json!({"agent_id": "agent-123"}))
            .await
            .unwrap();

        assert!(post_result.injected_context.as_ref()
            .map(|c| c.get("agent_spawned").is_some())
            .unwrap_or(false));

        // Agent should be tracked
        let agents = harness.orchestrator.list_active_agents().await;
        assert!(!agents.is_empty());
    }

    #[tokio::test]
    async fn test_agent_inherits_session_context() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Start parent session
        harness.hook_dispatcher
            .dispatch_session_start("parent-session", json!({
                "project": "contextgraph",
                "goals": ["implement auth", "write tests"]
            }))
            .await
            .unwrap();

        // Spawn agent via Task tool
        let task_event = ClaudeCodeEvent {
            tool_name: "Task".to_string(),
            tool_input: json!({
                "description": "Write authentication tests",
                "mode": "tester"
            }),
            session_id: "parent-session".to_string(),
            environment: ClaudeCodeEnvironment {
                cwd: "/project".to_string(),
                user: "user".to_string(),
                platform: "linux".to_string(),
            },
        };

        let result = harness.hook_dispatcher
            .dispatch_pre_tool_use(&task_event)
            .await
            .unwrap();

        // Agent should inherit parent context
        let context = result.injected_context.unwrap();
        assert!(context.get("inherited_goals").is_some());
    }

    #[tokio::test]
    async fn test_multiple_agents_spawn_tracking() {
        let harness = AgentLifecycleTestHarness::new().await;

        let agent_types = ["researcher", "coder", "tester", "reviewer"];

        for agent_type in agent_types {
            let task_event = ClaudeCodeEvent {
                tool_name: "Task".to_string(),
                tool_input: json!({
                    "description": format!("Work as {}", agent_type),
                    "mode": agent_type
                }),
                session_id: "multi-agent-session".to_string(),
                environment: ClaudeCodeEnvironment {
                    cwd: "/project".to_string(),
                    user: "user".to_string(),
                    platform: "linux".to_string(),
                },
            };

            harness.hook_dispatcher
                .dispatch_pre_tool_use(&task_event)
                .await
                .unwrap();

            harness.hook_dispatcher
                .dispatch_post_tool_use(&task_event, json!({"success": true}))
                .await
                .unwrap();
        }

        let agents = harness.orchestrator.list_active_agents().await;
        assert_eq!(agents.len(), 4, "Should track all 4 agents");
    }
}

mod inter_agent_communication_tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_memory_sharing() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn two agents
        let researcher_id = harness.orchestrator
            .spawn_agent("researcher", "parent-session")
            .await
            .unwrap();

        let coder_id = harness.orchestrator
            .spawn_agent("coder", "parent-session")
            .await
            .unwrap();

        // Researcher stores finding
        harness.store.inject(InjectionRequest {
            content: "Authentication should use bcrypt for password hashing".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "research".to_string(),
            metadata: json!({ "agent_id": researcher_id }),
            ..Default::default()
        }).await.unwrap();

        // Coder should be able to find researcher's memory
        let search_results = harness.store.search(SearchRequest {
            query: "password hashing".to_string(),
            namespace: Some("research".to_string()),
            limit: 5,
            ..Default::default()
        }).await.unwrap();

        assert!(!search_results.is_empty());
        assert!(search_results[0].content.contains("bcrypt"));
    }

    #[tokio::test]
    async fn test_agent_message_passing() {
        let harness = AgentLifecycleTestHarness::new().await;

        let agent1_id = harness.orchestrator
            .spawn_agent("researcher", "session")
            .await
            .unwrap();

        let agent2_id = harness.orchestrator
            .spawn_agent("coder", "session")
            .await
            .unwrap();

        // Send message from researcher to coder
        let message = AgentMessage {
            from: agent1_id.clone(),
            to: agent2_id.clone(),
            message_type: MessageType::Request,
            payload: json!({
                "type": "implementation_request",
                "finding": "Use bcrypt for passwords",
                "priority": "high"
            }),
        };

        harness.orchestrator.send_message(message).await.unwrap();

        // Coder should receive message
        let messages = harness.orchestrator
            .get_pending_messages(&agent2_id)
            .await
            .unwrap();

        assert!(!messages.is_empty());
        assert_eq!(messages[0].from, agent1_id);
    }

    #[tokio::test]
    async fn test_agent_broadcast_channel() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn multiple agents
        let agent_ids: Vec<_> = futures::future::join_all(
            ["researcher", "coder", "tester"].iter().map(|t| {
                harness.orchestrator.spawn_agent(t, "session")
            })
        ).await.into_iter().map(|r| r.unwrap()).collect();

        // Broadcast message to all agents
        let broadcast = AgentBroadcast {
            from: agent_ids[0].clone(),
            channel: "all".to_string(),
            payload: json!({
                "type": "announcement",
                "content": "Goal updated: prioritize security"
            }),
        };

        harness.orchestrator.broadcast(broadcast).await.unwrap();

        // All agents should receive broadcast
        for agent_id in &agent_ids[1..] {
            let messages = harness.orchestrator
                .get_pending_messages(agent_id)
                .await
                .unwrap();

            assert!(!messages.is_empty(), "Agent {} should receive broadcast", agent_id);
        }
    }
}

mod subagent_stop_tests {
    use super::*;

    #[tokio::test]
    async fn test_subagent_stop_triggers_consolidation() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn agent
        let agent_id = harness.orchestrator
            .spawn_agent("researcher", "session")
            .await
            .unwrap();

        // Agent stores findings
        for i in 0..10 {
            harness.store.inject(InjectionRequest {
                content: format!("Research finding {}", i),
                memory_type: MemoryType::CodeContext,
                namespace: format!("agent-{}", agent_id),
                ..Default::default()
            }).await.unwrap();
        }

        // Stop agent (triggers SubagentStop hook)
        let stop_result = harness.orchestrator
            .stop_agent(&agent_id)
            .await
            .unwrap();

        // Should consolidate agent memories
        assert!(stop_result.consolidated_count >= 10);
        assert!(stop_result.summary.is_some());
    }

    #[tokio::test]
    async fn test_subagent_stop_transfers_to_parent() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Start parent session
        harness.hook_dispatcher
            .dispatch_session_start("parent-session", json!({}))
            .await
            .unwrap();

        // Spawn child agent
        let agent_id = harness.orchestrator
            .spawn_agent("researcher", "parent-session")
            .await
            .unwrap();

        // Agent creates knowledge
        harness.store.inject(InjectionRequest {
            content: "Important discovery: use JWT for tokens".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: format!("agent-{}", agent_id),
            ..Default::default()
        }).await.unwrap();

        // Stop agent
        harness.orchestrator.stop_agent(&agent_id).await.unwrap();

        // Parent session should have access to agent's key findings
        let parent_memories = harness.store.search(SearchRequest {
            query: "JWT tokens".to_string(),
            namespace: Some("parent-session".to_string()),
            include_consolidated: true,
            ..Default::default()
        }).await.unwrap();

        assert!(!parent_memories.is_empty(), "Parent should inherit agent findings");
    }

    #[tokio::test]
    async fn test_subagent_stop_cleanup() {
        let harness = AgentLifecycleTestHarness::new().await;

        let agent_id = harness.orchestrator
            .spawn_agent("coder", "session")
            .await
            .unwrap();

        // Verify agent exists
        let agents = harness.orchestrator.list_active_agents().await;
        assert!(agents.iter().any(|a| a.id == agent_id));

        // Stop agent
        harness.orchestrator.stop_agent(&agent_id).await.unwrap();

        // Agent should be removed from active list
        let agents = harness.orchestrator.list_active_agents().await;
        assert!(!agents.iter().any(|a| a.id == agent_id));
    }
}

mod coordination_mode_tests {
    use super::*;

    #[tokio::test]
    async fn test_hierarchical_coordination() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn coordinator and workers
        let coordinator = harness.orchestrator
            .spawn_agent("coordinator", "session")
            .await
            .unwrap();

        let workers: Vec<_> = futures::future::join_all(
            (0..3).map(|_| {
                harness.orchestrator.spawn_agent("coder", "session")
            })
        ).await.into_iter().map(|r| r.unwrap()).collect();

        // Set up hierarchy
        harness.orchestrator.set_coordination_mode(
            CoordinationMode::Hierarchical {
                coordinator: coordinator.clone(),
                workers: workers.clone(),
            }
        ).await.unwrap();

        // Coordinator assigns task
        let task = AgentTask {
            id: "task-1".to_string(),
            description: "Implement login".to_string(),
            assigned_by: coordinator.clone(),
            assigned_to: workers[0].clone(),
        };

        harness.orchestrator.assign_task(task).await.unwrap();

        // Worker should have task
        let worker_tasks = harness.orchestrator
            .get_agent_tasks(&workers[0])
            .await
            .unwrap();

        assert!(!worker_tasks.is_empty());
    }

    #[tokio::test]
    async fn test_mesh_coordination() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn peer agents
        let peers: Vec<_> = futures::future::join_all(
            ["researcher", "coder", "tester"].iter().map(|t| {
                harness.orchestrator.spawn_agent(t, "session")
            })
        ).await.into_iter().map(|r| r.unwrap()).collect();

        // Set up mesh
        harness.orchestrator.set_coordination_mode(
            CoordinationMode::Mesh { peers: peers.clone() }
        ).await.unwrap();

        // Any peer can communicate with any other
        for i in 0..peers.len() {
            for j in 0..peers.len() {
                if i != j {
                    let can_communicate = harness.orchestrator
                        .can_communicate(&peers[i], &peers[j])
                        .await;
                    assert!(can_communicate, "Mesh peers should be able to communicate");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_swarm_coordination() {
        let harness = AgentLifecycleTestHarness::new().await;

        // Spawn swarm agents
        let agents: Vec<_> = futures::future::join_all(
            (0..5).map(|i| {
                harness.orchestrator.spawn_agent(
                    if i == 0 { "coordinator" } else { "coder" },
                    "session"
                )
            })
        ).await.into_iter().map(|r| r.unwrap()).collect();

        // Set up swarm with queen
        harness.orchestrator.set_coordination_mode(
            CoordinationMode::Swarm {
                queen: agents[0].clone(),
                workers: agents[1..].to_vec(),
                consensus: ConsensusMode::Byzantine,
            }
        ).await.unwrap();

        // Verify swarm structure
        let swarm_status = harness.orchestrator.get_swarm_status().await.unwrap();
        assert_eq!(swarm_status.queen, agents[0]);
        assert_eq!(swarm_status.workers.len(), 4);
    }
}
```
    </section>

    <section name="HooksIntegrationTests">
```rust
// crates/context-graph-mcp/tests/hooks_integration.rs

use context_graph_mcp::hooks::*;
use context_graph_mcp::test_utils::*;
use context_graph_storage::teleological::TeleologicalStore;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

/// Test harness for hook integration tests
struct HookTestHarness {
    store: Arc<TeleologicalStore>,
    hook_registry: Arc<HookRegistry>,
    pre_task_handler: PreTaskHandler,
    post_task_handler: PostTaskHandler,
    session_handler: SessionHandler,
    pre_file_write_handler: PreFileWriteHandler,
    post_file_write_handler: PostFileWriteHandler,
    pre_file_read_handler: PreFileReadHandler,
    post_file_read_handler: PostFileReadHandler,
    pre_bash_exec_handler: PreBashExecHandler,
    post_bash_exec_handler: PostBashExecHandler,
}

impl HookTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let hook_registry = Arc::new(HookRegistry::new());

        // Create all handlers
        Self {
            store: store.clone(),
            hook_registry: hook_registry.clone(),
            pre_task_handler: PreTaskHandler::new(store.clone(), Default::default()),
            post_task_handler: PostTaskHandler::new(store.clone(), Default::default()),
            session_handler: SessionHandler::new(store.clone(), Default::default()),
            pre_file_write_handler: PreFileWriteHandler::new(
                store.clone(),
                Arc::new(TeleologicalAlignmentCalculator::new()),
                Arc::new(TeleologicalDriftDetector::new()),
                Arc::new(EmbedderPipeline::new().await.unwrap()),
                Default::default(),
            ),
            post_file_write_handler: PostFileWriteHandler::new(
                store.clone(),
                Arc::new(EmbedderPipeline::new().await.unwrap()),
                Arc::new(TrajectoryTracker::new()),
                Arc::new(EditPatternExtractor::new()),
                Arc::new(BackgroundTaskQueue::new()),
                Default::default(),
            ),
            pre_file_read_handler: PreFileReadHandler::new(
                store.clone(),
                Arc::new(SearchEngine::new(store.clone())),
                Arc::new(FileAccessTracker::new(store.clone())),
                Default::default(),
            ),
            post_file_read_handler: PostFileReadHandler::new(
                store.clone(),
                Arc::new(EmbedderPipeline::new().await.unwrap()),
                Default::default(),
            ),
            pre_bash_exec_handler: PreBashExecHandler::new(
                Arc::new(CommandSafetyEngine::new()),
                store.clone(),
                Default::default(),
            ),
            post_bash_exec_handler: PostBashExecHandler::new(
                store.clone(),
                Arc::new(EmbedderPipeline::new().await.unwrap()),
                Arc::new(TrajectoryTracker::new()),
                Arc::new(CommandPatternLearner::new(store.clone())),
                Default::default(),
            ),
        }
    }
}

mod session_hooks {
    use super::*;

    #[tokio::test]
    async fn test_session_start_initializes_context() {
        let harness = HookTestHarness::new().await;

        let payload = SessionPayload {
            session_id: "test-session-001".to_string(),
            user_context: Some(serde_json::json!({"project": "test"})),
        };

        let response = harness.session_handler.handle_start(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Success);
        assert!(response.injected_context.is_some());

        // Verify session was stored
        let stored = harness.store
            .get_session("test-session-001")
            .await
            .unwrap();
        assert!(stored.is_some());
    }

    #[tokio::test]
    async fn test_session_end_triggers_consolidation() {
        let harness = HookTestHarness::new().await;

        // Start session first
        let start_payload = SessionPayload {
            session_id: "test-session-002".to_string(),
            user_context: None,
        };
        harness.session_handler.handle_start(start_payload).await.unwrap();

        // Inject some memories
        for i in 0..10 {
            harness.store.inject(InjectionRequest {
                content: format!("Test content {}", i),
                memory_type: MemoryType::General,
                namespace: "test".to_string(),
                ..Default::default()
            }).await.unwrap();
        }

        // End session
        let end_response = harness.session_handler
            .handle_end("test-session-002".to_string())
            .await
            .unwrap();

        assert_eq!(end_response.status, HookStatus::Success);

        // Verify consolidation occurred
        let context = end_response.injected_context.unwrap();
        assert!(context.get("consolidated_count").is_some());
    }

    #[tokio::test]
    async fn test_session_restore_loads_context() {
        let harness = HookTestHarness::new().await;

        // Create and end a session
        let session_id = "test-session-003".to_string();
        harness.session_handler.handle_start(SessionPayload {
            session_id: session_id.clone(),
            user_context: Some(serde_json::json!({"key": "value"})),
        }).await.unwrap();
        harness.session_handler.handle_end(session_id.clone()).await.unwrap();

        // Restore session
        let restore_response = harness.session_handler
            .handle_restore(session_id.clone())
            .await
            .unwrap();

        assert_eq!(restore_response.status, HookStatus::Success);
        let context = restore_response.injected_context.unwrap();
        assert_eq!(context.get("key").unwrap(), "value");
    }
}

mod file_write_hooks {
    use super::*;

    #[tokio::test]
    async fn test_pre_file_write_validates_alignment() {
        let harness = HookTestHarness::new().await;

        // Create a goal first
        let goal_array = create_test_teleological_array("authentication system").await;
        harness.store.store_goal(GoalNode {
            goal_id: "goal-001".to_string(),
            label: "Implement authentication".to_string(),
            level: GoalLevel::Strategic,
            teleological_array: goal_array,
            ..Default::default()
        }).await.unwrap();

        // Test aligned write
        let aligned_payload = FileWritePayload {
            file_path: "src/auth/handler.rs".to_string(),
            new_content: "fn verify_token(token: &str) -> Result<Claims, AuthError>".to_string(),
            original_content: None,
            operation: FileOperation::Create,
            source_tool: "Write".to_string(),
        };

        let response = harness.pre_file_write_handler.handle(aligned_payload).await.unwrap();
        assert!(matches!(response.status, HookStatus::Proceed | HookStatus::ProceedWithWarning));

        // Test misaligned write
        let misaligned_payload = FileWritePayload {
            file_path: "src/random/unrelated.rs".to_string(),
            new_content: "fn calculate_tax(amount: f64) -> f64".to_string(),
            original_content: None,
            operation: FileOperation::Create,
            source_tool: "Write".to_string(),
        };

        let response = harness.pre_file_write_handler.handle(misaligned_payload).await.unwrap();
        // Should have warnings about low alignment
        assert!(!response.warnings.is_empty() || response.status == HookStatus::ProceedWithWarning);
    }

    #[tokio::test]
    async fn test_post_file_write_stores_memory() {
        let harness = HookTestHarness::new().await;

        let payload = FileWritePayload {
            file_path: "src/test.rs".to_string(),
            new_content: r#"
                pub fn test_function() {
                    println!("Hello, world!");
                }
            "#.to_string(),
            original_content: None,
            operation: FileOperation::Create,
            source_tool: "Write".to_string(),
        };

        let response = harness.post_file_write_handler.handle(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Success);
        let context = response.injected_context.unwrap();
        let memory_id = context.get("memory_id").unwrap().as_str().unwrap();

        // Verify memory was stored
        let stored = harness.store.get_memory(memory_id.parse().unwrap()).await.unwrap();
        assert!(stored.is_some());
    }

    #[tokio::test]
    async fn test_post_file_write_triggers_pattern_learning() {
        let harness = HookTestHarness::new().await;

        // Write with before content for pattern extraction
        let payload = FileWritePayload {
            file_path: "src/refactor.rs".to_string(),
            new_content: r#"
                pub fn get_user_by_id(id: UserId) -> Result<User, Error> {
                    self.repository.find_by_id(id)
                }
            "#.to_string(),
            original_content: Some(r#"
                pub fn get_user(id: i64) -> User {
                    self.users.get(&id).unwrap()
                }
            "#.to_string()),
            operation: FileOperation::Update,
            source_tool: "Edit".to_string(),
        };

        let response = harness.post_file_write_handler.handle(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Success);
        let context = response.injected_context.unwrap();
        assert!(context.get("pattern_training_queued").is_some());
    }

    #[tokio::test]
    async fn test_pre_file_write_latency_under_threshold() {
        let harness = HookTestHarness::new().await;

        let payload = FileWritePayload {
            file_path: "src/test.rs".to_string(),
            new_content: "fn test() {}".to_string(),
            original_content: None,
            operation: FileOperation::Create,
            source_tool: "Write".to_string(),
        };

        let start = std::time::Instant::now();
        let response = harness.pre_file_write_handler.handle(payload).await.unwrap();
        let elapsed = start.elapsed();

        // Pre-write should be under 100ms
        assert!(elapsed.as_millis() < 100, "Pre-write took {}ms", elapsed.as_millis());
        assert!(response.metrics.latency_ms < 100);
    }
}

mod file_read_hooks {
    use super::*;

    #[tokio::test]
    async fn test_pre_file_read_injects_context() {
        let harness = HookTestHarness::new().await;

        // Store some related content first
        harness.store.inject(InjectionRequest {
            content: "Authentication handler for JWT tokens".to_string(),
            memory_type: MemoryType::CodeContext,
            namespace: "auth".to_string(),
            metadata: serde_json::json!({"file_path": "src/auth/jwt.rs"}),
            ..Default::default()
        }).await.unwrap();

        let payload = FileReadPayload {
            file_path: "src/auth/handler.rs".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };

        let response = harness.pre_file_read_handler.handle(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Proceed);
        if let Some(context) = response.injected_context {
            // Should find related memories
            let related = context.get("related_memories").and_then(|r| r.as_array());
            // May or may not find related content depending on embedding similarity
        }
    }

    #[tokio::test]
    async fn test_post_file_read_embeds_content() {
        let harness = HookTestHarness::new().await;

        let payload = PostFileReadPayload {
            file_path: "src/test.rs".to_string(),
            content: r#"
                /// Test module for authentication
                pub mod auth {
                    pub fn verify(token: &str) -> bool {
                        !token.is_empty()
                    }
                }
            "#.to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };

        let response = harness.post_file_read_handler.handle(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Success);
        let context = response.injected_context.unwrap();
        assert!(context.get("memory_id").is_some());
        assert_eq!(context.get("embedders_generated").unwrap(), 13);
    }

    #[tokio::test]
    async fn test_post_file_read_deduplicates() {
        let harness = HookTestHarness::new().await;

        let content = "fn duplicate_content() {}".to_string();

        // First read
        let payload1 = PostFileReadPayload {
            file_path: "src/test.rs".to_string(),
            content: content.clone(),
            line_range: None,
            source_tool: "Read".to_string(),
        };
        let response1 = harness.post_file_read_handler.handle(payload1).await.unwrap();
        let memory_id1 = response1.injected_context.unwrap()
            .get("memory_id").unwrap().as_str().unwrap().to_string();

        // Second read of same content
        let payload2 = PostFileReadPayload {
            file_path: "src/test.rs".to_string(),
            content,
            line_range: None,
            source_tool: "Read".to_string(),
        };
        let response2 = harness.post_file_read_handler.handle(payload2).await.unwrap();

        // Should be deduplicated
        let context = response2.injected_context.unwrap();
        assert_eq!(context.get("deduplicated").unwrap(), true);
    }

    #[tokio::test]
    async fn test_pre_file_read_latency_under_threshold() {
        let harness = HookTestHarness::new().await;

        let payload = FileReadPayload {
            file_path: "src/test.rs".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };

        let start = std::time::Instant::now();
        let response = harness.pre_file_read_handler.handle(payload).await.unwrap();
        let elapsed = start.elapsed();

        // Pre-read should be under 50ms
        assert!(elapsed.as_millis() < 50, "Pre-read took {}ms", elapsed.as_millis());
    }
}

mod bash_hooks {
    use super::*;

    #[tokio::test]
    async fn test_pre_bash_blocks_dangerous_commands() {
        let harness = HookTestHarness::new().await;

        // Dangerous command
        let payload = BashExecPayload {
            command: "rm -rf /".to_string(),
            working_dir: None,
            env_vars: None,
            timeout_ms: None,
        };

        let response = harness.pre_bash_exec_handler.handle(payload).await.unwrap();
        assert_eq!(response.status, HookStatus::Block);
        assert!(!response.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_pre_bash_allows_safe_commands() {
        let harness = HookTestHarness::new().await;

        let payload = BashExecPayload {
            command: "cargo test".to_string(),
            working_dir: None,
            env_vars: None,
            timeout_ms: None,
        };

        let response = harness.pre_bash_exec_handler.handle(payload).await.unwrap();
        assert!(matches!(response.status, HookStatus::Proceed | HookStatus::ProceedWithWarning));
    }

    #[tokio::test]
    async fn test_pre_bash_warns_on_medium_risk() {
        let harness = HookTestHarness::new().await;

        let payload = BashExecPayload {
            command: "git push --force".to_string(),
            working_dir: None,
            env_vars: None,
            timeout_ms: None,
        };

        let response = harness.pre_bash_exec_handler.handle(payload).await.unwrap();
        assert_eq!(response.status, HookStatus::ProceedWithWarning);
        assert!(response.warnings.iter().any(|w| w.contains("force")));
    }

    #[tokio::test]
    async fn test_post_bash_stores_output() {
        let harness = HookTestHarness::new().await;

        let payload = PostBashExecPayload {
            command: "cargo test".to_string(),
            exit_code: 0,
            stdout: "running 5 tests\ntest result: ok. 5 passed".to_string(),
            stderr: "".to_string(),
            duration_ms: 1500,
            working_dir: Some("/project".to_string()),
        };

        let response = harness.post_bash_exec_handler.handle(payload).await.unwrap();

        assert_eq!(response.status, HookStatus::Success);
        let context = response.injected_context.unwrap();
        assert!(context.get("memory_id").is_some());
        assert_eq!(context.get("success").unwrap(), true);
    }

    #[tokio::test]
    async fn test_post_bash_extracts_error_patterns() {
        let harness = HookTestHarness::new().await;

        let payload = PostBashExecPayload {
            command: "cargo build".to_string(),
            exit_code: 1,
            stdout: "".to_string(),
            stderr: "error[E0425]: cannot find value `foo` in this scope".to_string(),
            duration_ms: 500,
            working_dir: None,
        };

        let response = harness.post_bash_exec_handler.handle(payload).await.unwrap();

        let context = response.injected_context.unwrap();
        assert!(context.get("error_patterns").is_some());
        // Should include troubleshooting suggestions
    }

    #[tokio::test]
    async fn test_pre_bash_latency_under_threshold() {
        let harness = HookTestHarness::new().await;

        let payload = BashExecPayload {
            command: "echo hello".to_string(),
            working_dir: None,
            env_vars: None,
            timeout_ms: None,
        };

        let start = std::time::Instant::now();
        let response = harness.pre_bash_exec_handler.handle(payload).await.unwrap();
        let elapsed = start.elapsed();

        // Pre-bash should be under 20ms
        assert!(elapsed.as_millis() < 20, "Pre-bash took {}ms", elapsed.as_millis());
    }
}

mod hook_chaining {
    use super::*;

    #[tokio::test]
    async fn test_session_start_to_file_read_chain() {
        let harness = HookTestHarness::new().await;

        // 1. Start session
        let session_response = harness.session_handler.handle_start(SessionPayload {
            session_id: "chain-test-001".to_string(),
            user_context: None,
        }).await.unwrap();
        assert_eq!(session_response.status, HookStatus::Success);

        // 2. Read a file
        let read_payload = FileReadPayload {
            file_path: "src/main.rs".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };
        let read_response = harness.pre_file_read_handler.handle(read_payload).await.unwrap();
        assert_eq!(read_response.status, HookStatus::Proceed);

        // 3. Post read to store
        let post_read = PostFileReadPayload {
            file_path: "src/main.rs".to_string(),
            content: "fn main() { println!(\"Hello\"); }".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };
        let post_read_response = harness.post_file_read_handler.handle(post_read).await.unwrap();
        assert_eq!(post_read_response.status, HookStatus::Success);

        // Verify memory was stored in session context
        let memory_id = post_read_response.injected_context.unwrap()
            .get("memory_id").unwrap().as_str().unwrap().to_string();

        let session_memories = harness.store
            .get_session_memories("chain-test-001")
            .await
            .unwrap();
        // Memory should be associated with session
    }

    #[tokio::test]
    async fn test_file_write_to_goal_alignment_chain() {
        let harness = HookTestHarness::new().await;

        // 1. Create a goal
        let goal_array = create_test_teleological_array("user authentication").await;
        harness.store.store_goal(GoalNode {
            goal_id: "chain-goal-001".to_string(),
            label: "Implement user auth".to_string(),
            level: GoalLevel::Strategic,
            teleological_array: goal_array,
            ..Default::default()
        }).await.unwrap();

        // 2. Pre-write check
        let pre_write = FileWritePayload {
            file_path: "src/auth/login.rs".to_string(),
            new_content: "pub fn login(credentials: Credentials) -> Result<Token, AuthError>".to_string(),
            original_content: None,
            operation: FileOperation::Create,
            source_tool: "Write".to_string(),
        };
        let pre_response = harness.pre_file_write_handler.handle(pre_write.clone()).await.unwrap();

        // Should check alignment
        let context = pre_response.injected_context.unwrap();
        assert!(context.get("alignment_scores").is_some());

        // 3. Post-write storage
        let post_response = harness.post_file_write_handler.handle(pre_write).await.unwrap();
        assert_eq!(post_response.status, HookStatus::Success);
    }
}

mod error_handling {
    use super::*;

    #[tokio::test]
    async fn test_hook_timeout_handling() {
        let harness = HookTestHarness::new().await;

        // Configure very short timeout
        let mut config = PreFileReadConfig::default();
        config.max_preload_latency_ms = 1; // 1ms timeout

        let handler = PreFileReadHandler::new(
            harness.store.clone(),
            Arc::new(SearchEngine::new(harness.store.clone())),
            Arc::new(FileAccessTracker::new(harness.store.clone())),
            config,
        );

        let payload = FileReadPayload {
            file_path: "src/test.rs".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };

        // Should complete without error even with timeout
        let response = handler.handle(payload).await.unwrap();
        assert_eq!(response.status, HookStatus::Proceed);
    }

    #[tokio::test]
    async fn test_hook_graceful_degradation() {
        // Test that hooks degrade gracefully when store is unavailable
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());

        // Simulate store being temporarily unavailable by dropping it
        // In practice, handlers should catch errors and return degraded responses

        let payload = FileReadPayload {
            file_path: "src/test.rs".to_string(),
            line_range: None,
            source_tool: "Read".to_string(),
        };

        // Handler should not panic, should return safe default
    }
}
```
    </section>

    <section name="SkillsIntegrationTests">
```rust
// crates/context-graph-mcp/tests/skills_integration.rs

use context_graph_mcp::skills::*;
use context_graph_mcp::test_utils::*;

struct SkillTestHarness {
    skill_loader: Arc<SkillLoader>,
    mcp_client: Arc<MockMcpClient>,
    store: Arc<TeleologicalStore>,
}

impl SkillTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let mcp_client = Arc::new(MockMcpClient::new(store.clone()));
        let skills_dir = PathBuf::from(".claude/skills");
        let skill_loader = Arc::new(SkillLoader::new(skills_dir, mcp_client.clone()));

        Self {
            skill_loader,
            mcp_client,
            store,
        }
    }
}

#[tokio::test]
async fn test_skill_loader_discovers_skills() {
    let harness = SkillTestHarness::new().await;

    let count = harness.skill_loader.load_all().await.unwrap();

    // Should find at least the 5 core skills
    assert!(count >= 5, "Expected at least 5 skills, found {}", count);

    let skills = harness.skill_loader.list().await;
    assert!(skills.contains(&"memory-search".to_string()));
    assert!(skills.contains(&"goal-alignment".to_string()));
    assert!(skills.contains(&"pattern-learning".to_string()));
    assert!(skills.contains(&"context-injection".to_string()));
    assert!(skills.contains(&"drift-check".to_string()));
}

#[tokio::test]
async fn test_memory_search_skill() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Inject some test content
    harness.store.inject(InjectionRequest {
        content: "JWT token validation and refresh logic".to_string(),
        memory_type: MemoryType::CodeContext,
        namespace: "auth".to_string(),
        ..Default::default()
    }).await.unwrap();

    // Invoke skill
    let result = harness.skill_loader.invoke(
        "memory-search",
        serde_json::json!({
            "query": "authentication tokens",
            "limit": 5
        }),
    ).await.unwrap();

    assert!(result.success);
    let data = result.data;
    assert!(data.get("found").is_some());
}

#[tokio::test]
async fn test_goal_alignment_skill() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Create a goal
    let goal_array = create_test_teleological_array("implement caching").await;
    harness.store.store_goal(GoalNode {
        goal_id: "cache-goal-001".to_string(),
        label: "Implement caching layer".to_string(),
        level: GoalLevel::Strategic,
        teleological_array: goal_array,
        ..Default::default()
    }).await.unwrap();

    // Check alignment
    let result = harness.skill_loader.invoke(
        "goal-alignment",
        serde_json::json!({
            "content": "Adding Redis cache for session storage",
            "goal_id": "cache-goal-001"
        }),
    ).await.unwrap();

    assert!(result.success);
    let data = result.data;
    assert!(data.get("score").is_some());
    assert!(data.get("status").is_some());
}

#[tokio::test]
async fn test_pattern_learning_skill() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    let result = harness.skill_loader.invoke(
        "pattern-learning",
        serde_json::json!({
            "pattern_type": "code",
            "content": "pub fn handle_error<E: Error>(e: E) -> Response { ... }",
            "context": {
                "language": "rust",
                "framework": "actix-web"
            },
            "tags": ["error-handling", "web"]
        }),
    ).await.unwrap();

    assert!(result.success);
    let data = result.data;
    assert!(data.get("patternId").is_some());
}

#[tokio::test]
async fn test_context_injection_skill() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    let result = harness.skill_loader.invoke(
        "context-injection",
        serde_json::json!({
            "content": "Important context about the authentication flow",
            "memory_type": "documentation",
            "namespace": "auth"
        }),
    ).await.unwrap();

    assert!(result.success);
    let data = result.data;
    assert!(data.get("memoryId").is_some());
    assert_eq!(data.get("embeddings").unwrap(), 13);
}

#[tokio::test]
async fn test_drift_check_skill() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Create goal and memories
    let goal_array = create_test_teleological_array("payment processing").await;
    harness.store.store_goal(GoalNode {
        goal_id: "payment-goal-001".to_string(),
        label: "Implement payment processing".to_string(),
        level: GoalLevel::Strategic,
        teleological_array: goal_array,
        ..Default::default()
    }).await.unwrap();

    let memory_id = harness.store.inject(InjectionRequest {
        content: "Shopping cart implementation".to_string(),
        memory_type: MemoryType::CodeContext,
        namespace: "commerce".to_string(),
        ..Default::default()
    }).await.unwrap();

    let result = harness.skill_loader.invoke(
        "drift-check",
        serde_json::json!({
            "memory_ids": [memory_id.to_string()],
            "goal_id": "payment-goal-001"
        }),
    ).await.unwrap();

    assert!(result.success);
    let data = result.data;
    assert!(data.get("status").is_some());
    assert!(data.get("hasDrifted").is_some());
}

#[tokio::test]
async fn test_skill_error_handling_and_retry() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Configure mock to fail first two times
    harness.mcp_client.set_failure_count(2);

    let result = harness.skill_loader.invoke(
        "memory-search",
        serde_json::json!({
            "query": "test query"
        }),
    ).await;

    // Should succeed after retries
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_skill_parameter_validation() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Missing required parameter
    let result = harness.skill_loader.invoke(
        "memory-search",
        serde_json::json!({}), // Missing 'query'
    ).await;

    assert!(result.is_err());
    match result {
        Err(SkillError::MissingParam(param)) => {
            assert_eq!(param, "query");
        }
        _ => panic!("Expected MissingParam error"),
    }
}

#[tokio::test]
async fn test_skill_default_parameters() {
    let harness = SkillTestHarness::new().await;
    harness.skill_loader.load_all().await.unwrap();

    // Only provide required parameter
    let result = harness.skill_loader.invoke(
        "memory-search",
        serde_json::json!({
            "query": "test"
        }),
    ).await.unwrap();

    // Should use defaults for limit, threshold, etc.
    assert!(result.success);
}
```
    </section>

    <section name="AgentsIntegrationTests">
```rust
// crates/context-graph-mcp/tests/agents_integration.rs

use context_graph_mcp::agents::*;
use context_graph_mcp::test_utils::*;

struct AgentTestHarness {
    orchestrator: Arc<SubagentOrchestrator>,
    store: Arc<TeleologicalStore>,
}

impl AgentTestHarness {
    async fn new() -> Self {
        let store = Arc::new(TeleologicalStore::in_memory().await.unwrap());
        let skill_loader = Arc::new(SkillLoader::new(
            PathBuf::from(".claude/skills"),
            Arc::new(MockMcpClient::new(store.clone())),
        ));

        let orchestrator = Arc::new(SubagentOrchestrator::new(
            store.clone(),
            skill_loader,
            OrchestratorConfig {
                agents_dir: PathBuf::from(".claude/agents"),
                ..Default::default()
            },
        ));

        Self { orchestrator, store }
    }
}

#[tokio::test]
async fn test_agent_lifecycle_start_stop() {
    let harness = AgentTestHarness::new().await;

    // Start agent
    let agent_id = harness.orchestrator.start_agent("goal-tracker").await.unwrap();
    assert!(!agent_id.is_empty());

    // Check status
    let status = harness.orchestrator.get_status("goal-tracker").await.unwrap();
    assert!(matches!(status.status, AgentStatus::Running | AgentStatus::Starting));

    // Stop agent
    harness.orchestrator.stop_agent("goal-tracker").await.unwrap();

    // Should no longer be running
    let result = harness.orchestrator.get_status("goal-tracker").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_agent_list_available() {
    let harness = AgentTestHarness::new().await;

    let definitions = harness.orchestrator.load_definitions().await.unwrap();

    // Should find all 4 agents
    assert!(definitions.len() >= 4);

    let names: Vec<_> = definitions.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"goal-tracker"));
    assert!(names.contains(&"context-curator"));
    assert!(names.contains(&"pattern-miner"));
    assert!(names.contains(&"learning-coach"));
}

#[tokio::test]
async fn test_inter_agent_communication() {
    let harness = AgentTestHarness::new().await;

    // Start two agents
    harness.orchestrator.start_agent("goal-tracker").await.unwrap();
    harness.orchestrator.start_agent("context-curator").await.unwrap();

    // Wait for startup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send message from goal-tracker to context-curator
    let message = AgentMessage {
        message_id: Uuid::new_v4(),
        from: "goal-tracker".to_string(),
        to: "context-curator".to_string(),
        message_type: MessageType::Query,
        payload: serde_json::json!({"query": "recent file changes"}),
        timestamp: Utc::now(),
    };

    harness.orchestrator.send_message("context-curator", message).await.unwrap();

    // Clean up
    harness.orchestrator.stop_agent("goal-tracker").await.unwrap();
    harness.orchestrator.stop_agent("context-curator").await.unwrap();
}

#[tokio::test]
async fn test_agent_health_monitoring() {
    let harness = AgentTestHarness::new().await;

    // Start an agent
    harness.orchestrator.start_agent("pattern-miner").await.unwrap();

    // Check health
    let health = harness.orchestrator.health_check().await;
    assert!(!health.is_empty());

    let agent_health = health.iter().find(|h| h.name == "pattern-miner").unwrap();
    assert!(agent_health.healthy);
    assert!(agent_health.issues.is_empty());

    harness.orchestrator.stop_agent("pattern-miner").await.unwrap();
}

#[tokio::test]
async fn test_max_agents_limit() {
    let harness = AgentTestHarness::new().await;

    // Try to start more agents than allowed
    for agent in &["goal-tracker", "context-curator", "pattern-miner", "learning-coach"] {
        let _ = harness.orchestrator.start_agent(agent).await;
    }

    // Depending on max_agents setting, some may fail
    let running = harness.orchestrator.list_agents().await;
    assert!(running.len() <= 10); // Default max

    // Clean up
    for info in running {
        let _ = harness.orchestrator.stop_agent(&info.name).await;
    }
}

#[tokio::test]
async fn test_agent_store_integration() {
    let harness = AgentTestHarness::new().await;

    // Start goal-tracker
    harness.orchestrator.start_agent("goal-tracker").await.unwrap();

    // Inject a memory
    harness.store.inject(InjectionRequest {
        content: "New authentication feature completed".to_string(),
        memory_type: MemoryType::CodeContext,
        namespace: "auth".to_string(),
        ..Default::default()
    }).await.unwrap();

    // Goal tracker should be able to search and analyze
    // (In practice, this would happen via message passing)

    harness.orchestrator.stop_agent("goal-tracker").await.unwrap();
}

#[tokio::test]
async fn test_agent_skill_invocation() {
    let harness = AgentTestHarness::new().await;

    // Start context-curator which uses memory-search skill
    harness.orchestrator.start_agent("context-curator").await.unwrap();

    // Send task message
    let message = AgentMessage {
        message_id: Uuid::new_v4(),
        from: "test".to_string(),
        to: "context-curator".to_string(),
        message_type: MessageType::Task,
        payload: serde_json::json!({
            "skill": "memory-search",
            "query": "authentication"
        }),
        timestamp: Utc::now(),
    };

    harness.orchestrator.send_message("context-curator", message).await.unwrap();

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    harness.orchestrator.stop_agent("context-curator").await.unwrap();
}
```
    </section>

    <section name="FullIntegrationTests">
```rust
// crates/context-graph-mcp/tests/full_integration.rs

use context_graph_mcp::*;

/// Full system integration test
#[tokio::test]
async fn test_complete_session_workflow() {
    let system = IntegrationTestSystem::new().await;

    // 1. Start session
    let session_id = system.start_session().await.unwrap();

    // 2. Read some files (triggers hooks)
    let file_content = system.read_file("src/auth/handler.rs").await.unwrap();
    assert!(system.memory_count().await > 0);

    // 3. Write a file (triggers alignment check and storage)
    system.write_file(
        "src/auth/validator.rs",
        "pub fn validate(token: &str) -> bool { !token.is_empty() }",
    ).await.unwrap();

    // 4. Run a command (triggers safety check and learning)
    let output = system.run_command("cargo test --lib").await.unwrap();

    // 5. Check goal alignment
    let alignment = system.check_alignment("authentication").await.unwrap();
    assert!(alignment.score > 0.0);

    // 6. End session (triggers consolidation)
    system.end_session(session_id).await.unwrap();

    // Verify consolidation occurred
    let consolidated_count = system.consolidated_memory_count().await;
    assert!(consolidated_count > 0);
}

#[tokio::test]
async fn test_hook_skill_agent_interaction() {
    let system = IntegrationTestSystem::new().await;

    // 1. Start goal-tracker agent
    system.start_agent("goal-tracker").await.unwrap();

    // 2. Create a goal via skill
    system.invoke_skill("context-injection", serde_json::json!({
        "content": "Implement user authentication with OAuth2",
        "memory_type": "documentation"
    })).await.unwrap();

    // 3. Discover goals (should find the one we just created)
    system.trigger_goal_discovery().await.unwrap();

    // 4. Write code (triggers alignment via hook)
    system.write_file(
        "src/oauth/client.rs",
        "pub struct OAuthClient { /* ... */ }",
    ).await.unwrap();

    // 5. Agent should detect and track alignment
    let status = system.get_agent_status("goal-tracker").await.unwrap();
    assert_eq!(status.status, AgentStatus::Running);

    system.stop_agent("goal-tracker").await.unwrap();
}

#[tokio::test]
async fn test_error_recovery_scenario() {
    let system = IntegrationTestSystem::new().await;

    // 1. Start with a session
    let session_id = system.start_session().await.unwrap();

    // 2. Simulate store becoming temporarily unavailable
    system.simulate_store_failure().await;

    // 3. Try operations - should degrade gracefully
    let read_result = system.read_file("src/test.rs").await;
    // Should not panic, should return degraded response

    // 4. Restore store
    system.restore_store().await;

    // 5. Operations should resume
    system.write_file("src/test.rs", "fn test() {}").await.unwrap();

    system.end_session(session_id).await.unwrap();
}

#[tokio::test]
async fn test_learning_feedback_loop() {
    let system = IntegrationTestSystem::new().await;

    // 1. Create initial pattern
    for i in 0..5 {
        system.write_file(
            &format!("src/handlers/handler{}.rs", i),
            &format!("pub fn handle_{i}(req: Request) -> Response {{ /* ... */ }}"),
        ).await.unwrap();
    }

    // 2. Run pattern mining
    system.start_agent("pattern-miner").await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // 3. Check if patterns were discovered
    let patterns = system.search_patterns("handler").await.unwrap();
    // Should find the handler pattern

    system.stop_agent("pattern-miner").await.unwrap();
}
```
    </section>

    <section name="PerformanceBenchmarks">
```rust
// crates/context-graph-mcp/benches/integration_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use context_graph_mcp::*;

fn hook_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("Hooks");

    // Pre-file-read benchmark
    group.bench_function("pre_file_read", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_hook_harness().await;
                harness.pre_file_read_handler.handle(FileReadPayload {
                    file_path: "src/test.rs".to_string(),
                    line_range: None,
                    source_tool: "Read".to_string(),
                }).await.unwrap()
            })
        })
    });

    // Pre-file-write benchmark
    group.bench_function("pre_file_write", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_hook_harness().await;
                harness.pre_file_write_handler.handle(FileWritePayload {
                    file_path: "src/test.rs".to_string(),
                    new_content: "fn test() {}".to_string(),
                    original_content: None,
                    operation: FileOperation::Create,
                    source_tool: "Write".to_string(),
                }).await.unwrap()
            })
        })
    });

    // Pre-bash-exec benchmark
    group.bench_function("pre_bash_exec", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_hook_harness().await;
                harness.pre_bash_exec_handler.handle(BashExecPayload {
                    command: "cargo test".to_string(),
                    working_dir: None,
                    env_vars: None,
                    timeout_ms: None,
                }).await.unwrap()
            })
        })
    });

    group.finish();
}

fn skill_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("Skills");

    // Memory search skill
    group.bench_function("memory_search_skill", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_skill_harness().await;
                harness.skill_loader.invoke(
                    "memory-search",
                    serde_json::json!({"query": "authentication"}),
                ).await.unwrap()
            })
        })
    });

    // Goal alignment skill
    group.bench_function("goal_alignment_skill", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_skill_harness().await;
                harness.skill_loader.invoke(
                    "goal-alignment",
                    serde_json::json!({
                        "content": "test content",
                        "goal_id": "test-goal"
                    }),
                ).await.unwrap()
            })
        })
    });

    group.finish();
}

fn agent_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("Agents");

    // Agent start time
    group.bench_function("agent_start", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_agent_harness().await;
                let id = harness.orchestrator.start_agent("goal-tracker").await.unwrap();
                harness.orchestrator.stop_agent("goal-tracker").await.unwrap();
                id
            })
        })
    });

    // Message send time
    group.bench_function("agent_message", |b| {
        b.iter(|| {
            rt.block_on(async {
                let harness = create_agent_harness().await;
                harness.orchestrator.start_agent("context-curator").await.unwrap();

                let message = AgentMessage {
                    message_id: Uuid::new_v4(),
                    from: "test".to_string(),
                    to: "context-curator".to_string(),
                    message_type: MessageType::Query,
                    payload: serde_json::json!({}),
                    timestamp: Utc::now(),
                };

                harness.orchestrator.send_message("context-curator", message).await.unwrap();
                harness.orchestrator.stop_agent("context-curator").await.unwrap();
            })
        })
    });

    group.finish();
}

fn assert_performance_targets(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Performance targets from spec
    let targets = [
        ("hook_execution", 50_u64),       // <50ms
        ("skill_invocation", 100_u64),    // <100ms
        ("agent_response", 500_u64),      // <500ms
        ("store_operation", 10_u64),      // <10ms
    ];

    let mut group = c.benchmark_group("Performance Targets");

    for (name, target_ms) in targets {
        group.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let start = std::time::Instant::now();

                    match name {
                        "hook_execution" => {
                            let harness = create_hook_harness().await;
                            harness.pre_file_read_handler.handle(FileReadPayload {
                                file_path: "test.rs".to_string(),
                                line_range: None,
                                source_tool: "Read".to_string(),
                            }).await.unwrap();
                        }
                        "skill_invocation" => {
                            let harness = create_skill_harness().await;
                            harness.skill_loader.invoke(
                                "context-injection",
                                serde_json::json!({"content": "test"}),
                            ).await.unwrap();
                        }
                        "agent_response" => {
                            // Agent response time
                        }
                        "store_operation" => {
                            let store = TeleologicalStore::in_memory().await.unwrap();
                            store.inject(InjectionRequest {
                                content: "test".to_string(),
                                ..Default::default()
                            }).await.unwrap();
                        }
                        _ => {}
                    }

                    let elapsed = start.elapsed().as_millis() as u64;
                    assert!(
                        elapsed < target_ms,
                        "{} took {}ms, target is {}ms",
                        name, elapsed, target_ms
                    );
                })
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    hook_benchmarks,
    skill_benchmarks,
    agent_benchmarks,
    assert_performance_targets,
);
criterion_main!(benches);
```
    </section>
  </pseudo_code>
</task_spec>
```
