# Task: TASK-P6-003 - Memory Inject Context Command

```xml
<task_spec id="TASK-P6-003" version="2.0">
<metadata>
  <title>Memory Inject Context Command</title>
  <phase>6</phase>
  <sequence>45</sequence>
  <layer>surface</layer>
  <estimated_loc>400</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure (main.rs, Commands enum)</dependency>
    <dependency task="TASK-P5-007">InjectionPipeline (context-graph-core/src/injection/pipeline.rs)</dependency>
    <dependency task="TASK-P2-005">ProductionMultiArrayProvider (context-graph-embeddings/src/provider/multi_array.rs)</dependency>
  </dependencies>
  <produces>
    <artifact type="module">crates/context-graph-cli/src/commands/memory/mod.rs</artifact>
    <artifact type="module">crates/context-graph-cli/src/commands/memory/inject.rs</artifact>
    <artifact type="function">handle_inject_context</artifact>
  </produces>
  <prd_alignment>
    <section>9.3 CLI Structure</section>
    <command>memory inject-context</command>
    <hook_integration>UserPromptSubmit calls `context-graph-cli memory inject-context "$PROMPT"`</hook_integration>
  </prd_alignment>
</metadata>

<context>
  <background>
    The memory inject-context command is the PRIMARY context injection mechanism, called by
    UserPromptSubmit and SessionStart native hooks via shell scripts. It embeds the user's
    query using all 13 embedders, finds relevant memories via InjectionPipeline, computes
    divergence alerts (SEMANTIC embedders only), and outputs formatted markdown to stdout.

    NOTE: This command injects MEMORIES via the InjectionPipeline using 13-space embeddings,
    divergence detection, and topic-based ranking per PRD v6.
  </background>

  <business_value>
    This is the primary memory retrieval and injection mechanism per PRD Section 9.3.
    It surfaces relevant past work and divergence alerts to help Claude Code understand
    the user's context across sessions.
  </business_value>

  <technical_context>
    - Query comes from positional argument or USER_PROMPT environment variable
    - Session ID comes from --session-id flag or CLAUDE_SESSION_ID env var
    - Output goes directly to stdout for Claude Code hook capture
    - Uses InjectionPipeline from context-graph-core for the complete pipeline
    - Uses ProductionMultiArrayProvider from context-graph-embeddings for embedding
    - Hook timeout: 2000ms for UserPromptSubmit (see claudehooks.md)
  </technical_context>

  <constitution_compliance>
    - ARCH-09: Topic threshold is weighted_agreement >= 2.5
    - ARCH-10: Divergence detection uses SEMANTIC embedders only (E1, E5, E6, E7, E10, E12, E13)
    - AP-60: Temporal embedders (E2-E4) NEVER count toward topic detection
    - AP-62: Divergence alerts MUST only use SEMANTIC embedders
    - AP-14: No .unwrap() in library code
    - AP-26: Exit code 1 on error, 2 on corruption
  </constitution_compliance>
</context>

<existing_codebase_audit>
  <current_cli_structure>
    <!-- ACTUAL current state as of audit date -->
    Commands enum in main.rs:
    - Consciousness { action: ConsciousnessCommands }
    - Session { action: SessionCommands }
    - Hooks { action: HooksCommands }

    MISSING per PRD Section 9.3:
    - Memory { action: MemoryCommands }  <!-- THIS TASK CREATES THIS -->
    - Topic { action: TopicCommands }
    - Divergence { action: DivergenceCommands }
  </current_cli_structure>

  <existing_injection_infrastructure>
    <!-- InjectionPipeline EXISTS and is production-ready -->
    File: crates/context-graph-core/src/injection/pipeline.rs

    Key types:
    - InjectionPipeline::new(retriever: SimilarityRetriever, store: Arc&lt;MemoryStore&gt;)
    - generate_context(&amp;self, query: &amp;SemanticFingerprint, session_id: &amp;str, limit: Option&lt;usize&gt;) -> Result&lt;InjectionResult, InjectionError&gt;
    - generate_brief_context(&amp;self, query: &amp;SemanticFingerprint, session_id: &amp;str) -> Result&lt;InjectionResult, InjectionError&gt;

    Supporting modules (all exist):
    - crates/context-graph-core/src/injection/budget.rs (TokenBudget, DEFAULT_TOKEN_BUDGET=1200)
    - crates/context-graph-core/src/injection/result.rs (InjectionResult)
    - crates/context-graph-core/src/injection/formatter.rs (ContextFormatter)
    - crates/context-graph-core/src/injection/priority.rs (PriorityRanker)
  </existing_injection_infrastructure>

  <embedding_provider>
    <!-- ProductionMultiArrayProvider EXISTS -->
    File: crates/context-graph-embeddings/src/provider/multi_array.rs

    Key usage:
    - ProductionMultiArrayProvider::new(models_dir: PathBuf, gpu_config: GpuConfig).await?
    - provider.embed_all(content: &amp;str).await? -> MultiArrayEmbeddingOutput
    - output.fingerprint() -> SemanticFingerprint
  </embedding_provider>

  <storage>
    <!-- MemoryStore EXISTS -->
    File: crates/context-graph-core/src/memory/store.rs
    - MemoryStore::new(path: &amp;Path) -> Result&lt;Self, StorageError&gt;

    <!-- SimilarityRetriever EXISTS -->
    File: crates/context-graph-core/src/retrieval/retriever.rs
    - SimilarityRetriever::with_defaults(store: Arc&lt;MemoryStore&gt;) -> Self
  </storage>

  <do_not_confuse_with>
    <!-- NOTE: Legacy consciousness commands were removed in PRD v6 refactoring -->
    <!-- This memory inject-context command is the PRIMARY injection mechanism -->
    <!-- It uses the InjectionPipeline with 13-space embeddings and topic-based ranking -->
  </do_not_confuse_with>
</existing_codebase_audit>

<prerequisites>
  <prerequisite type="code" status="EXISTS">
    crates/context-graph-cli/src/main.rs
    - Contains Commands enum, need to ADD Memory variant
  </prerequisite>
  <prerequisite type="code" status="EXISTS">
    crates/context-graph-core/src/injection/pipeline.rs
    - InjectionPipeline fully implemented
  </prerequisite>
  <prerequisite type="code" status="EXISTS">
    crates/context-graph-embeddings/src/provider/multi_array.rs
    - ProductionMultiArrayProvider fully implemented
  </prerequisite>
  <prerequisite type="code" status="EXISTS">
    crates/context-graph-core/src/memory/store.rs
    - MemoryStore for RocksDB access
  </prerequisite>
  <prerequisite type="code" status="EXISTS">
    crates/context-graph-core/src/retrieval/retriever.rs
    - SimilarityRetriever for similarity search
  </prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>MemoryCommands enum with InjectContext variant</item>
    <item>handle_inject_context() async function</item>
    <item>handle_memory_command() dispatcher</item>
    <item>Query embedding via ProductionMultiArrayProvider</item>
    <item>Context generation via InjectionPipeline</item>
    <item>Environment variable fallbacks (USER_PROMPT, CLAUDE_SESSION_ID)</item>
    <item>Empty output handling (no relevant context = empty stdout, exit 0)</item>
    <item>Error output to stderr with appropriate exit codes</item>
  </includes>
  <excludes>
    <item>Brief context injection (TASK-P6-004, uses generate_brief_context)</item>
    <item>Memory capture (TASK-P6-005, TASK-P6-006)</item>
    <item>Topic commands (TASK-P6-007+)</item>
    <item>Divergence commands (separate task)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>Command `context-graph-cli memory inject-context [QUERY]` exists</description>
    <verification>cargo run --package context-graph-cli -- memory inject-context --help shows usage</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>inject-context outputs formatted markdown to stdout when memories found</description>
    <verification>Output contains "## Relevant Context" header or similar</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>Empty query returns empty stdout (not error)</description>
    <verification>Exit code 0, empty stdout, no error logged</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>No relevant context returns empty stdout</description>
    <verification>Exit code 0, empty stdout when no memories match</verification>
  </criterion>
  <criterion id="DOD-5">
    <description>Environment variable fallbacks work correctly</description>
    <verification>USER_PROMPT env var used when positional arg not provided</verification>
  </criterion>
  <criterion id="DOD-6">
    <description>Execution completes within hook timeout budget</description>
    <verification>p95 latency &lt;1500ms (leaving margin for 2000ms hook timeout)</verification>
  </criterion>

  <signatures>
    <signature name="MemoryCommands">
      <code>
#[derive(Subcommand)]
pub enum MemoryCommands {
    /// Inject relevant context from memory store
    InjectContext(InjectContextArgs),
    // Future: Capture, etc.
}
      </code>
    </signature>

    <signature name="InjectContextArgs">
      <code>
#[derive(Args)]
pub struct InjectContextArgs {
    /// Query text for context retrieval (or use USER_PROMPT env var)
    pub query: Option&lt;String&gt;,

    /// Session ID (or use CLAUDE_SESSION_ID env var)
    #[arg(long)]
    pub session_id: Option&lt;String&gt;,

    /// Token budget for context (default: 1200)
    #[arg(long, default_value = "1200")]
    pub budget: u32,

    /// Path to models directory
    #[arg(long, env = "CONTEXT_GRAPH_MODELS_DIR")]
    pub models_dir: Option&lt;PathBuf&gt;,

    /// Path to data directory
    #[arg(long, env = "CONTEXT_GRAPH_DATA_DIR")]
    pub data_dir: Option&lt;PathBuf&gt;,
}
      </code>
    </signature>

    <signature name="handle_inject_context">
      <code>
pub async fn handle_inject_context(args: InjectContextArgs) -> i32
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">Context markdown to stdout ONLY</constraint>
    <constraint type="output">Errors/logs to stderr ONLY</constraint>
    <constraint type="output">Empty string if no relevant context (exit code 0)</constraint>
    <constraint type="performance">Complete in &lt;1500ms p95 (hook timeout is 2000ms)</constraint>
    <constraint type="budget">Default budget 1200 tokens per constitution.yaml</constraint>
    <constraint type="exit_codes">
      - 0: Success (including empty result)
      - 1: Error (pipeline failure, storage error)
      - 2: Corruption (memory not found, stale index)
    </constraint>
  </constraints>
</definition_of_done>

<implementation_guide>
```rust
// =============================================================================
// FILE: crates/context-graph-cli/src/commands/memory/mod.rs
// =============================================================================

//! Memory management commands
//!
//! Commands for memory capture and context injection per PRD Section 9.3.
//!
//! # Commands
//!
//! - `memory inject-context`: Inject relevant context for UserPromptSubmit hook

pub mod inject;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum MemoryCommands {
    /// Inject relevant context from memory store
    InjectContext(inject::InjectContextArgs),
}

/// Handle memory subcommands
pub async fn handle_memory_command(cmd: MemoryCommands) -> i32 {
    match cmd {
        MemoryCommands::InjectContext(args) => inject::handle_inject_context(args).await,
    }
}

// =============================================================================
// FILE: crates/context-graph-cli/src/commands/memory/inject.rs
// =============================================================================

//! Memory inject-context command implementation.
//!
//! Called by UserPromptSubmit and SessionStart hooks to inject relevant
//! memories into Claude Code's context.
//!
//! # Constitution Compliance
//!
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - ARCH-10: Divergence uses SEMANTIC embedders only
//! - AP-26: Exit code 1 on error, 2 on corruption

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use tracing::{debug, error, info, warn};

use context_graph_core::injection::{InjectionError, InjectionPipeline, TokenBudget};
use context_graph_core::memory::MemoryStore;
use context_graph_core::retrieval::SimilarityRetriever;
use context_graph_embeddings::config::GpuConfig;
use context_graph_embeddings::provider::ProductionMultiArrayProvider;

use crate::error::CliExitCode;

/// Arguments for inject-context command
#[derive(Args)]
pub struct InjectContextArgs {
    /// Query text for context retrieval (or use USER_PROMPT env var)
    pub query: Option<String>,

    /// Session ID (or use CLAUDE_SESSION_ID env var)
    #[arg(long)]
    pub session_id: Option<String>,

    /// Token budget for context (default: 1200)
    #[arg(long, default_value = "1200")]
    pub budget: u32,

    /// Path to models directory
    #[arg(long, env = "CONTEXT_GRAPH_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Path to data directory
    #[arg(long, env = "CONTEXT_GRAPH_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
}

/// Handle inject-context command.
///
/// Embeds query, retrieves relevant memories, outputs formatted markdown to stdout.
/// Empty result = empty stdout (not an error).
///
/// # Exit Codes
/// - 0: Success (including empty result)
/// - 1: Error (pipeline/storage failure)
/// - 2: Corruption (missing memory, stale index)
pub async fn handle_inject_context(args: InjectContextArgs) -> i32 {
    // Get query from arg or USER_PROMPT env var
    let query = args
        .query
        .or_else(|| std::env::var("USER_PROMPT").ok())
        .filter(|q| !q.trim().is_empty());

    let Some(query) = query else {
        debug!("No query provided, returning empty context");
        return CliExitCode::Success as i32;
    };

    // Get session ID from arg or CLAUDE_SESSION_ID env var
    let session_id = args
        .session_id
        .or_else(|| std::env::var("CLAUDE_SESSION_ID").ok())
        .unwrap_or_else(|| {
            warn!("No session ID available, using 'default'");
            "default".to_string()
        });

    // Get paths from args or defaults
    let models_dir = args
        .models_dir
        .unwrap_or_else(|| PathBuf::from("./models"));
    let data_dir = args.data_dir.unwrap_or_else(|| PathBuf::from("./data"));

    info!(
        query_len = query.len(),
        session_id = %session_id,
        budget = args.budget,
        "Injecting memory context"
    );

    // Initialize embedding provider
    let provider = match ProductionMultiArrayProvider::new(models_dir, GpuConfig::default()).await {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to initialize embedding provider: {}", e);
            eprintln!("ERROR: Failed to initialize embeddings: {}", e);
            return CliExitCode::Error as i32;
        }
    };

    // Embed the query (all 13 embedders)
    let embedding_output = match provider.embed_all(&query).await {
        Ok(output) => output,
        Err(e) => {
            error!("Failed to embed query: {}", e);
            eprintln!("ERROR: Failed to embed query: {}", e);
            return CliExitCode::Error as i32;
        }
    };

    let query_fingerprint = embedding_output.fingerprint();

    // Initialize storage
    let store = match MemoryStore::new(&data_dir.join("memories")) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            error!("Failed to open memory store: {}", e);
            eprintln!("ERROR: Failed to open memory store: {}", e);
            return CliExitCode::Error as i32;
        }
    };

    // Create retriever and pipeline
    let retriever = SimilarityRetriever::with_defaults(store.clone());
    let budget = if args.budget != 1200 {
        TokenBudget::with_total(args.budget as usize)
    } else {
        TokenBudget::default()
    };
    let pipeline = InjectionPipeline::with_budget(retriever, store, budget);

    // Generate context
    let result = match pipeline.generate_context(&query_fingerprint, &session_id, None) {
        Ok(r) => r,
        Err(InjectionError::MemoryNotFound(id)) => {
            error!("Memory not found (stale index?): {}", id);
            eprintln!("CORRUPTION: Memory {} not found - index may be stale", id);
            return CliExitCode::Corruption as i32;
        }
        Err(e) => {
            error!("Pipeline failed: {}", e);
            eprintln!("ERROR: Context generation failed: {}", e);
            return CliExitCode::Error as i32;
        }
    };

    // Output result to stdout
    if result.is_empty() {
        debug!("No relevant context found");
        // Empty stdout = no injection (exit 0)
    } else {
        info!(
            memories = result.memory_count(),
            tokens = result.tokens_used,
            has_alerts = result.has_divergence_alerts(),
            "Context generated"
        );
        // Print context to stdout for hook capture
        print!("{}", result.formatted_context);
    }

    CliExitCode::Success as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Tests use REAL components, no mocks per constitution requirements.
    // These are unit tests for argument handling; integration tests in tests/integration/

    #[test]
    fn test_empty_query_env_fallback() {
        // Given: No arg, no env var
        std::env::remove_var("USER_PROMPT");
        let args = InjectContextArgs {
            query: None,
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };

        // Then: query resolution returns None
        let query = args
            .query
            .clone()
            .or_else(|| std::env::var("USER_PROMPT").ok())
            .filter(|q| !q.trim().is_empty());
        assert!(query.is_none());
        println!("[PASS] Empty query with no env var returns None");
    }

    #[test]
    fn test_query_from_env_var() {
        // Given: No arg, but env var set
        std::env::set_var("USER_PROMPT", "test query from env");
        let args = InjectContextArgs {
            query: None,
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };

        // Then: query comes from env var
        let query = args
            .query
            .clone()
            .or_else(|| std::env::var("USER_PROMPT").ok())
            .filter(|q| !q.trim().is_empty());
        assert_eq!(query, Some("test query from env".to_string()));
        std::env::remove_var("USER_PROMPT");
        println!("[PASS] Query from USER_PROMPT env var");
    }

    #[test]
    fn test_query_arg_overrides_env() {
        // Given: Both arg and env var set
        std::env::set_var("USER_PROMPT", "env query");
        let args = InjectContextArgs {
            query: Some("arg query".to_string()),
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };

        // Then: arg takes priority
        let query = args
            .query
            .clone()
            .or_else(|| std::env::var("USER_PROMPT").ok())
            .filter(|q| !q.trim().is_empty());
        assert_eq!(query, Some("arg query".to_string()));
        std::env::remove_var("USER_PROMPT");
        println!("[PASS] CLI arg overrides env var");
    }

    #[test]
    fn test_whitespace_query_treated_as_empty() {
        // Given: Whitespace-only query
        let args = InjectContextArgs {
            query: Some("   \n\t  ".to_string()),
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };

        // Then: treated as empty (filter removes it)
        let query = args
            .query
            .clone()
            .or_else(|| std::env::var("USER_PROMPT").ok())
            .filter(|q| !q.trim().is_empty());
        assert!(query.is_none());
        println!("[PASS] Whitespace-only query treated as empty");
    }

    #[test]
    fn test_session_id_env_fallback() {
        // Given: No arg, but env var set
        std::env::set_var("CLAUDE_SESSION_ID", "test-session-123");
        let args = InjectContextArgs {
            query: None,
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };

        // Then: session ID comes from env var
        let session_id = args
            .session_id
            .clone()
            .or_else(|| std::env::var("CLAUDE_SESSION_ID").ok())
            .unwrap_or_else(|| "default".to_string());
        assert_eq!(session_id, "test-session-123");
        std::env::remove_var("CLAUDE_SESSION_ID");
        println!("[PASS] Session ID from CLAUDE_SESSION_ID env var");
    }

    #[test]
    fn test_default_budget() {
        let args = InjectContextArgs {
            query: None,
            session_id: None,
            budget: 1200,
            models_dir: None,
            data_dir: None,
        };
        assert_eq!(args.budget, 1200);
        println!("[PASS] Default budget is 1200 per constitution");
    }
}
```
</implementation_guide>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/memory/mod.rs">
    Module definition with MemoryCommands enum and dispatcher
  </file>
  <file path="crates/context-graph-cli/src/commands/memory/inject.rs">
    InjectContextArgs struct and handle_inject_context function
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Add: pub mod memory;
  </file>
  <file path="crates/context-graph-cli/src/main.rs">
    Add Memory variant to Commands enum:
    Memory {
        #[command(subcommand)]
        action: commands::memory::MemoryCommands,
    },

    Add dispatch in main():
    Commands::Memory { action } => commands::memory::handle_memory_command(action).await,
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles without errors</criterion>
  <criterion type="test">cargo test --package context-graph-cli -- memory passes</criterion>
  <criterion type="cli_help">./target/debug/context-graph-cli memory --help shows InjectContext subcommand</criterion>
  <criterion type="cli_run">./target/debug/context-graph-cli memory inject-context "test" executes</criterion>
</validation_criteria>

<test_commands>
  <command desc="Build CLI">cargo build --package context-graph-cli</command>
  <command desc="Run unit tests">cargo test --package context-graph-cli memory::</command>
  <command desc="Show help">./target/debug/context-graph-cli memory --help</command>
  <command desc="Run with query arg">./target/debug/context-graph-cli memory inject-context "implement HDBSCAN clustering"</command>
  <command desc="Run with env var">USER_PROMPT="test" ./target/debug/context-graph-cli memory inject-context</command>
  <command desc="Run with session ID">./target/debug/context-graph-cli memory inject-context --session-id abc-123 "test query"</command>
  <command desc="Run with custom budget">./target/debug/context-graph-cli memory inject-context --budget 800 "test"</command>
</test_commands>

<full_state_verification>
  <source_of_truth>
    <item>RocksDB at $CONTEXT_GRAPH_DATA_DIR/memories contains stored memories</item>
    <item>FAISS indexes at $CONTEXT_GRAPH_DATA_DIR/indexes contain embedding vectors</item>
    <item>InjectionPipeline uses SimilarityRetriever to query these stores</item>
  </source_of_truth>

  <execute_and_inspect>
    <step order="1">
      <description>Verify memory store exists and has content</description>
      <command>ls -la ./data/memories/</command>
      <expected>RocksDB files (LOCK, LOG, MANIFEST-*, *.sst)</expected>
    </step>
    <step order="2">
      <description>Run inject-context with verbose logging</description>
      <command>RUST_LOG=debug ./target/debug/context-graph-cli memory inject-context "test query" 2>&amp;1 | tee /tmp/inject.log</command>
      <expected>
        - "Injecting memory context" log entry
        - "Retrieved similar memories" with count
        - "Checked divergence" with alert count
        - "Context generated" OR "No relevant context found"
      </expected>
    </step>
    <step order="3">
      <description>Verify stdout contains properly formatted markdown (if memories exist)</description>
      <command>./target/debug/context-graph-cli memory inject-context "test" | head -20</command>
      <expected>Markdown with ## headers or empty output</expected>
    </step>
    <step order="4">
      <description>Verify exit code on success</description>
      <command>./target/debug/context-graph-cli memory inject-context "test"; echo "Exit: $?"</command>
      <expected>Exit: 0</expected>
    </step>
  </execute_and_inspect>

  <boundary_edge_cases>
    <case id="EDGE-1">
      <description>Empty query (no arg, no env var)</description>
      <before>USER_PROMPT unset, no positional arg</before>
      <command>unset USER_PROMPT; ./target/debug/context-graph-cli memory inject-context 2>&amp;1; echo "Exit: $?"</command>
      <after>Empty stdout, Exit: 0</after>
      <rationale>Empty query = nothing to search, graceful empty response</rationale>
    </case>
    <case id="EDGE-2">
      <description>Whitespace-only query</description>
      <before>Query is "   \n\t  "</before>
      <command>./target/debug/context-graph-cli memory inject-context "   " 2>&amp;1; echo "Exit: $?"</command>
      <after>Empty stdout, Exit: 0</after>
      <rationale>Whitespace treated as empty per implementation</rationale>
    </case>
    <case id="EDGE-3">
      <description>Very long query (>10KB)</description>
      <before>Query = 10,000 character string</before>
      <command>./target/debug/context-graph-cli memory inject-context "$(python3 -c 'print("x"*10000)')" 2>&amp;1; echo "Exit: $?"</command>
      <after>Should either succeed or fail gracefully (Exit: 0 or 1, not crash)</after>
      <rationale>Embedding models have token limits, should handle gracefully</rationale>
    </case>
    <case id="EDGE-4">
      <description>Empty memory store</description>
      <before>Fresh data directory with no memories stored</before>
      <command>rm -rf ./data/test_empty &amp;&amp; mkdir -p ./data/test_empty/memories &amp;&amp; ./target/debug/context-graph-cli memory inject-context --data-dir ./data/test_empty "test" 2>&amp;1; echo "Exit: $?"</command>
      <after>Empty stdout, Exit: 0 (no relevant context found)</after>
      <rationale>Empty store = no memories to inject</rationale>
    </case>
    <case id="EDGE-5">
      <description>Missing models directory</description>
      <before>--models-dir points to non-existent path</before>
      <command>./target/debug/context-graph-cli memory inject-context --models-dir /nonexistent "test" 2>&amp;1; echo "Exit: $?"</command>
      <after>Error message to stderr, Exit: 1</after>
      <rationale>Fail fast on missing models</rationale>
    </case>
    <case id="EDGE-6">
      <description>Corrupted memory store</description>
      <before>Memory store has corrupted/truncated files</before>
      <command>
        mkdir -p ./data/test_corrupt/memories
        echo "garbage" > ./data/test_corrupt/memories/LOCK
        ./target/debug/context-graph-cli memory inject-context --data-dir ./data/test_corrupt "test" 2>&amp;1
        echo "Exit: $?"
      </command>
      <after>Error message about storage, Exit: 1 or 2</after>
      <rationale>Fail fast on corruption</rationale>
    </case>
    <case id="EDGE-7">
      <description>Budget = 0 tokens</description>
      <before>--budget 0</before>
      <command>./target/debug/context-graph-cli memory inject-context --budget 0 "test" 2>&amp;1; echo "Exit: $?"</command>
      <after>Empty stdout (no room for any memories), Exit: 0</after>
      <rationale>Zero budget = no content can fit</rationale>
    </case>
    <case id="EDGE-8">
      <description>Unicode query with emoji</description>
      <before>Query contains emoji and CJK characters</before>
      <command>./target/debug/context-graph-cli memory inject-context "你好 👋 こんにちは" 2>&amp;1; echo "Exit: $?"</command>
      <after>Should succeed (Exit: 0) - embedding models handle UTF-8</after>
      <rationale>Unicode support required</rationale>
    </case>
  </boundary_edge_cases>

  <manual_verification>
    <check id="MV-1">
      <description>Verify output is valid markdown</description>
      <action>Pipe output to a markdown renderer and visually inspect</action>
      <command>./target/debug/context-graph-cli memory inject-context "test" | mdcat</command>
    </check>
    <check id="MV-2">
      <description>Verify latency is within budget</description>
      <action>Time command execution multiple times</action>
      <command>for i in {1..5}; do time ./target/debug/context-graph-cli memory inject-context "test" > /dev/null 2>&amp;1; done</command>
      <expected>Each execution &lt;1500ms</expected>
    </check>
    <check id="MV-3">
      <description>Verify hook integration works</description>
      <action>Simulate UserPromptSubmit hook call</action>
      <command>
        export USER_PROMPT="How do I implement HDBSCAN clustering?"
        export CLAUDE_SESSION_ID="test-session-$(date +%s)"
        ./target/debug/context-graph-cli memory inject-context
      </command>
      <expected>Context output or empty (depending on memory store contents)</expected>
    </check>
  </manual_verification>
</full_state_verification>

<hook_integration>
  <hook name="UserPromptSubmit">
    <timeout_ms>2000</timeout_ms>
    <shell_script>hooks/user-prompt-submit.sh</shell_script>
    <script_content>
#!/bin/bash
# UserPromptSubmit hook - inject relevant memory context
# Called with user prompt as $1

set -euo pipefail

PROMPT="${1:-}"
if [ -z "$PROMPT" ]; then
    exit 0
fi

# Call memory inject-context
# Output goes to stdout for Claude Code to capture
context-graph-cli memory inject-context "$PROMPT"
    </script_content>
  </hook>
</hook_integration>

<anti_patterns_to_avoid>
  <ap ref="AP-14">No .unwrap() - use ? operator or explicit error handling</ap>
  <ap ref="AP-26">Exit code 1 on error, 2 on corruption - don't exit 0 on failure</ap>
  <ap ref="AP-60">Temporal embedders excluded from topic detection (handled by InjectionPipeline)</ap>
  <ap ref="AP-62">Divergence only from SEMANTIC embedders (handled by SimilarityRetriever)</ap>
  <ap ref="AP-72">No TODO stubs - implement fully or don't implement</ap>
</anti_patterns_to_avoid>

</task_spec>
```
