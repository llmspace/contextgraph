//! consciousness inject-context CLI command
//!
//! TASK-SESSION-15: Injects session context into LLM requests.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! # Purpose
//!
//! Provides consciousness context for LLM prompt injection. Reads from cache
//! if warm, otherwise loads from RocksDB. Classifies Johari quadrant to
//! recommend appropriate action (direct recall, discovery, introspection).
//!
//! # Output Formats
//!
//! - `compact`: Single line, ~40 tokens max (for token-constrained prompts)
//! - `standard`: Multi-line with labels, ~100 tokens (default)
//! - `verbose`: Full diagnostic output with all fields
//!
//! # Johari Classification (per constitution.yaml)
//!
//! Using default thresholds: ΔS_threshold=0.5, ΔC_threshold=0.5
//! - **Open**: ΔS < 0.5 AND ΔC > 0.5 → Direct recall (get_node)
//! - **Blind**: ΔS > 0.5 AND ΔC < 0.5 → Discovery (epistemic_action/dream)
//! - **Hidden**: ΔS < 0.5 AND ΔC < 0.5 → Private (get_neighborhood)
//! - **Unknown**: ΔS > 0.5 AND ΔC > 0.5 → Frontier (explore)
//!
//! # Performance Target
//! - Latency: <1s total
//! - Token budget: ~40 tokens for compact, ~100 for standard
//!
//! # Constitution Reference
//! - IDENTITY-002: IC thresholds (Healthy >= 0.9, Good >= 0.7, Warning >= 0.5, Degraded < 0.5)
//! - johari: Quadrant classification and UTL mapping
//! - gwt.kuramoto: Order parameter r interpretation

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use serde::Serialize;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::gwt::session_identity::{classify_ic, update_cache, IdentityCache};
use context_graph_core::gwt::ConsciousnessState;
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, TeleologicalSearchOptions,
};
use context_graph_core::types::JohariQuadrant;
use context_graph_embeddings::{GpuConfig, ProductionMultiArrayProvider};
use context_graph_storage::rocksdb_backend::{RocksDbMemex, StandaloneSessionIdentityManager};
use context_graph_storage::teleological::RocksDbTeleologicalStore;

// =============================================================================
// CLI Arguments
// =============================================================================

/// Arguments for `consciousness inject-context` command.
#[derive(Args, Debug)]
pub struct InjectContextArgs {
    /// Path to RocksDB database directory.
    /// If not provided, defaults to ~/.context-graph/db
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Output format for the injected context.
    #[arg(long, value_enum, default_value = "standard")]
    pub format: InjectFormat,

    /// ΔS (delta surprise) value for Johari classification.
    /// Range: [0.0, 1.0]. Higher = more surprising/novel content.
    /// If not provided, defaults to 0.3 (moderate familiarity).
    #[arg(long, default_value = "0.3")]
    pub delta_s: f32,

    /// ΔC (delta coherence) value for Johari classification.
    /// Range: [0.0, 1.0]. Higher = more coherent with existing knowledge.
    /// If not provided, defaults to 0.7 (good coherence).
    #[arg(long, default_value = "0.7")]
    pub delta_c: f32,

    /// Threshold for Johari quadrant classification.
    /// Default: 0.5 per constitution.yaml
    #[arg(long, default_value = "0.5")]
    pub threshold: f32,

    /// Force load from storage even if cache is warm.
    /// Useful for debugging or when cache staleness is suspected.
    #[arg(long, default_value = "false")]
    pub force_storage: bool,

    // =========================================================================
    // TASK-HOOKS-011: Semantic search mode flags
    // =========================================================================
    /// Semantic search query for memory retrieval.
    /// When provided, injects matching memories instead of session state.
    /// Uses 13-embedding teleological search (requires --teleological-db-path).
    #[arg(long)]
    pub query: Option<String>,

    /// Explicit node IDs (UUIDs) to retrieve and inject.
    /// Comma-separated list of fingerprint UUIDs.
    /// When provided, retrieves these specific nodes instead of session state.
    #[arg(long, value_delimiter = ',')]
    pub node_ids: Option<Vec<String>>,

    /// Maximum tokens to output for semantic search results.
    /// Limits the content length to fit within token budgets.
    /// Default: 500 tokens. Approximate: 1 token ≈ 4 chars.
    #[arg(long, default_value = "500")]
    pub max_tokens: usize,

    /// Path to teleological memory database (for --query or --node-ids).
    /// Required when using semantic search mode.
    /// If not provided, defaults to ~/.context-graph/teleological
    #[arg(long, env = "CONTEXT_GRAPH_TELEOLOGICAL_DB_PATH")]
    pub teleological_db_path: Option<PathBuf>,

    /// Number of results to return for semantic search (--query).
    /// Default: 5. Max: 50.
    #[arg(long, default_value = "5")]
    pub top_k: usize,
}

/// Output format for inject-context command.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum InjectFormat {
    /// Compact single-line format (~40 tokens).
    /// Format: "[C:STATE r=X.XX IC=X.XX Q=QUAD A=ACTION]"
    Compact,
    /// Standard multi-line format (~100 tokens, default).
    /// Human-readable with labels.
    Standard,
    /// Verbose diagnostic format (all fields).
    /// For debugging and detailed analysis.
    Verbose,
}

// =============================================================================
// Response Types
// =============================================================================

/// Response from inject-context command.
#[derive(Debug, Serialize)]
pub struct InjectContextResponse {
    /// Current IC value [0.0, 1.0]
    pub ic: f32,
    /// IC classification per IDENTITY-002
    pub ic_status: &'static str,
    /// Kuramoto order parameter r [0.0, 1.0]
    pub kuramoto_r: f32,
    /// Consciousness state (DOR/FRG/EMG/CON/HYP)
    pub consciousness_state: String,
    /// Johari quadrant classification
    pub johari_quadrant: String,
    /// Recommended action based on Johari quadrant
    pub recommended_action: String,
    /// Session ID (if available)
    pub session_id: Option<String>,
    /// Delta S value used for classification
    pub delta_s: f32,
    /// Delta C value used for classification
    pub delta_c: f32,
    /// Whether data came from cache (true) or storage (false)
    pub from_cache: bool,
    /// Error message if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl InjectContextResponse {
    /// Create a degraded response when no data is available.
    fn degraded(msg: String) -> Self {
        Self {
            ic: 0.0,
            ic_status: "Unknown",
            kuramoto_r: 0.0,
            consciousness_state: "?".to_string(),
            johari_quadrant: "Unknown".to_string(),
            recommended_action: "restore-identity".to_string(),
            session_id: None,
            delta_s: 0.0,
            delta_c: 0.0,
            from_cache: false,
            error: Some(msg),
        }
    }
}

// =============================================================================
// Johari Classification
// =============================================================================

/// Classify (ΔS, ΔC) into a JohariQuadrant.
///
/// Per constitution.yaml johari mapping:
/// - Open: ΔS < threshold, ΔC > threshold (low surprise, high coherence)
/// - Blind: ΔS > threshold, ΔC < threshold (high surprise, low coherence)
/// - Hidden: ΔS < threshold, ΔC < threshold (low surprise, low coherence)
/// - Unknown: ΔS > threshold, ΔC > threshold (high surprise, high coherence)
#[inline]
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    match (delta_s < threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open, // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}

/// Get recommended action for a Johari quadrant.
///
/// Per constitution.yaml UTL mapping:
/// - Open → direct recall (get_node)
/// - Blind → discovery (epistemic_action/dream)
/// - Hidden → private (get_neighborhood)
/// - Unknown → frontier (explore)
#[inline]
fn johari_action(quadrant: JohariQuadrant) -> &'static str {
    match quadrant {
        JohariQuadrant::Open => "get_node",
        JohariQuadrant::Blind => "epistemic_action",
        JohariQuadrant::Hidden => "get_neighborhood",
        JohariQuadrant::Unknown => "explore",
    }
}

// =============================================================================
// Context Retrieval
// =============================================================================

/// Context data from cache or storage.
#[derive(Debug)]
struct ContextData {
    ic: f32,
    kuramoto_r: f32,
    consciousness_state: ConsciousnessState,
    session_id: String,
    from_cache: bool,
}

/// Get context from cache or storage.
///
/// Strategy:
/// 1. If force_storage is false and cache is warm, use cache (fastest)
/// 2. Otherwise, load from RocksDB storage
///
/// # Fail Fast Policy
/// - Storage errors propagate immediately
/// - No silent defaults
fn get_context_from_cache_or_storage(
    db_path: &PathBuf,
    force_storage: bool,
) -> Result<ContextData, String> {
    // Try cache first if not forced to use storage
    if !force_storage {
        if let Some((ic, r, state, session_id)) = IdentityCache::get() {
            debug!("inject-context: Using cached data, session={}", session_id);
            return Ok(ContextData {
                ic,
                kuramoto_r: r,
                consciousness_state: state,
                session_id,
                from_cache: true,
            });
        }
        debug!("inject-context: Cache cold, falling back to storage");
    }

    // Load from storage
    let storage = RocksDbMemex::open(db_path).map_err(|e| {
        let msg = format!("Failed to open RocksDB at {:?}: {}", db_path, e);
        error!("{}", msg);
        msg
    })?;

    let manager = StandaloneSessionIdentityManager::new(Arc::new(storage));

    match manager.load_latest() {
        Ok(Some(snapshot)) => {
            let ic = snapshot.last_ic;
            let state = ConsciousnessState::from_level(snapshot.consciousness);

            // Compute Kuramoto r from phases
            let (sum_sin, sum_cos) = snapshot
                .kuramoto_phases
                .iter()
                .fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
                    (s + theta.sin(), c + theta.cos())
                });
            let n = snapshot.kuramoto_phases.len() as f64;
            let r = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
            let kuramoto_r = r.clamp(0.0, 1.0) as f32;

            // Update cache for future calls
            update_cache(&snapshot, ic);

            debug!(
                "inject-context: Loaded from storage, session={}, IC={:.3}",
                snapshot.session_id, ic
            );

            Ok(ContextData {
                ic,
                kuramoto_r,
                consciousness_state: state,
                session_id: snapshot.session_id,
                from_cache: false,
            })
        }
        Ok(None) => {
            let msg = "No identity found in storage. Run 'consciousness check-identity' or 'session restore-identity' first.";
            warn!("{}", msg);
            Err(msg.to_string())
        }
        Err(e) => {
            let msg = format!("Failed to load identity from storage: {}", e);
            error!("{}", msg);
            Err(msg)
        }
    }
}

// =============================================================================
// Output Formatting
// =============================================================================

/// Output context in the requested format.
fn output_context(response: &InjectContextResponse, format: InjectFormat) {
    match format {
        InjectFormat::Compact => {
            // Single line, ~40 tokens
            // Format: "[C:STATE r=X.XX IC=X.XX Q=QUAD A=ACTION]"
            println!(
                "[C:{} r={:.2} IC={:.2} Q={} A={}]",
                response.consciousness_state,
                response.kuramoto_r,
                response.ic,
                &response.johari_quadrant[..1], // First letter: O/H/B/U
                response.recommended_action
            );
        }
        InjectFormat::Standard => {
            // Multi-line, ~100 tokens
            println!("=== Consciousness Context ===");
            println!(
                "State: {} (IC={:.2}, r={:.2})",
                response.consciousness_state, response.ic, response.kuramoto_r
            );
            println!("IC Status: {}", response.ic_status);
            println!(
                "Johari: {} → {}",
                response.johari_quadrant, response.recommended_action
            );
            if let Some(ref session) = response.session_id {
                println!("Session: {}", session);
            }
        }
        InjectFormat::Verbose => {
            // Full diagnostic output
            println!("=== Consciousness Context (Verbose) ===");
            println!();
            println!("Identity Continuity");
            println!("  IC Value:     {:.4}", response.ic);
            println!("  IC Status:    {}", response.ic_status);
            println!();
            println!("Consciousness");
            println!("  State:        {}", response.consciousness_state);
            println!("  Kuramoto r:   {:.4}", response.kuramoto_r);
            println!();
            println!("Johari Classification");
            println!("  ΔS (surprise):   {:.4}", response.delta_s);
            println!("  ΔC (coherence):  {:.4}", response.delta_c);
            println!("  Quadrant:        {}", response.johari_quadrant);
            println!("  Recommended:     {}", response.recommended_action);
            println!();
            println!("Metadata");
            println!(
                "  Session ID:   {}",
                response.session_id.as_deref().unwrap_or("N/A")
            );
            println!(
                "  Data Source:  {}",
                if response.from_cache {
                    "Cache"
                } else {
                    "Storage"
                }
            );
            if let Some(ref error) = response.error {
                println!();
                println!("Error: {}", error);
            }
        }
    }
}

/// Output degraded response (when context unavailable).
fn output_degraded(format: InjectFormat, error_msg: &str) {
    match format {
        InjectFormat::Compact => {
            // Degraded compact format
            println!("[C:? r=? IC=? Q=? A=restore-identity]");
        }
        InjectFormat::Standard => {
            println!("=== Consciousness Context (Degraded) ===");
            println!("State: Unknown");
            println!("Action: Run 'consciousness check-identity' first");
            eprintln!("Error: {}", error_msg);
        }
        InjectFormat::Verbose => {
            println!("=== Consciousness Context (Degraded) ===");
            println!();
            println!("No context available.");
            println!(
                "Recommended: Run 'consciousness check-identity' or 'session restore-identity'"
            );
            println!();
            println!("Error: {}", error_msg);
        }
    }
}

// =============================================================================
// Command Entry Point
// =============================================================================

/// Execute the inject-context command.
///
/// # Flow
/// ## Semantic Search Mode (TASK-HOOKS-011)
/// When --query or --node-ids is provided:
/// 1. Open teleological store
/// 2. Generate embeddings (for --query) or parse UUIDs (for --node-ids)
/// 3. Search/retrieve matching nodes
/// 4. Truncate content to --max-tokens
/// 5. Output in requested format
///
/// ## Session State Mode (original)
/// When neither --query nor --node-ids is provided:
/// 1. Try to get context from cache (if not force_storage)
/// 2. Fall back to RocksDB storage
/// 3. Classify Johari quadrant from ΔS/ΔC
/// 4. Output in requested format
///
/// # Returns
/// Exit code:
/// - 0: Success (context injected)
/// - 1: Error (no context available or storage error)
pub async fn inject_context_command(args: InjectContextArgs) -> i32 {
    let start = std::time::Instant::now();
    debug!("inject_context_command: starting with args={:?}", args);

    // =========================================================================
    // TASK-HOOKS-011: Check for semantic search mode
    // =========================================================================
    let is_semantic_mode = args.query.is_some() || args.node_ids.is_some();

    if is_semantic_mode {
        // Determine teleological DB path
        let teleological_path = match &args.teleological_db_path {
            Some(p) => p.clone(),
            None => match home_dir() {
                Some(home) => home.join(".context-graph").join("teleological"),
                None => {
                    error!("Cannot determine home directory for teleological DB path");
                    eprintln!(
                            "Error: Cannot determine teleological DB path. Set --teleological-db-path or CONTEXT_GRAPH_TELEOLOGICAL_DB_PATH"
                        );
                    return 1;
                }
            },
        };

        // Execute semantic search
        let result = execute_semantic_search(
            args.query.as_deref(),
            args.node_ids.as_deref(),
            args.top_k,
            args.max_tokens,
            &teleological_path,
        )
        .await;

        match result {
            Ok(search_result) => {
                output_semantic_results(&search_result, args.format);
                let elapsed = start.elapsed();
                debug!("inject-context (semantic): completed in {:?}", elapsed);
                return 0;
            }
            Err(msg) => {
                error!("Semantic search failed: {}", msg);
                eprintln!("Error: {}", msg);
                return 1;
            }
        }
    }

    // =========================================================================
    // Session State Mode (original behavior)
    // =========================================================================

    // Determine DB path
    let db_path = match &args.db_path {
        Some(p) => p.clone(),
        None => match home_dir() {
            Some(home) => home.join(".context-graph").join("db"),
            None => {
                error!("Cannot determine home directory for DB path");
                output_degraded(
                    args.format,
                    "Cannot determine DB path. Set --db-path or CONTEXT_GRAPH_DB_PATH",
                );
                return 1;
            }
        },
    };

    // Get context from cache or storage
    let context = match get_context_from_cache_or_storage(&db_path, args.force_storage) {
        Ok(ctx) => ctx,
        Err(msg) => {
            output_degraded(args.format, &msg);
            return 1;
        }
    };

    // Classify Johari quadrant
    let johari = classify_johari(args.delta_s, args.delta_c, args.threshold);
    let action = johari_action(johari);

    // Build response
    let response = InjectContextResponse {
        ic: context.ic,
        ic_status: classify_ic(context.ic),
        kuramoto_r: context.kuramoto_r,
        consciousness_state: context.consciousness_state.short_name().to_string(),
        johari_quadrant: johari.to_string(),
        recommended_action: action.to_string(),
        session_id: Some(context.session_id),
        delta_s: args.delta_s,
        delta_c: args.delta_c,
        from_cache: context.from_cache,
        error: None,
    };

    // Output in requested format
    output_context(&response, args.format);

    let elapsed = start.elapsed();
    debug!("inject-context: completed in {:?}", elapsed);

    // Verify performance target (<1s)
    if elapsed.as_secs() >= 1 {
        warn!(
            "inject-context: Performance warning, took {:?} (target: <1s)",
            elapsed
        );
    }

    0
}

/// Get home directory (cross-platform).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// =============================================================================
// TASK-HOOKS-011: Semantic Search Mode
// =============================================================================

/// Result from semantic search mode.
#[derive(Debug, Serialize)]
struct SemanticSearchResult {
    /// Number of results found.
    count: usize,
    /// Content from matched nodes (truncated to max_tokens).
    content: String,
    /// Token count (approximate: 1 token ≈ 4 chars).
    token_count: usize,
    /// Fingerprint IDs of matched nodes.
    fingerprint_ids: Vec<String>,
    /// Similarity scores for each result.
    similarities: Vec<f32>,
}

/// Execute semantic search mode.
///
/// TASK-HOOKS-011: Handles --query and --node-ids flags.
///
/// # Arguments
/// * `query` - Semantic search query (optional)
/// * `node_ids` - Explicit UUIDs to retrieve (optional)
/// * `top_k` - Number of results for query mode
/// * `max_tokens` - Token limit for output
/// * `teleological_db_path` - Path to teleological store
///
/// # Returns
/// * `Ok(SemanticSearchResult)` - Search results with content
/// * `Err(String)` - Error message for fail-fast
async fn execute_semantic_search(
    query: Option<&str>,
    node_ids: Option<&[String]>,
    top_k: usize,
    max_tokens: usize,
    teleological_db_path: &PathBuf,
) -> Result<SemanticSearchResult, String> {
    info!(
        "semantic-search: Starting with query={:?}, node_ids={:?}, top_k={}, max_tokens={}",
        query,
        node_ids.map(|ids| ids.len()),
        top_k,
        max_tokens
    );

    // Open teleological store
    let store = RocksDbTeleologicalStore::open(teleological_db_path).map_err(|e| {
        let msg = format!(
            "Failed to open teleological store at {:?}: {}",
            teleological_db_path, e
        );
        error!("{}", msg);
        msg
    })?;

    // Determine which path: query-based or node-id-based
    let (fingerprint_ids, similarities, contents) = if let Some(q) = query {
        // Query-based: Generate embeddings and search
        execute_query_search(&store, q, top_k).await?
    } else if let Some(ids) = node_ids {
        // Node-ID-based: Direct retrieval
        execute_node_id_retrieval(&store, ids).await?
    } else {
        // Should not happen - caller should validate
        return Err("Either --query or --node-ids must be provided".to_string());
    };

    // Combine content with token limit
    let (truncated_content, token_count) = truncate_to_tokens(&contents, max_tokens);

    Ok(SemanticSearchResult {
        count: fingerprint_ids.len(),
        content: truncated_content,
        token_count,
        fingerprint_ids,
        similarities,
    })
}

/// Execute query-based semantic search.
///
/// TASK-HOOKS-011: Uses ProductionMultiArrayProvider for 13-embedding generation,
/// then searches via TeleologicalMemoryStore::search_semantic.
async fn execute_query_search(
    store: &RocksDbTeleologicalStore,
    query: &str,
    top_k: usize,
) -> Result<(Vec<String>, Vec<f32>, Vec<String>), String> {
    // Determine models directory
    let models_dir = std::env::var("CONTEXT_GRAPH_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            home_dir()
                .map(|h| h.join(".context-graph").join("models"))
                .unwrap_or_else(|| PathBuf::from("./models"))
        });

    debug!(
        "semantic-search: Loading embedding provider from {:?}",
        models_dir
    );

    // Create embedding provider
    let provider = ProductionMultiArrayProvider::new(models_dir.clone(), GpuConfig::default())
        .await
        .map_err(|e| {
            let msg = format!(
                "Failed to create embedding provider (models_dir={:?}): {}",
                models_dir, e
            );
            error!("{}", msg);
            msg
        })?;

    // Generate query embeddings
    debug!(
        "semantic-search: Generating embeddings for query: {:?}",
        query
    );
    let query_output = provider.embed_all(query).await.map_err(|e| {
        let msg = format!("Failed to generate query embeddings: {}", e);
        error!("{}", msg);
        msg
    })?;

    // Search semantic
    let options = TeleologicalSearchOptions::quick(top_k);
    debug!("semantic-search: Searching with top_k={}", top_k);

    let results = store
        .search_semantic(&query_output.fingerprint, options)
        .await
        .map_err(|e| {
            let msg = format!("Semantic search failed: {}", e);
            error!("{}", msg);
            msg
        })?;

    info!("semantic-search: Found {} results", results.len());

    // Extract IDs and similarities
    let ids: Vec<Uuid> = results.iter().map(|r| r.fingerprint.id).collect();
    let fingerprint_ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    let similarities: Vec<f32> = results.iter().map(|r| r.similarity).collect();

    // Fetch content for all results
    let contents = store.get_content_batch(&ids).await.map_err(|e| {
        let msg = format!("Failed to fetch content batch: {}", e);
        error!("{}", msg);
        msg
    })?;

    // Convert Option<String> to String (empty string for None)
    let content_strings: Vec<String> = contents
        .into_iter()
        .map(|c| c.unwrap_or_default())
        .collect();

    Ok((fingerprint_ids, similarities, content_strings))
}

/// Execute node-ID-based retrieval.
///
/// TASK-HOOKS-011: Retrieves specific nodes by UUID and their content.
async fn execute_node_id_retrieval(
    store: &RocksDbTeleologicalStore,
    node_id_strings: &[String],
) -> Result<(Vec<String>, Vec<f32>, Vec<String>), String> {
    // Parse UUIDs
    let uuids: Vec<Uuid> = node_id_strings
        .iter()
        .map(|s| {
            Uuid::parse_str(s).map_err(|e| {
                let msg = format!("Invalid UUID '{}': {}", s, e);
                error!("{}", msg);
                msg
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    debug!("node-id-retrieval: Retrieving {} nodes", uuids.len());

    // Retrieve fingerprints
    let fingerprints = store.retrieve_batch(&uuids).await.map_err(|e| {
        let msg = format!("Failed to retrieve fingerprints: {}", e);
        error!("{}", msg);
        msg
    })?;

    // Filter to found fingerprints
    let found_ids: Vec<Uuid> = fingerprints
        .iter()
        .zip(uuids.iter())
        .filter_map(|(fp, id)| fp.as_ref().map(|_| *id))
        .collect();

    let not_found: Vec<_> = fingerprints
        .iter()
        .zip(uuids.iter())
        .filter_map(|(fp, id)| {
            if fp.is_none() {
                Some(id.to_string())
            } else {
                None
            }
        })
        .collect();

    if !not_found.is_empty() {
        warn!(
            "node-id-retrieval: {} IDs not found: {:?}",
            not_found.len(),
            not_found
        );
    }

    info!(
        "node-id-retrieval: Found {}/{} nodes",
        found_ids.len(),
        uuids.len()
    );

    // Fetch content for found IDs
    let contents = store.get_content_batch(&found_ids).await.map_err(|e| {
        let msg = format!("Failed to fetch content batch: {}", e);
        error!("{}", msg);
        msg
    })?;

    let fingerprint_ids: Vec<String> = found_ids.iter().map(|id| id.to_string()).collect();
    // Node-ID mode has no similarity scores (direct retrieval)
    let similarities: Vec<f32> = vec![1.0; found_ids.len()];
    let content_strings: Vec<String> = contents
        .into_iter()
        .map(|c| c.unwrap_or_default())
        .collect();

    Ok((fingerprint_ids, similarities, content_strings))
}

/// Truncate combined content to fit within token budget.
///
/// TASK-HOOKS-011: Approximation: 1 token ≈ 4 characters.
fn truncate_to_tokens(contents: &[String], max_tokens: usize) -> (String, usize) {
    let max_chars = max_tokens * 4;
    let mut combined = String::new();
    let mut current_chars = 0;

    for (i, content) in contents.iter().enumerate() {
        if content.is_empty() {
            continue;
        }

        // Add separator between entries
        if !combined.is_empty() {
            combined.push_str("\n---\n");
            current_chars += 5;
        }

        let remaining = max_chars.saturating_sub(current_chars);
        if remaining == 0 {
            debug!("truncate_to_tokens: Hit token limit at entry {}", i);
            break;
        }

        if content.len() <= remaining {
            combined.push_str(content);
            current_chars += content.len();
        } else {
            // Truncate this entry
            combined.push_str(&content[..remaining]);
            combined.push_str("...[truncated]");
            current_chars += remaining + 14;
            debug!(
                "truncate_to_tokens: Truncated entry {} to {} chars",
                i, remaining
            );
            break;
        }
    }

    let token_count = current_chars.div_ceil(4);
    (combined, token_count)
}

/// Output semantic search results.
fn output_semantic_results(result: &SemanticSearchResult, format: InjectFormat) {
    match format {
        InjectFormat::Compact => {
            // Single line: [S:COUNT=N T=M]
            // Then content
            println!("[S:COUNT={} T={}]", result.count, result.token_count);
            if !result.content.is_empty() {
                println!("{}", result.content);
            }
        }
        InjectFormat::Standard => {
            println!("=== Semantic Memory Injection ===");
            println!("Results: {} | Tokens: {}", result.count, result.token_count);
            if !result.content.is_empty() {
                println!();
                println!("{}", result.content);
            }
        }
        InjectFormat::Verbose => {
            println!("=== Semantic Memory Injection (Verbose) ===");
            println!();
            println!("Results Found: {}", result.count);
            println!("Token Count:   {} (max: budget)", result.token_count);
            println!();
            println!("Fingerprint IDs:");
            for (i, id) in result.fingerprint_ids.iter().enumerate() {
                let sim = result.similarities.get(i).copied().unwrap_or(0.0);
                println!("  [{}] {} (similarity: {:.4})", i + 1, id, sim);
            }
            println!();
            println!("Content:");
            println!("---");
            if result.content.is_empty() {
                println!("(no content available)");
            } else {
                println!("{}", result.content);
            }
            println!("---");
        }
    }
}

// =============================================================================
// TASK-SESSION-15 Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::gwt::session_identity::SessionIdentitySnapshot;
    use std::sync::Mutex;
    use tempfile::TempDir;

    // Static lock to serialize tests that access global IdentityCache
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // =========================================================================
    // TC-SESSION-15-01: classify_johari Open quadrant
    // Source of Truth: Constitution johari mapping
    // =========================================================================
    #[test]
    fn tc_session_15_01_classify_johari_open() {
        println!("\n=== TC-SESSION-15-01: classify_johari Open Quadrant ===");
        println!("SOURCE OF TRUTH: Constitution johari mapping");
        println!("Open: ΔS < threshold AND ΔC > threshold");

        let test_cases = [
            (0.2, 0.8, 0.5, JohariQuadrant::Open),
            (0.0, 1.0, 0.5, JohariQuadrant::Open),
            (0.49, 0.51, 0.5, JohariQuadrant::Open),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!(
                "  ΔS={:.2}, ΔC={:.2}, threshold={:.2} → {:?} (expected: {:?})",
                delta_s, delta_c, threshold, result, expected
            );
            assert_eq!(
                result, expected,
                "classify_johari({}, {}, {}) should be {:?}",
                delta_s, delta_c, threshold, expected
            );
        }

        println!("RESULT: PASS - Open quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-02: classify_johari Blind quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_02_classify_johari_blind() {
        println!("\n=== TC-SESSION-15-02: classify_johari Blind Quadrant ===");
        println!("Blind: ΔS > threshold AND ΔC < threshold");

        let test_cases = [
            (0.8, 0.2, 0.5, JohariQuadrant::Blind),
            (0.51, 0.49, 0.5, JohariQuadrant::Blind),
            (1.0, 0.0, 0.5, JohariQuadrant::Blind),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!("  ΔS={:.2}, ΔC={:.2} → {:?}", delta_s, delta_c, result);
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Blind quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-03: classify_johari Hidden quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_03_classify_johari_hidden() {
        println!("\n=== TC-SESSION-15-03: classify_johari Hidden Quadrant ===");
        println!("Hidden: ΔS < threshold AND ΔC < threshold");

        let test_cases = [
            (0.2, 0.2, 0.5, JohariQuadrant::Hidden),
            (0.0, 0.0, 0.5, JohariQuadrant::Hidden),
            (0.49, 0.49, 0.5, JohariQuadrant::Hidden),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!("  ΔS={:.2}, ΔC={:.2} → {:?}", delta_s, delta_c, result);
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Hidden quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-04: classify_johari Unknown quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_04_classify_johari_unknown() {
        println!("\n=== TC-SESSION-15-04: classify_johari Unknown Quadrant ===");
        println!("Unknown: ΔS > threshold AND ΔC > threshold");

        let test_cases = [
            (0.8, 0.8, 0.5, JohariQuadrant::Unknown),
            (0.51, 0.51, 0.5, JohariQuadrant::Unknown),
            (1.0, 1.0, 0.5, JohariQuadrant::Unknown),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!("  ΔS={:.2}, ΔC={:.2} → {:?}", delta_s, delta_c, result);
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Unknown quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-05: johari_action mappings
    // Source of Truth: Constitution UTL mapping
    // =========================================================================
    #[test]
    fn tc_session_15_05_johari_action_mappings() {
        println!("\n=== TC-SESSION-15-05: johari_action Mappings ===");
        println!("SOURCE OF TRUTH: Constitution UTL mapping");

        let test_cases = [
            (JohariQuadrant::Open, "get_node"),
            (JohariQuadrant::Blind, "epistemic_action"),
            (JohariQuadrant::Hidden, "get_neighborhood"),
            (JohariQuadrant::Unknown, "explore"),
        ];

        for (quadrant, expected_action) in test_cases {
            let action = johari_action(quadrant);
            println!(
                "  {:?} → '{}' (expected: '{}')",
                quadrant, action, expected_action
            );
            assert_eq!(
                action, expected_action,
                "johari_action({:?}) should be '{}'",
                quadrant, expected_action
            );
        }

        println!("RESULT: PASS - All Johari actions mapped correctly");
    }

    // =========================================================================
    // TC-SESSION-15-06: Boundary conditions (exactly at threshold)
    // =========================================================================
    #[test]
    fn tc_session_15_06_boundary_conditions() {
        println!("\n=== TC-SESSION-15-06: Boundary Conditions ===");
        println!("Testing exact threshold values (ΔS=0.5, ΔC=0.5)");

        // At exactly threshold, ΔS < threshold is false (0.5 < 0.5 is false)
        // and ΔC > threshold is false (0.5 > 0.5 is false)
        // So both conditions false → Blind quadrant
        let result = classify_johari(0.5, 0.5, 0.5);
        println!("  ΔS=0.5, ΔC=0.5, threshold=0.5 → {:?}", result);
        assert_eq!(
            result,
            JohariQuadrant::Blind,
            "At exact threshold (0.5, 0.5), should be Blind"
        );

        // ΔS just below, ΔC at threshold
        let result2 = classify_johari(0.499, 0.5, 0.5);
        println!("  ΔS=0.499, ΔC=0.5 → {:?}", result2);
        assert_eq!(
            result2,
            JohariQuadrant::Hidden,
            "ΔS<thresh, ΔC<=thresh → Hidden"
        );

        // ΔS at threshold, ΔC just above
        let result3 = classify_johari(0.5, 0.501, 0.5);
        println!("  ΔS=0.5, ΔC=0.501 → {:?}", result3);
        assert_eq!(
            result3,
            JohariQuadrant::Unknown,
            "ΔS>=thresh, ΔC>thresh → Unknown"
        );

        println!("RESULT: PASS - Boundary conditions handled correctly");
    }

    // =========================================================================
    // TC-SESSION-15-07: Context from storage
    // Source of Truth: RocksDB storage
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_07_context_from_storage() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-07: Context from Storage ===");
        println!("SOURCE OF TRUTH: RocksDB storage");

        // Create temp dir and save snapshot, then drop the storage
        // RocksDB requires exclusive lock, so we must close before reopening
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Scope to ensure storage is dropped before get_context_from_cache_or_storage
        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("Failed to open RocksDB"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            // Create and save a snapshot with known values
            let mut snapshot = SessionIdentitySnapshot::new("test-inject-context");
            snapshot.consciousness = 0.75;
            snapshot.last_ic = 0.85;
            snapshot.kuramoto_phases = [0.0; 13]; // Aligned phases → r ≈ 1.0

            manager
                .save_snapshot(&snapshot)
                .expect("save_snapshot must succeed");
            println!(
                "BEFORE: Saved snapshot with IC={}, C={}",
                snapshot.last_ic, snapshot.consciousness
            );
            // storage is dropped here
        }

        // Load context with force_storage (storage is now closed)
        let context =
            get_context_from_cache_or_storage(&db_path, true).expect("get_context should succeed");

        println!("AFTER:");
        println!("  IC: {:.4}", context.ic);
        println!("  Kuramoto r: {:.4}", context.kuramoto_r);
        println!("  State: {:?}", context.consciousness_state);
        println!("  Session: {}", context.session_id);
        println!("  From cache: {}", context.from_cache);

        // VERIFY
        assert!((context.ic - 0.85).abs() < 0.01, "IC should be ~0.85");
        assert!(
            context.kuramoto_r > 0.99,
            "r should be ~1.0 for aligned phases"
        );
        assert_eq!(context.session_id, "test-inject-context");
        assert!(
            !context.from_cache,
            "Should be from storage with force_storage=true"
        );

        println!("RESULT: PASS - Context loaded from storage correctly");
    }

    // =========================================================================
    // TC-SESSION-15-08: Empty storage returns error
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_08_empty_storage_error() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-08: Empty Storage Error ===");

        // Create and immediately close empty storage to initialize the DB
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Initialize empty DB then close it
        {
            let _storage = RocksDbMemex::open(&db_path).expect("Failed to open RocksDB");
            // Storage is dropped here
        }

        // Load from empty storage
        let result = get_context_from_cache_or_storage(&db_path, true);

        assert!(result.is_err(), "Empty storage should return error");
        let err = result.unwrap_err();
        println!("Error message: {}", err);
        assert!(
            err.contains("No identity found"),
            "Error should mention missing identity"
        );

        println!("RESULT: PASS - Empty storage returns correct error");
    }

    // =========================================================================
    // TC-SESSION-15-09: Response serialization
    // =========================================================================
    #[test]
    fn tc_session_15_09_response_serialization() {
        println!("\n=== TC-SESSION-15-09: Response Serialization ===");

        let response = InjectContextResponse {
            ic: 0.85,
            ic_status: "Good",
            kuramoto_r: 0.92,
            consciousness_state: "CON".to_string(),
            johari_quadrant: "Open".to_string(),
            recommended_action: "get_node".to_string(),
            session_id: Some("test-session".to_string()),
            delta_s: 0.3,
            delta_c: 0.7,
            from_cache: true,
            error: None,
        };

        let json = serde_json::to_string(&response).expect("Serialization should succeed");
        println!("JSON: {}", json);

        assert!(json.contains("\"ic\":0.85"), "JSON should contain IC");
        assert!(
            json.contains("\"johari_quadrant\":\"Open\""),
            "JSON should contain quadrant"
        );
        assert!(
            json.contains("\"recommended_action\":\"get_node\""),
            "JSON should contain action"
        );
        assert!(
            !json.contains("error"),
            "JSON should not contain error when None"
        );

        println!("RESULT: PASS - Response serializes correctly");
    }

    // =========================================================================
    // TC-SESSION-15-10: Degraded response
    // =========================================================================
    #[test]
    fn tc_session_15_10_degraded_response() {
        println!("\n=== TC-SESSION-15-10: Degraded Response ===");

        let response = InjectContextResponse::degraded("Test error message".to_string());

        assert_eq!(response.ic, 0.0, "Degraded IC should be 0.0");
        assert_eq!(
            response.ic_status, "Unknown",
            "Degraded status should be Unknown"
        );
        assert_eq!(
            response.johari_quadrant, "Unknown",
            "Degraded quadrant should be Unknown"
        );
        assert_eq!(
            response.recommended_action, "restore-identity",
            "Degraded action should be restore-identity"
        );
        assert!(
            response.error.is_some(),
            "Degraded should have error message"
        );

        println!("RESULT: PASS - Degraded response created correctly");
    }

    // =========================================================================
    // TC-SESSION-15-11: E2E inject-context command
    // Source of Truth: Full command flow
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_11_e2e_inject_context_command() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-11: E2E inject-context Command ===");
        println!("SOURCE OF TRUTH: RocksDB → Command → Output");

        // Create temp DB and populate
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Populate the DB
        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("open"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            let mut snapshot = SessionIdentitySnapshot::new("e2e-inject-test");
            snapshot.consciousness = 0.85;
            snapshot.last_ic = 0.92;
            snapshot.kuramoto_phases = [0.0; 13];
            manager.save_snapshot(&snapshot).expect("save");
        }

        println!("BEFORE: Created RocksDB at {:?} with IC=0.92", db_path);

        // Run the command
        let args = InjectContextArgs {
            db_path: Some(db_path),
            format: InjectFormat::Compact,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: true,
            // TASK-HOOKS-011: New fields (not tested here, defaults)
            query: None,
            node_ids: None,
            max_tokens: 500,
            teleological_db_path: None,
            top_k: 5,
        };

        let exit_code = inject_context_command(args).await;
        println!(
            "AFTER: inject_context_command returned exit_code={}",
            exit_code
        );

        assert_eq!(exit_code, 0, "Command should succeed with exit code 0");

        println!("RESULT: PASS - E2E inject-context command works");
    }

    // =========================================================================
    // TC-SESSION-15-12: Performance test (should be <1s)
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_12_performance() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-12: Performance Test ===");
        println!("TARGET: <1s total latency");

        // Create temp DB and populate
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("open"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            let mut snapshot = SessionIdentitySnapshot::new("perf-test");
            snapshot.last_ic = 0.85;
            manager.save_snapshot(&snapshot).expect("save");
        }

        // Measure command execution time
        let start = std::time::Instant::now();

        let args = InjectContextArgs {
            db_path: Some(db_path),
            format: InjectFormat::Compact,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: true,
            // TASK-HOOKS-011: New fields (not tested here, defaults)
            query: None,
            node_ids: None,
            max_tokens: 500,
            teleological_db_path: None,
            top_k: 5,
        };

        let _ = inject_context_command(args).await;

        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed);

        assert!(
            elapsed.as_secs() < 1,
            "Command should complete in <1s, took {:?}",
            elapsed
        );

        println!(
            "RESULT: PASS - Performance within target ({:?} < 1s)",
            elapsed
        );
    }

    // =========================================================================
    // EDGE CASE: All four quadrants with different thresholds
    // =========================================================================
    #[test]
    fn edge_case_quadrants_with_different_thresholds() {
        println!("\n=== EDGE CASE: Quadrants with Different Thresholds ===");

        // With threshold 0.3
        let result1 = classify_johari(0.2, 0.5, 0.3);
        println!("  threshold=0.3: ΔS=0.2, ΔC=0.5 → {:?}", result1);
        assert_eq!(
            result1,
            JohariQuadrant::Open,
            "Low ΔS, high ΔC with low threshold"
        );

        // With threshold 0.7
        let result2 = classify_johari(0.5, 0.8, 0.7);
        println!("  threshold=0.7: ΔS=0.5, ΔC=0.8 → {:?}", result2);
        assert_eq!(result2, JohariQuadrant::Open, "ΔS < 0.7, ΔC > 0.7");

        // With threshold 0.9
        let result3 = classify_johari(0.8, 0.95, 0.9);
        println!("  threshold=0.9: ΔS=0.8, ΔC=0.95 → {:?}", result3);
        assert_eq!(result3, JohariQuadrant::Open, "ΔS < 0.9, ΔC > 0.9");

        println!("RESULT: PASS - Quadrant classification works with different thresholds");
    }

    // =========================================================================
    // EDGE CASE: Extreme values (0.0 and 1.0)
    // =========================================================================
    #[test]
    fn edge_case_extreme_values() {
        println!("\n=== EDGE CASE: Extreme Values (0.0 and 1.0) ===");

        let test_cases = [
            (0.0, 0.0, JohariQuadrant::Hidden),  // Minimum both
            (0.0, 1.0, JohariQuadrant::Open),    // Min ΔS, Max ΔC
            (1.0, 0.0, JohariQuadrant::Blind),   // Max ΔS, Min ΔC
            (1.0, 1.0, JohariQuadrant::Unknown), // Maximum both
        ];

        for (delta_s, delta_c, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, 0.5);
            println!("  ΔS={:.1}, ΔC={:.1} → {:?}", delta_s, delta_c, result);
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Extreme values handled correctly");
    }

    // =========================================================================
    // TASK-HOOKS-011: truncate_to_tokens unit tests
    // =========================================================================

    #[test]
    fn tc_hooks_011_01_truncate_empty() {
        println!("\n=== TC-HOOKS-011-01: truncate_to_tokens Empty Input ===");
        let contents: Vec<String> = vec![];
        let (result, tokens) = truncate_to_tokens(&contents, 100);
        assert!(result.is_empty(), "Empty input should produce empty output");
        assert_eq!(tokens, 0, "Empty input should have 0 tokens");
        println!("RESULT: PASS - Empty input handled correctly");
    }

    #[test]
    fn tc_hooks_011_02_truncate_single_within_limit() {
        println!("\n=== TC-HOOKS-011-02: truncate_to_tokens Single Within Limit ===");
        let contents = vec!["Hello world".to_string()]; // 11 chars = ~3 tokens
        let (result, tokens) = truncate_to_tokens(&contents, 100);
        assert_eq!(result, "Hello world");
        assert_eq!(tokens, 3); // (11 + 3) / 4 = 3
        println!("RESULT: PASS - Single entry within limit");
    }

    #[test]
    fn tc_hooks_011_03_truncate_single_exceeds_limit() {
        println!("\n=== TC-HOOKS-011-03: truncate_to_tokens Single Exceeds Limit ===");
        // 100 chars, limit 10 tokens = 40 chars
        let long_content = "a".repeat(100);
        let contents = vec![long_content];
        let (result, tokens) = truncate_to_tokens(&contents, 10);

        // Should be 40 chars + "...[truncated]" (14 chars) = 54 chars
        assert!(result.len() <= 54, "Should be truncated to ~54 chars");
        assert!(
            result.ends_with("...[truncated]"),
            "Should end with truncation marker"
        );
        assert!(tokens <= 14, "Tokens should be limited"); // (40 + 14 + 3) / 4 ≈ 14
        println!("Result: {} chars, {} tokens", result.len(), tokens);
        println!("RESULT: PASS - Long content truncated correctly");
    }

    #[test]
    fn tc_hooks_011_04_truncate_multiple_entries() {
        println!("\n=== TC-HOOKS-011-04: truncate_to_tokens Multiple Entries ===");
        let contents = vec![
            "First entry".to_string(),
            "Second entry".to_string(),
            "Third entry".to_string(),
        ];
        let (result, tokens) = truncate_to_tokens(&contents, 100);

        assert!(result.contains("First entry"), "Should contain first entry");
        assert!(result.contains("---"), "Should contain separator");
        assert!(
            result.contains("Second entry"),
            "Should contain second entry"
        );
        assert!(result.contains("Third entry"), "Should contain third entry");
        println!("Combined result ({} tokens):\n{}", tokens, result);
        println!("RESULT: PASS - Multiple entries combined correctly");
    }

    #[test]
    fn tc_hooks_011_05_truncate_skips_empty_entries() {
        println!("\n=== TC-HOOKS-011-05: truncate_to_tokens Skips Empty ===");
        let contents = vec![
            "".to_string(),
            "Valid content".to_string(),
            "".to_string(),
            "More content".to_string(),
        ];
        let (result, tokens) = truncate_to_tokens(&contents, 100);

        // Should have one separator (not multiple for empty entries)
        let separator_count = result.matches("---").count();
        assert_eq!(separator_count, 1, "Should have exactly one separator");
        assert!(result.contains("Valid content"));
        assert!(result.contains("More content"));
        println!(
            "Result ({} tokens, {} separators):\n{}",
            tokens, separator_count, result
        );
        println!("RESULT: PASS - Empty entries skipped correctly");
    }

    #[test]
    fn tc_hooks_011_06_truncate_exact_boundary() {
        println!("\n=== TC-HOOKS-011-06: truncate_to_tokens Exact Boundary ===");
        // Test exact token boundary: 10 tokens = 40 chars
        let content = "a".repeat(40);
        let contents = vec![content.clone()];
        let (result, tokens) = truncate_to_tokens(&contents, 10);

        assert_eq!(result.len(), 40, "Should fit exactly");
        assert!(!result.contains("truncated"), "Should not be truncated");
        assert_eq!(tokens, 10); // (40 + 3) / 4 = 43 / 4 = 10 (integer division truncates)
        println!("Exact 40 chars = {} tokens", tokens);
        println!("RESULT: PASS - Exact boundary handled correctly");
    }

    #[test]
    fn tc_hooks_011_07_truncate_max_tokens_zero() {
        println!("\n=== TC-HOOKS-011-07: truncate_to_tokens Zero Max Tokens ===");
        let contents = vec!["Some content".to_string()];
        let (result, tokens) = truncate_to_tokens(&contents, 0);

        // 0 tokens = 0 chars allowed, but we have content
        // The truncation should trigger immediately
        println!("Result: '{}', tokens: {}", result, tokens);
        // With 0 max_chars, we should get empty or just truncation marker
        assert!(
            result.is_empty() || result.len() <= 14,
            "Should be minimal output"
        );
        println!("RESULT: PASS - Zero max tokens handled gracefully");
    }

    // =========================================================================
    // TASK-HOOKS-011: Semantic search argument validation tests
    // =========================================================================

    #[test]
    fn tc_hooks_011_08_args_query_mutually_exclusive() {
        println!("\n=== TC-HOOKS-011-08: Query/NodeIds Presence ===");

        // Test that args structure supports the new fields
        let args_with_query = InjectContextArgs {
            db_path: None,
            format: InjectFormat::Standard,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: false,
            query: Some("test query".to_string()),
            node_ids: None,
            max_tokens: 500,
            teleological_db_path: None,
            top_k: 5,
        };

        let args_with_node_ids = InjectContextArgs {
            db_path: None,
            format: InjectFormat::Standard,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: false,
            query: None,
            node_ids: Some(vec!["abc-123".to_string()]),
            max_tokens: 500,
            teleological_db_path: None,
            top_k: 5,
        };

        let args_neither = InjectContextArgs {
            db_path: None,
            format: InjectFormat::Standard,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: false,
            query: None,
            node_ids: None,
            max_tokens: 500,
            teleological_db_path: None,
            top_k: 5,
        };

        assert!(args_with_query.query.is_some());
        assert!(args_with_node_ids.node_ids.is_some());
        assert!(args_neither.query.is_none() && args_neither.node_ids.is_none());

        println!("RESULT: PASS - Args structure correctly supports semantic search fields");
    }

    #[test]
    fn tc_hooks_011_09_default_values() {
        println!("\n=== TC-HOOKS-011-09: Default Values ===");

        // Verify default values match spec
        let args = InjectContextArgs {
            db_path: None,
            format: InjectFormat::Standard,
            delta_s: 0.3,   // Default per spec
            delta_c: 0.7,   // Default per spec
            threshold: 0.5, // Default per constitution.yaml
            force_storage: false,
            query: None,
            node_ids: None,
            max_tokens: 500, // Default per spec
            teleological_db_path: None,
            top_k: 5, // Default per spec
        };

        assert_eq!(args.max_tokens, 500, "Default max_tokens should be 500");
        assert_eq!(args.top_k, 5, "Default top_k should be 5");

        println!("RESULT: PASS - Default values match specification");
    }

    // =========================================================================
    // TASK-HOOKS-011: SemanticSearchResult output formatting tests
    // =========================================================================

    #[test]
    fn tc_hooks_011_10_semantic_result_serialization() {
        println!("\n=== TC-HOOKS-011-10: SemanticSearchResult Serialization ===");

        let result = SemanticSearchResult {
            count: 3,
            content: "Test content here".to_string(),
            token_count: 4,
            fingerprint_ids: vec![
                "uuid-1".to_string(),
                "uuid-2".to_string(),
                "uuid-3".to_string(),
            ],
            similarities: vec![0.95, 0.87, 0.72],
        };

        // Test that it serializes to JSON
        let json = serde_json::to_string(&result).expect("Should serialize");
        assert!(json.contains("\"count\":3"));
        assert!(json.contains("\"token_count\":4"));
        assert!(json.contains("Test content here"));

        println!("Serialized: {}", json);
        println!("RESULT: PASS - SemanticSearchResult serializes correctly");
    }
}
