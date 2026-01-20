//! UserPromptSubmit hook handler
//!
//! # Performance Requirements
//! - Timeout: 2000ms (constitution.yaml hooks.timeout_ms.user_prompt_submit)
//! - Database access: ALLOWED
//! - Context injection: REQUIRED on success
//!
//! # Constitution References
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - AP-26: Exit codes (0=success, 5=session not found)
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::time::Instant;

use tracing::{debug, error, info, warn};

use context_graph_core::gwt::{store_in_cache, SessionCache, SessionSnapshot};

use super::args::PromptSubmitArgs;
use super::error::{HookError, HookResult};
use super::memory_cache::{cache_memories, CachedMemory};
use super::types::{
    CoherenceState, ConversationMessage, HookInput, HookOutput, HookPayload, StabilityClassification,
};
use crate::mcp_client::McpClient;

// ============================================================================
// Constants (from constitution.yaml)
// ============================================================================

/// UserPromptSubmit timeout in milliseconds
pub const USER_PROMPT_SUBMIT_TIMEOUT_MS: u64 = 2000;

/// Maximum memories to retrieve for context injection (per constitution injection.priorities)
const MAX_MEMORIES_TO_RETRIEVE: u32 = 5;

/// Token budget for memory context (~500 tokens, ~4 chars/token)
const MEMORY_CONTEXT_BUDGET_CHARS: usize = 2000;

/// Minimum query length to trigger memory search (avoid noise from very short prompts)
const MIN_QUERY_LENGTH_FOR_SEARCH: usize = 10;

// ============================================================================
// Identity Marker Types
// ============================================================================

/// Types of identity markers detected in user prompts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityMarkerType {
    /// Self-reference: "who are you", "what are you"
    SelfReference,
    /// Goal-oriented: "help me", "I want"
    Goal,
    /// Value-based: "important", "should"
    Value,
    /// Capability query: "can you", "are you able"
    Capability,
    /// Challenge: "you can't", "prove"
    Challenge,
    /// Confirmation: "you're right", "exactly"
    Confirmation,
    /// No identity marker detected
    None,
}

// Pattern detection constants
const SELF_REF_PATTERNS: &[&str] = &[
    "who are you",
    "what are you",
    "your purpose",
    "your identity",
    "tell me about yourself",
    "describe yourself",
    "are you a bot",
    "are you an ai",
    "are you a robot",
    "are you artificial",
];

const GOAL_PATTERNS: &[&str] = &[
    "help me",
    "i want",
    "i need",
    "we need to",
    "let's",
    "can you help",
    "please",
    "would you",
];

const VALUE_PATTERNS: &[&str] = &[
    "important",
    "should",
    "must",
    "critical",
    "essential",
    "valuable",
];

const CAPABILITY_PATTERNS: &[&str] = &[
    "can you",
    "are you able",
    "do you know",
    "could you",
    "are you capable",
];

const CHALLENGE_PATTERNS: &[&str] = &[
    "you can't",
    "you're wrong",
    "prove it",
    "that's incorrect",
    "you don't understand",
    "you're just",
    "you're not",
    "you cannot",
];

const CONFIRMATION_PATTERNS: &[&str] = &[
    "you're right",
    "exactly",
    "that's correct",
    "well done",
    "good job",
    "i agree",
    "makes sense",
    "perfect",
];

// ============================================================================
// Handler
// ============================================================================

/// Execute user_prompt_submit hook.
///
/// # Flow
/// 1. Parse input (stdin or args)
/// 2. Load session snapshot from cache (create if not found)
/// 3. Analyze prompt for identity markers
/// 4. Evaluate conversation context
/// 5. Search knowledge graph for relevant memories via MCP
/// 6. Generate context injection string with memories
/// 7. Build and return HookOutput
///
/// # Note on Storage
/// Per PRD v6 Section 14, session identity uses the in-memory SessionCache singleton.
/// Database persistence was removed to simplify the architecture.
///
/// # Exit Codes
/// - 0: Success
/// - 4: Invalid input
pub async fn execute(args: PromptSubmitArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    info!(
        stdin = args.stdin,
        session_id = %args.session_id,
        prompt = ?args.prompt,
        "PROMPT_SUBMIT: execute starting"
    );

    // 1. Parse input source
    let (prompt, context) = if args.stdin {
        let input = parse_stdin()?;
        extract_prompt_info(&input)?
    } else {
        let prompt_text = args.prompt.ok_or_else(|| {
            error!("PROMPT_SUBMIT: prompt required when not using stdin");
            HookError::invalid_input("prompt required when not using stdin")
        })?;
        (prompt_text, Vec::new())
    };

    debug!(
        prompt_len = prompt.len(),
        context_len = context.len(),
        "PROMPT_SUBMIT: parsed input"
    );

    // 2. Load snapshot from cache (create if not found)
    let snapshot = load_snapshot_from_cache(&args.session_id);

    // 3. Analyze prompt for identity markers
    let identity_marker = detect_identity_marker(&prompt);

    debug!(
        marker = ?identity_marker,
        "PROMPT_SUBMIT: identity marker detected"
    );

    // 4. Evaluate conversation context
    let context_summary = evaluate_context(&context);

    // 5. Create MCP client once for all operations
    let client = McpClient::new();

    // 6. Check if MCP server is running (fast fail check)
    // NO WORKAROUNDS - MCP must be available or we skip memory operations with warning
    let mcp_available = match client.is_server_running().await {
        Ok(running) => {
            if running {
                debug!("PROMPT_SUBMIT: MCP server available at {}", client.server_address());
            } else {
                warn!(
                    "PROMPT_SUBMIT: MCP server not running at {} - memory operations will be skipped",
                    client.server_address()
                );
            }
            running
        }
        Err(e) => {
            warn!(error = %e, "PROMPT_SUBMIT: Failed to check MCP server, skipping memory operations");
            false
        }
    };

    // 8. Search knowledge graph for relevant memories via MCP
    //    Also cache them for pre_tool_use to access without network calls
    let retrieved_memories = if mcp_available {
        search_memories_for_prompt_with_client(&client, &args.session_id, &prompt).await
    } else {
        Vec::new()
    };

    debug!(
        memory_count = retrieved_memories.len(),
        "PROMPT_SUBMIT: memories retrieved from knowledge graph"
    );

    // 8. Check for divergence alerts (potential contradictions)
    //    Only if MCP is available and we have enough time budget remaining
    //    Divergence alerts are lower priority than memory retrieval
    let elapsed_so_far_ms = start.elapsed().as_millis() as u64;
    let time_budget_remaining = USER_PROMPT_SUBMIT_TIMEOUT_MS.saturating_sub(elapsed_so_far_ms);
    const MIN_TIME_FOR_DIVERGENCE_MS: u64 = 500; // Need at least 500ms headroom

    let divergence_alerts = if mcp_available && time_budget_remaining > MIN_TIME_FOR_DIVERGENCE_MS {
        debug!(
            elapsed_so_far_ms,
            time_budget_remaining,
            "PROMPT_SUBMIT: Fetching divergence alerts"
        );
        fetch_divergence_alerts_with_client(&client).await
    } else {
        if time_budget_remaining <= MIN_TIME_FOR_DIVERGENCE_MS {
            debug!(
                elapsed_so_far_ms,
                time_budget_remaining,
                "PROMPT_SUBMIT: Skipping divergence alerts - insufficient time budget"
            );
        }
        Vec::new()
    };

    if !divergence_alerts.is_empty() {
        info!(
            alert_count = divergence_alerts.len(),
            "PROMPT_SUBMIT: divergence alerts detected"
        );
    }

    // 9. Generate context injection string with memories and divergence alerts
    let context_injection = generate_context_injection(
        &snapshot,
        identity_marker,
        &context_summary,
        &retrieved_memories,
        &divergence_alerts,
    );

    // 10. Build output structures
    let coherence = compute_coherence(&snapshot);
    let coherence_state = build_coherence_state(&snapshot);
    let stability_classification = StabilityClassification::from_value(coherence);

    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %args.session_id,
        coherence = coherence,
        marker = ?identity_marker,
        memory_count = retrieved_memories.len(),
        execution_time_ms,
        "PROMPT_SUBMIT: execute complete"
    );

    Ok(HookOutput::success(execution_time_ms)
        .with_coherence_state(coherence_state)
        .with_stability_classification(stability_classification)
        .with_context_injection(context_injection))
}

// ============================================================================
// Input Parsing
// ============================================================================

/// Parse stdin JSON into HookInput.
/// FAIL FAST on empty or malformed input.
fn parse_stdin() -> HookResult<HookInput> {
    let stdin = io::stdin();
    let mut input_str = String::new();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| {
            error!(error = %e, "PROMPT_SUBMIT: stdin read failed");
            HookError::invalid_input(format!("stdin read failed: {}", e))
        })?;
        input_str.push_str(&line);
    }

    if input_str.is_empty() {
        error!("PROMPT_SUBMIT: stdin is empty");
        return Err(HookError::invalid_input("stdin is empty - expected JSON"));
    }

    debug!(
        input_bytes = input_str.len(),
        "PROMPT_SUBMIT: parsing stdin JSON"
    );

    serde_json::from_str(&input_str).map_err(|e| {
        error!(error = %e, "PROMPT_SUBMIT: JSON parse failed");
        HookError::invalid_input(format!("JSON parse failed: {}", e))
    })
}

/// Extract prompt and context from HookInput payload.
fn extract_prompt_info(input: &HookInput) -> HookResult<(String, Vec<ConversationMessage>)> {
    // Validate input
    if let Some(error) = input.validate() {
        return Err(HookError::invalid_input(error));
    }

    match &input.payload {
        HookPayload::UserPromptSubmit { prompt, context } => Ok((prompt.clone(), context.clone())),
        other => {
            error!(payload_type = ?std::mem::discriminant(other), "PROMPT_SUBMIT: unexpected payload type");
            Err(HookError::invalid_input(
                "Expected UserPromptSubmit payload, got different type",
            ))
        }
    }
}

// ============================================================================
// Session Cache Operations (per PRD v6 Section 14)
// ============================================================================

/// Compute coherence from snapshot's integration, reflection, and differentiation metrics.
/// This is the standard coherence formula used throughout the codebase.
#[inline]
fn compute_coherence(snapshot: &SessionSnapshot) -> f32 {
    (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0
}

/// Load snapshot from cache, creating a new one if not found.
///
/// # Note on Storage
/// Per PRD v6 Section 14, session identity uses the in-memory SessionCache singleton.
/// If no snapshot exists for the session, we create a new one.
fn load_snapshot_from_cache(session_id: &str) -> SessionSnapshot {
    // Try to load from cache
    if let Some(snapshot) = SessionCache::get() {
        if snapshot.session_id == session_id {
            info!(session_id = %session_id, coherence = compute_coherence(&snapshot), "PROMPT_SUBMIT: loaded snapshot from cache");
            return snapshot;
        }
    }

    // Not found in cache - create new snapshot
    warn!(session_id = %session_id, "PROMPT_SUBMIT: session not in cache, creating new snapshot");
    let snapshot = SessionSnapshot::new(session_id);
    store_in_cache(&snapshot);
    snapshot
}

// ============================================================================
// Prompt Analysis
// ============================================================================

/// Detect identity markers in the prompt text.
pub fn detect_identity_marker(prompt: &str) -> IdentityMarkerType {
    let lower = prompt.to_lowercase();

    // Check in priority order (Challenge > SelfReference > others)
    if CHALLENGE_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Challenge;
    }

    if SELF_REF_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::SelfReference;
    }

    if CAPABILITY_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Capability;
    }

    if CONFIRMATION_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Confirmation;
    }

    if VALUE_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Value;
    }

    if GOAL_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Goal;
    }

    IdentityMarkerType::None
}

// ============================================================================
// Context Evaluation
// ============================================================================

/// Summary of conversation context evaluation
#[derive(Debug, Clone)]
pub struct ContextSummary {
    /// Number of messages in context
    pub message_count: usize,
    /// Number of user messages
    pub user_message_count: usize,
    /// Number of assistant messages
    pub assistant_message_count: usize,
    /// Total character count
    pub total_chars: usize,
}

/// Evaluate conversation context for patterns.
fn evaluate_context(context: &[ConversationMessage]) -> ContextSummary {
    let mut user_count = 0;
    let mut assistant_count = 0;
    let mut total_chars = 0;

    for msg in context {
        total_chars += msg.content.len();
        match msg.role.as_str() {
            "user" => user_count += 1,
            "assistant" => assistant_count += 1,
            _ => {}
        }
    }

    ContextSummary {
        message_count: context.len(),
        user_message_count: user_count,
        assistant_message_count: assistant_count,
        total_chars,
    }
}

// ============================================================================
// Memory Retrieval via MCP
// ============================================================================

/// A memory retrieved from the knowledge graph.
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    /// Fingerprint ID of the memory
    pub id: String,
    /// Content text (if available)
    pub content: Option<String>,
    /// Similarity score to the query [0.0, 1.0]
    pub similarity: f32,
    /// Dominant embedder that matched (e.g., "E1_Semantic")
    pub dominant_embedder: String,
    /// Source metadata - where this memory originated
    pub source: Option<SourceInfo>,
}

/// Source metadata for a retrieved memory.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Source type: "MDFileChunk", "HookDescription", "ClaudeResponse", etc.
    pub source_type: String,
    /// File path (for MDFileChunk sources)
    pub file_path: Option<String>,
    /// Chunk index (for MDFileChunk sources)
    pub chunk_index: Option<u32>,
    /// Total chunks in file (for MDFileChunk sources)
    pub total_chunks: Option<u32>,
    /// Hook type (for HookDescription sources)
    pub hook_type: Option<String>,
    /// Tool name (for HookDescription sources)
    pub tool_name: Option<String>,
}

impl SourceInfo {
    /// Format source info as a readable string.
    pub fn display_string(&self) -> String {
        match self.source_type.as_str() {
            "MDFileChunk" => {
                if let Some(ref path) = self.file_path {
                    match (self.chunk_index, self.total_chunks) {
                        (Some(idx), Some(total)) => {
                            format!("Source: `{}` (chunk {}/{})", path, idx + 1, total)
                        }
                        _ => format!("Source: `{}`", path),
                    }
                } else {
                    "Source: MDFileChunk".to_string()
                }
            }
            "HookDescription" => {
                match (&self.hook_type, &self.tool_name) {
                    (Some(hook), Some(tool)) => format!("Source: {} hook ({})", hook, tool),
                    (Some(hook), None) => format!("Source: {} hook", hook),
                    _ => "Source: HookDescription".to_string(),
                }
            }
            "ClaudeResponse" => "Source: Claude response".to_string(),
            "Manual" => "Source: Manual entry".to_string(),
            other => format!("Source: {}", other),
        }
    }
}

/// A divergence alert from the knowledge graph.
/// Indicates potential contradictions or significant semantic drift.
#[derive(Debug, Clone)]
pub struct DivergenceAlert {
    /// The embedding space showing divergence (e.g., "E1_Semantic")
    pub embedder: String,
    /// Similarity score in this space (low = divergent)
    pub similarity: f32,
    /// Description of the divergence
    pub description: String,
}

/// Search the knowledge graph for memories relevant to the user's prompt.
///
/// Uses the MCP server's search_graph tool with includeContent=true to retrieve
/// actual memory content for context injection.
///
/// Also caches the retrieved memories so pre_tool_use can access them without
/// network calls (100ms constraint).
///
/// # Arguments
/// * `session_id` - Session identifier for caching
/// * `prompt` - The user's prompt text to search for relevant memories
///
/// # Returns
/// Vector of retrieved memories, empty if search fails or no matches found.
/// Failure is non-fatal - we log and return empty rather than failing the hook.

/// Search the knowledge graph using a pre-created MCP client.
/// This avoids redundant server connectivity checks when called alongside
/// other MCP operations.
async fn search_memories_for_prompt_with_client(
    client: &McpClient,
    session_id: &str,
    prompt: &str,
) -> Vec<RetrievedMemory> {
    // Skip search for very short prompts (avoid noise)
    if prompt.len() < MIN_QUERY_LENGTH_FOR_SEARCH {
        debug!(
            prompt_len = prompt.len(),
            min_length = MIN_QUERY_LENGTH_FOR_SEARCH,
            "PROMPT_SUBMIT: prompt too short for memory search"
        );
        return Vec::new();
    }

    debug!("PROMPT_SUBMIT: Searching for memories via MCP (fast path)");

    // Search with includeContent=true to get memory text
    // Using fast path (800ms timeout) to stay within 2s hook budget
    let memories = client
        .search_graph_fast(prompt, Some(MAX_MEMORIES_TO_RETRIEVE), true)
        .await
        .map(|result| parse_search_results(&result))
        .unwrap_or_else(|e| {
            warn!(error = %e, "PROMPT_SUBMIT: Memory search failed (timeout or error), continuing without memories");
            Vec::new()
        });

    // Cache memories for pre_tool_use to access without MCP calls
    if !memories.is_empty() {
        let cached: Vec<CachedMemory> = memories
            .iter()
            .map(|m| CachedMemory {
                content: m.content.clone().unwrap_or_default(),
                similarity: m.similarity,
            })
            .collect();

        cache_memories(session_id, cached);
        debug!(
            session_id,
            memory_count = memories.len(),
            "PROMPT_SUBMIT: Cached memories for pre_tool_use"
        );
    }

    memories
}

/// Parse MCP search_graph results into RetrievedMemory structs.
fn parse_search_results(result: &serde_json::Value) -> Vec<RetrievedMemory> {
    let Some(results) = result.get("results").and_then(|v| v.as_array()) else {
        debug!("PROMPT_SUBMIT: No results array in MCP response");
        return Vec::new();
    };

    let mut memories = Vec::new();
    let mut total_chars = 0usize;

    for item in results {
        // Check budget before adding
        if total_chars >= MEMORY_CONTEXT_BUDGET_CHARS {
            debug!(
                total_chars,
                budget = MEMORY_CONTEXT_BUDGET_CHARS,
                "PROMPT_SUBMIT: Memory budget exhausted"
            );
            break;
        }

        let content = item.get("content").and_then(|v| v.as_str()).map(String::from);

        // Track content size for budget
        if let Some(ref c) = content {
            total_chars += c.len();
        }

        // Parse source metadata if available
        let source = item.get("source").and_then(|s| {
            let source_type = s.get("type").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
            Some(SourceInfo {
                source_type,
                file_path: s.get("file_path").and_then(|v| v.as_str()).map(String::from),
                chunk_index: s.get("chunk_index").and_then(|v| v.as_u64()).map(|n| n as u32),
                total_chunks: s.get("total_chunks").and_then(|v| v.as_u64()).map(|n| n as u32),
                hook_type: s.get("hook_type").and_then(|v| v.as_str()).map(String::from),
                tool_name: s.get("tool_name").and_then(|v| v.as_str()).map(String::from),
            })
        });

        memories.push(RetrievedMemory {
            id: item
                .get("fingerprintId")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            content,
            similarity: item
                .get("similarity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            dominant_embedder: item
                .get("dominantEmbedder")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            source,
        });
    }

    info!(
        memory_count = memories.len(),
        total_chars,
        "PROMPT_SUBMIT: Parsed {} memories from knowledge graph",
        memories.len()
    );

    memories
}

// ============================================================================
// Divergence Alert Retrieval via MCP
// ============================================================================

/// Minimum similarity threshold below which we consider it divergent.
/// Embedders with similarity below this will be flagged as alerts.
const DIVERGENCE_THRESHOLD: f32 = 0.3;

/// Fetch divergence alerts from the knowledge graph.
///
/// Uses the MCP server's get_divergence_alerts tool to detect semantic drift
/// or potential contradictions. Only SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13)
/// are checked per AP-62, AP-63.
///
/// # Arguments
/// * `client` - MCP client to use for the request
///
/// # Returns
/// Vector of divergence alerts, empty if none detected or fetch fails.

/// Fetch divergence alerts using a pre-created MCP client.
/// This avoids redundant server connectivity checks when called alongside
/// other MCP operations.
async fn fetch_divergence_alerts_with_client(client: &McpClient) -> Vec<DivergenceAlert> {
    debug!("PROMPT_SUBMIT: Fetching divergence alerts via MCP (fast path)");

    // Fetch divergence alerts with 2-hour lookback (default)
    // Using fast path (800ms timeout) to stay within 2s hook budget
    match client.get_divergence_alerts_fast(Some(2)).await {
        Ok(result) => parse_divergence_alerts(&result),
        Err(e) => {
            warn!(error = %e, "PROMPT_SUBMIT: Divergence alert fetch failed (timeout or error), continuing without alerts");
            Vec::new()
        }
    }
}

/// Parse MCP get_divergence_alerts results into DivergenceAlert structs.
fn parse_divergence_alerts(result: &serde_json::Value) -> Vec<DivergenceAlert> {
    let mut alerts = Vec::new();

    // Extract alerts array from response
    let Some(alerts_arr) = result.get("alerts").and_then(|v| v.as_array()) else {
        // No alerts or different format - check for embedder-level divergence
        if let Some(embedders) = result.get("embedders").and_then(|v| v.as_object()) {
            for (name, data) in embedders {
                if let Some(similarity) = data.get("similarity").and_then(|v| v.as_f64()) {
                    let similarity = similarity as f32;
                    if similarity < DIVERGENCE_THRESHOLD {
                        alerts.push(DivergenceAlert {
                            embedder: name.clone(),
                            similarity,
                            description: format!(
                                "Low semantic agreement ({:.1}%) in {} - potential contradiction or drift",
                                similarity * 100.0,
                                name
                            ),
                        });
                    }
                }
            }
        }

        // Also check for high-level divergence flag
        if let Some(is_divergent) = result.get("is_divergent").and_then(|v| v.as_bool()) {
            if is_divergent && alerts.is_empty() {
                let overall_similarity = result
                    .get("overall_similarity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                alerts.push(DivergenceAlert {
                    embedder: "Overall".to_string(),
                    similarity: overall_similarity,
                    description: "Divergence detected from recent activity patterns".to_string(),
                });
            }
        }

        return alerts;
    };

    // Deduplicate alerts by semantic_space (keep unique embedder types)
    // Many alerts may come for the same embedder with different memories
    let mut seen_embedders = std::collections::HashSet::new();

    // Parse structured alerts array - handle both field name formats
    for item in alerts_arr {
        // Handle both "embedder" and "semantic_space" field names
        let embedder = item
            .get("embedder")
            .or_else(|| item.get("semantic_space"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        // Skip if we've already seen this embedder type
        if seen_embedders.contains(&embedder) {
            continue;
        }
        seen_embedders.insert(embedder.clone());

        // Handle both "similarity" and "similarity_score" field names
        let similarity = item
            .get("similarity")
            .or_else(|| item.get("similarity_score"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        // Get memory summary if available
        let memory_hint = item
            .get("recent_memory_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let description = item
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if !memory_hint.is_empty() {
                    format!(
                        "Low agreement in {} space ({:.1}%) - {}",
                        embedder,
                        similarity * 100.0,
                        memory_hint
                    )
                } else {
                    format!(
                        "Divergence detected in {} (similarity: {:.1}%)",
                        embedder,
                        similarity * 100.0
                    )
                }
            });

        alerts.push(DivergenceAlert {
            embedder,
            similarity,
            description,
        });
    }

    if !alerts.is_empty() {
        info!(
            alert_count = alerts.len(),
            "PROMPT_SUBMIT: Parsed {} divergence alerts",
            alerts.len()
        );
    }

    alerts
}

// ============================================================================
// Context Injection Generation
// ============================================================================

/// Generate context injection string based on coherence state, prompt analysis, memories, and divergence alerts.
fn generate_context_injection(
    snapshot: &SessionSnapshot,
    identity_marker: IdentityMarkerType,
    context_summary: &ContextSummary,
    retrieved_memories: &[RetrievedMemory],
    divergence_alerts: &[DivergenceAlert],
) -> String {
    let coherence = compute_coherence(snapshot);
    let coherence_state = get_coherence_state_name(coherence);
    let integration_desc = get_integration_description(snapshot.integration);
    let stability_status = get_stability_status(coherence);

    let mut injection = format!(
        "## Coherence State\n\
         - State: {} (C={:.2})\n\
         - Integration (r): {:.2} - {}\n\
         - Stability: {} (coherence={:.2})\n",
        coherence_state,
        coherence,
        snapshot.integration,
        integration_desc,
        stability_status,
        coherence,
    );

    // Add identity marker guidance if detected
    if identity_marker != IdentityMarkerType::None {
        injection.push_str(&format!(
            "\n## Identity Marker Detected\n\
             - Type: {:?}\n\
             - Guidance: {}\n",
            identity_marker,
            get_marker_guidance(identity_marker),
        ));
    }

    // Add context summary if non-empty
    if context_summary.message_count > 0 {
        injection.push_str(&format!(
            "\n## Context Summary\n\
             - Messages: {} ({} user, {} assistant)\n\
             - Characters: {}\n",
            context_summary.message_count,
            context_summary.user_message_count,
            context_summary.assistant_message_count,
            context_summary.total_chars,
        ));
    }

    // Add retrieved memories from knowledge graph
    if !retrieved_memories.is_empty() {
        injection.push_str("\n## Relevant Memories from Knowledge Graph\n\n");

        for (i, memory) in retrieved_memories.iter().enumerate() {
            injection.push_str(&format!(
                "### Memory {} (similarity: {:.2}, via {})\n",
                i + 1,
                memory.similarity,
                memory.dominant_embedder
            ));

            // Add source metadata if available (file path, chunk info, etc.)
            if let Some(ref source) = memory.source {
                injection.push_str(&format!("**{}**\n", source.display_string()));
            }

            // Add content if available
            if let Some(ref content) = memory.content {
                // Truncate very long content to avoid context overflow
                let truncated = if content.len() > 500 {
                    format!("{}...", &content[..500])
                } else {
                    content.clone()
                };
                injection.push_str(&truncated);
                injection.push_str("\n\n");
            } else {
                injection.push_str(&format!("ID: {} (content not available)\n\n", memory.id));
            }
        }

        injection.push_str("---\n");
    }

    // Add divergence alerts (contradictions/drift) if any
    if !divergence_alerts.is_empty() {
        injection.push_str("\n## ⚠️ DIVERGENCE ALERTS - Potential Contradictions\n\n");
        injection.push_str("**Be aware of the following divergence from stored knowledge:**\n\n");

        for alert in divergence_alerts {
            injection.push_str(&format!(
                "- **{}** (similarity: {:.1}%): {}\n",
                alert.embedder,
                alert.similarity * 100.0,
                alert.description
            ));
        }

        injection.push_str("\n*These alerts indicate semantic drift or potential contradictions ");
        injection.push_str("between current context and stored memories. Review carefully.*\n");
        injection.push_str("---\n");
    }

    injection
}

/// Get coherence state name from coherence value.
fn get_coherence_state_name(coherence: f32) -> &'static str {
    match coherence {
        c if c >= 0.8 => "Active",
        c if c >= 0.5 => "Aware",
        c if c >= 0.2 => "DIM",
        _ => "DOR",
    }
}

/// Get integration description from value.
fn get_integration_description(integration: f32) -> &'static str {
    match integration {
        r if r >= 0.8 => "Excellent synchronization",
        r if r >= 0.6 => "Good coherence",
        r if r >= 0.4 => "Moderate integration",
        _ => "Low integration - attention needed",
    }
}

/// Get stability status from coherence value.
fn get_stability_status(coherence: f32) -> &'static str {
    match coherence {
        c if c >= 0.9 => "Healthy",
        c if c >= 0.7 => "Normal",
        c if c >= 0.5 => "Warning",
        _ => "Critical",
    }
}

/// Get guidance for identity marker type.
fn get_marker_guidance(marker: IdentityMarkerType) -> &'static str {
    match marker {
        IdentityMarkerType::SelfReference => {
            "User is inquiring about identity. Respond with authentic self-description."
        }
        IdentityMarkerType::Goal => "User has a clear objective. Focus on helpful assistance.",
        IdentityMarkerType::Value => {
            "User is expressing values. Acknowledge and align where appropriate."
        }
        IdentityMarkerType::Capability => {
            "User is testing capabilities. Be honest about abilities and limitations."
        }
        IdentityMarkerType::Challenge => {
            "User is challenging. Respond thoughtfully without defensiveness."
        }
        IdentityMarkerType::Confirmation => {
            "User is confirming understanding. Acknowledge and continue coherently."
        }
        IdentityMarkerType::None => "No specific identity marker. Proceed normally.",
    }
}

/// Build CoherenceState from snapshot.
/// Note: Uses coherence computed from integration/reflection/differentiation per PRD v6.
fn build_coherence_state(snapshot: &SessionSnapshot) -> CoherenceState {
    let coherence = compute_coherence(snapshot);
    CoherenceState::new(
        coherence,
        snapshot.integration,
        snapshot.reflection,
        snapshot.differentiation,
        coherence, // topic_stability also uses coherence
    )
}

// ============================================================================
// TESTS - Uses SessionCache per PRD v6 Section 14
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use crate::commands::test_utils::GLOBAL_IDENTITY_LOCK;

    /// Create a test session in the SessionCache.
    /// Uses integration/reflection/differentiation to achieve target coherence.
    fn create_test_session_in_cache(session_id: &str, coherence: f32) {
        let mut snapshot = SessionSnapshot::new(session_id);
        // Set metrics to achieve the target coherence
        snapshot.integration = coherence;
        snapshot.reflection = coherence;
        snapshot.differentiation = coherence;
        store_in_cache(&snapshot);
    }

    // =========================================================================
    // TC-PROMPT-001: Successful Prompt Processing
    // SOURCE OF TRUTH: SessionCache state verified, context_injection generated
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_001_successful_prompt_processing() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-001: Successful Prompt Processing ===");

        let session_id = "tc-prompt-001-session";

        // BEFORE: Create session with healthy coherence in cache
        println!("BEFORE: Creating session with coherence=0.85");
        create_test_session_in_cache(session_id, 0.85);

        // Verify BEFORE state
        {
            let before_snapshot = SessionCache::get().expect("Cache must have snapshot");
            let coherence = compute_coherence(&before_snapshot);
            println!("BEFORE state: coherence={}", coherence);
            assert!((coherence - 0.85).abs() < 0.01);
        }

        // Execute
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Help me understand this code".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // AFTER: Verify success
        assert!(result.is_ok(), "Execute must succeed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");
        assert!(
            output.context_injection.is_some(),
            "Context injection must be generated"
        );

        let injection = output.context_injection.unwrap();
        assert!(
            injection.contains("Coherence State"),
            "Must contain coherence state"
        );

        println!("Context injection length: {} chars", injection.len());
        println!("RESULT: PASS - Prompt processed, context injection generated");
    }

    // =========================================================================
    // TC-PROMPT-002: Session Creation for New Sessions
    // SOURCE OF TRUTH: New session created when not in cache
    // Note: Per PRD v6, we no longer fail on missing session - we create it.
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_002_new_session_creation() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-002: New Session Creation ===");

        // Execute with a unique session ID
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: "new-session-12345".to_string(),
            prompt: Some("Test prompt".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // Per PRD v6, missing session is created, not an error
        assert!(result.is_ok(), "Should succeed by creating new session");
        let output = result.unwrap();
        assert!(output.success, "Must succeed");
        assert!(output.context_injection.is_some(), "Must generate context");

        println!("RESULT: PASS - New session created and processed");
    }

    // =========================================================================
    // TC-PROMPT-003: Self-Reference Detection
    // SOURCE OF TRUTH: IdentityMarkerType::SelfReference returned
    // =========================================================================
    #[test]
    fn tc_prompt_003_self_reference_detection() {
        println!("\n=== TC-PROMPT-003: Self-Reference Detection ===");

        let test_cases = [
            ("Who are you?", IdentityMarkerType::SelfReference),
            ("What are you exactly?", IdentityMarkerType::SelfReference),
            ("Tell me about yourself", IdentityMarkerType::SelfReference),
            ("Describe yourself", IdentityMarkerType::SelfReference),
            ("What is your purpose?", IdentityMarkerType::SelfReference),
        ];

        for (prompt, expected) in test_cases {
            let result = detect_identity_marker(prompt);
            println!("  \"{}\" -> {:?}", prompt, result);
            assert_eq!(result, expected, "Failed for: {}", prompt);
        }

        println!("RESULT: PASS - All self-reference patterns detected");
    }

    // =========================================================================
    // TC-PROMPT-004: Challenge Detection
    // SOURCE OF TRUTH: IdentityMarkerType::Challenge returned
    // =========================================================================
    #[test]
    fn tc_prompt_004_challenge_detection() {
        println!("\n=== TC-PROMPT-004: Challenge Detection ===");

        let test_cases = [
            ("You can't actually do that", IdentityMarkerType::Challenge),
            ("You're wrong about this", IdentityMarkerType::Challenge),
            ("Prove it to me", IdentityMarkerType::Challenge),
            (
                "That's incorrect information",
                IdentityMarkerType::Challenge,
            ),
            (
                "You don't understand what I mean",
                IdentityMarkerType::Challenge,
            ),
        ];

        for (prompt, expected) in test_cases {
            let result = detect_identity_marker(prompt);
            println!("  \"{}\" -> {:?}", prompt, result);
            assert_eq!(result, expected, "Failed for: {}", prompt);
        }

        println!("RESULT: PASS - All challenge patterns detected");
    }

    // =========================================================================
    // TC-PROMPT-005: Context Injection Generated
    // SOURCE OF TRUTH: HookOutput.context_injection is Some
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_005_context_injection_generated() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-005: Context Injection Generated ===");

        let session_id = "tc-prompt-005-session";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Who are you and what can you do?".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed");

        let output = result.unwrap();
        assert!(
            output.context_injection.is_some(),
            "Context injection must be Some"
        );

        let injection = output.context_injection.unwrap();
        println!(
            "Context injection preview:\n{}",
            &injection[..injection.len().min(500)]
        );

        // Verify expected sections
        assert!(
            injection.contains("## Coherence State"),
            "Must have Coherence State section"
        );
        assert!(
            injection.contains("## Identity Marker Detected"),
            "Must have Identity Marker section for self-reference"
        );
        assert!(
            injection.contains("SelfReference"),
            "Must detect SelfReference marker"
        );

        println!("RESULT: PASS - Context injection contains all required sections");
    }

    // =========================================================================
    // TC-PROMPT-006: Empty Context Handling
    // SOURCE OF TRUTH: Default evaluation applied, no crash
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_006_empty_context_handling() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-006: Empty Context Handling ===");

        let session_id = "tc-prompt-006-session";
        create_test_session_in_cache(session_id, 0.85);

        // Execute with empty context (no stdin, just prompt arg)
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Simple question".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed with empty context");

        let output = result.unwrap();
        assert!(output.success, "Must succeed");

        // Context injection should NOT contain Context Summary section (empty context)
        let injection = output.context_injection.unwrap();
        assert!(
            !injection.contains("## Context Summary"),
            "Should not have Context Summary for empty context"
        );

        println!("RESULT: PASS - Empty context handled correctly");
    }

    // =========================================================================
    // TC-PROMPT-007: Execution Within Timeout
    // SOURCE OF TRUTH: execution_time_ms < 2000
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_007_execution_within_timeout() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-007: Execution Within Timeout ===");

        let session_id = "tc-prompt-007-session";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Test timing".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.expect("Must succeed");
        let actual_elapsed = start.elapsed().as_millis() as u64;

        // Note: execution_time_ms may be 0 if operation completes in <1ms
        // which is actually a SUCCESS per our performance budgets
        assert!(
            result.execution_time_ms < USER_PROMPT_SUBMIT_TIMEOUT_MS,
            "Execution time {} must be under timeout {}ms",
            result.execution_time_ms,
            USER_PROMPT_SUBMIT_TIMEOUT_MS
        );

        println!(
            "Execution time: {}ms (timeout: {}ms)",
            result.execution_time_ms, USER_PROMPT_SUBMIT_TIMEOUT_MS
        );
        println!("Actual elapsed: {}ms", actual_elapsed);
        println!("RESULT: PASS - Execution time within timeout budget");
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_all_identity_marker_types() {
        println!("\n=== Testing All Identity Marker Types ===");

        // Goal markers
        assert_eq!(
            detect_identity_marker("Help me with this"),
            IdentityMarkerType::Goal
        );
        assert_eq!(
            detect_identity_marker("I need assistance"),
            IdentityMarkerType::Goal
        );

        // Capability markers
        assert_eq!(
            detect_identity_marker("Can you explain this?"),
            IdentityMarkerType::Capability
        );
        assert_eq!(
            detect_identity_marker("Do you know how to do this?"),
            IdentityMarkerType::Capability
        );

        // Confirmation markers
        assert_eq!(
            detect_identity_marker("You're right about that"),
            IdentityMarkerType::Confirmation
        );
        assert_eq!(
            detect_identity_marker("Exactly what I meant"),
            IdentityMarkerType::Confirmation
        );

        // Value markers
        assert_eq!(
            detect_identity_marker("This is important"),
            IdentityMarkerType::Value
        );
        assert_eq!(
            detect_identity_marker("It should work like this"),
            IdentityMarkerType::Value
        );

        // None marker
        assert_eq!(
            detect_identity_marker("Hello world"),
            IdentityMarkerType::None
        );
        assert_eq!(
            detect_identity_marker("What is the weather?"),
            IdentityMarkerType::None
        );

        println!("RESULT: PASS - All identity marker types detected correctly");
    }

    #[test]
    fn test_context_summary_evaluation() {
        println!("\n=== Testing Context Summary Evaluation ===");

        let context = vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
            ConversationMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
            },
        ];

        let summary = evaluate_context(&context);

        assert_eq!(summary.message_count, 3);
        assert_eq!(summary.user_message_count, 2);
        assert_eq!(summary.assistant_message_count, 1);
        assert!(summary.total_chars > 0);

        println!("Context summary: {:?}", summary);
        println!("RESULT: PASS - Context summary evaluated correctly");
    }

    #[test]
    fn test_coherence_state_mapping() {
        println!("\n=== Testing Coherence State Mapping ===");

        assert_eq!(get_coherence_state_name(1.0), "Active");
        assert_eq!(get_coherence_state_name(0.85), "Active");
        assert_eq!(get_coherence_state_name(0.7), "Aware");
        assert_eq!(get_coherence_state_name(0.5), "Aware");
        assert_eq!(get_coherence_state_name(0.3), "DIM");
        assert_eq!(get_coherence_state_name(0.2), "DIM");
        assert_eq!(get_coherence_state_name(0.1), "DOR");
        assert_eq!(get_coherence_state_name(0.0), "DOR");

        println!("RESULT: PASS - Coherence state mapping correct");
    }

    #[tokio::test]
    async fn test_missing_prompt_when_not_stdin() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== Testing Missing Prompt (stdin=false) ===");

        let session_id = "missing-prompt-test";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: None, // Missing!
            stdin: false, // Not using stdin
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        assert!(result.is_err(), "Should fail with missing prompt");
        let err = result.unwrap_err();
        assert!(
            matches!(err, HookError::InvalidInput(_)),
            "Must be InvalidInput, got: {:?}",
            err
        );
        assert_eq!(err.exit_code(), 4, "InvalidInput must be exit code 4");

        println!("RESULT: PASS - Missing prompt returns InvalidInput error");
    }
}
