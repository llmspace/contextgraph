//! Tool names as constants for dispatch matching.

// ========== CORE TOOLS ==========

pub const INJECT_CONTEXT: &str = "inject_context";
pub const STORE_MEMORY: &str = "store_memory";
pub const GET_MEMETIC_STATUS: &str = "get_memetic_status";
pub const GET_GRAPH_MANIFEST: &str = "get_graph_manifest";
pub const SEARCH_GRAPH: &str = "search_graph";
pub const UTL_STATUS: &str = "utl_status";

// ========== GWT TOOLS (TASK-GWT-001) ==========
// Note: Consciousness tools removed in PRD v6.
// Topic-based coherence scoring replaces consciousness calculations.
// Phase synchronization is handled by per-space clustering coordination.

/// TASK-GWT-001: Get Global Workspace status (active memory, competing, broadcast)
pub const GET_WORKSPACE_STATUS: &str = "get_workspace_status";
/// TASK-GWT-001: Get Topic Profile state (purpose vector, topic stability)
pub const GET_EGO_STATE: &str = "get_ego_state";
/// TASK-GWT-001: Trigger workspace broadcast with a memory
pub const TRIGGER_WORKSPACE_BROADCAST: &str = "trigger_workspace_broadcast";

// ========== UTL TOOLS (TASK-UTL-P1-001) ==========

/// TASK-UTL-P1-001: Compute per-embedder delta-S and aggregate delta-C
pub const COMPUTE_DELTA_SC: &str = "gwt/compute_delta_sc";

// ========== ADAPTIVE THRESHOLD CALIBRATION (ATC) TOOLS (TASK-ATC-001) ==========

/// TASK-ATC-001: Get current ATC threshold status
pub const GET_THRESHOLD_STATUS: &str = "get_threshold_status";
/// TASK-ATC-001: Get calibration quality metrics (ECE, MCE, Brier)
pub const GET_CALIBRATION_METRICS: &str = "get_calibration_metrics";
/// TASK-ATC-001: Manually trigger recalibration at a specific level
pub const TRIGGER_RECALIBRATION: &str = "trigger_recalibration";

// ========== DREAM TOOLS (TASK-DREAM-MCP) ==========

/// TASK-DREAM-MCP: Manually trigger a dream consolidation cycle
pub const TRIGGER_DREAM: &str = "trigger_dream";
/// TASK-DREAM-MCP: Get current dream system status
pub const GET_DREAM_STATUS: &str = "get_dream_status";
/// TASK-DREAM-MCP: Abort current dream cycle
pub const ABORT_DREAM: &str = "abort_dream";
/// TASK-DREAM-MCP: Get shortcut candidates from amortized learning
pub const GET_AMORTIZED_SHORTCUTS: &str = "get_amortized_shortcuts";
/// TASK-37: Get GPU utilization and dream eligibility status
pub const GET_GPU_STATUS: &str = "get_gpu_status";
/// TASK-S01: Trigger mental_check workflow based on entropy threshold
pub const TRIGGER_MENTAL_CHECK: &str = "trigger_mental_check";
/// TASK-S02: Get current trigger configuration
pub const GET_TRIGGER_CONFIG: &str = "get_trigger_config";
/// TASK-S03: Get trigger history
pub const GET_TRIGGER_HISTORY: &str = "get_trigger_history";

// ========== NEUROMODULATION TOOLS (TASK-NEUROMOD-MCP) ==========

/// TASK-NEUROMOD-MCP: Get all 4 neuromodulator levels
pub const GET_NEUROMODULATION_STATE: &str = "get_neuromodulation_state";
/// TASK-NEUROMOD-MCP: Adjust a specific modulator
pub const ADJUST_NEUROMODULATOR: &str = "adjust_neuromodulator";

// ========== STEERING TOOLS (TASK-STEERING-001) ==========

/// TASK-STEERING-001: Get steering feedback from Gardener, Curator, Assessor
pub const GET_STEERING_FEEDBACK: &str = "get_steering_feedback";

// ========== CAUSAL INFERENCE TOOLS (TASK-CAUSAL-001) ==========

/// TASK-CAUSAL-001: Perform omni-directional causal inference
pub const OMNI_INFER: &str = "omni_infer";

// ========== TELEOLOGICAL TOOLS (TELEO-007 through TELEO-011) ==========

/// TELEO-007: Cross-correlation search across all 13 embedders
pub const SEARCH_TELEOLOGICAL: &str = "search_teleological";
/// TELEO-008: Compute full 13-embedder teleological vector
pub const COMPUTE_TELEOLOGICAL_VECTOR: &str = "compute_teleological_vector";
/// TELEO-009: Fuse embeddings using synergy matrix
pub const FUSE_EMBEDDINGS: &str = "fuse_embeddings";
/// TELEO-010: Adaptively update synergy matrix from feedback
pub const UPDATE_SYNERGY_MATRIX: &str = "update_synergy_matrix";
/// TELEO-011: CRUD operations for task-specific teleological profiles
pub const MANAGE_TELEOLOGICAL_PROFILE: &str = "manage_teleological_profile";

// ========== AUTONOMOUS TOOLS (TASK-AUTONOMOUS-MCP) ==========

/// TASK-AUTONOMOUS-MCP: Get current drift state and history
pub const GET_ALIGNMENT_DRIFT: &str = "get_alignment_drift";
/// TASK-FIX-002/NORTH-010: Get historical drift measurements and trend data
pub const GET_DRIFT_HISTORY: &str = "get_drift_history";
/// TASK-AUTONOMOUS-MCP: Manually trigger drift correction
pub const TRIGGER_DRIFT_CORRECTION: &str = "trigger_drift_correction";
/// TASK-AUTONOMOUS-MCP: Get memories that are candidates for pruning
pub const GET_PRUNING_CANDIDATES: &str = "get_pruning_candidates";
/// TASK-AUTONOMOUS-MCP: Trigger memory consolidation
pub const TRIGGER_CONSOLIDATION: &str = "trigger_consolidation";
/// TASK-AUTONOMOUS-MCP: Discover potential sub-goals from memory clusters
pub const DISCOVER_SUB_GOALS: &str = "discover_sub_goals";
/// TASK-AUTONOMOUS-MCP: Get comprehensive autonomous system status
pub const GET_AUTONOMOUS_STATUS: &str = "get_autonomous_status";
/// SPEC-AUTONOMOUS-001: Get Meta-UTL learner state (accuracy, domain_stats, lambda_weights)
pub const GET_LEARNER_STATE: &str = "get_learner_state";
/// SPEC-AUTONOMOUS-001: Record learning outcome for Meta-UTL prediction
pub const OBSERVE_OUTCOME: &str = "observe_outcome";
/// SPEC-AUTONOMOUS-001: Execute pruning on identified candidates (NORTH-012)
pub const EXECUTE_PRUNE: &str = "execute_prune";
/// SPEC-AUTONOMOUS-001: Get system-wide health status (UTL, GWT, Dream, Storage)
pub const GET_HEALTH_STATUS: &str = "get_health_status";
/// SPEC-AUTONOMOUS-001: Trigger self-healing protocol for subsystem (NORTH-020)
pub const TRIGGER_HEALING: &str = "trigger_healing";

// ========== META-UTL TOOLS (TASK-MCP-P0-001) ==========

/// TASK-MCP-P0-001: Get current self-correction status
pub const GET_META_LEARNING_STATUS: &str = "get_meta_learning_status";
/// TASK-MCP-P0-001: Manually trigger lambda recalibration
pub const TRIGGER_LAMBDA_RECALIBRATION: &str = "trigger_lambda_recalibration";
/// TASK-MCP-P0-001: Query meta-learning event log
pub const GET_META_LEARNING_LOG: &str = "get_meta_learning_log";

// ========== EPISTEMIC TOOLS (TASK-MCP-001) ==========

/// TASK-MCP-001: Perform epistemic action on GWT workspace
/// Used when embedder category indicates Unknown (high entropy + high coherence)
pub const EPISTEMIC_ACTION: &str = "epistemic_action";

// ========== MERGE TOOLS (TASK-MCP-003) ==========

/// TASK-MCP-003: Merge related concept nodes into a unified node
/// Returns reversal_hash for 30-day undo per SEC-06
pub const MERGE_CONCEPTS: &str = "merge_concepts";

// ========== COHERENCE STATE TOOL (TASK-34) ==========

/// TASK-34: Get high-level coherence state from GWT system
/// Returns coherence_level (High/Medium/Low), is_broadcasting, has_conflict
/// This returns a focused coherence summary for quick status checks.
pub const GET_COHERENCE_STATE: &str = "get_coherence_state";

// ========== SESSION TOOLS (TASK-013) ==========

/// TASK-013: Initialize new MCP session per ARCH-07.
/// Returns session_id, created_at, expires_at, ttl_minutes.
pub const SESSION_START: &str = "session_start";
/// TASK-013: Terminate MCP session per ARCH-07.
/// Returns session_id, duration_seconds, tool_count, status.
pub const SESSION_END: &str = "session_end";
/// TASK-013: Pre-tool use hook per ARCH-07.
/// Records tool invocation before execution.
pub const PRE_TOOL_USE: &str = "pre_tool_use";
/// TASK-013: Post-tool use hook per ARCH-07.
/// Records tool completion and outcome.
pub const POST_TOOL_USE: &str = "post_tool_use";
