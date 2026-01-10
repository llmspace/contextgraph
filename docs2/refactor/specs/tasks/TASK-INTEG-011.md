# TASK-INTEG-011: Security Implementation (SEC-01 through SEC-08)

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-011 |
| **Title** | Security Implementation (SEC-01 through SEC-08) |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 31 |
| **Estimated Days** | 4 |
| **Complexity** | High |

## Implements

- **SEC-01**: Input validation
- **SEC-02**: PII detection
- **SEC-03**: Rate limiting
- **SEC-04**: Session authentication
- **SEC-05**: Authorization
- **SEC-06**: Secrets management
- **SEC-07**: Subagent isolation
- **SEC-08**: Security logging

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handlers to secure |
| TASK-INTEG-004 | Hook system for security events |
| TASK-INTEG-003 | Consciousness integration for threat awareness |
| TASK-LOGIC-002 | 13-embedder system for PII semantic detection |

## Objective

Implement all security requirements from the constitution including input validation, PII detection, rate limiting, authentication, authorization, secrets management, subagent isolation, and security logging.

## Context

**Constitution Security Requirements (SEC-01 through SEC-08, lines 461-516):**

The constitution mandates comprehensive security controls that are currently **NOT COVERED** by any existing INTEG task. This is a critical gap.

## Scope

### In Scope

- **SEC-01**: Input validation and sanitization for all MCP tool inputs
- **SEC-02**: PII detection patterns (SSN, credit cards, emails, phones, addresses)
- **SEC-03**: Rate limiting per tool per session
- **SEC-04**: Session authentication with token expiry
- **SEC-05**: Authorization enforcement per tool
- **SEC-06**: Environment variable secrets management
- **SEC-07**: Subagent memory isolation
- **SEC-08**: Security event logging

### Out of Scope

- Network-level security (firewall, TLS termination)
- Audit compliance reporting
- Multi-tenant isolation

---

## Claude Code Integration

This section specifies how security integrates with Claude Code's hook system and session management.

### Claude Code Hook Security Integration

Security checks run as part of Claude Code's hook system, validating all inputs before tool execution.

#### PreToolUse Security Hook

```rust
// Security runs on every PreToolUse hook invocation
pub struct PreToolUseSecurityCheck {
    input_validator: InputValidator,
    pii_detector: PiiDetector,
    rate_limiter: RateLimiter,
    authorizer: Authorizer,
}

impl PreToolUseSecurityCheck {
    /// Called by Claude Code before any tool execution
    pub async fn validate_tool_use(
        &self,
        session: &ClaudeCodeSession,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> SecurityResult<()> {
        // 1. Rate limit check (per session, per tool type)
        self.rate_limiter.check(session.id, tool_name)?;

        // 2. Authorization check
        self.authorizer.authorize(&session.auth_session, tool_name)?;

        // 3. Input validation
        self.input_validator.validate_json(tool_input)?;

        // 4. PII detection
        let text_content = extract_text_fields(tool_input);
        for text in text_content {
            if self.pii_detector.contains_pii(&text) {
                return Err(SecurityError::PiiDetected {
                    tool: tool_name.to_string(),
                    action: self.pii_detector.action,
                });
            }
        }

        // 5. Consume rate limit token
        self.rate_limiter.consume(session.id, tool_name);

        Ok(())
    }
}
```

#### Claude Code settings.json Configuration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "command": "security_check",
        "config": {
          "rate_limit_enabled": true,
          "pii_detection_enabled": true,
          "pii_action": "mask",
          "max_text_length": 100000,
          "authorization_required": true
        }
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "command": "security_audit_log"
      }
    ]
  },
  "security": {
    "session_token_expiry_hours": 24,
    "rate_limits": {
      "inject_context": { "max_requests": 100, "window_seconds": 60 },
      "store_memory": { "max_requests": 50, "window_seconds": 60 },
      "search_graph": { "max_requests": 200, "window_seconds": 60 },
      "discover_goals": { "max_requests": 10, "window_seconds": 60 },
      "consolidate_memories": { "max_requests": 1, "window_seconds": 60 }
    },
    "pii_patterns": ["ssn", "credit_card", "email", "phone", "address"]
  }
}
```

### Claude Code Session Security

Session hooks initialize and clean up security context, tying tokens to Claude Code session IDs.

```rust
// crates/context-graph-mcp/src/security/claude_code_session.rs

use chrono::{DateTime, Utc};

/// Claude Code session security context
pub struct ClaudeCodeSessionSecurity {
    session_auth: SessionAuth,
    security_contexts: RwLock<HashMap<String, SessionSecurityContext>>,
}

/// Security context initialized per Claude Code session
#[derive(Debug, Clone)]
pub struct SessionSecurityContext {
    pub claude_code_session_id: String,     // From Claude Code SessionStart
    pub auth_token: SessionToken,            // Our internal token
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub tool_usage_counts: HashMap<String, usize>,
    pub security_events: Vec<SecurityEvent>,
    pub permissions: Permissions,
}

impl ClaudeCodeSessionSecurity {
    /// Called on Claude Code SessionStart hook
    pub async fn on_session_start(
        &self,
        claude_code_session_id: &str,
        permissions: Permissions,
    ) -> Result<SessionSecurityContext, SecurityError> {
        // Create internal auth session tied to Claude Code session ID
        let auth_session = self.session_auth.create_session(permissions.clone());

        let context = SessionSecurityContext {
            claude_code_session_id: claude_code_session_id.to_string(),
            auth_token: auth_session.token.clone(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            tool_usage_counts: HashMap::new(),
            security_events: Vec::new(),
            permissions,
        };

        self.security_contexts.write().await
            .insert(claude_code_session_id.to_string(), context.clone());

        tracing::info!(
            target: "security",
            session_id = %claude_code_session_id,
            "Security context initialized for Claude Code session"
        );

        Ok(context)
    }

    /// Called on Claude Code SessionEnd hook
    pub async fn on_session_end(
        &self,
        claude_code_session_id: &str,
    ) -> SessionSecuritySummary {
        let context = self.security_contexts.write().await
            .remove(claude_code_session_id);

        match context {
            Some(ctx) => {
                // Invalidate auth token
                self.session_auth.invalidate(&ctx.auth_token);

                // Generate summary
                let summary = SessionSecuritySummary {
                    session_id: claude_code_session_id.to_string(),
                    duration_seconds: (Utc::now() - ctx.created_at).num_seconds(),
                    total_tool_calls: ctx.tool_usage_counts.values().sum(),
                    security_events_count: ctx.security_events.len(),
                    rate_limit_hits: ctx.security_events.iter()
                        .filter(|e| matches!(e, SecurityEvent::RateLimitExceeded { .. }))
                        .count(),
                    pii_detections: ctx.security_events.iter()
                        .filter(|e| matches!(e, SecurityEvent::PiiDetected { .. }))
                        .count(),
                };

                tracing::info!(
                    target: "security",
                    session_id = %claude_code_session_id,
                    summary = ?summary,
                    "Security context cleaned up for Claude Code session"
                );

                summary
            }
            None => SessionSecuritySummary::empty(claude_code_session_id),
        }
    }

    /// Validate that a Claude Code session is still valid
    pub async fn validate_session(
        &self,
        claude_code_session_id: &str,
    ) -> Result<SessionSecurityContext, SecurityError> {
        let contexts = self.security_contexts.read().await;
        let ctx = contexts.get(claude_code_session_id)
            .ok_or(SecurityError::SessionNotFound)?;

        // Validate internal token
        self.session_auth.validate(&ctx.auth_token)?;

        Ok(ctx.clone())
    }
}

#[derive(Debug, Clone)]
pub struct SessionSecuritySummary {
    pub session_id: String,
    pub duration_seconds: i64,
    pub total_tool_calls: usize,
    pub security_events_count: usize,
    pub rate_limit_hits: usize,
    pub pii_detections: usize,
}
```

## Definition of Done

### SEC-01: Input Validation

```rust
// crates/context-graph-mcp/src/security/input_validation.rs

use regex::Regex;

/// Input validator for MCP tool parameters
pub struct InputValidator {
    max_text_length: usize,
    max_array_length: usize,
    forbidden_patterns: Vec<Regex>,
}

impl InputValidator {
    pub fn new(config: ValidationConfig) -> Self;

    /// Validate and sanitize text input
    pub fn validate_text(&self, input: &str) -> ValidationResult<String>;

    /// Validate numeric input within bounds
    pub fn validate_number<T: Num>(&self, input: T, min: T, max: T) -> ValidationResult<T>;

    /// Validate UUID format
    pub fn validate_uuid(&self, input: &str) -> ValidationResult<Uuid>;

    /// Validate JSON structure
    pub fn validate_json(&self, input: &serde_json::Value) -> ValidationResult<()>;

    /// Sanitize string for storage (escape, normalize)
    pub fn sanitize(&self, input: &str) -> String;
}

#[derive(Debug)]
pub struct ValidationConfig {
    pub max_text_length: usize,      // Default: 100_000
    pub max_array_length: usize,     // Default: 1_000
    pub reject_html: bool,           // Default: true
    pub reject_null_bytes: bool,     // Default: true
}
```

### SEC-02: PII Detection with 13-Embedder Semantic Analysis

PII detection combines regex pattern matching with semantic analysis using the 13-embedder system for context-aware detection.

```rust
// crates/context-graph-mcp/src/security/pii_detection.rs

use crate::embeddings::{Embedder, E1_Semantic, E5_Moral, E8_Contextual, E11_Emotional};

/// PII detection with regex patterns AND semantic embedder analysis
pub struct PiiDetector {
    patterns: Vec<(PiiType, Regex)>,
    action: PiiAction,
    semantic_embedder: E1_Semantic,      // Understanding content meaning
    moral_embedder: E5_Moral,            // Detecting sensitive content
    contextual_embedder: E8_Contextual,  // Understanding PII context
    emotional_embedder: E11_Emotional,   // Personal information patterns
}

#[derive(Debug, Clone, Copy)]
pub enum PiiType {
    Ssn,
    CreditCard,
    Email,
    PhoneNumber,
    Address,
    DateOfBirth,
    DriversLicense,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiAction {
    Reject,     // Reject input entirely
    Mask,       // Replace PII with [REDACTED]
    Log,        // Log detection but allow
}

impl PiiDetector {
    pub fn new(action: PiiAction) -> Self;

    /// Detect PII in text using regex patterns
    pub fn detect_patterns(&self, text: &str) -> Vec<PiiMatch>;

    /// Semantic PII detection using 13-embedder system
    pub async fn detect_semantic(&self, text: &str) -> Vec<SemanticPiiMatch> {
        let mut matches = Vec::new();

        // E1_Semantic: Understand if content discusses personal information
        let semantic_vec = self.semantic_embedder.embed(text).await;
        let personal_info_similarity = cosine_similarity(
            &semantic_vec,
            &PERSONAL_INFO_REFERENCE_EMBEDDING
        );

        // E5_Moral: Detect ethically sensitive content (privacy violations)
        let moral_vec = self.moral_embedder.embed(text).await;
        let privacy_concern = cosine_similarity(
            &moral_vec,
            &PRIVACY_VIOLATION_REFERENCE_EMBEDDING
        );

        // E8_Contextual: Understand if context indicates PII sharing
        let context_vec = self.contextual_embedder.embed(text).await;
        let pii_context_score = cosine_similarity(
            &context_vec,
            &PII_CONTEXT_REFERENCE_EMBEDDING
        );

        // E11_Emotional: Detect personal/intimate information patterns
        let emotional_vec = self.emotional_embedder.embed(text).await;
        let personal_pattern = cosine_similarity(
            &emotional_vec,
            &PERSONAL_EMOTIONAL_REFERENCE_EMBEDDING
        );

        // Combine signals with weighted scoring
        let combined_score = (personal_info_similarity * 0.3)
            + (privacy_concern * 0.3)
            + (pii_context_score * 0.2)
            + (personal_pattern * 0.2);

        if combined_score > PII_SEMANTIC_THRESHOLD {
            matches.push(SemanticPiiMatch {
                confidence: combined_score,
                detected_by: vec!["E1_Semantic", "E5_Moral", "E8_Contextual", "E11_Emotional"],
                explanation: format!(
                    "Semantic PII detection: personal={:.2}, privacy={:.2}, context={:.2}, emotional={:.2}",
                    personal_info_similarity, privacy_concern, pii_context_score, personal_pattern
                ),
            });
        }

        matches
    }

    /// Combined detection: regex patterns + semantic analysis
    pub async fn detect(&self, text: &str) -> PiiDetectionResult {
        let pattern_matches = self.detect_patterns(text);
        let semantic_matches = self.detect_semantic(text).await;

        PiiDetectionResult {
            pattern_matches,
            semantic_matches,
            contains_pii: !pattern_matches.is_empty() || !semantic_matches.is_empty(),
        }
    }

    /// Apply action (mask or reject)
    pub fn process(&self, text: &str) -> PiiResult<String>;

    /// Check if text contains any PII (pattern-based, synchronous)
    pub fn contains_pii(&self, text: &str) -> bool;

    /// Check if text contains any PII (includes semantic, async)
    pub async fn contains_pii_semantic(&self, text: &str) -> bool;
}

#[derive(Debug)]
pub struct SemanticPiiMatch {
    pub confidence: f32,
    pub detected_by: Vec<&'static str>,
    pub explanation: String,
}

#[derive(Debug)]
pub struct PiiDetectionResult {
    pub pattern_matches: Vec<PiiMatch>,
    pub semantic_matches: Vec<SemanticPiiMatch>,
    pub contains_pii: bool,
}

#[derive(Debug)]
pub struct PiiMatch {
    pub pii_type: PiiType,
    pub start: usize,
    pub end: usize,
    pub matched: String,
}
```

### SEC-03: Rate Limiting

```rust
// crates/context-graph-mcp/src/security/rate_limiter.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Per-session, per-tool rate limiter
pub struct RateLimiter {
    limits: HashMap<String, RateLimit>,  // tool -> limit
    buckets: RwLock<HashMap<(SessionId, String), TokenBucket>>,
}

/// Rate limit configuration per tool (from constitution)
pub fn default_rate_limits() -> HashMap<String, RateLimit> {
    let mut limits = HashMap::new();
    limits.insert("inject_context".into(), RateLimit::new(100, Duration::from_secs(60)));
    limits.insert("store_memory".into(), RateLimit::new(50, Duration::from_secs(60)));
    limits.insert("search_graph".into(), RateLimit::new(200, Duration::from_secs(60)));
    limits.insert("discover_goals".into(), RateLimit::new(10, Duration::from_secs(60)));
    limits.insert("consolidate_memories".into(), RateLimit::new(1, Duration::from_secs(60)));
    limits
}

impl RateLimiter {
    pub fn new(limits: HashMap<String, RateLimit>) -> Self;

    /// Check if request is allowed (token bucket algorithm)
    pub fn check(&self, session: SessionId, tool: &str) -> RateLimitResult;

    /// Consume a token (call after allowing request)
    pub fn consume(&self, session: SessionId, tool: &str);

    /// Get remaining tokens for session/tool
    pub fn remaining(&self, session: SessionId, tool: &str) -> usize;

    /// Reset rate limit for session (admin)
    pub fn reset(&self, session: SessionId);
}

#[derive(Debug)]
pub struct RateLimit {
    pub max_requests: usize,
    pub window: Duration,
}
```

### SEC-04: Session Authentication

```rust
// crates/context-graph-mcp/src/security/auth.rs

use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Session authentication manager
pub struct SessionAuth {
    token_expiry: ChronoDuration,  // Default: 24 hours
    sessions: RwLock<HashMap<SessionToken, Session>>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub token: SessionToken,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub permissions: Permissions,
}

impl SessionAuth {
    pub fn new(token_expiry_hours: i64) -> Self;

    /// Create new session with token
    pub fn create_session(&self, permissions: Permissions) -> Session;

    /// Validate token, return session if valid
    pub fn validate(&self, token: &SessionToken) -> AuthResult<Session>;

    /// Invalidate session (logout)
    pub fn invalidate(&self, token: &SessionToken);

    /// Refresh session expiry
    pub fn refresh(&self, token: &SessionToken) -> AuthResult<Session>;

    /// Clean up expired sessions
    pub fn cleanup_expired(&self);
}
```

### SEC-05: Authorization

```rust
// crates/context-graph-mcp/src/security/authz.rs

/// Tool-level authorization
pub struct Authorizer {
    tool_permissions: HashMap<String, Permission>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,           // inject_context, search_graph
    Write,          // store_memory
    Admin,          // consolidate_memories
    GoalDiscovery,  // discover_goals (read-only by default)
}

impl Authorizer {
    pub fn new() -> Self;

    /// Check if session has permission for tool
    pub fn authorize(&self, session: &Session, tool: &str) -> AuthzResult<()>;

    /// Get required permission for tool
    pub fn required_permission(&self, tool: &str) -> Permission;
}
```

### SEC-06: Secrets Management

```rust
// crates/context-graph-mcp/src/security/secrets.rs

/// Environment-based secrets management
pub struct SecretsManager;

impl SecretsManager {
    /// Get database path from env
    pub fn db_path() -> Result<PathBuf, ConfigError> {
        std::env::var("CONTEXT_GRAPH_DB_PATH")
            .map(PathBuf::from)
            .map_err(|_| ConfigError::EnvNotSet("CONTEXT_GRAPH_DB_PATH".into()))
    }

    /// Get model directory from env
    pub fn model_dir() -> Result<PathBuf, ConfigError> {
        std::env::var("CONTEXT_GRAPH_MODEL_DIR")
            .map(PathBuf::from)
            .map_err(|_| ConfigError::EnvNotSet("CONTEXT_GRAPH_MODEL_DIR".into()))
    }

    /// Required env vars
    pub fn required_vars() -> Vec<&'static str> {
        vec![
            "CONTEXT_GRAPH_DB_PATH",
            "CONTEXT_GRAPH_MODEL_DIR",
        ]
    }

    /// Validate all required vars are set
    pub fn validate_environment() -> Result<(), ConfigError>;
}
```

### SEC-07: Subagent Isolation with Claude Code Task Tool Integration

When Claude Code spawns subagents via the Task tool, each gets isolated memory namespace with permission-controlled consolidation.

```rust
// crates/context-graph-mcp/src/security/isolation.rs

/// Subagent memory isolation for Claude Code Task tool spawned agents
pub struct IsolationManager {
    session_scopes: RwLock<HashMap<SessionId, MemoryScope>>,
    subagent_registry: RwLock<HashMap<SubagentId, SubagentSecurityContext>>,
}

#[derive(Debug, Clone)]
pub struct MemoryScope {
    pub session_id: SessionId,
    pub allowed_namespaces: Vec<String>,
    pub cross_session_access: bool,
}

/// Security context for Task tool spawned subagents
#[derive(Debug, Clone)]
pub struct SubagentSecurityContext {
    pub subagent_id: SubagentId,
    pub parent_session_id: SessionId,
    pub memory_namespace: String,           // Isolated namespace: "subagent_{id}"
    pub spawned_at: DateTime<Utc>,
    pub permissions: SubagentPermissions,
    pub cross_agent_grants: Vec<SubagentId>, // Explicit grants for cross-agent communication
}

#[derive(Debug, Clone)]
pub struct SubagentPermissions {
    pub can_read_parent_memory: bool,       // Default: false
    pub can_write_parent_memory: bool,      // Default: false
    pub can_communicate_peers: bool,        // Default: false
    pub allowed_tools: Vec<String>,         // Subset of parent's tools
    pub max_memory_bytes: usize,            // Memory quota
}

impl IsolationManager {
    /// Called when Claude Code Task tool spawns a subagent
    pub async fn on_subagent_spawn(
        &self,
        parent_session: SessionId,
        subagent_id: SubagentId,
        permissions: SubagentPermissions,
    ) -> SubagentSecurityContext {
        let namespace = format!("subagent_{}", subagent_id);

        let context = SubagentSecurityContext {
            subagent_id: subagent_id.clone(),
            parent_session_id: parent_session,
            memory_namespace: namespace.clone(),
            spawned_at: Utc::now(),
            permissions,
            cross_agent_grants: Vec::new(),
        };

        // Create isolated memory namespace
        self.create_isolated_namespace(&namespace).await;

        // Register subagent
        self.subagent_registry.write().await
            .insert(subagent_id, context.clone());

        tracing::info!(
            target: "security",
            subagent_id = %subagent_id,
            namespace = %namespace,
            "Subagent spawned with isolated memory namespace"
        );

        context
    }

    /// Called when Claude Code SubagentStop occurs - consolidates with permission check
    pub async fn on_subagent_stop(
        &self,
        subagent_id: SubagentId,
        consolidation_request: Option<ConsolidationRequest>,
    ) -> Result<ConsolidationResult, SecurityError> {
        let context = self.subagent_registry.write().await
            .remove(&subagent_id)
            .ok_or(SecurityError::SubagentNotFound)?;

        let result = match consolidation_request {
            Some(request) => {
                // Check if subagent has permission to write to parent
                if !context.permissions.can_write_parent_memory {
                    return Err(SecurityError::ConsolidationDenied {
                        reason: "Subagent lacks can_write_parent_memory permission".into(),
                    });
                }

                // Consolidate selected memories to parent namespace
                self.consolidate_memories(
                    &context.memory_namespace,
                    &format!("session_{}", context.parent_session_id),
                    &request.memory_ids,
                ).await?
            }
            None => ConsolidationResult::NoConsolidation,
        };

        // Clean up isolated namespace
        self.cleanup_namespace(&context.memory_namespace).await;

        tracing::info!(
            target: "security",
            subagent_id = %subagent_id,
            result = ?result,
            "Subagent stopped, memory consolidated"
        );

        Ok(result)
    }

    /// Grant explicit cross-agent communication permission
    pub async fn grant_cross_agent_access(
        &self,
        from_subagent: SubagentId,
        to_subagent: SubagentId,
    ) -> Result<(), SecurityError> {
        let mut registry = self.subagent_registry.write().await;

        let from_ctx = registry.get_mut(&from_subagent)
            .ok_or(SecurityError::SubagentNotFound)?;

        if !from_ctx.permissions.can_communicate_peers {
            return Err(SecurityError::CrossAgentDenied {
                reason: "Subagent lacks can_communicate_peers permission".into(),
            });
        }

        from_ctx.cross_agent_grants.push(to_subagent.clone());

        tracing::info!(
            target: "security",
            from = %from_subagent,
            to = %to_subagent,
            "Cross-agent communication granted"
        );

        Ok(())
    }

    /// Check if cross-agent communication is allowed
    pub async fn check_cross_agent_access(
        &self,
        from_subagent: SubagentId,
        to_subagent: SubagentId,
    ) -> bool {
        let registry = self.subagent_registry.read().await;

        registry.get(&from_subagent)
            .map(|ctx| ctx.cross_agent_grants.contains(&to_subagent))
            .unwrap_or(false)
    }

    /// Create isolated scope for subagent
    pub fn create_scope(&self, parent_session: SessionId) -> MemoryScope;

    /// Check if access to memory is allowed
    pub fn check_access(&self, scope: &MemoryScope, memory_id: Uuid) -> bool;

    /// Grant cross-session access (explicit consolidation)
    pub fn grant_cross_session(&self, scope: &mut MemoryScope);
}

#[derive(Debug)]
pub struct ConsolidationRequest {
    pub memory_ids: Vec<Uuid>,
    pub consolidation_strategy: ConsolidationStrategy,
}

#[derive(Debug)]
pub enum ConsolidationResult {
    Consolidated { memories_transferred: usize },
    NoConsolidation,
}
```

### SEC-08: Security Logging

```rust
// crates/context-graph-mcp/src/security/audit_log.rs

use serde::Serialize;
use tracing::{info, warn, error};

/// Security event types
#[derive(Debug, Clone, Serialize)]
pub enum SecurityEvent {
    AuthenticationFailure { session_token: String, reason: String },
    RateLimitExceeded { session_id: String, tool: String, limit: usize },
    PiiDetected { session_id: String, pii_type: String, action_taken: String },
    InvalidInputRejected { session_id: String, tool: String, reason: String },
    AuthorizationDenied { session_id: String, tool: String, required: String },
    SessionExpired { session_id: String },
}

/// Structured security audit logger
pub struct SecurityAuditLog;

impl SecurityAuditLog {
    /// Log security event with structured JSON
    pub fn log(event: SecurityEvent) {
        let json = serde_json::to_string(&event).unwrap();
        match &event {
            SecurityEvent::AuthenticationFailure { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::RateLimitExceeded { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::PiiDetected { .. } => info!(target: "security", "{}", json),
            SecurityEvent::InvalidInputRejected { .. } => info!(target: "security", "{}", json),
            SecurityEvent::AuthorizationDenied { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::SessionExpired { .. } => info!(target: "security", "{}", json),
        }
    }
}
```

---

## GWT Consciousness Security Integration

Security awareness scales with consciousness level, enabling adaptive threat response.

### SecurityConsciousnessContext

```rust
// crates/context-graph-mcp/src/security/consciousness_security.rs

use crate::consciousness::{ConsciousnessState, GwtMetrics};

/// Security context integrated with GWT consciousness model
pub struct SecurityConsciousnessContext {
    pub integration_score: f32,      // I(t) - security posture integration
    pub reflection_depth: f32,       // R(t) - threat analysis depth
    pub differentiation_index: f32,  // D(t) - anomaly detection capability
    pub consciousness_level: f32,    // C(t) = I(t) * R(t) * D(t)
    pub threat_awareness: ThreatAwareness,
}

/// Threat awareness levels based on consciousness C(t)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreatAwareness {
    Normal,         // C < 0.3, standard security checks only
    Elevated,       // 0.3 <= C < 0.6, enhanced monitoring active
    HighAlert,      // 0.6 <= C < 0.8, full threat analysis engaged
    CriticalAlert,  // C >= 0.8, maximum security posture
}

impl ThreatAwareness {
    pub fn from_consciousness_level(c: f32) -> Self {
        match c {
            c if c >= 0.8 => ThreatAwareness::CriticalAlert,
            c if c >= 0.6 => ThreatAwareness::HighAlert,
            c if c >= 0.3 => ThreatAwareness::Elevated,
            _ => ThreatAwareness::Normal,
        }
    }
}

impl SecurityConsciousnessContext {
    /// Create security context from current consciousness state
    pub fn from_consciousness(state: &ConsciousnessState) -> Self {
        let gwt = &state.gwt_metrics;

        // Map GWT metrics to security dimensions
        let integration_score = gwt.global_availability;      // How integrated is security posture
        let reflection_depth = gwt.attention_capacity;        // Depth of threat analysis
        let differentiation_index = gwt.workspace_coherence;  // Anomaly detection capability

        let consciousness_level = integration_score * reflection_depth * differentiation_index;
        let threat_awareness = ThreatAwareness::from_consciousness_level(consciousness_level);

        Self {
            integration_score,
            reflection_depth,
            differentiation_index,
            consciousness_level,
            threat_awareness,
        }
    }

    /// Get security check intensity based on consciousness level
    pub fn security_check_config(&self) -> SecurityCheckConfig {
        match self.threat_awareness {
            ThreatAwareness::Normal => SecurityCheckConfig {
                pii_detection: PiiDetectionMode::PatternOnly,
                input_validation_depth: ValidationDepth::Standard,
                rate_limit_strictness: 1.0,
                log_verbosity: LogVerbosity::Minimal,
            },
            ThreatAwareness::Elevated => SecurityCheckConfig {
                pii_detection: PiiDetectionMode::PatternAndSemantic,
                input_validation_depth: ValidationDepth::Enhanced,
                rate_limit_strictness: 0.8,  // 20% stricter
                log_verbosity: LogVerbosity::Standard,
            },
            ThreatAwareness::HighAlert => SecurityCheckConfig {
                pii_detection: PiiDetectionMode::Full,
                input_validation_depth: ValidationDepth::Deep,
                rate_limit_strictness: 0.5,  // 50% stricter
                log_verbosity: LogVerbosity::Detailed,
            },
            ThreatAwareness::CriticalAlert => SecurityCheckConfig {
                pii_detection: PiiDetectionMode::Full,
                input_validation_depth: ValidationDepth::Paranoid,
                rate_limit_strictness: 0.3,  // 70% stricter
                log_verbosity: LogVerbosity::Full,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecurityCheckConfig {
    pub pii_detection: PiiDetectionMode,
    pub input_validation_depth: ValidationDepth,
    pub rate_limit_strictness: f32,  // Multiplier for rate limits (lower = stricter)
    pub log_verbosity: LogVerbosity,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiDetectionMode {
    PatternOnly,        // Regex patterns only (fast)
    PatternAndSemantic, // Patterns + semantic embedder analysis
    Full,               // All detection methods including behavioral analysis
}

#[derive(Debug, Clone, Copy)]
pub enum ValidationDepth {
    Standard,   // Basic validation
    Enhanced,   // Additional sanitization
    Deep,       // Full content analysis
    Paranoid,   // Maximum scrutiny, may reject ambiguous inputs
}

#[derive(Debug, Clone, Copy)]
pub enum LogVerbosity {
    Minimal,    // Security events only
    Standard,   // Events + context
    Detailed,   // Full request/response logging
    Full,       // Everything including timing and memory state
}
```

### Consciousness-Aware Security Dispatcher

```rust
// crates/context-graph-mcp/src/security/conscious_dispatcher.rs

/// Security dispatcher that adapts based on consciousness level
pub struct ConsciousSecurityDispatcher {
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
    security_check: PreToolUseSecurityCheck,
}

impl ConsciousSecurityDispatcher {
    /// Dispatch security checks with consciousness-aware configuration
    pub async fn check_with_consciousness(
        &self,
        session: &ClaudeCodeSession,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> SecurityResult<()> {
        // Get current consciousness level
        let consciousness = self.consciousness_state.read().await;
        let security_context = SecurityConsciousnessContext::from_consciousness(&consciousness);
        let config = security_context.security_check_config();

        // Adjust rate limits based on consciousness
        let adjusted_limit = self.security_check.rate_limiter
            .get_limit(tool_name)
            .map(|l| (l.max_requests as f32 * config.rate_limit_strictness) as usize);

        // Run PII detection with appropriate mode
        match config.pii_detection {
            PiiDetectionMode::PatternOnly => {
                self.security_check.pii_detector.detect_patterns(tool_input)?;
            }
            PiiDetectionMode::PatternAndSemantic | PiiDetectionMode::Full => {
                self.security_check.pii_detector.detect(tool_input).await?;
            }
        }

        // Log with appropriate verbosity
        if config.log_verbosity >= LogVerbosity::Detailed {
            tracing::info!(
                target: "security",
                consciousness_level = %security_context.consciousness_level,
                threat_awareness = ?security_context.threat_awareness,
                tool = %tool_name,
                "Consciousness-aware security check completed"
            );
        }

        Ok(())
    }
}
```

---

## Security Event Memory Integration

Security events are stored in teleological memory for pattern learning and incident analysis.

### Security Memory Storage

```rust
// crates/context-graph-mcp/src/security/memory_integration.rs

use crate::memory::{TeleologicalMemory, MemoryNamespace};

/// Security event memory integration
pub struct SecurityMemoryIntegration {
    memory: Arc<TeleologicalMemory>,
}

/// Security event for memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMemoryEvent {
    pub event_id: Uuid,
    pub session_id: String,
    pub event_type: SecurityEventType,
    pub timestamp: DateTime<Utc>,
    pub severity: SecuritySeverity,
    pub details: serde_json::Value,
    pub embedding: Option<Vec<f32>>,  // For pattern matching
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Info,
    Warning,
    Critical,
    Alert,
}

impl SecurityMemoryIntegration {
    pub fn new(memory: Arc<TeleologicalMemory>) -> Self {
        Self { memory }
    }

    /// Store security event in teleological memory
    pub async fn store_event(&self, event: SecurityEvent) -> Result<Uuid, MemoryError> {
        let memory_event = SecurityMemoryEvent {
            event_id: Uuid::new_v4(),
            session_id: event.session_id().to_string(),
            event_type: SecurityEventType::from(&event),
            timestamp: Utc::now(),
            severity: event.severity(),
            details: serde_json::to_value(&event)?,
            embedding: None,  // Computed lazily for pattern matching
        };

        // Store in security namespace with memory type annotation
        self.memory.store(
            MemoryNamespace::Security,
            memory_event.event_id,
            MemoryType::SecurityEvent,
            memory_event,
        ).await?;

        Ok(memory_event.event_id)
    }

    /// Search for similar security patterns
    pub async fn search_similar_patterns(
        &self,
        event: &SecurityEvent,
        k: usize,
    ) -> Result<Vec<SecurityMemoryEvent>, MemoryError> {
        // Generate embedding for pattern search
        let event_embedding = self.compute_event_embedding(event).await?;

        // Search in security namespace
        self.memory.search_similar(
            MemoryNamespace::Security,
            &event_embedding,
            k,
        ).await
    }

    /// Analyze security patterns over time window
    pub async fn analyze_patterns(
        &self,
        window: Duration,
    ) -> SecurityPatternAnalysis {
        let events = self.memory.query_recent(
            MemoryNamespace::Security,
            window,
        ).await.unwrap_or_default();

        SecurityPatternAnalysis {
            total_events: events.len(),
            by_type: group_by_type(&events),
            by_severity: group_by_severity(&events),
            anomalies: detect_anomalies(&events),
            recommendations: generate_recommendations(&events),
        }
    }

    /// Learn from security incident for future detection
    pub async fn learn_from_incident(
        &self,
        incident: SecurityIncident,
    ) -> Result<(), MemoryError> {
        // Store incident with high priority for pattern learning
        self.memory.store_with_priority(
            MemoryNamespace::Security,
            incident.id,
            MemoryType::SecurityIncident,
            incident,
            MemoryPriority::High,
        ).await
    }
}

#[derive(Debug)]
pub struct SecurityPatternAnalysis {
    pub total_events: usize,
    pub by_type: HashMap<SecurityEventType, usize>,
    pub by_severity: HashMap<SecuritySeverity, usize>,
    pub anomalies: Vec<SecurityAnomaly>,
    pub recommendations: Vec<SecurityRecommendation>,
}
```

---

## Hook-Level Security Enforcement

Shell script hooks provide an additional security layer at the Claude Code hook level.

### Security Check Hook Script

```bash
#!/bin/bash
# .claude/hooks/security_check.sh
# PreToolUse matcher: all tools (regex: .*)
# Runs input validation before every tool execution

set -e

# Read hook input from stdin
HOOK_INPUT=$(cat)

# Parse JSON input
TOOL_NAME=$(echo "$HOOK_INPUT" | jq -r '.tool_name')
TOOL_INPUT=$(echo "$HOOK_INPUT" | jq -r '.tool_input')
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id')

# 1. Check for blocked patterns (injection attempts)
BLOCKED_PATTERNS=(
    "rm -rf /"
    "DROP TABLE"
    "DELETE FROM"
    "; --"
    "eval("
    "__import__"
)

for pattern in "${BLOCKED_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qi "$pattern"; then
        echo "{\"status\": \"blocked\", \"reason\": \"Blocked pattern detected: $pattern\"}" >&2
        exit 1
    fi
done

# 2. Check input size limits
INPUT_SIZE=$(echo "$TOOL_INPUT" | wc -c)
MAX_SIZE=100000

if [ "$INPUT_SIZE" -gt "$MAX_SIZE" ]; then
    echo "{\"status\": \"blocked\", \"reason\": \"Input exceeds maximum size: $INPUT_SIZE > $MAX_SIZE\"}" >&2
    exit 1
fi

# 3. PII pattern detection (basic)
PII_PATTERNS=(
    "[0-9]{3}-[0-9]{2}-[0-9]{4}"  # SSN
    "[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}"  # Credit card
)

for pattern in "${PII_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qE "$pattern"; then
        echo "{\"status\": \"warning\", \"reason\": \"Potential PII detected\", \"pattern\": \"$pattern\"}" >&2
        # Log but allow (Rust layer handles actual rejection)
    fi
done

# 4. Rate limit check via shared state file
RATE_FILE="/tmp/claude_security_rate_${SESSION_ID}_${TOOL_NAME}"
CURRENT_TIME=$(date +%s)
WINDOW=60  # 1 minute window

if [ -f "$RATE_FILE" ]; then
    LAST_TIME=$(head -1 "$RATE_FILE")
    COUNT=$(tail -1 "$RATE_FILE")

    if [ $((CURRENT_TIME - LAST_TIME)) -lt $WINDOW ]; then
        COUNT=$((COUNT + 1))
        echo "$LAST_TIME" > "$RATE_FILE"
        echo "$COUNT" >> "$RATE_FILE"

        # Check against limits (conservative shell-level check)
        if [ "$COUNT" -gt 200 ]; then
            echo "{\"status\": \"blocked\", \"reason\": \"Rate limit exceeded in shell check\"}" >&2
            exit 1
        fi
    else
        # Reset window
        echo "$CURRENT_TIME" > "$RATE_FILE"
        echo "1" >> "$RATE_FILE"
    fi
else
    echo "$CURRENT_TIME" > "$RATE_FILE"
    echo "1" >> "$RATE_FILE"
fi

# 5. Log security check completion
echo "{\"status\": \"passed\", \"tool\": \"$TOOL_NAME\", \"session\": \"$SESSION_ID\", \"timestamp\": \"$(date -Iseconds)\"}"
exit 0
```

### Claude Code settings.json Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "type": "command",
        "command": ".claude/hooks/security_check.sh",
        "timeout": 5000,
        "failOpen": false
      }
    ],
    "SessionStart": [
      {
        "type": "command",
        "command": ".claude/hooks/session_security_init.sh"
      }
    ],
    "SessionEnd": [
      {
        "type": "command",
        "command": ".claude/hooks/session_security_cleanup.sh"
      }
    ],
    "SubagentSpawn": [
      {
        "type": "command",
        "command": ".claude/hooks/subagent_isolation.sh"
      }
    ],
    "SubagentStop": [
      {
        "type": "command",
        "command": ".claude/hooks/subagent_consolidate.sh"
      }
    ]
  }
}
```

### Session Security Initialization Hook

```bash
#!/bin/bash
# .claude/hooks/session_security_init.sh
# Called on SessionStart

set -e

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id')

# Initialize security state directory
SECURITY_DIR="/tmp/claude_security_${SESSION_ID}"
mkdir -p "$SECURITY_DIR"

# Initialize rate limit tracking
touch "$SECURITY_DIR/rate_limits"

# Initialize security event log
echo "[]" > "$SECURITY_DIR/events.json"

# Store session start time
echo "$(date +%s)" > "$SECURITY_DIR/started_at"

echo "{\"status\": \"initialized\", \"session\": \"$SESSION_ID\", \"security_dir\": \"$SECURITY_DIR\"}"
```

### Session Security Cleanup Hook

```bash
#!/bin/bash
# .claude/hooks/session_security_cleanup.sh
# Called on SessionEnd

set -e

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id')

SECURITY_DIR="/tmp/claude_security_${SESSION_ID}"

if [ -d "$SECURITY_DIR" ]; then
    # Generate security summary
    START_TIME=$(cat "$SECURITY_DIR/started_at" 2>/dev/null || echo "0")
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    EVENTS=$(cat "$SECURITY_DIR/events.json" 2>/dev/null || echo "[]")
    EVENT_COUNT=$(echo "$EVENTS" | jq 'length')

    # Output summary
    echo "{\"status\": \"cleaned\", \"session\": \"$SESSION_ID\", \"duration_seconds\": $DURATION, \"security_events\": $EVENT_COUNT}"

    # Clean up
    rm -rf "$SECURITY_DIR"
else
    echo "{\"status\": \"no_cleanup_needed\", \"session\": \"$SESSION_ID\"}"
fi
```

---

### Constraints

| Constraint | Target |
|------------|--------|
| Input validation latency | < 1ms |
| PII detection latency (pattern) | < 5ms |
| PII detection latency (semantic) | < 50ms |
| Rate limit check | < 0.1ms |
| Auth validation | < 1ms |
| Hook script execution | < 100ms |
| Consciousness context computation | < 1ms |

## Verification

### Core Security (SEC-01 through SEC-08)

- [ ] All MCP tool inputs validated before processing
- [ ] PII patterns detect SSN, credit cards, emails, phones
- [ ] Rate limits enforced per constitution (100/50/200/1 per min)
- [ ] Session tokens expire after 24 hours
- [ ] Unauthorized tool access returns 403
- [ ] Security events logged in JSON format
- [ ] Subagent memory isolated from other sessions

### Claude Code Integration

- [ ] PreToolUse hook validates all tool inputs
- [ ] SessionStart hook initializes security context
- [ ] SessionEnd hook cleans up and logs summary
- [ ] Session tokens tied to Claude Code session IDs
- [ ] Rate limiting enforced per session per tool type

### 13-Embedder Semantic PII Detection

- [ ] E1_Semantic detects personal information content
- [ ] E5_Moral detects privacy-sensitive content
- [ ] E8_Contextual understands PII sharing context
- [ ] E11_Emotional detects personal information patterns
- [ ] Combined semantic score threshold configurable

### GWT Consciousness Integration

- [ ] SecurityConsciousnessContext computed from GWT metrics
- [ ] ThreatAwareness levels (Normal/Elevated/HighAlert/CriticalAlert) active
- [ ] Security check intensity scales with consciousness level
- [ ] Rate limits tighten at higher threat awareness

### Security Memory Integration

- [ ] Security events stored in "security" namespace
- [ ] Pattern learning from security incidents enabled
- [ ] Similar security patterns searchable via embeddings
- [ ] Incident analysis generates recommendations

### Subagent Isolation

- [ ] Task tool spawned agents get isolated memory namespace
- [ ] SubagentStop consolidates with permission check
- [ ] Cross-agent communication requires explicit grants
- [ ] Subagent permissions enforced (read/write parent, peer communication)

### Hook-Level Security

- [ ] .claude/hooks/security_check.sh blocks dangerous patterns
- [ ] .claude/hooks/session_security_init.sh creates security state
- [ ] .claude/hooks/session_security_cleanup.sh generates summary
- [ ] Hook script execution completes within 100ms

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/security/mod.rs` | Security module root |
| `crates/context-graph-mcp/src/security/input_validation.rs` | SEC-01 |
| `crates/context-graph-mcp/src/security/pii_detection.rs` | SEC-02 with 13-embedder integration |
| `crates/context-graph-mcp/src/security/rate_limiter.rs` | SEC-03 |
| `crates/context-graph-mcp/src/security/auth.rs` | SEC-04 |
| `crates/context-graph-mcp/src/security/authz.rs` | SEC-05 |
| `crates/context-graph-mcp/src/security/secrets.rs` | SEC-06 |
| `crates/context-graph-mcp/src/security/isolation.rs` | SEC-07 with subagent support |
| `crates/context-graph-mcp/src/security/audit_log.rs` | SEC-08 |
| `crates/context-graph-mcp/src/security/claude_code_session.rs` | Claude Code session security |
| `crates/context-graph-mcp/src/security/consciousness_security.rs` | GWT consciousness integration |
| `crates/context-graph-mcp/src/security/conscious_dispatcher.rs` | Consciousness-aware dispatcher |
| `crates/context-graph-mcp/src/security/memory_integration.rs` | Security memory storage |
| `.claude/hooks/security_check.sh` | PreToolUse security hook |
| `.claude/hooks/session_security_init.sh` | SessionStart hook |
| `.claude/hooks/session_security_cleanup.sh` | SessionEnd hook |
| `.claude/hooks/subagent_isolation.sh` | SubagentSpawn hook |
| `.claude/hooks/subagent_consolidate.sh` | SubagentStop hook |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PII bypass | Medium | Critical | Multiple pattern sources + semantic embedder analysis |
| Rate limit bypass | Low | Medium | Token bucket algorithm + shell-level backup check |
| Token leak | Low | Critical | Short expiry, secure generation, session binding |
| Semantic PII false positives | Medium | Low | Configurable threshold, pattern confirmation |
| Consciousness level manipulation | Low | Medium | GWT metrics computed from verified state |
| Subagent isolation escape | Low | Critical | Namespace isolation + permission enforcement |
| Cross-agent unauthorized access | Low | High | Explicit grants required, audit logging |
| Hook script timeout | Medium | Low | 100ms timeout, fail-closed policy |
| Security memory overflow | Low | Medium | Memory quotas per namespace |

## Traceability

- Source: Constitution security_requirements (lines 461-516)
- Claude Code hooks: PreToolUse, SessionStart, SessionEnd, SubagentSpawn, SubagentStop
- 13-Embedder: E1_Semantic, E5_Moral, E8_Contextual, E11_Emotional
- GWT Consciousness: C(t) = I(t) * R(t) * D(t) formula
- Memory: TeleologicalMemory with security namespace
