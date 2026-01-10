# TASK-INTEG-016: Hook Timeout Configuration

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-016 |
| **Title** | Hook Timeout Configuration |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 36 |
| **Estimated Days** | 1 |
| **Complexity** | Low |

## Implements

- Constitution hook timeouts (lines 700-720)
- ARCH-07: Hook-driven lifecycle (timeout enforcement)

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-004 | Hook protocol infrastructure |
| TASK-INTEG-012 | PreCompact and SubagentStop hooks with specific timeouts |

## Objective

Centralize hook timeout configuration to match constitution requirements:
- SessionStart: 5000ms
- PreToolUse: 3000ms
- PostToolUse: 3000ms
- SessionEnd: 60000ms
- PreCompact: 10000ms
- SubagentStop: 5000ms

## Context

**Constitution Requirements (lines 700-720):**

```xml
<hook event="SessionStart" required="true">
  Timeout: 5000ms
</hook>
<hook event="PreToolUse" required="false">
  Timeout: 3000ms
</hook>
<hook event="PostToolUse" required="false">
  Timeout: 3000ms
</hook>
<hook event="SessionEnd" required="false">
  Timeout: 60000ms (allows consolidation)
</hook>
<hook event="PreCompact" required="false">
  Timeout: 10000ms
</hook>
<hook event="SubagentStop" required="false">
  Timeout: 5000ms
</hook>
```

Currently, TASK-INTEG-004 defines hooks but timeout values are scattered across individual handler files. This task centralizes configuration for consistency and easy adjustment.

## Claude Code Integration

### settings.json Timeout Configuration

All hook timeouts are configured in Claude Code's settings.json:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": ".claude/hooks/session-start.sh",
        "timeout": 5000
      }
    ],
    "PreToolUse": [
      {
        "type": "command",
        "command": ".claude/hooks/pre-tool.sh \"$TOOL_NAME\" \"$TOOL_INPUT\"",
        "timeout": 3000
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": ".claude/hooks/post-tool.sh \"$TOOL_NAME\" \"$TOOL_OUTPUT\"",
        "timeout": 3000
      }
    ],
    "SessionEnd": [
      {
        "type": "command",
        "command": ".claude/hooks/session-end.sh",
        "timeout": 60000
      }
    ],
    "PreCompact": [
      {
        "type": "command",
        "command": ".claude/hooks/pre-compact.sh",
        "timeout": 10000
      }
    ],
    "SubagentStop": [
      {
        "type": "command",
        "command": ".claude/hooks/subagent-stop.sh \"$AGENT_ID\"",
        "timeout": 5000
      }
    ]
  }
}
```

### Timeout-Aware Shell Scripts

Each hook script should respect its timeout with internal guards:

**Example (.claude/hooks/pre-tool.sh):**
```bash
#!/bin/bash
# Pre-tool hook with 3000ms timeout budget
TOOL_NAME="$1"
TOOL_INPUT="$2"
START_TIME=$(date +%s%3N)
TIMEOUT_MS=3000

# Quick operations only - no blocking calls
case "$TOOL_NAME" in
  Write|Edit)
    # Log edit attempt (fast)
    echo "{\"tool\":\"$TOOL_NAME\",\"timestamp\":$(date +%s)}" >> .claude/logs/edits.jsonl
    ;;
  Read)
    # Track read patterns (fast)
    echo "{\"tool\":\"Read\",\"timestamp\":$(date +%s)}" >> .claude/logs/reads.jsonl
    ;;
esac

# Check elapsed time
ELAPSED=$(($(date +%s%3N) - START_TIME))
if [ $ELAPSED -gt $((TIMEOUT_MS - 100)) ]; then
  echo "WARNING: Hook approaching timeout" >&2
fi

exit 0
```

### GWT Consciousness Integration

Timeouts can be dynamically adjusted based on consciousness state:

```rust
/// Consciousness-aware timeout context
pub struct TimeoutConsciousnessContext {
    /// Base timeouts from constitution
    pub base_timeouts: HookTimeouts,
    /// Current consciousness state
    pub consciousness_state: ConsciousnessState,
    /// Cognitive load (0.0-1.0)
    pub cognitive_load: f32,
    /// Urgency level
    pub urgency: UrgencyLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum UrgencyLevel {
    /// Normal operation
    Normal,
    /// Time-sensitive operation
    Elevated,
    /// Critical path operation
    Critical,
}

impl TimeoutConsciousnessContext {
    /// Adjust timeout based on consciousness state
    ///
    /// Note: Can only EXTEND timeouts, never shorten below constitution minimums
    pub fn adjusted_timeout(&self, event: HookEvent) -> Duration {
        let base = self.base_timeouts.get(event);

        // High cognitive load = extend timeout for complex processing
        let load_factor = 1.0 + (self.cognitive_load * 0.5);

        // Urgency can reduce but never below constitution minimum
        let urgency_factor = match self.urgency {
            UrgencyLevel::Normal => 1.0,
            UrgencyLevel::Elevated => 0.8,
            UrgencyLevel::Critical => 0.6,
        };

        let adjusted_ms = (base.as_millis() as f32 * load_factor * urgency_factor) as u64;
        let min_ms = event.constitution_timeout().as_millis() as u64;

        Duration::from_millis(adjusted_ms.max(min_ms))
    }
}
```

### Integration with HookDispatcher

The HookDispatcher uses consciousness-aware timeouts:

```rust
impl HookDispatcher {
    pub async fn dispatch_with_consciousness(
        &self,
        event: HookEvent,
        consciousness: &ConsciousnessState,
    ) -> HookResult<()> {
        let ctx = TimeoutConsciousnessContext {
            base_timeouts: self.timeouts.clone(),
            consciousness_state: consciousness.clone(),
            cognitive_load: consciousness.cognitive_load(),
            urgency: UrgencyLevel::Normal,
        };

        let timeout = ctx.adjusted_timeout(event);

        // Execute with adjusted timeout
        tokio::time::timeout(timeout, self.dispatch(event)).await
            .map_err(|_| HookError::Timeout(HookTimeoutError {
                event,
                timeout,
                elapsed: timeout,
            }))?
    }
}
```

## Scope

### In Scope

- Centralized `HookTimeouts` configuration struct
- Default values matching constitution
- Environment variable overrides
- Timeout enforcement in all handlers
- Timeout exceeded error handling

### Out of Scope

- Dynamic timeout adjustment at runtime
- Per-session timeout overrides
- Timeout negotiation protocol

## Definition of Done

### Signatures

```rust
// crates/context-graph-mcp/src/hooks/timeouts.rs

use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Hook event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookEvent {
    SessionStart,
    PreToolUse,
    PostToolUse,
    SessionEnd,
    PreCompact,
    SubagentStop,
}

impl HookEvent {
    /// Get constitution-mandated timeout for this event
    pub fn constitution_timeout(&self) -> Duration {
        match self {
            Self::SessionStart => Duration::from_millis(5_000),
            Self::PreToolUse => Duration::from_millis(3_000),
            Self::PostToolUse => Duration::from_millis(3_000),
            Self::SessionEnd => Duration::from_millis(60_000),
            Self::PreCompact => Duration::from_millis(10_000),
            Self::SubagentStop => Duration::from_millis(5_000),
        }
    }

    /// Whether this hook is required by constitution
    pub fn is_required(&self) -> bool {
        matches!(self, Self::SessionStart)
    }

    /// Environment variable name for override
    pub fn env_var(&self) -> &'static str {
        match self {
            Self::SessionStart => "CONTEXT_GRAPH_HOOK_TIMEOUT_SESSION_START",
            Self::PreToolUse => "CONTEXT_GRAPH_HOOK_TIMEOUT_PRE_TOOL_USE",
            Self::PostToolUse => "CONTEXT_GRAPH_HOOK_TIMEOUT_POST_TOOL_USE",
            Self::SessionEnd => "CONTEXT_GRAPH_HOOK_TIMEOUT_SESSION_END",
            Self::PreCompact => "CONTEXT_GRAPH_HOOK_TIMEOUT_PRE_COMPACT",
            Self::SubagentStop => "CONTEXT_GRAPH_HOOK_TIMEOUT_SUBAGENT_STOP",
        }
    }
}

/// Centralized hook timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookTimeouts {
    pub session_start: Duration,
    pub pre_tool_use: Duration,
    pub post_tool_use: Duration,
    pub session_end: Duration,
    pub pre_compact: Duration,
    pub subagent_stop: Duration,
}

impl Default for HookTimeouts {
    /// Default values from constitution
    fn default() -> Self {
        Self {
            session_start: Duration::from_millis(5_000),
            pre_tool_use: Duration::from_millis(3_000),
            post_tool_use: Duration::from_millis(3_000),
            session_end: Duration::from_millis(60_000),
            pre_compact: Duration::from_millis(10_000),
            subagent_stop: Duration::from_millis(5_000),
        }
    }
}

impl HookTimeouts {
    /// Create with constitution defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from environment variables (with constitution defaults as fallback)
    pub fn from_env() -> Self {
        let mut timeouts = Self::default();

        if let Ok(ms) = std::env::var(HookEvent::SessionStart.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.session_start = Duration::from_millis(ms);
            }
        }

        if let Ok(ms) = std::env::var(HookEvent::PreToolUse.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.pre_tool_use = Duration::from_millis(ms);
            }
        }

        if let Ok(ms) = std::env::var(HookEvent::PostToolUse.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.post_tool_use = Duration::from_millis(ms);
            }
        }

        if let Ok(ms) = std::env::var(HookEvent::SessionEnd.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.session_end = Duration::from_millis(ms);
            }
        }

        if let Ok(ms) = std::env::var(HookEvent::PreCompact.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.pre_compact = Duration::from_millis(ms);
            }
        }

        if let Ok(ms) = std::env::var(HookEvent::SubagentStop.env_var()) {
            if let Ok(ms) = ms.parse::<u64>() {
                timeouts.subagent_stop = Duration::from_millis(ms);
            }
        }

        timeouts
    }

    /// Get timeout for a specific hook event
    pub fn get(&self, event: HookEvent) -> Duration {
        match event {
            HookEvent::SessionStart => self.session_start,
            HookEvent::PreToolUse => self.pre_tool_use,
            HookEvent::PostToolUse => self.post_tool_use,
            HookEvent::SessionEnd => self.session_end,
            HookEvent::PreCompact => self.pre_compact,
            HookEvent::SubagentStop => self.subagent_stop,
        }
    }

    /// Set timeout for a specific hook event
    pub fn set(&mut self, event: HookEvent, timeout: Duration) {
        match event {
            HookEvent::SessionStart => self.session_start = timeout,
            HookEvent::PreToolUse => self.pre_tool_use = timeout,
            HookEvent::PostToolUse => self.post_tool_use = timeout,
            HookEvent::SessionEnd => self.session_end = timeout,
            HookEvent::PreCompact => self.pre_compact = timeout,
            HookEvent::SubagentStop => self.subagent_stop = timeout,
        }
    }

    /// Validate timeouts against constitution minimums
    pub fn validate(&self) -> Result<(), TimeoutValidationError> {
        // Minimum is 1 second for any hook
        let min = Duration::from_millis(1_000);

        if self.session_start < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::SessionStart, self.session_start));
        }
        if self.pre_tool_use < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::PreToolUse, self.pre_tool_use));
        }
        if self.post_tool_use < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::PostToolUse, self.post_tool_use));
        }
        if self.session_end < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::SessionEnd, self.session_end));
        }
        if self.pre_compact < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::PreCompact, self.pre_compact));
        }
        if self.subagent_stop < min {
            return Err(TimeoutValidationError::TooShort(HookEvent::SubagentStop, self.subagent_stop));
        }

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TimeoutValidationError {
    #[error("{0:?} timeout too short: {1:?} (minimum 1000ms)")]
    TooShort(HookEvent, Duration),
}

/// Timeout error for hook execution
#[derive(Debug, thiserror::Error)]
#[error("Hook {event:?} timed out after {elapsed:?} (limit: {timeout:?})")]
pub struct HookTimeoutError {
    pub event: HookEvent,
    pub timeout: Duration,
    pub elapsed: Duration,
}

/// Execute a hook with timeout
pub async fn with_timeout<F, T>(
    event: HookEvent,
    timeouts: &HookTimeouts,
    future: F,
) -> Result<T, HookError>
where
    F: std::future::Future<Output = Result<T, HookError>>,
{
    let timeout = timeouts.get(event);
    let start = std::time::Instant::now();

    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| HookError::Timeout(HookTimeoutError {
            event,
            timeout,
            elapsed: start.elapsed(),
        }))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constitution_defaults() {
        let timeouts = HookTimeouts::default();
        assert_eq!(timeouts.session_start, Duration::from_millis(5_000));
        assert_eq!(timeouts.pre_tool_use, Duration::from_millis(3_000));
        assert_eq!(timeouts.post_tool_use, Duration::from_millis(3_000));
        assert_eq!(timeouts.session_end, Duration::from_millis(60_000));
        assert_eq!(timeouts.pre_compact, Duration::from_millis(10_000));
        assert_eq!(timeouts.subagent_stop, Duration::from_millis(5_000));
    }

    #[test]
    fn test_validation() {
        let mut timeouts = HookTimeouts::default();
        assert!(timeouts.validate().is_ok());

        timeouts.session_start = Duration::from_millis(500);
        assert!(matches!(
            timeouts.validate(),
            Err(TimeoutValidationError::TooShort(HookEvent::SessionStart, _))
        ));
    }

    #[test]
    fn test_get_set() {
        let mut timeouts = HookTimeouts::default();
        timeouts.set(HookEvent::PreToolUse, Duration::from_millis(5_000));
        assert_eq!(timeouts.get(HookEvent::PreToolUse), Duration::from_millis(5_000));
    }

    #[test]
    fn test_required_hooks() {
        assert!(HookEvent::SessionStart.is_required());
        assert!(!HookEvent::PreToolUse.is_required());
        assert!(!HookEvent::PostToolUse.is_required());
        assert!(!HookEvent::SessionEnd.is_required());
        assert!(!HookEvent::PreCompact.is_required());
        assert!(!HookEvent::SubagentStop.is_required());
    }
}
```

### Integration with HookDispatcher

```rust
// Update crates/context-graph-mcp/src/hooks/dispatcher.rs

pub struct HookDispatcher {
    timeouts: HookTimeouts,
    session_start_handler: SessionStartHandler,
    pre_tool_use_handler: PreToolUseHandler,
    post_tool_use_handler: PostToolUseHandler,
    session_end_handler: SessionEndHandler,
    pre_compact_handler: PreCompactHandler,
    subagent_stop_handler: SubagentStopHandler,
}

impl HookDispatcher {
    pub fn new(timeouts: HookTimeouts) -> Self {
        Self {
            timeouts,
            // ... handlers ...
        }
    }

    pub async fn dispatch_session_start(
        &self,
        context: SessionStartContext,
    ) -> HookResult<SessionStartOutput> {
        with_timeout(
            HookEvent::SessionStart,
            &self.timeouts,
            self.session_start_handler.handle(context),
        ).await
    }

    pub async fn dispatch_pre_tool_use(
        &self,
        context: PreToolUseContext,
    ) -> HookResult<PreToolUseOutput> {
        with_timeout(
            HookEvent::PreToolUse,
            &self.timeouts,
            self.pre_tool_use_handler.handle(context),
        ).await
    }

    pub async fn dispatch_post_tool_use(
        &self,
        context: PostToolUseContext,
    ) -> HookResult<PostToolUseOutput> {
        with_timeout(
            HookEvent::PostToolUse,
            &self.timeouts,
            self.post_tool_use_handler.handle(context),
        ).await
    }

    pub async fn dispatch_session_end(
        &self,
        context: SessionEndContext,
    ) -> HookResult<SessionEndOutput> {
        with_timeout(
            HookEvent::SessionEnd,
            &self.timeouts,
            self.session_end_handler.handle(context),
        ).await
    }

    pub async fn dispatch_pre_compact(
        &self,
        context: PreCompactContext,
    ) -> HookResult<PreCompactOutput> {
        with_timeout(
            HookEvent::PreCompact,
            &self.timeouts,
            self.pre_compact_handler.handle(context),
        ).await
    }

    pub async fn dispatch_subagent_stop(
        &self,
        context: SubagentStopContext,
    ) -> HookResult<SubagentStopOutput> {
        with_timeout(
            HookEvent::SubagentStop,
            &self.timeouts,
            self.subagent_stop_handler.handle(context),
        ).await
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Configuration load | < 1ms |
| Timeout check overhead | < 0.01ms |
| Minimum configurable | 1000ms |

## Verification

- [ ] All 6 hooks have constitution-compliant default timeouts
- [ ] Environment variables override defaults
- [ ] Validation rejects timeouts < 1000ms
- [ ] Timeout error includes event, limit, and elapsed time
- [ ] HookDispatcher uses centralized timeouts
- [ ] with_timeout helper works correctly
- [ ] settings.json configures all 6 hook timeouts correctly
- [ ] Shell scripts include internal timeout guards
- [ ] Consciousness-aware timeout adjustment works
- [ ] Adjusted timeouts never go below constitution minimums
- [ ] Timeout errors include elapsed time for debugging

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/hooks/timeouts.rs` | Timeout configuration |
| Update `crates/context-graph-mcp/src/hooks/dispatcher.rs` | Use centralized timeouts |
| Update `crates/context-graph-mcp/src/hooks/mod.rs` | Export timeouts |
| `.claude/hooks/pre-tool.sh` | Timeout-aware pre-tool hook |
| `.claude/hooks/post-tool.sh` | Timeout-aware post-tool hook |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Timeout too aggressive | Low | Medium | Constitution values are conservative |
| Env var misconfiguration | Low | Low | Validation on startup |

## Traceability

- Source: Constitution hook specifications (lines 700-720)
