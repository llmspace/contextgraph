# E3 (V_periodicity) Optimization Plan

**Version**: 1.0.0
**Date**: 2026-01-25
**Status**: Implementation Ready

## Executive Summary

E3 (V_periodicity) is a 512D Fourier-based temporal embedder that captures cyclical time patterns (hour-of-day, day-of-week, month-of-year). The model infrastructure is **fully implemented**, but it lacks **user-facing exposure** via MCP tools. This plan addresses how to optimally integrate E3 into the context-graph system to enhance E1's semantic search by finding time-based patterns that E1 cannot detect.

### Current State

| Component | Status | Notes |
|-----------|--------|-------|
| E3 Model | COMPLETE | Fourier-based, 512D, 5 periods (hour/day/week/month/year) |
| Embedding Generation | COMPLETE | Runs in parallel with E1-E13 |
| HNSW Indexing | COMPLETE | Dense index, RocksDB storage |
| Post-retrieval Functions | COMPLETE | `compute_e3_periodic_score()`, `compute_periodic_match_fallback()` |
| PeriodicOptions | COMPLETE | target_hour, target_day_of_week, auto_detect |
| MCP Tool Exposure | **MISSING** | No `search_periodic` tool |
| Query Type Detection | **MISSING** | No periodic pattern auto-detection |

### What E3 Finds That E1 Misses

E1 (semantic) cannot encode temporal patterns. E3 complements E1 by finding:

| E1 Query | E1 Misses | E3 Finds |
|----------|-----------|----------|
| "standup meeting notes" | Meetings at 9am vs 3pm | Memories from 9am (standup time) |
| "deployment issues" | Friday vs Monday patterns | Friday deployment patterns |
| "morning productivity" | Actual morning times | Memories created 6am-10am |
| "weekend project" | Semantic "weekend" | Memories from Saturday/Sunday |

---

## Architecture

### E3 Signal Definition

Per Constitution v6.5, E3 captures **periodic time-of-day patterns**:

```yaml
E3:
  name: V_periodicity
  dimension: 512
  finds: "Time-of-day patterns, day-of-week cycles"
  misses_by_E1: "E1 treats timestamps as metadata, not signal"
  role: "POST-RETRIEVAL ONLY (per ARCH-25, AP-73)"
  topic_weight: 0.0  # NEVER contributes to topics
```

### Fourier Encoding

E3 uses Fourier basis functions to encode cyclical patterns:

```
For each period P in [hour, day, week, month, year]:
  position = timestamp_secs mod P
  phase = 2 * pi * position / P
  For harmonic n in 1..51:
    features.push(sin(n * phase))
    features.push(cos(n * phase))
```

This produces 5 periods * 51 harmonics * 2 (sin/cos) = 510 features + 2 padding = 512D.

### Why Fourier Works

Fourier encoding enables:
- **Same-time similarity**: 10:30 AM on different days has near-identical hour features (cosine > 0.99)
- **Opposite-time dissimilarity**: 6 AM vs 6 PM are orthogonal (cosine < 0.5)
- **Cyclic wrap-around**: 11 PM and 1 AM are close (2 hours apart), not 22 hours

---

## Implementation Plan

### Phase 1: search_periodic MCP Tool

**Objective**: Expose E3's periodic pattern matching via a dedicated MCP tool.

#### 1.1 Tool Definition

File: `crates/context-graph-mcp/src/tools/definitions/temporal.rs`

```rust
// Add to definitions() function
ToolDefinition::new(
    "search_periodic",
    "Search for memories matching periodic time patterns. \
     Finds memories from similar times of day or days of week. \
     Use for queries like 'morning meetings', 'Friday deployments', \
     'what do I usually work on at 3pm?'. \
     Per ARCH-25: Periodic boost is applied POST-retrieval.",
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query text"
            },
            "topK": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "Maximum number of results"
            },
            "targetHour": {
                "type": "integer",
                "minimum": 0,
                "maximum": 23,
                "description": "Target hour (0-23). If omitted and autoDetect=true, uses current hour."
            },
            "targetDayOfWeek": {
                "type": "integer",
                "minimum": 0,
                "maximum": 6,
                "description": "Target day (0=Sunday, 6=Saturday). If omitted and autoDetect=true, uses current day."
            },
            "autoDetect": {
                "type": "boolean",
                "default": false,
                "description": "Auto-detect target from current time. When true, targetHour/targetDayOfWeek are computed from now."
            },
            "periodicWeight": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "default": 0.3,
                "description": "Weight for periodic boost [0.1, 1.0]. Higher = more periodic preference."
            },
            "includeContent": {
                "type": "boolean",
                "default": true,
                "description": "Include content text in results"
            },
            "minSimilarity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.1,
                "description": "Minimum semantic similarity threshold"
            }
        },
        "required": ["query"]
    }),
),
```

#### 1.2 Tool Name Constant

File: `crates/context-graph-mcp/src/tools/names.rs`

```rust
// Add after SEARCH_RECENT
pub const SEARCH_PERIODIC: &str = "search_periodic";
```

#### 1.3 DTOs

File: `crates/context-graph-mcp/src/handlers/tools/temporal_dtos.rs`

Add new DTOs for search_periodic:

```rust
/// Parameters for search_periodic tool.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPeriodicParams {
    /// The search query text.
    pub query: String,

    /// Maximum number of results (default: 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Target hour of day (0-23).
    /// If omitted and auto_detect=true, uses current hour.
    pub target_hour: Option<u8>,

    /// Target day of week (0=Sunday, 6=Saturday).
    /// If omitted and auto_detect=true, uses current day.
    pub target_day_of_week: Option<u8>,

    /// Auto-detect target from current time.
    #[serde(default)]
    pub auto_detect: bool,

    /// Periodic boost weight [0.1, 1.0] (default: 0.3).
    #[serde(default = "default_periodic_weight")]
    pub periodic_weight: f32,

    /// Include content in results (default: true).
    #[serde(default = "default_true")]
    pub include_content: bool,

    /// Minimum semantic similarity threshold (default: 0.1).
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
}

fn default_periodic_weight() -> f32 {
    0.3
}

/// Result entry for search_periodic.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PeriodicSearchResultEntry {
    /// Memory ID.
    pub id: String,

    /// Original semantic similarity score.
    pub semantic_score: f32,

    /// Periodic pattern match score [0.0, 1.0].
    pub periodic_score: f32,

    /// Final boosted score.
    pub final_score: f32,

    /// Memory's hour of creation (0-23).
    pub memory_hour: u8,

    /// Memory's day of week (0=Sunday, 6=Saturday).
    pub memory_day_of_week: u8,

    /// Content text (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Created at timestamp (ISO 8601).
    pub created_at: String,
}

/// Response for search_periodic.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPeriodicResponse {
    /// Query that was executed.
    pub query: String,

    /// Results sorted by final score.
    pub results: Vec<PeriodicSearchResultEntry>,

    /// Number of results.
    pub count: usize,

    /// Periodic configuration used.
    pub periodic_config: PeriodicConfigSummary,
}

/// Summary of periodic configuration used.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PeriodicConfigSummary {
    /// Target hour used (if any).
    pub target_hour: Option<u8>,

    /// Target day of week used (if any).
    pub target_day_of_week: Option<u8>,

    /// Periodic boost weight used.
    pub periodic_weight: f32,

    /// Whether auto-detect was used.
    pub auto_detected: bool,
}
```

#### 1.4 Handler Implementation

File: `crates/context-graph-mcp/src/handlers/tools/temporal_tools.rs`

```rust
impl Handlers {
    /// search_periodic tool implementation.
    ///
    /// Searches with E3 periodic boost applied POST-RETRIEVAL per ARCH-25.
    /// Returns results sorted by boosted score (semantic * periodic boost).
    pub(crate) async fn call_search_periodic(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse parameters
        let params: SearchPeriodicParams = match serde_json::from_value(args) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "search_periodic: Parameter parsing FAILED");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        if params.query.is_empty() {
            return self.tool_error_with_pulse(id, "Query cannot be empty");
        }

        // Validate hour and day if provided
        if let Some(hour) = params.target_hour {
            if hour > 23 {
                return self.tool_error_with_pulse(id, "targetHour must be 0-23");
            }
        }
        if let Some(dow) = params.target_day_of_week {
            if dow > 6 {
                return self.tool_error_with_pulse(id, "targetDayOfWeek must be 0-6 (Sun-Sat)");
            }
        }

        // Clamp periodic weight
        let periodic_weight = params.periodic_weight.clamp(0.1, 1.0);

        // Compute effective targets (auto-detect if needed)
        let now = Utc::now();
        let effective_hour = if params.auto_detect && params.target_hour.is_none() {
            Some(now.hour() as u8)
        } else {
            params.target_hour
        };
        let effective_dow = if params.auto_detect && params.target_day_of_week.is_none() {
            Some(now.weekday().num_days_from_sunday() as u8)
        } else {
            params.target_day_of_week
        };

        debug!(
            query_preview = %params.query.chars().take(50).collect::<String>(),
            top_k = params.top_k,
            effective_hour = ?effective_hour,
            effective_dow = ?effective_dow,
            periodic_weight = periodic_weight,
            "search_periodic: Starting periodic search"
        );

        // Generate query embedding with timestamp instruction for E3
        let timestamp_instruction = format!("timestamp:{}", now.to_rfc3339());
        let query_embedding = match self.multi_array_provider
            .embed_all_with_instruction(&params.query, &timestamp_instruction)
            .await
        {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "search_periodic: Query embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Query embedding failed: {}", e));
            }
        };

        // Build search options - use E1Only strategy for base search
        let mut options = TeleologicalSearchOptions::default();
        options.top_k = params.top_k * 2; // Over-fetch for periodic reranking
        options.min_similarity = params.min_similarity;
        options.strategy = SearchStrategy::E1Only;
        options.include_content = params.include_content;

        // Run base semantic search
        let results = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "search_periodic: Search FAILED");
                return self.tool_error_with_pulse(id, &format!("Search failed: {}", e));
            }
        };

        if results.is_empty() {
            return self.tool_result_with_pulse(
                id,
                serde_json::to_value(SearchPeriodicResponse {
                    query: params.query,
                    results: vec![],
                    count: 0,
                    periodic_config: PeriodicConfigSummary {
                        target_hour: effective_hour,
                        target_day_of_week: effective_dow,
                        periodic_weight,
                        auto_detected: params.auto_detect,
                    },
                })
                .unwrap(),
            );
        }

        // Apply E3 periodic boost POST-RETRIEVAL
        let mut boosted_results: Vec<PeriodicSearchResultEntry> = results
            .into_iter()
            .map(|r| {
                let memory_ts = r.fingerprint.created_at.timestamp_millis();
                let (memory_hour, memory_dow) = extract_temporal_components(memory_ts);

                // Compute E3 periodic score
                let periodic_score = if !query_embedding.e3_temporal_periodic.is_empty()
                    && !r.fingerprint.e3_temporal_periodic.is_empty()
                {
                    // Prefer embedding-based similarity
                    compute_e3_periodic_score(
                        &query_embedding.e3_temporal_periodic,
                        &r.fingerprint.e3_temporal_periodic,
                    )
                } else {
                    // Fall back to hour/day matching
                    compute_periodic_match_fallback(
                        effective_hour,
                        memory_hour,
                        effective_dow,
                        memory_dow,
                    )
                };

                // Per ARCH-25: Periodic boost is multiplicative POST-retrieval
                let boost_factor = (1.0 + periodic_weight * (periodic_score - 0.5)).clamp(0.8, 1.2);
                let final_score = r.similarity * boost_factor;

                PeriodicSearchResultEntry {
                    id: r.fingerprint.id.to_string(),
                    semantic_score: r.similarity,
                    periodic_score,
                    final_score,
                    memory_hour,
                    memory_day_of_week: memory_dow,
                    content: r.content,
                    created_at: r.fingerprint.created_at.to_rfc3339(),
                }
            })
            .collect();

        // Sort by final score descending
        boosted_results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to requested top_k
        boosted_results.truncate(params.top_k);
        let count = boosted_results.len();

        debug!(
            result_count = count,
            "search_periodic: Periodic search complete"
        );

        // Build response
        let response = SearchPeriodicResponse {
            query: params.query,
            results: boosted_results,
            count,
            periodic_config: PeriodicConfigSummary {
                target_hour: effective_hour,
                target_day_of_week: effective_dow,
                periodic_weight,
                auto_detected: params.auto_detect,
            },
        };

        match serde_json::to_value(&response) {
            Ok(json) => self.tool_result_with_pulse(id, json),
            Err(e) => {
                error!(error = %e, "search_periodic: Response serialization FAILED");
                self.tool_error_with_pulse(id, &format!("Response serialization failed: {}", e))
            }
        }
    }
}
```

#### 1.5 Dispatch Registration

File: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

```rust
// Add in the TEMPORAL TOOLS section
tool_names::SEARCH_PERIODIC => self.call_search_periodic(id, arguments).await,
```

### Phase 2: Query Type Detection Enhancement

**Objective**: Auto-detect periodic queries in the enrichment pipeline.

#### 2.1 Periodic Query Patterns

File: `crates/context-graph-core/src/retrieval/query_detection.rs` (or similar)

```rust
/// Patterns that indicate a periodic/temporal query
pub const PERIODIC_PATTERNS: &[&str] = &[
    // Time-of-day patterns
    "morning", "afternoon", "evening", "night",
    "AM", "PM", "o'clock",
    "breakfast", "lunch", "dinner",
    "at 9", "at 10", "at 3pm",

    // Day-of-week patterns
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "weekday", "weekend",
    "every monday", "on fridays",

    // Routine patterns
    "usually", "typically", "always", "daily", "weekly",
    "standup", "meeting", "review",
    "morning routine", "end of day",
];

/// Detect if a query has periodic intent
pub fn detect_periodic_query(query: &str) -> Option<PeriodicQueryHint> {
    let lower = query.to_lowercase();

    // Hour detection
    let hour_hint = detect_hour_hint(&lower);

    // Day-of-week detection
    let dow_hint = detect_day_of_week_hint(&lower);

    if hour_hint.is_some() || dow_hint.is_some() {
        Some(PeriodicQueryHint {
            hour: hour_hint,
            day_of_week: dow_hint,
            confidence: compute_confidence(&lower),
        })
    } else {
        None
    }
}

pub struct PeriodicQueryHint {
    pub hour: Option<u8>,
    pub day_of_week: Option<u8>,
    pub confidence: f32,
}
```

#### 2.2 Enrichment Integration

In the autonomous enrichment system, add E3 as a potential enhancer for periodic queries:

```rust
// In query_type_detection
TEMPORAL: ["morning", "afternoon", "evening", "monday", "friday", ...] => [E3]

// But remember: E3 is POST-RETRIEVAL only (ARCH-25)
// So we don't add E3 to similarity fusion, but we DO flag the query
// for periodic post-retrieval boost
```

### Phase 3: embed_all_with_instruction Enhancement

**Objective**: Pass timestamp metadata to E3 during embedding.

Currently, `embed_all_with_metadata()` in `multi_array.rs` doesn't pass the timestamp to E3. This should be fixed:

```rust
// In embed_all_with_metadata()
// Current comment: "E2/E3: Currently use default behavior (could be extended to use metadata.timestamp)"

// Fix: Pass timestamp instruction to E3
let e3_instruction = metadata.timestamp
    .map(|ts| format!("timestamp:{}", ts.to_rfc3339()))
    .unwrap_or_default();

// Then use embed_with_instruction for E3
Self::timed_embed("E3_TemporalPeriodic", {
    let c = content_owned.clone();
    let inst = e3_instruction.clone();
    async move { e3.embed_with_instruction(&c, &inst).await }
}),
```

---

## Testing Plan

### Unit Tests

#### 1. E3 Embedding Correctness (Already Exists)

File: `crates/context-graph-embeddings/src/models/custom/temporal_periodic/tests/periodicity.rs`

- Same time-of-day similarity (cosine > 0.99)
- Different time-of-day dissimilarity (6AM vs 6PM cosine < 0.5)
- Same day-of-week similarity
- Annual cycle preservation

#### 2. search_periodic Tool Tests (New)

File: `crates/context-graph-mcp/src/handlers/tests/search_periodic_test.rs`

```rust
#[tokio::test]
async fn test_search_periodic_with_hour_target() {
    // Store memories at different hours
    // Query with targetHour=9
    // Verify memories from 9am ranked higher
}

#[tokio::test]
async fn test_search_periodic_with_day_target() {
    // Store memories on different days
    // Query with targetDayOfWeek=5 (Friday)
    // Verify Friday memories ranked higher
}

#[tokio::test]
async fn test_search_periodic_auto_detect() {
    // Query with autoDetect=true
    // Verify current time is used for matching
}

#[tokio::test]
async fn test_search_periodic_validation_errors() {
    // targetHour=25 -> error
    // targetDayOfWeek=7 -> error
    // empty query -> error
}
```

### Integration Tests

#### 1. E3 Post-Retrieval Boost Accuracy

```rust
#[tokio::test]
async fn test_e3_boosts_correct_memories() {
    // Create memories:
    // - Memory A: 9am Monday, content "standup notes"
    // - Memory B: 9pm Monday, content "standup notes"
    // - Memory C: 9am Friday, content "standup notes"

    // Query: "standup" with targetHour=9
    // Expected: A ranked highest (same hour)
    // B should be ranked lowest (12 hours apart = opposite)
}
```

#### 2. E3 + E1 Combination

```rust
#[tokio::test]
async fn test_e3_enhances_e1_results() {
    // Query: "morning meeting" with autoDetect=true (run at 9am)
    // Verify E3 boosts 9am memories
    // Verify E1 still contributes semantic relevance
}
```

### Manual Test Scenarios

| Test | Input | Expected Output |
|------|-------|-----------------|
| Hour matching | Query "deployment" at 3pm | Memories from 2-4pm ranked higher |
| Day matching | Query "code review" on Friday | Friday memories ranked higher |
| Auto-detect | Query with autoDetect=true | Uses current time for matching |
| Combined | Query "friday standup" | Friday 9am memories highest |

---

## Performance Considerations

### E3 is Lightweight

E3 uses Fourier computation (pure math, no ML inference):
- No GPU required
- No model loading delay
- Computation: O(510) arithmetic operations per embedding
- Memory: 512 * 4 bytes = 2KB per embedding

### Post-Retrieval is Efficient

Per ARCH-25, E3 is applied POST-retrieval:
- No impact on initial HNSW search latency
- Only computed for top-K results (typically 10-100)
- Adds ~1ms for 100 results

---

## Error Handling

### Fail Fast Principle

No workarounds or fallbacks. If something fails, the system should error clearly:

```rust
// GOOD: Clear error with context
if params.target_hour.map_or(false, |h| h > 23) {
    return Err(ToolError::InvalidParameter {
        param: "targetHour",
        expected: "0-23",
        actual: params.target_hour.unwrap().to_string(),
    });
}

// BAD: Silent clamping that hides bugs
let hour = params.target_hour.unwrap_or(12).clamp(0, 23);
```

### Error Logging

All errors should include:
- Tool name
- Operation that failed
- Parameter values
- Root cause

```rust
error!(
    tool = "search_periodic",
    operation = "query_embedding",
    query_len = params.query.len(),
    error = %e,
    "search_periodic: Query embedding FAILED - check model availability"
);
```

---

## Migration Notes

### No Breaking Changes

This implementation is additive:
- New tool `search_periodic` doesn't affect existing tools
- E3 embeddings are already being generated
- No schema changes required

### Backwards Compatibility

Not applicable - this is new functionality.

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/tools/names.rs` | Add `SEARCH_PERIODIC` constant |
| `crates/context-graph-mcp/src/tools/definitions/temporal.rs` | Add tool definition |
| `crates/context-graph-mcp/src/handlers/tools/temporal_dtos.rs` | Add DTOs |
| `crates/context-graph-mcp/src/handlers/tools/temporal_tools.rs` | Add handler |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Add dispatch entry |
| `crates/context-graph-mcp/src/tools/definitions/mod.rs` | Include temporal module (if not already) |
| `crates/context-graph-embeddings/src/provider/multi_array.rs` | Pass timestamp to E3 (Phase 3) |

---

## Verification Checklist

- [ ] E3 embeddings are generated with correct timestamps
- [ ] search_periodic tool is registered in dispatch
- [ ] search_periodic appears in tools/list
- [ ] Hour validation rejects values > 23
- [ ] Day-of-week validation rejects values > 6
- [ ] Auto-detect correctly uses current time
- [ ] Periodic boost is applied POST-retrieval (not in similarity fusion)
- [ ] Results include semantic_score, periodic_score, final_score
- [ ] Memories from target hour/day are ranked higher
- [ ] Error messages are clear and actionable

---

## Summary

E3's infrastructure is complete. The optimization focuses on:

1. **Phase 1**: Expose E3 via `search_periodic` MCP tool (similar to `search_recent` for E2)
2. **Phase 2**: Add periodic query detection to auto-enhance periodic queries
3. **Phase 3**: Pass timestamp metadata to E3 during embedding

This aligns with the Constitution's philosophy that E3 provides **signal** (time-of-day patterns) that E1 cannot encode. E3 enhances E1 by finding "what did I work on at 9am?" when E1 only finds "what is semantically similar?".
