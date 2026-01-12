//! History and trend analysis for drift detection.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use super::types::{DriftThresholds, DriftTrend, MIN_TREND_SAMPLES, NUM_EMBEDDERS};

// ============================================
// TREND ANALYSIS
// ============================================

/// Trend analysis over time using linear regression.
#[derive(Debug)]
pub struct TrendAnalysis {
    /// Direction of the trend
    pub direction: DriftTrend,
    /// Velocity: absolute value of slope
    pub velocity: f32,
    /// Number of history samples used
    pub samples: usize,
    /// Projected time until critical drift (if worsening)
    pub projected_critical_in: Option<String>,
}

// ============================================
// HISTORY TYPES
// ============================================

/// History of drift measurements for trend analysis.
#[derive(Debug, Default)]
pub struct DriftHistory {
    /// Entries keyed by goal_id
    entries: HashMap<String, Vec<DriftHistoryEntry>>,
    /// Maximum entries to keep per goal
    max_entries_per_goal: usize,
}

impl DriftHistory {
    /// Create a new history with specified max entries per goal.
    pub fn new(max_entries_per_goal: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries_per_goal,
        }
    }

    /// Add an entry for a goal.
    pub fn add(&mut self, goal_id: &str, entry: DriftHistoryEntry) {
        let entries = self.entries.entry(goal_id.to_string()).or_default();
        entries.push(entry);

        // Trim to max entries
        while entries.len() > self.max_entries_per_goal {
            entries.remove(0);
        }
    }

    /// Get entries for a goal.
    pub fn get(&self, goal_id: &str) -> Option<&Vec<DriftHistoryEntry>> {
        self.entries.get(goal_id)
    }

    /// Clear entries for a goal.
    pub fn clear(&mut self, goal_id: &str) {
        self.entries.remove(goal_id);
    }

    /// Get entry count for a goal.
    pub fn len(&self, goal_id: &str) -> usize {
        self.entries.get(goal_id).map(|e| e.len()).unwrap_or(0)
    }
}

/// Single history entry with per-embedder breakdown.
#[derive(Debug, Clone)]
pub struct DriftHistoryEntry {
    /// Timestamp of this measurement
    pub timestamp: DateTime<Utc>,
    /// Overall similarity score
    pub overall_similarity: f32,
    /// Per-embedder similarity scores
    pub per_embedder: [f32; NUM_EMBEDDERS],
    /// Number of memories analyzed
    pub memories_analyzed: usize,
}

/// Compute trend from history using linear regression.
pub fn compute_trend(
    entries: &[DriftHistoryEntry],
    thresholds: &DriftThresholds,
) -> Option<TrendAnalysis> {
    if entries.len() < MIN_TREND_SAMPLES {
        return None;
    }

    // Simple linear regression on overall similarity
    let n = entries.len() as f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_xy = 0.0f32;
    let mut sum_xx = 0.0f32;

    for (i, entry) in entries.iter().enumerate() {
        let x = i as f32;
        let y = entry.overall_similarity;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    // slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < f32::EPSILON {
        return Some(TrendAnalysis {
            direction: DriftTrend::Stable,
            velocity: 0.0,
            samples: entries.len(),
            projected_critical_in: None,
        });
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;

    let direction = if slope.abs() < 0.01 {
        DriftTrend::Stable
    } else if slope > 0.0 {
        DriftTrend::Improving
    } else {
        DriftTrend::Worsening
    };

    // Project time to critical (similarity < 0.40) if worsening
    let projected_critical_in = if direction == DriftTrend::Worsening {
        let current_sim = entries.last()?.overall_similarity;
        let critical_threshold = thresholds.high_min; // 0.40

        if current_sim > critical_threshold && slope < 0.0 {
            // Steps until critical = (current_sim - critical_threshold) / |slope|
            let steps = (current_sim - critical_threshold) / slope.abs();
            // Assuming ~1 measurement per check, convert to human-readable
            Some(format!("{:.1} checks at current rate", steps))
        } else {
            None
        }
    } else {
        None
    };

    Some(TrendAnalysis {
        direction,
        velocity: slope.abs(),
        samples: entries.len(),
        projected_critical_in,
    })
}
