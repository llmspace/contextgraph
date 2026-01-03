---
id: "M04-T05"
title: "Implement PoincareBall Mobius Operations"
description: |
  Implement PoincareBall struct with Mobius algebra operations for hyperbolic geometry.
  Methods: mobius_add(x, y), distance(x, y), exp_map(x, v), log_map(x, y).
  Performance target: <10us per distance computation.
layer: "foundation"
status: "pending"
priority: "critical"
estimated_hours: 4
sequence: 8
depends_on:
  - "M04-T04"  # PoincarePoint - VERIFIED COMPLETE
spec_refs:
  - "TECH-GRAPH-004 Section 5.2"
  - "REQ-KG-051"
files_to_create:
  - path: "crates/context-graph-graph/src/hyperbolic/mobius.rs"
    description: "PoincareBall struct and Mobius operations"
files_to_modify:
  - path: "crates/context-graph-graph/src/hyperbolic/mod.rs"
    description: "Add mobius module and re-export PoincareBall"
  - path: "crates/context-graph-graph/src/lib.rs"
    description: "Re-export PoincareBall at crate root"
test_file: "crates/context-graph-graph/src/hyperbolic/mobius.rs"
  # Tests are co-located in #[cfg(test)] per constitution
---

## CRITICAL CODEBASE STATE (Verified 2026-01-03)

### Dependency Status: ALL SATISFIED

| Dependency | Status | Location |
|------------|--------|----------|
| M04-T00 (Crate) | COMPLETE | `crates/context-graph-graph/` exists |
| M04-T02 (HyperbolicConfig) | COMPLETE | `src/config.rs:137-330` |
| M04-T04 (PoincarePoint) | COMPLETE | `src/hyperbolic/poincare.rs:52-340` |
| GraphError | COMPLETE | `src/error.rs:24-160` |

### Current File Structure (ACTUAL)

```
crates/context-graph-graph/
├── Cargo.toml
└── src/
    ├── lib.rs                    # Crate root, re-exports
    ├── config.rs                 # IndexConfig, HyperbolicConfig, ConeConfig
    ├── error.rs                  # GraphError enum
    └── hyperbolic/
        ├── mod.rs                # Module declaration (needs mobius added)
        └── poincare.rs           # PoincarePoint (64D, 256 bytes, 64-byte aligned)
```

### Verified Types Available

```rust
// From crates/context-graph-graph/src/config.rs:137-168
pub struct HyperbolicConfig {
    pub dim: usize,           // Default: 64
    pub curvature: f32,       // Default: -1.0 (MUST be negative)
    pub eps: f32,             // Default: 1e-7
    pub max_norm: f32,        // Default: 0.99999 (1.0 - 1e-5)
}
impl HyperbolicConfig {
    pub fn abs_curvature(&self) -> f32;  // Returns |curvature|
    pub fn scale(&self) -> f32;           // Returns sqrt(|curvature|)
    pub fn validate(&self) -> Result<(), GraphError>;
}

// From crates/context-graph-graph/src/hyperbolic/poincare.rs:52-58
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct PoincarePoint {
    pub coords: [f32; 64],    // 64D coordinates
}
impl PoincarePoint {
    pub fn origin() -> Self;
    pub fn from_coords(coords: [f32; 64]) -> Self;
    pub fn from_coords_projected(coords: [f32; 64], config: &HyperbolicConfig) -> Self;
    pub fn norm_squared(&self) -> f32;
    pub fn norm(&self) -> f32;
    pub fn project(&mut self, config: &HyperbolicConfig);
    pub fn projected(&self, config: &HyperbolicConfig) -> Self;
    pub fn is_valid(&self) -> bool;          // norm_squared < 1.0
    pub fn is_valid_for_config(&self, config: &HyperbolicConfig) -> bool;
}

// From crates/context-graph-graph/src/error.rs
pub enum GraphError {
    InvalidHyperbolicPoint { norm: f32 },
    MobiusOperationFailed(String),
    InvalidConfig(String),
    // ... other variants
}
pub type GraphResult<T> = Result<T, GraphError>;
```

## Implementation Requirements

### File: `crates/context-graph-graph/src/hyperbolic/mobius.rs`

Create this file with the following exact implementation:

```rust
//! PoincareBall implementation with Mobius algebra operations.
//!
//! # Poincare Ball Model
//!
//! The Poincare ball model represents hyperbolic space as the open unit ball.
//! Mobius operations provide the algebra for vector addition, distances, and
//! exponential/logarithmic maps between tangent spaces and the manifold.
//!
//! # Mathematics
//!
//! - Mobius addition: x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
//!                           (1 + 2c<x,y> + c²||x||²||y||²)
//! - Distance: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
//! - Exp map: Maps tangent vector at x to point on manifold
//! - Log map: Maps point y to tangent vector at x (inverse of exp_map)
//!
//! # Performance Targets
//!
//! - distance(): <10μs per pair
//! - mobius_add(): <5μs per operation
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms (this contributes ~1% budget)
//! - contextprd.md Section 4.4: Poincare Ball d(x,y) formula

use crate::config::HyperbolicConfig;
use crate::hyperbolic::poincare::PoincarePoint;

/// Poincare ball model with Mobius algebra operations.
///
/// Provides hyperbolic geometry operations for the knowledge graph's
/// hierarchical embeddings. Points near origin represent general concepts;
/// points near boundary represent specific concepts.
///
/// # Example
///
/// ```
/// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let config = HyperbolicConfig::default();
/// let ball = PoincareBall::new(config);
///
/// let origin = PoincarePoint::origin();
/// let mut coords = [0.0f32; 64];
/// coords[0] = 0.5;
/// let point = PoincarePoint::from_coords(coords);
///
/// // Distance from origin
/// let d = ball.distance(&origin, &point);
/// assert!(d > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct PoincareBall {
    config: HyperbolicConfig,
}

impl PoincareBall {
    /// Create a new Poincare ball with given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - HyperbolicConfig with curvature, eps, max_norm settings
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincareBall;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default();
    /// let ball = PoincareBall::new(config);
    /// assert_eq!(ball.config().curvature, -1.0);
    /// ```
    #[inline]
    pub fn new(config: HyperbolicConfig) -> Self {
        Self { config }
    }

    /// Get reference to configuration.
    #[inline]
    pub fn config(&self) -> &HyperbolicConfig {
        &self.config
    }

    /// Mobius addition in Poincare ball.
    ///
    /// Formula: x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
    ///                  (1 + 2c<x,y> + c²||x||²||y||²)
    ///
    /// where c = |curvature|
    ///
    /// # Arguments
    ///
    /// * `x` - First point in Poincare ball
    /// * `y` - Second point in Poincare ball
    ///
    /// # Returns
    ///
    /// Result point, projected to stay inside ball if needed.
    ///
    /// # Performance
    ///
    /// Target: <5μs. O(64) operations with potential SIMD optimization.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.3;
    /// let point = PoincarePoint::from_coords(coords);
    ///
    /// // Adding origin returns the other point
    /// let result = ball.mobius_add(&origin, &point);
    /// assert!((result.coords[0] - 0.3).abs() < 1e-6);
    /// ```
    pub fn mobius_add(&self, x: &PoincarePoint, y: &PoincarePoint) -> PoincarePoint {
        let c = self.config.abs_curvature();
        let x_norm_sq = x.norm_squared();
        let y_norm_sq = y.norm_squared();

        // Inner product <x, y>
        let xy_dot: f32 = x.coords.iter()
            .zip(y.coords.iter())
            .map(|(a, b)| a * b)
            .sum();

        let num_coeff_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
        let num_coeff_y = 1.0 - c * x_norm_sq;
        let denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

        // Avoid division by zero
        let safe_denom = if denom.abs() < self.config.eps {
            self.config.eps
        } else {
            denom
        };

        let mut result = PoincarePoint::origin();
        for i in 0..64 {
            result.coords[i] = (num_coeff_x * x.coords[i] + num_coeff_y * y.coords[i]) / safe_denom;
        }

        // Project to ensure result stays inside ball
        result.project(&self.config);
        result
    }

    /// Compute Poincare ball distance between two points.
    ///
    /// Formula: d(x,y) = (2/√c) * arctanh(√(c * ||x-y||² / ((1 - c||x||²)(1 - c||y||²))))
    ///
    /// # Arguments
    ///
    /// * `x` - First point
    /// * `y` - Second point
    ///
    /// # Returns
    ///
    /// Hyperbolic distance (always >= 0).
    ///
    /// # Performance
    ///
    /// Target: <10μs. O(64) with one sqrt and one atanh.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    ///
    /// // Distance from point to itself is 0
    /// let point = PoincarePoint::origin();
    /// assert_eq!(ball.distance(&point, &point), 0.0);
    ///
    /// // Distance is symmetric
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.5;
    /// let p1 = PoincarePoint::origin();
    /// let p2 = PoincarePoint::from_coords(coords);
    /// let d1 = ball.distance(&p1, &p2);
    /// let d2 = ball.distance(&p2, &p1);
    /// assert!((d1 - d2).abs() < 1e-6);
    /// ```
    pub fn distance(&self, x: &PoincarePoint, y: &PoincarePoint) -> f32 {
        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // ||x - y||²
        let diff_norm_sq: f32 = x.coords.iter()
            .zip(y.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        // Early exit for identical points
        if diff_norm_sq < self.config.eps * self.config.eps {
            return 0.0;
        }

        let x_norm_sq = x.norm_squared();
        let y_norm_sq = y.norm_squared();

        // Conformal factors: λ_x = 1 - c||x||², λ_y = 1 - c||y||²
        let lambda_x = 1.0 - c * x_norm_sq;
        let lambda_y = 1.0 - c * y_norm_sq;

        // Avoid division by zero near boundary
        let denom = (lambda_x * lambda_y).max(self.config.eps);

        // arctanh argument, clamped to avoid NaN
        let arg = (c * diff_norm_sq / denom).sqrt().min(1.0 - self.config.eps);

        (2.0 / sqrt_c) * arg.atanh()
    }

    /// Exponential map: tangent vector at x -> point on manifold.
    ///
    /// Maps a vector v in the tangent space at x to a point on the Poincare ball
    /// along the geodesic starting at x with initial direction v.
    ///
    /// # Arguments
    ///
    /// * `x` - Base point on manifold
    /// * `v` - Tangent vector at x (64D array)
    ///
    /// # Returns
    ///
    /// Point on Poincare ball.
    ///
    /// # Mathematical Formula
    ///
    /// exp_x(v) = x ⊕ (tanh(√c ||v|| / λ_x) * v / (√c ||v||))
    ///
    /// where λ_x = 1 - c||x||² is the conformal factor.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    ///
    /// // Zero tangent vector returns base point
    /// let zero_v = [0.0f32; 64];
    /// let result = ball.exp_map(&origin, &zero_v);
    /// assert_eq!(result.coords[0], 0.0);
    /// ```
    pub fn exp_map(&self, x: &PoincarePoint, v: &[f32; 64]) -> PoincarePoint {
        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // ||v|| in tangent space
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle zero tangent vector
        if v_norm < self.config.eps {
            return x.clone();
        }

        // Conformal factor at x
        let x_norm_sq = x.norm_squared();
        let lambda_x = (1.0 - c * x_norm_sq).max(self.config.eps);

        // Scaled tangent norm
        let scaled_norm = sqrt_c * v_norm / lambda_x;

        let tanh_factor = scaled_norm.tanh();

        // Create direction point: (tanh_factor / (sqrt_c * v_norm)) * v
        let mut direction = PoincarePoint::origin();
        let scale = tanh_factor / (sqrt_c * v_norm);
        for i in 0..64 {
            direction.coords[i] = scale * v[i];
        }

        // Mobius add with x
        self.mobius_add(x, &direction)
    }

    /// Logarithmic map: point on manifold -> tangent vector at x.
    ///
    /// Returns the tangent vector at x that points toward y.
    /// This is the inverse of exp_map.
    ///
    /// # Arguments
    ///
    /// * `x` - Base point
    /// * `y` - Target point
    ///
    /// # Returns
    ///
    /// Tangent vector at x pointing toward y.
    ///
    /// # Mathematical Formula
    ///
    /// log_x(y) = (2 λ_x / √c) * arctanh(√c ||(-x) ⊕ y||) * ((-x) ⊕ y) / ||(-x) ⊕ y||
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    ///
    /// // Log map from point to itself returns zero vector
    /// let v = ball.log_map(&origin, &origin);
    /// assert!(v.iter().all(|&x| x.abs() < 1e-6));
    /// ```
    pub fn log_map(&self, x: &PoincarePoint, y: &PoincarePoint) -> [f32; 64] {
        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // Compute (-x) ⊕ y
        let mut neg_x = x.clone();
        for coord in &mut neg_x.coords {
            *coord = -*coord;
        }
        let diff = self.mobius_add(&neg_x, y);

        let diff_norm = diff.norm();

        // Handle identical points
        if diff_norm < self.config.eps {
            return [0.0; 64];
        }

        // Conformal factor at x
        let x_norm_sq = x.norm_squared();
        let lambda_x = (1.0 - c * x_norm_sq).max(self.config.eps);

        // arctanh(√c * ||(-x) ⊕ y||), clamped to avoid NaN
        let arg = (sqrt_c * diff_norm).min(1.0 - self.config.eps);
        let scale = (2.0 * lambda_x / sqrt_c) * arg.atanh() / diff_norm;

        let mut result = [0.0; 64];
        for i in 0..64 {
            result[i] = scale * diff.coords[i];
        }

        result
    }
}

// ============================================================================
// TESTS - REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ball() -> PoincareBall {
        PoincareBall::new(HyperbolicConfig::default())
    }

    fn make_point(first_coord: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = first_coord;
        PoincarePoint::from_coords(coords)
    }

    fn make_point_2d(x: f32, y: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = x;
        coords[1] = y;
        PoincarePoint::from_coords(coords)
    }

    // ========== CONSTRUCTION TESTS ==========

    #[test]
    fn test_new_creates_ball_with_config() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config.clone());
        assert_eq!(ball.config().curvature, -0.5);
    }

    #[test]
    fn test_config_accessor() {
        let ball = default_ball();
        assert_eq!(ball.config().dim, 64);
        assert_eq!(ball.config().curvature, -1.0);
    }

    // ========== MOBIUS ADDITION TESTS ==========

    #[test]
    fn test_mobius_add_with_origin_returns_other() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let point = make_point(0.3);

        // x ⊕ 0 = x
        let result = ball.mobius_add(&point, &origin);
        assert!((result.coords[0] - 0.3).abs() < 1e-5);

        // 0 ⊕ y = y
        let result2 = ball.mobius_add(&origin, &point);
        assert!((result2.coords[0] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_mobius_add_result_inside_ball() {
        let ball = default_ball();
        let p1 = make_point(0.5);
        let p2 = make_point(0.3);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid(), "Mobius add result must be inside ball");
    }

    #[test]
    fn test_mobius_add_near_boundary() {
        let ball = default_ball();
        // Points close to boundary
        let p1 = make_point(0.9);
        let p2 = make_point(0.8);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid_for_config(ball.config()));
    }

    #[test]
    fn test_mobius_add_opposite_directions() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.3);

        // Opposite points should partially cancel
        let result = ball.mobius_add(&p1, &p2);
        assert!(result.norm() < 0.3);
    }

    // ========== DISTANCE TESTS ==========

    #[test]
    fn test_distance_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.5);
        assert_eq!(ball.distance(&point, &point), 0.0);
    }

    #[test]
    fn test_distance_origin_to_origin_is_zero() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        assert_eq!(ball.distance(&origin, &origin), 0.0);
    }

    #[test]
    fn test_distance_is_symmetric() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        let d1 = ball.distance(&p1, &p2);
        let d2 = ball.distance(&p2, &p1);
        assert!((d1 - d2).abs() < 1e-6, "Distance must be symmetric");
    }

    #[test]
    fn test_distance_is_nonnegative() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.5);
        assert!(ball.distance(&p1, &p2) >= 0.0);
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let ball = default_ball();
        let p1 = make_point(0.1);
        let p2 = make_point(0.3);
        let p3 = make_point(0.5);

        let d12 = ball.distance(&p1, &p2);
        let d23 = ball.distance(&p2, &p3);
        let d13 = ball.distance(&p1, &p3);

        // d(p1, p3) <= d(p1, p2) + d(p2, p3)
        assert!(d13 <= d12 + d23 + 1e-6, "Triangle inequality violated");
    }

    #[test]
    fn test_distance_from_origin_monotonic() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();

        // Distance increases as we move further from origin
        let p1 = make_point(0.1);
        let p2 = make_point(0.5);
        let p3 = make_point(0.9);

        let d1 = ball.distance(&origin, &p1);
        let d2 = ball.distance(&origin, &p2);
        let d3 = ball.distance(&origin, &p3);

        assert!(d1 < d2, "d(0, 0.1) < d(0, 0.5)");
        assert!(d2 < d3, "d(0, 0.5) < d(0, 0.9)");
    }

    #[test]
    fn test_distance_near_boundary_larger() {
        let ball = default_ball();
        // In hyperbolic space, distances near boundary are larger
        let origin = PoincarePoint::origin();
        let near_boundary = make_point(0.99);

        let d = ball.distance(&origin, &near_boundary);
        // For c=-1, d(0, r) = 2 * arctanh(r), so d(0, 0.99) ≈ 5.3
        assert!(d > 4.0, "Distance near boundary should be large: {}", d);
    }

    // ========== EXP MAP TESTS ==========

    #[test]
    fn test_exp_map_zero_tangent_returns_base() {
        let ball = default_ball();
        let base = make_point(0.3);
        let zero_v = [0.0f32; 64];

        let result = ball.exp_map(&base, &zero_v);
        assert!((result.coords[0] - base.coords[0]).abs() < 1e-6);
    }

    #[test]
    fn test_exp_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v = [0.0f32; 64];
        v[0] = 0.5;

        let result = ball.exp_map(&origin, &v);
        assert!(result.is_valid());
        assert!(result.coords[0] > 0.0, "Should move in direction of v");
    }

    #[test]
    fn test_exp_map_result_inside_ball() {
        let ball = default_ball();
        let base = make_point(0.5);
        let mut v = [0.0f32; 64];
        v[0] = 10.0; // Large tangent vector

        let result = ball.exp_map(&base, &v);
        assert!(result.is_valid_for_config(ball.config()), "exp_map result must be inside ball");
    }

    // ========== LOG MAP TESTS ==========

    #[test]
    fn test_log_map_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.3);

        let v = ball.log_map(&point, &point);
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(v_norm < 1e-6, "log_map(x, x) should be zero vector");
    }

    #[test]
    fn test_log_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let target = make_point(0.3);

        let v = ball.log_map(&origin, &target);
        // Should point in positive x direction
        assert!(v[0] > 0.0);
    }

    // ========== ROUND-TRIP TESTS (CRITICAL) ==========

    #[test]
    fn test_exp_log_roundtrip_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v_orig = [0.0f32; 64];
        v_orig[0] = 0.5;
        v_orig[1] = 0.3;

        // exp_map -> log_map should recover original tangent vector
        let point = ball.exp_map(&origin, &v_orig);
        let v_recovered = ball.log_map(&origin, &point);

        for i in 0..64 {
            assert!(
                (v_orig[i] - v_recovered[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i, v_orig[i], v_recovered[i]
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let ball = default_ball();
        let base = make_point(0.2);
        let target = make_point(0.5);

        // log_map -> exp_map should approximately recover target
        let v = ball.log_map(&base, &target);
        let recovered = ball.exp_map(&base, &v);

        for i in 0..64 {
            assert!(
                (target.coords[i] - recovered.coords[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i, target.coords[i], recovered.coords[i]
            );
        }
    }

    // ========== EDGE CASES ==========

    #[test]
    fn test_distance_with_nan_coords_returns_nan() {
        let ball = default_ball();
        let mut coords = [0.0f32; 64];
        coords[0] = f32::NAN;
        let p1 = PoincarePoint::from_coords(coords);
        let p2 = PoincarePoint::origin();

        let d = ball.distance(&p1, &p2);
        assert!(d.is_nan(), "Distance with NaN input should be NaN");
    }

    #[test]
    fn test_mobius_add_handles_small_denominator() {
        let ball = default_ball();
        // Create points that might cause small denominator
        let mut coords1 = [0.0f32; 64];
        let mut coords2 = [0.0f32; 64];
        coords1[0] = 0.99;
        coords2[0] = -0.99;

        let p1 = PoincarePoint::from_coords(coords1);
        let p2 = PoincarePoint::from_coords(coords2);

        let result = ball.mobius_add(&p1, &p2);
        // Should not panic or produce NaN
        assert!(!result.coords[0].is_nan(), "Should handle small denominator");
        assert!(result.is_valid());
    }

    #[test]
    fn test_custom_curvature() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config);

        let p1 = PoincarePoint::origin();
        let p2 = make_point(0.5);

        let d = ball.distance(&p1, &p2);
        // With lower curvature magnitude, distances should be different
        assert!(d > 0.0);
        assert!(!d.is_nan());
    }

    // ========== MATHEMATICAL PROPERTY TESTS ==========

    #[test]
    fn test_distance_formula_verification() {
        // Verify against known formula: d(0, r) = 2 * arctanh(r) for c=-1
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let r = 0.5;
        let point = make_point(r);

        let computed = ball.distance(&origin, &point);
        let expected = 2.0 * r.atanh();

        assert!(
            (computed - expected).abs() < 1e-5,
            "Distance formula mismatch: computed={}, expected={}",
            computed, expected
        );
    }

    // ========== PERFORMANCE SANITY TESTS ==========

    #[test]
    fn test_distance_many_calls() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        // Run many iterations to check for consistency
        let first_distance = ball.distance(&p1, &p2);
        for _ in 0..1000 {
            let d = ball.distance(&p1, &p2);
            assert!((d - first_distance).abs() < 1e-10, "Distance should be deterministic");
        }
    }
}
```

### File: `crates/context-graph-graph/src/hyperbolic/mod.rs`

Update to add mobius module:

```rust
//! Hyperbolic geometry module using Poincare ball model.
//!
//! This module implements hyperbolic geometry operations for representing
//! hierarchical relationships in the knowledge graph. Points closer to the
//! boundary represent more specific concepts; points near origin are general.
//!
//! # Mathematics
//!
//! The Poincare ball model uses the unit ball B^n = {x in R^n : ||x|| < 1}
//! with the metric:
//!
//! ```text
//! d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
//! ```
//!
//! # Components
//!
//! - [`PoincarePoint`]: 64D point in hyperbolic space
//! - [`PoincareBall`]: Mobius operations (add, distance, exp/log maps)
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//! - hyperbolic.curvature: -1.0 (default)
//!
//! # GPU Acceleration
//!
//! CUDA kernels for batch operations: TODO: M04-T23

pub mod poincare;
pub mod mobius;

pub use poincare::PoincarePoint;
pub use mobius::PoincareBall;
```

### File: `crates/context-graph-graph/src/lib.rs`

Update the re-exports to include PoincareBall:

Find this line:
```rust
pub use hyperbolic::PoincarePoint;
```

Replace with:
```rust
pub use hyperbolic::{PoincareBall, PoincarePoint};
```

## Acceptance Criteria

### Signatures (MUST MATCH EXACTLY)

- [ ] `PoincareBall::new(config: HyperbolicConfig) -> Self`
- [ ] `PoincareBall::config(&self) -> &HyperbolicConfig`
- [ ] `PoincareBall::mobius_add(&self, x: &PoincarePoint, y: &PoincarePoint) -> PoincarePoint`
- [ ] `PoincareBall::distance(&self, x: &PoincarePoint, y: &PoincarePoint) -> f32`
- [ ] `PoincareBall::exp_map(&self, x: &PoincarePoint, v: &[f32; 64]) -> PoincarePoint`
- [ ] `PoincareBall::log_map(&self, x: &PoincarePoint, y: &PoincarePoint) -> [f32; 64]`

### Mathematical Constraints (MUST HOLD)

- [ ] `mobius_add(x, origin) == x` (identity)
- [ ] `mobius_add(origin, y) == y` (identity)
- [ ] `distance(x, x) == 0` (reflexivity)
- [ ] `distance(x, y) == distance(y, x)` (symmetry)
- [ ] `distance(x, z) <= distance(x, y) + distance(y, z)` (triangle inequality)
- [ ] `exp_map(x, log_map(x, y)) ≈ y` (roundtrip)
- [ ] `log_map(x, exp_map(x, v)) ≈ v` (roundtrip)
- [ ] All results stay inside Poincare ball (norm < max_norm)

### Performance (VERIFY WITH TIMING)

- [ ] `distance()` completes in <10μs per call
- [ ] `mobius_add()` completes in <5μs per call

## Verification Commands

```bash
# 1. Build the crate
cargo build -p context-graph-graph

# 2. Run all tests (must pass 100%)
cargo test -p context-graph-graph mobius -- --nocapture

# 3. Run specific tests
cargo test -p context-graph-graph test_distance_is_symmetric
cargo test -p context-graph-graph test_exp_log_roundtrip

# 4. Run clippy (must have 0 warnings)
cargo clippy -p context-graph-graph -- -D warnings

# 5. Run doc tests
cargo test -p context-graph-graph --doc

# 6. Check documentation
cargo doc -p context-graph-graph --open
```

## Full State Verification Protocol

After completing implementation, you MUST verify the entire state:

### 1. Source of Truth Identification

The source of truth for M04-T05 is:
- **File existence**: `crates/context-graph-graph/src/hyperbolic/mobius.rs`
- **Module registration**: `pub mod mobius;` in `hyperbolic/mod.rs`
- **Re-export**: `pub use hyperbolic::PoincareBall;` in `lib.rs`
- **Test results**: `cargo test -p context-graph-graph mobius`

### 2. Execute & Inspect Protocol

After writing code, run these commands and capture output:

```bash
# Verify file exists
ls -la crates/context-graph-graph/src/hyperbolic/mobius.rs

# Verify module is registered
grep "pub mod mobius" crates/context-graph-graph/src/hyperbolic/mod.rs

# Verify re-export
grep "PoincareBall" crates/context-graph-graph/src/lib.rs

# Build and capture output
cargo build -p context-graph-graph 2>&1

# Run tests and capture output
cargo test -p context-graph-graph mobius 2>&1

# Run clippy
cargo clippy -p context-graph-graph -- -D warnings 2>&1
```

### 3. Boundary & Edge Case Audit

Manually test these 3 edge cases by adding temporary test output:

**Edge Case 1: Points at boundary (norm ≈ 0.99)**
```rust
#[test]
fn edge_case_boundary() {
    let ball = default_ball();
    let p1 = make_point(0.999);
    let p2 = make_point(0.998);
    println!("BEFORE: p1.norm={}, p2.norm={}", p1.norm(), p2.norm());
    let result = ball.mobius_add(&p1, &p2);
    println!("AFTER: result.norm={}, is_valid={}", result.norm(), result.is_valid());
    assert!(result.is_valid_for_config(ball.config()));
}
```

**Edge Case 2: Origin operations**
```rust
#[test]
fn edge_case_origin() {
    let ball = default_ball();
    let origin = PoincarePoint::origin();
    let p = make_point(0.5);
    println!("BEFORE: origin.norm={}, p.norm={}", origin.norm(), p.norm());
    let d = ball.distance(&origin, &p);
    let expected = 2.0 * 0.5_f32.atanh();
    println!("AFTER: distance={}, expected={}", d, expected);
    assert!((d - expected).abs() < 1e-5);
}
```

**Edge Case 3: Roundtrip precision**
```rust
#[test]
fn edge_case_roundtrip() {
    let ball = default_ball();
    let base = make_point(0.3);
    let target = make_point(0.7);
    println!("BEFORE: base={:?}, target.coords[0]={}", base.coords[0], target.coords[0]);
    let v = ball.log_map(&base, &target);
    let recovered = ball.exp_map(&base, &v);
    println!("AFTER: recovered.coords[0]={}, diff={}", recovered.coords[0], (target.coords[0] - recovered.coords[0]).abs());
    assert!((target.coords[0] - recovered.coords[0]).abs() < 1e-4);
}
```

### 4. Evidence of Success

Your final verification should produce a log showing:

```
=== M04-T05 VERIFICATION LOG ===

File exists: crates/context-graph-graph/src/hyperbolic/mobius.rs ✓
Module registered: pub mod mobius; ✓
Re-export found: pub use hyperbolic::PoincareBall; ✓

Build result: SUCCESS (0 errors, 0 warnings)
Test result: XX passed; 0 failed
Clippy result: 0 warnings

Edge Case 1 (boundary): PASS - result.norm=0.9999, is_valid=true
Edge Case 2 (origin): PASS - distance=1.0986, expected=1.0986
Edge Case 3 (roundtrip): PASS - diff=0.00001

=== TASK COMPLETE ===
```

## FINAL VERIFICATION: Sherlock-Holmes Agent

After completing ALL above steps, you MUST spawn the `sherlock-holmes` subagent to perform forensic verification:

```
Task("Verify M04-T05 complete",
     "Investigate M04-T05 PoincareBall Mobius implementation completion.
      Verify:
      1. File mobius.rs exists at crates/context-graph-graph/src/hyperbolic/mobius.rs
      2. All 6 methods implemented: new, config, mobius_add, distance, exp_map, log_map
      3. All tests pass: cargo test -p context-graph-graph mobius
      4. No clippy warnings
      5. Doc tests pass
      6. Mathematical properties verified (identity, symmetry, triangle inequality, roundtrip)
      Report any issues found.",
     "sherlock-holmes")
```

Fix ANY issues identified by sherlock-holmes before marking task complete.

## Anti-Patterns to AVOID

Per constitution.yaml:
- **AP-001**: Never use `unwrap()` - use proper error handling
- **AP-003**: No magic numbers - use `self.config.eps`, `self.config.max_norm`
- **AP-009**: Clamp values to avoid NaN/Infinity in distance calculation
- **NO MOCKS**: Tests must use real HyperbolicConfig, real PoincarePoint
- **NO BACKWARDS COMPATIBILITY**: If something breaks, error immediately with clear message

## Notes for Implementer

1. The `HyperbolicConfig.eps` is available at `self.config.eps` (default 1e-7)
2. The `HyperbolicConfig.max_norm` is available at `self.config.max_norm` (default 0.99999)
3. The `HyperbolicConfig.abs_curvature()` method returns `|curvature|`
4. The `PoincarePoint.project()` method ensures points stay inside ball
5. All tests MUST be in `#[cfg(test)]` module inside mobius.rs (co-located per constitution)
6. Dimension is always 64 (hardcoded in PoincarePoint, matches HyperbolicConfig.dim)

---

*Task Version: 2.0.0*
*Last Updated: 2026-01-03*
*Dependencies Verified: M04-T00, M04-T02, M04-T02a, M04-T04 all COMPLETE*
