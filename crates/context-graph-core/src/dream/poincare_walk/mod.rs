//! Poincare Ball Math Utilities for Dream Walks
//!
//! Implements hyperbolic geometry operations for REM phase exploration:
//! - Mobius addition for random walk steps
//! - Geodesic distance for blind spot detection
//! - Random direction sampling with temperature
//! - Projection to keep points inside the ball
//!
//! Constitution Reference: docs2/constitution.yaml lines 391-394
//!   - temperature: 2.0
//!   - semantic_leap: 0.7
//!
//! ## Module Structure
//!
//! - [`config`]: Configuration for Poincare ball operations
//! - [`math`]: Core vector math and ball validation
//! - [`mobius`]: Mobius addition and geodesic operations
//! - [`sampling`]: Random direction sampling with temperature

pub mod config;
pub mod math;
pub mod mobius;
pub mod sampling;

// Re-export all public items for backward compatibility
pub use config::PoincareBallConfig;
pub use math::{
    inner_product_64,
    norm_64,
    norm_squared_64,
    project_to_ball,
    validate_in_ball,
};
pub use mobius::{
    direction_toward,
    geodesic_distance,
    mobius_add,
};
pub use sampling::{
    is_far_from_all,
    random_direction,
    sample_direction_with_temperature,
    scale_direction,
    softmax_temperature,
};
