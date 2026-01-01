//! Core types for the 12-model embedding pipeline.

mod embedding;
mod model_id;
pub mod dimensions;

pub use embedding::ModelEmbedding;
pub use model_id::ModelId;
pub use model_id::TokenizerFamily;
