//! Core types for the 12-model embedding pipeline.

mod concatenated;
pub mod dimensions;
mod embedding;
mod fused;
mod input;
mod model_id;

pub use concatenated::ConcatenatedEmbedding;
pub use embedding::ModelEmbedding;
pub use fused::{AuxiliaryEmbeddingData, FusedEmbedding};
pub use input::{ImageFormat, InputType, ModelInput};
pub use model_id::ModelId;
pub use model_id::TokenizerFamily;
