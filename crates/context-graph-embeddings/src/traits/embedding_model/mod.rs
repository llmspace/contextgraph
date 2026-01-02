//! Core trait for embedding model implementations.
//!
//! The `EmbeddingModel` trait defines the contract that all 12 embedding models
//! in the fusion pipeline must implement. Each model (E1-E12) has different
//! input requirements, dimensions, and processing characteristics.
//!
//! # Model Compatibility Matrix
//!
//! | Model | Text | Code | Image | Audio |
//! |-------|------|------|-------|-------|
//! | Semantic (E1) | ✓ | ✓* | ✗ | ✗ |
//! | TemporalRecent (E2) | ✓ | ✓ | ✗ | ✗ |
//! | TemporalPeriodic (E3) | ✓ | ✓ | ✗ | ✗ |
//! | TemporalPositional (E4) | ✓ | ✓ | ✗ | ✗ |
//! | Causal (E5) | ✓ | ✓ | ✗ | ✗ |
//! | Sparse (E6) | ✓ | ✓* | ✗ | ✗ |
//! | Code (E7) | ✓* | ✓ | ✗ | ✗ |
//! | Graph (E8) | ✓ | ✓* | ✗ | ✗ |
//! | HDC (E9) | ✓ | ✓ | ✗ | ✗ |
//! | Multimodal (E10) | ✓ | ✗ | ✓ | ✗ |
//! | Entity (E11) | ✓ | ✓* | ✗ | ✗ |
//! | LateInteraction (E12) | ✓ | ✓* | ✗ | ✗ |
//!
//! *Model can process but is not optimized for this type
//!
//! # Thread Safety
//!
//! The trait requires `Send + Sync` bounds to ensure safe usage in
//! multi-threaded async contexts. All implementations must be thread-safe.
//!
//! # Example Implementation
//!
//! ```rust,ignore
//! use context_graph_embeddings::traits::EmbeddingModel;
//! use context_graph_embeddings::types::{ModelId, ModelEmbedding, ModelInput, InputType};
//! use context_graph_embeddings::error::{EmbeddingError, EmbeddingResult};
//! use async_trait::async_trait;
//!
//! struct SemanticModel {
//!     initialized: bool,
//! }
//!
//! #[async_trait]
//! impl EmbeddingModel for SemanticModel {
//!     fn model_id(&self) -> ModelId {
//!         ModelId::Semantic
//!     }
//!
//!     fn supported_input_types(&self) -> &[InputType] {
//!         &[InputType::Text, InputType::Code]
//!     }
//!
//!     async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
//!         // Implementation...
//!         todo!()
//!     }
//!
//!     fn is_initialized(&self) -> bool {
//!         self.initialized
//!     }
//! }
//! ```

mod trait_def;

#[cfg(test)]
mod tests;

pub use trait_def::EmbeddingModel;
