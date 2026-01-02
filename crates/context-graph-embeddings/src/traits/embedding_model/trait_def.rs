//! Core trait definition for embedding models.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};
use async_trait::async_trait;

/// Core trait for embedding model implementations.
///
/// All 12 embedding models in the fusion pipeline must implement this trait.
/// The trait provides a unified async interface for:
/// - Querying model capabilities (ID, supported input types, dimensions)
/// - Generating embeddings from multi-modal inputs
/// - Managing model lifecycle (initialization state)
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` bounds, ensuring implementations
/// can be safely shared across async tasks and threads.
///
/// # Error Handling
///
/// All methods that can fail return `EmbeddingResult<T>`. Implementations
/// must return appropriate error variants:
/// - `EmbeddingError::UnsupportedModality` for incompatible input types
/// - `EmbeddingError::NotInitialized` if model not ready
/// - `EmbeddingError::EmptyInput` for empty content
/// - Other variants as appropriate
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::traits::EmbeddingModel;
/// use context_graph_embeddings::types::{ModelInput, InputType};
///
/// async fn generate_embedding(model: &dyn EmbeddingModel, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
///     // Check if model supports text
///     if !model.supports_input_type(InputType::Text) {
///         return Err("Model doesn't support text".into());
///     }
///
///     // Generate embedding
///     let input = ModelInput::text(text)?;
///     let embedding = model.embed(&input).await?;
///     Ok(embedding.vector)
/// }
/// ```
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Returns the unique identifier for this model.
    ///
    /// Each implementation returns one of the 12 `ModelId` variants (E1-E12).
    /// This ID is used for:
    /// - Routing inputs to appropriate models
    /// - Validating embedding dimensions
    /// - Logging and debugging
    ///
    /// # Returns
    /// The `ModelId` variant identifying this model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = SemanticModel::new();
    /// assert_eq!(model.model_id(), ModelId::Semantic);
    /// ```
    fn model_id(&self) -> ModelId;

    /// Returns the list of input types this model supports.
    ///
    /// Models should only list types they are designed to handle well.
    /// Attempting to embed an unsupported type should return
    /// `EmbeddingError::UnsupportedModality`.
    ///
    /// # Returns
    /// Static slice of supported `InputType` variants.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = MultimodalModel::new();
    /// let types = model.supported_input_types();
    /// assert!(types.contains(&InputType::Text));
    /// assert!(types.contains(&InputType::Image));
    /// ```
    fn supported_input_types(&self) -> &[InputType];

    /// Generate an embedding for the given input.
    ///
    /// This is the core embedding generation method. Implementations must:
    /// 1. Validate the input type is supported
    /// 2. Process the input through the model
    /// 3. Return a properly dimensioned `ModelEmbedding`
    ///
    /// # Arguments
    /// * `input` - The input to embed (text, code, image, or audio)
    ///
    /// # Returns
    /// - `Ok(ModelEmbedding)` with the generated embedding vector
    /// - `Err(EmbeddingError)` on failure
    ///
    /// # Errors
    /// - `EmbeddingError::UnsupportedModality` if input type not supported
    /// - `EmbeddingError::NotInitialized` if model not initialized
    /// - `EmbeddingError::EmptyInput` if input content is empty
    /// - `EmbeddingError::InputTooLong` if input exceeds max tokens
    /// - `EmbeddingError::Timeout` if processing exceeds time budget
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = ModelInput::text("Hello, world!")?;
    /// let embedding = model.embed(&input).await?;
    /// assert_eq!(embedding.model_id, ModelId::Semantic);
    /// assert_eq!(embedding.dimension(), 1024);
    /// ```
    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding>;

    /// Returns whether the model is initialized and ready for inference.
    ///
    /// Models may require initialization (loading weights, warming up GPU)
    /// before they can process inputs. This method allows checking readiness.
    ///
    /// # Returns
    /// - `true` if model is ready for `embed()` calls
    /// - `false` if model needs initialization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if !model.is_initialized() {
    ///     return Err(EmbeddingError::NotInitialized { model_id: model.model_id() });
    /// }
    /// ```
    fn is_initialized(&self) -> bool;

    // =========================================================================
    // Default implementations
    // =========================================================================

    /// Check if this model supports the given input type.
    ///
    /// This is a convenience method that checks if the input type
    /// is in the list returned by `supported_input_types()`.
    ///
    /// # Arguments
    /// * `input_type` - The input type to check
    ///
    /// # Returns
    /// `true` if the model supports this input type
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if model.supports_input_type(InputType::Image) {
    ///     let embedding = model.embed(&image_input).await?;
    /// }
    /// ```
    fn supports_input_type(&self, input_type: InputType) -> bool {
        self.supported_input_types().contains(&input_type)
    }

    /// Returns the native output dimension for this model.
    ///
    /// This is a convenience method that delegates to `ModelId::dimension()`.
    /// Returns the raw model output size before any projection.
    ///
    /// # Returns
    /// The embedding dimension for this model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dim = model.dimension();
    /// assert_eq!(dim, 1024); // For Semantic model
    /// ```
    fn dimension(&self) -> usize {
        self.model_id().dimension()
    }

    /// Returns the projected dimension for FuseMoE input.
    ///
    /// Some models (Sparse, Code, HDC) project their outputs to
    /// different dimensions for the fusion pipeline.
    ///
    /// # Returns
    /// The projected dimension used in concatenation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let proj_dim = model.projected_dimension();
    /// assert_eq!(proj_dim, 1536); // For Sparse model
    /// ```
    fn projected_dimension(&self) -> usize {
        self.model_id().projected_dimension()
    }

    /// Returns the latency budget in milliseconds for this model.
    ///
    /// Each model has a performance target from constitution.yaml.
    /// Implementations should aim to complete within this budget.
    ///
    /// # Returns
    /// Maximum expected latency in milliseconds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let budget = model.latency_budget_ms();
    /// assert_eq!(budget, 5); // Semantic: 5ms
    /// ```
    fn latency_budget_ms(&self) -> u32 {
        self.model_id().latency_budget_ms()
    }

    /// Validate that the input is compatible with this model.
    ///
    /// Checks the input type against supported types and returns
    /// an appropriate error if incompatible.
    ///
    /// # Arguments
    /// * `input` - The input to validate
    ///
    /// # Returns
    /// - `Ok(())` if input is compatible
    /// - `Err(EmbeddingError::UnsupportedModality)` if not supported
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.validate_input(&input)?;
    /// let embedding = model.embed(&input).await?;
    /// ```
    fn validate_input(&self, input: &ModelInput) -> EmbeddingResult<()> {
        let input_type = InputType::from(input);
        if self.supports_input_type(input_type) {
            Ok(())
        } else {
            Err(EmbeddingError::UnsupportedModality {
                model_id: self.model_id(),
                input_type,
            })
        }
    }

    /// Returns the maximum input token count for this model.
    ///
    /// Convenience method delegating to `ModelId::max_tokens()`.
    ///
    /// # Returns
    /// Maximum token count (varies by model: 77-4096, or MAX for custom)
    fn max_tokens(&self) -> usize {
        self.model_id().max_tokens()
    }

    /// Returns whether this model uses pretrained weights.
    ///
    /// Convenience method delegating to `ModelId::is_pretrained()`.
    ///
    /// # Returns
    /// - `true` for models with HuggingFace weights (8 models)
    /// - `false` for custom implementations (Temporal*, HDC)
    fn is_pretrained(&self) -> bool {
        self.model_id().is_pretrained()
    }
}
