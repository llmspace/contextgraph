//! Type definitions and constants for the Entity embedding model.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::gpu::BertWeights;
use crate::traits::SingleModelConfig;

pub(crate) use crate::models::pretrained::shared::ModelState;

/// Concrete state type for entity model (BERT weights).
pub(crate) type EntityModelState = ModelState<Box<BertWeights>>;

/// Native dimension for MiniLM entity embeddings (legacy).
/// Note: Production E11 uses KEPLER (768D) via ModelId::Kepler.
/// This legacy Entity model remains at 384D (all-MiniLM-L6-v2).
pub const ENTITY_DIMENSION: usize = 384;

/// Maximum tokens for KEPLER (standard BERT-family limit).
pub const ENTITY_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const ENTITY_LATENCY_BUDGET_MS: u64 = 2;

/// HuggingFace model repository name.
/// Note: This is the deprecated MiniLM model. Production uses ModelId::Kepler with KEPLER 768D.
pub const ENTITY_MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// LEGACY entity embedding model -- use [`KeplerModel`] for production E11 embedding.
///
/// # WARNING: Legacy Model
///
/// This is the **legacy** MiniLM-based entity model producing **384D** vectors
/// (`ModelId::Entity`, `ENTITY_DIMENSION = 384`).
///
/// **Production uses [`KeplerModel`]** (RoBERTa-base + TransE) at **768D**
/// (`ModelId::Kepler`). The two model IDs have **different dimensions**:
///
/// | Model | ModelId | Dimension | Status |
/// |-------|---------|-----------|--------|
/// | `EntityModel` | `ModelId::Entity` | 384 | **LEGACY** (MiniLM-L6-v2) |
/// | `KeplerModel` | `ModelId::Kepler` | 768 | **PRODUCTION** (E11 slot) |
///
/// This model is **only** used when `ModelId::Entity` is explicitly requested.
/// Do NOT confuse `ModelId::Entity` (384D) with `ModelId::Kepler` (768D) -- they
/// are separate model identities with incompatible vector spaces.
///
/// # Entity-Specific Features
///
/// - **encode_entity**: Encodes entity names with optional type context
/// - **encode_relation**: Encodes relation predicates for TransE operations
/// - **transe_score**: Computes TransE triple score
/// - **predict_tail**: Predicts tail entity from head + relation
/// - **predict_relation**: Predicts relation from head and tail
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::EntityModel;
/// use context_graph_embeddings::traits::SingleModelConfig;
/// use context_graph_embeddings::error::EmbeddingResult;
/// use std::path::Path;
///
/// async fn example() -> EmbeddingResult<()> {
///     let model = EntityModel::new(
///         Path::new("models/entity"),
///         SingleModelConfig::default(),
///     )?;
///     model.load().await?;  // Must load before embed
///
///     // Encode an entity with type
///     let entity_text = EntityModel::encode_entity("Alice", Some("PERSON"));
///     // => "[PERSON] Alice"
///     Ok(())
/// }
/// ```
// EMB-M1: NOT deprecated via attribute — too disruptive across the crate.
// The doc comment above provides sufficient warning about legacy status.
pub struct EntityModel {
    /// Model weights and inference engine.
    #[allow(dead_code)]
    pub(crate) model_state: std::sync::RwLock<EntityModelState>,

    /// Path to model weights directory.
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,

    /// Configuration for this model instance.
    #[allow(dead_code)]
    pub(crate) config: SingleModelConfig,

    /// Whether model weights are loaded and ready.
    pub(crate) loaded: AtomicBool,
}

// Implement Send and Sync manually since RwLock is involved
unsafe impl Send for EntityModel {}
unsafe impl Sync for EntityModel {}
