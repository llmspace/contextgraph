//! Embedding layer weight loading for BERT models.
//!
//! Loads word, position, token_type embeddings and LayerNorm from safetensors.

use candle_nn::VarBuilder;
use std::path::Path;

use super::config::BertConfig;
use super::error::ModelLoadError;
use super::tensor_utils::get_tensor;
use super::weights::{EmbeddingWeights, PoolerWeights};

/// Load embedding layer weights with optional model prefix.
pub fn load_embeddings(
    vb: &VarBuilder,
    config: &BertConfig,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<EmbeddingWeights, ModelLoadError> {
    let prefix = if model_prefix.is_empty() {
        "embeddings".to_string()
    } else {
        format!("{}embeddings", model_prefix)
    };
    let model_path = model_dir.display().to_string();

    // Word embeddings: [vocab_size, hidden_size]
    let word_embeddings = get_tensor(
        vb,
        &format!("{}.word_embeddings.weight", prefix),
        &[config.vocab_size, config.hidden_size],
        &model_path,
    )?;

    // Position embeddings: [max_position_embeddings, hidden_size]
    let position_embeddings = get_tensor(
        vb,
        &format!("{}.position_embeddings.weight", prefix),
        &[config.max_position_embeddings, config.hidden_size],
        &model_path,
    )?;

    // Token type embeddings: [type_vocab_size, hidden_size]
    let token_type_embeddings = get_tensor(
        vb,
        &format!("{}.token_type_embeddings.weight", prefix),
        &[config.type_vocab_size, config.hidden_size],
        &model_path,
    )?;

    // LayerNorm
    let layer_norm_weight = get_tensor(
        vb,
        &format!("{}.LayerNorm.weight", prefix),
        &[config.hidden_size],
        &model_path,
    )?;
    let layer_norm_bias = get_tensor(
        vb,
        &format!("{}.LayerNorm.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(EmbeddingWeights {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm_weight,
        layer_norm_bias,
    })
}

/// Load pooler weights with optional model prefix.
pub fn load_pooler(
    vb: &VarBuilder,
    config: &BertConfig,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<PoolerWeights, ModelLoadError> {
    let model_path = model_dir.display().to_string();

    let dense_weight = get_tensor(
        vb,
        &format!("{}pooler.dense.weight", model_prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let dense_bias = get_tensor(
        vb,
        &format!("{}pooler.dense.bias", model_prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(PoolerWeights {
        dense_weight,
        dense_bias,
    })
}
