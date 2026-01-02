//! Auxiliary data tests for AuxiliaryEmbeddingData.

#[cfg(test)]
mod tests {
    use crate::types::dimensions::{COLBERT_V3_DIM, FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};
    use crate::types::fused::{AuxiliaryEmbeddingData, FusedEmbedding};
    use crate::types::ModelId;

    fn make_valid_vector() -> Vec<f32> {
        vec![0.1; FUSED_OUTPUT]
    }

    fn make_valid_weights() -> [f32; 8] {
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25]
    }

    fn make_valid_selected() -> [u8; TOP_K_EXPERTS] {
        [0, 1, 4, 5]
    }

    fn make_valid_fused() -> FusedEmbedding {
        FusedEmbedding::new(
            make_valid_vector(),
            make_valid_weights(),
            make_valid_selected(),
            1000,
            12345,
        )
        .expect("Test helper should create valid embedding")
    }

    fn make_token_vector() -> Vec<f32> {
        vec![0.5; COLBERT_V3_DIM]
    }

    // ========== Auxiliary Data Tests ==========

    #[test]
    fn test_auxiliary_data_new_validates_128d_tokens() {
        let token_vecs = vec![make_token_vector(), make_token_vector()];

        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);

        assert_eq!(aux.num_tokens, 2);
        assert_eq!(aux.token_vectors.len(), 2);
        assert_eq!(aux.token_vectors[0].len(), COLBERT_V3_DIM);
    }

    #[test]
    #[should_panic(expected = "dimension")]
    fn test_auxiliary_data_new_panics_on_wrong_dimension() {
        let bad_vec = vec![0.5; 64];
        let token_vecs = vec![bad_vec];

        let _aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);
    }

    #[test]
    fn test_auxiliary_to_blob_from_blob_round_trip() {
        let token_vecs = vec![make_token_vector(); 5];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);

        let blob = aux.to_blob();
        let reconstructed =
            AuxiliaryEmbeddingData::from_blob(&blob).expect("Should deserialize");

        assert_eq!(aux.num_tokens, reconstructed.num_tokens);
        assert_eq!(aux.token_vectors, reconstructed.token_vectors);
    }

    #[test]
    fn test_compress_aux_data_creates_blob() {
        let token_vecs = vec![make_token_vector(); 3];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);
        let mut emb = make_valid_fused().with_aux_data(aux);

        assert!(emb.aux_data.as_ref().unwrap().blob.is_none());

        emb.compress_aux_data().expect("Should compress");

        assert!(emb.aux_data.as_ref().unwrap().blob.is_some());
    }

    #[test]
    fn test_decompress_aux_data_restores_token_vectors() {
        let original_vecs = vec![make_token_vector(); 4];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, original_vecs.clone());
        let mut emb = make_valid_fused().with_aux_data(aux);

        emb.compress_aux_data().expect("Should compress");

        emb.aux_data.as_mut().unwrap().token_vectors.clear();
        emb.aux_data.as_mut().unwrap().num_tokens = 0;

        emb.decompress_aux_data().expect("Should decompress");

        assert_eq!(emb.aux_data.as_ref().unwrap().num_tokens, 4);
        assert_eq!(
            emb.aux_data.as_ref().unwrap().token_vectors,
            original_vecs
        );
    }

    #[test]
    fn test_auxiliary_memory_size_returns_correct_byte_count() {
        let token_vecs = vec![make_token_vector(); 10];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);

        let expected_size = 10 * COLBERT_V3_DIM * 4;
        let actual_size = aux.memory_size();

        assert_eq!(actual_size, expected_size);
    }

    #[test]
    fn test_edge_case_aux_data_empty_tokens() {
        let empty_tokens: Vec<Vec<f32>> = vec![];

        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, empty_tokens);

        assert_eq!(aux.num_tokens, 0);
        assert_eq!(aux.memory_size(), 0);

        let blob = aux.to_blob();
        assert_eq!(blob.len(), 5);
    }

    #[test]
    fn test_edge_case_expert_weights_boundary_0_99() {
        let mut emb = make_valid_fused();
        let weights_low = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.29];
        emb.expert_weights = weights_low;

        let result = emb.validate();

        assert!(result.is_ok(), "0.99 should be within tolerance");
    }

    #[test]
    fn test_edge_case_expert_weights_boundary_1_01() {
        let mut emb = make_valid_fused();
        let weights_high = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.31];
        emb.expert_weights = weights_high;

        let result = emb.validate();

        assert!(result.is_ok(), "1.01 should be within tolerance");
    }

    // ========== Constants Validation Tests ==========

    #[test]
    fn test_constants_match_dimensions_rs() {
        assert_eq!(FusedEmbedding::DIMENSION, FUSED_OUTPUT);
        assert_eq!(FusedEmbedding::DIMENSION, 1536);
        assert_eq!(FusedEmbedding::NUM_EXPERTS, NUM_EXPERTS);
        assert_eq!(FusedEmbedding::NUM_EXPERTS, 8);
        assert_eq!(FusedEmbedding::TOP_K, TOP_K_EXPERTS);
        assert_eq!(FusedEmbedding::TOP_K, 4);
        assert_eq!(AuxiliaryEmbeddingData::TOKEN_DIM, COLBERT_V3_DIM);
        assert_eq!(AuxiliaryEmbeddingData::TOKEN_DIM, 128);
    }
}
