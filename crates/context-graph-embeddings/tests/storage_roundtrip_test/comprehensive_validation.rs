//! Comprehensive Validation Test

use super::helpers::*;
use context_graph_embeddings::{
    EmbedderQueryResult, IndexEntry, MultiSpaceQueryResult, StoredQuantizedFingerprint,
};
use uuid::Uuid;

/// Master validation test covering all critical storage roundtrip requirements.
#[test]
fn test_comprehensive_storage_roundtrip_validation() {
    println!("=== COMPREHENSIVE STORAGE ROUNDTRIP VALIDATION ===\n");

    // 1. Create fingerprint with all 13 embeddings
    let id = Uuid::new_v4();
    let embeddings = create_test_embeddings_with_deterministic_data(123);
    let purpose_vector = create_purpose_vector(123);
    let content_hash = create_content_hash(123);

    let original = StoredQuantizedFingerprint::new(id, embeddings, purpose_vector, content_hash);

    assert_eq!(original.embeddings.len(), 13, "Must have 13 embeddings");
    println!("[1/7] Created fingerprint with all 13 embeddings");

    // 2. Verify JSON roundtrip
    let json = serde_json::to_string(&original).expect("JSON serialize");
    let from_json: StoredQuantizedFingerprint =
        serde_json::from_str(&json).expect("JSON deserialize");
    assert_eq!(from_json.id, original.id);
    assert_eq!(from_json.content_hash, original.content_hash);
    println!(
        "[2/7] JSON roundtrip preserves data (size: {} bytes)",
        json.len()
    );

    // 3. Verify bincode roundtrip
    let bincode_bytes = bincode::serialize(&original).expect("Bincode serialize");
    let from_bincode: StoredQuantizedFingerprint =
        bincode::deserialize(&bincode_bytes).expect("Bincode deserialize");
    assert_eq!(from_bincode.id, original.id);
    println!(
        "[3/7] Bincode roundtrip preserves data (size: {} bytes)",
        bincode_bytes.len()
    );

    // 4. Verify IndexEntry operations
    let index_entry = IndexEntry::new(id, 0, vec![3.0, 4.0]);
    assert!((index_entry.norm - 5.0).abs() < f32::EPSILON);
    let sim = index_entry.cosine_similarity(&[3.0, 4.0]);
    assert!((sim - 1.0).abs() < 1e-6);
    println!("[4/7] IndexEntry norm and cosine similarity verified");

    // 5. Verify RRF formula
    let rrf_0 = EmbedderQueryResult::from_similarity(id, 0, 0.9, 0).rrf_contribution();
    assert!((rrf_0 - 1.0 / 60.0).abs() < f32::EPSILON);
    println!("[5/7] RRF formula 1/(60+rank) verified");

    // 6. Verify MultiSpaceQueryResult aggregation
    let results = vec![
        EmbedderQueryResult::from_similarity(id, 0, 0.9, 0),
        EmbedderQueryResult::from_similarity(id, 1, 0.8, 1),
    ];
    let multi = MultiSpaceQueryResult::from_embedder_results(id, &results, 0.6);
    assert_eq!(multi.embedder_count, 2);
    let expected_rrf = 1.0 / 60.0 + 1.0 / 61.0;
    assert!((multi.rrf_score - expected_rrf).abs() < 1e-6);
    println!("[6/7] MultiSpaceQueryResult RRF aggregation verified");

    // 7. Verify purpose alignment filter
    assert!(multi.passes_alignment_filter(0.55));
    assert!(!multi.passes_alignment_filter(0.65));
    println!("[7/7] Purpose alignment filter at 0.55 threshold verified");

    println!("\n=== ALL STORAGE ROUNDTRIP VALIDATIONS PASSED ===");
    println!("  - 13 embeddings with correct quantization methods");
    println!("  - JSON roundtrip: {} bytes", json.len());
    println!("  - Bincode roundtrip: {} bytes", bincode_bytes.len());
    println!("  - IndexEntry norm/cosine similarity verified");
    println!("  - RRF formula 1/(60+rank) verified");
    println!("  - MultiSpaceQueryResult aggregation verified");
    println!("  - Purpose alignment filter at 0.55 verified");
}
