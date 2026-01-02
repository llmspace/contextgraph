//! Tests for batch encoding functionality.

use super::*;
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS};
use crate::types::FusedEmbedding;
use std::fs;
use tempfile::tempdir;

fn make_test_embedding(hash: u64) -> FusedEmbedding {
    FusedEmbedding::new(
        vec![0.1; FUSED_OUTPUT],
        [0.125; NUM_EXPERTS],
        [0, 1, 2, 3],
        1000,
        hash,
    )
    .expect("test embedding creation")
}

#[test]
fn test_index_header_is_24_bytes() {
    let size = std::mem::size_of::<EmbeddingIndexHeader>();
    println!("BEFORE: Expected size = 24 bytes");
    println!("AFTER: Actual size = {} bytes", size);
    assert_eq!(size, 24);
    println!("PASSED: EmbeddingIndexHeader is exactly 24 bytes");
}

#[test]
fn test_batch_encoder_push() {
    let mut encoder = BatchBinaryEncoder::with_capacity(10);

    println!("BEFORE: encoder.len() = {}", encoder.len());

    for i in 0..5 {
        let embedding = make_test_embedding(i as u64);
        encoder.push(&embedding).expect("push should succeed");
    }

    println!("AFTER: encoder.len() = {}", encoder.len());
    assert_eq!(encoder.len(), 5);
    println!("PASSED: batch encoder accumulates embeddings");
}

#[test]
fn test_batch_encoder_buffer_size() {
    let mut encoder = BatchBinaryEncoder::with_capacity(10);

    let embedding = make_test_embedding(0);
    encoder.push(&embedding).expect("push");

    println!("BEFORE: expected buffer_size = 6244");
    println!("AFTER: actual buffer_size = {}", encoder.buffer_size());
    assert_eq!(encoder.buffer_size(), 6244);
    println!("PASSED: buffer size matches single embedding");
}

#[test]
fn test_batch_encoder_finalize() {
    let mut encoder = BatchBinaryEncoder::with_capacity(10);

    for i in 0..3 {
        let embedding = make_test_embedding(i as u64);
        encoder.push(&embedding).expect("push");
    }

    let (buffer, offsets) = encoder.finalize();

    println!("BEFORE: expected 3 offsets");
    println!("AFTER: got {} offsets", offsets.len());
    println!("AFTER: offsets = {:?}", offsets);
    println!("AFTER: buffer.len() = {}", buffer.len());

    assert_eq!(offsets.len(), 3);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[1], 6244);
    assert_eq!(offsets[2], 12488);
    assert_eq!(buffer.len(), 6244 * 3);
    println!("PASSED: finalize returns correct buffer and offsets");
}

#[test]
fn test_batch_encoder_write_gds_file() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_batch");

    let mut encoder = BatchBinaryEncoder::with_capacity(10);
    for i in 0..3 {
        let embedding = make_test_embedding(i as u64);
        encoder.push(&embedding).expect("push");
    }

    encoder.write_gds_file(&path).expect("write_gds_file");

    // Verify files exist
    let data_path = path.with_extension("cgeb");
    let index_path = path.with_extension("cgei");

    println!("BEFORE: expecting .cgeb and .cgei files");
    println!("AFTER: data file exists = {}", data_path.exists());
    println!("AFTER: index file exists = {}", index_path.exists());

    assert!(data_path.exists());
    assert!(index_path.exists());

    // Verify index file content
    let index_bytes = fs::read(&index_path).expect("read index");
    println!("AFTER: index file size = {} bytes", index_bytes.len());

    // Header (24) + 3 offsets (3 * 8 = 24) = 48 bytes
    assert_eq!(index_bytes.len(), 48);

    // Verify magic
    assert_eq!(&index_bytes[0..4], &INDEX_MAGIC);
    println!("PASSED: write_gds_file creates valid files");
}

#[test]
fn test_batch_encoder_page_alignment() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_aligned");

    let mut encoder = BatchBinaryEncoder::with_capacity(10);
    for i in 0..3 {
        let embedding = make_test_embedding(i as u64);
        encoder.push(&embedding).expect("push");
    }

    encoder.write_gds_file(&path).expect("write_gds_file");

    // Read index to get offsets
    let index_path = path.with_extension("cgei");
    let index_bytes = fs::read(&index_path).expect("read index");

    // Parse offsets (skip 24-byte header)
    let offset_0 = u64::from_be_bytes(index_bytes[24..32].try_into().unwrap());
    let offset_1 = u64::from_be_bytes(index_bytes[32..40].try_into().unwrap());
    let offset_2 = u64::from_be_bytes(index_bytes[40..48].try_into().unwrap());

    println!("BEFORE: expecting page-aligned offsets");
    println!("AFTER: offset_0 = {} (expect 0)", offset_0);
    println!("AFTER: offset_1 = {} (expect 8192)", offset_1);
    println!("AFTER: offset_2 = {} (expect 16384)", offset_2);

    // First embedding at offset 0, second at 8192 (2 pages), third at 16384
    assert_eq!(offset_0, 0);
    assert_eq!(offset_1 % 4096, 0, "offset_1 should be page-aligned");
    assert_eq!(offset_2 % 4096, 0, "offset_2 should be page-aligned");
    println!("PASSED: offsets are 4KB page-aligned");
}

#[test]
fn test_batch_encoder_write_unaligned() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_unaligned");

    let mut encoder = BatchBinaryEncoder::with_capacity(10);
    for i in 0..3 {
        let embedding = make_test_embedding(i as u64);
        encoder.push(&embedding).expect("push");
    }

    encoder.write_unaligned(&path).expect("write_unaligned");

    let data_path = path.with_extension("cgeb");
    let data_bytes = fs::read(&data_path).expect("read data");

    println!("BEFORE: expecting compact data file (no padding)");
    println!("AFTER: data file size = {} bytes", data_bytes.len());

    // 3 embeddings x 6244 bytes = 18732 bytes (no padding)
    assert_eq!(data_bytes.len(), 6244 * 3);
    println!("PASSED: write_unaligned produces compact file");
}

#[test]
fn test_batch_encoder_empty_fails() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_empty");

    let encoder = BatchBinaryEncoder::with_capacity(10);

    let result = encoder.write_gds_file(&path);

    println!("BEFORE: attempting to write empty batch");
    println!("AFTER: result = {:?}", result);

    assert!(result.is_err());
    println!("PASSED: empty batch write fails fast");
}

#[test]
fn test_batch_encoder_is_empty() {
    let encoder = BatchBinaryEncoder::with_capacity(10);

    println!("BEFORE: new encoder");
    println!("AFTER: is_empty() = {}", encoder.is_empty());

    assert!(encoder.is_empty());
    println!("PASSED: new encoder is empty");
}

#[test]
fn test_batch_encoder_magic_in_data_file() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_magic");

    let mut encoder = BatchBinaryEncoder::with_capacity(10);
    encoder.push(&make_test_embedding(42)).expect("push");

    encoder.write_gds_file(&path).expect("write");

    let data_path = path.with_extension("cgeb");
    let data_bytes = fs::read(&data_path).expect("read");

    println!("BEFORE: expecting CGEB magic at start of data file");
    println!("AFTER: first 4 bytes = {:02x?}", &data_bytes[0..4]);

    assert_eq!(&data_bytes[0..4], &crate::storage::binary::EMBEDDING_MAGIC);
    println!("PASSED: data file starts with CGEB magic");
}

#[test]
fn test_batch_encoder_with_aux_data() {
    let encoder = BatchBinaryEncoder::with_aux_data(10);

    println!("BEFORE: created encoder with aux_data support");
    println!("AFTER: encoder.is_empty() = {}", encoder.is_empty());

    assert!(encoder.is_empty());
    println!("PASSED: encoder with aux_data initializes correctly");
}
