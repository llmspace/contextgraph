//! Tests for GDS file reader operations.

use crate::storage::gds::GdsFile;
use crate::storage::BatchBinaryEncoder;
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS};
use crate::types::FusedEmbedding;
use std::path::Path;
use tempfile::tempdir;

fn make_test_embedding(hash: u64) -> FusedEmbedding {
    FusedEmbedding::new(
        vec![0.1 * hash as f32; FUSED_OUTPUT],
        [0.125; NUM_EXPERTS],
        [0, 1, 2, 3],
        hash * 100,
        hash,
    )
    .expect("test embedding creation")
}

fn create_test_gds_files(path: &Path, count: usize) {
    let mut encoder = BatchBinaryEncoder::with_capacity(count);
    for i in 0..count {
        encoder.push(&make_test_embedding(i as u64)).expect("push");
    }
    encoder.write_gds_file(path).expect("write_gds_file");
}

#[test]
fn test_gds_file_open() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_gds");

    create_test_gds_files(&path, 5);

    let gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: created GDS files with 5 embeddings");
    println!("AFTER: gds.len() = {}", gds.len());

    assert_eq!(gds.len(), 5);
    println!("PASSED: GdsFile opens correctly");
}

#[test]
fn test_gds_file_read() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_read");

    create_test_gds_files(&path, 5);

    let mut gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: reading embedding at index 2");

    let embedding = gds.read(2).expect("read");

    println!("AFTER: content_hash = {:#x}", embedding.content_hash);
    println!(
        "AFTER: pipeline_latency_us = {}",
        embedding.pipeline_latency_us
    );

    assert_eq!(embedding.content_hash, 2);
    assert_eq!(embedding.pipeline_latency_us, 200);
    println!("PASSED: GdsFile reads correct embedding");
}

#[test]
fn test_gds_file_read_all() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_read_all");

    create_test_gds_files(&path, 10);

    let mut gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: reading all 10 embeddings");

    for i in 0..10 {
        let embedding = gds.read(i).expect(&format!("read {}", i));
        assert_eq!(
            embedding.content_hash, i as u64,
            "embedding {} has wrong hash",
            i
        );
    }

    println!("AFTER: all 10 embeddings read successfully");
    println!("PASSED: GdsFile reads all embeddings correctly");
}

#[test]
fn test_gds_file_is_empty() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_empty_check");

    create_test_gds_files(&path, 3);

    let gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: file with 3 embeddings");
    println!("AFTER: is_empty() = {}", gds.is_empty());

    assert!(!gds.is_empty());
    println!("PASSED: is_empty returns false for non-empty file");
}

#[test]
fn test_gds_file_get_offset() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_offset");

    create_test_gds_files(&path, 5);

    let gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: getting offsets for 5 embeddings");

    let offset_0 = gds.get_offset(0).expect("offset 0");
    let offset_1 = gds.get_offset(1).expect("offset 1");

    println!("AFTER: offset_0 = {}", offset_0);
    println!("AFTER: offset_1 = {}", offset_1);

    assert_eq!(offset_0, 0);
    // offset_1 should be page-aligned (4096 or 8192)
    assert!(offset_1 % 4096 == 0, "offset_1 should be page-aligned");
    println!("PASSED: get_offset returns correct values");
}

#[test]
fn test_gds_file_read_batch() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_batch_read");

    create_test_gds_files(&path, 10);

    let mut gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: batch reading indices [0, 5, 9]");

    let embeddings = gds.read_batch(&[0, 5, 9]).expect("read_batch");

    println!("AFTER: got {} embeddings", embeddings.len());
    println!(
        "AFTER: hashes = {:?}",
        embeddings
            .iter()
            .map(|e| e.content_hash)
            .collect::<Vec<_>>()
    );

    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].content_hash, 0);
    assert_eq!(embeddings[1].content_hash, 5);
    assert_eq!(embeddings[2].content_hash, 9);
    println!("PASSED: read_batch returns correct embeddings");
}

#[test]
fn test_gds_file_iter() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_iter");

    create_test_gds_files(&path, 5);

    let mut gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: iterating over 5 embeddings");

    let embeddings: Vec<_> = gds.iter().collect::<Result<_, _>>().expect("iter");

    println!("AFTER: collected {} embeddings", embeddings.len());

    assert_eq!(embeddings.len(), 5);
    for (i, e) in embeddings.iter().enumerate() {
        assert_eq!(e.content_hash, i as u64);
    }
    println!("PASSED: iter yields all embeddings in order");
}

#[test]
fn test_gds_file_iter_size_hint() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_iter_size");

    create_test_gds_files(&path, 5);

    let mut gds = GdsFile::open(&path).expect("open");
    let iter = gds.iter();

    println!("BEFORE: iterator size_hint");
    println!("AFTER: size_hint = {:?}", iter.size_hint());

    assert_eq!(iter.size_hint(), (5, Some(5)));
    println!("PASSED: iter size_hint is correct");
}

#[test]
fn test_gds_file_data_file_hash() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_hash");

    create_test_gds_files(&path, 3);

    let gds = GdsFile::open(&path).expect("open");
    let hash = gds.data_file_hash();

    println!("BEFORE: checking data file hash");
    println!("AFTER: data_file_hash = {:#x}", hash);

    // Hash should be non-zero for non-empty data
    assert_ne!(hash, 0);
    println!("PASSED: data_file_hash is non-zero");
}
