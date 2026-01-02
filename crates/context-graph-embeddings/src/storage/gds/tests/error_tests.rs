//! Tests for GDS file error handling.

use crate::storage::gds::{GdsFile, GdsFileError};
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
fn test_gds_file_out_of_bounds() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_oob");

    create_test_gds_files(&path, 5);

    let mut gds = GdsFile::open(&path).expect("open");

    println!("BEFORE: attempting to read index 10 (out of bounds)");

    let result = gds.read(10);

    println!("AFTER: result = {:?}", result);

    match result {
        Err(GdsFileError::IndexOutOfBounds { index, len }) => {
            assert_eq!(index, 10);
            assert_eq!(len, 5);
        }
        _ => panic!("Expected IndexOutOfBounds"),
    }
    println!("PASSED: GdsFile fails fast on out-of-bounds");
}

#[test]
fn test_gds_file_invalid_magic() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_bad_magic");

    // Create valid files first
    create_test_gds_files(&path, 5);

    // Corrupt the index file magic
    let index_path = path.with_extension("cgei");
    let mut bytes = std::fs::read(&index_path).expect("read index");
    bytes[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    std::fs::write(&index_path, bytes).expect("write corrupted");

    println!("BEFORE: corrupted index magic bytes");

    let result = GdsFile::open(&path);

    println!("AFTER: result = {:?}", result);

    assert!(matches!(result, Err(GdsFileError::InvalidIndexMagic)));
    println!("PASSED: GdsFile fails fast on invalid magic");
}

#[test]
fn test_gds_file_missing_data_file() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_missing_data");

    // Create only index file (simulate missing data file)
    create_test_gds_files(&path, 3);
    std::fs::remove_file(path.with_extension("cgeb")).expect("remove data file");

    println!("BEFORE: opening with missing data file");

    let result = GdsFile::open(&path);

    println!("AFTER: result = {:?}", result);

    assert!(matches!(result, Err(GdsFileError::Io(_))));
    println!("PASSED: fails on missing data file");
}

#[test]
fn test_gds_file_missing_index_file() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test_missing_index");

    // Create only data file (simulate missing index file)
    create_test_gds_files(&path, 3);
    std::fs::remove_file(path.with_extension("cgei")).expect("remove index file");

    println!("BEFORE: opening with missing index file");

    let result = GdsFile::open(&path);

    println!("AFTER: result = {:?}", result);

    assert!(matches!(result, Err(GdsFileError::Io(_))));
    println!("PASSED: fails on missing index file");
}
