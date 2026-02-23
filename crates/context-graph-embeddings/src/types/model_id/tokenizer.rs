//! Tokenizer families for shared tokenization caching.

/// Tokenizer families for shared tokenization caching.
///
/// Models using the same family can share tokenized inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerFamily {
    /// BERT WordPiece tokenization (e5, SPLADE, MiniLM, ColBERT)
    BertWordpiece,
    /// RoBERTa BPE tokenization (KEPLER)
    RobertaBpe,
    /// Custom models with no tokenization
    None,
}
