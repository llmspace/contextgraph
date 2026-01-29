# PRD 08: Search & Retrieval

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. 4-Stage Search Pipeline

```
+-----------------------------------------------------------------------+
|                        4-STAGE SEARCH PIPELINE                         |
+-----------------------------------------------------------------------+
|                                                                       |
|  Query: "What does the contract say about early termination?"         |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  | STAGE 1: BM25 RECALL                                  [<5ms]   |   |
|  |                                                                |   |
|  | - E13 inverted index lookup                                   |   |
|  | - Terms: "contract", "early", "termination"                   |   |
|  | - Fast lexical matching                                       |   |
|  |                                                                |   |
|  | Output: 500 candidate chunks                                  |   |
|  +---------------------------------------------------------------+   |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+   |
|  | STAGE 2: SEMANTIC RANKING                             [<80ms]  |   |
|  |                                                                |   |
|  | - E1-LEGAL: Semantic similarity (384D dense cosine)           |   |
|  | - E6-LEGAL: Keyword expansion (sparse dot product)            |   |
|  | - E7: Structured text similarity (Free) / boost (Pro)         |   |
|  | - Score fusion via Reciprocal Rank Fusion (RRF)               |   |
|  |                                                                |   |
|  | Output: 100 candidates, ranked                                |   |
|  +---------------------------------------------------------------+   |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+   |
|  | STAGE 3: MULTI-SIGNAL BOOST (PRO TIER ONLY)          [<30ms]  |   |
|  |                                                                |   |
|  | - E8-LEGAL: Boost citation similarity                         |   |
|  | - E11-LEGAL: Boost entity matches                             |   |
|  |                                                                |   |
|  | Weights: 0.6 x semantic + 0.2 x structure                    |   |
|  |        + 0.1 x citation + 0.1 x entity                       |   |
|  |                                                                |   |
|  | Output: 50 candidates, re-ranked                              |   |
|  +---------------------------------------------------------------+   |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+   |
|  | STAGE 4: COLBERT RERANK (PRO TIER ONLY)              [<100ms] |   |
|  |                                                                |   |
|  | - E12: Token-level MaxSim scoring                             |   |
|  | - Ensures exact phrase matches rank highest                   |   |
|  | - "early termination" > "termination that was early"          |   |
|  |                                                                |   |
|  | Output: Top K results with provenance                         |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  LATENCY TARGETS                                                      |
|  ----------------                                                     |
|  Free tier (Stages 1-2):  <100ms                                     |
|  Pro tier (Stages 1-4):   <200ms                                     |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## 2. Search Engine Implementation

```rust
pub struct SearchEngine {
    embedder: Arc<EmbeddingEngine>,
    tier: LicenseTier,
}

impl SearchEngine {
    pub fn search(
        &self,
        case: &CaseHandle,
        query: &str,
        top_k: usize,
        document_filter: Option<Uuid>,
    ) -> Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();

        // Stage 1: BM25 recall
        let bm25_candidates = self.bm25_recall(case, query, 500, document_filter)?;

        if bm25_candidates.is_empty() {
            return Ok(vec![]);
        }

        // Stage 2: Semantic ranking
        let query_e1 = self.embedder.embed_query(query, EmbedderId::E1Legal)?;
        let query_e6 = self.embedder.embed_query(query, EmbedderId::E6Legal)?;
        let query_e7 = self.embedder.embed_query(query, EmbedderId::E7)?;

        let mut scored: Vec<(Uuid, f32)> = bm25_candidates
            .iter()
            .map(|chunk_id| {
                let e1_score = self.score_dense(case, "e1", chunk_id, &query_e1)?;
                let e6_score = self.score_sparse(case, "e6", chunk_id, &query_e6)?;
                let e7_score = self.score_dense(case, "e7", chunk_id, &query_e7)?;

                let rrf = rrf_fusion(&[
                    (e1_score, 1.0),   // E1: weight 1.0
                    (e6_score, 0.8),   // E6: weight 0.8
                    (e7_score, 0.6),   // E7: weight 0.6
                ]);

                Ok((*chunk_id, rrf))
            })
            .collect::<Result<Vec<_>>>()?;

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(100);

        // Stage 3: Multi-signal boost (Pro only)
        if self.tier.is_pro() {
            scored = self.multi_signal_boost(case, query, scored)?;
            scored.truncate(50);
        }

        // Stage 4: ColBERT rerank (Pro only)
        if self.tier.is_pro() {
            scored = self.colbert_rerank(case, query, scored)?;
        }

        // Build results with provenance
        let results: Vec<SearchResult> = scored
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| self.build_result(case, chunk_id, score))
            .collect::<Result<Vec<_>>>()?;

        let elapsed = start.elapsed();
        tracing::info!(
            "Search completed: {} results in {}ms (query: '{}')",
            results.len(),
            elapsed.as_millis(),
            query
        );

        Ok(results)
    }

    fn build_result(
        &self,
        case: &CaseHandle,
        chunk_id: Uuid,
        score: f32,
    ) -> Result<SearchResult> {
        let chunk = case.get_chunk(chunk_id)?;
        let (ctx_before, ctx_after) = case.get_surrounding_context(&chunk, 1)?;

        Ok(SearchResult {
            text: chunk.text,
            score,
            provenance: chunk.provenance.clone(),
            citation: chunk.provenance.cite(),
            citation_short: chunk.provenance.cite_short(),
            context_before: ctx_before,
            context_after: ctx_after,
        })
    }
}
```

---

## 3. BM25 Implementation

Standard BM25 with `k1=1.2, b=0.75`. Stored in `bm25_index` column family.

**Key schema**: `term:{token}` → bincode `PostingList`, `stats` → bincode `Bm25Stats`

**Tokenization**: lowercase, split on non-alphanumeric (preserving apostrophes), filter stopwords and single-char tokens.

```rust
pub struct Bm25Index;

impl Bm25Index {
    /// Tokenize query → lookup postings per term → accumulate BM25 scores
    /// per chunk → apply optional document_filter → return top `limit` chunk IDs
    pub fn search(case: &CaseHandle, query: &str, limit: usize,
                  document_filter: Option<Uuid>) -> Result<Vec<Uuid>>;

    /// Tokenize chunk text → upsert PostingList per term → update Bm25Stats
    pub fn index_chunk(case: &CaseHandle, chunk: &Chunk) -> Result<()>;
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Bm25Stats {
    pub total_docs: u32,
    pub total_tokens: u64,
    pub avg_doc_length: f32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PostingList {
    pub doc_freq: u32,
    pub entries: Vec<PostingEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PostingEntry {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub term_freq: u32,
    pub doc_length: u32,
}
```

---

## 4. Reciprocal Rank Fusion (RRF)

```rust
/// Combine scores from multiple embedders using RRF
/// Each (score, weight) pair represents one embedder's score and its importance
pub fn rrf_fusion(scored_weights: &[(f32, f32)]) -> f32 {
    const K: f32 = 60.0;

    scored_weights
        .iter()
        .map(|(score, weight)| {
            if *score <= 0.0 {
                0.0
            } else {
                // Convert similarity score to rank-like value, then apply RRF
                weight / (K + (1.0 / score))
            }
        })
        .sum()
}

/// Alternative: weighted combination for Stage 3 multi-signal boost
pub fn weighted_combination(
    base_score: f32,
    signals: &[(f32, f32)],  // (score, weight) pairs
) -> f32 {
    let total_weight: f32 = signals.iter().map(|(_, w)| w).sum();
    let weighted_sum: f32 = signals.iter().map(|(s, w)| s * w).sum();
    base_score * 0.6 + (weighted_sum / total_weight) * 0.4
}
```

---

## 5. ColBERT Reranking (Stage 4)

```rust
impl SearchEngine {
    fn colbert_rerank(
        &self,
        case: &CaseHandle,
        query: &str,
        candidates: Vec<(Uuid, f32)>,
    ) -> Result<Vec<(Uuid, f32)>> {
        // Embed query at token level
        let query_tokens = self.embedder.embed_query(query, EmbedderId::E12)?;
        let query_vecs = match query_tokens {
            QueryEmbedding::Token(t) => t,
            _ => unreachable!(),
        };

        let mut reranked: Vec<(Uuid, f32)> = candidates
            .into_iter()
            .map(|(chunk_id, base_score)| {
                // Load chunk's token embeddings
                let chunk_tokens = self.load_token_embeddings(case, &chunk_id)?;

                // MaxSim: for each query token, find max similarity to any chunk token
                let maxsim_score = query_vecs.vectors.iter()
                    .map(|q_vec| {
                        chunk_tokens.vectors.iter()
                            .map(|c_vec| cosine_similarity(q_vec, c_vec))
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum::<f32>() / query_vecs.vectors.len() as f32;

                // Blend ColBERT score with previous ranking
                let final_score = base_score * 0.4 + maxsim_score * 0.6;
                Ok((chunk_id, final_score))
            })
            .collect::<Result<Vec<_>>>()?;

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(reranked)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenEmbeddings {
    pub vectors: Vec<Vec<f32>>,  // One 64D vector per token
    pub token_count: usize,
}
```

---

## 6. Search Response Format (Canonical)

This is the canonical MCP response format for `search_case` (also referenced by PRD 09).
Document-scoped search uses the same pipeline via the `document_filter` parameter on `SearchEngine::search`.

```json
{
  "query": "early termination clause",
  "case": "Smith v. Jones Corp",
  "results_count": 5,
  "search_time_ms": 87,
  "tier": "pro",
  "stages_used": ["bm25", "semantic", "multi_signal", "colbert"],
  "results": [
    {
      "text": "Either party may terminate this Agreement upon thirty (30) days written notice...",
      "score": 0.94,
      "citation": "Contract.pdf, p. 12, para. 8",
      "citation_short": "Contract, p. 12",
      "source": {
        "document": "Contract.pdf",
        "document_path": "/Users/sarah/Documents/CaseTrack/cases/smith-v-jones/originals/Contract.pdf",
        "document_id": "abc-123",
        "chunk_id": "chunk-456",
        "chunk_index": 14,
        "page": 12,
        "paragraph_start": 8,
        "paragraph_end": 8,
        "line_start": 1,
        "line_end": 4,
        "char_start": 24580,
        "char_end": 26580,
        "bates": null,
        "extraction_method": "Native",
        "ocr_confidence": null,
        "chunk_created_at": "2026-01-15T14:30:00Z",
        "chunk_embedded_at": "2026-01-15T14:30:12Z",
        "document_ingested_at": "2026-01-15T14:29:48Z"
      },
      "context": {
        "before": "...the previous paragraph text...",
        "after": "...the next paragraph text..."
      }
    }
  ]
}
```

---

*CaseTrack PRD v4.0.0 -- Document 8 of 10*
