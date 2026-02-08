# Context Graph

### The World's First Knowledge System That Actually Understands *Why*

---

## The Problem

Every AI memory system on the market today is fundamentally blind.

You store knowledge. You ask a question. You get back documents that *look* similar. But the system has no idea *why* things happen, *how* code actually works, *which* entities are involved, or *what* changed since yesterday. It treats "the server crash caused data loss" and "data loss caused the server crash" as the same thing. It can't tell you who stored a memory, what happened to it since, or whether an AI hallucinated part of it.

Pinecone, Weaviate, ChromaDB, Qdrant, pgvector, Milvus -- they all work the same way: one embedding model, one vector, one perspective. A single opinion about what your knowledge means. If that opinion is wrong for your question, you get the wrong answer. And you'll never know why.

**Context Graph changes the equation entirely.**

---

## What Context Graph Is

Context Graph is a **13-perspective GPU-accelerated knowledge graph** that embeds every piece of content through 13 specialized neural models simultaneously, stores them atomically, and retrieves them through tunable weighted fusion. It combines semantic understanding, causal reasoning, code intelligence, entity knowledge, temporal awareness, noise tolerance, and graph structure into a single unified retrieval system.

It doesn't just store what you know. It understands what it means, why it matters, how it connects, and what happened to it over time -- with a complete, immutable audit trail for every operation.

**57 MCP tools. 13 embedding models. 18 indexes. 54 RocksDB column families. 924 passing tests. Zero compromises.**

---

## Three Capabilities No One Else Has

### 1. An Embedding Model That Reasons

Every vector database treats text as a bag of semantics. The direction of causation -- the single most important relationship in diagnostics, incident response, scientific inquiry, and decision-making -- is invisible.

Context Graph solves this with a live LLM (Hermes-2-Pro-Mistral-7B) that sits *inside* the embedding pipeline. Before content is embedded, the LLM reads it, identifies cause-effect spans, determines causal direction, and injects directional markers into the embedding process itself. The result: **asymmetric dual vectors** -- one for the cause side, one for the effect side.

When you ask "why did production go down?", the system doesn't just find documents mentioning outages. It traverses asymmetric causal vectors to find **root causes**, weighted by how strongly the LLM judged the causal directionality, dampened by 0.8x to reflect the inherent uncertainty of reasoning backwards from effects to causes.

This is not RAG. This is **reasoning-augmented embedding**. The LLM doesn't generate answers from retrieved chunks. It gives the embedding model the ability to understand *why things happen*.

### 2. A Search Engine With 13 Independent Opinions

A single embedding model gives you a single ranking. If it's wrong for your query type, you're stuck.

Context Graph runs **13 specialized models simultaneously** and fuses their opinions through weighted Reciprocal Rank Fusion. Each model is a specialist that sees content from a fundamentally different angle:

| Perspective | What It Sees | What Others Miss |
|-------------|-------------|-----------------|
| **Semantic (E1)** | General meaning and paraphrase | Exact keywords, code structure, causal direction |
| **Causal (E5)** | Why things happen, cause-effect chains | Only possible with LLM-guided asymmetric vectors |
| **Code (E7)** | Function signatures, design patterns, AST structure | 69% better than semantic-only for code queries |
| **Entity (E11)** | Named entities, TransE knowledge relationships | "Diesel" = Rust ORM, not fuel |
| **Graph (E8)** | Dependencies, imports, structural connections | Direction-aware: "A imports B" differs from "B imports A" |
| **Keyword (E6)** | Exact term matches, error codes, identifiers | Preserves what semantic models average away |
| **Noise-Tolerant (E9)** | Typos, misspellings, encoding artifacts | "authetication" still finds "authentication" |
| **Intent (E10)** | Purpose and goals behind text | Same purpose, completely different vocabulary |
| **SPLADE (E13)** | Learned term expansion for high recall | "fast" also finds "quick", "rapid", "efficient" |
| **ColBERT (E12)** | Token-level precision for reranking | Exact phrase-level disambiguation |
| **Temporal (E2-E4)** | Recency, periodic patterns, conversation order | Applied post-retrieval so time never overrides meaning |

**The same knowledge base. 16 built-in weight profiles plus unlimited custom profiles. Change a 13-element weight vector and the system *thinks differently about what's relevant*.**

No reindexing. No separate collections. No pipeline changes.

### 3. Complete Lifecycle Provenance for Every Memory

Every mainstream vector database is a black box. You put embeddings in. You get results out. But you can't answer: Who stored this? What happened to it? Was it merged? Did an AI influence it? Can I undo it?

Context Graph tracks the **complete life story** of every piece of knowledge through 9 dedicated RocksDB column families:

- **Who did it** -- Operator attribution on every operation
- **Where it came from** -- Source type, file path, line range, content hash
- **How it was derived** -- Permanent merge lineage with 30-day undo
- **What AI changed** -- LLM model name, version, quantization, temperature, prompt hash, confidence
- **What triggered it** -- Tool call ID, MCP request ID, hook execution details
- **What happened over time** -- Append-only audit log: creation, merging, boosting, deletion, recovery

When an AI agent creates a memory through a hook, and the LLM influences its embedding, and another agent merges it with two other memories, and a human boosts its importance, and it gets soft-deleted with a reason -- **every single step is recorded**. Immutably. With operator, timestamp, rationale, parameters, and previous state.

No other system provides this level of transparency into how knowledge evolves.

---

## What You Can Build With This

### Incident Response That Finds Root Causes, Not Just Keywords

Create a custom profile: E5 (causal) = 0.40, E7 (code) = 0.30, E11 (entity) = 0.20, E1 (semantic) = 0.10.

Ask: "Why did the authentication service fail at 3am?"

The system uses causal vectors to find root causes, code vectors to find the relevant implementation, entity vectors to identify affected services, and semantic vectors for context. One query, four specialized perspectives, weighted for incident response. The causal search applies 0.8x dampening for abductive reasoning because the system knows that tracing backwards from effects to causes is inherently uncertain.

**What you'd get from Pinecone:** Documents that mention "authentication" and "fail."

### Code Review That Understands Structure, Not Just Text

Use the `code_search` profile to find the review target. Switch to `graph_reasoning` to map its dependency surface. Switch to `causal_reasoning` to find historical cause-effect relationships involving the module.

The same knowledge base serves three different analytical perspectives without reindexing.

**What you'd get from ChromaDB:** Documents with similar words.

### Knowledge Validation With Built-In Fact Checking

Use `fact_checking` (E11 dominant) with TransE scoring:

```
score = -||subject + relation - object||
```

Validate arbitrary knowledge triples: Does Tokio depend on PostgreSQL? Is Diesel a Rust ORM? The answer comes from the geometry of learned entity relationships, not from keyword matching.

Then use `search_cross_embedder_anomalies` to find cases where entity knowledge and semantic understanding disagree. These disagreements surface stale knowledge, incorrect assumptions, or domain-specific jargon that needs clarification.

**What you'd get from Qdrant:** No entity reasoning capability at all.

### Noisy Input That Still Works

Error logs. Stack traces. User typos. OCR artifacts. Use the `typo_tolerant` profile with E9 hyperdimensional computing.

E9 uses 10,000-bit binary hypervectors projected to 1024D dense space. Character-level trigram overlap means "authetication middlware impelementation" still finds what you need -- because the embedding itself is noise-tolerant at the mathematical level. This isn't spell-correction. The geometry handles it natively.

**What you'd get from Milvus:** Query failure.

### Causal Discovery That Runs Automatically

The system doesn't just embed causal content passively. It actively **discovers** causal relationships:

1. Finds candidate memory pairs with high semantic similarity
2. Feeds both to the LLM for causal analysis
3. If confirmed: generates explanation, mechanism type, confidence score
4. Stores as dual-vector CausalEdge with full LLM provenance

Tested: 50 memories analyzed, 40 causal relationships discovered, 0 errors. Each with natural-language explanation, mechanism type, and full model provenance.

---

## Architecture at a Glance

### 13 Embedders, 18 Indexes, One Atomic Write

Every memory is embedded through all 13 models in a single atomic operation. If any embedder fails, nothing is stored. Zero partial embeddings. Zero missing dimensions. Complete coverage guaranteed.

| Component | Count |
|-----------|-------|
| Embedding models | 13 (7 semantic, 3 temporal, 2 relational, 1 structural) |
| HNSW indexes | 15 (11 standard + 4 asymmetric variants) |
| Inverted indexes | 2 (E6 keyword, E13 SPLADE) |
| Specialized indexes | 1 (E12 ColBERT MaxSim) |
| Weight profiles | 16 built-in + unlimited custom |
| MCP tools | 57 |
| RocksDB column families | 54 |
| Provenance column families | 9 (all permanent except consolidation recommendations) |

### 5-Stage Retrieval Pipeline

```
Stage 1  SPLADE recall         ~10,000 candidates    Cast the widest net
Stage 2  Matryoshka128 filter  ~1,000 candidates     Fast approximate scoring
Stage 3  13-embedder WRRF      ~100 candidates       Multi-perspective fusion
Stage 4  ColBERT MaxSim        ~top_k results        Token-level precision
Stage 5  Validation + badges   Final results         Temporal signals applied
```

Or skip the pipeline entirely. E1-only search runs in under 50ms. Multi-space WRRF runs in 50-150ms. Full pipeline in 150-300ms. Choose the right tool for the job.

### Performance

| Operation | Latency |
|-----------|---------|
| Single HNSW search | < 50ms |
| Multi-space weighted fusion (6 embedders) | 50-150ms |
| Full 5-stage pipeline | 150-300ms |
| Query embedding (all 13 spaces) | ~500ms |
| ColBERT reranking (50 candidates) | < 15ms |
| Consensus computation (13 embedders) | < 10ms |

### Storage Efficiency

~46 KB per memory across all 13 embeddings. GPU-accelerated. RocksDB-backed with LSM-tree compaction. Tested with 156+ fingerprints, real data, live GPU inference.

---

## 57 Tools, Complete Coverage

| Category | Tools | Highlights |
|----------|-------|------------|
| **Search** | 15 | Multi-space RRF, causal, code, entity, intent, keyword, graph, noise-tolerant, temporal, adaptive, anomaly detection |
| **Entity Intelligence** | 6 | NER extraction, TransE inference, relationship prediction, knowledge validation, entity graph |
| **Graph Traversal** | 7 | K-NN neighbors, BFS/DFS, causal chains, shortest path, similarity chains |
| **Topic Analysis** | 4 | HDBSCAN clustering, portfolio metrics, stability tracking, divergence alerts |
| **Discovery** | 4 | LLM-powered causal discovery, graph relationship detection, link validation |
| **Provenance** | 4 | Audit trail, merge history, provenance chain, file reconciliation |
| **Embedder Navigation** | 6 | System health, index listing, fingerprint inspection, custom profiles, anomaly detection, adaptive query classification |
| **Memory Operations** | 3 | Atomic store, importance boost, soft-delete with recovery |
| **Session** | 3 | Timeline, context, cross-session comparison |
| **Maintenance** | 5 | Consolidation, merging, causal repair, file watching |

---

## The Competitive Landscape

| Capability | Context Graph | Everyone Else |
|-----------|:---:|:---:|
| Simultaneous multi-model embedding | **13 models** | 1 model |
| LLM reasoning inside embedding pipeline | **Yes** | No |
| Asymmetric directional vectors | **Yes** (E5, E8, E10) | No |
| Weighted fusion across models | **WRRF, 16+ profiles** | None or partial |
| Custom weight profiles at runtime | **Unlimited** | No |
| LLM-powered relationship discovery | **Yes** | No |
| TransE entity knowledge inference | **Yes** | No |
| Noise-tolerant HDC retrieval | **Yes** | No |
| Three-dimensional temporal intelligence | **Yes** | No |
| Cross-embedder blind spot detection | **Yes** | No |
| Adaptive query classification | **Yes** | No |
| Code-aware AST embeddings | **Yes** | No |
| Append-only audit log | **Yes** | No |
| Operator attribution on all operations | **Yes** | No |
| Permanent merge lineage with undo | **Yes** | No |
| LLM influence provenance | **Yes** | No |
| Tool call traceability | **Yes** | No |
| Embedding version tracking per memory | **Yes** | No |
| Soft-delete with 30-day recovery | **Yes** | No |
| Dedicated provenance column families | **9** | 0 |

---

## Who This Is For

**AI Agent Platforms** -- Give your agents memory that actually understands context, causation, and code. Full provenance means you can audit every decision an agent made and trace it back to the knowledge it relied on.

**DevOps & SRE Teams** -- Root cause analysis that finds *causes*, not just correlated keywords. Causal chain traversal. Asymmetric search that distinguishes between what triggered an incident and what the incident triggered.

**Knowledge Management** -- Store organizational knowledge once. Retrieve it through 13 different lenses depending on what you're looking for. Automatic topic detection. Cross-embedder anomaly detection finds blind spots no single model could identify.

**Compliance & Audit** -- Immutable provenance for every piece of knowledge. Who created it, who changed it, what AI influenced it, when it was deleted, and whether it can be recovered. 9 dedicated column families for lifecycle tracking.

**Code Intelligence** -- E7 (Qodo-Embed 1.5B) understands code as code, not prose. 69% improvement over semantic-only search for function signatures. Graph-aware dependency analysis through E8.

**Research & Analysis** -- TransE entity inference validates knowledge triples. Causal discovery finds relationships between concepts automatically. 13 independent perspectives surface insights that any single model would miss.

---

## The Bottom Line

Every other vector database gives you one perspective on your knowledge and hopes it's the right one.

Context Graph gives you thirteen. It lets you tune which perspectives matter for each query. It uses an LLM to understand causation inside the embedding process itself. It tracks the complete lifecycle of every piece of knowledge with immutable provenance. And it exposes all of this through 57 battle-tested MCP tools.

**One knowledge base. Thirteen ways of understanding it. Complete transparency into how it got there.**

This isn't an incremental improvement on vector search. This is a fundamentally different architecture for how AI systems remember, reason about, and account for knowledge.

---

*Context Graph: 13 embedders. 57 tools. 54 column families. 924 tests passing. The knowledge graph that actually knows what it knows -- and can prove it.*
