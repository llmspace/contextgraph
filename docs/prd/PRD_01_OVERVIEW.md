# PRD 01: CaseTrack Overview

## One-Click Legal Document Analysis for Claude Code & Claude Desktop

**Version**: 4.0.0
**Date**: 2026-01-28
**Status**: Draft
**Scope**: Fresh greenfield project build
**Language**: Rust (entire project -- no exceptions)

> **BUILD MANDATE**: CaseTrack is built entirely in Rust. The binary crate, core
> library, MCP server, document processing, embedding engine, storage layer,
> search engine, license validation, CLI, and all tooling are Rust. The only
> non-Rust code is a Python helper script for one-time ONNX model conversion
> (a build-time tool, not shipped to users). There is no JavaScript, TypeScript,
> Python, Go, or C++ in the product. All dependencies are Rust crates. The
> output is a single statically-linked Rust binary with zero runtime dependencies.

> **PROVENANCE MANDATE**: Every piece of information CaseTrack returns MUST trace
> back to its exact source. This is non-negotiable. The provenance chain is:
>
> **Embedding vector → Chunk → Provenance → Source document (file path + filename)**
>
> Every chunk stores: source file path, document filename, page number, paragraph,
> line number, character offsets, extraction method, and timestamps (created_at,
> embedded_at). Every embedding vector is keyed to a chunk_id. Every entity mention,
> citation, and graph edge stores the chunk_id and document_id it came from. Every
> search result, every MCP tool response, every piece of retrieved text includes
> its full provenance. There are ZERO orphaned vectors -- every embedding can be
> traced back to the original document, page, and paragraph it came from.
> **If the provenance chain is broken, the data is useless.**

---

## Document Index

This PRD is split across 10 documents. Each is self-contained but references the others.

| Doc | Title | Covers |
|-----|-------|--------|
| **01 (this)** | Overview | Executive summary, vision, principles, glossary |
| [02](PRD_02_TARGET_USER_HARDWARE.md) | Target User & Hardware | Users, hardware tiers, performance targets |
| [03](PRD_03_DISTRIBUTION_INSTALLATION.md) | Distribution & Installation | Channels, MCPB, manifest, install flows, updates |
| [04](PRD_04_STORAGE_ARCHITECTURE.md) | Storage Architecture | Local storage, RocksDB schema, data versioning |
| [05](PRD_05_EMBEDDER_STACK.md) | Embedder Stack | 7 embedders, ONNX, model management |
| [06](PRD_06_DOCUMENT_INGESTION.md) | Document Ingestion | PDF, DOCX, OCR, chunking |
| [07](PRD_07_CASE_MANAGEMENT.md) | Case Management & Provenance | Case model, isolation, citations |
| [08](PRD_08_SEARCH_RETRIEVAL.md) | Search & Retrieval | 4-stage pipeline, RRF, ranking |
| [09](PRD_09_MCP_TOOLS.md) | MCP Tools | All tool specs, examples, error responses |
| [10](PRD_10_TECHNICAL_BUILD.md) | Technical Build Guide | Bootstrap, crate structure, CI/CD, testing, security |

---

## 1. What is CaseTrack?

CaseTrack is a **one-click installable MCP server** that plugs into **Claude Code** and **Claude Desktop**, giving Claude the ability to ingest, search, and analyze legal documents. Everything runs on the user's machine -- **all embeddings, vectors, and databases are stored locally** on the user's device with zero cloud dependencies.

```
+---------------------------------------------------------------------------+
|  CASETRACK -- "Install once. Everything runs on YOUR machine."            |
+---------------------------------------------------------------------------+
|  - Ingests PDFs, DOCX, scanned images                                    |
|  - Embeds documents with 7 specialized legal embedders                   |
|  - Stores all vectors/embeddings locally (RocksDB)                       |
|  - Provides semantic search with full source citations                   |
|  - MCP server for Claude Code + Claude Desktop                           |
|  - Your data NEVER leaves your computer                                  |
+---------------------------------------------------------------------------+
```

---

## 2. The Problem

Legal professionals waste hours searching through case documents:

- **Keyword search fails**: "breach of duty" won't find "violation of fiduciary obligation"
- **No AI integration**: Can't ask questions about documents in natural language
- **No provenance**: When you find something, you can't cite the exact source
- **Complex tools**: Existing legal tech requires IT departments and training
- **Expensive**: Enterprise legal tech costs $200-500+/seat/month

---

## 3. The Solution

CaseTrack solves this with:

1. **One-click install** -- single command or MCPB file, embedders and database included
2. **100% local** -- all data stored on YOUR device in per-case RocksDB instances (case and customer isolation)
3. **7 specialized embedders** -- semantic search that understands legal language
4. **Full provenance** -- every answer cites source file path, document name, page, paragraph, and line number
5. **2000-char chunks** -- 10% overlap, each chunk stores its exact origin
6. **Claude Code + Desktop** -- works with both CLI and Desktop via MCP stdio
7. **Auto-sync** -- watches folders for changes; optional scheduled reindexing (daily/hourly/custom)
8. **Runs anywhere** -- 8GB MacBook Air, no GPU needed; free tier useful, Pro $29/month

---

## 4. Key Metrics

| Metric | Target |
|--------|--------|
| Install time | < 2 minutes |
| First search after install | < 5 minutes |
| Search latency | < 200ms on any laptop |
| PDF ingestion | < 1 second per page |
| RAM usage | < 2GB peak |
| Model download | ~400MB one-time |

---

## 5. Vision Statement

> **Any attorney can ask Claude questions about their case documents and get accurate, cited answers -- without IT support, cloud accounts, or technical knowledge.**

---

## 6. Design Principles

```
DESIGN PRINCIPLES
=================================================================================

1. ZERO CONFIGURATION
   User downloads file -> double-clicks -> starts using
   No terminal, no config files, no environment variables
   Claude Code: single curl command + one settings.json entry

2. RUNS ON ANYTHING
   8GB RAM laptop from 2020 should work fine
   No GPU required, ever
   Intel, AMD, Apple Silicon all supported

3. PRIVACY FIRST
   Documents never leave the device
   No telemetry, no analytics, no cloud
   Attorney-client privilege preserved
   License validation works offline after first activation

4. INSTANT VALUE
   First useful search within 5 minutes of download
   No training required
   Works like asking a research assistant

5. PROVENANCE ALWAYS (THE MOST IMPORTANT PRINCIPLE)
   Every answer includes exact source citation
   Document name, file path, page, paragraph, line number, character offsets
   Every embedding vector links back to its chunk, which links to its source
   Every entity, citation, and graph edge traces to its source chunk
   Timestamps on everything: when ingested, when embedded, when last synced
   One click to view original context
   If you can't cite the source, you can't return the information

6. GRACEFUL DEGRADATION
   Low RAM? Use fewer models (lazy loading)
   Slow CPU? Longer ingestion, same quality
   Free tier? Fewer features, still useful

7. FAIL LOUDLY
   Errors are specific and actionable
   No silent failures -- every operation reports success or explains failure
   MCP error responses include recovery instructions
```

---

## 7. What CaseTrack is NOT

- **Not a document management system**: Use Dropbox/OneDrive for storage
- **Not a practice management tool**: No billing, calendaring, or client management
- **Not e-discovery software**: Not built for litigation holds or productions
- **Not a cloud service**: Everything runs locally, we never see your data
- **Not an LLM**: CaseTrack provides tools to Claude; it does not generate answers itself

---

## 8. Architecture at a Glance

```
+-----------------------------------------------------------------------+
|                         USER'S MACHINE                                 |
+-----------------------------------------------------------------------+
|                                                                       |
|  +----------------------------+                                       |
|  | Claude Code / Desktop      |                                       |
|  |                            |                                       |
|  |  User asks a question      |                                       |
|  |        |                   |                                       |
|  |        v  MCP (stdio)      |                                       |
|  +--------+-------------------+                                       |
|           |                                                           |
|  +--------v-------------------+                                       |
|  | CaseTrack MCP Server       |   Single Rust binary                  |
|  |  (casetrack binary)        |   No runtime dependencies             |
|  |                            |                                       |
|  |  +----------+  +--------+  |                                       |
|  |  | Document |  | Search |  |                                       |
|  |  | Parser   |  | Engine |  |                                       |
|  |  +----------+  +--------+  |                                       |
|  |  +----------+  +--------+  |                                       |
|  |  | Chunking |  | 7 ONNX |  |                                       |
|  |  | Engine   |  | Models |  |                                       |
|  |  +----------+  +--------+  |                                       |
|  +--------+-------------------+                                       |
|           |                                                           |
|  +--------v-------------------+                                       |
|  | Local Storage              |   ~/Documents/CaseTrack/              |
|  |  +-------+  +-----------+  |                                       |
|  |  |Case A |  | Case B    |  |   Each case = isolated RocksDB       |
|  |  |RocksDB|  | RocksDB   |  |   Vectors, chunks, provenance        |
|  |  +-------+  +-----------+  |                                       |
|  +----------------------------+                                       |
|                                                                       |
|  NOTHING LEAVES THIS MACHINE                                          |
+-----------------------------------------------------------------------+
```

---

## 9. Technology Summary

| Component | Technology | Why |
|-----------|------------|-----|
| Language | Rust | Single binary, no runtime, cross-platform |
| MCP SDK | rmcp | Official Rust MCP SDK, stdio transport |
| Storage | RocksDB | Embedded KV store, zero-config, local disk |
| ML Inference | ONNX Runtime | CPU-optimized, cross-platform, quantized INT8 |
| PDF | pdf-extract + lopdf | Pure Rust |
| DOCX | docx-rs | Pure Rust |
| OCR | Tesseract (bundled) | Best open-source OCR |
| Model Download | hf-hub | Hugging Face model registry |
| Serialization | bincode + serde | Fast binary serialization for vectors |
| Async | tokio | Standard Rust async runtime |
| File watching | notify | Cross-platform OS file notifications (inotify/FSEvents/ReadDirectoryChanges) |
| CLI | clap | Standard Rust CLI parsing |
| Logging | tracing | Structured logging with subscriber |
| License | ed25519-dalek | Offline cryptographic validation |
| Build/Release | cargo-dist | Cross-platform binary distribution |
| CI | GitHub Actions | Multi-platform CI/CD |

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **BM25** | Best Match 25 -- classical keyword ranking algorithm |
| **Chunk** | A 2000-character segment of a document with 10% (200 char) overlap, the unit of search. Every chunk stores full provenance: source file path, document name, page number, paragraph, line number, and character offsets. |
| **ColBERT** | Contextualized Late Interaction over BERT -- token-level reranking |
| **Embedder** | A model that converts text to a numerical vector |
| **MCP** | Model Context Protocol -- standard for AI tool integration |
| **MCPB** | MCP Bundle -- a ZIP file format for distributing MCP servers |
| **ONNX** | Open Neural Network Exchange -- cross-platform ML model format |
| **Provenance** | The exact source location of text: file path, document name, page number, paragraph number, line number, and character offsets. Attached to every chunk and included in every search result and MCP tool response. |
| **RocksDB** | Embedded key-value database by Meta, used for local storage |
| **RRF** | Reciprocal Rank Fusion -- method to combine search rankings |
| **rmcp** | Official Rust MCP SDK |
| **SPLADE** | Sparse Lexical and Expansion Model -- keyword expansion embedder |
| **stdio** | Standard input/output transport for MCP server communication |
| **Case Map** | A per-case summary structure containing parties, key dates, legal issues, top authorities, entity statistics, and document counts. Built incrementally during ingestion. |
| **Citation Network** | The graph of which documents cite which authorities, and how (Cited, Followed, Distinguished, Overruled). Stored in the `citations` column family. |
| **Context Graph** | The graph layer built on top of chunks and embeddings that stores entities, citations, document relationships, chunk similarity edges, and the case map. Enables AI navigation of 1000+ document cases. |
| **Document Graph** | Relationship edges between documents based on shared entities, shared citations, semantic similarity, or explicit references (ResponseTo, Amends, Exhibits). |
| **Entity** | A named thing extracted from document text: person, organization, court, statute, case citation, date, monetary amount, or legal concept. Stored with mentions linking to source chunks. |

---

*CaseTrack PRD v4.0.0 -- Document 1 of 10*
