# PRD 07: Case Management & Provenance

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Case Model

```rust
/// A legal case/matter containing documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub id: Uuid,
    pub name: String,
    pub case_number: Option<String>,
    pub case_type: CaseType,
    pub status: CaseStatus,
    pub created_at: i64,     // Unix timestamp
    pub updated_at: i64,     // Unix timestamp
    pub stats: CaseStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseStats {
    pub document_count: u32,
    pub page_count: u32,
    pub chunk_count: u32,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CaseType {
    Civil,
    Criminal,
    Family,
    Bankruptcy,
    Contract,
    Employment,
    PersonalInjury,
    RealEstate,
    IntellectualProperty,
    Immigration,
    Other,
}

// Derive FromStr via case-insensitive match on variant names. Default: Other.
// Note: "ip" is an alias for IntellectualProperty.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CaseStatus {
    Active,
    Closed,
    Archived,
}
```

---

## 2. Case Registry

Shared RocksDB instance indexing all cases. Key schema: `case:{uuid}` → bincode-serialized `Case`.

```rust
pub struct CaseRegistry {
    db: rocksdb::DB,        // registry.db in data_dir
    data_dir: PathBuf,
    active_case: Option<Uuid>,
}

pub struct CreateCaseParams {
    pub name: String,
    pub case_number: Option<String>,
    pub case_type: Option<CaseType>,
}

impl CaseRegistry {
    /// Opens registry.db from data_dir
    pub fn open(data_dir: &Path) -> Result<Self>;

    /// Creates case dir + originals subdir, initializes CaseHandle DB,
    /// stores in registry, auto-switches active_case to new case
    pub fn create_case(&mut self, params: CreateCaseParams) -> Result<Case>;

    /// Lookup by "case:{id}" key. Error: CaseNotFound
    pub fn get_case(&self, case_id: Uuid) -> Result<Case>;

    /// Prefix scan "case:", returns all cases sorted by updated_at DESC
    pub fn list_cases(&self) -> Result<Vec<Case>>;

    /// Upsert case metadata
    pub fn update_case(&mut self, case: &Case) -> Result<()>;

    /// Deletes registry entry + entire case directory. Clears active_case if matched.
    pub fn delete_case(&mut self, case_id: Uuid) -> Result<()>;

    /// Validates case exists, opens CaseHandle, sets active_case
    pub fn switch_case(&mut self, case_id: Uuid) -> Result<CaseHandle>;

    pub fn active_case_id(&self) -> Option<Uuid>;
    pub fn count_cases(&self) -> Result<u32>;
}
```

---

## 3. Case Handle

Each case has its own `case.db` RocksDB with column families defined in `super::COLUMN_FAMILIES`.

Key schemas:
- Documents CF: `doc:{uuid}` → bincode `DocumentMetadata`
- Chunks CF: `chunk:{uuid}` → bincode `Chunk`
- Chunks CF index: `doc_chunks:{doc_uuid}:{sequence:06}` → chunk UUID string

```rust
/// Handle to an open case database
pub struct CaseHandle {
    pub db: rocksdb::DB,
    pub case_id: Uuid,       // Parsed from case_dir directory name
    pub case_dir: PathBuf,
}

impl CaseHandle {
    /// Create case.db with all column families (DB dropped after init, reopened by open())
    pub fn initialize(case_dir: &Path) -> Result<()>;

    /// Open existing case.db. Error: CaseDbOpenFailed
    pub fn open(case_dir: &Path) -> Result<Self>;

    // --- Document Operations (all use "documents" CF) ---
    pub fn store_document(&self, doc: &DocumentMetadata) -> Result<()>;
    pub fn get_document(&self, doc_id: Uuid) -> Result<DocumentMetadata>;
    /// Prefix scan "doc:", sorted by ingested_at DESC
    pub fn list_documents(&self) -> Result<Vec<DocumentMetadata>>;
    /// Deletes doc metadata + all chunks via doc_chunks index + embeddings + provenance
    pub fn delete_document(&self, doc_id: Uuid) -> Result<()>;

    // --- Chunk Operations (all use "chunks" CF) ---
    /// Stores chunk + doc_chunks index entry (keyed by doc_id + zero-padded sequence)
    pub fn store_chunk(&self, chunk: &Chunk) -> Result<()>;
    pub fn get_chunk(&self, chunk_id: Uuid) -> Result<Chunk>;
}
```

---

## 4. Provenance System (THE MOST IMPORTANT SYSTEM IN CASETRACK)

### 4.1 Provenance Model

```
PROVENANCE IS NON-NEGOTIABLE
=================================================================================

Every piece of information CaseTrack stores or returns MUST trace back to:
  1. The SOURCE FILE (file path + filename on disk)
  2. The exact LOCATION (page, paragraph, line, character offsets)
  3. The EXTRACTION METHOD (Native text, OCR, Hybrid)
  4. TIMESTAMPS (when created, when last embedded)

This applies to:
  - Every text chunk
  - Every embedding vector (linked via chunk_id)
  - Every entity mention (stores chunk_id + char offsets)
  - Every citation record (stores chunk_id + document_id)
  - Every search result (includes full provenance)
  - Every MCP tool response that returns text

If the provenance chain is broken, the data is USELESS.
A search result without a source citation is worthless to an attorney.
```

Every chunk tracks exactly where it came from:

```rust
/// EVERY chunk stores full provenance. This is THE MOST IMPORTANT DATA STRUCTURE
/// in CaseTrack. When the AI returns information, the user must know EXACTLY where
/// it came from -- which document, which file on disk, which page, which paragraph,
/// which line, which character range. Without provenance, the data is useless.
///
/// The Provenance chain: Embedding vector → chunk_id → ChunkData.provenance → source file
/// This chain is NEVER broken. Every embedding, every entity mention, every citation,
/// every search result carries its Provenance. If you can't cite the source, you can't
/// return the information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    // === Source Document (WHERE did this come from?) ===
    /// UUID of the ingested document
    pub document_id: Uuid,
    /// Original filename ("Contract.pdf") -- always stored, never empty
    pub document_name: String,
    /// Full filesystem path where the file was when ingested
    /// ("/Users/sarah/Cases/Smith/Contract.pdf")
    /// Used for: reindexing (re-reads the file), sync (detects changes), display
    pub document_path: Option<PathBuf>,

    // === Location in Document (EXACTLY where in the document?) ===
    /// Page number (1-indexed) -- which page of the PDF/DOCX
    pub page: u32,
    /// First paragraph index included in this chunk (0-indexed within page)
    pub paragraph_start: u32,
    /// Last paragraph index included in this chunk
    pub paragraph_end: u32,
    /// First line number (1-indexed within page)
    pub line_start: u32,
    /// Last line number
    pub line_end: u32,

    // === Character Offsets (for exact highlighting and cursor positioning) ===
    /// Character offset from start of page -- pinpoints exactly where the text starts
    pub char_start: u64,
    /// Character offset end -- pinpoints exactly where the text ends
    pub char_end: u64,

    // === Extraction Metadata (HOW was the text obtained?) ===
    /// How the text was extracted from the original file
    pub extraction_method: ExtractionMethod,
    /// OCR confidence score (0.0-1.0) if extracted via OCR. Lets the AI warn when
    /// text may be unreliable ("This text was OCR'd with 72% confidence").
    pub ocr_confidence: Option<f32>,

    // === Legal Metadata ===
    /// Optional Bates stamp number (for litigation document production)
    pub bates_number: Option<String>,

    // === Chunk Position ===
    /// Sequential position of this chunk within the entire document (0-indexed)
    pub chunk_index: u32,

    // === Timestamps (WHEN was this data created/updated?) ===
    /// When this chunk was first created from the source document (Unix timestamp)
    pub created_at: i64,
    /// When the embedding vectors for this chunk were last computed (Unix timestamp)
    /// Updated on reindex. Lets the system detect stale embeddings.
    pub embedded_at: i64,
}

impl Provenance {
    /// Generate a legal citation string
    pub fn cite(&self) -> String {
        let mut parts = vec![self.document_name.clone()];
        parts.push(format!("p. {}", self.page));

        if self.paragraph_start == self.paragraph_end {
            parts.push(format!("para. {}", self.paragraph_start));
        } else {
            parts.push(format!("paras. {}-{}", self.paragraph_start, self.paragraph_end));
        }

        if self.line_start > 0 {
            parts.push(format!("ll. {}-{}", self.line_start, self.line_end));
        }

        if let Some(bates) = &self.bates_number {
            parts.push(format!("({})", bates));
        }

        parts.join(", ")
    }

    /// Short citation for inline use
    pub fn cite_short(&self) -> String {
        if let Some(bates) = &self.bates_number {
            bates.clone()
        } else {
            format!("{}, p. {}",
                self.document_name.split('.').next().unwrap_or(&self.document_name),
                self.page
            )
        }
    }
}
```

### 4.2 Search Results with Provenance

```rust
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub provenance: Provenance,
    pub citation: String,
    pub citation_short: String,
    pub context_before: Option<String>,
    pub context_after: Option<String>,
}

impl SearchResult {
    pub fn to_mcp_content(&self) -> serde_json::Value {
        json!({
            "text": self.text,
            "score": self.score,
            "citation": self.citation,
            "citation_short": self.citation_short,
            "source": {
                "document": self.provenance.document_name,
                "page": self.provenance.page,
                "paragraph_start": self.provenance.paragraph_start,
                "paragraph_end": self.provenance.paragraph_end,
                "lines": format!("{}-{}", self.provenance.line_start, self.provenance.line_end),
                "bates": self.provenance.bates_number,
                "extraction_method": format!("{:?}", self.provenance.extraction_method),
                "ocr_confidence": self.provenance.ocr_confidence,
            },
            "context": {
                "before": self.context_before,
                "after": self.context_after,
            }
        })
    }
}
```

### 4.3 Context Window

Search results include surrounding chunks for comprehension. Uses the `doc_chunks` index to look up adjacent chunks by `sequence +/- window`.

```rust
impl CaseHandle {
    /// Returns (before_text, after_text) by looking up adjacent chunks
    /// via doc_chunks:{doc_id}:{sequence +/- 1} index keys
    pub fn get_surrounding_context(
        &self,
        chunk: &Chunk,
        window: usize,
    ) -> Result<(Option<String>, Option<String>)>;
}
```

---

## 5. Case Lifecycle

```
CASE LIFECYCLE
=================================================================================

  create_case("Smith v. Jones")
       |
       v
  [ACTIVE] -----> ingest_pdf, ingest_docx, search_case
       |
       |  close_case()          reopen_case()
       v                             |
  [CLOSED] --------> (read-only) ---+
       |
       |  archive_case()
       v
  [ARCHIVED] -----> (read-only, not shown in default list)
       |
       |  delete_case()
       v
  [DELETED] -----> case directory removed from disk

Notes:
  - ACTIVE: Full read/write. Can ingest, search, modify.
  - CLOSED: Read-only. Search works. Cannot ingest new documents.
  - ARCHIVED: Same as closed but hidden from default list_cases.
  - DELETED: Completely removed. Not recoverable.
```

---

## 6. Case Management via MCP Tools -- Operations Guide

This section is the definitive reference for how the AI (Claude) and the user manage cases, documents, embeddings, and databases through MCP tools. **Every operation below is exposed as an MCP tool** (see PRD 09 for full input/output schemas).

### 6.1 Isolation Guarantee

```
CRITICAL: DATA NEVER CROSSES CASE BOUNDARIES
=================================================================================

- Each case = its own RocksDB database on disk (separate files, separate directory)
- Embeddings from Case A are in a DIFFERENT DATABASE FILE than Case B
- Search operates within a SINGLE CASE ONLY -- there is no cross-case search
- Ingestion targets the ACTIVE CASE ONLY -- documents go into exactly one case
- Deleting a case deletes ONLY that case's database, chunks, embeddings, and index
- No shared vector index, no shared embedding store, no shared anything

The AI MUST switch_case before performing ANY operation on a different case.
There is no way to accidentally mix data between cases.
```

### 6.2 Case Lifecycle Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Create a case | `create_case` | Creates a new case directory, initializes an empty RocksDB instance with all column families, registers in the case registry, auto-switches to the new case | New database on disk |
| List all cases | `list_cases` | Lists all cases with status, document count, chunk count, creation date | Read-only |
| Switch active case | `switch_case` | Changes which case all subsequent operations target. Opens that case's RocksDB database. | Changes active DB handle |
| Get case details | `get_case_info` | Shows all documents, total pages, total chunks, storage usage, embedder info | Read-only |
| Delete a case | `delete_case` | **Permanently removes**: case directory, RocksDB database, ALL chunks, ALL embeddings, ALL indexes, ALL provenance records, optionally stored original files. Requires `confirm=true`. Not recoverable. | **Destroys entire database** |

### 6.3 Document Management Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Ingest one file | `ingest_document` | Reads file → extracts text → chunks into 2000-char segments → embeds with all active models → stores in active case's DB | Adds chunks + embeddings to active case |
| Ingest a folder | `ingest_folder` | Recursively walks directory → ingests all supported files → skips already-ingested (SHA256) | Bulk add to active case |
| Sync a folder | `sync_folder` | Compares disk vs DB → ingests new files, reindexes changed files, optionally removes deleted | Add/update/remove in active case |
| List documents | `list_documents` | Lists all documents in active case with page count, chunk count, type | Read-only |
| Get document details | `get_document` | Shows one document's metadata, extraction method, chunk stats | Read-only |
| **Delete a document** | `delete_document` | **Removes from active case**: document metadata, ALL chunks for that document, ALL embeddings for those chunks, ALL provenance records, ALL BM25 index entries. Requires `confirm=true`. | **Destroys document data** |

### 6.4 Embedding & Index Management Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Check index health | `get_index_status` | Per-document report: embedder coverage (4/7 vs 7/7), SHA256 staleness, missing source files | Read-only |
| Reindex one document | `reindex_document` | Deletes ALL old chunks + embeddings → re-reads source file → re-chunks → re-embeds → rebuilds BM25 entries. Option: `reparse=false` keeps chunks, only rebuilds embeddings. | **Replaces** old embeddings with fresh ones |
| Reindex entire case | `reindex_case` | Full rebuild of every document in the case. Option: `skip_unchanged=true` only touches stale documents. Requires `confirm=true`. | **Replaces** all embeddings in case |
| Get chunk provenance | `get_chunk` | Retrieves one chunk with full text and provenance (file, page, paragraph, line, char offsets) | Read-only |
| List document chunks | `get_document_chunks` | Lists all chunks in a document with their provenance | Read-only |
| Get surrounding context | `get_source_context` | Gets the chunks before/after a given chunk for context | Read-only |

### 6.5 Folder Watch & Auto-Sync Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Watch a folder | `watch_folder` | Starts OS-level file monitoring. New/modified/deleted files automatically trigger ingestion/reindex/removal in the target case. | Automatic ongoing changes |
| Stop watching | `unwatch_folder` | Stops auto-sync. Existing case data is untouched. | No data change |
| List watches | `list_watches` | Shows all active watches, their schedule, last sync, health status | Read-only |
| Change schedule | `set_sync_schedule` | Changes how often a watch syncs (on_change, hourly, daily, manual) | No data change |

### 6.6 Typical AI Workflow

```
User: "New case, Smith v. Jones. Docs in ~/Cases/Smith/"

Claude:
  1. create_case("Smith v. Jones", case_type="contract")  → isolated DB, auto-switched
  2. ingest_folder("~/Cases/Smith/", recursive=true)      → chunks + embeds all files
  3. watch_folder("~/Cases/Smith/", schedule="on_change") → auto-sync future changes

User: "Search for termination clauses"
  4. search_case("termination clauses", top_k=5)          → results with full provenance

User: "Switch to Doe case and search witness testimony"
  5. switch_case("Doe v. State")                          → separate DB, Smith inaccessible
  6. search_case("witness testimony")                     → Doe-only results

Key invariant: delete_case/delete_document/reindex always cascade through
chunks → embeddings → provenance → BM25 entries. Original source files on
disk are NEVER removed. See PRD 09 for full tool schemas.
```

---

*CaseTrack PRD v4.0.0 -- Document 7 of 10*
