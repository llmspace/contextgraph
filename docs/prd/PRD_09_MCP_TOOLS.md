# PRD 09: MCP Tools

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Tool Overview

| Tool | Description | Tier | Requires Active Case |
|------|-------------|------|---------------------|
| `create_case` | Create a new case | Free | No |
| `list_cases` | List all cases | Free | No |
| `switch_case` | Switch active case | Free | No |
| `delete_case` | Delete a case and all its data | Free | No |
| `get_case_info` | Get details about active case | Free | Yes |
| `ingest_document` | Ingest a PDF, DOCX, or image | Free | Yes |
| `ingest_folder` | Ingest all supported files in a folder and subfolders | Free | Yes |
| `sync_folder` | Sync a folder -- ingest new/changed files, optionally remove deleted | Free | Yes |
| `list_documents` | List documents in active case | Free | Yes |
| `get_document` | Get document details and stats | Free | Yes |
| `delete_document` | Remove a document from a case | Free | Yes |
| `search_case` | Search across all documents | Free (limited) | Yes |
| `find_entity` | Find mentions of a legal entity | Pro | Yes |
| `get_chunk` | Get a specific chunk with full provenance | Free | Yes |
| `get_document_chunks` | List all chunks in a document with provenance | Free | Yes |
| `get_source_context` | Get surrounding text for a chunk (context window) | Free | Yes |
| `reindex_document` | Delete old embeddings/indexes for a document and rebuild from scratch | Free | Yes |
| `reindex_case` | Rebuild all embeddings and indexes for the entire active case | Free | Yes |
| `get_index_status` | Show embedding/index health for all documents in active case | Free | Yes |
| `watch_folder` | Start watching a folder for file changes -- auto-sync on change or schedule | Free | Yes |
| `unwatch_folder` | Stop watching a folder | Free | Yes |
| `list_watches` | List all active folder watches and their sync status | Free | No |
| `set_sync_schedule` | Set the auto-sync schedule (on_change, hourly, daily, manual) | Free | Yes |
| `get_status` | Get server status and model info | Free | No |
| | | | |
| **--- Context Graph: Case Overview ---** | | | |
| `get_case_map` | High-level case briefing: parties, key dates, issues, document categories, top entities, top authorities, statistics | Free | Yes |
| `get_case_timeline` | Chronological view of key dates and events extracted from documents | Free | Yes |
| `get_case_statistics` | Document counts, page counts, chunk counts, entity counts, authority counts, embedder coverage | Free | Yes |
| | | | |
| **--- Context Graph: Entity & Citation Search ---** | | | |
| `list_entities` | List all extracted entities in the case, grouped by type (person, org, court, statute, etc.) | Free | Yes |
| `get_entity_mentions` | Get all chunks mentioning a specific entity, with context snippets | Free | Yes |
| `search_entity_relationships` | Find chunks mentioning two or more entities together | Pro | Yes |
| `list_authorities` | List all cited legal authorities (statutes, case law) with citation counts | Free | Yes |
| `get_authority_citations` | Get all chunks citing a specific authority, with treatment (cited, followed, distinguished) | Free | Yes |
| | | | |
| **--- Context Graph: Document Navigation ---** | | | |
| `get_document_structure` | Get headings, sections, and table of contents for a document | Free | Yes |
| `browse_pages` | Get all chunks from a specific page range within a document | Free | Yes |
| `find_related_documents` | Find documents similar to a given document (by shared entities, authorities, or semantic similarity) | Free | Yes |
| `list_documents_by_type` | List documents filtered by type (pleading, discovery, exhibit, etc.) | Free | Yes |
| `traverse_chunks` | Navigate forward/backward through chunks in a document from a starting point | Free | Yes |
| | | | |
| **--- Context Graph: Advanced Search ---** | | | |
| `search_similar_chunks` | Find chunks semantically similar to a given chunk across all documents | Free | Yes |
| `compare_documents` | Compare what two documents say about a topic (side-by-side search) | Pro | Yes |
| `find_document_clusters` | Group documents by theme/topic using semantic clustering | Pro | Yes |

---

## 2. Tool Specifications

> **PROVENANCE IN EVERY RESPONSE**: Every MCP tool that returns text from a document
> MUST include the full provenance chain: source document filename, file path on disk,
> page number, paragraph range, line range, character offsets, extraction method, OCR
> confidence (if applicable), and timestamps (when ingested, when last embedded).
> A tool response that returns document text without telling the user exactly where
> it came from is a **bug**. The AI must always be able to cite its sources.

### Common Error Patterns

All tools return errors in a consistent MCP format. The four common error types:

```json
// NoCaseActive -- returned by any tool that requires an active case
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "No active case. Create or switch to a case first:\n  - create_case: Create a new case\n  - switch_case: Switch to an existing case\n  - list_cases: See all cases"
  }]
}

// FileNotFound -- returned by ingest_document, reindex_document, etc.
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "File not found: /Users/sarah/Downloads/Complaint.pdf\n\nCheck that the path is correct and the file exists."
  }]
}

// FreeTierLimit -- returned when a free tier quota is exceeded
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "Free tier allows 3 cases (you have 3). Delete a case or upgrade to Pro for unlimited cases: https://casetrack.legal/upgrade"
  }]
}

// NotFound -- returned when a case, document, or chunk ID is not found
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "Case not found: \"Smith\". Did you mean:\n  - Smith v. Jones Corp (ID: a1b2c3d4)\nUse the full name or ID."
  }]
}
```

Per-tool error examples are omitted below; all errors follow these patterns.

---

### 2.1 `create_case`

```json
{
  "name": "create_case",
  "description": "Create a new legal case. Creates an isolated database for this case on your machine. Automatically switches to the new case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Case name (e.g., 'Smith v. Jones')"
      },
      "case_number": {
        "type": "string",
        "description": "Optional docket or case number"
      },
      "case_type": {
        "type": "string",
        "enum": ["civil", "criminal", "family", "bankruptcy", "contract", "employment", "personal_injury", "real_estate", "intellectual_property", "immigration", "other"],
        "description": "Type of legal case"
      }
    },
    "required": ["name"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Created case \"Smith v. Jones Corp\" (ID: a1b2c3d4).\nType: Contract\nThis is now your active case.\n\nNext: Ingest documents with ingest_document."
  }]
}
```

---

### 2.2 `list_cases`

```json
{
  "name": "list_cases",
  "description": "List all cases. Shows name, type, status, document count, and which case is active.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "status_filter": {
        "type": "string",
        "enum": ["active", "closed", "archived", "all"],
        "default": "active",
        "description": "Filter by case status"
      }
    }
  }
}
```

---

### 2.3 `switch_case`

```json
{
  "name": "switch_case",
  "description": "Switch to a different case. All subsequent operations (ingest, search) will use this case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "case_name": {
        "type": "string",
        "description": "Case name or ID to switch to"
      }
    },
    "required": ["case_name"]
  }
}
```

---

### 2.4 `delete_case`

```json
{
  "name": "delete_case",
  "description": "Permanently delete a case and all its documents, embeddings, and data. This cannot be undone.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "case_name": {
        "type": "string",
        "description": "Case name or ID to delete"
      },
      "confirm": {
        "type": "boolean",
        "description": "Must be true to confirm deletion",
        "default": false
      }
    },
    "required": ["case_name", "confirm"]
  }
}
```

---

### 2.5 `get_case_info`

```json
{
  "name": "get_case_info",
  "description": "Get detailed information about the active case including document list and storage usage.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.6 `ingest_document`

```json
{
  "name": "ingest_document",
  "description": "Ingest a document (PDF, DOCX, or image) into the active case. Extracts text (with OCR for scans), chunks the text, computes embeddings, and indexes for search. All processing and storage happens locally on your machine.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Absolute path to the file on your computer"
      },
      "document_name": {
        "type": "string",
        "description": "Optional display name (defaults to filename)"
      },
      "document_type": {
        "type": "string",
        "enum": ["pleading", "motion", "brief", "contract", "exhibit", "correspondence", "deposition", "discovery", "statute", "case_law", "other"],
        "description": "Type of legal document"
      },
      "copy_original": {
        "type": "boolean",
        "default": false,
        "description": "Copy the original file into the case folder"
      }
    },
    "required": ["file_path"]
  }
}
```

---

### 2.7 `ingest_folder`

```json
{
  "name": "ingest_folder",
  "description": "Ingest all supported documents in a folder and all subfolders. Walks the entire directory tree recursively. Automatically skips files already ingested (matched by SHA256 hash). Supported formats: PDF, DOCX, DOC, TXT, RTF, JPG, PNG, TIFF. Each file is chunked into 2000-character segments with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to folder containing documents. All subfolders are included automatically."
      },
      "recursive": {
        "type": "boolean",
        "default": true,
        "description": "Include subfolders (default: true). Set to false to only process the top-level folder."
      },
      "skip_existing": {
        "type": "boolean",
        "default": true,
        "description": "Skip files already ingested (matched by SHA256 hash). Set to false to re-ingest everything."
      },
      "document_type": {
        "type": "string",
        "enum": ["pleading", "motion", "brief", "contract", "exhibit", "correspondence", "deposition", "discovery", "statute", "case_law", "other"],
        "description": "Default document type for all files. If omitted, CaseTrack infers from file content."
      },
      "file_extensions": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional filter: only ingest files with these extensions (e.g., [\"pdf\", \"docx\"]). Default: all supported formats."
      }
    },
    "required": ["folder_path"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Folder ingestion complete for Smith v. Jones Corp\n\n  Folder:     ~/Cases/Smith/Documents/\n  Subfolders: 4 (Pleadings/, Discovery/, Exhibits/, Correspondence/)\n  Found:      47 supported files\n  New:        23 (ingested)\n  Skipped:    22 (already ingested, matching SHA256)\n  Failed:     2\n  Duration:   4 minutes 12 seconds\n\n  New documents ingested:\n  - Pleadings/Complaint.pdf (45 pages, 234 chunks)\n  - Pleadings/Answer.pdf (12 pages, 67 chunks)\n  - Discovery/Interrogatories.docx (8 pages, 42 chunks)\n  ... 20 more\n\n  Failures:\n  - Exhibits/corrupted.pdf: PDF parsing error (file may be corrupted)\n  - Exhibits/scan_2019.tiff: OCR failed (image too low resolution)\n\nAll 23 new documents are now searchable."
  }]
}
```

---

### 2.8 `sync_folder`

```json
{
  "name": "sync_folder",
  "description": "Sync a folder with the active case. Compares files on disk against what is already ingested and: (1) ingests new files not yet in the case, (2) re-ingests files that have changed since last ingestion (detected by SHA256 mismatch), (3) optionally removes documents whose source files no longer exist on disk. This is the easiest way to keep a case up to date with a directory of documents -- just point it at the folder and run it whenever files change.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to folder to sync. All subfolders are included."
      },
      "remove_deleted": {
        "type": "boolean",
        "default": false,
        "description": "If true, documents whose source files no longer exist on disk will be removed from the case (chunks + embeddings deleted). Default: false (only add/update, never remove)."
      },
      "document_type": {
        "type": "string",
        "enum": ["pleading", "motion", "brief", "contract", "exhibit", "correspondence", "deposition", "discovery", "statute", "case_law", "other"],
        "description": "Default document type for newly ingested files."
      },
      "dry_run": {
        "type": "boolean",
        "default": false,
        "description": "If true, report what would change without actually ingesting or removing anything. Useful for previewing a sync."
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.9 `list_documents`

```json
{
  "name": "list_documents",
  "description": "List all documents in the active case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sort_by": {
        "type": "string",
        "enum": ["name", "date", "pages", "type"],
        "default": "date",
        "description": "Sort order"
      }
    }
  }
}
```

---

### 2.10 `get_document`

```json
{
  "name": "get_document",
  "description": "Get detailed information about a specific document including page count, extraction method, and chunk statistics.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.11 `delete_document`

```json
{
  "name": "delete_document",
  "description": "Remove a document and all its chunks, embeddings, and index entries from the active case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to delete"
      },
      "confirm": {
        "type": "boolean",
        "default": false,
        "description": "Must be true to confirm deletion"
      }
    },
    "required": ["document_name", "confirm"]
  }
}
```

---

### 2.12 `search_case`

```json
{
  "name": "search_case",
  "description": "Search across all documents in the active case using semantic and keyword search. Returns results with FULL provenance: source document filename, file path, page, paragraph, line numbers, character offsets, extraction method, timestamps. Every result is traceable to its exact source location.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query (e.g., 'What are the termination provisions?')"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 50,
        "description": "Number of results to return"
      },
      "document_filter": {
        "type": "string",
        "description": "Optional: restrict search to a specific document name or ID"
      }
    },
    "required": ["query"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Search: \"early termination clause\"\nCase: Smith v. Jones Corp | 5 documents, 1,051 chunks searched\nTime: 87ms | Tier: Pro (4-stage pipeline)\n\n--- Result 1 (score: 0.94) ---\nContract.pdf, p. 12, para. 8, ll. 1-4\n\n\"Either party may terminate this Agreement upon thirty (30) days written notice to the other party. In the event of material breach, the non-breaching party may terminate immediately upon written notice specifying the breach.\"\n\n--- Result 2 (score: 0.89) ---\nContract.pdf, p. 13, para. 10, ll. 1-6\n\n\"In the event of early termination, the non-breaching party shall be entitled to recover all damages, including but not limited to lost profits, reasonable attorney's fees, and costs of enforcement.\"\n\n--- Result 3 (score: 0.76) ---\nComplaint.pdf, p. 8, para. 22, ll. 3-5\n\n\"Defendant terminated the Agreement without the required thirty days notice, in direct violation of Section 8.1 of the Agreement.\""
  }]
}
```

---

### 2.13 `find_entity`

```json
{
  "name": "find_entity",
  "description": "Find all mentions of a legal entity (person, court, statute, case citation) across documents. Pro tier only. Uses E11-LEGAL entity embedder.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity": {
        "type": "string",
        "description": "Entity to find (e.g., 'Judge Smith', '42 USC 1983', 'Miranda v. Arizona')"
      },
      "entity_type": {
        "type": "string",
        "enum": ["person", "court", "statute", "case_citation", "organization", "any"],
        "default": "any",
        "description": "Type of entity to search for"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entity"]
  }
}
```

---

### 2.14 `reindex_document`

```json
{
  "name": "reindex_document",
  "description": "Rebuild all embeddings, chunks, and search indexes for a single document. Deletes all existing chunks and embeddings for the document, re-extracts text from the original file, re-chunks into 2000-character segments, re-embeds with all active models, and rebuilds the BM25 index. Use this when: (1) a document's source file has been updated on disk, (2) you upgraded to Pro tier and want the document embedded with all 7 models, (3) embeddings seem stale or corrupt, (4) OCR results need refreshing. The original file path stored in provenance is used to re-read the source file.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to reindex"
      },
      "force": {
        "type": "boolean",
        "default": false,
        "description": "If true, reindex even if the source file SHA256 has not changed. Default: only reindex if the file has changed."
      },
      "reparse": {
        "type": "boolean",
        "default": true,
        "description": "If true (default), re-extract text from the source file and re-chunk. If false, keep existing chunks but only rebuild embeddings and indexes (faster, useful after tier upgrade)."
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.15 `reindex_case`

```json
{
  "name": "reindex_case",
  "description": "Rebuild all embeddings, chunks, and search indexes for every document in the active case. This is a full rebuild -- it deletes ALL existing chunks and embeddings, re-reads every source file, re-chunks, re-embeds with all active models, and rebuilds the entire BM25 index. Use this when: (1) upgrading from Free to Pro tier (re-embed everything with 7 models instead of 4), (2) after a CaseTrack update that changes chunking or embedding logic, (3) the case index seems corrupted or stale, (4) you want a clean rebuild. WARNING: This can be slow for large cases (hundreds of documents).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "confirm": {
        "type": "boolean",
        "default": false,
        "description": "Must be true to confirm. This deletes and rebuilds ALL embeddings in the case."
      },
      "reparse": {
        "type": "boolean",
        "default": true,
        "description": "If true (default), re-extract text from source files and re-chunk everything. If false, keep existing chunks but only rebuild embeddings and indexes (faster, useful after tier upgrade)."
      },
      "skip_unchanged": {
        "type": "boolean",
        "default": false,
        "description": "If true, skip documents whose source files have not changed (SHA256 match) and whose embeddings are complete for the current tier. Default: false (rebuild everything)."
      }
    },
    "required": ["confirm"]
  }
}
```

---

### 2.16 `get_index_status`

```json
{
  "name": "get_index_status",
  "description": "Show the embedding and index health status for all documents in the active case. Reports which documents have complete embeddings for the current tier, which need reindexing (source file changed, missing embedder coverage, stale embeddings), and overall case index health. Use this to diagnose issues or decide whether to run reindex_document or reindex_case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_filter": {
        "type": "string",
        "description": "Optional: check a specific document instead of all"
      }
    }
  }
}
```

---

### 2.17 `get_status`

```json
{
  "name": "get_status",
  "description": "Get CaseTrack server status including version, license tier, loaded models, and storage usage.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.18 `get_chunk`

```json
{
  "name": "get_chunk",
  "description": "Get a specific chunk by ID with its full text, provenance (source file, page, paragraph, line, character offsets), and embedding status.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk"
      }
    },
    "required": ["chunk_id"]
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Chunk abc-123 (2000 chars)\n\nText:\n\"Either party may terminate this Agreement upon thirty (30) days written notice to the other party...\"\n\nProvenance:\n  Document:   Contract.pdf\n  File Path:  /Users/sarah/Cases/Smith/Contract.pdf\n  Page:       12\n  Paragraphs: 8-9\n  Lines:      1-14\n  Chars:      2401-4401 (within page)\n  Extraction: Native text\n  Chunk Index: 47 of 234\n\nEmbeddings: E1-Legal, E6-Legal, E7, E8-Legal, E11-Legal, E12"
  }]
}
```

---

### 2.19 `get_document_chunks`

```json
{
  "name": "get_document_chunks",
  "description": "List all chunks in a document with their provenance. Shows where every piece of text came from: page, paragraph, line numbers, and character offsets. Use this to understand how a document was chunked and indexed.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      },
      "page_filter": {
        "type": "integer",
        "description": "Optional: only show chunks from this page number"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.20 `get_source_context`

```json
{
  "name": "get_source_context",
  "description": "Get the surrounding context for a chunk -- the chunks immediately before and after it in the original document. Useful for understanding the full context around a search result.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk to get context for"
      },
      "window": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 5,
        "description": "Number of chunks before and after to include"
      }
    },
    "required": ["chunk_id"]
  }
}
```

---

### 2.21 `watch_folder`

```json
{
  "name": "watch_folder",
  "description": "Start watching a folder for file changes. When files are added, modified, or deleted in the watched folder (or any subfolder), CaseTrack automatically syncs the changes into the active case -- new files are ingested, modified files are reindexed (old chunks/embeddings deleted, new ones created), and optionally deleted files are removed from the case. Uses OS-level file notifications (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows) for instant detection. Also supports scheduled sync as a safety net (daily, hourly, or custom interval). Watch persists across server restarts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to the folder to watch. All subfolders are included."
      },
      "schedule": {
        "type": "string",
        "enum": ["on_change", "hourly", "daily", "every_6h", "every_12h", "manual"],
        "default": "on_change",
        "description": "When to sync: 'on_change' = real-time via OS file notifications (recommended), 'hourly'/'daily'/'every_6h'/'every_12h' = scheduled interval (runs in addition to on_change), 'manual' = only sync when you call sync_folder."
      },
      "auto_remove_deleted": {
        "type": "boolean",
        "default": false,
        "description": "If true, documents whose source files are deleted from disk will be automatically removed from the case (chunks + embeddings deleted). Default: false (only add/update, never auto-remove)."
      },
      "file_extensions": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional filter: only watch files with these extensions (e.g., [\"pdf\", \"docx\"]). Default: all supported formats."
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.22 `unwatch_folder`

```json
{
  "name": "unwatch_folder",
  "description": "Stop watching a folder. Removes the watch but does NOT delete any documents already ingested from that folder. The case data remains intact -- only the automatic sync is stopped.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Path to the folder to stop watching (or watch ID)"
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.23 `list_watches`

```json
{
  "name": "list_watches",
  "description": "List all active folder watches across all cases. Shows the watched folder, which case it syncs to, the schedule, last sync time, and current status.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "case_filter": {
        "type": "string",
        "description": "Optional: only show watches for a specific case name or ID"
      }
    }
  }
}
```

---

### 2.24 `set_sync_schedule`

```json
{
  "name": "set_sync_schedule",
  "description": "Change the sync schedule for an existing folder watch. Controls how often CaseTrack checks for file changes and reindexes.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Path to the watched folder (or watch ID)"
      },
      "schedule": {
        "type": "string",
        "enum": ["on_change", "hourly", "daily", "every_6h", "every_12h", "manual"],
        "description": "New schedule: 'on_change' = real-time OS notifications, 'hourly'/'daily' etc = interval-based, 'manual' = only when you call sync_folder"
      },
      "auto_remove_deleted": {
        "type": "boolean",
        "description": "Optionally update auto-remove behavior"
      }
    },
    "required": ["folder_path", "schedule"]
  }
}
```

---

## 2b. Context Graph Tool Specifications

The context graph tools give the AI structured navigation of the case beyond flat search. They are built on the entity, citation, and document graph data extracted during ingestion (see PRD 04 Section 8).

### 2.25 `get_case_map`

```json
{
  "name": "get_case_map",
  "description": "Get a high-level briefing on the active case. Returns: all identified parties and their roles, key dates and events, key legal issues, document breakdown by category, most-cited legal authorities, most-mentioned entities, and case statistics. This is the FIRST tool the AI should call when starting work on a case -- it provides the structural overview needed to plan search strategy for 1000+ documents.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "CASE MAP: Smith v. Jones Corp (Contract)\n\n  PARTIES:\n    Plaintiff:  Smith Corp (CEO: John Smith)\n    Defendant:  Jones LLC (CEO: Mary Jones)\n    Judge:      Hon. Robert Anderson, U.S. District Court, S.D.N.Y.\n    Attorneys:  Sarah Chen (Plaintiff), Michael Brown (Defendant)\n\n  KEY DATES:\n    2022-01-15  Contract signed (Contract.pdf, p.1)\n    2023-06-01  Alleged breach (Complaint.pdf, p.5)\n    2023-07-01  Complaint filed (Complaint.pdf, p.1)\n    2023-09-15  Answer filed (Answer.pdf, p.1)\n    2024-01-10  Discovery deadline (Scheduling_Order.pdf, p.2)\n    2024-06-15  Trial date (Scheduling_Order.pdf, p.3)\n\n  KEY LEGAL ISSUES:\n    1. Breach of non-compete clause (§4.2) -- 23 documents, 187 chunks\n    2. Damages calculation -- 18 documents, 145 chunks\n    3. Injunctive relief eligibility -- 8 documents, 42 chunks\n    4. Attorney's fees -- 5 documents, 28 chunks\n\n  DOCUMENTS (47 total, 2,341 pages, 12,450 chunks):\n    Pleadings:       5 docs (Complaint, Answer, Motions...)\n    Discovery:      20 docs (Interrogatories, Depositions...)\n    Exhibits:       15 docs (Contracts, Emails, Invoices...)\n    Correspondence:  7 docs (Settlement letters, Notices...)\n\n  TOP AUTHORITIES (most cited):\n    1. 42 U.S.C. § 1983 -- 47 citations across 15 documents\n    2. Fed. R. Civ. P. 12(b)(6) -- 23 citations across 8 documents\n    3. Smith v. Board of Regents, 429 U.S. 438 -- 12 citations across 6 documents\n\n  TOP ENTITIES:\n    Smith Corp -- 892 mentions in 45 documents\n    Jones LLC -- 756 mentions in 42 documents\n    John Smith -- 234 mentions in 28 documents\n    Non-compete agreement -- 187 mentions in 23 documents\n\n  EMBEDDINGS: 7/7 embedders (Pro tier), all 12,450 chunks fully embedded"
  }]
}
```

---

### 2.26 `get_case_timeline`

```json
{
  "name": "get_case_timeline",
  "description": "Get a chronological timeline of key dates and events extracted from documents in the active case. Each event includes the date, description, and source document/chunk provenance. Use this to understand the narrative sequence of the case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_date": {
        "type": "string",
        "description": "Optional: filter events from this date (YYYY-MM-DD)"
      },
      "end_date": {
        "type": "string",
        "description": "Optional: filter events until this date (YYYY-MM-DD)"
      }
    }
  }
}
```

---

### 2.27 `get_case_statistics`

```json
{
  "name": "get_case_statistics",
  "description": "Get detailed statistics about the active case: document counts by type, page/chunk totals, entity and citation counts, embedder coverage, storage usage. Useful for understanding case scope and data quality.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.28 `list_entities`

```json
{
  "name": "list_entities",
  "description": "List all entities extracted from documents in the active case, grouped by type. Shows name, type, mention count, and number of documents mentioning each entity. Entities include: persons, organizations, courts, statutes, case citations, dates, monetary amounts, and legal concepts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "court", "statute", "case_citation", "date", "monetary_amount", "legal_concept", "all"],
        "default": "all",
        "description": "Filter by entity type"
      },
      "sort_by": {
        "type": "string",
        "enum": ["mentions", "documents", "name"],
        "default": "mentions",
        "description": "Sort order"
      },
      "top_k": {
        "type": "integer",
        "default": 50,
        "maximum": 500,
        "description": "Maximum entities to return"
      }
    }
  }
}
```

---

### 2.29 `get_entity_mentions`

```json
{
  "name": "get_entity_mentions",
  "description": "Get all chunks that mention a specific entity, with context snippets showing how the entity is referenced. Uses the entity index built during ingestion. Supports fuzzy matching on entity name.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_name": {
        "type": "string",
        "description": "Name of the entity to find (e.g., 'John Smith', '42 USC 1983', 'breach of contract')"
      },
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "court", "statute", "case_citation", "date", "monetary_amount", "legal_concept", "any"],
        "default": "any"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entity_name"]
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Mentions of \"John Smith\" (person) -- 234 total, showing top 20:\n\n  1. Complaint.pdf, p.2, para.3\n     \"Plaintiff JOHN SMITH, CEO of Smith Corp, brings this action...\"\n\n  2. Deposition_Smith.pdf, p.15, para.8\n     \"Q: Mr. Smith, when did you first learn of the alleged breach?\"\n     \"A: I received a call from our VP on March 10, 2023...\"\n\n  3. Contract.pdf, p.12, para.1 (signature block)\n     \"John Smith, Chief Executive Officer, Smith Corp\"\n\n  ... 17 more mentions"
  }]
}
```

---

### 2.30 `search_entity_relationships`

```json
{
  "name": "search_entity_relationships",
  "description": "Find chunks where two or more entities are mentioned together. Use this to trace relationships (who interacted with whom, what statute applies to which party). Pro tier only.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entities": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2,
        "maxItems": 5,
        "description": "Entity names to find together (e.g., ['Smith Corp', 'Jones LLC'])"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entities"]
  }
}
```

---

### 2.31 `list_authorities`

```json
{
  "name": "list_authorities",
  "description": "List all legal authorities (statutes, case law, rules, regulations) cited in the active case. Shows the authority, type, citation count, and number of citing documents. Use this to understand which legal authorities matter most in the case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "authority_type": {
        "type": "string",
        "enum": ["federal_statute", "state_statute", "federal_rule", "state_rule", "case_law", "regulation", "all"],
        "default": "all"
      },
      "sort_by": {
        "type": "string",
        "enum": ["citations", "documents", "name"],
        "default": "citations"
      },
      "top_k": {
        "type": "integer",
        "default": 50,
        "maximum": 200
      }
    }
  }
}
```

---

### 2.32 `get_authority_citations`

```json
{
  "name": "get_authority_citations",
  "description": "Get all chunks that cite a specific legal authority. Shows the context of each citation and the treatment (cited, followed, distinguished, overruled). Use this to understand how an authority is used throughout the case.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "authority": {
        "type": "string",
        "description": "The legal authority to look up (e.g., '42 USC 1983', 'Miranda v. Arizona')"
      },
      "treatment_filter": {
        "type": "string",
        "enum": ["cited", "followed", "distinguished", "overruled", "discussed", "all"],
        "default": "all"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["authority"]
  }
}
```

---

### 2.33 `get_document_structure`

```json
{
  "name": "get_document_structure",
  "description": "Get the structural outline of a document: headings, sections, numbered clauses, and their page/chunk locations. This gives the AI a table-of-contents view for navigation. Works best with structured documents (contracts, briefs, statutes).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.34 `browse_pages`

```json
{
  "name": "browse_pages",
  "description": "Get all chunks from a specific page range within a document. Use this to read through a section of a document sequentially. Returns chunks in order with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      },
      "start_page": {
        "type": "integer",
        "minimum": 1,
        "description": "First page to read"
      },
      "end_page": {
        "type": "integer",
        "minimum": 1,
        "description": "Last page to read"
      }
    },
    "required": ["document_name", "start_page", "end_page"]
  }
}
```

---

### 2.35 `find_related_documents`

```json
{
  "name": "find_related_documents",
  "description": "Find documents related to a given document. Relationships detected: shared entities, shared legal authorities, semantic similarity (E1 cosine), response chains (complaint->answer->reply), and amendment chains. Returns related documents ranked by relationship strength.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to find relationships for"
      },
      "relationship_type": {
        "type": "string",
        "enum": ["all", "shared_entities", "shared_authorities", "semantic_similar", "response_chain", "amendment_chain"],
        "default": "all"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.36 `list_documents_by_type`

```json
{
  "name": "list_documents_by_type",
  "description": "List all documents in the active case filtered by document type (pleading, discovery, exhibit, etc.). Includes page count, chunk count, and ingestion date.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_type": {
        "type": "string",
        "enum": ["pleading", "motion", "brief", "contract", "exhibit", "correspondence", "deposition", "discovery", "statute", "case_law", "other"],
        "description": "Type to filter by"
      }
    },
    "required": ["document_type"]
  }
}
```

---

### 2.37 `traverse_chunks`

```json
{
  "name": "traverse_chunks",
  "description": "Navigate forward or backward through chunks in a document from a starting point. Use this to read through a document sequentially from any position. Returns N chunks in document order with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_chunk_id": {
        "type": "string",
        "description": "UUID of the starting chunk"
      },
      "direction": {
        "type": "string",
        "enum": ["forward", "backward"],
        "default": "forward",
        "description": "Direction to traverse"
      },
      "count": {
        "type": "integer",
        "default": 5,
        "minimum": 1,
        "maximum": 20,
        "description": "Number of chunks to return"
      }
    },
    "required": ["start_chunk_id"]
  }
}
```

---

### 2.38 `search_similar_chunks`

```json
{
  "name": "search_similar_chunks",
  "description": "Find chunks across all documents that are semantically similar to a given chunk. Uses E1 cosine similarity. Use this to find related passages in other documents (e.g., 'find other places in the case that discuss the same issue as this paragraph').",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk to find similar content for"
      },
      "exclude_same_document": {
        "type": "boolean",
        "default": true,
        "description": "Exclude results from the same document (default: true, for cross-document discovery)"
      },
      "min_similarity": {
        "type": "number",
        "default": 0.6,
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Minimum cosine similarity threshold"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      }
    },
    "required": ["chunk_id"]
  }
}
```

---

### 2.39 `compare_documents`

```json
{
  "name": "compare_documents",
  "description": "Compare what two documents say about a specific topic. Searches both documents independently, then returns side-by-side results showing how each document addresses the topic. Pro tier only. Use this for: contract vs. complaint comparison, deposition A vs. deposition B, any 'what does X say vs. what does Y say' question.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_a": {
        "type": "string",
        "description": "First document name or ID"
      },
      "document_b": {
        "type": "string",
        "description": "Second document name or ID"
      },
      "topic": {
        "type": "string",
        "description": "Topic to compare (e.g., 'termination provisions', 'damages', 'breach of duty')"
      },
      "top_k_per_document": {
        "type": "integer",
        "default": 5,
        "maximum": 20
      }
    },
    "required": ["document_a", "document_b", "topic"]
  }
}
```

---

### 2.40 `find_document_clusters`

```json
{
  "name": "find_document_clusters",
  "description": "Group all documents in the case by theme or topic using semantic clustering. Returns clusters of related documents with a label describing what they share. Pro tier only. Use this to understand the structure of a large case (100+ documents) at a glance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "strategy": {
        "type": "string",
        "enum": ["topical", "entity", "authority", "document_type"],
        "default": "topical",
        "description": "Clustering strategy: 'topical' = semantic similarity, 'entity' = shared parties, 'authority' = shared citations, 'document_type' = by type"
      },
      "max_clusters": {
        "type": "integer",
        "default": 10,
        "maximum": 20
      }
    }
  }
}
```

---

## 3. Background Watch System

The folder watch system runs as background tasks inside the MCP server process using the `notify` crate for cross-platform OS file notifications. Key data structures:

```rust
pub struct WatchManager {
    watches: Arc<RwLock<Vec<ActiveWatch>>>,
    fs_watcher: notify::RecommendedWatcher,
    event_tx: mpsc::Sender<FsEvent>,
}

struct ActiveWatch {
    config: FolderWatch,
    case_handle: Arc<CaseHandle>,
}

enum FsEventKind { Created, Modified, Deleted }
```

Behavior: On startup, `WatchManager::init` restores saved watches from `watches.json`, starts OS watchers, and spawns two background tasks -- an event processor (with 2-second debounce) and a scheduled sync runner (checks every 60 seconds). Events are batched: Created triggers ingest, Modified triggers reindex, Deleted triggers removal (if `auto_remove_deleted` is enabled).

For full implementation details (server initialization, tool registration, error handling), see [PRD 10: Technical Build Guide](PRD_10_TECHNICAL_BUILD.md).

---

## 4. Active Case State

The server maintains an "active case" that all document and search operations target. The server starts with no active case; `create_case` automatically switches to the new case, and `switch_case` explicitly changes it. Tools requiring a case return a `NoCaseActive` error if none is set. The active case persists for the MCP session duration but not across sessions.

---

*CaseTrack PRD v4.0.0 -- Document 9 of 10*
