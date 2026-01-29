# PRD 02: Target User & Hardware

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Target Users

| Segment | Profile | Key Pain Point | Why CaseTrack |
|---------|---------|---------------|---------------|
| **Primary: Solo/Small Firms** (1-10 attorneys) | No IT staff, consumer hardware, 5-50 active matters, already uses Claude | Can't semantically search documents; no budget for enterprise legal tech ($500+/seat) | Works on existing laptop, no setup, $29/mo or free tier |
| **Secondary: Paralegals** | Support 1-5 attorneys, manage case document collections | Manual document review is tedious; need exact citations for attorney review | Batch ingest folders, search returns cited sources, 70%+ review reduction |
| **Tertiary: Law Students** | Studying case law, limited budget, CLI-comfortable | Organizing research, finding cross-document connections, accurate citations | Free tier (3 cases), provenance generates proper citations |

---

## 2. User Personas

| Persona | Role / Practice | Hardware | CaseTrack Use | Key Need |
|---------|----------------|----------|---------------|----------|
| **Sarah** | Solo family law attorney | MacBook Air M1, 8GB | Ingests case docs, asks Claude questions about depositions | Zero setup friction |
| **Mike** | Small firm partner (5 attorneys), commercial litigation | Windows 11, 16GB | Firm license, each attorney searches own cases | Windows support, multi-seat, worth $99/mo |
| **Alex** | Paralegal, personal injury firm | Windows 10, 8GB | Batch ingests case folders, builds searchable databases | Fast ingestion, reliable OCR |

---

## 3. Minimum Hardware Requirements

```
MINIMUM REQUIREMENTS (Must Run)
=================================================================================

CPU:     Any 64-bit processor (2018 or newer recommended)
         - Intel Core i3 or better
         - AMD Ryzen 3 or better
         - Apple M1 or better

RAM:     8GB minimum
         - 16GB recommended for large cases (1000+ pages)

Storage: 5GB available
         - 400MB for embedding models (one-time download)
         - 4.6GB for case data (scales with usage)
         - SSD strongly recommended (HDD works but slower ingestion)

OS:      - macOS 11 (Big Sur) or later
         - Windows 10 (64-bit) or later
         - Ubuntu 20.04 or later (other Linux distros likely work)

GPU:     NOT REQUIRED
         - Optional: Metal (macOS), CUDA (NVIDIA), DirectML (Windows)
         - GPU provides ~2x speedup for ingestion if available
         - Search latency unaffected (small batch sizes)

Network: Required ONLY for:
         - Initial model download (~400MB, one-time)
         - License activation (one-time, then cached offline)
         - Software updates (optional)
         ALL document processing is 100% offline

Prerequisites:
         - Claude Code or Claude Desktop installed
         - No other runtime dependencies (Rust binary is self-contained)
         - Tesseract OCR bundled with binary (no separate install)
```

---

## 4. Performance by Hardware Tier

### 4.1 Ingestion Performance

| Hardware | 50-page PDF | 500-page PDF | OCR (50 scanned pages) |
|----------|-------------|--------------|------------------------|
| **Entry** (M1 Air 8GB) | 45 seconds | 7 minutes | 3 minutes |
| **Mid** (M2 Pro 16GB) | 25 seconds | 4 minutes | 2 minutes |
| **High** (i7 32GB) | 20 seconds | 3 minutes | 90 seconds |
| **With GPU** (RTX 3060) | 10 seconds | 90 seconds | 45 seconds |

### 4.2 Search Performance

| Hardware | Free Tier (2-stage) | Pro Tier (4-stage) | Concurrent Models |
|----------|--------------------|--------------------|-------------------|
| **Entry** (M1 Air 8GB) | 100ms | 200ms | 3 (lazy loaded) |
| **Mid** (M2 Pro 16GB) | 60ms | 120ms | 5 |
| **High** (i7 32GB) | 40ms | 80ms | 7 (all loaded) |
| **With GPU** (RTX 3060) | 20ms | 50ms | 7 (all loaded) |

### 4.3 Memory Usage

| Scenario | RAM Usage |
|----------|-----------|
| Idle (server running, no models loaded) | ~50MB |
| Free tier (3 models loaded) | ~800MB |
| Pro tier (6 models loaded) | ~1.5GB |
| During ingestion (peak) | +300MB above baseline |
| During search (peak) | +100MB above baseline |

---

## 5. Supported Platforms

### 5.1 Build Targets

| Platform | Architecture | Binary Name | Status |
|----------|-------------|-------------|--------|
| macOS | x86_64 (Intel) | `casetrack-darwin-x64` | Supported |
| macOS | aarch64 (Apple Silicon) | `casetrack-darwin-arm64` | Supported |
| Windows | x86_64 | `casetrack-win32-x64.exe` | Supported |
| Linux | x86_64 | `casetrack-linux-x64` | Supported |
| Linux | aarch64 | `casetrack-linux-arm64` | Future |

### 5.2 Platform-Specific Notes

**macOS:**
- CoreML execution provider available for ~2x inference speedup
- Universal binary option (fat binary for Intel + Apple Silicon)
- Code signing required for Gatekeeper (`codesign --sign`)
- Notarization required for distribution outside App Store

**Windows:**
- DirectML execution provider available for GPU acceleration
- Binary should be signed with Authenticode certificate
- Windows Defender may flag unsigned binaries
- Long path support: use `\\?\` prefix or registry setting

**Linux:**
- CPU-only by default; CUDA available if NVIDIA drivers present
- Statically linked against musl for maximum compatibility
- AppImage format as alternative distribution

### 5.3 Claude Integration Compatibility

| Client | Transport | Config Location | Status |
|--------|-----------|-----------------|--------|
| Claude Code (CLI) | stdio | `~/.claude/settings.json` | Primary target |
| Claude Desktop (macOS) | stdio | `~/Library/Application Support/Claude/claude_desktop_config.json` | Supported |
| Claude Desktop (Windows) | stdio | `%APPDATA%\Claude\claude_desktop_config.json` | Supported |
| Claude Desktop (Linux) | stdio | `~/.config/Claude/claude_desktop_config.json` | Supported |

---

## 6. Graceful Degradation Strategy

| Tier | RAM | Models Loaded | Behavior |
|------|-----|---------------|----------|
| **Full** | 16GB+ | All 7 simultaneously | Zero load latency, parallel embedding |
| **Standard** | 8-16GB | Free models always (E1, E6, E7 ~800MB); Pro lazy-loaded | ~200ms first-use penalty; models stay loaded until memory pressure |
| **Constrained** | <8GB | E1 + E13 only (~400MB); others loaded one-at-a-time | Sequential embedding, higher search latency, startup warning |

**Detection**: On startup, check available RAM via `sysinfo` crate. Set tier automatically, log the decision. User override: `--memory-mode=full|standard|constrained`.

---

*CaseTrack PRD v4.0.0 -- Document 2 of 10*
