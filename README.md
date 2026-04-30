<p align="center">
  <h1 align="center">🧠 OPC DeepBrain</h1>
  <p align="center"><strong>Self-learning knowledge base that runs 100% on your machine.</strong></p>
  <p align="center">No cloud. No API keys. No cost. Your data never leaves your computer.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/v/opc-deepbrain?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/dm/opc-deepbrain" alt="Downloads"></a>
  <a href="https://github.com/Deepleaper/opc-deepbrain/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
</p>

---

## Why DeepBrain?

| | Mem0 | RAG | **DeepBrain** |
|---|---|---|---|
| Runs locally | ❌ Cloud | ❌ Needs vector DB | ✅ SQLite only |
| Self-learning | ✅ | ❌ | ✅ |
| Knowledge evolution | ❌ | ❌ | ✅ 6-layer |
| Conflict detection | ❌ | ❌ | ✅ |
| Cost | $$ API fees | $ Infra | **$0** |
| Privacy | Data on their server | Depends | **Never leaves your machine** |

---

## Quick Start

### Prerequisites

```bash
# 1. Install Ollama (local AI model runner)
# macOS / Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# 2. Pull required models
ollama pull qwen2.5:7b          # for knowledge extraction
ollama pull nomic-embed-text    # for semantic search
```

### Install & Use

```bash
# Install
pip install opc-deepbrain

# Initialize
deepbrain init

# Learn something
deepbrain learn "Our company uses Python 3.12 and PostgreSQL"

# Ingest a directory of documents
deepbrain ingest ~/Documents

# Search your knowledge
deepbrain search "what database do we use"

# Watch a directory (auto-ingest on change)
deepbrain watch ~/Notes

# Check stats
deepbrain stats
```

### As a Python Library

```python
from deepbrain import DeepBrain

brain = DeepBrain()

# Learn
brain.learn("FastAPI is our web framework", claim_type="fact")
brain.learn("Never deploy on Fridays", claim_type="constraint")

# Search
results = brain.search("web framework")
for r in results:
    print(f"[{r['claim_type']}] {r['content']}")

# Detect conflicts
conflicts = brain.detect_conflicts(entry_id)

# Evolve (aggregate, deduplicate, decay)
brain.evolve()
```

---

## Core Features

### 📂 Document Ingestion
Scans PDF, Word, Markdown, code files. Incremental — only processes new/modified files.

### 🧠 Structured Extraction
Uses local Ollama to extract typed knowledge entries (fact / inference / preference / constraint) with evidence and confidence scores.

### 🔍 Hybrid Search
Keyword matching + semantic vectors, fused with RRF ranking. Chinese tokenization supported.

### 🔄 Knowledge Evolution
- Cross-document aggregation
- Automatic deduplication
- Time-based confidence decay
- **Conflict detection** — contradicting knowledge flagged automatically

### 📋 Knowledge State
Every entry has:
- `claim_type`: fact / inference / preference / constraint / observation
- `evidence`: source text citation
- `confidence`: 0-1, decays over time
- `valid_from` / `valid_until`: temporal validity

### 👁️ Directory Watch
Real-time monitoring with auto-ingest. Uses watchdog (fast) with polling fallback.

### 🔒 100% Local
- Storage: SQLite (single `.db` file)
- AI: Ollama (runs on your machine)
- Network: **zero** outbound connections

---

## Supported Formats

| Format | Extensions | Extra Dependency |
|--------|-----------|-----------------|
| Markdown | `.md` `.rst` | — |
| Plain text | `.txt` | — |
| PDF | `.pdf` | `pip install opc-deepbrain[pdf]` |
| Word | `.docx` | `pip install opc-deepbrain[docx]` |
| Code | `.py` `.js` `.ts` `.java` `.go` `.rs` `.c` `.cpp` `.rb` | — |
| Data | `.json` `.yaml` `.toml` `.csv` | — |

---

## Configuration

```yaml
# ~/.deepbrain/config.yaml (auto-created by `deepbrain init`)
storage:
  path: ~/.deepbrain/brain.db

ollama:
  base_url: http://localhost:11434
  model: qwen2.5:7b
  embed_model: nomic-embed-text

ingest:
  ignore_patterns:
    - "node_modules/**"
    - ".git/**"
    - "__pycache__/**"
  max_file_size_mb: 50
```

---

## Integration with OPC Agent

DeepBrain is the memory engine inside [OPC Agent](https://github.com/Deepleaper/opc-agent):

```python
from deepbrain import DeepBrain

brain = DeepBrain()
context = brain.search("relevant to user question")
# Inject into AI conversation as context
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| RAM | 4 GB | 8 GB+ |
| Ollama | Any | Latest |
| Disk | 100 MB | 1 GB+ (for models) |

---

## Roadmap

- [x] Document ingestion with smart chunking
- [x] Hybrid search (keyword + semantic)
- [x] Structured extraction (fact/inference/preference/constraint)
- [x] Conflict detection & resolution
- [x] Directory watch with auto-ingest
- [ ] Web UI for knowledge browsing
- [ ] Multi-brain sync
- [ ] Plugin system

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[Apache-2.0](LICENSE) — Use it however you want. Commercial use OK.

---

<p align="center">
  <strong>Part of the <a href="https://github.com/Deepleaper">Deepleaper</a> open source ecosystem</strong><br>
  🧠 <a href="https://github.com/Deepleaper/opc-deepbrain">DeepBrain</a> · 🤖 <a href="https://github.com/Deepleaper/opc-agent">OPC Agent</a> · 🚀 <a href="https://github.com/Deepleaper/leaper-agent">Leaper Agent</a>
</p>

<p align="center">
  ⭐ If DeepBrain helps you, give it a star!
</p>
