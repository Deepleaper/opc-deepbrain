<div align="center">

# 🧠 OPC DeepBrain

**Embeddable Personal Knowledge Base — Self-Learning, 100% Local, in <2000 Lines**

[![PyPI](https://img.shields.io/pypi/v/opc-deepbrain?color=%2334D058&label=PyPI)](https://pypi.org/project/opc-deepbrain/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://pypi.org/project/opc-deepbrain/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Deepleaper/opc-deepbrain?style=social)](https://github.com/Deepleaper/opc-deepbrain)

</div>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#6-layer-architecture">Architecture</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#use-cases">Use Cases</a>
</p>

---

> **OPC DeepBrain** is a standalone knowledge engine that gives any AI agent long-term, evolving memory. SQLite-based, zero cloud dependency, zero heavy dependencies — just Python stdlib. Plug it into your existing agent framework in 3 lines of code.
>
> **OPC DeepBrain** 是一个**个人自学习知识库**。本地 SQLite 存储，零云端依赖，你的数据永远在你的电脑上。3 行代码让任何 AI 拥有持久记忆。

## Why DeepBrain?

| | Vector DB (Pinecone, Chroma) | Key-Value Memory (Mem0) | **OPC DeepBrain** |
|---|---|---|---|
| **Memory evolution** | ❌ Static embeddings | ✅ Basic | ✅ **6-layer (L0→L5)** |
| **Knowledge governance** | ❌ | ❌ | ✅ Conflict/decay/merge |
| **Local-first** | ❌ Cloud | Partial | ✅ **SQLite, 100% local** |
| **Dependencies** | Heavy | Moderate | ✅ **Zero** (stdlib only) |
| **Codebase** | Large | Medium | ✅ **<2,000 lines** |
| **Embeddable** | SDK | SDK | ✅ **3 lines** |

## Quick Start

```bash
pip install opc-deepbrain
```

```python
from opc_deepbrain import DeepBrain

brain = DeepBrain("./knowledge.db")

# Learn — information enters L0, evolves upward automatically
brain.learn("User prefers concise answers", namespace="preferences")
brain.learn("Project deadline is March 2026", namespace="project")

# Recall — searches across all layers
results = brain.recall("what does the user prefer", top_k=5)
for r in results:
    print(r.content, r.score)
```

That's it. No config files, no API keys, no Docker.

## 6-Layer Architecture

```
┌──────────────────────────────────────────────────┐
│  L5  Consistency Guard   回归测试 · 异常检测       │
├──────────────────────────────────────────────────┤
│  L4  User Profiling      多维特征 · 偏好建模       │
├──────────────────────────────────────────────────┤
│  L3  Knowledge Gov.      冲突 · 淘汰 · 晋升       │
├──────────────────────────────────────────────────┤
│  L2  Skill Synthesis     跨会话聚类 · 合并 · 去重  │
├──────────────────────────────────────────────────┤
│  L1  Structured Extract  4Gate 门控过滤            │
├──────────────────────────────────────────────────┤
│  L0  Raw Storage         原始输入                  │
├──────────────────────────────────────────────────┤
│  💾  SQLite              本地 · 零依赖 · 可控      │
└──────────────────────────────────────────────────┘
```

**How it works:**
1. **L0**: Raw data goes in (conversations, documents, any text)
2. **L1**: 4Gate filter decides what's worth keeping (novelty / actionability / durability / relevance)
3. **L2**: Cross-session clustering merges related knowledge
4. **L3**: Governance layer resolves conflicts, decays stale info, promotes proven knowledge
5. **L4**: Builds multi-dimensional user/context profiles
6. **L5**: Consistency guardian runs regression checks, detects anomalies

## Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **6-Layer Evolution** | Knowledge automatically matures from raw input to governed, tested wisdom |
| 💾 **SQLite Storage** | Single file, zero setup, fully portable |
| 🔍 **Hybrid Search** | Keyword search built-in; semantic search with optional embeddings |
| 📊 **Knowledge Visualization** | Export to HTML, browse in any browser |
| 🏗️ **Namespace Isolation** | Multiple agents share one DB, knowledge stays separated |
| ⚡ **Minimal Footprint** | 1,794 lines of Python. No numpy, no torch, no heavy deps |
| 🔌 **Embeddable** | Standard Python API — works with LangChain, CrewAI, custom agents, anything |

## API Reference

```python
brain = DeepBrain("path/to/db.sqlite")

# Core operations
brain.learn(content, namespace="default")       # Store & begin evolution
brain.recall(query, top_k=5, namespace=None)    # Search across layers
brain.forget(content_id)                        # Remove specific knowledge
brain.evolve()                                  # Trigger manual evolution cycle

# Visualization
brain.export_html("output/")                    # Browse knowledge in browser

# Namespace management
brain.namespaces()                              # List all namespaces
brain.stats()                                   # Knowledge base statistics
```

## Use Cases

**Chatbots** — Give your bot persistent memory across sessions. It remembers user preferences, past conversations, and learned patterns.

**Knowledge Management** — Build a personal/team knowledge base that self-organizes. No manual tagging required.

**AI Assistants** — Embed in personal assistants so they improve with use. Calendar preferences, communication style, domain knowledge — all learned automatically.

**Customer Service** — Agents that remember client history, common issues, and successful resolutions.

**Any "AI that gets smarter"** scenario — If your project needs an AI that improves over time, DeepBrain is the memory layer.

## System Requirements

- Python >= 3.10
- No additional dependencies (embedding support is optional)
- Works on Linux / macOS / Windows

## Project Stats

| Metric | Value |
|--------|-------|
| Version | **v0.3.1** |
| Source code | **1,794** lines Python |
| Tests | **211** lines |
| Dependencies | **0** (stdlib only) |

## License

[MIT](LICENSE) — use it anywhere, commercial or otherwise.

## Related Projects

> **DeepBrain 技术落地顺序：** Leaper Agent CN（首发）→ Leaper Agent Global → OPC DeepBrain
>
> OPC DeepBrain 定位是**个人自学习知识库**——纯本地、零成本、数据完全私有。适合个人用户和隐私敏感场景。

| Project | Description |
|---------|-------------|
| [leaper-agent](https://github.com/Deepleaper/leaper-agent) | 🚀 Full AI Agent framework with DeepBrain built in |
| [leaper-agent-cn](https://github.com/Deepleaper/leaper-agent-cn) | 🇨🇳 中国版 Agent 框架，内置 DeepBrain |

---

<div align="center">
  <a href="https://github.com/Deepleaper"><strong>Deepleaper 跃盟开源</strong></a><br>
  <sub>Memory is what makes AI intelligent. 记忆让 AI 真正聪明。</sub>
</div>

<div align="center">
  ⭐ If your AI should remember, star this repo.
</div>
