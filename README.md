<div align="center">

# 🧠 OPC DeepBrain

### Self-Evolving Knowledge Engine for AI Agents — 6-Layer Memory That Grows With You

### AI Agent 自进化知识引擎 — 6 层记忆，越用越聪明

[![PyPI version](https://img.shields.io/pypi/v/opc-deepbrain.svg)](https://pypi.org/project/opc-deepbrain/)
[![Downloads](https://img.shields.io/pypi/dm/opc-deepbrain.svg)](https://pypi.org/project/opc-deepbrain/)
[![GitHub stars](https://img.shields.io/github/stars/deepleaper/opc-deepbrain.svg)](https://github.com/deepleaper/opc-deepbrain/stargazers)
[![License](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Dependencies](https://img.shields.io/badge/Dependencies-0-green.svg)](#)

[Website](https://www.deepleaper.com) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [vs Mem0](#-comparison)

</div>

---

## 💡 Why DeepBrain?

Memory solutions like Mem0 store facts. **DeepBrain evolves knowledge.**

| Problem | Mem0 / Others | DeepBrain |
|---------|--------------|-----------|
| Memory model | Flat key-value | 6-layer evolving hierarchy |
| Quality control | None | 4-Gate validation system |
| Knowledge growth | Manual CRUD | Auto-promotion through layers |
| Dependencies | Redis, Qdrant, OpenAI… | **Zero** (stdlib only) |
| Storage | Cloud vectors | SQLite (100% local) |
| Self-awareness | ❌ | ✅ Meta-knowledge layer |

**DeepBrain** is a standalone, embeddable knowledge engine that gives any AI agent **long-term, self-evolving memory** — in 3 lines of code, with zero dependencies.

## ✨ Key Features

- 🏗️ **6-Layer Memory Architecture** — From flash memory to meta-knowledge, just like the human brain
- 🚪 **4-Gate Quality Control** — Every piece of knowledge passes Relevance → Novelty → Consistency → Utility gates
- 📦 **Zero Dependencies** — Pure Python stdlib. No numpy, no torch, no API keys
- 💾 **100% Local** — SQLite storage. Your knowledge never leaves your machine
- 🔌 **Embeddable** — Drop into any Python agent framework in 3 lines
- 🔄 **Auto-Evolution** — Knowledge automatically promotes, consolidates, and archives

## 🚀 Quick Start

```bash
pip install opc-deepbrain
```

```python
from opc_deepbrain import DeepBrain

# Initialize
brain = DeepBrain("./my_brain.db")

# Learn
brain.learn("User prefers concise, technical answers", source="conversation")

# Recall
results = brain.recall("What communication style does the user prefer?")
print(results[0].content)  # → "User prefers concise, technical answers"
```

**That's it.** No API keys. No config. No cloud. 3 lines to persistent, evolving memory.

## 🏗️ 6-Layer Memory Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 5: 🔮 Meta-Knowledge                     │
│  "I know that I know X well, but Y is uncertain"│
├─────────────────────────────────────────────────┤
│  Layer 4: 🗄️ Archived                           │
│  Historical reference, low-access but preserved  │
├─────────────────────────────────────────────────┤
│  Layer 3: 🏗️ Consolidated                       │
│  Cross-session patterns, validated over time     │
├─────────────────────────────────────────────────┤
│  Layer 2: 📚 Long-Term                          │
│  Validated knowledge, frequently accessed        │
├─────────────────────────────────────────────────┤
│  Layer 1: 📝 Short-Term                         │
│  Recent interactions, hours to days              │
├─────────────────────────────────────────────────┤
│  Layer 0: ⚡ Flash Memory                        │
│  Current session buffer, minutes                 │
└─────────────────────────────────────────────────┘
         ↑ Auto-promotion based on relevance,
           frequency, and validation scores
```

### How Knowledge Evolves

1. **Ingestion** → New knowledge enters Layer 0 (Flash)
2. **4-Gate Check** → Relevance, Novelty, Consistency, Utility scoring
3. **Promotion** → High-quality knowledge moves up layers over time
4. **Consolidation** → Related facts merge into coherent understanding
5. **Meta-Learning** → The system learns its own knowledge strengths/gaps

## 🚪 4-Gate Quality Control

Every piece of knowledge must pass through 4 gates:

| Gate | Purpose | Question Asked |
|------|---------|---------------|
| 🎯 **Relevance** | Is this useful? | Does this relate to active contexts? |
| 🆕 **Novelty** | Is this new? | Do we already know this? |
| ✅ **Consistency** | Does this fit? | Does it contradict existing knowledge? |
| 🔧 **Utility** | Is this actionable? | Can this improve future responses? |

## 📖 API Reference

### Core API

```python
from opc_deepbrain import DeepBrain

brain = DeepBrain(db_path="./brain.db")

# Learn — store knowledge
brain.learn(
    content="FastAPI is preferred over Flask for new projects",
    source="architecture-review",
    category="tech-decisions",
    tags=["python", "web", "architecture"]
)

# Recall — retrieve relevant knowledge
results = brain.recall(
    query="Which web framework should we use?",
    top_k=5,
    min_score=0.3
)

# Search — keyword search
results = brain.search("FastAPI", category="tech-decisions")

# Stats — memory statistics
stats = brain.stats()
print(f"Total entries: {stats['total']}")
print(f"By layer: {stats['by_layer']}")

# Evolve — trigger manual evolution cycle
brain.evolve()

# Export / Import
brain.export("backup.json")
brain.load("backup.json")
```

### Embedding in Your Agent

```python
# Works with any agent framework
class MyAgent:
    def __init__(self):
        self.brain = DeepBrain("./agent_brain.db")

    def chat(self, user_message):
        # Recall relevant context
        context = self.brain.recall(user_message, top_k=3)

        # Generate response (your LLM call here)
        response = self.llm.generate(user_message, context=context)

        # Learn from the interaction
        self.brain.learn(
            f"User asked about: {user_message}",
            source="conversation"
        )

        return response
```

## ⚖️ Comparison / 对比

| Feature | **OPC DeepBrain** | Mem0 | ChromaDB | Pinecone |
|---------|:-:|:-:|:-:|:-:|
| Memory Model | 6-layer evolving | Flat store | Vector store | Vector store |
| Quality Control | 4-Gate system | ❌ | ❌ | ❌ |
| Auto-Evolution | ✅ | ❌ | ❌ | ❌ |
| Meta-Knowledge | ✅ | ❌ | ❌ | ❌ |
| Dependencies | **0** | 5+ | 3+ | 2+ |
| Storage | SQLite (local) | Redis + Qdrant | Local/Cloud | Cloud only |
| Cloud Required | ❌ | ⚠️ Optional | ⚠️ Optional | ✅ Yes |
| Pricing | **Free** | Free/Paid | Free/Paid | Paid |
| Self-Evolving | ✅ | ❌ | ❌ | ❌ |
| Python stdlib only | ✅ | ❌ | ❌ | ❌ |

## 🔌 Integrations

DeepBrain powers memory in:
- [OPC Agent](https://github.com/deepleaper/opc-agent) — Local AI agent
- [Leaper Agent](https://github.com/deepleaper/leaper-agent) — Global agent framework
- [Leaper Agent CN](https://github.com/deepleaper/leaper-agent-cn) — China-optimized agent
- **Your project** — `pip install opc-deepbrain` and go

## 📄 License

[BSL-1.1](LICENSE) — see LICENSE for details.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

📧 Contact: [tech@deepleaper.com](mailto:tech@deepleaper.com)

---

<div align="center">

**Built with ❤️ by [Deepleaper Technology / 跃盟科技](https://www.deepleaper.com)**

*Give your AI agent a brain that evolves.*

</div>
