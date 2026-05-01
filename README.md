<p align="center">
  <h1 align="center">🧠 OPC DeepBrain</h1>
  <p align="center">
    <strong>本地运行的自学习知识库，六层记忆进化引擎。</strong><br>
    <em>The self-learning knowledge base that runs on your machine. 6-layer memory evolution engine.</em>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/v/opc-deepbrain?color=%2334D058&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/dm/opc-deepbrain" alt="Downloads"></a>
  <a href="https://github.com/Deepleaper/opc-deepbrain"><img src="https://img.shields.io/github/stars/Deepleaper/opc-deepbrain?style=social" alt="Stars"></a>
  <a href="https://github.com/Deepleaper/opc-deepbrain/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
</p>

<p align="center">
  <a href="#性能对比">性能对比</a> ·
  <a href="#30-秒上手">30 秒上手</a> ·
  <a href="#六层引擎">六层引擎</a> ·
  <a href="#python-api">Python API</a> ·
  <a href="./docs/README_EN.md">English</a>
</p>

---

## 性能对比

### DeepBrain vs Mem0 vs RAG

|  | **DeepBrain** | Mem0 | 传统 RAG |
|---|---|---|---|
| **部署** | `pip install` + Ollama | 云端 API / 自建 Qdrant+Redis | 自建向量库+Embedding服务 |
| **隐私** | 数据永不离机 | 数据在 Mem0 服务器 | 取决于部署 |
| **月费** | **$0** | $99+/月 | 向量库+Embedding费用 |
| **记忆模型** | 六层进化（提取→聚合→治理→画像→守护） | 单层存取 | 无（只做检索） |
| **冲突检测** | ✅ 矛盾知识自动标注 | ❌ | ❌ |
| **时效衰减** | ✅ 过期知识自动降权 | ❌ | ❌ |
| **知识类型** | 事实/推断/偏好/约束/观察 | 仅文本 | 仅文本片段 |
| **外部依赖** | SQLite（内置） | Qdrant + Redis + OpenAI | 向量DB + Embedding API |
| **中文支持** | ✅ 原生分词 | 有限 | 取决于Embedding |

> 💡 DeepBrain 不只是"存住"信息——它**提炼、进化、治理**知识。这是和所有竞品的本质区别。

---

## 30 秒上手

```bash
# 前置：安装 Ollama（本地AI引擎）
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download
ollama pull qwen2.5:7b && ollama pull nomic-embed-text

# 安装 DeepBrain
pip install opc-deepbrain

# 开始使用
deepbrain init
deepbrain learn "我们的技术栈是 Python + FastAPI + PostgreSQL"
deepbrain learn "周五不许发版，这是铁律"
deepbrain search "我们用什么技术"
```

输出：
```
[fact] 我们的技术栈是 Python + FastAPI + PostgreSQL (confidence: 0.95)
[constraint] 周五不许发版，这是铁律 (confidence: 0.98)
```

**每条知识都带有类型、置信度、证据链。不是存文本，是理解知识。**

---

## 六层引擎

这是 DeepBrain 的核心技术——**模仿人类记忆的知识进化系统**：

```
┌─────────────────────────────────────────────────┐
│  L5 一致性守护  回归测试 · 时间衰减 · 异常检测    │
├─────────────────────────────────────────────────┤
│  L4 用户画像    多维特征 · 偏好建模 · 行为预测    │
├─────────────────────────────────────────────────┤
│  L3 知识治理    MERGE · DEPRECATE · 漂移检测     │
├─────────────────────────────────────────────────┤
│  L2 技能合成    聚类 · 跨源聚合 · 去重           │
├─────────────────────────────────────────────────┤
│  L1 结构化提取  4Gate门控: 新颖性·可操作性·持久性·关联性 │
├─────────────────────────────────────────────────┤
│  L0 原始存储    对话/文档/手动输入               │
└─────────────────────────────────────────────────┘
```

### L1: 4Gate 门控提取

不是什么都往知识库里塞——每条信息经过四道门：

| 门 | 作用 | 示例 |
|---|---|---|
| 新颖性 | 重复的不存 | "今天天气不错" → 拒绝 |
| 可操作性 | 无用的不存 | "嗯嗯好的" → 拒绝 |
| 持久性 | 临时的不存 | "明天开会" → 短期存储 |
| 关联性 | 无关的不存 | 与已有知识无关的噪音 → 拒绝 |

### L3: 知识治理

```python
# 自动检测矛盾
brain.learn("我们用 MySQL", claim_type="fact")
brain.learn("我们用 PostgreSQL", claim_type="fact")
# → 冲突检测触发！自动标注需要人工决策

# 过时知识自动降权
brain.learn("王总的手机号是 138xxxx", valid_until="2025-12-31")
# → 过期后 confidence 自动衰减
```

---

## Python API

```python
from deepbrain import DeepBrain

brain = DeepBrain()

# 学习——自动分类、打分、存储
brain.learn("FastAPI 是我们的 Web 框架", claim_type="fact")
brain.learn("团队偏好函数式编程", claim_type="preference")
brain.learn("绝对不能用 eval()", claim_type="constraint")

# 搜索——混合检索（关键词 + 语义）
results = brain.search("Web 开发")

# 批量导入文档
brain.ingest("~/Documents/company-wiki/")

# 目录监控（文件变动自动入库）
brain.watch("~/Notes/", namespace="personal")

# 冲突检测
conflicts = brain.detect_conflicts(entry_id)

# 知识进化（聚合、去重、衰减、治理）
brain.evolve()

# 统计
stats = brain.stats()
print(f"共 {stats['total']} 条知识，{stats['facts']} 个事实，{stats['conflicts']} 个冲突")
```

---

## CLI 完整命令

```bash
deepbrain init                          # 初始化知识库
deepbrain learn "知识内容"               # 手动学习
deepbrain ingest ~/Documents            # 批量导入
deepbrain search "关键词"               # 搜索
deepbrain watch ~/Notes                 # 实时监控
deepbrain stats                         # 统计概览
deepbrain conflicts                     # 查看冲突
deepbrain evolve                        # 触发进化
deepbrain export --format json          # 导出
```

---

## 支持的文件格式

| 格式 | 扩展名 | 额外依赖 |
|------|--------|---------|
| Markdown | `.md` `.rst` | — |
| 纯文本 | `.txt` `.log` | — |
| PDF | `.pdf` | `pip install opc-deepbrain[pdf]` |
| Word | `.docx` | `pip install opc-deepbrain[docx]` |
| 代码 | `.py` `.js` `.ts` `.java` `.go` `.rs` `.c` `.cpp` | — |
| 数据 | `.json` `.yaml` `.toml` `.csv` | — |
| 全部格式 | — | `pip install opc-deepbrain[all]` |

---

## 技术架构

```
┌──────────────────────────────────────────┐
│              DeepBrain CLI / API          │
├──────────────────────────────────────────┤
│  Evolution Engine (L0-L5)                │
├──────────────┬───────────────────────────┤
│  Extraction  │  Hybrid Search            │
│  (Ollama)    │  (Keyword + Vector + RRF) │
├──────────────┴───────────────────────────┤
│  SQLite Storage (单文件，可复制可备份)     │
└──────────────────────────────────────────┘
```

- **零外部服务**: 不需要 Redis、Qdrant、Elasticsearch
- **单文件存储**: 一个 `brain.db` 搞定一切，复制就是备份
- **本地AI**: Ollama 跑在你的 CPU/GPU 上

---

## 配置

```yaml
# ~/.deepbrain/config.yaml（deepbrain init 自动生成）
storage:
  path: ~/.deepbrain/brain.db

ollama:
  base_url: http://localhost:11434
  model: qwen2.5:7b           # 知识提取/进化
  embed_model: nomic-embed-text # 语义搜索

ingest:
  ignore_patterns: ["node_modules/**", ".git/**", "__pycache__/**"]
  max_file_size_mb: 50

evolution:
  auto_evolve: true            # 知识量达阈值自动进化
  conflict_threshold: 0.8      # 冲突检测灵敏度
  decay_days: 90               # 未访问知识开始衰减的天数
```

---

## 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10 | 3.12+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 5 GB（含模型） | 10 GB+ |
| GPU | 不需要 | 有则更快 |
| Ollama | 必需 | 最新版 |

---

## 路线图

- [x] 六层记忆进化引擎
- [x] 4Gate 门控提取
- [x] 混合搜索（关键词 + 语义 + RRF）
- [x] 冲突检测与解决
- [x] 目录实时监控
- [x] 知识时效衰减
- [ ] Web UI 知识浏览器
- [ ] Benchmark 评测（LoCoMo / LongMemEval）
- [ ] 多知识库联邦
- [ ] 插件系统
- [ ] MCP 协议支持

---

## 生态

```
┌─────────────────────────────────────────────┐
│  🚀 Leaper Agent — 自进化 AI 员工团队         │
├─────────────────────────────────────────────┤
│  🤖 OPC Agent — 纯本地 AI 助手               │
├─────────────────────────────────────────────┤
│  🧠 OPC DeepBrain — 知识库引擎 ← 你在这里     │
└─────────────────────────────────────────────┘
```

| 你需要… | 用这个 |
|---------|--------|
| 只要知识库 | **OPC DeepBrain**（本项目） |
| 要本地 AI 助手 | [OPC Agent](https://github.com/Deepleaper/opc-agent)（内置 DeepBrain） |
| 要 AI 员工团队 | [Leaper Agent](https://github.com/Deepleaper/leaper-agent)（六层引擎 + CXO 模板） |

---

## 许可证

[Apache-2.0](LICENSE) — 自由使用，商用无限制。

---

<p align="center">
  <a href="https://github.com/Deepleaper"><strong>Deepleaper 跃盟开源</strong></a><br>
  <sub>让每个人都有 AI 超能力。</sub>
</p>

<p align="center">
  ⭐ 如果 DeepBrain 对你有用，请给个 Star —— 这是对我们最大的鼓励。
</p>
