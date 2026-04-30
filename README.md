<p align="center">
  <h1 align="center">🧠 OPC DeepBrain</h1>
  <p align="center"><strong>纯本地自学习知识库，数据永远不离开你的电脑。</strong></p>
  <p align="center">A self-learning knowledge base that runs 100% on your machine.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/v/opc-deepbrain?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/opc-deepbrain/"><img src="https://img.shields.io/pypi/dm/opc-deepbrain?label=%E4%B8%8B%E8%BD%BD%E9%87%8F" alt="Downloads"></a>
  <a href="https://github.com/Deepleaper/opc-deepbrain/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
  <a href="https://github.com/Deepleaper/opc-deepbrain/stargazers"><img src="https://img.shields.io/github/stars/Deepleaper/opc-deepbrain?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#核心能力">核心能力</a> ·
  <a href="#python-api">Python API</a> ·
  <a href="./README_EN.md">English</a>
</p>

---

## 为什么选 DeepBrain？

|  | Mem0 | RAG | **DeepBrain** |
|---|---|---|---|
| 本地运行 | ❌ 云端 | ❌ 需要向量数据库 | ✅ 只需 SQLite |
| 自学习 | ✅ | ❌ | ✅ |
| 知识进化 | ❌ | ❌ | ✅ 六层引擎 |
| 冲突检测 | ❌ | ❌ | ✅ 自动标注 |
| 费用 | $$ API 按量收费 | $ 基础设施成本 | **$0 完全免费** |
| 隐私 | 数据在他们服务器 | 看部署方式 | **数据永不离开你的电脑** |

---

## 快速开始

### 第一步：安装 Ollama（本地 AI 模型）

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows：从 https://ollama.com/download 下载安装包
```

```bash
# 拉取模型（约 5GB，只需一次）
ollama pull qwen2.5:7b          # 知识提取
ollama pull nomic-embed-text    # 语义搜索
```

### 第二步：安装 DeepBrain

```bash
pip install opc-deepbrain
```

### 第三步：开始使用

```bash
# 初始化知识库
deepbrain init

# 学习一条知识
deepbrain learn "我们公司用 Python 3.12 和 PostgreSQL"

# 导入整个文档目录
deepbrain ingest ~/Documents

# 搜索知识库
deepbrain search "我们用什么数据库"

# 实时监控目录（文件变动自动入库）
deepbrain watch ~/Notes

# 查看统计
deepbrain stats
```

**就这么简单。** 30 秒安装，即刻拥有一个会自己学习的知识库。

---

## 核心能力

### 📂 智能文档摄入
指向目录，自动处理所有文档。增量扫描——只处理新增和修改的文件。

支持格式：Markdown、PDF、Word、代码文件、JSON/YAML/CSV

### 🧠 结构化知识提取
不是存原文，是**提炼知识**。用本地模型把每段文本提取为：
- 类型（事实 / 推断 / 偏好 / 约束）
- 证据（原文引用）
- 置信度（0-1，随时间衰减）

### 🔍 混合搜索
关键词精确匹配 + 向量语义搜索，RRF 融合排序。**中文分词原生支持。**

### 🔄 知识进化
- 跨文档聚合同主题知识
- 自动去重
- 过时知识自动降权
- **冲突检测**：矛盾的知识自动标注

### 👁️ 目录监控
```bash
deepbrain watch ~/Documents --namespace work
```
文件变了自动入库，删了自动标记过期。用 watchdog 实时监控，没装也有轮询兜底。

### 🔒 完全本地
- 存储：SQLite（一个 `.db` 文件，可以复制带走）
- 模型：Ollama（在你电脑上跑）
- 网络：**零外发连接**

---

## Python API

```python
from deepbrain import DeepBrain

brain = DeepBrain()

# 学习
brain.learn("FastAPI 是我们的 Web 框架", claim_type="fact")
brain.learn("周五不许发版", claim_type="constraint")

# 搜索
results = brain.search("Web 框架")
for r in results:
    print(f"[{r['claim_type']}] {r['content']}")

# 导入文档
brain.ingest("~/Documents")

# 冲突检测
conflicts = brain.detect_conflicts(entry_id)

# 进化（聚合、去重、衰减）
brain.evolve()
```

---

## 配置

```yaml
# ~/.deepbrain/config.yaml（deepbrain init 自动创建）
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

## 支持的文件格式

| 格式 | 扩展名 | 额外依赖 |
|------|--------|---------|
| Markdown | `.md` `.rst` | — |
| 纯文本 | `.txt` | — |
| PDF | `.pdf` | `pip install opc-deepbrain[pdf]` |
| Word | `.docx` | `pip install opc-deepbrain[docx]` |
| 代码 | `.py` `.js` `.ts` `.java` `.go` `.rs` `.c` `.cpp` `.rb` | — |
| 数据 | `.json` `.yaml` `.toml` `.csv` | — |

---

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.12+ |
| 内存 | 4 GB | 8 GB+ |
| Ollama | 任意版本 | 最新版 |
| 磁盘 | 100 MB | 1 GB+（模型占用） |

---

## 路线图

- [x] 文档摄入 + 智能切片
- [x] 混合搜索（关键词 + 语义）
- [x] 结构化提取（事实/推断/偏好/约束）
- [x] 冲突检测与解决
- [x] 目录实时监控
- [ ] Web UI 知识浏览器
- [ ] 多知识库同步
- [ ] 插件系统

---

## 与其他产品的关系

DeepBrain 是 [OPC Agent](https://github.com/Deepleaper/opc-agent) 的记忆引擎：

```
┌─────────────────────────────────┐
│  Leaper Agent (AI 员工团队)      │
├─────────────────────────────────┤
│  OPC Agent (本地 AI 助手)        │
├─────────────────────────────────┤
│  OPC DeepBrain (知识库引擎) ← 你在这里
└─────────────────────────────────┘
```

---

## 参与贡献

欢迎 PR 和 Issue！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 许可证

[Apache-2.0](LICENSE) — 商用自由，无附加条件。

---

<p align="center">
  <strong><a href="https://github.com/Deepleaper">Deepleaper 跃盟开源</a></strong><br>
  🧠 <a href="https://github.com/Deepleaper/opc-deepbrain">DeepBrain</a> · 🤖 <a href="https://github.com/Deepleaper/opc-agent">OPC Agent</a> · 🚀 <a href="https://github.com/Deepleaper/leaper-agent">Leaper Agent</a>
</p>

<p align="center">⭐ 觉得有用？点个 Star 支持一下！</p>
