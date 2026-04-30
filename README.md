# OPC DeepBrain

> 纯本地的自学习知识库。把你电脑里的文档变成可搜索、可进化的知识。

## 它做什么？

把你指定目录里的文档（PDF、Word、Markdown、文本文件）自动读取、切片、摘要、建索引。之后你可以用自然语言搜索你自己的知识库。

不上传任何数据。不需要网络。不花钱。

## 安装

```bash
pip install opc-deepbrain
```

## 快速开始

```bash
# 初始化知识库
deepbrain init

# 摄入文档目录
deepbrain ingest ~/Documents
deepbrain ingest ~/Notes

# 搜索
deepbrain search "关于融资的策略"

# 查看知识库统计
deepbrain stats
```

## 核心能力

**📂 文档摄入**

指定目录，自动扫描 PDF / Word / Markdown / TXT 文件。增量处理——只处理新增和修改的文件。

**🧠 智能摘要**

用本地模型（Ollama）给每个文档生成摘要。不是存原文，是提炼知识。

**🔍 混合搜索**

关键词精确匹配 + 向量语义搜索，RRF 融合排序。中文分词支持。

**🔄 知识进化**

- 跨文档聚合同主题内容
- 自动去重
- 知识衰减（过时的自动降权）
- 冲突检测（矛盾的知识标注出来）

**📋 知识状态**

每条知识都有：
- 类型（事实 / 推断 / 偏好 / 约束）
- 证据（来源原文引用）
- 时效（生效时间 / 失效时间）
- 置信度（0-1，随时间衰减）

**🔒 完全本地**

- 存储：SQLite（一个 .db 文件）
- 模型：Ollama（本地运行）
- 零网络依赖

## 作为 Python 库使用

```python
from deepbrain import DeepBrain

brain = DeepBrain()

# 摄入
brain.ingest("~/Documents")

# 学习一条知识
brain.learn("跃盟科技是一家AI情景智能公司", claim_type="fact")

# 搜索
results = brain.search("跃盟科技做什么")
for r in results:
    print(f"[{r['claim_type']}] {r['content']}")

# 进化（聚合、去重、衰减）
brain.evolve()
```

## 与 OPC Agent 集成

OPC DeepBrain 可以被 OPC Agent 调用，让你的 AI 助手自动使用你的知识库：

```python
from deepbrain import DeepBrain

brain = DeepBrain()
context = brain.search("用户问题相关的内容")
# 注入到 AI 对话的 system prompt
```

## 支持的文件格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| Markdown | .md | 无 |
| 纯文本 | .txt | 无 |
| PDF | .pdf | PyMuPDF |
| Word | .docx | python-docx |
| 代码文件 | .py .js .ts .java 等 | 无 |

## 配置

```yaml
# ~/.deepbrain/config.yaml
storage:
  path: ~/.deepbrain/brain.db

ollama:
  base_url: http://localhost:11434
  model: qwen2.5:7b          # 摘要/提取用
  embed_model: nomic-embed-text  # 向量化用

ingest:
  watch_dirs:
    - ~/Documents
    - ~/Notes
  ignore_patterns:
    - "*.tmp"
    - "node_modules/**"
  max_file_size_mb: 50
```

## 系统要求

- Python 3.10+
- Ollama（推荐 qwen2.5:7b + nomic-embed-text）
- 8GB+ 内存

## 许可证

Apache-2.0
