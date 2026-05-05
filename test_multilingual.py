# -*- coding: utf-8 -*-
import os, sys, tempfile
os.environ["DEEPBRAIN_EMBED_PROVIDER"] = "sentence-transformers"
os.environ["DEEPBRAIN_ST_MODEL"] = "paraphrase-multilingual-MiniLM-L12-v2"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

from deepbrain.brain import DeepBrain

D2_RECALL = [
    {"stored": ["公司用 Python 3.12", "部署在阿里云", "CTO 是李明"],
     "query": "我们用什么语言开发的？", "expected": "Python 3.12"},
    {"stored": ["目标 Q3 DAU 10万", "现在 DAU 3万", "主要渠道是抖音"],
     "query": "我们的增长目标是什么？", "expected": "Q3 DAU 10万"},
    {"stored": ["客户是中大型电商", "客单价 5000-20000", "决策周期 2-3 个月"],
     "query": "我们的客户画像？", "expected": "中大型电商"},
    {"stored": ["技术债在支付模块", "用 PHP 5.6", "计划 Q4 重构"],
     "query": "有什么技术债？", "expected": "支付模块"},
    {"stored": ["张总负责销售", "王总负责产品", "李明是 CTO"],
     "query": "谁管技术？", "expected": "李明"},
    {"stored": ["竞品融了B轮5000万", "我们A轮2000万", "投资人是红杉"],
     "query": "竞品的融资情况？", "expected": "B轮5000万"},
    {"stored": ["RAG 召回率 30%", "试过向量数据库 Milvus", "最终选了 Elasticsearch"],
     "query": "之前用 RAG 效果怎么样？", "expected": "召回率30%"},
    {"stored": ["公司50人", "技术20人", "年底扩到80人"],
     "query": "团队多大？", "expected": "50人"},
    {"stored": ["放弃ToC转ToB", "获客成本太高", "ToB 客单价高"],
     "query": "为什么转型？", "expected": "获客成本"},
    {"stored": ["付费转化率 3.8%", "上月 2.1%", "新定价策略"],
     "query": "最近业务数据怎么样？", "expected": "付费转化率"},
]

brain = DeepBrain(os.path.join(tempfile.mkdtemp(), 'test.db'))
for case in D2_RECALL:
    for item in case["stored"]:
        brain.learn(item, namespace="recall")

score = 0
for case in D2_RECALL:
    results = brain.search(case["query"], top_k=5, namespace="recall")
    combined = " ".join(r["content"] for r in results).lower().replace("-","").replace(" ","")
    expected = case["expected"].lower().replace("-","").replace(" ","")
    hit = expected in combined
    score += 1 if hit else 0
    mark = "OK" if hit else "MISS"
    print(f"{mark}: q={case['query'][:15]}  exp={case['expected']}  got={[r['content'][:20] for r in results[:3]]}")

print(f"\nScore: {score}/10")
