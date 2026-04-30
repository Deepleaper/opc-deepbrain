"""DeepBrain core — SQLite knowledge store with hybrid search and knowledge state."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Chinese segmentation ────────────────────────────────────────────────────

try:
    import jieba
    jieba.setLogLevel(logging.WARNING)
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# ─── Embedding ───────────────────────────────────────────────────────────────

_EMBED_MODEL = os.environ.get("DEEPBRAIN_EMBED_MODEL", "nomic-embed-text")
_EMBED_URL = os.environ.get("DEEPBRAIN_EMBED_URL", "http://localhost:11434")
_embed_available: bool | None = None
_embed_lock = threading.Lock()


def _get_embedding(text: str) -> list[float] | None:
    global _embed_available
    if _embed_available is False:
        return None
    try:
        import urllib.request
        url = f"{_EMBED_URL.rstrip('/')}/api/embeddings"
        body = json.dumps({"model": _EMBED_MODEL, "prompt": text[:2000]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        with _embed_lock:
            _embed_available = True
        return data["embedding"]
    except Exception as e:
        with _embed_lock:
            if _embed_available is None:
                logger.info("Embedding unavailable (%s) — keyword fallback", e)
                _embed_available = False
        return None


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_keywords(text: str) -> list[str]:
    import re
    if _HAS_JIEBA:
        tokens = [w.strip().lower() for w in jieba.cut(text, cut_all=False) if len(w.strip()) >= 2]
    else:
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", text)]
    return list(dict.fromkeys(tokens))[:30]


_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could this that these those it its i me my we our "
    "you your he him his she her they them their what which who whom how when where "
    "why all each every both few more most other some such no not only own same so "
    "than too very just because but and or if then else for at by from in into of "
    "on to with about between through during before after above below up down out "
    "off over under again further once here there".split()
)

_RRF_K = 60


# ─── Schema ──────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS deepbrain (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    keywords    TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    namespace   TEXT NOT NULL DEFAULT 'default',
    layer       TEXT NOT NULL DEFAULT 'l0',
    entry_type  TEXT NOT NULL DEFAULT 'raw',
    confidence  REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'active',
    embedding   BLOB,
    evidence    TEXT,
    valid_from  TEXT,
    valid_until TEXT,
    claim_type  TEXT NOT NULL DEFAULT 'observation',
    supersedes  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ns ON deepbrain (namespace);
CREATE INDEX IF NOT EXISTS idx_layer ON deepbrain (layer);
CREATE INDEX IF NOT EXISTS idx_status ON deepbrain (status);
CREATE INDEX IF NOT EXISTS idx_claim ON deepbrain (claim_type);
"""


class DeepBrain:
    """Local-first self-learning knowledge base."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            home = Path.home() / ".deepbrain"
            home.mkdir(exist_ok=True)
            db_path = str(home / "brain.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(_CREATE_SQL)
            self.conn.commit()

    # ── Core API ─────────────────────────────────────────────────────────────

    def learn(
        self,
        content: str,
        source: str = "",
        namespace: str = "default",
        claim_type: str = "observation",
        confidence: float = 0.5,
        evidence: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
        entry_type: str = "raw",
        layer: str = "l0",
    ) -> str:
        """Store a knowledge entry. Returns entry id."""
        if not content.strip():
            return ""
        entry_id = str(uuid.uuid4())
        now = _now()
        keywords = _extract_keywords(content)
        meta_json = json.dumps(metadata or {})

        with self._lock:
            self.conn.execute(
                """INSERT INTO deepbrain
                  (id, content, keywords, source, namespace, layer, entry_type,
                   confidence, access_count, metadata, status, embedding,
                   evidence, valid_from, valid_until, claim_type, supersedes,
                   created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,0,?,'active',NULL,?,?,?,?,?,?,?)""",
                (entry_id, content.strip(), json.dumps(keywords), source, namespace,
                 layer, entry_type, confidence, meta_json,
                 evidence, valid_from or now, valid_until, claim_type, supersedes,
                 now, now),
            )
            if supersedes:
                self.conn.execute(
                    "UPDATE deepbrain SET status='superseded', valid_until=? WHERE id=?",
                    (now, supersedes),
                )
            self.conn.commit()

        # Async embedding
        def _embed():
            vec = _get_embedding(content)
            if vec:
                with self._lock:
                    self.conn.execute(
                        "UPDATE deepbrain SET embedding=? WHERE id=?",
                        (_vec_to_blob(vec), entry_id),
                    )
                    self.conn.commit()
        threading.Thread(target=_embed, daemon=True).start()

        # Auto conflict detection (async, best-effort)
        def _detect():
            try:
                conflicts = self.detect_conflicts(entry_id)
                if conflicts:
                    meta = metadata or {}
                    meta["conflict_with"] = [c["id"] for c in conflicts]
                    with self._lock:
                        self.conn.execute(
                            "UPDATE deepbrain SET metadata=? WHERE id=?",
                            (json.dumps(meta), entry_id),
                        )
                        self.conn.commit()
            except Exception:
                pass
        threading.Thread(target=_detect, daemon=True).start()

        return entry_id

    def detect_conflicts(self, entry_id: str) -> list[dict[str, Any]]:
        """Detect knowledge entries that conflict with the given entry.
        
        Uses keyword overlap to find similar entries, then checks for contradiction
        signals (opposite claims about same subject).
        """
        with self._lock:
            row = self.conn.execute("SELECT * FROM deepbrain WHERE id=?", (entry_id,)).fetchone()
        if not row:
            return []

        content = row["content"]
        keywords = json.loads(row["keywords"] or "[]")
        if not keywords:
            return []

        # Find similar entries (same namespace, different id, active)
        like_parts = " OR ".join(["content LIKE ?"] * min(len(keywords), 3))
        params = [f"%{k}%" for k in keywords[:3]]
        with self._lock:
            candidates = self.conn.execute(
                f"""SELECT * FROM deepbrain 
                    WHERE id != ? AND status='active' AND namespace=? AND ({like_parts})
                    LIMIT 20""",
                [entry_id, row["namespace"]] + params,
            ).fetchall()

        conflicts = []
        content_lower = content.lower()
        
        # Contradiction signals
        _NEGATION_PAIRS = [
            ("not ", ""), ("don't ", "do "), ("doesn't ", "does "),
            ("won't ", "will "), ("can't ", "can "), ("isn't ", "is "),
            ("no longer ", "still "), ("停止", "继续"), ("不是", "是"),
            ("不再", "仍然"), ("废弃", "使用"),
        ]

        for cand in candidates:
            cand_content = cand["content"].lower()
            # Check keyword overlap (high overlap = same topic)
            cand_keywords = set(json.loads(cand["keywords"] or "[]"))
            overlap = len(set(keywords) & cand_keywords)
            if overlap < 2:
                continue

            # Check for contradiction signals
            is_conflict = False
            for neg, pos in _NEGATION_PAIRS:
                if (neg in content_lower and pos in cand_content) or \
                   (pos in content_lower and neg in cand_content):
                    is_conflict = True
                    break

            # Also flag if same claim_type=fact about very similar content
            if not is_conflict and row["claim_type"] == "fact" and cand["claim_type"] == "fact":
                # High keyword overlap + different content = potential conflict
                if overlap >= 3 and content_lower != cand_content:
                    is_conflict = True

            if is_conflict:
                conflicts.append(self._row_to_dict(cand))

        return conflicts

    def resolve_conflict(self, keep_id: str, discard_id: str) -> None:
        """Resolve a conflict: keep one entry, supersede the other."""
        now = _now()
        with self._lock:
            self.conn.execute(
                "UPDATE deepbrain SET status='superseded', valid_until=?, updated_at=? WHERE id=?",
                (now, now, discard_id),
            )
            # Update winner metadata
            row = self.conn.execute("SELECT metadata FROM deepbrain WHERE id=?", (keep_id,)).fetchone()
            if row:
                meta = json.loads(row["metadata"] or "{}")
                resolved = meta.get("resolved_conflicts", [])
                resolved.append(discard_id)
                meta["resolved_conflicts"] = resolved
                if "conflict_with" in meta and discard_id in meta["conflict_with"]:
                    meta["conflict_with"].remove(discard_id)
                self.conn.execute(
                    "UPDATE deepbrain SET metadata=?, updated_at=? WHERE id=?",
                    (json.dumps(meta), now, keep_id),
                )
            self.conn.commit()

    def _now_str(self) -> str:
        """Return current UTC timestamp as ISO string."""
        return _now()

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        claim_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: keyword + vector, RRF fusion."""
        if not query.strip():
            return []

        now_str = _now()
        query_vec = _get_embedding(query)
        query_terms = [t.lower() for t in _extract_keywords(query) if t not in _STOP_WORDS]

        # Base filter
        where = ["(status='active' OR status IS NULL)", "(valid_until IS NULL OR valid_until > ?)"]
        params: list[Any] = [now_str]
        if namespace:
            where.append("namespace=?")
            params.append(namespace)
        if claim_type:
            where.append("claim_type=?")
            params.append(claim_type)
        where_sql = " AND ".join(where)

        # Keyword search
        kw_results: list[tuple[str, float]] = []
        if query_terms:
            like_parts = " OR ".join(["content LIKE ?"] * len(query_terms))
            kw_params = params + [f"%{t}%" for t in query_terms[:5]]
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT id, content, keywords FROM deepbrain WHERE {where_sql} AND ({like_parts}) LIMIT 50",
                    kw_params,
                ).fetchall()
            for row in rows:
                content_lower = row["content"].lower()
                score = sum(1 for t in query_terms if t in content_lower)
                kw_results.append((row["id"], score))
            kw_results.sort(key=lambda x: -x[1])

        # Vector search
        vec_results: list[tuple[str, float]] = []
        if query_vec:
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT id, embedding FROM deepbrain WHERE {where_sql} AND embedding IS NOT NULL LIMIT 200",
                    params,
                ).fetchall()
            for row in rows:
                sim = _cosine_similarity(query_vec, _blob_to_vec(row["embedding"]))
                vec_results.append((row["id"], sim))
            vec_results.sort(key=lambda x: -x[1])
            vec_results = vec_results[:20]

        # RRF fusion
        scores: dict[str, float] = {}
        for rank, (eid, _) in enumerate(kw_results[:20]):
            scores[eid] = scores.get(eid, 0) + 1.0 / (_RRF_K + rank + 1)
        for rank, (eid, _) in enumerate(vec_results):
            scores[eid] = scores.get(eid, 0) + 1.0 / (_RRF_K + rank + 1)

        # Fallback: if nothing found, return recent entries
        if not scores:
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT * FROM deepbrain WHERE {where_sql} ORDER BY updated_at DESC LIMIT ?",
                    params + [top_k],
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

        # Fetch top-k
        sorted_ids = sorted(scores, key=lambda x: -scores[x])[:top_k]
        results = []
        for eid in sorted_ids:
            with self._lock:
                row = self.conn.execute("SELECT * FROM deepbrain WHERE id=?", (eid,)).fetchone()
            if row:
                d = self._row_to_dict(row)
                # Update access count
                self.conn.execute(
                    "UPDATE deepbrain SET access_count=access_count+1, last_accessed=? WHERE id=?",
                    (_now(), eid),
                )
                results.append(d)
        if results:
            self.conn.commit()
        return results

    def evolve(self) -> dict[str, int]:
        """Run knowledge evolution: decay old entries, expire invalid ones."""
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        decayed = 0
        expired = 0

        with self._lock:
            rows = self.conn.execute(
                "SELECT id, confidence, updated_at, valid_until FROM deepbrain WHERE status='active'"
            ).fetchall()

        for row in rows:
            # Expire
            if row["valid_until"] and row["valid_until"] < now_str:
                with self._lock:
                    self.conn.execute(
                        "UPDATE deepbrain SET status='expired', updated_at=? WHERE id=?",
                        (now_str, row["id"]),
                    )
                expired += 1
                continue

            # Decay confidence
            try:
                updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
                age_days = (now - updated).total_seconds() / 86400
            except Exception:
                continue

            conf = row["confidence"] or 0.5
            if age_days >= 90:
                new_conf = conf * 0.7
            elif age_days >= 30:
                new_conf = conf * 0.9
            else:
                continue

            if new_conf < conf - 0.001:
                with self._lock:
                    self.conn.execute(
                        "UPDATE deepbrain SET confidence=?, updated_at=? WHERE id=?",
                        (new_conf, now_str, row["id"]),
                    )
                decayed += 1

        with self._lock:
            self.conn.commit()
        return {"decayed": decayed, "expired": expired}

    def stats(self) -> dict[str, Any]:
        """Return knowledge base statistics."""
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) FROM deepbrain").fetchone()[0]
            active = self.conn.execute("SELECT COUNT(*) FROM deepbrain WHERE status='active'").fetchone()[0]
            by_type = dict(self.conn.execute(
                "SELECT claim_type, COUNT(*) FROM deepbrain WHERE status='active' GROUP BY claim_type"
            ).fetchall())
            by_ns = dict(self.conn.execute(
                "SELECT namespace, COUNT(*) FROM deepbrain WHERE status='active' GROUP BY namespace"
            ).fetchall())
        return {
            "total": total,
            "active": active,
            "superseded": total - active,
            "by_claim_type": by_type,
            "by_namespace": by_ns,
            "db_path": self.db_path,
        }

    def get(self, entry_id: str) -> dict[str, Any] | None:
        """Get a single entry by ID."""
        with self._lock:
            row = self.conn.execute("SELECT * FROM deepbrain WHERE id=?", (entry_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["keywords"] = json.loads(d.get("keywords") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        d.pop("embedding", None)
        return d
