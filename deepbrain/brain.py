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

# Negation words used for keyword-scoped contradiction detection
_NEGATION_WORDS: frozenset[str] = frozenset([
    "not", "no", "never", "neither", "nor", "lack", "fail", "false",
    "don't", "doesn't", "didn't", "won't", "can't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "no longer",
    "不", "没", "非", "无", "否", "停止", "不是", "不再", "废弃",
])

# Explicit contraction pairs (negated_form, positive_form) for cross-entry contradiction checks
_NEGATION_CONTRACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("don't", "do"), ("doesn't", "does"), ("won't", "will"),
    ("can't", "can"), ("isn't", "is"), ("aren't", "are"),
    ("wasn't", "was"), ("weren't", "were"), ("haven't", "have"),
    ("hasn't", "has"), ("hadn't", "had"), ("no longer", "still"),
    ("停止", "继续"), ("不是", "是"), ("不再", "仍然"), ("废弃", "使用"),
)

_RRF_K = 60


def _has_negation_near(text_lower: str, keyword: str, window: int = 60) -> bool:
    """Return True if a negation word appears within *window* chars of *keyword*."""
    idx = text_lower.find(keyword.lower())
    if idx == -1:
        return False
    start = max(0, idx - window)
    end = min(len(text_lower), idx + len(keyword) + window)
    snippet = text_lower[start:end]
    return any(neg in snippet for neg in _NEGATION_WORDS)


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
        """Detect knowledge entries that contradict the given entry.

        Strategy:
        1. Find candidates in the same namespace with ≥2 overlapping keywords.
        2. Flag a conflict when an explicit contraction pair (e.g. "don't" vs "do")
           appears asymmetrically across the two entries.
        3. Flag a conflict when a shared keyword is negated in one entry but
           affirmed in the other (keyword-scoped negation via context window).
        4. Flag two ``fact`` entries as conflicting when they share ≥3 keywords
           but differ in content.

        Args:
            entry_id: ID of the entry to check for conflicts.

        Returns:
            List of conflicting entry dicts (embedding field excluded).
        """
        with self._lock:
            row = self.conn.execute("SELECT * FROM deepbrain WHERE id=?", (entry_id,)).fetchone()
        if not row:
            return []

        content: str = row["content"]
        keywords: list[str] = json.loads(row["keywords"] or "[]")
        if not keywords:
            return []

        like_parts = " OR ".join(["content LIKE ?"] * min(len(keywords), 3))
        params: list[Any] = [f"%{k}%" for k in keywords[:3]]
        with self._lock:
            candidates = self.conn.execute(
                f"""SELECT * FROM deepbrain
                    WHERE id != ? AND status='active' AND namespace=? AND ({like_parts})
                    LIMIT 20""",
                [entry_id, row["namespace"]] + params,
            ).fetchall()

        content_lower = content.lower()
        conflicts: list[dict[str, Any]] = []

        for cand in candidates:
            cand_lower: str = cand["content"].lower()
            cand_keywords: set[str] = set(json.loads(cand["keywords"] or "[]"))
            shared: set[str] = set(keywords) & cand_keywords
            if len(shared) < 2:
                continue

            is_conflict = False

            # Pass 1: explicit contraction-pair contradiction
            for neg_form, pos_form in _NEGATION_CONTRACTION_PAIRS:
                entry_has_neg = neg_form in content_lower
                entry_has_pos = pos_form in content_lower and not entry_has_neg
                cand_has_neg = neg_form in cand_lower
                cand_has_pos = pos_form in cand_lower and not cand_has_neg
                if (entry_has_neg and cand_has_pos) or (cand_has_neg and entry_has_pos):
                    is_conflict = True
                    break

            # Pass 2: keyword-scoped negation — one entry negates a shared keyword, other affirms it
            if not is_conflict:
                for kw in shared:
                    if _has_negation_near(content_lower, kw) != _has_negation_near(cand_lower, kw):
                        is_conflict = True
                        break

            # Pass 3: two 'fact' entries with high keyword overlap but different content
            if not is_conflict and row["claim_type"] == "fact" and cand["claim_type"] == "fact":
                if len(shared) >= 3 and content_lower != cand_lower:
                    is_conflict = True

            if is_conflict:
                conflicts.append(self._row_to_dict(cand))

        return conflicts

    def resolve_conflict(self, keep_id: str, discard_id: str) -> None:
        """Mark one conflicting entry as superseded, retaining the other as authoritative.

        Sets ``discard_id`` status to ``'superseded'`` and updates ``keep_id``'s
        metadata: appends to ``resolved_conflicts`` and removes from ``conflict_with``.

        Args:
            keep_id: ID of the entry to retain.
            discard_id: ID of the entry to supersede.

        Raises:
            ValueError: If ``keep_id`` and ``discard_id`` are identical.
        """
        if keep_id == discard_id:
            raise ValueError("keep_id and discard_id must differ")
        now = _now()
        with self._lock:
            self.conn.execute(
                "UPDATE deepbrain SET status='superseded', valid_until=?, updated_at=? WHERE id=?",
                (now, now, discard_id),
            )
            row = self.conn.execute("SELECT metadata FROM deepbrain WHERE id=?", (keep_id,)).fetchone()
            if row:
                meta: dict[str, Any] = json.loads(row["metadata"] or "{}")
                resolved: list[str] = meta.get("resolved_conflicts", [])
                if discard_id not in resolved:
                    resolved.append(discard_id)
                meta["resolved_conflicts"] = resolved
                conflict_with: list[str] = meta.get("conflict_with", [])
                if discard_id in conflict_with:
                    conflict_with.remove(discard_id)
                meta["conflict_with"] = conflict_with
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
