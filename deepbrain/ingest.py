"""Document ingestion — scan directories, extract text, feed into DeepBrain."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported extensions
_TEXT_EXTS = {".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
_CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".sh", ".sql"}
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}

_DEFAULT_IGNORE = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def ingest_directory(
    brain: Any,
    directory: str,
    namespace: str = "documents",
    ignore_patterns: set[str] | None = None,
    max_file_size: int = _MAX_FILE_SIZE,
) -> dict[str, int]:
    """Recursively ingest all supported files in a directory.
    
    Returns: {"ingested": N, "skipped": M, "errors": E}
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    ignore = ignore_patterns or _DEFAULT_IGNORE
    stats = {"ingested": 0, "skipped": 0, "errors": 0}

    for root, dirs, files in os.walk(directory):
        # Prune ignored dirs
        dirs[:] = [d for d in dirs if d not in ignore]

        for fname in files:
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()

            if ext not in (_TEXT_EXTS | _CODE_EXTS | _PDF_EXTS | _DOCX_EXTS):
                stats["skipped"] += 1
                continue

            if os.path.getsize(fpath) > max_file_size:
                stats["skipped"] += 1
                continue

            try:
                result = ingest_file(brain, fpath, namespace=namespace)
                if result:
                    stats["ingested"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.warning("Failed to ingest %s: %s", fpath, e)
                stats["errors"] += 1

    return stats


def ingest_file(
    brain: Any,
    filepath: str,
    namespace: str = "documents",
) -> str | None:
    """Ingest a single file. Returns entry_id or None if skipped (duplicate)."""
    filepath = os.path.expanduser(filepath)
    if not os.path.isfile(filepath):
        return None

    ext = Path(filepath).suffix.lower()
    text = _extract_text(filepath, ext)
    if not text or len(text.strip()) < 20:
        return None

    # Dedup by content hash
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Check if already ingested (store hash in metadata)
    existing = brain.conn.execute(
        "SELECT id FROM deepbrain WHERE source=? AND status='active'",
        (filepath,),
    ).fetchone()
    if existing:
        # Check if content changed
        entry = brain.get(existing["id"])
        if entry and entry.get("metadata", {}).get("hash") == content_hash:
            return None  # No change
        # Content changed — supersede old entry
        old_id = existing["id"]
    else:
        old_id = None

    # Truncate for storage (keep first 5000 chars for now)
    content = text[:5000]

    # Generate summary with local model (best-effort)
    summary = _summarize(content)
    store_content = summary if summary else content[:2000]

    entry_id = brain.learn(
        content=store_content,
        source=filepath,
        namespace=namespace,
        claim_type="fact",
        evidence=content[:500],
        supersedes=old_id,
        metadata={
            "hash": content_hash,
            "filename": os.path.basename(filepath),
            "ext": ext,
            "size": os.path.getsize(filepath),
            "full_content_chars": len(text),
        },
    )
    return entry_id


def _extract_text(filepath: str, ext: str) -> str | None:
    """Extract text content from a file."""
    if ext in _TEXT_EXTS | _CODE_EXTS:
        return _read_text(filepath)
    elif ext in _PDF_EXTS:
        return _read_pdf(filepath)
    elif ext in _DOCX_EXTS:
        return _read_docx(filepath)
    return None


def _read_text(filepath: str) -> str | None:
    """Read plain text file."""
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    return None


def _read_pdf(filepath: str) -> str | None:
    """Read PDF using PyMuPDF (optional dependency)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        logger.info("PyMuPDF not installed — skipping PDF: %s", filepath)
        return None
    except Exception as e:
        logger.warning("PDF read error %s: %s", filepath, e)
        return None


def _read_docx(filepath: str) -> str | None:
    """Read Word document (optional dependency)."""
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        logger.info("python-docx not installed — skipping: %s", filepath)
        return None
    except Exception as e:
        logger.warning("DOCX read error %s: %s", filepath, e)
        return None


def _summarize(text: str) -> str | None:
    """Use local Ollama to summarize text. Returns None on failure."""
    model = os.environ.get("DEEPBRAIN_MODEL", "qwen2.5:7b")
    base_url = os.environ.get("DEEPBRAIN_OLLAMA_URL", "http://localhost:11434")

    prompt = f"""用一段话总结以下内容的核心知识点（不超过200字）：

{text[:3000]}

只返回总结，不要其他内容。"""

    try:
        import urllib.request
        url = f"{base_url.rstrip('/')}/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip() or None
    except Exception:
        return None


# Need json for _summarize
import json
