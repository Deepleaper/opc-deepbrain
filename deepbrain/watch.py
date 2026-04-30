"""Directory watch — auto-ingest on file changes."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {
    '.md', '.txt', '.rst', '.csv', '.json', '.yaml', '.yml', '.toml',
    '.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.rb', '.sh', '.sql',
    '.pdf', '.docx',
}
_IGNORE_DIRS = {'node_modules', '.git', '__pycache__', '.venv', 'venv', 'dist', 'build'}


def watch_directory(
    brain: Any,
    directory: str,
    namespace: str = "documents",
    interval: float = 5.0,
    daemon: bool = False,
) -> None:
    """Watch a directory for changes and auto-ingest.
    
    Uses watchdog if available, otherwise falls back to polling.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        _watch_with_watchdog(brain, directory, namespace, Observer, FileSystemEventHandler)
    except ImportError:
        logger.info("watchdog not installed, using polling (pip install opc-deepbrain[watch])")
        _watch_with_polling(brain, directory, namespace, interval)


def _watch_with_watchdog(brain, directory, namespace, Observer, FileSystemEventHandler):
    """Watch using watchdog library."""
    from deepbrain.ingest import ingest_file

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                self._handle(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                self._handle(event.src_path)

        def on_deleted(self, event):
            if not event.is_directory:
                self._expire(event.src_path)

        def _handle(self, filepath):
            ext = Path(filepath).suffix.lower()
            if ext not in _SUPPORTED_EXTS:
                return
            # Skip ignored dirs
            parts = Path(filepath).parts
            if any(p in _IGNORE_DIRS for p in parts):
                return
            try:
                result = ingest_file(brain, filepath, namespace=namespace)
                if result:
                    logger.info("Ingested: %s -> %s", filepath, result[:8])
            except Exception as e:
                logger.warning("Ingest failed: %s: %s", filepath, e)

        def _expire(self, filepath):
            """Mark entries from deleted file as expired."""
            try:
                with brain._lock:
                    brain.conn.execute(
                        "UPDATE deepbrain SET status='expired', updated_at=? WHERE source=? AND status='active'",
                        (brain._now_str(), filepath),
                    )
                    brain.conn.commit()
                logger.info("Expired entries from deleted: %s", filepath)
            except Exception as e:
                logger.warning("Expire failed: %s: %s", filepath, e)

    observer = Observer()
    observer.schedule(Handler(), directory, recursive=True)
    observer.start()
    print(f"Watching: {directory} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _watch_with_polling(brain, directory, namespace, interval):
    """Fallback: poll for changes."""
    from deepbrain.ingest import ingest_file

    # Build initial state
    state: dict[str, float] = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for f in files:
            fp = os.path.join(root, f)
            ext = Path(f).suffix.lower()
            if ext in _SUPPORTED_EXTS:
                try:
                    state[fp] = os.path.getmtime(fp)
                except OSError:
                    pass

    print(f"Watching: {directory} (polling every {interval}s, Ctrl+C to stop)")
    print(f"Tracking {len(state)} files")

    try:
        while True:
            time.sleep(interval)
            current: dict[str, float] = {}
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
                for f in files:
                    fp = os.path.join(root, f)
                    ext = Path(f).suffix.lower()
                    if ext in _SUPPORTED_EXTS:
                        try:
                            current[fp] = os.path.getmtime(fp)
                        except OSError:
                            pass

            # New or modified
            for fp, mtime in current.items():
                if fp not in state or state[fp] < mtime:
                    try:
                        result = ingest_file(brain, fp, namespace=namespace)
                        if result:
                            print(f"  + Ingested: {os.path.basename(fp)}")
                    except Exception as e:
                        logger.warning("Ingest failed: %s: %s", fp, e)

            # Deleted
            for fp in set(state) - set(current):
                with brain._lock:
                    brain.conn.execute(
                        "UPDATE deepbrain SET status='expired', updated_at=? WHERE source=? AND status='active'",
                        (_now_helper(), fp),
                    )
                    brain.conn.commit()
                print(f"  - Expired: {os.path.basename(fp)}")

            state = current
    except KeyboardInterrupt:
        print("\nStopped watching.")


def _now_helper():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
