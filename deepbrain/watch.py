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
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
        _watch_with_watchdog(brain, directory, namespace, Observer, FileSystemEventHandler)
    except ImportError:
        logger.info("watchdog not installed, using polling (pip install opc-deepbrain[watch])")
        _watch_with_polling(brain, directory, namespace, interval)


def _expire_file(brain: Any, filepath: str) -> None:
    """Mark active entries from a deleted file as expired."""
    try:
        with brain._lock:
            brain.conn.execute(
                "UPDATE deepbrain SET status='expired', updated_at=? WHERE source=? AND status='active'",
                (brain._now_str(), filepath),
            )
            brain.conn.commit()
        logger.info("Expired entries for deleted file: %s", filepath)
    except Exception as e:
        logger.warning("Expire failed for %s: %s", filepath, e)


def _scan_directory(directory: str) -> dict[str, float]:
    """Return {filepath: mtime} for all supported files under directory."""
    result: dict[str, float] = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for f in files:
            if Path(f).suffix.lower() not in _SUPPORTED_EXTS:
                continue
            fp = os.path.join(root, f)
            try:
                result[fp] = os.path.getmtime(fp)
            except OSError as e:
                logger.debug("Could not stat %s: %s", fp, e)
    return result


def _watch_with_watchdog(
    brain: Any,
    directory: str,
    namespace: str,
    Observer: type[Any],
    FileSystemEventHandler: type[Any],
) -> None:
    """Watch using the watchdog library."""
    from deepbrain.ingest import ingest_file

    class _Handler(FileSystemEventHandler):
        def on_created(self, event: Any) -> None:
            if not event.is_directory:
                self._handle(event.src_path)

        def on_modified(self, event: Any) -> None:
            if not event.is_directory:
                self._handle(event.src_path)

        def on_deleted(self, event: Any) -> None:
            if not event.is_directory:
                _expire_file(brain, event.src_path)

        def _handle(self, filepath: str) -> None:
            if Path(filepath).suffix.lower() not in _SUPPORTED_EXTS:
                return
            if any(p in _IGNORE_DIRS for p in Path(filepath).parts):
                return
            try:
                result = ingest_file(brain, filepath, namespace=namespace)
                if result:
                    logger.info("Ingested: %s -> %s", filepath, result[:8])
            except Exception as e:
                logger.warning("Ingest failed for %s: %s", filepath, e)

    observer = Observer()
    observer.schedule(_Handler(), directory, recursive=True)
    observer.start()
    logger.info("Started watchdog observer on %s", directory)
    print(f"Watching: {directory} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching.")
    finally:
        observer.join()


def _watch_with_polling(
    brain: Any,
    directory: str,
    namespace: str,
    interval: float,
) -> None:
    """Fallback file watcher: poll for mtime changes."""
    from deepbrain.ingest import ingest_file

    state = _scan_directory(directory)
    print(f"Watching: {directory} (polling every {interval}s, Ctrl+C to stop)")
    print(f"Tracking {len(state)} files")
    logger.info("Started polling watcher on %s with %d files", directory, len(state))

    try:
        while True:
            time.sleep(interval)
            current = _scan_directory(directory)

            for fp, mtime in current.items():
                if fp not in state or state[fp] < mtime:
                    try:
                        result = ingest_file(brain, fp, namespace=namespace)
                        if result:
                            logger.info("Ingested: %s", fp)
                    except Exception as e:
                        logger.warning("Ingest failed for %s: %s", fp, e)

            for fp in set(state) - set(current):
                _expire_file(brain, fp)

            state = current
    except KeyboardInterrupt:
        print("\nStopped watching.")
