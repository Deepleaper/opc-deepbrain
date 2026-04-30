"""OPC DeepBrain — Local-first self-learning knowledge base."""

from deepbrain.brain import DeepBrain
from deepbrain.ingest import ingest_directory, ingest_file
from deepbrain.chunker import chunk_document
from deepbrain.watch import watch_directory

__all__ = ["DeepBrain", "ingest_directory", "ingest_file", "chunk_document", "watch_directory"]
__version__ = "0.2.0"
