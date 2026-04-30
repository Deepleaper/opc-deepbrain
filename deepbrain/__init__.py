"""OPC DeepBrain — Local-first self-learning knowledge base."""

from deepbrain.brain import DeepBrain
from deepbrain.ingest import ingest_directory, ingest_file

__all__ = ["DeepBrain", "ingest_directory", "ingest_file"]
__version__ = "0.1.0"
