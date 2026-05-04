"""Tests for embedding providers and hybrid search with vector similarity."""

import json
import time
import threading
from unittest.mock import patch, MagicMock

import deepbrain.brain as bm
from deepbrain.brain import (
    DeepBrain,
    _local_embedding,
    _get_ollama_embedding,
    _get_st_embedding,
    _get_embedding,
    _cosine_similarity,
    _vec_to_blob,
    _blob_to_vec,
)


class TestLocalEmbedding:
    """Test the local n-gram hash embedding (always available)."""

    def test_returns_vector(self):
        vec = _local_embedding("hello world")
        assert len(vec) == 256
        # Should be normalized
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_empty_string(self):
        vec = _local_embedding("")
        assert all(x == 0.0 for x in vec)

    def test_chinese_text(self):
        vec = _local_embedding("知识图谱构建")
        assert len(vec) == 256
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_similar_texts_higher_similarity(self):
        v1 = _local_embedding("machine learning algorithms")
        v2 = _local_embedding("machine learning models")
        v3 = _local_embedding("chocolate cake recipe")
        sim_close = _cosine_similarity(v1, v2)
        sim_far = _cosine_similarity(v1, v3)
        assert sim_close > sim_far


class TestOllamaEmbedding:
    """Test Ollama embedding with mocked HTTP."""

    def test_success(self):
        fake_vec = [0.1] * 768
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": fake_vec}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            vec = _get_ollama_embedding("test text")
        assert vec == fake_vec

    def test_failure_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            vec = _get_ollama_embedding("test text")
        assert vec is None


class TestSentenceTransformersEmbedding:
    """Test sentence-transformers provider with mock."""

    def test_success(self):
        fake_vec = [0.5] * 384
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: fake_vec)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            with patch("deepbrain.brain._st_model", mock_model):
                vec = _get_st_embedding("hello")
        assert vec == fake_vec

    def test_import_error_returns_none(self):
        # Reset singleton
        original = bm._st_model
        bm._st_model = None
        try:
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                # Force ImportError by removing module
                import builtins
                real_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "sentence_transformers":
                        raise ImportError("no module")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    vec = _get_st_embedding("test")
            assert vec is None
        finally:
            bm._st_model = original


class TestGetEmbedding:
    """Test the provider fallback chain."""

    def setup_method(self):
        self._orig_available = bm._embed_available
        self._orig_provider = bm._EMBED_PROVIDER

    def teardown_method(self):
        bm._embed_available = self._orig_available
        bm._EMBED_PROVIDER = self._orig_provider

    def test_auto_ollama_success(self):
        bm._embed_available = None
        bm._EMBED_PROVIDER = "auto"
        fake_vec = [0.2] * 768
        with patch("deepbrain.brain._get_ollama_embedding", return_value=fake_vec):
            vec = _get_embedding("test")
        assert vec == fake_vec
        assert bm._embed_available is True

    def test_auto_fallback_to_st(self):
        bm._embed_available = False
        bm._EMBED_PROVIDER = "auto"
        fake_vec = [0.3] * 384
        with patch("deepbrain.brain._get_st_embedding", return_value=fake_vec):
            vec = _get_embedding("test")
        assert vec == fake_vec

    def test_auto_fallback_to_local(self):
        bm._embed_available = False
        bm._EMBED_PROVIDER = "auto"
        bm._USE_LOCAL_EMBED = True
        with patch("deepbrain.brain._get_st_embedding", return_value=None):
            vec = _get_embedding("test text")
        assert vec is not None
        assert len(vec) == 256

    def test_explicit_local_provider(self):
        bm._EMBED_PROVIDER = "local"
        bm._USE_LOCAL_EMBED = True
        vec = _get_embedding("hello")
        assert vec is not None
        assert len(vec) == 256

    def test_no_embedding_when_all_disabled(self):
        bm._embed_available = False
        bm._EMBED_PROVIDER = "auto"
        bm._USE_LOCAL_EMBED = False
        with patch("deepbrain.brain._get_st_embedding", return_value=None):
            vec = _get_embedding("test")
        assert vec is None


class TestVecSerialization:
    """Test blob <-> vector conversion."""

    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = _vec_to_blob(vec)
        restored = _blob_to_vec(blob)
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-6


class TestHybridSearch:
    """Test that hybrid search uses both keyword and vector results."""

    def test_vector_boosts_ranking(self, tmp_path):
        """Entry with embedding match should rank higher than keyword-only."""
        # Disable embedding during learn to control it manually
        bm._embed_available = False
        orig_local = bm._USE_LOCAL_EMBED
        orig_recency = bm._USE_RECENCY_BIAS
        bm._USE_LOCAL_EMBED = False
        bm._USE_RECENCY_BIAS = False  # Disable recency to test pure vector boost

        db = DeepBrain(str(tmp_path / "test.db"))
        id1 = db.learn("Python is a programming language", namespace="test")
        id2 = db.learn("Python snake is found in Asia", namespace="test")
        time.sleep(0.1)  # let async threads finish

        # Manually set embedding only for id1 (close to query vec)
        # id2 has no embedding — so only id1 gets vector RRF boost
        query_vec = [0.0] * 256
        query_vec[0] = 1.0
        match_vec = [0.0] * 256
        match_vec[0] = 0.95
        match_vec[1] = 0.05

        db.conn.execute("UPDATE deepbrain SET embedding=? WHERE id=?", (_vec_to_blob(match_vec), id1))
        db.conn.commit()

        with patch("deepbrain.brain._get_embedding", return_value=query_vec):
            results = db.search("Python", namespace="test", top_k=2)

        assert len(results) == 2
        # id1 should rank first due to vector boost
        assert results[0]["id"] == id1

        bm._USE_LOCAL_EMBED = orig_local
        bm._USE_RECENCY_BIAS = orig_recency
        db.conn.close()

    def test_keyword_only_fallback(self, tmp_path):
        """Without embeddings, keyword search still works."""
        bm._embed_available = False
        orig_local = bm._USE_LOCAL_EMBED
        bm._USE_LOCAL_EMBED = False

        db = DeepBrain(str(tmp_path / "test.db"))
        db.learn("Kubernetes orchestrates containers", namespace="test")
        time.sleep(0.1)

        with patch("deepbrain.brain._get_embedding", return_value=None):
            results = db.search("Kubernetes", namespace="test")

        assert len(results) == 1
        assert "Kubernetes" in results[0]["content"]

        bm._USE_LOCAL_EMBED = orig_local
        db.conn.close()
