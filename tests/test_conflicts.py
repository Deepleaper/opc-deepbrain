"""Tests for DeepBrain.detect_conflicts and resolve_conflict."""

import json
import tempfile
import time

import pytest

from deepbrain.brain import DeepBrain


@pytest.fixture
def brain():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = DeepBrain(db_path=db_path)
    yield db
    db.conn.close()


# ── detect_conflicts ──────────────────────────────────────────────────────────


class TestDetectConflicts:
    def test_returns_empty_for_unknown_id(self, brain):
        assert brain.detect_conflicts("nonexistent-id") == []

    def test_returns_empty_when_no_candidates(self, brain):
        eid = brain.learn("Python is a programming language", claim_type="observation")
        assert brain.detect_conflicts(eid) == []

    def test_contraction_pair_triggers_conflict(self, brain):
        # "won't" vs "will" is an explicit contraction pair
        a = brain.learn(
            "The service will scale automatically under load",
            claim_type="observation",
        )
        b = brain.learn(
            "The service won't scale automatically under load",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        ids = [c["id"] for c in conflicts]
        assert a in ids

    def test_keyword_scoped_negation_triggers_conflict(self, brain):
        # "not" near shared keyword "cache" vs no negation
        a = brain.learn(
            "Redis cache stores session tokens efficiently",
            claim_type="observation",
        )
        b = brain.learn(
            "Redis does not cache session tokens by default",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        ids = [c["id"] for c in conflicts]
        assert a in ids

    def test_fact_entries_high_keyword_overlap_conflict(self, brain):
        a = brain.learn(
            "The Python Django framework uses MVT architecture pattern",
            claim_type="fact",
        )
        b = brain.learn(
            "The Python Django framework uses MVC architecture pattern",
            claim_type="fact",
        )
        conflicts = brain.detect_conflicts(b)
        ids = [c["id"] for c in conflicts]
        assert a in ids

    def test_fact_entries_identical_content_no_conflict(self, brain):
        text = "Python uses indentation for code blocks"
        a = brain.learn(text, claim_type="fact")
        b = brain.learn(text, claim_type="fact")
        conflicts = brain.detect_conflicts(b)
        ids = [c["id"] for c in conflicts]
        assert a not in ids

    def test_different_namespaces_no_conflict(self, brain):
        a = brain.learn(
            "The service won't scale automatically under load",
            namespace="ns-a",
            claim_type="observation",
        )
        b = brain.learn(
            "The service will scale automatically under load",
            namespace="ns-b",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        assert all(c["id"] != a for c in conflicts)

    def test_superseded_entry_not_returned(self, brain):
        a = brain.learn(
            "The service won't scale automatically under load",
            claim_type="observation",
        )
        # Mark a as superseded
        brain.conn.execute("UPDATE deepbrain SET status='superseded' WHERE id=?", (a,))
        brain.conn.commit()

        b = brain.learn(
            "The service will scale automatically under load",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        assert all(c["id"] != a for c in conflicts)

    def test_conflict_dict_excludes_embedding(self, brain):
        a = brain.learn(
            "The service won't scale automatically under load",
            claim_type="observation",
        )
        b = brain.learn(
            "The service will scale automatically under load",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        for c in conflicts:
            assert "embedding" not in c

    def test_insufficient_keyword_overlap_no_conflict(self, brain):
        # Only one shared keyword — below the minimum of 2
        a = brain.learn("Cats are friendly pets", claim_type="observation")
        b = brain.learn("Python is a programming language", claim_type="observation")
        conflicts = brain.detect_conflicts(b)
        assert all(c["id"] != a for c in conflicts)

    def test_observation_high_overlap_no_fact_conflict(self, brain):
        # Pass 3 only fires for claim_type='fact'; observations with shared keywords but
        # no negation contradiction should not conflict.
        a = brain.learn(
            "The Python Django framework is a web development tool",
            claim_type="observation",
        )
        b = brain.learn(
            "The Python Django framework is a backend development tool",
            claim_type="observation",
        )
        conflicts = brain.detect_conflicts(b)
        assert all(c["id"] != a for c in conflicts)


# ── resolve_conflict ──────────────────────────────────────────────────────────


class TestResolveConflict:
    def test_discard_status_becomes_superseded(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        brain.resolve_conflict(keep, discard)
        row = brain.get(discard)
        assert row["status"] == "superseded"

    def test_keep_status_remains_active(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        brain.resolve_conflict(keep, discard)
        row = brain.get(keep)
        assert row["status"] == "active"

    def test_resolved_conflicts_appended_to_keep_metadata(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        brain.resolve_conflict(keep, discard)
        meta = brain.get(keep)["metadata"]
        assert discard in meta.get("resolved_conflicts", [])

    def test_discard_removed_from_conflict_with(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        # Seed conflict_with manually
        brain.conn.execute(
            "UPDATE deepbrain SET metadata=? WHERE id=?",
            (json.dumps({"conflict_with": [discard]}), keep),
        )
        brain.conn.commit()
        brain.resolve_conflict(keep, discard)
        meta = brain.get(keep)["metadata"]
        assert discard not in meta.get("conflict_with", [])

    def test_same_id_raises_value_error(self, brain):
        eid = brain.learn("Any content", claim_type="observation")
        with pytest.raises(ValueError):
            brain.resolve_conflict(eid, eid)

    def test_idempotent_resolve(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        brain.resolve_conflict(keep, discard)
        brain.resolve_conflict(keep, discard)  # second call must not duplicate
        meta = brain.get(keep)["metadata"]
        assert meta.get("resolved_conflicts", []).count(discard) == 1

    def test_resolve_nonexistent_keep_does_not_raise(self, brain):
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        # Should not raise even when keep_id doesn't exist
        brain.resolve_conflict("nonexistent-id", discard)
        row = brain.get(discard)
        assert row["status"] == "superseded"

    def test_valid_until_set_on_discard(self, brain):
        keep = brain.learn("Python is fast enough for production", claim_type="fact")
        discard = brain.learn("Python is not fast enough for production", claim_type="fact")
        brain.resolve_conflict(keep, discard)
        row = brain.get(discard)
        assert row["valid_until"] is not None
