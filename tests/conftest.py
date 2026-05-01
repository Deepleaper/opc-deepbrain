import deepbrain.brain as _bm

# Disable all embedding in test environment:
# - _embed_available=False skips Ollama network calls (prevents Windows AV crash)
# - _USE_LOCAL_EMBED=False means _embed() returns None and skips DB commit,
#   avoiding a race with tests that do raw conn.execute/commit directly.
_bm._embed_available = False
_bm._USE_LOCAL_EMBED = False
