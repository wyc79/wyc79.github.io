"""RAG pipeline for the portfolio chat agent (build-time write path).

Normal-mode retrieval happens server-side (Task 29 Part 1); light mode and
degraded mode do it client-side in scripts/chat-widget.js against the
data/chunks_{model_preset}.json / data/chunks_en_minilm.json this package
produces (see config.py's Settings for the full file layout).
"""
