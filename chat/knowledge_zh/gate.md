# 中文守门语料（zh gate corpus）— 目前为空，功能未启用

This file intentionally has no `## ` sections yet, so the Chinese first-pass
gate stays disabled and CJK questions bypass the gate (they're still guarded
by the LLM system prompt, layer 3).

To enable it later:
1. Write Chinese summary sections here (same format as knowledge/*.md:
   each `## 标题` block is one passage, optional `link:` line). Cover the
   topics visitors ask about — bio, education, combat design, engine work,
   skills, publications, the AI chat project, resume highlights.
2. Fetch the gate model: Xenova/bge-small-zh-v1.5 into
   chat/models/Xenova/bge-small-zh-v1.5 (build_package.py will do this
   automatically once sections exist here).
3. Rebuild: `python scripts/build_index.py --model e5`. The build calibrates
   the zh gate on the zh query sets in gate_calibration.py and only enables
   it if the on/off-topic distributions actually separate.
4. Repackage + redeploy the function (gate_model_zh/ + gate_vectors.json).
