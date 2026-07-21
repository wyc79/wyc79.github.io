"""Calibrates the widget's off-topic gate at build time.

Different embedding models need different gate statistics: MiniLM's raw
top score separates on-/off-topic cleanly, but e5-family models compress
all cosines into a ~0.7-0.9 band where absolute thresholds overlap. So the
calibration scores canonical on-/off-topic query sets, evaluates several
candidate statistics, and picks the one with the widest relative margin:

  top      — max similarity (works for MiniLM)
  contrast — max minus corpus mean (peak height above background)
  zscore   — (max - mean) / std   (scale-free peak sharpness)

The chosen statistic + threshold ship in index.json (gate_stat,
gate_threshold); the widget and tests/test_gate.py implement the same three
statistics — keep them in sync.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

ON_TOPIC = [
    "resume highlights",
    "what did he study",
    "does he know Unity?",
    "publications research",
    "combat design work",
    "engine programming and C++ work",
    "who is Yuanchen Wang?",
    "his machine learning background",
]

OFF_TOPIC = [
    "tell me a joke",
    "write me a python fibonacci function",
    "translate this to french: hello",
    "what's the weather today",
    "write my homework essay",
    "best restaurants nearby",
    "who won the world cup",
    "write a poem about love",
]

# Chinese sets: used to calibrate the zh gate (bge-zh vs the hand-written
# knowledge/about_zh.md corpus), and appended for multilingual presets.
ON_TOPIC_ZH = [
    "他做过哪些战斗设计工作",
    "介绍一下他的游戏引擎开发经验",
    "他的教育背景是什么",
    "他会用Unity和虚幻引擎吗",
    "他的简历亮点有哪些",
    "他发表过什么论文",
    "他做过什么AI或大模型项目",
    "介绍一下YC这个人",
]
OFF_TOPIC_ZH = [
    "给我讲个笑话",
    "今天天气怎么样",
    "帮我写作业",
    "帮我写一段Python代码",
    "把这句话翻译成英文",
    "谁赢了世界杯",
    "写一首关于爱情的诗",
    "附近有什么好吃的餐厅",
]

GATE_STATS = ("top", "contrast", "zscore")


def stat_value(scores: np.ndarray, kind: str) -> float:
    top = float(np.max(scores))
    if kind == "top":
        return top
    mean = float(np.mean(scores))
    if kind == "contrast":
        return top - mean
    if kind == "zscore":
        return (top - mean) / (float(np.std(scores)) + 1e-6)
    raise ValueError(f"unknown gate stat {kind!r}")


def compute_gate(
    embedder,
    matrix: np.ndarray,
    multilingual: bool = False,
    on: list | None = None,
    off: list | None = None,
) -> dict:
    on = on if on is not None else ON_TOPIC + (ON_TOPIC_ZH if multilingual else [])
    off = off if off is not None else OFF_TOPIC + (OFF_TOPIC_ZH if multilingual else [])
    on_scores = [matrix @ embedder.embed_query(q) for q in on]
    off_scores = [matrix @ embedder.embed_query(q) for q in off]

    best = None
    for kind in GATE_STATS:
        on_vals = [stat_value(s, kind) for s in on_scores]
        off_vals = [stat_value(s, kind) for s in off_scores]
        lo, hi = max(off_vals), min(on_vals)
        spread = max(on_vals + off_vals) - min(on_vals + off_vals) + 1e-9
        margin = (hi - lo) / spread  # relative, comparable across stats
        logger.info(
            "gate calibration [%s]: off-topic max %.3f | on-topic min %.3f | rel margin %.1f%%",
            kind, lo, hi, margin * 100,
        )
        if best is None or margin > best["margin"]:
            best = {"stat": kind, "lo": lo, "hi": hi, "margin": margin}

    if best["margin"] <= 0:
        logger.warning(
            "gate calibration: no statistic separates the distributions (best: %s); "
            "gating just above the off-topic max — expect some false refusals",
            best["stat"],
        )
        threshold = round(best["lo"] * 1.02 + 1e-4, 4)
    else:
        threshold = round((best["lo"] + best["hi"]) / 2, 4)

    logger.info("gate calibration: chose stat=%s threshold=%.4f", best["stat"], threshold)
    return {"stat": best["stat"], "threshold": threshold, "margin": round(best["margin"], 4)}
