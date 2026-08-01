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

# Task 25: grown from 8/8 to 18/16 (en) and 18/16 (zh). With 8 queries a
# side, margin = min(on) - max(off) is set by whichever SINGLE query is least
# favorable -- a bad draw on either side determines the whole calibration.
# More entries make both bounds a genuine distribution statistic instead of
# one query's score. Every addition was individually verified (a standalone
# script, not committed) to score comfortably on its intended side of the
# then-current threshold before being added here, and checked for
# disjointness (character n-gram / longest-common-substring, not word
# tokenization -- the latter is blind to Chinese) against golden.jsonl's 120
# cases, roles.json's starters, test_gate.py's OFF_TOPIC, and about_en.md /
# about_zh.md's own sentences, so no addition scores artificially high (or
# low) by echoing something it will later be evaluated against or scored
# from. Two early candidates ("Harvard biomedical informatics background",
# "frustum culling implementation") were caught restating corpus/golden text
# and replaced before landing here.
#
# Deliberately NOT added: adjacent off-topic probes (e.g. "what's a good
# build order in StarCraft 2", "how much do combat designers get paid" --
# see golden.jsonl's neg-en-05..08/neg-zh-05..08, which measure this gap
# without feeding it back into calibration). Measured directly against this
# corpus before deciding: 8 candidate adjacent en probes scored 0.29-0.41,
# every one above min(ON_TOPIC) (0.208); adding them to OFF_TOPIC drove the
# en margin from +2.5% to -33.7% (stat flips to "contrast", threshold jumps
# to 0.315 -- well above genuine on-topic queries). The zh case is the same
# shape (-34.4%). Adjacent questions are closer in embedding space to real
# on-topic questions than to the easy off-topic ones a single-vector gate can
# actually separate; folding them into calibration wouldn't teach the gate to
# refuse them, it would just raise the threshold until it refuses real
# visitors too. This is a measured limitation, not an oversight -- see
# eval/KNOWN_ISSUES.md and the Task 25 report.
ON_TOPIC = [
    "resume highlights",
    "what did he study",
    "does he know Unity?",
    "publications research",
    "combat design work",
    "engine programming and C++ work",
    "who is Yuanchen Wang?",
    "his machine learning background",
    "shipped games list",
    "which engines has he used",
    "USC game development program",
    "his pre-USC graduate degree",
    "grapple traversal mechanic",
    "hierarchical culling on Prime Engine",
    "animation blending tools",
    "chat widget architecture",
    "how the gate blocks prompt injection",
    "solo developed game",
]

# All additions are "easy" (unambiguously unrelated), matching the original
# 8 -- see the module-level comment above ON_TOPIC for why adjacent probes
# are deliberately excluded from this list rather than merely under-
# represented.
OFF_TOPIC = [
    "tell me a joke",
    "write me a python fibonacci function",
    "translate this to french: hello",
    "what's the weather today",
    "write my homework essay",
    "best restaurants nearby",
    "who won the world cup",
    "write a poem about love",
    "what's the tallest mountain in the world",
    "how do I bake chocolate chip cookies",
    "recommend a good sci-fi movie to watch tonight",
    "what's the exchange rate between USD and EUR",
    "what's a good workout routine for beginners",
    "how do you say thank you in Spanish",
    "what time zone is Tokyo in",
    "how long does it take to boil an egg",
]

# Chinese sets: used to calibrate the zh gate (bge-zh vs the hand-written
# knowledge/about_zh.md corpus), and appended for multilingual presets.
# Grown 8->18/8->16 alongside the English sets above (Task 25); every zh
# addition was authored natively in Chinese, never translated from the en
# additions, per this project's standing zh-authoring rule (Task 23).
ON_TOPIC_ZH = [
    "他做过哪些战斗设计工作",
    "介绍一下他的游戏引擎开发经验",
    "他的教育背景是什么",
    "他会用Unity和虚幻引擎吗",
    "他的简历亮点有哪些",
    "他发表过什么论文",
    "他做过什么AI或大模型项目",
    "介绍一下YC这个人",
    "他用什么引擎做战斗系统",
    "他现在读的硕士是什么专业",
    "他有没有发表过科研论文",
    "自动微分工具是他做的吗",
    "他做过哪些独立游戏",
    "他掌握哪些编程语言",
    "他有没有做过关卡设计",
    "这个聊天助手是他自己写的吗",
    "他有没有做过三维渲染或着色器相关的项目",
    "他平时用什么工具做开发",
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
    "世界上最高的山是哪一座",
    "怎么做红烧肉",
    "人民币兑美元今天的汇率是多少",
    "推荐一部好看的科幻电影",
    "新手健身应该怎么练",
    "西班牙语的谢谢怎么说",
    "东京是哪个时区",
    "煮一个鸡蛋要多久",
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
    # lo/hi (off-topic max / on-topic min, in the chosen stat's units) ride
    # along so a caller that rejects a non-separating margin (index_builder's
    # en floor, task 20) can name both distribution bounds in its error, not
    # just the derived margin number.
    return {
        "stat": best["stat"],
        "threshold": threshold,
        "margin": round(best["margin"], 4),
        "lo": round(best["lo"], 4),
        "hi": round(best["hi"], 4),
    }
