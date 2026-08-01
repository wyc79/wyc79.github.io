# knowledge/about_zh.md — 中文 gate 校准语料（不只是补充检索内容）

**这是 gate 校准语料，删改需谨慎。** 本文件同时喂给两处，任何改动都会同时
影响两边：(1) **检索索引** —— load_site 经 load_knowledge(dir, "zh") 把每个
`## ` 段落并入索引；(2) **中文首轮守门（gate）的校准集** —— index_builder.py
直接读取本文件，用其内容计算 on/off-topic 分离间隔（margin）。**间隔一旦为负，
gate 会自动保持禁用**，中文问题转为 bypass，只剩 LLM system prompt 兜底，向量
层不再拦截离题提问。

**历史教训（commit 10be374）：** 仅仅改写了本文件里 1 行措辞（没有增删段落），
校准间隔就从正值跌到负值，gate 自此持续禁用，且没有任何测试察觉——因为改写前后
语料读起来都还算合理，只有实际校准数字才暴露问题。

**数字对照（历史 vs 当前，历史数字已不代表现状）：** 改写前——on-topic 最低
0.492、off-topic 最高 0.448，正向间隔约 +11%；改写后即当前——约 -0.4%，不再
分离，gate 保持禁用。两个数字并列，是为了证明"精选语料能校准出正间隔"这件事
本身成立，坏的是那次改写，不是这个方法——这正是后续任务继续扩充语料的理由。

**编辑规则：** 每段要短、只讲一个主题、按访客提问的措辞起头（小模型在长段落上
会把向量"平均糊化"，导致离题提问也命中）；中文一律称「王元辰」，不用「YC」。
**编辑后必须重新构建并检查构建日志**（`python scripts/build_index.py --model
e5`，看 `zh gate: enabled ... margin ...` 还是 `skipped`）——不要凭旧数字判断
是否健康，一切以最近一次构建日志的 margin 为准。

与 about_en.md 同处 chat/knowledge/ 目录，按语言后缀区分：load_knowledge(dir,
"zh") 只读 *_zh.md，中文语料不会与英文混在一起。

## 介绍一下王元辰
link: index.html
介绍一下王元辰（Yuanchen Wang）：他是一名游戏开发者，正在南加州大学攻读计算机科学硕士（游戏开发方向），做过战斗设计、玩法编程和游戏引擎开发。

## 介绍一下王元辰这个人
link: index.html
介绍一下王元辰这个人：王元辰是一名游戏开发者，做游戏设计、玩法编程和引擎开发，也做过科研。

## 王元辰的个人简介
link: index.html
王元辰个人简介：一名游戏开发者，正在读游戏开发方向的计算机硕士，擅长战斗设计、玩法编程与引擎开发。

## 王元辰是谁
link: index.html
王元辰是谁：一名游戏开发者，也是本作品集网站的作者。

## 关于他这个人
link: index.html
关于王元辰这个人：他是做游戏的开发者，擅长战斗设计、玩法编程和引擎开发，之前还做过科研。

## 简历亮点
link: pages/projects.html
王元辰的简历亮点：游戏战斗设计、玩法编程、关卡设计，以及 C++ 游戏引擎开发经验。

## 教育背景与学历
link: pages/education.html
王元辰的教育背景：南加州大学计算机科学硕士（游戏开发）、哈佛医学院生物医学信息学硕士、罗切斯特大学脑与认知科学学士与心理学学士。

## 就读学校与专业
link: pages/education.html
王元辰在南加州大学读游戏开发方向的计算机硕士，本科在罗切斯特大学读脑与认知科学与心理学。

## 他的学历
link: pages/education.html
王元辰的学历：计算机科学硕士（游戏开发方向）、生物医学信息学硕士，以及脑与认知科学学士和心理学学士。

## 战斗设计工作
link: pages/cemented-dreams.html
王元辰在 Cemented Dreams 里做战斗设计，设计并实现了钩爪位移与滑行战斗等核心战斗移动机制。

## 玩法与关卡设计
link: pages/cemented-dreams.html
王元辰负责玩法编程与关卡设计，用 Unreal Engine 的 C++ 与 Blueprint 构建移动与战斗系统。

## 游戏引擎开发经验
link: pages/prime-engine.html
王元辰在 Prime Engine 做 C++ 引擎开发，实现了视锥剔除、BVH 剔除、碰撞与移动滑行物理、骨骼动画混合。

## 会用哪些游戏引擎
link: pages/skills.html
王元辰会用 Unreal Engine 5、Unity 和 Godot 这三款游戏引擎做游戏开发。

## 独立开发的游戏
link: pages/gyrotris.html
王元辰独自开发了解谜游戏 Gyrotris，用 Godot 完成设计、编程与像素美术，是他第一款完整发布的作品。

## game jam 项目
link: pages/nothing-can-go-wrong.html
王元辰在 game jam 中担任主程序，做了 Nothing Can Go Wrong 与 Codebreaker 的核心玩法系统。

## 实时渲染与着色器项目
link: pages/3d-rendering.html
王元辰在 Unity URP 里做实时渲染，用 HLSL 写程序化熔岩与裂缝着色器，实现小行星撞击特效。

## CAD 建模项目
link: pages/aegis-sword.html
王元辰用 Onshape CAD 参数化建模，复刻了 Xenoblade 2 的 Aegis Sword 道具。

## 自动微分工具项目
link: pages/automatic-differentiation.html
王元辰在一个五人团队里开发前向模式自动微分工具，负责核心引擎与测试质量保证。

## 发表的论文与研究
link: pages/publications.html
王元辰发表过神经科学与医学影像方向的论文，涉及 fMRI、催产素、语义分割等主题。

## 科研背景
link: pages/publications.html
进入游戏行业前，王元辰做过神经科学、心理学与生物医学信息学研究。

## AI 与大模型项目
link: index.html
王元辰用 Python 为本作品集网站做了一个 AI 智能问答助手，用检索增强生成（RAG）技术回答访客关于他项目与经历的问题。

## 他会什么、都会什么、擅长什么
link: pages/skills.html
王元辰会什么、都会什么、擅长什么：他会用 Unreal Engine 5、Unity、Godot 做游戏开发，会 C++、C#、Python 编程，擅长战斗设计、玩法编程和关卡设计，也做过机器学习与科研。
