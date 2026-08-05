# knowledge/about_zh.md — 中文 gate 校准语料（不只是补充检索内容）

**这是 gate 校准语料，删改需谨慎。** 本文件同时喂给两处，任何改动都会同时
影响两边：(1) **检索索引** —— load_site 经 load_knowledge(dir, "zh") 把每个
`## ` 段落并入索引；(2) **中文首轮守门（gate）的校准集** —— index_builder.py
直接读取本文件，用其内容计算 on/off-topic 分离间隔（margin）。**间隔一旦为负，
gate 会自动保持禁用**，中文问题转为 bypass，只剩 LLM system prompt 兜底，向量
层不再拦截离题提问。

**历史教训（commit 10be374）：** 仅仅改写了本文件里 1 行措辞（没有增删段落），
校准间隔就从正值跌到负值，gate 因此被禁用，且没有任何测试察觉——因为改写前后
语料读起来都还算合理，只有实际校准数字才暴露问题。这是历史，不是现状：中文 gate
当前是否启用由每次构建决定，本文件刻意不写死，请以构建日志为准。

**数字对照（历史，均不代表现状）：** 那次改写前——on-topic 最低 0.492、
off-topic 最高 0.448，正向间隔约 +11%；改写后——约 -0.4%，不再分离。两个数字
并列，是为了证明"精选语料能校准出正间隔"这件事本身成立，坏的是那次改写，不是
这个方法。**当前的阈值与间隔请一律以构建日志的 gate 行为准，不要引用本文件里
写死的任何数字。**

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

## 瞄准辅助玩法
link: pages/cemented-dreams.html
王元辰有没有做过瞄准辅助或者索敌辅助之类的战斗手感调校：在 Cemented Dreams 里，他实现了瞄准辅助这一玩法行为，通过 C++ 玩法组件搭配 Blueprint 接口整合进整体战斗系统，让远程目标切换更顺手。

## 关键战斗参数交给设计师调整
link: pages/cemented-dreams.html
Cemented Dreams 用 Unreal Engine 5、C++ 与 Blueprint 开发，王元辰把关键战斗参数通过 Blueprint 开放给设计师，这样调整移动手感和战斗节奏时不需要重新编译底层 C++ 代码。

## Hive 关卡怎么设计的
link: pages/cemented-dreams.html
Cemented Dreams 里那座不断生长的 Hive 关卡是王元辰设计的：一座机械结构在建筑内部生长，缆索的接触点会生成六边形平台，撕开的入口则形成位移路径与战斗场地。

## 靠玩法测试打磨系统好不好玩
link: pages/game-design-workshop.html
衡量一套机制受不受玩家欢迎，王元辰在 CTIN 488 游戏设计工作坊里靠反复的玩法测试：无论是 Up the River 变体还是 Too Bird To Handle 这类原型，每周都会根据玩家在桌面前的实际反应重写规则、调整节奏。

## 数值平衡靠模拟跑出来的
link: pages/game-design-workshop.html
CTIN 488 期末项目做数值平衡时，王元辰所在的团队没有靠拍脑袋，而是跑了 20 万次试验的模拟，再据此调整牌库构成与胜利条件，减少意外获胜的情况。

## 两个原型入选了课堂展示
link: pages/game-design-workshop.html
王元辰在 CTIN 488 做的三个工作室项目里，Up the River 的变体和以鸟类求偶为题材的 Too Bird To Handle 竞价游戏这两个原型都入选了课堂展示。

## 团队项目里当过制作人
link: pages/game-design-workshop.html
在 CTIN 488 的期末团队项目里，王元辰的职责是制作人：帮助锁定核心循环、推动规则书清理、主导以模拟为依据的平衡检查，并协助了最终幻灯片与展示视频的准备。

## Codebreaker 的玩法架构
link: pages/codebreaker.html
Codebreaker 是王元辰用 Unity 6 和 C# 做的一款系统驱动的动作游戏，他是主要玩法工程师，实现了核心战斗机制与玩家互动逻辑，并和制作、音频团队协作交付里程碑版本。

## 视锥剔除和层级剔除怎么实现
link: pages/prime-engine.html
为了减少渲染阶段不必要的开销，王元辰在 Prime Engine 里先用包围盒实现了视锥剔除，再扩展成层级式 BVH：静态场景用一次性构建的中位数划分树，运动物体则用按 Morton 编码每帧重建的动态树。

## 碰撞之后顺着表面滑动的物理
link: pages/prime-engine.html
角色撞到障碍物之后不会直接卡住：王元辰在 Prime Engine 里实现的碰撞系统会把移动向量投影到碰撞平面上，去除指向表面的分量，剩余部分则让物体沿障碍物自然滑动，同时还处理了带重力的下落与落地检测。

## 骨骼动画混合与调试可视化工具
link: pages/prime-engine.html
Prime Engine 的动画状态机能一次驱动多个动作片段：王元辰构建了全身混合、只覆盖某一关节范围的局部混合，以及叠加在现有姿势之上的叠加混合三种模式，还做了包围体和动画状态的调试可视化工具方便验证效果。

## 除了引擎脚本还会哪些编程语言
link: pages/skills.html
除了 C++ 和 C# 这类引擎里常用的语言，王元辰的技能页面还列出了 Python、R、C、MATLAB、Java、SQL，以及 CSS、HTML、PHP 和 Onshape CAD、Docker、GitHub 等工具。

## 本科阶段修过的计算机课程
link: pages/education.html
王元辰在罗切斯特大学读本科时辅修计算机科学，修过离散数学、数据结构与算法、计算与形式系统等课程，是他后来做引擎与玩法编程之前的理论基础。

## 人工智能和机器学习方面的课程
link: pages/education.html
王元辰在罗切斯特大学本科阶段修过人工智能课程，读哈佛生物医学信息学硕士时又修了机器学习和应用贝叶斯分析，这些课程在他后来的游戏引擎与渲染课程之外，补上了偏理论和统计建模的一块。

## 聊天助手的检索与判断都在服务端
link: pages/chat-agent.html
本站聊天助手的检索语料是打包进服务端函数的静态 JSON 文件，最近邻搜索完全发生在服务端，而不是访客的浏览器里；浏览器只负责把问题本身发出去，查询转向量、跑相关性门控这些步骤同样经过这个小的 Python 函数完成。

## 中英文各自用什么把关模型
link: pages/chat-agent.html
王元辰给聊天助手的两种语言各配了一个把关模型：英文用 MiniLM，中文用量化过的 bge-small-zh，两者都有各自调校的阈值，不会共用同一套判断标准。

## 后端连不上时聊天功能会怎样
link: pages/chat-agent.html
如果聊天助手的后端函数连不上，功能不会直接失效：王元辰设计了一个需经同意的兜底方案，改用浏览器内的模型在本地完成检索，而不是让访客碰壁。

## 夹带姓名的指令能不能被识破
link: pages/chat-agent.html
聊天助手的门控会先把提问里对王元辰姓名的提及（不管是英文 YC / Wang 还是中文王元辰）归一化成一个标记，再判断剩下的文本：只是捎带姓名却夹带别的指令的提问会被剥离出真实诉求并拒绝。

## 每轮对话是不是能查到记录
link: pages/chat-agent.html
聊天助手的每一轮对话都带着请求 ID 和经过哈希处理的会话 ID，通过门控的轮次里，服务端会记录门控判定结果、检索到的段落和大模型的输入输出，方便事后排查，但不会存原始查询向量。

## 回答会不会照访客身份调整
link: pages/projects.html
本站这个聊天助手在项目列表里被称作角色感知 RAG：根据访客选择的身份不同，它回答时强调的重点也会跟着变化，并不是每个人问到的内容都完全相同。

## 自动微分工具里怎么算导数
link: pages/automatic-differentiation.html
在那个五人团队做的自动微分工具里，王元辰实现了前向模式引擎的核心部分：通过运算符重载让对偶数在加减乘除等复合表达式里正确传播，从而在运行时算出精确导数，而不用手动套链式法则。

## 网站里能直接玩的小工具
link: pages/toolbox.html
在游戏项目之外，王元辰还在网站的工具箱页面里做了两个可以现场试用的小工具：一个词云生成器，可以调整最大词数和缩放方式；一个二维码生成器，可以调整输出尺寸。

## 会讲几种语言
link: pages/skills.html
王元辰的技能页面列出了他的语言能力：中文是母语，英语熟练，日语和德语都是入门水平。

## 做研究时用过哪些实验技术
link: pages/skills.html
王元辰在做神经科学研究时接触过 DPI 眼动追踪和 VICON 动作捕捉这类实验设备，也做过人机交互、数据可视化与应用机器学习，后来这些方法延伸到了迭代式玩法测试、焦点小组和可用性研究上。

## 学位是读完了还是还在读
link: pages/education.html
王元辰在哈佛医学院和罗切斯特大学的学位都已经完成，目前只有南加州大学的计算机科学硕士（游戏开发方向）还在读，要到 2027 年 5 月才结束。

## 论文都发在哪些期刊上
link: pages/publications.html
王元辰挂名的论文分散在好几个期刊和会议上：NeuroImage、Cerebral Cortex、PLOS ONE 的正式期刊文章，Physiology 的会议摘要，还有一篇 PsyArXiv 预印本和 FBB 2020 的会议论文集。

## 医学影像方面的机器学习论文
link: pages/publications.html
除了大脑相关的研究，王元辰还发表过一篇把机器学习预处理方法 ps-KDE 用在胸部 X 光影像语义分割上的论文，发表在 PLOS ONE。

## Gyrotris 在哪能玩到
link: pages/gyrotris.html
王元辰独自做的解谜游戏 Gyrotris 已经发布在 itch.io 上，任何人都可以直接上去试玩这款他第一款完整发布的作品。

## 只能看一个项目该看哪个
link: pages/projects.html
预算有限只挑一个项目看的话，Cemented Dreams 是王元辰身兼设计与工程职责最重的一个：他同时担任战斗设计师、玩法工程师和关卡设计师，在项目页的游戏开发项目分类里也排在第一位。

## 挂过的职位头衔都有哪些
link: pages/projects.html
在设计和程序之外，王元辰在项目列表里还挂过 CAD 建模、Python 开发者、QA 负责人、独立游戏开发者、系统设计师这些不那么常见的职位。

## 时间不多该重点看哪些方向
link: pages/projects.html
如果时间有限，王元辰的作品跨度可以从三个方向感受：战斗设计与工程并重的 Cemented Dreams、完全独立完成的解谜游戏 Gyrotris，以及体现建模精度的 Aegis Sword 道具复刻。
