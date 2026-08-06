# Yuanchen Wang — Portfolio / 王元辰 · 作品集

Live site: [wyc79.github.io](https://wyc79.github.io)

Personal portfolio of **Yuanchen Wang** (王元辰) — game developer and USC MSCS (Game Development) student. Static site on GitHub Pages: bilingual EN/ZH, light/dark theme, project pages, and an optional RAG chat agent.

王元辰的个人作品集网站：游戏开发者，南加州大学计算机科学硕士（游戏开发方向）。托管于 GitHub Pages，支持中英文切换、明暗主题、项目详情页，以及可选的 RAG 问答助手。

---

## English

### What’s here

| Area | Description |
|------|-------------|
| **Landing** (`index.html`) | Interactive sphere menu into Projects, Skills, Education, Publications, Agents, Toolbox |
| **Projects** | Game work (Cemented Dreams, Nothing Can Go Wrong, Code Breaker, Gyrotris) plus engine / rendering / CAD / AD tooling |
| **Skills / Education / Publications** | Background, coursework, and research |
| **Agents** | Notes on agent skills used with this site |
| **Toolbox** | In-page word cloud and QR code generators |
| **Chat** (`chat/`) | Role-aware RAG assistant over site content (see [`chat/README.md`](chat/README.md)) |

### Stack

- Static **HTML / CSS / vanilla JS** (no site framework)
- Client scripts: theme, i18n, page header, landing canvas/menu, chat widget
- Optional chat backend: Tencent SCF + DeepSeek (API key never in the repo)
- Chat index pipeline: Python package under `chat/`

### Preview locally

Browsers block `fetch()` on `file://`, so serve over HTTP from the repo root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`.

### Chat index (after editing site content)

```bash
cd chat
pip install -e ".[dev]"
python build.py
```

Full pipeline, eval, and deploy notes: [`chat/README.md`](chat/README.md).

### License

[MIT](LICENSE) © 2026 wyc79

---

## 中文

### 内容概览

| 区域 | 说明 |
|------|------|
| **首页**（`index.html`） | 交互式球体菜单，进入项目、技能、教育、论文、智能体、工具箱 |
| **项目** | 游戏作品（Cemented Dreams、Nothing Can Go Wrong、Code Breaker、Gyrotris）及引擎 / 渲染 / CAD / 自动微分等工程项目 |
| **技能 / 教育 / 论文** | 背景、课程与科研经历 |
| **智能体** | 与本站相关的 Agent 技能说明 |
| **工具箱** | 站内词云与二维码生成器 |
| **问答**（`chat/`） | 基于站内内容的角色感知 RAG 助手（详见 [`chat/README.md`](chat/README.md)） |

### 技术栈

- 静态 **HTML / CSS / 原生 JS**（站点无前端框架）
- 客户端脚本：主题、中英文、页头、首页画布与菜单、聊天组件
- 可选聊天后端：腾讯云 SCF + DeepSeek（密钥不进仓库）
- 索引构建：`chat/` 下的 Python 包

### 本地预览

浏览器会拦截 `file://` 下的 `fetch()`，请从仓库根目录起一个 HTTP 服务：

```bash
python -m http.server 8000
```

然后打开 `http://localhost:8000`。

### 更新聊天索引（改站内内容后）

```bash
cd chat
pip install -e ".[dev]"
python build.py
```

完整流水线、评测与部署说明见 [`chat/README.md`](chat/README.md)。

### 许可证

[MIT](LICENSE) © 2026 wyc79
