// Portfolio chat agent — normally a thin client over the backend's own
// retrieval (Task 29); falls back to client-side RAG over the static index
// when there is no backend (light mode) or it's unreachable (degraded mode).
//
// Architecture (see chat/README.md):
//   chat/data/meta.json         small metadata sidecar (build-time, Python) —
//                              gate_threshold/gate_stat/gate_remote/model/
//                              query_prefix/chunks_file. Fetched on every load.
//   chat/data/{meta.chunks_file}  precomputed chunk vectors for the CONFIGURED
//                              model (chunks_e5.json in production). Fetched
//                              ONLY when the vendored in-browser model matches
//                              meta.model (light mode / a `--model minilm`
//                              build) — normal mode's retrieval happens
//                              server-side, and degraded mode never uses this
//                              file (see chat/data/chunks_en_minilm.json below).
//   chat/data/gate_en_minilm.json    English off-topic gate vectors (MiniLM,
//                              knowledge/about_en.md). Fetched only in
//                              degraded mode (backend unreachable).
//   chat/data/chunks_en_minilm.json  degraded mode's own retrieval corpus:
//                              the SAME 192 English chunks as chunks_e5.json's
//                              lang=="en" entries, re-embedded with MiniLM so
//                              the self-hosted model can score real chunk
//                              records — resolved by id, never by borrowing a
//                              position from an unrelated array (Task 29
//                              Part 2; this is the fix for the degraded-mode
//                              source links Task 24 broke). Fetched only in
//                              degraded mode.
//   chat/models/           self-hosted MiniLM ONNX — the SAME weights the
//                          build pipeline used, so query and document vectors
//                          share one embedding space
//   scripts/vendor/        self-hosted transformers.js runtime + ORT WASM
//   WORKER_URL             optional backend for LLM answers; leave empty for
//                          retrieval-only demo mode (no API key). Server-side
//                          retrieval is the Tencent SCF function
//                          (chat/functions/tencent) -- the only backend this
//                          widget speaks to.
//
// Everything the widget logs (every user input, retrieval result and answer)
// goes to console.debug, to Google Analytics when present, and — once the
// function is deployed — to the function's server-side log.
(function () {
  'use strict';

  // ── Config ────────────────────────────────────────────────────────────
  // Backend base URL (Tencent SCF 函数URL). Set it here
  // after deploying — see chat/functions/tencent/DEPLOY.md. A page may also
  // predefine window.YC_CHAT_WORKER_URL before this script loads.
  var WORKER_URL = (window.YC_CHAT_WORKER_URL || 'https://1302480548-79ajhb3iyj.ap-guangzhou.tencentscf.com').replace(/\/+$/, '');
  var TOP_K = 4;
  var MIN_SCORE = 0.18;  // per-source display floor
  // Off-topic gate: statistic + threshold are calibrated per embedding model
  // at build time and shipped in meta.json (gate_stat, gate_threshold) —
  // see chat/src/portfolio_rag/gate_calibration.py, mirrored in
  // chat/tests/test_gate.py. These constants are only the fallback for old
  // builds: raw top-score >= 0.22 (the MiniLM calibration).
  var OFFTOPIC_GATE = 0.22;

  // Every backend call gets a deadline. Without one, a connection that is
  // accepted and then goes silent leaves the thinking bubble animating
  // forever: every fallback in this widget (degraded mode, the error message,
  // the offline-search offer) runs only when a fetch actually settles.
  var EMBED_TIMEOUT_MS = 20000;
  var CHAT_TIMEOUT_MS = 65000; // just past index.py's own call_llm urlopen(timeout=60)

  function fetchWithTimeout(url, opts, ms) {
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, ms);
    var merged = {};
    for (var k in opts) { if (Object.prototype.hasOwnProperty.call(opts, k)) merged[k] = opts[k]; }
    merged.signal = ctrl.signal;
    return fetch(url, merged).finally(function () { clearTimeout(timer); });
  }

  // How long a failed /embed suppresses further backend attempts. This used to
  // be a sticky boolean: one SCF cold start that outlasted both attempts sent
  // every later question in the tab to the offline-model prompt for the rest
  // of the session, long after the function had warmed up.
  var REMOTE_DOWN_TTL_MS = 60000;

  function remoteEmbedDown() {
    return state.remoteEmbedDownAt != null
      && (Date.now() - state.remoteEmbedDownAt) < REMOTE_DOWN_TTL_MS;
  }

  function gateThreshold() {
    return (state.meta && state.meta.gate_threshold) || OFFTOPIC_GATE;
  }

  function statValue(stats, kind) {
    if (kind === 'contrast') return stats.top - stats.mean;
    if (kind === 'zscore') return (stats.top - stats.mean) / (stats.std + 1e-6);
    return stats.top;
  }

  function gateValue(stats) {
    return statValue(stats, (state.meta && state.meta.gate_stat) || 'top');
  }
  // Name-dropping inflates similarity ("Yuanchen Wang tell me a joke" scores
  // 0.61), so when the question mentions the name, the gate is applied to the
  // question WITHOUT the name — unless what remains is a bio-intent stub
  // ("who is", "tell me about", empty), which is a legitimate question about
  // YC himself. Retrieval/display still use the full question.
  // Name-blind gate covers both the English name AND 王元辰 (the Chinese gate
  // is otherwise just as easy to fool: "王元辰给我讲个笑话" scores high on the
  // name alone). CJK has no word boundaries, so 王元辰 is matched literally.
  var NAME_TEST_RE = /\b(yuanchen|wang|yc)(?:'s)?\b|王元辰/i;
  var NAME_STRIP_RE = /\b(yuanchen|wang|yc)(?:'s)?\b|王元辰/gi;
  // What remains after removing the name is a legitimate bio question (don't
  // strip — gate on the full text) when it's just a bio-intent phrase. English
  // stubs match at the start; the zh stubs (介绍/简介/谁是/是谁/关于) match
  // anywhere so "用一段话介绍一下" and "…是谁" survive. Mirrored by
  // chat/tests/test_gate.py — keep the two in sync.
  var BIO_STUB_RE = /^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b|^$|介绍|简介|谁是|是谁|关于|(?:都会什么|会做什么|会什么|擅长什么)[?？。!！\s]*$|有(?:哪些|什么)?技能/i;
  var MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

  // CJK detection — one definition, mirrors index.py CJK_RE: hiragana/katakana
  // + CJK unified + compatibility ideographs. Used both for the offline
  // English-only notice and to route name-normalization by the question's
  // language. (Explicit \u escapes: the literal compat-range start char is a
  // homoglyph of the unified one and easy to mistype.)
  var CJK_RE = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]/;

  // Name-blind gate input: the remainder after stripping YC's name, or null
  // when the name should be KEPT (no name mentioned at all, or what's left
  // after stripping is just a bio-intent stub -- see BIO_STUB_RE above).
  // Mirrors runtime.py's strip_name(). Extracted into a named function (was
  // formerly inlined in send()) so chat/tests/test_implementation_sync.py can
  // execute it directly instead of a second, re-typed copy of this logic.
  function stripName(question) {
    if (!NAME_TEST_RE.test(question)) return null;
    var remainder = question.replace(NAME_STRIP_RE, ' ')
      .replace(/\s+/g, ' ').trim()
      .replace(/^[\s:;,.!?—-]+|[\s:;,.!?—-]+$/g, '');
    return BIO_STUB_RE.test(remainder) ? null : remainder;
  }

  // Gate input. When the name was stripped (off-topic detection), gate on the
  // remainder. When it was KEPT (a bio question), normalize the name to the
  // gate's own language — so a Chinese question that uses "YC", or an English
  // one that uses 王元辰, still matches the corpus (which is single-language per
  // gate). Retrieval always uses the original, unmodified question.
  function gateForm(question, stripped) {
    if (stripped != null) return stripped;
    return CJK_RE.test(question)
      ? question.replace(NAME_STRIP_RE, '王元辰')
      : question.replace(/王元辰/g, 'YC');
  }

  var inSubpage = /\/pages\//.test(window.location.pathname);
  var PREFIX = inSubpage ? '../' : '';

  // ── i18n: the widget follows the site's 中/EN toggle (scripts/i18n.js) ──
  // Reads the active language on open and re-localizes on the 'yc-langchange'
  // event i18n.js fires. Only UI chrome is localized here; role label/tagline/
  // starters come from roles.json (each role's optional zh:{} block).
  function lang() {
    try { return (window.YCI18N && window.YCI18N.current()) || 'en'; }
    catch (e) { return 'en'; }
  }
  var STR = {
    en: {
      askBtn: '✦ ASK AI',
      openAria: 'Open AI chat about YC',
      panelAria: 'Chat with an AI about YC',
      header: 'Ask about YC',
      chooseRole: 'choose role',
      roleChipTitle: 'Change visitor role',
      closeAria: 'Close chat',
      placeholder: 'Ask about projects, skills…',
      send: 'Send',
      loadingGeneric: 'Loading…',
      loadingModel: 'Loading the on-device search model (~23 MB, cached after the first visit)…',
      loadingModelPct: function (pct) { return 'Loading the on-device search model… ' + pct + '%'; },
      assetsFail: function (msg) { return 'Could not load the chat assets (' + msg + ').'; },
      fileProtocol: 'The chat can\'t run from a file:// page (browsers block the model and index from loading). Serve the site locally instead:\n\npython -m http.server 8000\n\nthen open http://localhost:8000 — or just use the live site.',
      greeting: 'Hi! I answer questions about YC, grounded in the pages of this site. Who\'s visiting?',
      viewingAs: function (label) { return 'Viewing as: ' + label + '. Ask me anything about YC — answers link back to the relevant pages.'; },
      refused: 'That doesn\'t look like a question about YC, and that\'s all I can help with here — his projects, skills, education, and publications. Try one of these:',
      degradedCJK: 'The AI answer service is unreachable right now, and offline search only covers English. Here are some pages you can browse in the meantime:',
      degradedLoading: function (pct) { return 'Backend unreachable — loading offline search (' + pct + '%)…'; },
      degradedSources: 'The AI answer service is unreachable right now, but these pages look most relevant to your question:',
      // Task 24 broke this claim by pointing the (then single, dual-purpose)
      // fallback file's retrieval half at a curated corpus that no longer
      // chunk-order-aligned with the full index. Task 29 Part 2 fixed it for
      // real, not just reworded it: chat/data/chunks_en_minilm.json is now a
      // genuine retrieval corpus (real chunk records, resolved by id -- see
      // retrieveFallback), so degradedSources is accurate again. Kept as a
      // SEPARATE string from degradedNoSources (below) rather than merged
      // back into one, because that case -- gate passed, but nothing
      // individually cleared MIN_SCORE -- is still real and still deserves
      // its own honest wording.
      degradedNoSources: 'The AI answer service is unreachable right now. Offline search confirms this is something I can help with, but can\'t reliably point to a specific page yet — here are some pages you can browse:',
      backendDown: 'The chat backend is unreachable right now — please try again in a minute.',
      retrievalOnly: 'Demo is in retrieval-only mode (no LLM connected yet), but here\'s what the semantic index surfaces for that — sources below:',
      somethingWrong: function (msg) { return 'Something went wrong (' + msg + '). Please try again.'; },
      offlineConsent: 'The AI answer service is unreachable. I can download a small on-device search model (~23 MB, cached after this) to still find relevant pages — or you can just browse the site.',
      offlineUse: 'Use offline search (~23 MB)',
      offlineBrowse: 'Just browse the site',
      offlineDeclined: 'No problem — here are some pages to explore. Ask again anytime to turn on offline search.',
      clearLabel: 'Clear',
      clearTitle: 'Clear this conversation',
      interrupted: 'That answer didn\'t finish — the page changed while it was loading. Ask again?',
    },
    zh: {
      askBtn: '✦ 问 AI',
      openAria: '打开关于王元辰的 AI 聊天',
      panelAria: '与 AI 聊王元辰的作品集',
      header: '问问王元辰',
      chooseRole: '选择身份',
      roleChipTitle: '切换访客身份',
      closeAria: '关闭聊天',
      placeholder: '问问项目、技能…',
      send: '发送',
      loadingGeneric: '加载中…',
      loadingModel: '正在加载本地检索模型（约 23 MB，首次访问后会缓存）…',
      loadingModelPct: function (pct) { return '正在加载本地检索模型… ' + pct + '%'; },
      assetsFail: function (msg) { return '无法加载聊天所需资源（' + msg + '）。'; },
      fileProtocol: '聊天无法在 file:// 页面运行（浏览器会阻止模型和索引加载）。请在本地起一个服务器：\n\npython -m http.server 8000\n\n然后打开 http://localhost:8000 —— 或直接访问线上站点。',
      greeting: '你好！我可以回答关于王元辰的问题，内容都来自本站页面。请问你是？',
      viewingAs: function (label) { return '当前身份：' + label + '。有关王元辰的问题都可以问我 —— 回答下方会附上相关页面链接。'; },
      refused: '这看起来不像是关于王元辰的问题，而我只能回答这方面的内容 —— 他的项目、技能、教育和论文。可以试试这些：',
      degradedCJK: 'AI 回答服务暂时无法连接，而离线检索目前只支持英文。你可以先浏览这些页面：',
      degradedLoading: function (pct) { return '后端暂时无法连接 —— 正在加载离线检索（' + pct + '%）…'; },
      degradedSources: 'AI 回答服务暂时无法连接，不过这些页面看起来和你的问题最相关：',
      degradedNoSources: 'AI 回答服务暂时无法连接。离线检索确认这是我能回答的话题，但目前还不能可靠地指向具体某个页面 —— 你可以先看看这些页面：',
      backendDown: '聊天后端暂时无法连接 —— 请过一会儿再试。',
      retrievalOnly: '演示目前处于「仅检索」模式（还没接入 LLM），不过这是语义索引为该问题找到的内容 —— 来源见下方：',
      somethingWrong: function (msg) { return '出错了（' + msg + '）。请再试一次。'; },
      offlineConsent: 'AI 回答服务暂时无法连接。我可以下载一个很小的本地检索模型（约 23 MB，之后会缓存）来帮你找到相关页面 —— 你也可以直接浏览站点。',
      offlineUse: '使用离线检索（约 23 MB）',
      offlineBrowse: '直接浏览站点',
      offlineDeclined: '没问题 —— 这里有一些页面可以看看。你随时可以再问一次来启用离线检索。',
      clearLabel: '清空',
      clearTitle: '清空当前对话',
      interrupted: '上一个回答没能完成 —— 加载过程中页面发生了跳转。要再问一次吗？',
    },
  };
  function t(key) {
    var v = (STR[lang()] || STR.en)[key];
    if (v === undefined) v = STR.en[key];
    if (typeof v === 'function') return v.apply(null, Array.prototype.slice.call(arguments, 1));
    return v;
  }
  // Localized role field: role[lang][field] (e.g. role.zh.label) when present,
  // else the top-level English value. English stays top-level for the function.
  function L(obj, field) {
    if (!obj) return undefined;
    var lz = lang();
    if (lz !== 'en' && obj[lz] && obj[lz][field] != null) return obj[lz][field];
    return obj[field];
  }

  // ── State ─────────────────────────────────────────────────────────────
  var state = {
    open: false,
    role: null,
    roles: null,
    meta: null, // chat/data/meta.json — gate_threshold/gate_stat/gate_remote/model/query_prefix/chunks_file
    chunks: null, // chat/data/{meta.chunks_file}'s chunk array — fetched only when localModelMatchesIndex()
    degradedGate: null, // chat/data/gate_en_minilm.json — fetched lazily, degraded mode only
    degradedChunks: null, // chat/data/chunks_en_minilm.json — fetched lazily, degraded mode only
    extractor: null,
    loading: null, // Promise while core assets load
    extractorLoading: null, // Promise while the local model loads
    remoteEmbedDownAt: null, // ms timestamp of the last /embed give-up, or null
    busy: false,
    session: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()),
    history: [], // [{role:'user'|'assistant', content}], alias of transcripts[role].history
    offlineConsent: null, // null = undecided; true/false = this session's choice (TODO#5)
    transcripts: {}, // TODO#3: {roleId: {history:[...], log:[...]}} persisted in sessionStorage
  };

  // ── Session memory (TODO#3): per-role transcripts persist across page
  // navigations WITHIN the tab (sessionStorage, deliberately NOT localStorage,
  // so nothing survives closing the tab). We persist the session id too, so the
  // server-side logs correlate the same visitor across pages by sid. `log` is a
  // replayable record of rendered turns; `history` is the {role,content} pairs
  // fed back to /chat (capped when sent). ──────────────────────────────────
  var STORE_KEY = 'yc-chat-session';
  var HISTORY_CAP = 40; // stored user+assistant entries kept per role (sent is capped separately)

  function saveSession() {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify({
        v: 1,
        session: state.session,
        role: state.role,
        transcripts: state.transcripts,
      }));
    } catch (e) { /* private mode / quota: memory just won't persist */ }
  }

  function loadSession() {
    try {
      var data = JSON.parse(sessionStorage.getItem(STORE_KEY) || 'null');
      if (!data || data.v !== 1) return;
      if (data.session) state.session = data.session;
      state.transcripts = data.transcripts || {};
      state.role = data.role || null;
      if (state.role && state.transcripts[state.role]) {
        state.history = state.transcripts[state.role].history || [];
      }
    } catch (e) { /* corrupt store: start fresh */ }
  }

  function transcript(id) {
    if (!state.transcripts[id]) state.transcripts[id] = { history: [], log: [] };
    return state.transcripts[id];
  }

  // Append a replayable entry to the active role's transcript and persist.
  function pushLog(entry) {
    if (!state.role) return;
    transcript(state.role).log.push(entry);
    saveSession();
  }

  // Minimal source data needed to re-render a sources block after navigation.
  function sourcesForLog(results) {
    return dedupeForDisplay(results).map(function (r) {
      return {
        chunk: {
          url: r.chunk.url, anchor: r.chunk.anchor,
          page_title: r.chunk.page_title, section_title: r.chunk.section_title,
          text: r.chunk.text, id: r.chunk.id,
        },
        score: r.score,
      };
    });
  }

  // A turn whose answer never arrived persists as a trailing `user` entry:
  // pushLog records the bot reply only once the turn completes, so navigating
  // away mid-flight (clicking a source card does exactly that) stores the
  // question and nothing else. Structural check on the transcript, not a guess
  // about the text.
  function lastTurnInterrupted(log) {
    var last = log[log.length - 1];
    return !!(last && last.type === 'user');
  }

  // Replay a role's stored transcript into the body (on open / role switch /
  // page load). Messages keep the language they were written in; interactive
  // starters re-render in the current language.
  function renderLog() {
    els.body.textContent = '';
    var tr = state.transcripts[state.role];
    if (!tr) return;
    tr.log.forEach(function (e) {
      if (e.type === 'user' || e.type === 'bot' || e.type === 'note') addMsg(e.type, e.text);
      else if (e.type === 'sources') addSources(e.results);
      else if (e.type === 'pagelinks') addPageLinks();
      else if (e.type === 'starters' && state.roles && state.roles.roles[state.role]) addStarters(state.roles.roles[state.role]);
    });
    // Derived on replay, never pushed to the log: if the turn is later
    // completed in another tab or re-asked, the stored transcript stays clean.
    if (lastTurnInterrupted(tr.log)) addMsg('note', t('interrupted'));
    els.body.scrollTop = els.body.scrollHeight;
  }

  // Static page suggestions for the "just browse" path — no model, no embedding.
  var SUGGESTED_PAGES = [
    { url: 'pages/projects.html', en: 'Projects', zh: '项目' },
    { url: 'pages/skills.html', en: 'Skills', zh: '技能' },
    { url: 'pages/education.html', en: 'Education', zh: '教育' },
    { url: 'pages/publications.html', en: 'Publications', zh: '论文' },
    { url: 'pages/agents.html', en: 'Agents', zh: '智能体' },
  ];

  // ── Logging (everything in, everything out) ───────────────────────────
  function logTurn(record) {
    record.ts = new Date().toISOString();
    record.session = state.session;
    record.page = window.location.pathname;
    console.debug('[yc-chat]', record);
    if (typeof window.gtag === 'function') {
      window.gtag('event', 'chat_turn', {
        chat_role: record.role || '',
        chat_mode: record.mode || '',
        chat_question: (record.question || '').slice(0, 100),
      });
    }
    // Server-side logging: /chat turns are logged by the worker itself; send
    // the rest (retrieval-only turns, errors) to /log when a worker exists.
    if (WORKER_URL && record.mode !== 'llm') {
      try {
        fetch(WORKER_URL + '/log', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(record),
        }).catch(function () {});
      } catch (e) { /* logging must never break the chat */ }
    }
  }

  // ── Lazy asset loading ────────────────────────────────────────────────
  // Core (roles + meta) always loads on open — it's ~10KB, not the 1.4MB
  // chunk index (Task 29). The chunk index itself loads only when the
  // vendored in-browser model can actually use it (light mode / the local
  // half of degraded mode) — normal mode's retrieval happens server-side.
  // The 23MB local model loads separately, only when actually needed.
  function loadCore() {
    if (state.loading) return state.loading;
    state.loading = (async function () {
      // no-cache = revalidate with the server (cheap 304 when unchanged), so
      // visitors never see a stale index/roles after a rebuild is deployed.
      var rolesRes = fetch(PREFIX + 'chat/data/roles.json', { cache: 'no-cache' }).then(function (r) {
        if (!r.ok) throw new Error('roles.json ' + r.status);
        return r.json();
      });
      var metaRes = fetch(PREFIX + 'chat/data/meta.json', { cache: 'no-cache' }).then(function (r) {
        if (!r.ok) throw new Error('meta.json ' + r.status);
        return r.json();
      });
      state.roles = await rolesRes;
      state.meta = await metaRes;
      if (localModelMatchesIndex()) {
        // chunks_file (Task 29 Part 2) names the retrieval corpus for
        // WHATEVER preset actually built this meta.json -- fetched by name,
        // never inferred from `model` here.
        var indexRes = fetch(PREFIX + 'chat/data/' + state.meta.chunks_file, { cache: 'no-cache' }).then(function (r) {
          if (!r.ok) throw new Error(state.meta.chunks_file + ' ' + r.status);
          return r.json();
        });
        state.chunks = (await indexRes).chunks || [];
      } else {
        state.chunks = [];
      }
    })();
    return state.loading;
  }

  // The vendored in-browser model only matches indexes built with MiniLM —
  // an e5-built index needs the backend's own retrieval (Task 29) and has no
  // local fallback.
  function localModelMatchesIndex() {
    return !state.meta || /all-MiniLM-L6-v2/.test(state.meta.model || '');
  }

  function ensureExtractor(onProgress) {
    if (state.extractor) return Promise.resolve();
    if (state.extractorLoading) return state.extractorLoading;
    state.extractorLoading = (async function () {
      var T = await import(new URL(PREFIX + 'scripts/vendor/transformers.min.js', window.location.href).href);
      T.env.allowRemoteModels = false;
      T.env.localModelPath = new URL(PREFIX + 'chat/models/', window.location.href).href;
      T.env.backends.onnx.wasm.wasmPaths = new URL(PREFIX + 'scripts/vendor/', window.location.href).href;
      T.env.backends.onnx.wasm.numThreads = 1;
      state.extractor = await T.pipeline('feature-extraction', MODEL_ID, {
        quantized: true,
        progress_callback: function (p) {
          if (onProgress && p.status === 'progress' && /model_quantized/.test(p.file || '') && p.total) {
            onProgress(Math.round((p.loaded / p.total) * 100));
          }
        },
      });
    })();
    return state.extractorLoading;
  }

  // Embed `text` with the vendored in-browser model. Used only on the local
  // path: light mode (no backend, or the index matches the local model) and
  // the "classic local gate" branch in send(), which always needs a genuine
  // local vector regardless of whether the backend is otherwise reachable.
  async function localEmbed(text) {
    await ensureExtractor(null);
    var out = await state.extractor((state.meta.query_prefix || '') + text, { pooling: 'mean', normalize: true });
    return out.data; // Float32Array(384), unit norm
  }

  // Returns {vector, gate, rid}. Task 29: the backend is only ever asked for
  // a GATE decision now (gate_only: true) — normal-mode retrieval moved
  // server-side into /chat, so there is no more use for a remotely-computed
  // query vector. `vector` is therefore populated only via the local
  // in-browser model (light mode, or the local half of degraded mode); gate
  // is non-null only when the backend embeds AND has the gate model packaged
  // — it judges gateText (the name-stripped question) with MiniLM, since e5
  // can't separate on/off-topic.
  async function embedQuery(text, gateText) {
    // Preferred: the backend judges the gate (no client download). One retry
    // with a pause covers cold starts and transient hiccups before we give
    // up on the backend for this session.
    if (WORKER_URL && !remoteEmbedDown()) {
      for (var attempt = 0; attempt < 2; attempt++) {
        try {
          var res = await fetchWithTimeout(WORKER_URL + '/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, gate_text: gateText || text, gate_only: true }),
          }, EMBED_TIMEOUT_MS);
          if (res.ok) {
            var data = await res.json();
            return { vector: null, gate: data.gate || null, rid: data.rid || null };
          }
          if (res.status === 429) throw new Error('rate limited — please wait a minute and try again');
        } catch (e) {
          if (String(e && e.message).indexOf('rate limited') === 0) throw e;
        }
        if (attempt === 0) await new Promise(function (r) { setTimeout(r, 1500); });
      }
      state.remoteEmbedDownAt = Date.now();
      logTurn({ event: 'remote_embed_fallback' });
    }
    if (!localModelMatchesIndex()) {
      throw new Error('embedding service unavailable (the search index needs the server-side model)');
    }
    return { vector: await localEmbed(text), gate: null, rid: null };
  }

  // Dot-product retrieval + gate stats. vecAt(i) yields the i-th chunk vector,
  // chunkAt(i) the chunk it maps to — the only thing that differs between the
  // primary (state.chunks) and degraded (state.degradedChunks) paths. Both
  // paths index vecAt/chunkAt into the SAME array at the SAME position i, so
  // this is never a cross-array positional guess — see retrieveFallback below
  // for why that distinction matters.
  function scoreChunks(count, queryVec, vecAt, chunkAt) {
    var scored = [], sum = 0, sumSq = 0;
    for (var i = 0; i < count; i++) {
      var v = vecAt(i), s = 0;
      for (var j = 0; j < v.length; j++) s += v[j] * queryVec[j];
      scored.push({ chunk: chunkAt(i), score: s });
      sum += s;
      sumSq += s * s;
    }
    var n = scored.length || 1, mean = sum / n;
    scored.sort(function (a, b) { return b.score - a.score; });
    return {
      results: scored.slice(0, TOP_K).filter(function (r) { return r.score >= MIN_SCORE; }),
      stats: {
        top: scored.length ? scored[0].score : 0,
        mean: mean,
        std: Math.sqrt(Math.max(sumSq / n - mean * mean, 0)),
      },
    };
  }

  function retrieve(queryVec) {
    var chunks = state.chunks;
    return scoreChunks(chunks.length, queryVec,
      function (i) { return chunks[i].vector; }, function (i) { return chunks[i]; });
  }

  // Destinations that carry no information beyond "this site": index.html has
  // no section ids at all (its only <section> is the landing menu, so every
  // link lands on the sphere), and pages/projects.html is a listing. Several
  // curated knowledge/about_en.md chunks deliberately link: to both, and their
  // TEXT is good grounding -- so this is display-only. dedupeForDisplay,
  // sourcesForLog and record.retrieved all keep the full ranked list.
  var HUB_URLS = { 'index.html': 1, 'pages/projects.html': 1 };

  function displayableSources(results) {
    return results.filter(function (r) { return !HUB_URLS[r.chunk.url]; });
  }

  function dedupeForDisplay(results) {
    var seen = {}, out = [];
    results.forEach(function (r) {
      var key = r.chunk.url + '|' + r.chunk.section_title;
      if (!seen[key]) { seen[key] = true; out.push(r); }
    });
    return out;
  }

  // ── Degraded mode: backend down + e5 index (Task 29 Part 2) ───────────
  // Two SEPARATE single-purpose files now, never one file doing both jobs:
  //   chat/data/gate_en_minilm.json    — MiniLM vectors over knowledge/
  //                                      about_en.md's ~55 curated sections.
  //                                      GATE ONLY: no chunk records, so
  //                                      there is nothing here to resolve a
  //                                      source link from.
  //   chat/data/chunks_en_minilm.json  — the SAME 192 English chunks as the
  //                                      normal-mode retrieval corpus's
  //                                      lang=="en" entries (same ids, same
  //                                      order, same text), re-embedded with
  //                                      MiniLM. RETRIEVAL ONLY: real chunk
  //                                      records, each carrying its own real
  //                                      id/url/anchor/page_title/
  //                                      section_title/text.
  //
  // Task 24 broke degraded-mode source links by pointing the (then single,
  // dual-purpose) gate file at a curated corpus that no longer order-aligned
  // with the full chunk index, while retrieveFallback kept mapping
  // fb.vectors[i] -> state.index.chunks[i] positionally across those two
  // now-unrelated arrays. The fix here is not a rename: gate scoring and
  // retrieval now run against two INDEPENDENT arrays, and retrieveFallback
  // below resolves each result from chunks_en_minilm.json's OWN chunk
  // record at that SAME position — never a position borrowed from a
  // different array. That is exactly the "resolve by id, never by a
  // position borrowed from an unrelated array" property this split exists
  // to restore; addSources/dedupeForDisplay/sourcesForLog/record.retrieved
  // all read real chunk ids again, same as normal mode.
  function loadDegradedGate() {
    if (state.degradedGate) return Promise.resolve(state.degradedGate);
    return fetch(PREFIX + 'chat/data/gate_en_minilm.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error('offline gate unavailable');
      return r.json();
    }).then(function (g) { state.degradedGate = g; return g; });
  }

  function loadDegradedChunks() {
    if (state.degradedChunks) return Promise.resolve(state.degradedChunks);
    return fetch(PREFIX + 'chat/data/chunks_en_minilm.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error('offline search index unavailable');
      return r.json();
    }).then(function (c) { state.degradedChunks = c; return c; });
  }

  // Identical shape to retrieve() above, parameterized on WHICH chunk array
  // to score — chunks_en_minilm.json's real chunk records, not fb.vectors'
  // curated gate corpus. vecAt/chunkAt both index the SAME array at the
  // SAME position, so this is never a cross-array positional guess.
  function retrieveFallback(chunks, queryVec) {
    return scoreChunks(chunks.length, queryVec,
      function (i) { return chunks[i].vector; }, function (i) { return chunks[i]; });
  }

  async function degradedTurn(question, stripped, thinking, record) {
    record.mode = 'degraded-local';
    if (CJK_RE.test(question)) {
      // The local fallback model is English-only — be honest for CJK, never
      // download it for a question it can't serve, but still offer static page
      // links (localized) so a Chinese visitor isn't left at a dead end.
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = t('degradedCJK');
      addPageLinks();
      pushLog({ type: 'bot', text: thinking.textContent });
      pushLog({ type: 'pagelinks' });
      record.mode = 'degraded-cjk';
      record.answer = thinking.textContent;
      logTurn(record);
      return;
    }
    // TODO#5: never auto-download the ~23MB in-browser model. Ask once per
    // session; remember the choice; declining shows static page links only.
    if (state.offlineConsent === false) return offlineDeclined(thinking, record);
    if (state.offlineConsent === true) return runOfflineSearch(question, stripped, thinking, record);

    thinking.classList.remove('ycchat-dots');
    thinking.textContent = t('offlineConsent');
    var choices = h('div', 'ycchat-starters');
    var yes = h('button', 'ycchat-starter', t('offlineUse'));
    var no = h('button', 'ycchat-starter', t('offlineBrowse'));
    yes.type = 'button';
    no.type = 'button';
    yes.addEventListener('click', function () {
      state.offlineConsent = true;
      choices.remove();
      logTurn({ event: 'offline_consent', consent: true });
      runOfflineSearch(question, stripped, thinking, record);
    });
    no.addEventListener('click', function () {
      state.offlineConsent = false;
      choices.remove();
      logTurn({ event: 'offline_consent', consent: false });
      offlineDeclined(thinking, record);
    });
    choices.appendChild(yes);
    choices.appendChild(no);
    els.body.appendChild(choices);
    els.body.scrollTop = els.body.scrollHeight;
  }

  // Declined offline search: static page links, no model, no embedding.
  function offlineDeclined(thinking, record) {
    thinking.classList.remove('ycchat-dots');
    thinking.textContent = t('offlineDeclined');
    addPageLinks();
    pushLog({ type: 'bot', text: thinking.textContent });
    pushLog({ type: 'pagelinks' });
    record.mode = 'degraded-declined';
    record.answer = thinking.textContent;
    logTurn(record);
  }

  // Consent granted: pull the MiniLM model and do local fallback retrieval.
  async function runOfflineSearch(question, stripped, thinking, record) {
    try {
      var gate = await loadDegradedGate();
      var chunks = await loadDegradedChunks();
      thinking.classList.add('ycchat-dots');
      await ensureExtractor(function (pct) {
        thinking.textContent = t('degradedLoading', pct);
        thinking.classList.remove('ycchat-dots');
      });
      var embedLocal = async function (q, prefix) {
        var out = await state.extractor((prefix || '') + q, { pooling: 'mean', normalize: true });
        return out.data;
      };
      var retrieved = retrieveFallback(chunks.chunks, await embedLocal(question, chunks.query_prefix));
      // Gate on gateForm()'s output, not on `stripped ? stripped : question`.
      // Those differ when the name was KEPT (a bio question): gateForm then
      // normalizes the name to the gate corpus's own language, which the raw
      // question does not. runtime.py and functions/tencent/index.py both gate
      // on gateForm's output, so the shortcut was a silent third divergence —
      // the same class as BIO_STUB_RE's, which has already drifted twice.
      var degradedGateText = gateForm(question, stripped);
      var gateVec = await embedLocal(degradedGateText, gate.query_prefix);
      var gateStats = scoreChunks(
        gate.vectors.length, gateVec,
        function (i) { return gate.vectors[i]; }, function () { return null; }
      ).stats;
      var gateScore = statValue(gateStats, gate.gate_stat || 'top');
      thinking.classList.remove('ycchat-dots');
      if (gateScore < (gate.gate_threshold || OFFTOPIC_GATE)) {
        thinking.textContent = t('refused');
        addStarters(state.roles.roles[state.role]);
        pushLog({ type: 'bot', text: thinking.textContent });
        pushLog({ type: 'starters' });
      } else if (retrieved.results.length) {
        // Task 29 Part 2: retrieved.results now names REAL chunks from
        // chunks_en_minilm.json (see retrieveFallback above), so degraded
        // mode can show real source links again, same as normal mode.
        thinking.textContent = t('degradedSources');
        addSources(retrieved.results);
        pushLog({ type: 'bot', text: thinking.textContent });
        pushLog({ type: 'sources', results: sourcesForLog(retrieved.results) });
      } else {
        // Gate passed (on-topic) but nothing cleared MIN_SCORE -- genuinely
        // no reliable source to point to, so fall back to the static,
        // curated page-link list degradedCJK/offlineDeclined already use.
        thinking.textContent = t('degradedNoSources');
        addPageLinks();
        pushLog({ type: 'bot', text: thinking.textContent });
        pushLog({ type: 'pagelinks' });
      }
      // record.retrieved carries real chunk ids again (Task 29 Part 2) --
      // contrast the old fallback-corpus-position ids this used to log.
      record.retrieved = retrieved.results.map(function (r) { return { id: r.chunk.id, score: +r.score.toFixed(3) }; });
      record.answer = thinking.textContent;
      logTurn(record);
    } catch (err2) {
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = t('backendDown');
      logTurn({ event: 'error', role: state.role, question: question, error: String(err2) });
    }
  }

  // ── LLM call via worker ───────────────────────────────────────────────
  // Task 29: the client no longer sends contexts — the function retrieves
  // from its own bundled index and returns the chunks it used as `sources`.
  async function askWorker(question) {
    var res = await fetchWithTimeout(WORKER_URL + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session: state.session,
        role: state.role,
        lang: lang(),
        question: question,
        history: state.history.slice(-6),
      }),
    }, CHAT_TIMEOUT_MS);
    // Same rate-limit message /embed's remote branch throws (see
    // embedQuery) -- send()'s chatErr handler bubbles this one to the
    // generic "something went wrong" path instead of degraded mode, for the
    // same reason: a transient rate limit isn't "the backend is down," and
    // shouldn't prompt a 23MB offline-model download.
    if (res.status === 429) throw new Error('rate limited — please wait a minute and try again');
    if (!res.ok) throw new Error('worker ' + res.status);
    return await res.json(); // {answer, model, rid, sources[], refused?}
  }

  // Rebuilds the widget's internal {chunk, score} shape (what addSources,
  // dedupeForDisplay, sourcesForLog and record.retrieved all expect) from
  // /chat's `sources` array — the read-side equivalent of what retrieve()
  // used to return, now that retrieval happens server-side (Task 29).
  function resultsFromSources(sources) {
    return (sources || []).map(function (s) {
      // Guard at the boundary where server-supplied data becomes the
      // widget's internal {chunk, score} shape: every downstream reader
      // (addSources' r.score.toFixed(2), the record.retrieved/sourcesForLog
      // .toFixed(3) call sites) assumes a real number, the same guarantee
      // scoreChunks' own numeric loop always provides for the client-side
      // path. A backend that omits or malforms `score` must not throw deep
      // inside rendering.
      var score = typeof s.score === 'number' && isFinite(s.score) ? s.score : 0;
      return {
        chunk: {
          id: s.id, url: s.url, anchor: s.anchor,
          page_title: s.page_title, section_title: s.section_title, text: s.text,
        },
        score: score,
      };
    });
  }

  // The UI renders plain text, so strip stray markdown emphasis the LLM may
  // emit despite the prompt (e.g. **Prime Engine**). Returns '' when there is
  // nothing to render: `resp.answer` can legitimately be empty (a reasoner
  // model puts its text in reasoning_content, a content-filter stop returns
  // "") or missing, and both used to land in the bubble as-is -- an empty bot
  // bubble with the dots gone and no error anywhere.
  function normalizeAnswer(raw) {
    return String(raw == null ? '' : raw)
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/^#+\s+/gm, '')
      .trim();
  }

  // ── UI ────────────────────────────────────────────────────────────────
  var els = {};

  function h(tag, cls, text) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function injectStyles() {
    var css = [
      '.ycchat-btn{position:fixed;left:1.1rem;bottom:1.1rem;z-index:1200;border:1px solid var(--border,#ccc);',
      ' background:var(--card,#f7f7f8);color:var(--fg,#131313);border-radius:999px;padding:.65rem 1rem;cursor:pointer;',
      ' font:600 14px/1 system-ui,sans-serif;box-shadow:var(--shadow,0 10px 25px rgba(0,0,0,.15));}',
      '.ycchat-btn:hover{border-color:var(--link,#0b57d0);}',
      '.ycchat-panel{position:fixed;left:1.1rem;bottom:4.2rem;z-index:1200;width:min(400px,calc(100vw - 2rem));',
      ' height:min(560px,calc(100vh - 7rem));display:flex;flex-direction:column;background:var(--bg,#fff);',
      ' color:var(--fg,#131313);border:1px solid var(--border,#ccc);border-radius:var(--radius-lg,20px);',
      ' box-shadow:var(--shadow,0 10px 30px rgba(0,0,0,.2));overflow:hidden;font:14px/1.55 system-ui,sans-serif;}',
      '.ycchat-head{display:flex;align-items:center;gap:.5rem;padding:.7rem .9rem;border-bottom:1px solid var(--border,#ccc);background:var(--card,#f7f7f8);}',
      '.ycchat-head b{font-size:.95rem;}',
      '.ycchat-rolechip{margin-left:auto;font-size:.72rem;color:var(--muted,#666);border:1px solid var(--border,#ccc);',
      ' border-radius:999px;padding:.15rem .55rem;cursor:pointer;background:var(--bg,#fff);}',
      '.ycchat-clearchip{font-size:.72rem;color:var(--muted,#666);border:1px solid var(--border,#ccc);',
      ' border-radius:999px;padding:.15rem .55rem;cursor:pointer;background:var(--bg,#fff);}',
      '.ycchat-clearchip:hover{border-color:var(--link,#0b57d0);}',
      '.ycchat-x{border:0;background:none;color:var(--muted,#666);font-size:1.1rem;cursor:pointer;padding:.2rem .4rem;}',
      '.ycchat-body{flex:1;overflow-y:auto;padding:.9rem;display:flex;flex-direction:column;gap:.6rem;}',
      '.ycchat-msg{max-width:88%;padding:.55rem .75rem;border-radius:14px;white-space:pre-wrap;word-wrap:break-word;}',
      '.ycchat-msg.user{align-self:flex-end;background:var(--link,#0b57d0);color:#fff;border-bottom-right-radius:4px;}',
      '.ycchat-msg.bot{align-self:flex-start;background:var(--card,#f7f7f8);border:1px solid var(--border,#ccc);border-bottom-left-radius:4px;}',
      '.ycchat-msg.note{align-self:center;background:none;color:var(--muted,#666);font-size:.78rem;text-align:center;}',
      '.ycchat-srcs{align-self:flex-start;display:flex;flex-direction:column;gap:.35rem;max-width:88%;}',
      '.ycchat-src{display:block;text-decoration:none;border:1px solid var(--border,#ccc);border-radius:10px;padding:.45rem .6rem;',
      ' background:var(--card,#f7f7f8);color:var(--fg,#131313);font-size:.8rem;}',
      '.ycchat-src:hover{border-color:var(--link,#0b57d0);}',
      '.ycchat-src b{color:var(--link,#0b57d0);display:block;font-size:.8rem;}',
      '.ycchat-src span{color:var(--muted,#666);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}',
      '.ycchat-roles{display:flex;flex-direction:column;gap:.5rem;padding:.4rem 0;}',
      '.ycchat-role{border:1px solid var(--border,#ccc);border-radius:12px;background:var(--card,#f7f7f8);color:var(--fg,#131313);',
      ' padding:.6rem .8rem;cursor:pointer;text-align:left;font:inherit;}',
      '.ycchat-role:hover{border-color:var(--link,#0b57d0);}',
      '.ycchat-role b{display:block;} .ycchat-role span{color:var(--muted,#666);font-size:.78rem;}',
      '.ycchat-starters{display:flex;flex-wrap:wrap;gap:.35rem;}',
      '.ycchat-starter{border:1px solid var(--border,#ccc);border-radius:999px;background:var(--bg,#fff);color:var(--link,#0b57d0);',
      ' font-size:.75rem;padding:.3rem .6rem;cursor:pointer;}',
      '.ycchat-foot{display:flex;gap:.5rem;padding:.7rem;border-top:1px solid var(--border,#ccc);background:var(--card,#f7f7f8);}',
      '.ycchat-in{flex:1;border:1px solid var(--border,#ccc);border-radius:12px;padding:.5rem .7rem;font:inherit;',
      ' background:var(--bg,#fff);color:var(--fg,#131313);outline:none;}',
      '.ycchat-in:focus{border-color:var(--link,#0b57d0);}',
      '.ycchat-send{border:1px solid var(--border,#ccc);background:var(--link,#0b57d0);color:#fff;border-radius:12px;',
      ' padding:.5rem .9rem;cursor:pointer;font:600 13px/1 system-ui,sans-serif;}',
      '.ycchat-send:disabled{opacity:.5;cursor:default;}',
      '.ycchat-dots::after{content:"…";animation:ycchat-b 1.2s infinite;}',
      '@keyframes ycchat-b{0%{opacity:.2}50%{opacity:1}100%{opacity:.2}}',
    ].join('');
    var style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }

  function addMsg(kind, text) {
    var el = h('div', 'ycchat-msg ' + kind, text);
    els.body.appendChild(el);
    els.body.scrollTop = els.body.scrollHeight;
    return el;
  }

  function addSources(results) {
    var shown = dedupeForDisplay(displayableSources(results));
    if (!shown.length) return; // after filtering, so an all-hub set adds no empty wrapper
    var wrap = h('div', 'ycchat-srcs');
    shown.forEach(function (r) {
      var href = PREFIX + r.chunk.url + (r.chunk.anchor ? '#' + r.chunk.anchor : '');
      var a = h('a', 'ycchat-src');
      a.href = href;
      var sec = r.chunk.section_title || '';
      var titleText = (!sec || sec === r.chunk.page_title || sec.indexOf(r.chunk.page_title) === 0)
        ? (sec || r.chunk.page_title)
        : r.chunk.page_title + ' — ' + sec;
      var title = h('b', null, titleText);
      var snippet = h('span', null, r.chunk.text.slice(0, 110) + '…  (' + r.score.toFixed(2) + ')');
      a.appendChild(title);
      a.appendChild(snippet);
      wrap.appendChild(a);
    });
    els.body.appendChild(wrap);
    els.body.scrollTop = els.body.scrollHeight;
  }

  function addPageLinks() {
    var wrap = h('div', 'ycchat-srcs');
    SUGGESTED_PAGES.forEach(function (p) {
      var a = h('a', 'ycchat-src');
      a.href = PREFIX + p.url;
      a.appendChild(h('b', null, lang() === 'zh' ? p.zh : p.en));
      wrap.appendChild(a);
    });
    els.body.appendChild(wrap);
    els.body.scrollTop = els.body.scrollHeight;
  }

  function showRolePicker() {
    els.body.textContent = '';
    addMsg('note', t('greeting'));
    var wrap = h('div', 'ycchat-roles');
    Object.keys(state.roles.roles).forEach(function (id) {
      var role = state.roles.roles[id];
      var btn = h('button', 'ycchat-role');
      btn.type = 'button';
      btn.appendChild(h('b', null, L(role, 'label')));
      btn.appendChild(h('span', null, L(role, 'tagline')));
      btn.addEventListener('click', function () { pickRole(id); });
      wrap.appendChild(btn);
    });
    els.body.appendChild(wrap);
    updateClearVisibility();
  }

  // TODO#3: on open / page load, restore the remembered role and its transcript
  // instead of forcing the role picker. Falls back to the picker when no role
  // was chosen yet (or the stored role no longer exists after a rebuild).
  function restoreView() {
    if (state.role && state.roles && state.roles.roles[state.role]) {
      var role = state.roles.roles[state.role];
      els.roleChip.textContent = L(role, 'label');
      var tr = transcript(state.role);
      if (!tr.log.length) {
        pushLog({ type: 'note', text: t('viewingAs', L(role, 'label')) });
        pushLog({ type: 'starters' });
      }
      state.history = tr.history;
      renderLog();
      updateClearVisibility();
    } else {
      showRolePicker();
    }
  }

  // TODO#3: picking a role restores that role's stored transcript instead of
  // wiping history. Switching between two roles hops between two conversations;
  // switching to a fresh role starts one. The active role is remembered too.
  function pickRole(id) {
    state.role = id;
    var tr = transcript(id);
    state.history = tr.history;
    var role = state.roles.roles[id];
    els.roleChip.textContent = L(role, 'label');
    logTurn({ event: 'role_selected', role: id });
    if (!tr.log.length) {
      pushLog({ type: 'note', text: t('viewingAs', L(role, 'label')) });
      pushLog({ type: 'starters' });
    }
    renderLog();
    updateClearVisibility();
    saveSession();
    els.input.focus();
  }

  // Wipe the active role's conversation and start it over (the "clear chat"
  // affordance). Only the current role is cleared; other roles keep theirs.
  function clearChat() {
    if (!state.role) return;
    var tr = transcript(state.role);
    tr.log = [];
    tr.history.length = 0;
    logTurn({ event: 'chat_cleared', role: state.role });
    var role = state.roles.roles[state.role];
    pushLog({ type: 'note', text: t('viewingAs', L(role, 'label')) });
    pushLog({ type: 'starters' });
    renderLog();
    els.input.focus();
  }

  function updateClearVisibility() {
    if (els.clearBtn) els.clearBtn.style.display = state.role ? '' : 'none';
  }

  function addStarters(role) {
    var starters = h('div', 'ycchat-starters');
    (L(role, 'starters') || []).forEach(function (q) {
      var chip = h('button', 'ycchat-starter', q);
      chip.type = 'button';
      chip.addEventListener('click', function () { send(q); });
      starters.appendChild(chip);
    });
    els.body.appendChild(starters);
    els.body.scrollTop = els.body.scrollHeight;
  }

  async function send(question) {
    question = (question || els.input.value).trim();
    if (!question || state.busy || !state.role) return;
    els.input.value = '';
    state.busy = true;
    els.send.disabled = true;
    addMsg('user', question);
    pushLog({ type: 'user', text: question }); // TODO#3: persist as it happens
    var thinking = addMsg('bot', '');
    thinking.classList.add('ycchat-dots');

    try {
      // Name-blind gate input: strip YC's name unless the remainder is a
      // bio-intent stub (a genuine question about him).
      var stripped = stripName(question);

      var record = { event: 'turn', role: state.role, question: question };
      var gateText = gateForm(question, stripped);
      var emb;
      try {
        emb = await embedQuery(question, gateText);
      } catch (embErr) {
        if (String(embErr && embErr.message).indexOf('embedding service unavailable') !== 0) throw embErr;
        await degradedTurn(question, stripped, thinking, record);
        return;
      }
      record.embed_rid = emb.rid || undefined; // correlate with the server /embed log

      // emb.vector is populated only on the LOCAL embedding path (light
      // mode, or backend-down-but-local-model-matches) — normal mode's
      // retrieval now happens server-side inside /chat (Task 29), so there
      // is nothing to score locally there; results come from /chat's
      // `sources` instead, once the gate (below) has passed.
      var localRetrieval = emb.vector ? retrieve(emb.vector) : null;
      if (localRetrieval) {
        record.retrieved = localRetrieval.results.map(function (r) { return { id: r.chunk.id, score: +r.score.toFixed(3) }; });
      }

      // Off-topic gate: refuse before any retrieval/LLM call. Three cases:
      // the backend judged it (server-side gate_only /embed call), the index
      // expects a remote gate that wasn't available (fail open — the LLM
      // prompt still refuses), or the classic local gate on this browser's
      // own in-memory index.
      var refused = false;
      if (emb.gate) {
        refused = !emb.gate.pass;
        record.gate = { remote: true, value: emb.gate.value, reason: emb.gate.reason, stripped: stripped || undefined };
      } else if (state.meta.gate_remote) {
        record.gate = { remote: true, unavailable: true };
      } else {
        // Gate on gateForm()'s output via its OWN local embedding (may differ
        // from `question` itself when the name was stripped/normalized) —
        // mirrors runtime.py / functions/tencent/index.py's gate_form. Always
        // a genuine local embedding (not embedQuery), so this works even in
        // the edge case where the backend is reachable enough to answer
        // gate_only /embed calls but has no gate model packaged.
        var gateScore = gateValue(retrieve(await localEmbed(gateText)).stats);
        if (stripped) record.gate = { stripped: stripped, score: +gateScore.toFixed(3) };
        refused = gateScore < gateThreshold();
      }
      if (refused) {
        record.mode = 'off_topic_refused';
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = t('refused');
        addStarters(state.roles.roles[state.role]);
        pushLog({ type: 'bot', text: thinking.textContent });
        pushLog({ type: 'starters' });
        record.answer = thinking.textContent;
        logTurn(record);
        return;
      }

      var answer, results;
      if (WORKER_URL) {
        record.mode = 'llm';
        var resp;
        try {
          resp = await askWorker(question);
        } catch (chatErr) {
          if (String(chatErr && chatErr.message).indexOf('rate limited') === 0) throw chatErr;
          // Log before falling into degradedTurn -- otherwise a /chat
          // failure left no diagnostic anywhere (degradedTurn's own
          // logTurn calls describe the offline-consent flow, not why this
          // turn ended up there).
          logTurn({ event: 'error', role: state.role, question: question, error: String(chatErr) });
          // A gate_only /embed call can succeed even when /chat itself is
          // down or misconfigured (e.g. an old zip with no bundled index,
          // Task 29) — treat that the same as an unreachable backend.
          await degradedTurn(question, stripped, thinking, record);
          return;
        }
        record.rid = resp.rid || undefined; // server /chat log id, for correlation
        // /chat's `refused` field (Task 29): the function's own retrieval
        // found nothing above MIN_SCORE, so it answered 200 with a canned
        // refusal and never called the LLM. askWorker's return comment has
        // always declared this field; nothing read it, so the visitor got
        // index.py's REFUSAL -- a hardcoded ENGLISH string -- even with
        // lang() === 'zh', with no starters and no page links (a dead end),
        // and record.mode stayed 'llm' so neither GA nor the transcript
        // could tell a refusal from an answer. This is the same situation
        // degraded mode already handles honestly (degradedNoSources +
        // addPageLinks); handle it here the way the client-side gate
        // refusal below does, in the visitor's own language.
        if (resp.refused) {
          record.mode = 'off_topic_refused';
          // Distinguishes this from the client-side gate refusal, which
          // never reaches the backend at all. The server logged its own
          // chat_refused line under the same rid.
          record.refused_by = 'server_retrieval_empty';
          thinking.classList.remove('ycchat-dots');
          thinking.textContent = t('refused');
          addStarters(state.roles.roles[state.role]);
          pushLog({ type: 'bot', text: thinking.textContent });
          pushLog({ type: 'starters' });
          record.answer = thinking.textContent;
          logTurn(record);
          return;
        }
        results = resultsFromSources(resp.sources);
        record.retrieved = results.map(function (r) { return { id: r.chunk.id, score: +r.score.toFixed(3) }; });
        answer = normalizeAnswer(resp.answer);
        if (!answer) throw new Error('the model returned an empty answer');
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = answer;
      } else {
        record.mode = 'retrieval-only';
        results = localRetrieval ? localRetrieval.results : [];
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = t('retrievalOnly');
        answer = thinking.textContent;
      }
      addSources(results);
      state.history.push({ role: 'user', content: question }, { role: 'assistant', content: answer });
      if (state.history.length > HISTORY_CAP) state.history.splice(0, state.history.length - HISTORY_CAP);
      pushLog({ type: 'bot', text: answer });
      if (results.length) pushLog({ type: 'sources', results: sourcesForLog(results) });
      record.answer = answer;
      logTurn(record);
    } catch (err) {
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = t('somethingWrong', (err && err.message || err));
      pushLog({ type: 'bot', text: thinking.textContent }); // keep the transcript paired
      logTurn({ event: 'error', role: state.role, question: question, error: String(err) });
    } finally {
      state.busy = false;
      els.send.disabled = false;
      els.input.focus();
    }
  }

  function buildPanel() {
    els.panel = h('div', 'ycchat-panel');
    els.panel.setAttribute('role', 'dialog');
    els.panel.setAttribute('aria-label', t('panelAria'));

    var head = h('div', 'ycchat-head');
    els.header = h('b', null, t('header'));
    head.appendChild(els.header);
    els.roleChip = h('button', 'ycchat-rolechip', t('chooseRole'));
    els.roleChip.type = 'button';
    els.roleChip.title = t('roleChipTitle');
    els.roleChip.addEventListener('click', function () { if (state.roles) showRolePicker(); });
    head.appendChild(els.roleChip);
    els.clearBtn = h('button', 'ycchat-clearchip', t('clearLabel')); // TODO#3
    els.clearBtn.type = 'button';
    els.clearBtn.title = t('clearTitle');
    els.clearBtn.style.display = state.role ? '' : 'none';
    els.clearBtn.addEventListener('click', function () { if (state.role) clearChat(); });
    head.appendChild(els.clearBtn);
    var x = h('button', 'ycchat-x', '✕');
    els.closeBtn = x;
    x.type = 'button';
    x.setAttribute('aria-label', t('closeAria'));
    x.addEventListener('click', toggle);
    head.appendChild(x);

    els.body = h('div', 'ycchat-body');
    els.body.setAttribute('aria-live', 'polite');

    var foot = h('div', 'ycchat-foot');
    els.input = h('input', 'ycchat-in');
    els.input.type = 'text';
    els.input.placeholder = t('placeholder');
    els.input.maxLength = 500;
    els.input.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(); });
    els.send = h('button', 'ycchat-send', t('send'));
    els.send.type = 'button';
    els.send.addEventListener('click', function () { send(); });
    foot.appendChild(els.input);
    foot.appendChild(els.send);

    els.panel.appendChild(head);
    els.panel.appendChild(els.body);
    els.panel.appendChild(foot);

    // Keys typed inside the chat belong to the chat: stop them from reaching
    // page-level shortcuts (the landing page navigates on Enter / arrows).
    els.panel.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') toggle();
      e.stopPropagation();
    });

    document.body.appendChild(els.panel);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && state.open) toggle();
    });
  }

  function toggle() {
    if (!els.panel) buildPanel();
    state.open = !state.open;
    els.panel.style.display = state.open ? 'flex' : 'none';
    els.btn.setAttribute('aria-expanded', String(state.open));
    if (!state.open) return;

    if (!state.roles) {
      els.body.textContent = '';
      // file:// blocks module imports and fetch — the widget needs a web
      // server. Explain instead of failing with a cryptic import error.
      if (window.location.protocol === 'file:') {
        addMsg('note', t('fileProtocol'));
        return;
      }
      var needLocalModel = !WORKER_URL; // backend embeds server-side when configured
      var status = addMsg('note', needLocalModel ? t('loadingModel') : t('loadingGeneric'));
      var ready = loadCore();
      if (needLocalModel) {
        ready = ready.then(function () {
          return ensureExtractor(function (pct) {
            status.textContent = t('loadingModelPct', pct);
          });
        });
      }
      ready.then(function () {
        logTurn({ event: 'assets_loaded', remote_embed: !!WORKER_URL });
        restoreView();
      }).catch(function (err) {
        status.textContent = t('assetsFail', (err && err.message || err));
        logTurn({ event: 'error', error: String(err) });
      });
    }
  }

  // Re-localize every already-rendered piece of chrome when the site toggle
  // fires. Live conversation messages keep their original language; only the
  // role picker (if that's the current view) re-renders.
  function applyLang() {
    if (els.btn) {
      els.btn.textContent = t('askBtn');
      els.btn.setAttribute('aria-label', t('openAria'));
    }
    if (!els.panel) return;
    els.panel.setAttribute('aria-label', t('panelAria'));
    if (els.header) els.header.textContent = t('header');
    els.roleChip.title = t('roleChipTitle');
    els.roleChip.textContent = (state.role && state.roles)
      ? L(state.roles.roles[state.role], 'label')
      : t('chooseRole');
    if (els.clearBtn) { els.clearBtn.textContent = t('clearLabel'); els.clearBtn.title = t('clearTitle'); }
    updateClearVisibility();
    if (els.closeBtn) els.closeBtn.setAttribute('aria-label', t('closeAria'));
    els.input.placeholder = t('placeholder');
    els.send.textContent = t('send');
    if (state.open && state.roles && !state.role) showRolePicker();
  }

  function init() {
    loadSession(); // TODO#3: restore per-role transcripts + session id for this tab
    injectStyles();
    els.btn = h('button', 'ycchat-btn', t('askBtn'));
    els.btn.type = 'button';
    els.btn.setAttribute('aria-expanded', 'false');
    els.btn.setAttribute('aria-label', t('openAria'));
    els.btn.addEventListener('click', toggle);
    document.body.appendChild(els.btn);
    window.addEventListener('yc-langchange', applyLang);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
