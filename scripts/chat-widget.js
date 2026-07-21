// Portfolio chat agent — client-side RAG over the static index built by chat/.
//
// Architecture (see chat/README.md):
//   chat/data/index.json   precomputed chunk vectors (build-time, Python)
//   chat/models/           self-hosted MiniLM ONNX — the SAME weights the
//                          build pipeline used, so query and document vectors
//                          share one embedding space
//   scripts/vendor/        self-hosted transformers.js runtime + ORT WASM
//   WORKER_URL             optional Cloudflare Worker for LLM answers; leave
//                          empty for retrieval-only demo mode (no API key)
//
// Everything the widget logs (every user input, retrieval result and answer)
// goes to console.debug, to Google Analytics when present, and — once the
// worker is deployed — to the worker's server-side log.
(function () {
  'use strict';

  // ── Config ────────────────────────────────────────────────────────────
  // Backend base URL (Tencent SCF 函数URL or Cloudflare Worker). Set it here
  // after deploying — see chat/functions/tencent/DEPLOY.md. A page may also
  // predefine window.YC_CHAT_WORKER_URL before this script loads.
  var WORKER_URL = (window.YC_CHAT_WORKER_URL || 'https://1302480548-79ajhb3iyj.ap-guangzhou.tencentscf.com').replace(/\/+$/, '');
  var TOP_K = 4;
  var MIN_SCORE = 0.18;  // per-source display floor
  // Off-topic gate: statistic + threshold are calibrated per embedding model
  // at build time and shipped in index.json (gate_stat, gate_threshold) —
  // see chat/src/portfolio_rag/gate_calibration.py, mirrored in
  // chat/tests/test_gate.py. These constants are only the fallback for old
  // indexes: raw top-score >= 0.22 (the MiniLM calibration).
  var OFFTOPIC_GATE = 0.22;

  function gateThreshold() {
    return (state.index && state.index.gate_threshold) || OFFTOPIC_GATE;
  }

  function statValue(stats, kind) {
    if (kind === 'contrast') return stats.top - stats.mean;
    if (kind === 'zscore') return (stats.top - stats.mean) / (stats.std + 1e-6);
    return stats.top;
  }

  function gateValue(stats) {
    return statValue(stats, (state.index && state.index.gate_stat) || 'top');
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
  var BIO_STUB_RE = /^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b|^$|介绍|简介|谁是|是谁|关于/i;
  var MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

  // Gate input. When the name was stripped (off-topic detection), gate on the
  // remainder. When it was KEPT (a bio question), normalize the name to the
  // gate's own language — so a Chinese question that uses "YC", or an English
  // one that uses 王元辰, still matches the corpus (which is single-language per
  // gate). Retrieval always uses the original, unmodified question.
  function gateForm(question, stripped) {
    if (stripped != null) return stripped;
    return /[㐀-鿿]/.test(question)
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
      backendDown: 'The chat backend is unreachable right now — please try again in a minute.',
      retrievalOnly: 'Demo is in retrieval-only mode (no LLM connected yet), but here\'s what the semantic index surfaces for that — sources below:',
      somethingWrong: function (msg) { return 'Something went wrong (' + msg + '). Please try again.'; },
      offlineConsent: 'The AI answer service is unreachable. I can download a small on-device search model (~23 MB, cached after this) to still find relevant pages — or you can just browse the site.',
      offlineUse: 'Use offline search (~23 MB)',
      offlineBrowse: 'Just browse the site',
      offlineDeclined: 'No problem — here are some pages to explore. Ask again anytime to turn on offline search.',
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
      backendDown: '聊天后端暂时无法连接 —— 请过一会儿再试。',
      retrievalOnly: '演示目前处于「仅检索」模式（还没接入 LLM），不过这是语义索引为该问题找到的内容 —— 来源见下方：',
      somethingWrong: function (msg) { return '出错了（' + msg + '）。请再试一次。'; },
      offlineConsent: 'AI 回答服务暂时无法连接。我可以下载一个很小的本地检索模型（约 23 MB，之后会缓存）来帮你找到相关页面 —— 你也可以直接浏览站点。',
      offlineUse: '使用离线检索（约 23 MB）',
      offlineBrowse: '直接浏览站点',
      offlineDeclined: '没问题 —— 这里有一些页面可以看看。你随时可以再问一次来启用离线检索。',
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
    index: null,
    extractor: null,
    loading: null, // Promise while core assets load
    extractorLoading: null, // Promise while the local model loads
    remoteEmbedDown: false, // /embed said 503 or errored -> stop trying
    busy: false,
    session: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()),
    history: [], // [{role:'user'|'assistant', content}]
    offlineConsent: null, // null = undecided; true/false = this session's choice (TODO#5)
  };

  // Static page suggestions for the "just browse" path — no model, no embedding.
  var SUGGESTED_PAGES = [
    { url: 'pages/projects.html', en: 'Projects', zh: '项目' },
    { url: 'pages/skills.html', en: 'Skills', zh: '技能' },
    { url: 'pages/education.html', en: 'Education', zh: '教育' },
    { url: 'pages/publications.html', en: 'Publications', zh: '论文' },
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
  // Core (roles + index) always loads on open — it's ~600KB. The 23MB local
  // model loads only when actually needed: immediately when there's no
  // backend, lazily as a fallback when the backend embeds server-side.
  function loadCore() {
    if (state.loading) return state.loading;
    state.loading = (async function () {
      // no-cache = revalidate with the server (cheap 304 when unchanged), so
      // visitors never see a stale index/roles after a rebuild is deployed.
      var rolesRes = fetch(PREFIX + 'chat/data/roles.json', { cache: 'no-cache' }).then(function (r) {
        if (!r.ok) throw new Error('roles.json ' + r.status);
        return r.json();
      });
      var indexRes = fetch(PREFIX + 'chat/data/index.json', { cache: 'no-cache' }).then(function (r) {
        if (!r.ok) throw new Error('index.json ' + r.status);
        return r.json();
      });
      state.roles = await rolesRes;
      state.index = await indexRes;
    })();
    return state.loading;
  }

  // The vendored in-browser model only matches indexes built with MiniLM —
  // an e5-built index needs the backend's /embed and has no local fallback.
  function localModelMatchesIndex() {
    return !state.index || /all-MiniLM-L6-v2/.test(state.index.model || '');
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

  // Returns {vector, gate}. gate is non-null only when the backend embeds
  // AND has the gate model packaged — it judges gateText (the name-stripped
  // question) with MiniLM, since e5 can't separate on/off-topic.
  async function embedQuery(text, gateText) {
    // Preferred: the backend embeds (multilingual model, no client download).
    // One retry with a pause covers cold starts and transient hiccups before
    // we give up on the backend for this session.
    if (WORKER_URL && !state.remoteEmbedDown) {
      for (var attempt = 0; attempt < 2; attempt++) {
        try {
          var res = await fetch(WORKER_URL + '/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, gate_text: gateText || text }),
          });
          if (res.ok) {
            var data = await res.json();
            return { vector: new Float32Array(data.vector), gate: data.gate || null, rid: data.rid || null };
          }
          if (res.status === 429) throw new Error('rate limited — please wait a minute and try again');
        } catch (e) {
          if (String(e && e.message).indexOf('rate limited') === 0) throw e;
        }
        if (attempt === 0) await new Promise(function (r) { setTimeout(r, 1500); });
      }
      state.remoteEmbedDown = true;
      logTurn({ event: 'remote_embed_fallback' });
    }
    if (!localModelMatchesIndex()) {
      throw new Error('embedding service unavailable (the search index needs the server-side model)');
    }
    await ensureExtractor(null);
    var out = await state.extractor((state.index.query_prefix || '') + text, { pooling: 'mean', normalize: true });
    return { vector: out.data, gate: null, rid: null }; // Float32Array(384), unit norm
  }

  function retrieve(queryVec) {
    var chunks = state.index.chunks;
    var scored = [], sum = 0, sumSq = 0;
    for (var i = 0; i < chunks.length; i++) {
      var v = chunks[i].vector, s = 0;
      for (var j = 0; j < v.length; j++) s += v[j] * queryVec[j];
      scored.push({ chunk: chunks[i], score: s });
      sum += s;
      sumSq += s * s;
    }
    var n = scored.length || 1;
    var mean = sum / n;
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

  function dedupeForDisplay(results) {
    var seen = {}, out = [];
    results.forEach(function (r) {
      var key = r.chunk.url + '|' + r.chunk.section_title;
      if (!seen[key]) { seen[key] = true; out.push(r); }
    });
    return out;
  }

  // ── Degraded mode: backend down + e5 index ────────────────────────────
  // The published fallback_vectors.json holds MiniLM copies of the chunk
  // vectors (chunk-order aligned with index.json), so the local MiniLM model
  // can still recommend relevant pages even when server-side embedding and
  // LLM answers are unreachable.
  function loadFallbackVectors() {
    if (state.fallback) return Promise.resolve(state.fallback);
    return fetch(PREFIX + 'chat/data/fallback_vectors.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error('fallback index unavailable');
      return r.json();
    }).then(function (fb) { state.fallback = fb; return fb; });
  }

  function retrieveFallback(fb, queryVec) {
    var scored = [], sum = 0, sumSq = 0;
    for (var i = 0; i < fb.vectors.length; i++) {
      var v = fb.vectors[i], s = 0;
      for (var j = 0; j < v.length; j++) s += v[j] * queryVec[j];
      scored.push({ chunk: state.index.chunks[i], score: s });
      sum += s; sumSq += s * s;
    }
    var n = scored.length || 1, mean = sum / n;
    scored.sort(function (a, b) { return b.score - a.score; });
    return {
      results: scored.slice(0, TOP_K).filter(function (r) { return r.score >= MIN_SCORE; }),
      stats: { top: scored.length ? scored[0].score : 0, mean: mean, std: Math.sqrt(Math.max(sumSq / n - mean * mean, 0)) },
    };
  }

  async function degradedTurn(question, stripped, thinking, record) {
    record.mode = 'degraded-local';
    if (/[぀-ヿ㐀-鿿豈-﫿]/.test(question)) {
      // The local fallback model is English-only — be honest for CJK, never
      // download it for a question it can't serve, but still offer static page
      // links (localized) so a Chinese visitor isn't left at a dead end.
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = t('degradedCJK');
      addPageLinks();
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
    record.mode = 'degraded-declined';
    record.answer = thinking.textContent;
    logTurn(record);
  }

  // Consent granted: pull the MiniLM model and do local fallback retrieval.
  async function runOfflineSearch(question, stripped, thinking, record) {
    try {
      var fb = await loadFallbackVectors();
      thinking.classList.add('ycchat-dots');
      await ensureExtractor(function (pct) {
        thinking.textContent = t('degradedLoading', pct);
        thinking.classList.remove('ycchat-dots');
      });
      var embedFb = async function (q) {
        var out = await state.extractor((fb.query_prefix || '') + q, { pooling: 'mean', normalize: true });
        return out.data;
      };
      var retrieved = retrieveFallback(fb, await embedFb(question));
      var gateScore = stripped
        ? statValue(retrieveFallback(fb, await embedFb(stripped)).stats, fb.gate_stat || 'top')
        : statValue(retrieved.stats, fb.gate_stat || 'top');
      thinking.classList.remove('ycchat-dots');
      if (gateScore < (fb.gate_threshold || OFFTOPIC_GATE)) {
        thinking.textContent = t('refused');
        addStarters(state.roles.roles[state.role]);
      } else {
        thinking.textContent = t('degradedSources');
        addSources(retrieved.results);
      }
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
  async function askWorker(question, results) {
    var res = await fetch(WORKER_URL + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session: state.session,
        role: state.role,
        lang: lang(),
        question: question,
        history: state.history.slice(-6),
        contexts: results.map(function (r) {
          return {
            id: r.chunk.id,
            title: r.chunk.page_title + ' — ' + r.chunk.section_title,
            url: r.chunk.url,
            text: r.chunk.text,
            score: +r.score.toFixed(4), // client-side retrieval similarity — the
            // server logs it so 日志查询 shows retrieval quality per chunk.
          };
        }),
      }),
    });
    if (!res.ok) throw new Error('worker ' + res.status);
    return await res.json(); // {answer, model, rid}
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
    if (!results.length) return;
    var wrap = h('div', 'ycchat-srcs');
    dedupeForDisplay(results).forEach(function (r) {
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
  }

  function pickRole(id) {
    state.role = id;
    state.history = [];
    var role = state.roles.roles[id];
    els.roleChip.textContent = L(role, 'label') + ' ⌄';
    els.body.textContent = '';
    logTurn({ event: 'role_selected', role: id });
    addMsg('note', t('viewingAs', L(role, 'label')));
    addStarters(role);
    els.input.focus();
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
    var thinking = addMsg('bot', '');
    thinking.classList.add('ycchat-dots');

    try {
      // Name-blind gate input: strip YC's name unless the remainder is a
      // bio-intent stub (a genuine question about him).
      var stripped = null;
      if (NAME_TEST_RE.test(question)) {
        var remainder = question.replace(NAME_STRIP_RE, ' ')
          .replace(/\s+/g, ' ').trim()
          .replace(/^[\s:;,.!?—-]+|[\s:;,.!?—-]+$/g, '');
        if (!BIO_STUB_RE.test(remainder)) stripped = remainder;
      }

      var record = { event: 'turn', role: state.role, question: question };
      var emb;
      try {
        emb = await embedQuery(question, gateForm(question, stripped));
      } catch (embErr) {
        if (String(embErr && embErr.message).indexOf('embedding service unavailable') !== 0) throw embErr;
        await degradedTurn(question, stripped, thinking, record);
        return;
      }
      record.embed_rid = emb.rid || undefined; // correlate with the server /embed log
      var retrieved = retrieve(emb.vector);
      var results = retrieved.results;
      record.retrieved = results.map(function (r) { return { id: r.chunk.id, score: +r.score.toFixed(3) }; });

      // Off-topic gate: refuse before any LLM call. Three cases: the backend
      // judged it (server-side MiniLM gate), the index expects a remote gate
      // that wasn't available (fail open — the LLM prompt still refuses), or
      // the classic local gate on this index's own scores.
      var refused = false;
      if (emb.gate) {
        refused = !emb.gate.pass;
        record.gate = { remote: true, value: emb.gate.value, reason: emb.gate.reason, stripped: stripped || undefined };
      } else if (state.index.gate_remote) {
        record.gate = { remote: true, unavailable: true };
      } else {
        var gateScore = stripped
          ? gateValue(retrieve((await embedQuery(stripped, stripped)).vector).stats)
          : gateValue(retrieved.stats);
        if (stripped) record.gate = { stripped: stripped, score: +gateScore.toFixed(3) };
        refused = gateScore < gateThreshold();
      }
      if (refused) {
        record.mode = 'off_topic_refused';
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = t('refused');
        addStarters(state.roles.roles[state.role]);
        record.answer = thinking.textContent;
        logTurn(record);
        return;
      }

      var answer;
      if (WORKER_URL) {
        record.mode = 'llm';
        var resp = await askWorker(question, results);
        answer = resp.answer;
        record.rid = resp.rid || undefined; // server /chat log id, for correlation
        // The UI renders plain text; strip stray markdown emphasis the LLM
        // may emit despite the prompt (e.g. **Prime Engine**).
        answer = answer.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/^#+\s+/gm, '');
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = answer;
      } else {
        record.mode = 'retrieval-only';
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = t('retrievalOnly');
        answer = thinking.textContent;
      }
      addSources(results);
      state.history.push({ role: 'user', content: question }, { role: 'assistant', content: answer });
      record.answer = answer;
      logTurn(record);
    } catch (err) {
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = t('somethingWrong', (err && err.message || err));
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
        showRolePicker();
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
      ? L(state.roles.roles[state.role], 'label') + ' ⌄'
      : t('chooseRole');
    if (els.closeBtn) els.closeBtn.setAttribute('aria-label', t('closeAria'));
    els.input.placeholder = t('placeholder');
    els.send.textContent = t('send');
    if (state.open && state.roles && !state.role) showRolePicker();
  }

  function init() {
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
