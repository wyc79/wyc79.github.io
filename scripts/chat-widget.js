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
  var WORKER_URL = ''; // e.g. 'https://portfolio-chat.<account>.workers.dev' after deploy
  var TOP_K = 4;
  var MIN_SCORE = 0.18;  // per-source display floor
  // Off-topic gate: if even the BEST chunk scores below this, the question
  // isn't about YC — refuse locally, never call the LLM. Calibrated against
  // measured scores (on-topic ≥ ~0.30, jokes/weather/homework ≤ ~0.25).
  var OFFTOPIC_GATE = 0.22;
  // Name-dropping inflates similarity ("Yuanchen Wang tell me a joke" scores
  // 0.61), so when the question mentions the name, the gate is applied to the
  // question WITHOUT the name — unless what remains is a bio-intent stub
  // ("who is", "tell me about", empty), which is a legitimate question about
  // YC himself. Retrieval/display still use the full question.
  var NAME_TEST_RE = /\b(yuanchen|wang|yc)(?:'s)?\b/i;
  var NAME_STRIP_RE = /\b(yuanchen|wang|yc)(?:'s)?\b/gi;
  // Prefix match: "who is ... in one paragraph" is still a bio question.
  // Mirrored by chat/tests/test_gate.py — keep the two in sync.
  var BIO_STUB_RE = /^(who\s+is|who'?s|about|tell\s+me\s+(?:more\s+)?about|introduce|what\s+about|more\s+about)\b|^$/i;
  var MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

  var inSubpage = /\/pages\//.test(window.location.pathname);
  var PREFIX = inSubpage ? '../' : '';

  // ── State ─────────────────────────────────────────────────────────────
  var state = {
    open: false,
    role: null,
    roles: null,
    index: null,
    extractor: null,
    loading: null, // Promise while assets load
    busy: false,
    session: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()),
    history: [], // [{role:'user'|'assistant', content}]
  };

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

  // ── Lazy asset loading (model + index + roles) ────────────────────────
  function loadAssets(onProgress) {
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
      var T = await import(new URL(PREFIX + 'scripts/vendor/transformers.min.js', window.location.href).href);
      T.env.allowRemoteModels = false;
      T.env.localModelPath = new URL(PREFIX + 'chat/models/', window.location.href).href;
      T.env.backends.onnx.wasm.wasmPaths = new URL(PREFIX + 'scripts/vendor/', window.location.href).href;
      T.env.backends.onnx.wasm.numThreads = 1;
      var extractor = await T.pipeline('feature-extraction', MODEL_ID, {
        quantized: true,
        progress_callback: function (p) {
          if (p.status === 'progress' && /model_quantized/.test(p.file || '') && p.total) {
            onProgress(Math.round((p.loaded / p.total) * 100));
          }
        },
      });
      state.roles = await rolesRes;
      state.index = await indexRes;
      state.extractor = extractor;
    })();
    return state.loading;
  }

  async function embedQuery(text) {
    var out = await state.extractor(text, { pooling: 'mean', normalize: true });
    return out.data; // Float32Array(384), unit norm
  }

  function retrieve(queryVec) {
    var chunks = state.index.chunks;
    var scored = [];
    for (var i = 0; i < chunks.length; i++) {
      var v = chunks[i].vector, s = 0;
      for (var j = 0; j < v.length; j++) s += v[j] * queryVec[j];
      scored.push({ chunk: chunks[i], score: s });
    }
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.slice(0, TOP_K).filter(function (r) { return r.score >= MIN_SCORE; });
  }

  function dedupeForDisplay(results) {
    var seen = {}, out = [];
    results.forEach(function (r) {
      var key = r.chunk.url + '|' + r.chunk.section_title;
      if (!seen[key]) { seen[key] = true; out.push(r); }
    });
    return out;
  }

  // ── LLM call via worker ───────────────────────────────────────────────
  async function askWorker(question, results) {
    var res = await fetch(WORKER_URL + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session: state.session,
        role: state.role,
        question: question,
        history: state.history.slice(-6),
        contexts: results.map(function (r) {
          return {
            id: r.chunk.id,
            title: r.chunk.page_title + ' — ' + r.chunk.section_title,
            url: r.chunk.url,
            text: r.chunk.text,
          };
        }),
      }),
    });
    if (!res.ok) throw new Error('worker ' + res.status);
    return (await res.json()).answer;
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

  function showRolePicker() {
    els.body.textContent = '';
    addMsg('note', 'Hi! I answer questions about YC, grounded in the pages of this site. Who\'s visiting?');
    var wrap = h('div', 'ycchat-roles');
    Object.keys(state.roles.roles).forEach(function (id) {
      var role = state.roles.roles[id];
      var btn = h('button', 'ycchat-role');
      btn.type = 'button';
      btn.appendChild(h('b', null, role.label));
      btn.appendChild(h('span', null, role.tagline));
      btn.addEventListener('click', function () { pickRole(id); });
      wrap.appendChild(btn);
    });
    els.body.appendChild(wrap);
  }

  function pickRole(id) {
    state.role = id;
    state.history = [];
    var role = state.roles.roles[id];
    els.roleChip.textContent = role.label + ' ⌄';
    els.body.textContent = '';
    logTurn({ event: 'role_selected', role: id });
    addMsg('note', 'Viewing as: ' + role.label + '. Ask me anything about YC — answers link back to the relevant pages.');
    addStarters(role);
    els.input.focus();
  }

  function addStarters(role) {
    var starters = h('div', 'ycchat-starters');
    role.starters.forEach(function (q) {
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
      var qVec = await embedQuery(question);
      var results = retrieve(qVec);
      var record = {
        event: 'turn', role: state.role, question: question,
        retrieved: results.map(function (r) { return { id: r.chunk.id, score: +r.score.toFixed(3) }; }),
      };

      // Off-topic gate: nothing on the site is a decent match, so this chat
      // isn't the right tool for the question. Refuse locally — no LLM call.
      var gateScore = results.length ? results[0].score : 0;
      if (NAME_TEST_RE.test(question)) {
        var stripped = question.replace(NAME_STRIP_RE, ' ')
          .replace(/\s+/g, ' ').trim()
          .replace(/^[\s:;,.!?—-]+|[\s:;,.!?—-]+$/g, '');
        if (!BIO_STUB_RE.test(stripped)) {
          var sResults = retrieve(await embedQuery(stripped));
          gateScore = sResults.length ? sResults[0].score : 0;
          record.gate = { stripped: stripped, score: +gateScore.toFixed(3) };
        }
      }
      if (gateScore < OFFTOPIC_GATE) {
        record.mode = 'off_topic_refused';
        thinking.classList.remove('ycchat-dots');
        thinking.textContent =
          'That doesn\'t look like a question about YC, and that\'s all I can help with here — ' +
          'his projects, skills, education, and publications. Try one of these:';
        addStarters(state.roles.roles[state.role]);
        record.answer = thinking.textContent;
        logTurn(record);
        return;
      }

      var answer;
      if (WORKER_URL) {
        record.mode = 'llm';
        answer = await askWorker(question, results);
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = answer;
      } else {
        record.mode = 'retrieval-only';
        thinking.classList.remove('ycchat-dots');
        thinking.textContent = 'Demo is in retrieval-only mode (no LLM connected yet), but here\'s what the semantic index surfaces for that — sources below:';
        answer = thinking.textContent;
      }
      addSources(results);
      state.history.push({ role: 'user', content: question }, { role: 'assistant', content: answer });
      record.answer = answer;
      logTurn(record);
    } catch (err) {
      thinking.classList.remove('ycchat-dots');
      thinking.textContent = 'Something went wrong (' + (err && err.message || err) + '). Please try again.';
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
    els.panel.setAttribute('aria-label', 'Chat with an AI about YC');

    var head = h('div', 'ycchat-head');
    head.appendChild(h('b', null, 'Ask about YC'));
    els.roleChip = h('button', 'ycchat-rolechip', 'choose role');
    els.roleChip.type = 'button';
    els.roleChip.title = 'Change visitor role';
    els.roleChip.addEventListener('click', function () { if (state.roles) showRolePicker(); });
    head.appendChild(els.roleChip);
    var x = h('button', 'ycchat-x', '✕');
    x.type = 'button';
    x.setAttribute('aria-label', 'Close chat');
    x.addEventListener('click', toggle);
    head.appendChild(x);

    els.body = h('div', 'ycchat-body');
    els.body.setAttribute('aria-live', 'polite');

    var foot = h('div', 'ycchat-foot');
    els.input = h('input', 'ycchat-in');
    els.input.type = 'text';
    els.input.placeholder = 'Ask about projects, skills…';
    els.input.maxLength = 500;
    els.input.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(); });
    els.send = h('button', 'ycchat-send', 'Send');
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

    if (!state.extractor) {
      els.body.textContent = '';
      // file:// blocks module imports and fetch — the widget needs a web
      // server. Explain instead of failing with a cryptic import error.
      if (window.location.protocol === 'file:') {
        addMsg('note',
          'The chat can\'t run from a file:// page (browsers block the model and index from loading). ' +
          'Serve the site locally instead:\n\npython -m http.server 8000\n\n' +
          'then open http://localhost:8000 — or just use the live site.');
        return;
      }
      var status = addMsg('note', 'Loading the on-device search model (~23 MB, cached after the first visit)…');
      loadAssets(function (pct) {
        status.textContent = 'Loading the on-device search model… ' + pct + '%';
      }).then(function () {
        logTurn({ event: 'assets_loaded' });
        showRolePicker();
      }).catch(function (err) {
        status.textContent = 'Could not load the chat assets (' + (err && err.message || err) + ').';
        logTurn({ event: 'error', error: String(err) });
      });
    }
  }

  function init() {
    injectStyles();
    els.btn = h('button', 'ycchat-btn', '✦ ASK AI');
    els.btn.type = 'button';
    els.btn.setAttribute('aria-expanded', 'false');
    els.btn.setAttribute('aria-label', 'Open AI chat about YC');
    els.btn.addEventListener('click', toggle);
    document.body.appendChild(els.btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
