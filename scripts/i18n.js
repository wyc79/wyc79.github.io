// Lightweight bilingual (EN / 中文) layer.
// No dictionary file: every translatable string lives in the HTML as a pair of
// elements marked data-lang="en" / data-lang="zh" (the engine shows the active
// one and hides the other). A few single-element cases (the <title>, form
// placeholders, button labels) use data-en / data-zh (and data-ph-en /
// data-ph-zh, data-al-en / data-al-zh) attribute pairs instead.
//
// Driven by the existing #languageToggle button. Choice is persisted in
// localStorage under 'yc-lang'. Default language is English.
//
// Load this AFTER scripts/page-header.js so the injected header exists when the
// DOMContentLoaded handler runs and can be translated + wired.
(function () {
  'use strict';

  var KEY = 'yc-lang';
  var LANGS = ['en', 'zh'];

  function saved() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(l) {
    try { localStorage.setItem(KEY, l); } catch (e) {}
  }
  function current() {
    var s = saved();
    return (s === 'en' || s === 'zh') ? s : 'en';
  }

  // --- No-flash pre-style: runs while the <head> parses, before the body does.
  // Hides the *other* language via CSS so a zh visitor never sees an English
  // flash (and vice versa). JS later sets inline display, which overrides this.
  (function preHide() {
    var lang = current();
    var other = lang === 'en' ? 'zh' : 'en';
    var style = document.createElement('style');
    style.id = 'yc-i18n-prehide';
    style.textContent = '[data-lang="' + other + '"]{display:none}';
    (document.head || document.documentElement).appendChild(style);
    document.documentElement.setAttribute('lang', lang);
  })();

  function apply(lang) {
    if (LANGS.indexOf(lang) < 0) lang = 'en';
    document.documentElement.setAttribute('lang', lang);

    // Once JS is in control, drop the no-flash pre-style so that setting an
    // element's inline display back to '' reveals it instead of falling through
    // to the pre-style's `display:none` rule (which would re-hide the active
    // language after a toggle).
    var pre = document.getElementById('yc-i18n-prehide');
    if (pre && pre.parentNode) pre.parentNode.removeChild(pre);

    // 1. Dual-language element pairs — show the active language, hide the other.
    var blocks = document.querySelectorAll('[data-lang]');
    for (var i = 0; i < blocks.length; i++) {
      var l = blocks[i].getAttribute('data-lang');
      if (l !== 'en' && l !== 'zh') continue;
      blocks[i].style.display = (l === lang) ? '' : 'none';
    }

    // 2. Single-element text swaps (used for <title>, buttons): data-en / data-zh.
    var texts = document.querySelectorAll('[data-en][data-zh]');
    for (var j = 0; j < texts.length; j++) {
      texts[j].textContent = texts[j].getAttribute('data-' + lang);
    }

    // 3. Placeholder swaps: data-ph-en / data-ph-zh.
    var phs = document.querySelectorAll('[data-ph-en][data-ph-zh]');
    for (var k = 0; k < phs.length; k++) {
      phs[k].setAttribute('placeholder', phs[k].getAttribute('data-ph-' + lang));
    }

    // 4. aria-label swaps: data-al-en / data-al-zh.
    var als = document.querySelectorAll('[data-al-en][data-al-zh]');
    for (var m = 0; m < als.length; m++) {
      als[m].setAttribute('aria-label', als[m].getAttribute('data-al-' + lang));
    }

    // 5. Language toggle button shows the language you'd switch TO.
    var btn = document.getElementById('languageToggle');
    if (btn) btn.textContent = (lang === 'en') ? '中文' : 'EN';
  }

  function set(lang) {
    if (LANGS.indexOf(lang) < 0) lang = 'en';
    store(lang);
    apply(lang);
    // Let other components (e.g. the chat widget) re-localize on toggle.
    try { window.dispatchEvent(new CustomEvent('yc-langchange', { detail: lang })); } catch (e) {}
  }

  function init() {
    apply(current());
    var btn = document.getElementById('languageToggle');
    if (btn) {
      btn.addEventListener('click', function () {
        set(current() === 'en' ? 'zh' : 'en');
        btn.blur();
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Exposed so other scripts (e.g. a page that injects content later) can
  // re-apply the current language after adding DOM.
  window.YCI18N = {
    current: current,
    set: set,
    apply: function () { apply(current()); }
  };
})();
