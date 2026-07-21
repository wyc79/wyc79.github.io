// Shared sub-page header — injects the same .p3-brand / .p3-brand-links markup
// into every page that has <header class="page-head" data-page-header></header>.
// Must load BEFORE theme.js so its DOMContentLoaded handler registers first and
// the #themeToggle button exists when theme.js wires it up. Load i18n.js AFTER
// this file so the injected header (with its data-lang spans) is present when
// i18n's DOMContentLoaded handler translates and wires #languageToggle.
(function () {
  'use strict';

  function render() {
    var host = document.querySelector('[data-page-header]');
    if (!host) return;

    // Pages under /pages/ need ../ to reach site-root assets; anything at the
    // root (e.g. future top-level pages reusing this include) uses no prefix.
    var inSubpage = /\/pages\//.test(window.location.pathname);
    var prefix = inSubpage ? '../' : '';

    host.innerHTML = [
      '<div class="p3-brand">',
        '<a href="', prefix, 'index.html" aria-label="Back to landing page">',
          '<img src="', prefix, 'images/wyc.png" alt="Yuanchen Wang logo" />',
        '</a>',
        '<div>',
          '<div class="p3-brand-name">YUANCHEN WANG</div>',
          '<div class="p3-brand-sub">',
            '<span data-lang="en">GAME&nbsp;DEVELOPER&nbsp;&middot;&nbsp;USC&nbsp;MSCS</span>',
            '<span data-lang="zh">游戏开发者&nbsp;&middot;&nbsp;USC&nbsp;计算机硕士</span>',
          '</div>',
        '</div>',
        '<button type="button" id="themeToggle" class="p3-theme-toggle" aria-label="Toggle theme">THEME</button>',
        '<button type="button" id="languageToggle" class="p3-theme-toggle" aria-label="Change language">中文</button>',
      '</div>',
      '<div class="p3-brand-links">',
        '<a href="mailto:ywang217@usc.edu">EMAIL</a>',
        '<a href="https://www.linkedin.com/in/yuanchen-wang-9b1854271" target="_blank" rel="noopener">LINKEDIN</a>',
        '<a href="https://github.com/wyc79" target="_blank" rel="noopener">GITHUB</a>',
      '</div>'
    ].join('');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
