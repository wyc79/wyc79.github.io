// Shared masthead for every inner page - injects the brand, the numbered
// section index, contact links and the language / theme toggles into
// <header class="page-head" data-page-header></header>.
// Must load BEFORE theme.js so its DOMContentLoaded handler registers first and
// the #themeToggle button exists when theme.js wires it up. Load i18n.js AFTER
// this file so the injected header (with its data-lang spans) is present when
// i18n's DOMContentLoaded handler translates and wires #languageToggle.
(function () {
  'use strict';

  // Section order and numbering follow the landing menu (scripts/p3-menu.js).
  var SECTIONS = [
    { file: 'projects.html',     en: 'Projects',     zh: '项目' },
    { file: 'skills.html',       en: 'Skills',       zh: '技能' },
    { file: 'education.html',    en: 'Education',    zh: '教育' },
    { file: 'publications.html', en: 'Publications', zh: '论文' },
    { file: 'agents.html',       en: 'Agents',       zh: '智能体' },
    { file: 'toolbox.html',      en: 'Toolbox',      zh: '工具箱' }
  ];

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  function render() {
    var host = document.querySelector('[data-page-header]');
    if (!host) return;

    // Pages under /pages/ need ../ to reach site-root assets; anything at the
    // root (e.g. future top-level pages reusing this include) uses no prefix.
    var inSubpage = /\/pages\//.test(window.location.pathname);
    var prefix = inSubpage ? '../' : '';
    var pagesPrefix = inSubpage ? '' : 'pages/';

    // A project detail page belongs to the Projects section.
    var file = window.location.pathname.split('/').pop() || 'index.html';
    var inProject = !!document.querySelector('.project-hero');

    var nav = SECTIONS.map(function (s, i) {
      var current = s.file === file ? ' aria-current="page"'
        : (inProject && s.file === 'projects.html' ? ' aria-current="true"' : '');
      return '<li><a href="' + pagesPrefix + s.file + '"' + current + '>' +
        '<span class="mast-num">' + pad(i + 1) + '</span>' +
        '<span><span data-lang="en">' + s.en + '</span><span data-lang="zh">' + s.zh + '</span></span>' +
        '</a></li>';
    }).join('');

    host.innerHTML = [
      '<div class="mast">',
        '<a class="mast-brand" href="', prefix, 'index.html" aria-label="Back to landing page">',
          '<img class="mast-logo" src="', prefix, 'images/wyc.png" alt="Yuanchen Wang logo" />',
          '<span>',
            '<span class="mast-name">YUANCHEN WANG</span>',
            '<span class="mast-sub">',
              '<span data-lang="en">GAME&nbsp;DEVELOPER&nbsp;&middot;&nbsp;USC&nbsp;MSCS</span>',
              '<span data-lang="zh">游戏开发者&nbsp;&middot;&nbsp;USC&nbsp;计算机科学硕士生</span>',
            '</span>',
          '</span>',
        '</a>',
        '<nav class="mast-nav" aria-label="Sections"><ol>', nav, '</ol></nav>',
        '<div class="mast-links">',
          '<a href="mailto:ywang217@usc.edu">EMAIL</a>',
          '<a href="https://www.linkedin.com/in/yuanchen-wang-9b1854271" target="_blank" rel="noopener">LINKEDIN</a>',
          '<a href="https://github.com/wyc79" target="_blank" rel="noopener">GITHUB</a>',
        '</div>',
        '<div class="mast-utils">',
          // data-fixed-label: i18n.js leaves this label alone; CSS marks the
          // active option from <html lang>.
          '<button type="button" id="languageToggle" class="mast-toggle" data-fixed-label aria-label="Change language">',
            '<span data-opt="en" lang="en">EN</span><span class="mast-slash" aria-hidden="true">/</span><span data-opt="zh" lang="zh">中文</span>',
          '</button>',
          '<button type="button" id="themeToggle" class="mast-toggle" aria-label="Toggle theme">',
            '<span data-opt="light"><span data-lang="en">LIGHT</span><span data-lang="zh">浅色</span></span>',
            '<span class="mast-slash" aria-hidden="true">/</span>',
            '<span data-opt="dark"><span data-lang="en">DARK</span><span data-lang="zh">深色</span></span>',
          '</button>',
        '</div>',
      '</div>'
    ].join('');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
