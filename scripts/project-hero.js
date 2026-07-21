// Project detail banner enhancements, shared by every page that has a
// <section class="project-hero">:
//   1. Injects a round "back to projects" button (‹) on the LEFT of the banner.
//   2. Builds a compact sticky bar that freezes to the top of the viewport once
//      the big banner scrolls out of view — back button on the left, and
//      "GameName — role · role · role" on the right. Bilingual: it clones the
//      title + first meta-item (the Role/职责 line) in both languages and lets
//      i18n.js show the active one.
// Load AFTER i18n.js so window.YCI18N exists when we re-apply the language.
(function () {
  'use strict';

  var BACK_HREF = 'projects.html';   // pages live in /pages/, so relative is fine
  var ANGLE = '‹';              // ‹  single left-pointing angle

  function langText(el, lang) {
    if (!el) return '';
    var span = el.querySelector('[data-lang="' + lang + '"]');
    return ((span ? span.textContent : el.textContent) || '').trim();
  }

  function gameName(fullTitle) {
    // "Cemented Dreams — Gameplay Systems" -> "Cemented Dreams"
    var i = fullTitle.indexOf('—');            // em dash
    if (i < 0) i = fullTitle.indexOf(' - ');
    return (i > 0 ? fullTitle.slice(0, i) : fullTitle).trim();
  }

  function stripRolePrefix(s) {
    // drop a leading "Role:" / "职责：" label up to the first colon
    return s.replace(/^\s*[^:：]*[:：]\s*/, '').trim();
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : '&quot;';
    });
  }

  function init() {
    var hero = document.querySelector('.project-hero');
    if (!hero) return;

    var titleEl = hero.querySelector('.project-hero__title');
    var roleEl = hero.querySelector('.project-hero__meta-item'); // first = Role/职责

    // 1. Back button inside the expanded banner (left side).
    if (!hero.querySelector('.project-hero__back')) {
      var back = document.createElement('a');
      back.className = 'project-hero__back';
      back.href = BACK_HREF;
      back.setAttribute('aria-label', 'Back to Projects');
      back.setAttribute('title', 'Back to Projects');
      back.innerHTML = ANGLE;
      hero.appendChild(back);
    }

    // 2. Compact sticky bar. Build a "name — roles" string per language.
    var compact = { en: '', zh: '' };
    ['en', 'zh'].forEach(function (lang) {
      var name = titleEl ? gameName(langText(titleEl, lang)) : '';
      var roles = roleEl ? stripRolePrefix(langText(roleEl, lang)) : '';
      compact[lang] = roles ? (name + ' — ' + roles) : name;
    });

    var bar = document.createElement('div');
    bar.className = 'project-hero-bar';
    bar.innerHTML =
      '<a class="project-hero-bar__back" href="' + BACK_HREF + '" aria-label="Back to Projects" title="Back to Projects">' + ANGLE + '</a>' +
      '<div class="project-hero-bar__label">' +
        '<span data-lang="en">' + esc(compact.en) + '</span>' +
        '<span data-lang="zh">' + esc(compact.zh) + '</span>' +
      '</div>';
    document.body.appendChild(bar);

    // Let i18n show the right language inside the freshly-injected bar.
    if (window.YCI18N && typeof window.YCI18N.apply === 'function') {
      window.YCI18N.apply();
    }

    // Show the bar once the big banner has scrolled fully above the viewport.
    var ticking = false;
    function update() {
      ticking = false;
      var passed = hero.getBoundingClientRect().bottom <= 0;
      bar.classList.toggle('visible', passed);
    }
    function onScroll() {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    update();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
