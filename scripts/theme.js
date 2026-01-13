// Theme initialization and toggle functionality
(function() {
  'use strict';

  const root = document.documentElement;
  const saved = localStorage.getItem('yc-theme');

  // Initialize theme on page load
  if (saved === 'light' || saved === 'dark') {
    root.setAttribute('data-theme', saved);
  } else {
    const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
    const defaultTheme = prefersDark ? 'dark' : 'light';
    root.setAttribute('data-theme', defaultTheme);
    localStorage.setItem('yc-theme', defaultTheme);
  }

  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;

    function currentTheme() {
      const v = root.getAttribute('data-theme');
      if (v === 'auto') {
        return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }
      return v;
    }

    function setTheme(mode) {
      root.setAttribute('data-theme', mode);
      localStorage.setItem('yc-theme', mode);
    }

    btn.addEventListener('click', () => {
      const cur = currentTheme();
      setTheme(cur === 'dark' ? 'light' : 'dark');
      btn.blur();
    });

    // React to OS theme changes if the user hasn't set a preference explicitly
    if (!saved) {
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        setTheme(e.matches ? 'dark' : 'light');
      });
    }
  });
})();
