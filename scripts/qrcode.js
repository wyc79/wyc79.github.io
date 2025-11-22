(function () {
  'use strict';

  const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
  const panelByButton = new Map();

  for (const button of tabButtons) {
    const targetId = button.getAttribute('data-tab-target');
    if (!targetId) continue;
    const panel = document.getElementById(targetId);
    if (panel) {
      panelByButton.set(button, panel);
    }
  }

  function setActiveTab(button, { focus = false } = {}) {
    if (!button || !panelByButton.has(button)) return;
    for (const btn of tabButtons) {
      const isSelected = btn === button;
      btn.setAttribute('aria-selected', isSelected ? 'true' : 'false');
      btn.tabIndex = isSelected ? 0 : -1;
      const panel = panelByButton.get(btn);
      if (!panel) continue;
      if (isSelected) {
        panel.removeAttribute('hidden');
      } else if (!panel.hasAttribute('hidden')) {
        panel.setAttribute('hidden', '');
      }
    }
    if (focus) {
      button.focus();
    }
  }

  const initialTab =
    tabButtons.find(btn => btn.getAttribute('aria-selected') === 'true' && panelByButton.has(btn)) ||
    tabButtons.find(btn => panelByButton.has(btn));
  if (initialTab) {
    setActiveTab(initialTab);
  }

  tabButtons.forEach((button, index) => {
    if (!panelByButton.has(button)) return;
    button.addEventListener('click', () => setActiveTab(button));
    button.addEventListener('keydown', event => {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault();
        const direction = event.key === 'ArrowRight' ? 1 : -1;
        let nextIndex = index;
        let attempts = tabButtons.length;
        while (attempts-- > 0) {
          nextIndex = (nextIndex + direction + tabButtons.length) % tabButtons.length;
          const nextButton = tabButtons[nextIndex];
          if (panelByButton.has(nextButton)) {
            setActiveTab(nextButton, { focus: true });
            break;
          }
        }
        return;
      }
      if (event.key === 'Home') {
        event.preventDefault();
        const first = tabButtons.find(btn => panelByButton.has(btn));
        if (first) setActiveTab(first, { focus: true });
        return;
      }
      if (event.key === 'End') {
        event.preventDefault();
        const reversed = [...tabButtons].reverse();
        const last = reversed.find(btn => panelByButton.has(btn));
        if (last) setActiveTab(last, { focus: true });
      }
    });
  });

  const qr = {
    input: document.getElementById('qr-input'),
    size: document.getElementById('qr-size'),
    level: document.getElementById('qr-level'),
    generate: document.getElementById('qr-generate'),
    download: document.getElementById('qr-download'),
    output: document.getElementById('qr-output'),
  };

  const hasQrElements = Object.values(qr).every(Boolean);

  if (hasQrElements) {
    const MIN_SIZE = 64;
    const MAX_SIZE = 1024;
    const DEFAULT_SIZE = 256;

    function clampSize(value) {
      const numeric = Math.round(Number(value));
      if (!Number.isFinite(numeric)) {
        return DEFAULT_SIZE;
      }
      return Math.min(MAX_SIZE, Math.max(MIN_SIZE, numeric));
    }

    function showMessage(message) {
      qr.output.innerHTML = `<p class="meta" style="text-align:center;">${message}</p>`;
      qr.download.disabled = true;
    }

    function getLevelConstant(levelKey) {
      if (!window.QRCode || !window.QRCode.CorrectLevel) {
        return null;
      }
      return window.QRCode.CorrectLevel[levelKey] ?? null;
    }

    function ensureLibrary() {
      if (window.QRCode) return true;
      showMessage('QR library failed to load.');
      return false;
    }

    function getDataUrl() {
      const canvas = qr.output.querySelector('canvas');
      if (canvas && typeof canvas.toDataURL === 'function') {
        return canvas.toDataURL('image/png');
      }
      const img = qr.output.querySelector('img');
      if (img && typeof img.src === 'string' && img.src.startsWith('data:image')) {
        return img.src;
      }
      return null;
    }

    function enableDownloadWhenReady() {
      qr.download.disabled = true;
      requestAnimationFrame(() => {
        const dataUrl = getDataUrl();
        if (dataUrl) {
          qr.download.disabled = false;
        } else {
          setTimeout(() => {
            if (getDataUrl()) {
              qr.download.disabled = false;
            }
          }, 100);
        }
      });
    }

    function generateQr() {
      const value = (qr.input.value || '').trim();
      if (!value) {
        showMessage('Enter text to generate a QR code.');
        return;
      }
      if (!ensureLibrary()) return;

      const size = clampSize(qr.size.value || DEFAULT_SIZE);
      if (String(size) !== String(qr.size.value)) {
        qr.size.value = size;
      }

      const levelKey = (qr.level.value || 'M').toUpperCase();
      const levelConstant = getLevelConstant(levelKey);
      if (!levelConstant) {
        showMessage('Error correction level is unavailable.');
        return;
      }

      qr.output.innerHTML = '';
      try {
        new window.QRCode(qr.output, {
          text: value,
          width: size,
          height: size,
          colorDark: '#000000',
          colorLight: '#ffffff',
          correctLevel: levelConstant,
        });
        enableDownloadWhenReady();
      } catch (error) {
        console.error('[toolbox] Failed to generate QR code', error);
        showMessage('Failed to generate QR code. Please try again.');
      }
    }

    function handleDownload(event) {
      event.preventDefault();
      const dataUrl = getDataUrl();
      if (!dataUrl) return;
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = 'qr-code.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    qr.generate.addEventListener('click', generateQr);
    qr.download.addEventListener('click', handleDownload);
    qr.input.addEventListener('input', () => {
      qr.download.disabled = true;
    });
    qr.size.addEventListener('change', () => {
      const size = clampSize(qr.size.value || DEFAULT_SIZE);
      if (String(size) !== String(qr.size.value)) {
        qr.size.value = size;
      }
    });
    qr.input.addEventListener('keydown', event => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        generateQr();
      }
    });
  }
})();

