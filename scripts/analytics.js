/* Google Analytics 4 loader
 * Usage: <script src="scripts/analytics.js" data-ga-id="G-XXXXXXXXXX"></script>
 */
(function (window, document, scriptEl) {
  if (!scriptEl) {
    console.warn('analytics.js: unable to locate the current script tag.');
    return;
  }

  var measurementId = scriptEl.dataset.gaId;
  if (!measurementId) {
    console.warn('analytics.js: missing GA measurement ID. Set data-ga-id on the script tag.');
    return;
  }

  if (!window.dataLayer) {
    window.dataLayer = [];
  }

  window.gtag = window.gtag || function () {
    window.dataLayer.push(arguments);
  };

  function loadGtagLibrary() {
    if (document.getElementById('ga4-loader')) {
      return; // Already loaded
    }

    var script = document.createElement('script');
    script.id = 'ga4-loader';
    script.async = true;
    script.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(measurementId);
    document.head.appendChild(script);
  }

  loadGtagLibrary();

  window.gtag('js', new Date());
  window.gtag('config', measurementId);
})(window, document, document.currentScript);

