(function () {
  const images = [
    { key: 'nothing', src: 'images/nothing_thumbnail.jpg', alt: 'Nothing Can Go Wrong — screenshot' },
    { key: 'gyrotris', src: 'images/gyrotris_thumbnail.png', alt: 'Gyrotris — screenshot' },
    { key: 'ad', src: 'images/AD_thumbnail.png', alt: 'Automatic Differentiation Toolbox' },
    { key: 'sword', src: 'images/sword_thumbnail.png', alt: 'Aegis Sword — render' },
    { key: 'cemented-dreams', src: 'images/CD_thumbnail.png', alt: 'Cemented Dreams — screenshot' },
    { key: 'codebreaker', src: 'images/codebreaker_thumbnail.png', alt: 'CodeBreaker — screenshot' },
    { key: 'workshop', src: 'images/ctin488_thumbnail.png', alt: 'Game Design Workshop' },
    { key: '3Drendering', src: 'images/3Drendering_thumbnail.png', alt: '3D Rendering Project Video' }
  ];

  function initFlipGame() {
    const grid = document.getElementById('flip-grid');
    if (!grid) return;

    const movesEl = document.getElementById('flip-moves');
    const timeEl = document.getElementById('flip-time');
    const resetBtn = document.getElementById('flip-reset');
    const msgEl = document.getElementById('flip-message');

    let firstCard = null;
    let secondCard = null;
    let lock = false;
    let moves = 0;
    let matchedPairs = 0;
    let startTime = null;
    let timerId = null;

    function updateMoves() {
      if (movesEl) movesEl.textContent = 'Moves: ' + moves;
    }

    function updateTime() {
      if (!timeEl) return;
      if (!startTime) {
        timeEl.textContent = 'Time: 0.00s';
        return;
      }
      const secs = ((Date.now() - startTime) / 1000).toFixed(2);
      timeEl.textContent = 'Time: ' + secs + 's';
    }

    function startTimerIfNeeded() {
      if (timerId) return;
      startTime = Date.now();
      timerId = setInterval(updateTime, 10);
      updateTime();
    }

    function stopTimer() {
      if (timerId) {
        clearInterval(timerId);
        timerId = null;
      }
    }

    function shuffle(array) {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    }

    function buildDeck() {
      const pairs = images.flatMap((img) => [{ ...img }, { ...img }]);
      return shuffle(pairs);
    }

    function createCard(cardData) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'flip-card';
      btn.setAttribute('data-key', cardData.key);
      btn.setAttribute('aria-label', 'Hidden card');
      btn.innerHTML =
        '<div class="flip-card-inner">' +
        '<div class="flip-card-front"><span aria-hidden="true">?</span></div>' +
        '<div class="flip-card-back"><img src="' + cardData.src + '" alt="' + cardData.alt + '"></div>' +
        '</div>';
      btn.addEventListener('click', () => onCardClick(btn));
      return btn;
    }

    function resetGame() {
      lock = false;
      firstCard = null;
      secondCard = null;
      moves = 0;
      matchedPairs = 0;
      updateMoves();
      stopTimer();
      startTime = null;
      updateTime();
      if (msgEl) msgEl.hidden = true;
      grid.innerHTML = '';
      const deck = buildDeck();
      deck.forEach((data) => grid.appendChild(createCard(data)));
    }

    function onCardClick(btn) {
      if (lock) return;
      if (btn.classList.contains('is-flipped') || btn.classList.contains('is-matched')) return;

      startTimerIfNeeded();
      btn.classList.add('is-flipped');

      if (!firstCard) {
        firstCard = btn;
        return;
      }

      secondCard = btn;
      moves += 1;
      updateMoves();

      const isMatch = firstCard.getAttribute('data-key') === secondCard.getAttribute('data-key');
      if (isMatch) {
        firstCard.classList.add('is-matched');
        secondCard.classList.add('is-matched');
        firstCard.disabled = true;
        secondCard.disabled = true;
        firstCard = null;
        secondCard = null;
        matchedPairs += 1;
        if (matchedPairs === images.length) {
          stopTimer();
          if (msgEl) msgEl.hidden = false;
        }
      } else {
        lock = true;
        setTimeout(() => {
          if (firstCard) firstCard.classList.remove('is-flipped');
          if (secondCard) secondCard.classList.remove('is-flipped');
          firstCard = null;
          secondCard = null;
          lock = false;
        }, 700);
      }
    }

    if (resetBtn) resetBtn.addEventListener('click', resetGame);
    resetGame();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFlipGame);
  } else {
    initFlipGame();
  }
})();
