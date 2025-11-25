// Slideshows
(function() {
  const mechanicsImages = [
    'images/ctin488/Mechanics/6ed63bc3591bd6b68bd693c572c2b87f.JPG',
    'images/ctin488/Mechanics/8a1b4cd3c6cbdb0d772054192b8e798b.JPG',
    'images/ctin488/Mechanics/IMG_1213.jpg',
    'images/ctin488/Mechanics/IMG_1220.jpg'
  ];

  const upriverImages = [
    'images/ctin488/UpTheRiver/image0.jpeg',
    'images/ctin488/UpTheRiver/image1.jpeg',
    'images/ctin488/UpTheRiver/image2.jpeg',
    'images/ctin488/UpTheRiver/image3.jpeg'
  ];

  let slideIndex = { mechanics: 1, upriver: 1 };

  function createSlideshow(containerId, dotsId, images, key) {
    const container = document.getElementById(containerId);
    const dotsContainer = document.getElementById(dotsId);
    if (!container || !dotsContainer) return;

    // Create slides
    images.forEach((src, index) => {
      const slide = document.createElement('div');
      slide.className = 'mySlides fade';
      const numberText = document.createElement('div');
      numberText.className = 'numbertext';
      numberText.textContent = `${index + 1} / ${images.length}`;
      const img = document.createElement('img');
      img.src = src;
      img.alt = '';
      img.loading = 'lazy';
      slide.appendChild(numberText);
      slide.appendChild(img);
      container.appendChild(slide);

      // Create dot
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.setAttribute('onclick', `currentSlide(${index + 1}, '${key}')`);
      dotsContainer.appendChild(dot);
    });

    // Add prev/next buttons
    const prev = document.createElement('a');
    prev.className = 'prev';
    prev.setAttribute('onclick', `plusSlides(-1, '${key}')`);
    prev.innerHTML = '&#10094;';
    container.appendChild(prev);

    const next = document.createElement('a');
    next.className = 'next';
    next.setAttribute('onclick', `plusSlides(1, '${key}')`);
    next.innerHTML = '&#10095;';
    container.appendChild(next);
  }

  function plusSlides(n, key) {
    showSlides(slideIndex[key] += n, key);
  }

  function currentSlide(n, key) {
    showSlides(slideIndex[key] = n, key);
  }

  function showSlides(n, key) {
    const containerId = key === 'mechanics' ? 'slideshow-mechanics' : 'slideshow-upriver';
    const dotsId = key === 'mechanics' ? 'dots-mechanics' : 'dots-upriver';
    const container = document.getElementById(containerId);
    const dotsContainer = document.getElementById(dotsId);
    
    if (!container || !dotsContainer) return;

    const slides = container.getElementsByClassName('mySlides');
    const dots = dotsContainer.getElementsByClassName('dot');
    
    if (n > slides.length) { slideIndex[key] = 1; }
    if (n < 1) { slideIndex[key] = slides.length; }
    
    for (let i = 0; i < slides.length; i++) {
      slides[i].style.display = 'none';
    }
    
    for (let i = 0; i < dots.length; i++) {
      dots[i].className = dots[i].className.replace(' active', '');
    }
    
    slides[slideIndex[key] - 1].style.display = 'block';
    dots[slideIndex[key] - 1].className += ' active';
  }

  // Make functions global for onclick handlers
  window.plusSlides = plusSlides;
  window.currentSlide = currentSlide;

  // Initialize slideshows
  createSlideshow('slideshow-mechanics', 'dots-mechanics', mechanicsImages, 'mechanics');
  createSlideshow('slideshow-upriver', 'dots-upriver', upriverImages, 'upriver');

  // Show first slide for each
  showSlides(1, 'mechanics');
  showSlides(1, 'upriver');
})();

