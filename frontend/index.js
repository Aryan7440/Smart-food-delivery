/**
 * Smart Food Ordering — Frontend JavaScript
 *
 * This file connects the frontend forms to the backend APIs using fetch().
 * Each function: reads form → sends POST request → shows result.
 */

const API_BASE = 'http://localhost:8000';

// ============================================================
//  UTILITY HELPERS
// ============================================================

/** Show loading spinner on a button */
function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (loading) {
    btn.classList.add('loading');
    btn.disabled = true;
  } else {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

/** Show a result box with main text and tag-style details */
function showResult(boxId, mainHtml, tags) {
  const box = document.getElementById(boxId);
  const main = document.getElementById(boxId + '-main');
  const details = document.getElementById(boxId + '-details');

  box.classList.remove('error');
  box.classList.add('visible');
  main.innerHTML = mainHtml;
  details.innerHTML = tags
    .map(t => `<span class="result-box__tag">${t}</span>`)
    .join('');

  // Re-trigger slide-up animation
  box.style.animation = 'none';
  box.offsetHeight; // force reflow
  box.style.animation = '';
}

/** Show an error in a result box */
function showError(boxId, message) {
  const box = document.getElementById(boxId);
  const main = document.getElementById(boxId + '-main');
  const details = document.getElementById(boxId + '-details');

  box.classList.add('visible', 'error');
  main.innerHTML = `<span class="result-box__error-msg">❌ ${message}</span>`;
  details.innerHTML = '';
}

/**
 * Call a backend API endpoint.
 * This is the KEY function — it's how frontend talks to backend!
 */
async function callAPI(endpoint, data) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => null);
    const msg = err?.detail?.[0]?.msg || err?.detail || `Server error (${response.status})`;
    throw new Error(msg);
  }

  return response.json();
}

// ============================================================
//  1. DELIVERY TIME PREDICTION
// ============================================================

document.getElementById('delivery-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  setLoading('delivery-btn', true);

  try {
    const data = {
      distance_km: parseFloat(document.getElementById('delivery-distance').value),
      weather: document.getElementById('delivery-weather').value,
      traffic_level: document.getElementById('delivery-traffic').value,
      time_of_day: document.getElementById('delivery-time-of-day').value,
      vehicle_type: document.getElementById('delivery-vehicle').value,
      preparation_time_min: parseFloat(document.getElementById('delivery-prep-time').value),
      courier_experience_yrs: parseFloat(document.getElementById('delivery-experience').value),
    };

    const result = await callAPI('/order/delivery-time', data);

    showResult('delivery-result',
      `${result.predicted_delivery_time_minutes.toFixed(1)} min`,
      [
        `Model: ${result.model_used}`,
        `Confidence: ${result.confidence || 'N/A'}`,
      ]
    );
  } catch (err) {
    showError('delivery-result', err.message);
  } finally {
    setLoading('delivery-btn', false);
  }
});

// ============================================================
//  2. MENU RECOMMENDATION
// ============================================================

// Tag input management for past orders
const recommendTags = [];

function renderRecommendTags() {
  const container = document.getElementById('recommend-tags');
  const input = document.getElementById('recommend-input');

  // Remove existing tags (keep input)
  container.querySelectorAll('.tag').forEach(t => t.remove());

  recommendTags.forEach((tag, i) => {
    const el = document.createElement('span');
    el.className = 'tag';
    el.innerHTML = `${tag}<button type="button" data-index="${i}">&times;</button>`;
    container.insertBefore(el, input);
  });
}

document.getElementById('recommend-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const val = e.target.value.trim().toLowerCase();
    if (val && !recommendTags.includes(val)) {
      recommendTags.push(val);
      renderRecommendTags();
    }
    e.target.value = '';
  }
});

document.getElementById('recommend-tags').addEventListener('click', (e) => {
  if (e.target.tagName === 'BUTTON') {
    recommendTags.splice(parseInt(e.target.dataset.index), 1);
    renderRecommendTags();
  }
});

// Add some default tags
['pizza', 'burger', 'pizza', 'biryani', 'burger'].forEach(t => recommendTags.push(t));
renderRecommendTags();

document.getElementById('recommend-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (recommendTags.length === 0) {
    showError('recommend-result', 'Please add at least one past order');
    return;
  }
  setLoading('recommend-btn', true);

  try {
    const result = await callAPI('/menu/recommend', { past_orders: [...recommendTags] });

    showResult('recommend-result',
      `🍴 ${result.recommended_item.charAt(0).toUpperCase() + result.recommended_item.slice(1)}`,
      [
        `Model: ${result.model_used}`,
        `Confidence: ${(result.confidence_score * 100).toFixed(0)}%`,
        result.reasoning ? `Reason: ${result.reasoning}` : '',
      ].filter(Boolean)
    );
  } catch (err) {
    showError('recommend-result', err.message);
  } finally {
    setLoading('recommend-btn', false);
  }
});

// ============================================================
//  3. REVIEW CLASSIFICATION
// ============================================================

document.getElementById('review-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  setLoading('review-btn', true);

  try {
    const rating = parseFloat(document.getElementById('review-rating').value);
    const text = document.getElementById('review-text').value.trim();
    const result = await callAPI('/review/fake-or-real', { rating: rating, review_text: text });

    const verdict = result.is_genuine ? '✅ Genuine Review' : '🚩 Likely Fake';

    showResult('review-result',
      verdict,
      [
        `Model: ${result.model_used}`,
        `Confidence: ${(result.confidence_score * 100).toFixed(0)}%`,
        result.reason ? `Reason: ${result.reason}` : '',
      ].filter(Boolean)
    );
  } catch (err) {
    showError('review-result', err.message);
  } finally {
    setLoading('review-btn', false);
  }
});

// ============================================================
//  4. CUISINE CLASSIFICATION
// ============================================================

// Tag input management for menu items
const cuisineTags = [];

function renderCuisineTags() {
  const container = document.getElementById('cuisine-tags');
  const input = document.getElementById('cuisine-input');

  container.querySelectorAll('.tag').forEach(t => t.remove());

  cuisineTags.forEach((tag, i) => {
    const el = document.createElement('span');
    el.className = 'tag';
    el.innerHTML = `${tag}<button type="button" data-index="${i}">&times;</button>`;
    container.insertBefore(el, input);
  });
}

document.getElementById('cuisine-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const val = e.target.value.trim().toLowerCase();
    if (val && !cuisineTags.includes(val)) {
      cuisineTags.push(val);
      renderCuisineTags();
    }
    e.target.value = '';
  }
});

document.getElementById('cuisine-tags').addEventListener('click', (e) => {
  if (e.target.tagName === 'BUTTON') {
    cuisineTags.splice(parseInt(e.target.dataset.index), 1);
    renderCuisineTags();
  }
});

// Add default cuisine tags
['paneer tikka', 'naan', 'biryani', 'dal makhani'].forEach(t => cuisineTags.push(t));
renderCuisineTags();

document.getElementById('cuisine-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (cuisineTags.length === 0) {
    showError('cuisine-result', 'Please add at least one menu item');
    return;
  }
  setLoading('cuisine-btn', true);

  try {
    const result = await callAPI('/restaurant/cuisine-classifier', { menu_items: [...cuisineTags] });

    const flagEmoji = {
      'Indian': '🇮🇳', 'Chinese': '🇨🇳', 'Italian': '🇮🇹', 'Mexican': '🇲🇽'
    };

    showResult('cuisine-result',
      `${flagEmoji[result.cuisine_type] || '🍴'} ${result.cuisine_type}`,
      [
        `Model: ${result.model_used}`,
        `Confidence: ${(result.confidence_score * 100).toFixed(0)}%`,
        ...(result.matched_keywords || []).map(k => `🏷️ ${k}`),
      ]
    );
  } catch (err) {
    showError('cuisine-result', err.message);
  } finally {
    setLoading('cuisine-btn', false);
  }
});

// ============================================================
//  5. FOOD IMAGE ANALYSIS
// ============================================================

// File upload: click-to-browse + drag-and-drop
const foodimgDropzone = document.getElementById('foodimg-dropzone');
const foodimgFileInput = document.getElementById('foodimg-file');
const foodimgPreview = document.getElementById('foodimg-preview');
const foodimgPlaceholder = document.getElementById('foodimg-placeholder');

// Click dropzone to open file picker
foodimgDropzone.addEventListener('click', () => foodimgFileInput.click());

// Drag events
foodimgDropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  foodimgDropzone.classList.add('dragover');
});
foodimgDropzone.addEventListener('dragleave', () => {
  foodimgDropzone.classList.remove('dragover');
});
foodimgDropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  foodimgDropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    foodimgFileInput.files = e.dataTransfer.files;
    showImagePreview(file);
  }
});

// File input change
foodimgFileInput.addEventListener('change', () => {
  if (foodimgFileInput.files[0]) {
    showImagePreview(foodimgFileInput.files[0]);
  }
});

function showImagePreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    foodimgPreview.src = e.target.result;
    foodimgPreview.classList.add('visible');
    foodimgPlaceholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

// Submit: upload image as multipart/form-data
document.getElementById('foodimg-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = foodimgFileInput.files[0];
  if (!file) {
    showError('foodimg-result', 'Please select a food image');
    return;
  }
  setLoading('foodimg-btn', true);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/food-image/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => null);
      const msg = err?.detail || `Server error (${response.status})`;
      throw new Error(msg);
    }

    const result = await response.json();

    // Build the main display: dish name
    const dishName = result.dish_name
      .split(' ')
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');

    // Build description HTML
    const descHtml = `<div class="result-box__description">"${result.description}"</div>`;

    showResult('foodimg-result',
      `🍽️ ${dishName}`,
      [
        `Model: ${result.model_used}`,
        `Confidence: ${(result.confidence * 100).toFixed(0)}%`,
        ...result.top_5.slice(1).map(t =>
          `${t.name} (${(t.score * 100).toFixed(0)}%)`
        ),
      ]
    );

    // Inject description below the main title
    const mainEl = document.getElementById('foodimg-result-main');
    mainEl.insertAdjacentHTML('afterend', descHtml);

  } catch (err) {
    showError('foodimg-result', err.message);
  } finally {
    setLoading('foodimg-btn', false);
  }
});

// ============================================================
//  SERVER HEALTH CHECK ON LOAD
// ============================================================

(async () => {
  const badge = document.querySelector('.hero__badge');
  try {
    const resp = await fetch(`${API_BASE}/`);
    const data = await resp.json();
    badge.textContent = `✅ ${data.app_name} v${data.version} — ${data.ai_models_loaded ? 'AI Models Active' : 'Basic Mode'}`;
  } catch {
    badge.textContent = '❌ Cannot reach server — is it running on port 8000?';
    badge.style.background = 'rgba(239, 68, 68, 0.12)';
    badge.style.borderColor = 'rgba(239, 68, 68, 0.25)';
    badge.style.color = '#ef4444';
  }
})();
