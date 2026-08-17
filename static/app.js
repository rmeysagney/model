const form = document.querySelector('#analysis-form');
const input = document.querySelector('#message');
const button = document.querySelector('#analyze-button');
const statusElement = document.querySelector('#status');
const analysisEmpty = document.querySelector('#analysis-empty');
const analysisResult = document.querySelector('#analysis-result');
const categoryName = document.querySelector('#category-name');
const confidenceText = document.querySelector('#confidence-text');
const confidenceBar = document.querySelector('#confidence-bar');
const explanation = document.querySelector('#analysis-explanation');
const suggestionsElement = document.querySelector('#suggestions');
const suggestionCount = document.querySelector('#suggestion-count');
const draft = document.querySelector('#reply-draft');
const copyButton = document.querySelector('#copy-button');
const draftStatus = document.querySelector('#draft-status');
const clearButton = document.querySelector('#clear-button');
const messageCount = document.querySelector('#message-count');
const decisionSource = document.querySelector('#decision-source');
const languageSignal = document.querySelector('#language-signal');

function setDraft(text, selectedButton) {
  draft.value = text;
  draftStatus.textContent = 'Taslak hazır';
  draftStatus.classList.add('filled');
  copyButton.disabled = false;
  document.querySelectorAll('.use-reply').forEach((button) => button.classList.remove('selected'));
  if (selectedButton) selectedButton.classList.add('selected');
  draft.focus();
}

function updateMessageCount() {
  messageCount.textContent = `${input.value.length.toLocaleString('tr-TR')} / 5.000`;
}

function renderLoadingSuggestions() {
  suggestionsElement.innerHTML = '<div class="loading-card"><i></i><i></i><i></i><span>Benzer kayıtlar aranıyor…</span></div>';
  suggestionCount.textContent = 'aranıyor';
}

function renderSuggestions(suggestions) {
  suggestionsElement.innerHTML = '';
  suggestions.forEach((suggestion, index) => {
    const card = document.createElement('article');
    card.className = 'suggestion-card';
    const meta = document.createElement('div');
    meta.className = 'suggestion-meta';
    const label = document.createElement('span');
    label.textContent = `ÖNERİ ${String(index + 1).padStart(2, '0')}`;
    const score = document.createElement('span');
    score.className = 'match-score';
    score.textContent = suggestion.match_type === 'rule'
      ? 'Kural ile tanındı'
      : suggestion.match_type === 'none'
        ? 'Eşleşme yok'
      : `%${(suggestion.similarity * 100).toFixed(0)} eşleşme`;
    meta.append(label, score);
    const answer = document.createElement('p');
    answer.className = 'suggestion-answer';
    answer.textContent = suggestion.answer;
    answer.title = suggestion.answer;
    const bottom = document.createElement('div');
    bottom.className = 'suggestion-bottom';
    const source = document.createElement('span');
    source.className = 'source-message';
    source.textContent = `Kaynak: ${suggestion.matched_question}`;
    source.title = source.textContent;
    const useButton = document.createElement('button');
    useButton.className = 'use-reply';
    useButton.type = 'button';
    useButton.textContent = 'Taslağa ekle →';
    useButton.addEventListener('click', () => setDraft(suggestion.answer, useButton));
    bottom.append(source, useButton);
    card.append(meta, answer, bottom);
    suggestionsElement.append(card);
  });
}

function renderAnalysis(data) {
  analysisEmpty.classList.add('hidden');
  analysisResult.classList.remove('hidden');
  categoryName.textContent = data.category;
  const score = Math.round(data.similarity * 100);
  languageSignal.textContent = `Girdi dili: ${(data.detected_language || 'bilinmiyor').toUpperCase()} · Yanıt: EN`;
  if (data.match_type === 'rule') {
    confidenceText.textContent = 'Kural ile tanındı';
    confidenceBar.style.width = '100%';
    decisionSource.textContent = 'Doğrudan niyet kuralı';
    explanation.textContent = 'Kısa sosyal mesaj, rastgele bir destek kategorisine düşmemesi için doğrudan tanındı.';
  } else if (data.match_type === 'none') {
    confidenceText.textContent = 'Yeterli eşleşme yok';
    confidenceBar.style.width = `${Math.max(4, score)}%`;
    decisionSource.textContent = 'Belirsizlik koruması';
    explanation.textContent = 'Mesaj, eğitim kayıtlarından yeterince benzer bir konuyla eşleşmedi. Daha fazla ayrıntı isteniyor.';
  } else {
    confidenceText.textContent = `En iyi benzerlik %${score}`;
    confidenceBar.style.width = `${Math.max(4, score)}%`;
    const hasIntentRule = data.category_source === 'keyword_rule';
    decisionSource.textContent = hasIntentRule ? 'Açık konu kuralı + kayıt eşleştirme' : 'Yerel ML + kayıt eşleştirme';
    explanation.textContent = hasIntentRule
      ? 'Açıkça belirtilen konu güvenli kuralla seçildi; öneri aynı konuya ait geçmiş kayıttan geliyor. Yüzde, yanıt doğruluğu değil metinsel benzerliktir.'
      : 'Etiket yerel modelden, öneriler aynı kategori içindeki geçmiş kayıtlardan gelir. Yüzde, doğruluk değil metinsel benzerliktir.';
  }
  suggestionCount.textContent = `${data.suggestions.length} öneri`;
  renderSuggestions(data.suggestions);
}

async function checkStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    if (!data.ready) throw new Error(data.error);
    statusElement.className = 'status ready';
    statusElement.innerHTML = `<i></i>${data.metadata.training_examples.toLocaleString('tr-TR')} kayıt hazır`;
  } catch (error) {
    statusElement.className = 'status error';
    statusElement.innerHTML = '<i></i>Model kullanılamıyor';
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || button.disabled) return;
  button.disabled = true;
  button.innerHTML = 'Analiz ediliyor…';
  renderLoadingSuggestions();
  let timeoutId;
  try {
    const controller = new AbortController();
    timeoutId = setTimeout(() => controller.abort(), 30000);
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message}), signal: controller.signal,
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); }
    catch { throw new Error('Sunucudan geçerli bir yanıt alınamadı.'); }
    if (!response.ok) throw new Error(data.error || 'Analiz gerçekleştirilemedi.');
    renderAnalysis(data);
  } catch (error) {
    analysisEmpty.classList.remove('hidden');
    const message = error.name === 'AbortError'
      ? 'Hata: Analiz 30 saniye içinde tamamlanamadı. Lütfen tekrar deneyin.'
      : `Hata: ${error.message}`;
    analysisEmpty.querySelector('p').textContent = message;
  } finally {
    clearTimeout(timeoutId);
    button.disabled = false;
    button.innerHTML = 'Analiz et <span>→</span>';
  }
});

input.addEventListener('input', updateMessageCount);
input.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') form.requestSubmit();
});
clearButton.addEventListener('click', () => {
  input.value = '';
  updateMessageCount();
  input.focus();
});

draft.addEventListener('input', () => {
  const filled = Boolean(draft.value.trim());
  draftStatus.textContent = filled ? 'Taslak düzenlendi' : 'Taslak boş';
  draftStatus.classList.toggle('filled', filled);
  copyButton.disabled = !filled;
});
copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(draft.value); copyButton.textContent = 'Kopyalandı ✓'; setTimeout(() => { copyButton.textContent = 'Kopyala'; }, 1600); }
  catch { copyButton.textContent = 'Kopyalanamadı'; }
});
checkStatus();
updateMessageCount();
