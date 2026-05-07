/* ═══════════════════════════════════════════════════════════
   Sourashtra Translator — script.js
   English / Tamil → Sourashtra Script
   ═══════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────
let currentLang = 'english';
let dictPage = 1;
let dictCategory = '';
let dictQuery = '';
let dictDebounce = null;

// ── Elements ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const sourceInput = $('#sourceInput');
const translateBtn = $('#translateBtn');
const clearBtn = $('#clearBtn');
const resultsArea = $('#resultsArea');
const emptyState = $('#emptyState');
const primaryResult = $('#primaryResult');
const moreMatches = $('#moreMatches');
const matchesGrid = $('#matchesGrid');

// ══════════════════════════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════════════════════════

$$('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        $$('.nav-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.tab;
        $$('.tab-content').forEach(s => s.style.display = 'none');
        $(`#tab-${target}`).style.display = 'block';

        // Load data on first tab visit
        if (target === 'dictionary' && !dictLoaded) loadDictionary();
        if (target === 'about' && !aboutLoaded) loadAbout();
    });
});

// ══════════════════════════════════════════════════════════
// LANGUAGE TOGGLE
// ══════════════════════════════════════════════════════════

$$('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.lang-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentLang = btn.dataset.lang;

        if (currentLang === 'tamil') {
            sourceInput.placeholder = 'தமிழ் சொல்லை உள்ளிடவும் (e.g. பால், தண்ணீர், அம்மா)...';
            updateExamples('tamil');
        } else if (currentLang === 'sourashtra') {
            sourceInput.placeholder = 'Type a Sourashtra word in Roman script (e.g. paal, tanni, amme)...';
            updateExamples('sourashtra');
        } else {
            sourceInput.placeholder = 'Type an English word (e.g. milk, water, mother)...';
            updateExamples('english');
        }
        sourceInput.focus();
    });
});

function updateExamples(lang) {
    const container = $('#quickExamples');
    let chips;
    if (lang === 'tamil') {
        chips = [
            { word: 'பால்', label: 'பால்' },
            { word: 'தண்ணீர்', label: 'தண்ணீர்' },
            { word: 'அம்மா', label: 'அம்மா' },
            { word: 'வீடு', label: 'வீடு' },
            { word: 'சாப்பாடு', label: 'சாப்பாடு' },
            { word: 'சூரியன்', label: 'சூரியன்' },
            { word: 'மீன்', label: 'மீன்' },
            { word: 'சிவப்பு', label: 'சிவப்பு' },
        ];
    } else if (lang === 'sourashtra') {
        chips = [
            { word: 'paal', label: 'paal (milk)' },
            { word: 'tanni', label: 'tanni (water)' },
            { word: 'amme', label: 'amme (mother)' },
            { word: 'ghor', label: 'ghor (house)' },
            { word: 'soori', label: 'soori (sun)' },
            { word: 'maas', label: 'maas (fish)' },
            { word: 'kempu', label: 'kempu (red)' },
            { word: 'anna', label: 'anna (food)' },
        ];
    } else {
        chips = [
            { word: 'milk', label: 'milk' },
            { word: 'water', label: 'water' },
            { word: 'mother', label: 'mother' },
            { word: 'house', label: 'house' },
            { word: 'food', label: 'food' },
            { word: 'sun', label: 'sun' },
            { word: 'fish', label: 'fish' },
            { word: 'red', label: 'red' },
        ];
    }

    container.innerHTML = `<span class="examples-label">Try:</span>` +
        chips.map(c =>
            `<button class="example-chip" data-word="${c.word}">${c.label}</button>`
        ).join('');

    // Re-bind chip click handlers
    container.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            sourceInput.value = chip.dataset.word;
            doTranslate();
        });
    });
}

// ══════════════════════════════════════════════════════════
// TRANSLATE
// ══════════════════════════════════════════════════════════

translateBtn.addEventListener('click', doTranslate);
sourceInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') doTranslate();
});
clearBtn.addEventListener('click', () => {
    sourceInput.value = '';
    resultsArea.style.display = 'none';
    $('#neuralResultArea').style.display = 'none';
    emptyState.style.display = 'block';
    sourceInput.focus();
});

async function doTranslate() {
    const text = sourceInput.value.trim();
    if (!text) { toast('Please enter a word'); return; }

    // Show loader
    translateBtn.disabled = true;
    translateBtn.querySelector('.btn-text').style.display = 'none';
    translateBtn.querySelector('.btn-loader').style.display = 'block';

    try {
        if (currentLang === 'sourashtra') {
            // Neural AI translation: Sourashtra -> English
            $('#neuralResultArea').style.display = 'none';
            resultsArea.style.display = 'none';

            const resp = await fetch('/api/neural-translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            const data = await resp.json();

            if (data.error) {
                toast(data.error);
                return;
            }

            renderNeuralResults(data);
        } else {
            // English/Tamil input: try neural EN/TA → Sourashtra first, then fall back to dictionary.
            $('#neuralResultArea').style.display = 'none';

            let data;
            let usedFallbackDictOnly = false;

            try {
                const resp = await fetch('/api/reverse-neural-translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, lang: currentLang }),
                });
                data = await resp.json();

                if (!resp.ok && !data.mode) {
                    // If server returned an error without mode hint, treat as failure and fall back.
                    throw new Error(data.error || 'Reverse neural API error');
                }
            } catch (errInner) {
                console.error('Reverse neural translate failed, falling back to dictionary:', errInner);
                usedFallbackDictOnly = true;
            }

            if (!usedFallbackDictOnly && data) {
                if (data.mode === 'neural' && data.neural_available) {
                    renderNeuralResults(data);
                    return;
                }

                // If backend already fell back to dictionary, reuse its response.
                if (data.mode === 'dictionary' || Array.isArray(data.results)) {
                    renderResults(data);
                    return;
                }

                if (data.error) {
                    toast(data.error);
                    return;
                }
            }

            // Final fallback: call pure dictionary endpoint.
            const resp = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, lang: currentLang }),
            });
            const dictData = await resp.json();

            if (dictData.error) {
                toast(dictData.error);
                return;
            }

            renderResults(dictData);
        }
    } catch (err) {
        toast('Connection error \u2014 is the server running?');
        console.error(err);
    } finally {
        translateBtn.disabled = false;
        translateBtn.querySelector('.btn-text').style.display = '';
        translateBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function renderNeuralResults(data) {
    // Hide dictionary results, show neural results
    resultsArea.style.display = 'none';
    emptyState.style.display = 'none';
    const neuralArea = $('#neuralResultArea');
    neuralArea.style.display = 'block';

    // Update labels based on direction (Sourashtra→English vs English/Tamil→Sourashtra)
    const sourceLang = (data.source_lang || '').toLowerCase();
    const targetLang = (data.target_lang || '').toLowerCase();
    const inputLabelEl = document.querySelector('.neural-input-label');
    const outputLabelEl = document.querySelector('.neural-output-label');

    if (inputLabelEl && outputLabelEl) {
        if (sourceLang === 'english') {
            inputLabelEl.textContent = 'English Input:';
        } else if (sourceLang === 'tamil') {
            inputLabelEl.textContent = 'Tamil Input:';
        } else {
            inputLabelEl.textContent = 'Sourashtra Input:';
        }

        if (targetLang === 'sourashtra') {
            outputLabelEl.textContent = 'Sourashtra Translation:';
        } else if (targetLang === 'english') {
            outputLabelEl.textContent = 'English Translation:';
        } else {
            outputLabelEl.textContent = 'Translation:';
        }
    }

    $('#neuralInput').textContent = data.input;
    $('#neuralOutput').textContent = data.translation;
    $('#neuralModel').textContent = data.model || 'V5 ByT5-small';
    $('#neuralTime').textContent = `${data.time_ms}ms`;

    // Show dictionary cross-references if available
    const dictMatches = data.dict_matches || [];
    const neuralDictArea = $('#neuralDictMatches');
    const neuralGrid = $('#neuralMatchesGrid');

    if (dictMatches.length > 0) {
        neuralDictArea.style.display = 'block';
        neuralGrid.innerHTML = dictMatches.map(r => `
            <div class="match-card">
                <div class="match-script">${esc(r.sourashtra_script)}</div>
                <div class="match-roman">${esc(r.roman)}</div>
                <div class="match-meanings">
                    <span>EN: ${esc(r.english)}</span>
                    ${r.tamil ? `<span>TA: ${esc(r.tamil)}</span>` : ''}
                </div>
                <div class="match-meta">
                    <span class="match-cat">${esc(r.category)}</span>
                    <span class="match-conf">${(r.confidence * 100).toFixed(0)}%</span>
                </div>
            </div>
        `).join('');
    } else {
        neuralDictArea.style.display = 'none';
    }

    neuralArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderResults(data) {
    const results = data.results || [];

    if (results.length === 0) {
        resultsArea.style.display = 'none';
        emptyState.style.display = 'block';
        emptyState.querySelector('p').textContent =
            `No Sourashtra translation found for "${data.input}"`;
        return;
    }

    emptyState.style.display = 'none';
    resultsArea.style.display = 'block';

    // ── Primary result ──
    const top = results[0];
    $('#mainScript').textContent = top.sourashtra_script || '—';
    $('#mainRoman').textContent = top.roman || '';
    $('#mainEnglish').textContent = top.english || '';
    $('#mainTamil').textContent = top.tamil || '';
    $('#mainCategory').textContent = top.category || 'General';
    $('#mainConfidence').textContent = `Match: ${(top.confidence * 100).toFixed(0)}%`;
    $('#mainTime').textContent = `${data.time_ms}ms`;

    // ── More matches ──
    if (results.length > 1) {
        moreMatches.style.display = 'block';
        matchesGrid.innerHTML = results.slice(1).map(r => `
            <div class="match-card" onclick="selectMatch(this)" 
                 data-script="${esc(r.sourashtra_script)}"
                 data-roman="${esc(r.roman)}"
                 data-english="${esc(r.english)}"
                 data-tamil="${esc(r.tamil)}"
                 data-category="${esc(r.category)}"
                 data-confidence="${r.confidence}">
                <div class="match-script">${esc(r.sourashtra_script)}</div>
                <div class="match-roman">${esc(r.roman)}</div>
                <div class="match-meanings">
                    <span>EN: ${esc(r.english)}</span>
                    ${r.tamil ? `<span>TA: ${esc(r.tamil)}</span>` : ''}
                </div>
                <div class="match-meta">
                    <span class="match-cat">${esc(r.category)}</span>
                    <span class="match-conf">${(r.confidence * 100).toFixed(0)}%</span>
                </div>
            </div>
        `).join('');
    } else {
        moreMatches.style.display = 'none';
    }

    // Scroll into view
    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Click on a match card to promote it to primary
window.selectMatch = function (card) {
    $('#mainScript').textContent = card.dataset.script;
    $('#mainRoman').textContent = card.dataset.roman;
    $('#mainEnglish').textContent = card.dataset.english;
    $('#mainTamil').textContent = card.dataset.tamil;
    $('#mainCategory').textContent = card.dataset.category || 'General';
    $('#mainConfidence').textContent = `Match: ${(parseFloat(card.dataset.confidence) * 100).toFixed(0)}%`;
    primaryResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

// ══════════════════════════════════════════════════════════
// DICTIONARY
// ══════════════════════════════════════════════════════════

let dictLoaded = false;

$('#dictSearch').addEventListener('input', () => {
    clearTimeout(dictDebounce);
    dictDebounce = setTimeout(() => {
        dictQuery = $('#dictSearch').value;
        dictPage = 1;
        loadDictionary();
    }, 300);
});

$('#dictCategory').addEventListener('change', () => {
    dictCategory = $('#dictCategory').value;
    dictPage = 1;
    loadDictionary();
});

async function loadDictionary() {
    try {
        const params = new URLSearchParams({
            page: dictPage,
            per_page: 50,
        });
        if (dictQuery) params.set('q', dictQuery);
        if (dictCategory) params.set('category', dictCategory);

        const resp = await fetch(`/api/dictionary?${params}`);
        const data = await resp.json();
        dictLoaded = true;

        // Populate category dropdown (once)
        const catSelect = $('#dictCategory');
        if (catSelect.options.length <= 1 && data.categories) {
            data.categories.forEach(cat => {
                const opt = document.createElement('option');
                opt.value = cat;
                opt.textContent = cat;
                catSelect.appendChild(opt);
            });
        }

        // Count
        $('#dictCount').textContent = `${data.total.toLocaleString()} entries`;

        // Table body
        const tbody = $('#dictBody');
        tbody.innerHTML = data.items.map(item => `
            <tr>
                <td>${esc(item.sourashtra_script)}</td>
                <td>${esc(item.roman)}</td>
                <td>${esc(item.english)}</td>
                <td>${esc(item.tamil)}</td>
                <td>${esc(item.category)}</td>
            </tr>
        `).join('');

        // Pagination
        renderPagination(data.page, data.total_pages);

    } catch (err) {
        console.error('Dictionary load error:', err);
        toast('Failed to load dictionary');
    }
}

function renderPagination(current, total) {
    const container = $('#dictPagination');
    if (total <= 1) { container.innerHTML = ''; return; }

    let pages = [];
    const addPage = (n) => { if (!pages.includes(n) && n >= 1 && n <= total) pages.push(n); };

    addPage(1);
    for (let i = current - 2; i <= current + 2; i++) addPage(i);
    addPage(total);
    pages.sort((a, b) => a - b);

    let html = `<button class="page-btn" ${current === 1 ? 'disabled' : ''} onclick="goDictPage(${current - 1})">‹</button>`;
    let prev = 0;
    for (const p of pages) {
        if (p - prev > 1) html += `<span style="color:var(--text-muted);font-size:0.75rem;">…</span>`;
        html += `<button class="page-btn ${p === current ? 'active' : ''}" onclick="goDictPage(${p})">${p}</button>`;
        prev = p;
    }
    html += `<button class="page-btn" ${current === total ? 'disabled' : ''} onclick="goDictPage(${current + 1})">›</button>`;
    container.innerHTML = html;
}

window.goDictPage = function (p) {
    dictPage = p;
    loadDictionary();
    $('.dict-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
};

// ══════════════════════════════════════════════════════════
// ABOUT TAB
// ══════════════════════════════════════════════════════════

let aboutLoaded = false;

async function loadAbout() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        aboutLoaded = true;

        $('#dictSize').textContent = data.dictionary_size.toLocaleString();
        $('#catCount').textContent = data.category_count;

        // Version timeline
        const versions = data.versions || {};
        const timeline = $('#versionTimeline');
        timeline.innerHTML = Object.entries(versions).map(([key, v]) => `
            <div class="version-item">
                <span class="version-tag">${key}</span>
                <span class="version-name">${v.name} (${v.params})</span>
                <span class="version-em">${v.em}%</span>
            </div>
        `).join('');

    } catch (err) {
        console.error('Stats load error:', err);
    }
}

// ══════════════════════════════════════════════════════════
// INITIAL EXAMPLE CHIPS
// ══════════════════════════════════════════════════════════

$$('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        sourceInput.value = chip.dataset.word;
        doTranslate();
    });
});

// ══════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════

function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function toast(msg) {
    let el = document.querySelector('.toast');
    if (!el) {
        el = document.createElement('div');
        el.className = 'toast';
        document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 2500);
}

// ── Auto-focus input
sourceInput.focus();
