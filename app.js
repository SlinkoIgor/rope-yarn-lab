import {QwenModel, kvHeadForQuery} from './model.js';
import {TAU, clamp, dot, entropy, norm, pairContribution, rotate, scoreAt, softmax, yarnAmplitude, yarnFrequencies} from './math.js';

const $ = id => document.getElementById(id);
const fmt = (x, digits = 2) => Number(x).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
const integer = x => Math.round(x).toLocaleString('en-US');
const signed = (x, digits = 2) => `${x > 0 ? '+' : ''}${fmt(x, digits)}`;
const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const colors = {};
function syncPalette() {
  const style = getComputedStyle(document.documentElement);
  const value = name => style.getPropertyValue(`--${name}`).trim();
  Object.assign(colors, {
    query:value('query'), key:value('key'), rope:value('query'), yarn:value('key'),
    baseline:value('baseline'), grid:value('grid'), text:value('muted'),
    axis:value('axis'), selected:value('selected'),
  });
}
syncPalette();

const state = {
  q:null, k:null, modelQ:null, modelK:null, shared:false,
  queryIds:[], keyIds:[], queryIndex:0, keyIndex:0, words:['',''],
  head:0, activeHead:0,
  delta:10, offset:0, factor:4, yarnEnabled:true, temperature:true, originalLength:40960,
  betaFast:32, betaSlow:1, selected:0, display:'yarn', metric:'dot', chartRange:8192, showReference:true,
  ropeBase:1e6, temperatureMode:'auto', temperatureCoefficient:.1, logitMultiplier:yarnAmplitude(4)**2,
  attentionText:'The cat is soft and fluffy.', attentionLoadedText:'', attentionIds:[], attentionKeys:[],
  attentionHead:-1, attentionLoading:false, attentionError:'',
};
let initialization, requestVersion = 0, debounce, frame, chartCache = null, lastChartKey = '';
let attentionVersion = 0, attentionDebounce;
const model = new QwenModel(bytes => {
  if (!state.q) setStatus(`Loading checkpoint ranges… ${fmt(bytes / 1048576, 1)} MB`);
});

function setStatus(message, error = false, ready = false) {
  $('load-status').textContent = message;
  $('load-status').className = `load-status${error ? ' error' : ready ? ' ready' : ''}`;
  $('retry').classList.toggle('hidden', !error);
}

async function ensureModel() {
  if (!initialization) {
    initialization = model.initialize().catch(error => { initialization = null; throw error; });
  }
  return initialization;
}

function renderTokens(kind) {
  const ids = state[`${kind}Ids`];
  const selected = state[`${kind}Index`];
  $(`${kind}-tokens`).innerHTML = ids.map((id, i) => `<button type="button" class="token${i === selected ? ' active' : ''}" data-kind="${kind}" data-index="${i}" aria-pressed="${i === selected}" title="Token ID ${id} · ${escape(model.tokenizer.idToToken.get(id) ?? model.tokenizer.decode(id))}">${escape(model.tokenizer.label(id))}</button>`).join('');
}

async function loadWords(keepSelection = false) {
  clearTimeout(debounce);
  const version = ++requestVersion;
  const words = [$('query-word').value, $('key-word').value];
  const requestedHead = state.head;
  if (!words[0].length || (!state.shared && !words[1].length)) {
    setStatus(state.shared ? 'Enter a query token.' : 'Enter text in both input fields.', true);
    $('results').setAttribute('aria-busy', 'false');
    return;
  }
  $('results').setAttribute('aria-busy', 'true');
  $('load-words').disabled = true;
  setStatus(state.q ? `Loading Q head ${requestedHead} and computing vectors…` : 'Loading Qwen tokenizer and weights…');
  try {
    await ensureModel();
    if (version !== requestVersion) return;
    const queryIds = model.tokenizer.encode(words[0]);
    // Keep a real projected K available to restore when the experimental tie is removed.
    const keyIds = model.tokenizer.encode(words[1] || words[0]);
    if (!queryIds.length || !keyIds.length) throw new Error('Each input must contain at least one token.');
    const sameWords = words[0] === state.words[0] && words[1] === state.words[1];
    state.queryIndex = keepSelection && sameWords ? clamp(state.queryIndex, 0, queryIds.length - 1) : 0;
    state.keyIndex = keepSelection && sameWords ? clamp(state.keyIndex, 0, keyIds.length - 1) : 0;
    state.queryIds = queryIds; state.keyIds = keyIds; state.words = words;
    renderTokens('query'); renderTokens('key');
    const {q, k, kvHead} = await model.vectors(queryIds[state.queryIndex], keyIds[state.keyIndex], requestedHead);
    if (version !== requestVersion) return;
    state.modelQ = q; state.modelK = k; state.activeHead = requestedHead;
    applyVectorMode();
    $('results').classList.remove('stale');
    setStatus(state.shared
      ? `Ready · Q head ${requestedHead} · Q = K · token ${queryIds[state.queryIndex]}`
      : `Ready · head ${requestedHead} / KV ${kvHead} · tokens ${queryIds[state.queryIndex]} / ${keyIds[state.keyIndex]}`, false, true);
    lastChartKey = '';
    loadAttentionKeys();
    recalculate();
  } catch (error) {
    if (version === requestVersion) setStatus(error.message || 'Unable to load weights. Please retry.', true);
  } finally {
    if (version === requestVersion) {
      $('results').setAttribute('aria-busy', 'false');
      $('load-words').disabled = false;
    }
  }
}

function modelFrequencies() {
  return yarnFrequencies(128, state.ropeBase, state.factor, state.originalLength, state.betaFast, state.betaSlow);
}

function applyVectorMode() {
  state.q = state.modelQ;
  state.k = state.shared ? state.modelQ : state.modelK;
}
function configuredMultiplier() {
  return state.temperatureMode === 'manual' ? state.logitMultiplier : yarnAmplitude(state.factor, state.temperatureCoefficient) ** 2;
}
function effectiveMultiplier() { return state.yarnEnabled && state.temperature ? configuredMultiplier() : 1; }
function amplitude() { return Math.sqrt(effectiveMultiplier()); }
function metricValue(value) {
  if (state.metric === 'cosine') return value / (norm(state.q) * norm(state.k));
  return state.metric === 'logit' ? value / Math.sqrt(state.q.length) : value;
}

function updateControls() {
  const config = model.config ?? {num_attention_heads:16, num_key_value_heads:8};
  if ($('head-select').options.length !== config.num_attention_heads) {
    $('head-select').innerHTML = Array.from({length:config.num_attention_heads}, (_,head) => `<option value="${head}">Q head ${String(head).padStart(2,'0')} → KV ${String(kvHeadForQuery(head,config)).padStart(2,'0')}</option>`).join('');
  }
  Array.from($('head-select').options).forEach((option, head) => {
    option.textContent = state.shared ? `Q head ${String(head).padStart(2,'0')} · K = Q` : `Q head ${String(head).padStart(2,'0')} → KV ${String(kvHeadForQuery(head,config)).padStart(2,'0')}`;
  });
  $('head-select').value = state.head;
  $('previous-head').disabled = state.head === 0;
  $('next-head').disabled = state.head === config.num_attention_heads - 1;
  $('active-head').textContent = String(state.activeHead).padStart(2,'0');
  $('active-key-label').textContent = state.shared ? 'K = Q' : 'KV';
  $('active-kv-head').textContent = state.shared ? '' : String(kvHeadForQuery(state.activeHead,config)).padStart(2,'0');
  $('head-mapping').textContent = state.shared ? `Q head ${state.head} supplies both vectors. Learned K is bypassed.` : `Q head ${state.head} → KV head ${kvHeadForQuery(state.head,config)} · ${config.num_attention_heads} Q / ${config.num_key_value_heads} KV`;
  $('model-footer').textContent = `Qwen3-0.6B · layer 0 · Q head ${state.activeHead} · ${state.shared ? 'K = Q (experimental)' : `KV ${kvHeadForQuery(state.activeHead,config)}`} · d = 128`;
  $('shared-vectors').checked = state.shared;
  $('key-inputs').classList.toggle('hidden', state.shared);
  $('key-inputs').inert = state.shared;
  $('shared-note').classList.toggle('hidden', !state.shared);
  $('vector-mode-status').classList.toggle('hidden', !state.shared);
  const queryId = state.queryIds[state.queryIndex];
  $('shared-note').textContent = `K is a copy of ${queryId === undefined ? 'the selected Q vector' : `Q for “${model.tokenizer.label(queryId)}” (token ${queryId})`} before RoPE. This experimental tie replaces the model's learned K. Different positions still rotate Q and K differently.`;
  $('tokens-note').textContent = state.shared ? 'Select one Q token. Q and K share its projected vector before rotation.' : 'Select one token per input. A word may span multiple tokens. Input updates automatically.';
  $('base-readout').textContent = integer(state.ropeBase);
  syncInput('rope-base', state.ropeBase);
  $('base-slider').value = Math.log10(state.ropeBase);
  document.querySelectorAll('[data-base]').forEach(button => button.classList.toggle('active', Number(button.dataset.base) === state.ropeBase));
  $('distance').value = state.delta;
  if (state.delta > Number($('distance-slider').max)) {
    const ranges = [4096,32768,163840,1310720];
    const max = ranges.find(value => value >= state.delta) ?? ranges.at(-1);
    $('distance-slider').max = max; $('distance-range').value = max;
  }
  $('distance-slider').value = state.delta;
  $('query-position').textContent = integer(state.delta + state.offset);
  $('key-position').textContent = integer(state.offset);
  $('delta-tag').textContent = integer(state.delta);
  $('factor-tag').textContent = `${fmt(state.factor, state.factor % 1 ? 1 : 0)}×`;
  $('context-from').textContent = integer(state.originalLength);
  $('context-to').textContent = integer(state.originalLength * state.factor);
  $('yarn-controls').classList.toggle('disabled', !state.yarnEnabled);
  $('yarn-controls').inert = !state.yarnEnabled;
  const scalingActive = state.yarnEnabled && state.temperature;
  $('temperature').checked = state.temperature;
  $('softmax-apply').checked = state.temperature;
  $('temperature').disabled = !state.yarnEnabled;
  $('softmax-apply').disabled = !state.yarnEnabled;
  $('temperature-controls').classList.toggle('disabled', !scalingActive);
  $('temperature-controls').inert = !scalingActive;
  $('temperature-mode').value = state.temperatureMode;
  $('coefficient-controls').classList.toggle('hidden', state.temperatureMode !== 'auto');
  $('manual-temperature-controls').classList.toggle('hidden', state.temperatureMode !== 'manual');
  syncInput('temperature-coefficient', state.temperatureCoefficient);
  $('coefficient-slider').value = state.temperatureCoefficient;
  syncInput('logit-scale', configuredMultiplier());
  $('logit-scale-slider').value = configuredMultiplier();
  $('softmax-scale').max = Math.max(8, configuredMultiplier());
  $('softmax-scale').value = configuredMultiplier();
  $('softmax-scale').disabled = !scalingActive;
  $('softmax-scale-value').textContent = fmt(effectiveMultiplier(), 4);
  syncInput('attention-gap', state.delta);
  $('attention-query').textContent = queryId === undefined ? '—' : model.tokenizer.label(queryId);
  $('attention-description').textContent = state.shared
    ? 'Q = K is active: each key is replaced by your selected Q vector before rotation. Compare the same probabilities with and without correction; different token positions still produce different rotations.'
    : 'Compare attention from your selected Q to the tokens below. Each token has its own learned K. Temperature changes how strongly higher scores are favored, redistributing the same 100% of attention.';
  const rule = state.temperatureMode === 'auto' ? `λ = (1 + ${fmt(state.temperatureCoefficient,2)} ln(${fmt(state.factor,1)}))²` : 'Manual logit multiplier';
  $('temperature-formula').textContent = state.temperatureMode === 'auto'
    ? `a = 1 + c ln(s) = ${fmt(Math.sqrt(configuredMultiplier()),4)}\nλ = a² = ${fmt(configuredMultiplier(),4)}`
    : `λ = ${fmt(configuredMultiplier(),4)}\na = √λ = ${fmt(Math.sqrt(configuredMultiplier()),4)}`;
  $('temperature-effect').textContent = `Q and K lengths × a; logits × λ; softmax temperature T = 1/λ = ${fmt(1/configuredMultiplier(),3)}. Angles and cosine stay fixed.`;
  $('softmax-rule').textContent = !state.yarnEnabled ? 'YaRN is off · λ = 1. Enable YaRN in Parameters to experiment.' : !state.temperature ? 'Scaling is off · λ = 1. Check Apply scaling to experiment.' : `${rule} · a = √λ = ${fmt(amplitude(),4)} · T = 1/λ = ${fmt(1/effectiveMultiplier(),3)}`;
  $('restore-yarn-temperature').disabled = !scalingActive;
  document.body.classList.toggle('yarn-off', !state.yarnEnabled);
  document.querySelectorAll('[data-distance]').forEach(button => button.classList.toggle('active', Number(button.dataset.distance) === state.delta));
  document.querySelectorAll('[data-display]').forEach(button => {
    button.disabled = button.dataset.display === 'yarn' && !state.yarnEnabled;
    button.classList.toggle('active', button.dataset.display === state.display);
    button.setAttribute('aria-pressed', String(button.dataset.display === state.display));
  });
  $('show-reference').checked = state.showReference;
  document.querySelectorAll('.reference-legend').forEach(item=>item.classList.toggle('reference-hidden',!state.showReference));
  const displayedMethod=state.display==='yarn'&&state.yarnEnabled?'YaRN':'RoPE';
  $('clock-positions').textContent=`${displayedMethod} · Q at m = ${integer(state.delta+state.offset)} · K at n = ${integer(state.offset)}`;
  const rotationNote=state.offset===0
    ? 'K is at n = 0, so its positional rotation is zero.'
    : `Both vectors rotate: Q by ${integer(state.delta+state.offset)}θ and K by ${integer(state.offset)}θ. The shared offset does not change their dot product.`;
  $('clock-rotation-note').textContent=rotationNote+(scalingActive
    ? ` With YaRN, attention scaling also multiplies both vector lengths by ${fmt(amplitude(),3)}.`:'');
  $('rotate-both-demo').classList.toggle('hidden',state.offset!==0);
  document.querySelectorAll('[data-metric]').forEach(button => button.classList.toggle('active', button.dataset.metric === state.metric));
}

// Preserve the text being edited until blur; dependent plots still update on input.
function syncInput(id, value) { if (document.activeElement !== $(id)) $(id).value = value; }

function recalculate() {
  updateControls();
  if (!state.q) return;
  const f = modelFrequencies();
  const a = amplitude(), squared = a * a;
  const baseline = dot(state.q, state.k);
  const rope = scoreAt(state.q, state.k, state.delta, f.original);
  const yarn = scoreAt(state.q, state.k, state.delta, f.scaled);
  const denominator = norm(state.q) * norm(state.k);
  const digits = state.metric === 'cosine' ? 4 : 2;
  $('before-value').textContent = fmt(metricValue(baseline), digits);
  $('rope-value').textContent = fmt(metricValue(rope), digits);
  $('yarn-value').textContent = state.yarnEnabled ? fmt(metricValue(state.metric === 'cosine' ? yarn : yarn * squared), digits) : '—';
  $('before-detail').textContent = `cos θ = ${fmt(baseline / denominator, 4)}`;
  $('rope-detail').textContent = `cos θ = ${fmt(rope / denominator, 4)}`;
  $('yarn-detail').textContent = state.yarnEnabled ? `cos θ = ${fmt(yarn / denominator, 4)}${state.temperature ? ` · a² = ${fmt(squared, 3)}` : ' · no temperature scaling'}` : 'Enable YaRN to compare';
  $('invariant-text').textContent = `‖q‖ = ${fmt(norm(state.q), 3)} · ‖k‖ = ${fmt(norm(state.k), 3)}. RoPE preserves norms.${state.yarnEnabled && state.temperature ? ` YaRN temperature: both norms × ${fmt(a, 3)}.` : ''}`;
  renderClocks(f);
  renderPair(f);
  renderScoreChart(f);
  renderContributions(f);
  renderSoftmax(f);
}

function scheduleRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => { frame = null; recalculate(); });
}

function arrow(x, y, color, {ghost = false, width = 1.8} = {}) {
  const endX = 50 + x, endY = 50 - y;
  const length = Math.hypot(x, y);
  if (length < 0.15) return '';
  if (ghost) return `<line x1="50" y1="50" x2="${endX}" y2="${endY}" stroke="${color}" stroke-opacity=".4" stroke-width="1.1" stroke-dasharray="2 2"/>`;
  const head = Math.min(4.5, length * .4), ux = x / length, uy = -y / length;
  const baseX = endX - ux * head, baseY = endY - uy * head;
  return `<line x1="50" y1="50" x2="${endX}" y2="${endY}" stroke="${color}" stroke-width="${width}" stroke-linecap="round"/><path d="M${endX},${endY} L${baseX - uy * head * .5},${baseY + ux * head * .5} L${baseX + uy * head * .5},${baseY - ux * head * .5} Z" fill="${color}"/>`;
}

function clockSVG(qx, qy, kx, ky, scale, original = null, large = false) {
  const radius = 37;
  let ticks = '';
  for (let i = 0; i < 12; i++) {
    const angle = i * TAU / 12;
    const r = i % 3 ? 34 : 32;
    ticks += `<line x1="${50 + r * Math.cos(angle)}" y1="${50 - r * Math.sin(angle)}" x2="${50 + radius * Math.cos(angle)}" y2="${50 - radius * Math.sin(angle)}" stroke="${colors.axis}" stroke-width="${i % 3 ? '.7' : '1'}"/>`;
  }
  const s = radius * .88 / Math.max(scale, 1e-12);
  const qAngle = Math.atan2(qy, qx), kAngle = Math.atan2(ky, kx);
  let diff = ((qAngle - kAngle + 3 * Math.PI) % TAU) - Math.PI;
  if (diff < -Math.PI) diff += TAU;
  const arcR = 16;
  const arc = large ? `<path d="M${50 + arcR * Math.cos(kAngle)},${50 - arcR * Math.sin(kAngle)} A${arcR},${arcR} 0 0 ${diff >= 0 ? 0 : 1} ${50 + arcR * Math.cos(qAngle)},${50 - arcR * Math.sin(qAngle)}" fill="none" stroke="${colors.axis}" stroke-width="1.8"/>` : '';
  return `<svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="37" fill="none" stroke="${colors.axis}" stroke-width=".8"/><line x1="10" y1="50" x2="90" y2="50" stroke="${colors.grid}" stroke-width=".6"/><line x1="50" y1="10" x2="50" y2="90" stroke="${colors.grid}" stroke-width=".6"/>${ticks}${arc}${original ? arrow(original[0] * s, original[1] * s, colors.query, {ghost:true}) + arrow(original[2] * s, original[3] * s, colors.key, {ghost:true}) : ''}${arrow(qx * s, qy * s, colors.query)}${arrow(kx * s, ky * s, colors.key)}<circle cx="50" cy="50" r="1.7" fill="${colors.text}"/></svg>`;
}

function renderClocks(f) {
  const frequency = state.display === 'yarn' && state.yarnEnabled ? f.scaled : f.original;
  const activeAmplitude = state.display === 'yarn' ? amplitude() : 1;
  const q = rotate(state.q, state.delta + state.offset, frequency);
  const k = rotate(state.k, state.offset, frequency);
  const html = [];
  for (let i = 0; i < 64; i++) {
    const qx = q[i] * activeAmplitude, qy = q[i + 64] * activeAmplitude;
    const kx = k[i] * activeAmplitude, ky = k[i + 64] * activeAmplitude;
    const scale = Math.max(Math.hypot(qx,qy), Math.hypot(kx,ky), Math.hypot(state.q[i],state.q[i+64]), Math.hypot(state.k[i],state.k[i+64]));
    const contribution = qx * kx + qy * ky;
    const band = f.mix[i] === 1 ? 'fast' : f.mix[i] === 0 ? 'slow' : 'middle';
    const tooltip = `Pair ${i}: dimensions ${i}, ${i+64}. θ = ${frequency[i].toExponential(3)} rad/token; period ${integer(TAU / frequency[i])} tokens. Contribution ${fmt(contribution, 4)}.`;
    html.push(`<button type="button" class="clock-button${i === state.selected ? ' selected' : ''}" data-pair="${i}" aria-label="${escape(tooltip)}" aria-pressed="${i === state.selected}" title="${escape(tooltip)}"><span class="clock-top"><span>${String(i).padStart(2,'0')}</span><i class="band-dot ${band}"></i></span>${clockSVG(qx,qy,kx,ky,scale,state.showReference?[state.q[i],state.q[i+64],state.k[i],state.k[i+64]]:null)}<span class="clock-score ${contribution >= 0 ? 'positive' : 'negative'}">${signed(contribution,1)}</span></button>`);
  }
  $('clock-grid').innerHTML = html.join('');
  const fast = Array.from(f.mix).filter(x => x === 1).length;
  const slow = Array.from(f.mix).filter(x => x === 0).length;
  $('band-summary').textContent = `${fast} high-frequency · ${64-fast-slow} transition · ${slow} low-frequency`;
}

function renderPair(f) {
  const i = state.selected;
  $('selected-pair').textContent = String(i).padStart(2,'0');
  $('pair-coordinates').textContent = ` · dimensions ${i}, ${i+64}`;
  const qNorm = Math.hypot(state.q[i],state.q[i+64]), kNorm = Math.hypot(state.k[i],state.k[i+64]);
  const scale = Math.max(qNorm,kNorm) * Math.max(1, amplitude());
  const variants = [
    {name:'Before RoPE', frequency:0, positionQ:0, positionK:0, a:1},
    {name:'RoPE', frequency:f.original[i], positionQ:state.delta+state.offset, positionK:state.offset, a:1},
    {name:'YaRN', frequency:f.scaled[i], positionQ:state.delta+state.offset, positionK:state.offset, a:amplitude()},
  ];
  $('pair-comparison').innerHTML = variants.map((variant,index) => {
    const qa = variant.positionQ * variant.frequency, ka = variant.positionK * variant.frequency;
    const qx = (state.q[i] * Math.cos(qa) - state.q[i+64] * Math.sin(qa)) * variant.a;
    const qy = (state.q[i] * Math.sin(qa) + state.q[i+64] * Math.cos(qa)) * variant.a;
    const kx = (state.k[i] * Math.cos(ka) - state.k[i+64] * Math.sin(ka)) * variant.a;
    const ky = (state.k[i] * Math.sin(ka) + state.k[i+64] * Math.cos(ka)) * variant.a;
    const angle = Math.acos(clamp((qx*kx+qy*ky) / Math.max(1e-30, Math.hypot(qx,qy)*Math.hypot(kx,ky)), -1, 1)) * 180 / Math.PI;
    const phase = state.delta * variant.frequency;
    return `<div class="pair-state${index === 2 ? ` yarn-state${state.yarnEnabled ? '' : ' off'}` : ''}"><h3>${variant.name}${index === 2 && !state.yarnEnabled ? ' · off' : ''}</h3>${clockSVG(qx,qy,kx,ky,scale,index&&state.showReference ? [state.q[i],state.q[i+64],state.k[i],state.k[i+64]] : null,true)}<p class="pair-numbers">Angle: <b>${fmt(angle,1)}°</b><br><span class="angle-line">Position shift: ${fmt(phase / TAU,3)} turns</span></p><div class="state-contribution">${signed(qx*kx+qy*ky,3)}</div></div>`;
  }).join('');
  const speedRatio = f.scaled[i] / f.original[i];
  $('pair-explanation').textContent = `θ = ${f.original[i].toExponential(3)} rad/token · RoPE period: ${integer(TAU/f.original[i])} tokens · YaRN frequency: ${fmt(speedRatio * 100,1)}% of original${state.temperature && state.yarnEnabled ? ` · norms × ${fmt(amplitude(),3)}` : ''}. Values below show the pair contribution to q·k.`;
}

function canvasContext(canvas, height) {
  const width = canvas.clientWidth;
  if (!width) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.font = '10px Menlo, Consolas, monospace';
  return {ctx,width,height};
}

function chartData(f) {
  const key = JSON.stringify([state.words,state.queryIndex,state.keyIndex,state.activeHead,state.shared,state.ropeBase,state.factor,state.originalLength,state.betaFast,state.betaSlow,state.chartRange,state.metric,effectiveMultiplier(),state.yarnEnabled]);
  if (key === lastChartKey && chartCache) return chartCache;
  const count = Math.min(state.chartRange + 1,4097);
  const x = [], rope = [], yarn = [];
  for (let i = 0; i < count; i++) {
    const distance = Math.round(i * state.chartRange / (count - 1));
    x.push(distance);
    rope.push(metricValue(scoreAt(state.q,state.k,distance,f.original)));
    const rawYarn = scoreAt(state.q,state.k,distance,f.scaled);
    yarn.push(metricValue(rawYarn * (state.metric === 'cosine' ? 1 : amplitude() ** 2)));
  }
  const baseline = metricValue(dot(state.q,state.k));
  const all = state.yarnEnabled ? [...rope,...yarn,baseline,0] : [...rope,baseline,0];
  let min = Math.min(...all), max = Math.max(...all);
  const padding = Math.max((max-min) * .12, .01);
  chartCache = {x,rope,yarn,baseline,min:min-padding,max:max+padding};
  lastChartKey = key;
  return chartCache;
}

let chartGeometry;
function renderScoreChart(f) {
  const canvas = $('score-chart');
  const c = canvasContext(canvas,270); if (!c) return;
  const {ctx,width,height} = c;
  const d = chartData(f);
  const left = 44, right = 12, top = 12, bottom = 32;
  const plotWidth = width-left-right, plotHeight = height-top-bottom;
  const px = x => left + x / state.chartRange * plotWidth;
  const py = y => top + (d.max-y)/(d.max-d.min)*plotHeight;
  chartGeometry = {left,plotWidth,px,py};
  ctx.lineWidth = 1; ctx.textAlign = 'right'; ctx.fillStyle = colors.text;
  for (let i = 0; i <= 4; i++) {
    const y = d.min + (d.max-d.min)*i/4;
    ctx.strokeStyle = colors.grid;
    ctx.beginPath(); ctx.moveTo(left,py(y)); ctx.lineTo(width-right,py(y)); ctx.stroke();
    ctx.fillText(fmt(y,state.metric === 'cosine' ? 2 : 1),left-8,py(y)+3);
  }
  ctx.textAlign = 'center';
  for (let i=0;i<=4;i++) {
    const x = state.chartRange*i/4;
    ctx.fillText(x >= 10000 ? `${fmt(x/1000,x%1000 ? 1 : 0)}K` : integer(x),px(x),height-12);
  }
  ctx.setLineDash([4,4]);ctx.strokeStyle = colors.baseline;
  ctx.beginPath();ctx.moveTo(left,py(d.baseline));ctx.lineTo(width-right,py(d.baseline));ctx.stroke();ctx.setLineDash([]);
  const line = (series,color) => {
    ctx.strokeStyle = color;ctx.lineWidth = 1.25;ctx.globalAlpha = .85;ctx.beginPath();
    series.forEach((value,i) => i ? ctx.lineTo(px(d.x[i]),py(value)) : ctx.moveTo(px(d.x[i]),py(value)));
    ctx.stroke();ctx.globalAlpha = 1;
  };
  ctx.save();ctx.beginPath();ctx.rect(left,top,plotWidth,plotHeight);ctx.clip();
  line(d.rope,colors.rope);if(state.yarnEnabled)line(d.yarn,colors.yarn);
  if(state.delta <= state.chartRange){
    const x = px(state.delta);
    ctx.strokeStyle=colors.axis;ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,height-bottom);ctx.stroke();ctx.setLineDash([]);
    for(const [frequency,color,multiplier] of [[f.original,colors.rope,1],...(state.yarnEnabled ? [[f.scaled,colors.yarn,state.metric === 'cosine' ? 1 : amplitude()**2]] : [])]){
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,py(metricValue(scoreAt(state.q,state.k,state.delta,frequency)*multiplier)),3.5,0,TAU);ctx.fill();
    }
  }
  ctx.restore();
  $('chart-note').textContent = `Scores are oscillatory and need not decrease monotonically with distance.${state.chartRange > 4096 ? ` Showing ${integer(d.x.length)} sampled distances; high-frequency oscillations may be undersampled at this range.` : ' All integer distances are shown.'}${state.delta > state.chartRange ? ' Current Δ is outside the plot. Increase the chart range.' : ''}`;
}

let contributionGeometry;
function renderContributions(f) {
  const c = canvasContext($('contribution-chart'),180);if(!c)return;
  const {ctx,width,height}=c;
  const rope = Array.from({length:64},(_,i)=>pairContribution(state.q,state.k,i,state.delta,f.original[i]));
  const yarn = Array.from({length:64},(_,i)=>pairContribution(state.q,state.k,i,state.delta,f.scaled[i])*amplitude()**2);
  const max = Math.max(...rope.map(Math.abs),...(state.yarnEnabled ? yarn.map(Math.abs) : []),.001)*1.12;
  const left=34,top=8,bottom=27,plotHeight=height-top-bottom,plotWidth=width-left-4,step=plotWidth/64;
  const zero=top+plotHeight/2,py=y=>zero-y/max*plotHeight/2;
  contributionGeometry={left,step};
  ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.text;ctx.textAlign='right';
  for(const y of [-max/1.12,0,max/1.12]){
    ctx.beginPath();ctx.moveTo(left,py(y));ctx.lineTo(width,py(y));ctx.stroke();ctx.fillText(fmt(y,1),left-7,py(y)+3);
  }
  ctx.fillStyle=colors.selected;ctx.fillRect(left+state.selected*step,top,step,plotHeight);
  for(let i=0;i<64;i++){
    const w=state.yarnEnabled ? step*.34 : step*.65;
    for(const [series,color,offset] of [[rope,colors.rope,.1],...(state.yarnEnabled?[[yarn,colors.yarn,.5]]:[])]){
      ctx.fillStyle=color;ctx.globalAlpha=i===state.selected?1:.65;
      ctx.fillRect(left+i*step+step*offset,Math.min(zero,py(series[i])),Math.max(1,w),Math.max(.5,Math.abs(py(series[i])-zero)));
    }
  }
  ctx.globalAlpha=1;ctx.textAlign='center';ctx.fillStyle=colors.text;
  for(const i of [0,8,16,24,32,40,48,56,63])ctx.fillText(String(i).padStart(2,'0'),left+(i+.5)*step,height-9);
  $('contribution-temp-note').textContent=state.temperature&&state.yarnEnabled?`including λ = ${fmt(effectiveMultiplier(),3)}`:'no temperature scaling';
  $('contribution-sum').textContent=`Σ = ${fmt((state.yarnEnabled?yarn:rope).reduce((a,b)=>a+b,0),2)}`;
}

async function loadAttentionKeys() {
  clearTimeout(attentionDebounce);
  const text = $('attention-text').value;
  const head = state.activeHead;
  const version = ++attentionVersion;
  state.attentionText = text;
  if (state.attentionLoadedText === text && state.attentionHead === head && state.attentionKeys.length && !state.attentionError) {
    state.attentionLoading = false;
    scheduleRender();
    return;
  }
  state.attentionLoading = true; state.attentionError = '';
  scheduleRender();
  try {
    await ensureModel();
    if (version !== attentionVersion) return;
    const ids = model.tokenizer.encode(text);
    if (ids.length < 2) throw new Error('Enter at least two key tokens to compare attention probabilities.');
    if (ids.length > 64) throw new Error('Use up to 64 key tokens in this comparison. Shorten the text.');
    const {keys} = await model.keyVectors(ids, head);
    if (version !== attentionVersion) return;
    state.attentionIds = ids; state.attentionKeys = keys;
    state.attentionHead = head; state.attentionLoadedText = text;
  } catch (error) {
    if (version === attentionVersion) state.attentionError = error.message || 'Unable to compute key vectors. Please retry.';
  } finally {
    if (version === attentionVersion) { state.attentionLoading = false; scheduleRender(); }
  }
}

function softmaxData(f = modelFrequencies()) {
  if (!state.q || state.attentionLoading || state.attentionError || state.attentionLoadedText !== state.attentionText || state.attentionHead !== state.activeHead || !state.attentionKeys.length) return null;
  const frequency = state.yarnEnabled ? f.scaled : f.original;
  // Keys occupy consecutive positions; Δ is the distance to the final key.
  // The comparison is normalized over these keys only, using actual layer-0 K.
  const distances = state.attentionIds.map((_,i) => state.delta + state.attentionIds.length - 1 - i);
  const logits = state.attentionKeys.map((key,i) => scoreAt(state.q, state.shared ? state.q : key, distances[i], frequency) / Math.sqrt(state.q.length));
  return {ids:state.attentionIds, distances, logits, before:softmax(logits), after:softmax(logits, effectiveMultiplier())};
}

function renderSoftmax(f) {
  const container = $('softmax-distribution');
  const data = softmaxData(f);
  const pending = !state.attentionError && (state.attentionLoading || state.attentionLoadedText !== state.attentionText || state.attentionHead !== state.activeHead);
  container.setAttribute('aria-busy', String(pending));
  $('attention-status').classList.toggle('error', Boolean(state.attentionError));
  $('attention-status').textContent = state.attentionError || (pending ? `Loading keys for head ${state.activeHead}…` : `${state.attentionIds.length} Qwen tokens · ${state.shared ? 'Q = K experiment: every K is replaced by Q' : `learned K from KV head ${kvHeadForQuery(state.activeHead,model.config)}`} · probabilities normalized over these keys only`);
  $('softmax-scaled-legend').textContent = `With correction · λ = ${fmt(effectiveMultiplier(),4)}`;
  if (!data) {
    container.innerHTML = `<p class="attention-empty">${escape(state.attentionError || 'Computing token attention…')}</p>`;
    for (const id of ['softmax-peak','softmax-entropy','softmax-effective','softmax-redistributed']) $(id).textContent = '—';
    $('attention-effect').textContent = 'Compare the same attention scores with and without temperature correction.';
    $('attention-axis-scale').textContent = 'Shared probability scale';
    return;
  }
  const {ids, distances, logits, before, after} = data;
  const lambda = effectiveMultiplier();
  const peakIndex = logits.indexOf(Math.max(...logits));
  const hBefore = entropy(before), hAfter = entropy(after);
  const pct = x => `${fmt(x*100,2)}%`;
  const change = i => (after[i]-before[i])*100;
  const maximum = Math.min(1, Math.ceil(Math.max(...before, ...after)*10)/10);
  $('attention-axis-scale').textContent = `Both bar columns: 0–${fmt(maximum*100,0)}%`;
  const header = `<div class="attention-row attention-table-header" role="row"><span role="columnheader">Key token</span><span role="columnheader">Without correction</span><span role="columnheader">With correction</span><span role="columnheader">Change · pp</span></div>`;
  const probabilityCell = (probability, kind) => `<div class="attention-probability" role="cell"><div class="attention-track"><span class="attention-bar ${kind}" style="width:${probability/maximum*100}%"></span></div><span class="attention-probability-value">${pct(probability)}</span></div>`;
  container.innerHTML = header + ids.map((id,i) => {
    const label = model.tokenizer.label(id);
    return `<div class="attention-row${i===peakIndex?' attention-winner':''}" role="row" data-attention-index="${i}"><div class="attention-token" role="cell" title="Token ${id} · position ${i} · distance ${distances[i]} · logit ${fmt(logits[i],4)}"><span>${escape(label)}</span><small>#${i} · ${integer(distances[i])} tokens before Q${i===peakIndex?' · top score':''}</small></div>${probabilityCell(before[i],'query')}${probabilityCell(after[i],'key')}<div class="attention-change${change(i)>1e-8?' gain':change(i)<-1e-8?' loss':''}" role="cell">${signed(change(i),2)}<span class="mobile-pp"> pp</span></div></div>`;
  }).join('');
  const redistributed = after.reduce((sum,p,i) => sum+Math.abs(p-before[i]),0)/2;
  $('softmax-peak').textContent = `${pct(before[peakIndex])} → ${pct(after[peakIndex])}`;
  $('softmax-entropy').textContent = `${fmt(hBefore,3)} → ${fmt(hAfter,3)}`;
  $('softmax-effective').textContent = `${fmt(2**hBefore,2)} → ${fmt(2**hAfter,2)}`;
  $('softmax-redistributed').textContent = `${fmt(redistributed*100,2)} pp`;
  const token = model.tokenizer.label(ids[peakIndex]);
  const noCorrection = !state.yarnEnabled ? 'YaRN is off, so no temperature correction is applied. Both distributions are identical.'
    : !state.temperature ? 'Temperature correction is switched off. Both distributions are identical; check Apply correction to compare.'
    : 'λ = 1: no temperature correction is applied, so both distributions are identical. Move the slider to compare.';
  $('attention-effect').textContent = lambda === 1
    ? noCorrection
    : `“${token}” keeps the highest score: ${pct(before[peakIndex])} → ${pct(after[peakIndex])} (${signed(change(peakIndex),2)} percentage points). ${lambda>1?'The correction makes the distribution more concentrated.':'The lower multiplier makes the distribution more even.'} It changes confidence, not the ranking of tokens.`;
  const concentrated = Math.max(...before)>.95 && lambda>1 ? ' The original distribution is already concentrated above 95%, so there is little room to sharpen it further.' : '';
  $('softmax-explanation').textContent = `Each probability column sums to 100%. Change is measured in percentage points (pp): positive values gain attention, negative values lose it. Both columns use the same ${state.yarnEnabled?'YaRN':'RoPE'} frequencies and logits; only the multiplier changes.${concentrated} Layer 0, over the entered keys only; Q’s own key and unspecified intervening tokens are excluded.${state.shared?' Shared Q = K is active, so the keys are experimentally tied instead of using their learned vectors.':''}`;
}

function setDistance(value) {
  state.delta=clamp(Math.round(Number(value)||0),0,1310720);
  scheduleRender();
}

$('words-form').addEventListener('submit',event=>{event.preventDefault();loadWords();});
$('retry').addEventListener('click',()=>loadWords(true));
$('shared-vectors').addEventListener('change',event=>{
  state.shared=event.target.checked;
  // An empty hidden K input used Q's token as the checkpoint fallback.
  // Expose that text when restoring the learned K projection.
  if(!state.shared&&!$('key-word').value){
    $('key-word').value=$('query-word').value;
    loadWords(true);
  }
  applyVectorMode();
  if($('load-status').classList.contains('ready')) {
    setStatus(state.shared ? `Ready · Q head ${state.activeHead} · Q = K · token ${state.queryIds[state.queryIndex]}`
      : `Ready · head ${state.activeHead} / KV ${kvHeadForQuery(state.activeHead,model.config)} · tokens ${state.queryIds[state.queryIndex]} / ${state.keyIds[state.keyIndex]}`,false,true);
  }
  scheduleRender();
});
function selectHead(value) {
  state.head=clamp(Number(value),0,(model.config?.num_attention_heads??16)-1);
  $('results').classList.add('stale');
  updateControls();
  loadWords(true);
}
$('head-select').addEventListener('change',event=>selectHead(event.target.value));
$('previous-head').addEventListener('click',()=>selectHead(state.head-1));
$('next-head').addEventListener('click',()=>selectHead(state.head+1));
for(const id of ['query-word','key-word'])$(id).addEventListener('input',()=>{
  clearTimeout(debounce);requestVersion++;
  $('results').classList.add('stale');$('load-words').disabled=false;
  setStatus('Input changed. Updating Q and K…');
  debounce=setTimeout(()=>loadWords(),700);
});
for(const kind of ['query','key'])$(`${kind}-tokens`).addEventListener('click',event=>{
  const button=event.target.closest('[data-index]');if(!button)return;
  state[`${kind}Index`]=Number(button.dataset.index);
  loadWords(true);
});
$('distance').addEventListener('input',event=>setDistance(event.target.value));
$('distance-slider').addEventListener('input',event=>setDistance(event.target.value));
$('distance-range').addEventListener('change',event=>{
  $('distance-slider').max=event.target.value;
  setDistance(Math.min(state.delta,Number(event.target.value)));
});
document.querySelectorAll('[data-distance]').forEach(button=>button.addEventListener('click',()=>setDistance(button.dataset.distance)));
$('offset').addEventListener('input',event=>{state.offset=clamp(Math.round(Number(event.target.value)||0),0,1000000);scheduleRender();});
$('yarn-enabled').addEventListener('change',event=>{
  state.yarnEnabled=event.target.checked;state.display=state.yarnEnabled?'yarn':'rope';scheduleRender();
});
for(const id of ['factor','factor-slider'])$(id).addEventListener('input',event=>{
  state.factor=clamp(Number(event.target.value)||1,1,32);
  $(id==='factor'?'factor-slider':'factor').value=state.factor;scheduleRender();
});
for(const id of ['temperature','softmax-apply'])$(id).addEventListener('change',event=>{state.temperature=event.target.checked;scheduleRender();});
function setBase(value) {
  if(!Number.isFinite(Number(value))||value==='')return;
  state.ropeBase=clamp(Number(value),100,1e8);scheduleRender();
}
$('rope-base').addEventListener('input',event=>setBase(event.target.value));
$('rope-base').addEventListener('change',event=>{event.target.value=state.ropeBase;});
$('base-slider').addEventListener('input',event=>setBase(Math.round(10**Number(event.target.value))));
document.querySelectorAll('[data-base]').forEach(button=>button.addEventListener('click',()=>setBase(button.dataset.base)));
$('temperature-mode').addEventListener('change',event=>{
  if(event.target.value==='manual')state.logitMultiplier=configuredMultiplier();
  state.temperatureMode=event.target.value;scheduleRender();
});
for(const id of ['temperature-coefficient','coefficient-slider'])$(id).addEventListener('input',event=>{
  if(!Number.isFinite(event.target.valueAsNumber))return;
  state.temperatureCoefficient=clamp(event.target.valueAsNumber,0,.5);scheduleRender();
});
$('temperature-coefficient').addEventListener('change',event=>{event.target.value=state.temperatureCoefficient;});
for(const id of ['logit-scale','logit-scale-slider','softmax-scale'])$(id).addEventListener('input',event=>{
  if(!Number.isFinite(event.target.valueAsNumber))return;
  state.logitMultiplier=clamp(event.target.valueAsNumber,.01,8);
  state.temperatureMode='manual';scheduleRender();
});
$('logit-scale').addEventListener('change',event=>{event.target.value=state.logitMultiplier;});
$('restore-yarn-temperature').addEventListener('click',()=>{state.temperatureMode='auto';scheduleRender();});
$('original-length').addEventListener('input',event=>{
  if(!Number.isFinite(event.target.valueAsNumber))return;
  state.originalLength=clamp(Math.round(event.target.valueAsNumber),128,131072);scheduleRender();
});
$('original-length').addEventListener('change',event=>{event.target.value=state.originalLength;});
for(const [id,property,min,max] of [['beta-fast','betaFast',1.1,128],['beta-slow','betaSlow',.1,32]]){
  $(id).addEventListener('input',event=>{
    if(!Number.isFinite(event.target.valueAsNumber))return;
    state[property]=clamp(event.target.valueAsNumber,min,max);
    if(state.betaFast<=state.betaSlow){
      state.betaFast=state.betaSlow+.1;
      if(property!=='betaFast')$('beta-fast').value=state.betaFast;
    }
    scheduleRender();
  });
  $(id).addEventListener('change',event=>{event.target.value=state[property];});
}
document.querySelectorAll('[data-display]').forEach(button=>button.addEventListener('click',()=>{state.display=button.dataset.display;scheduleRender();}));
$('show-reference').addEventListener('change',event=>{state.showReference=event.target.checked;scheduleRender();});
$('rotate-both-demo').addEventListener('click',()=>{
  state.offset=100;$('offset').value=100;
  document.querySelector('.offset-details').open=true;
  scheduleRender();
});
document.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>{state.metric=button.dataset.metric;scheduleRender();}));
$('clock-grid').addEventListener('click',event=>{const button=event.target.closest('[data-pair]');if(button){state.selected=Number(button.dataset.pair);scheduleRender();}});
$('clock-grid').addEventListener('keydown',event=>{
  const columns=getComputedStyle($('clock-grid')).gridTemplateColumns.split(' ').length;
  const changes={ArrowLeft:-1,ArrowRight:1,ArrowUp:-columns,ArrowDown:columns};
  if(changes[event.key]===undefined)return;event.preventDefault();state.selected=(state.selected+changes[event.key]+64)%64;recalculate();
  $('clock-grid').querySelector(`[data-pair="${state.selected}"]`)?.focus();
});
$('previous-pair').addEventListener('click',()=>{state.selected=(state.selected+63)%64;scheduleRender();});
$('next-pair').addEventListener('click',()=>{state.selected=(state.selected+1)%64;scheduleRender();});
$('chart-range').addEventListener('change',event=>{state.chartRange=Number(event.target.value);scheduleRender();});
function chartPointer(event,select=false){
  if(!state.q||!chartGeometry)return;
  const rect=$('score-chart').getBoundingClientRect();
  const x=event.clientX-rect.left;
  const distance=clamp(Math.round((x-chartGeometry.left)/chartGeometry.plotWidth*state.chartRange),0,state.chartRange);
  if(select)setDistance(distance);
  const f=modelFrequencies();
  const r=metricValue(scoreAt(state.q,state.k,distance,f.original));
  const y=metricValue(scoreAt(state.q,state.k,distance,f.scaled)*(state.metric==='cosine'?1:amplitude()**2));
  const tooltip=$('chart-tooltip');
  tooltip.innerHTML=`Δ ${integer(distance)}<br><span style="color:${colors.rope}">RoPE ${fmt(r,state.metric==='cosine'?4:2)}</span>${state.yarnEnabled?`<br><span style="color:${colors.yarn}">YaRN ${fmt(y,state.metric==='cosine'?4:2)}</span>`:''}`;
  tooltip.style.left=`${clamp(x+12,0,rect.width-145)}px`;tooltip.style.top='12px';tooltip.classList.remove('hidden');
}
$('score-chart').addEventListener('pointerdown',event=>{if(event.button===0){$('score-chart').setPointerCapture(event.pointerId);chartPointer(event,true);}});
$('score-chart').addEventListener('pointermove',event=>chartPointer(event,event.buttons===1));
$('score-chart').addEventListener('pointerleave',()=>$('chart-tooltip').classList.add('hidden'));
$('score-chart').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();setDistance(state.delta+(event.key==='ArrowRight'?1:-1)*(event.shiftKey?100:1));}});
$('contribution-chart').addEventListener('click',event=>{
  if(!contributionGeometry)return;
  const rect=event.currentTarget.getBoundingClientRect();
  state.selected=clamp(Math.floor((event.clientX-rect.left-contributionGeometry.left)/contributionGeometry.step),0,63);scheduleRender();
});
$('contribution-chart').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();state.selected=(state.selected+(event.key==='ArrowRight'?1:63))%64;scheduleRender();}});
$('attention-form').addEventListener('submit',event=>{event.preventDefault();loadAttentionKeys();});
$('attention-text').addEventListener('input',()=>{
  clearTimeout(attentionDebounce); attentionVersion++;
  state.attentionText=$('attention-text').value;
  state.attentionLoading=true; state.attentionError='';
  scheduleRender();
  attentionDebounce=setTimeout(()=>loadAttentionKeys(),700);
});
$('attention-gap').addEventListener('input',event=>setDistance(event.target.value));
$('attention-gap').addEventListener('change',event=>{event.target.value=state.delta;});
$('reset').addEventListener('click',()=>{
  Object.assign(state,{head:0,shared:false,delta:10,offset:0,factor:4,yarnEnabled:true,temperature:true,originalLength:40960,betaFast:32,betaSlow:1,selected:0,display:'yarn',metric:'dot',chartRange:8192,showReference:true,ropeBase:1e6,temperatureMode:'auto',temperatureCoefficient:.1,logitMultiplier:yarnAmplitude(4)**2,attentionText:'The cat is soft and fluffy.'});
  applyVectorMode();
  $('attention-text').value=state.attentionText;
  for(const [id,value]of [['query-word','cat'],['key-word','fluffy'],['offset',0],['factor',4],['factor-slider',4],['original-length',40960],['beta-fast',32],['beta-slow',1],['chart-range',8192],['distance-range',4096]])$(id).value=value;
  $('distance-slider').max=4096;$('yarn-enabled').checked=true;$('temperature').checked=true;
  loadWords();scheduleRender();
});
new ResizeObserver(()=>{if(state.q)scheduleRender();}).observe($('results'));

const compactLayout=window.matchMedia('(max-width: 840px)');
function toggleParameters(open,returnFocus=false){
  document.body.classList.toggle('controls-open',open);
  $('parameters-toggle').setAttribute('aria-expanded',String(open));
  $('experiment-controls').inert=compactLayout.matches&&!open;
  if(open)requestAnimationFrame(()=>$('close-controls').focus());
  else if(returnFocus&&compactLayout.matches)$('parameters-toggle').focus();
}
$('parameters-toggle').addEventListener('click',()=>toggleParameters(!document.body.classList.contains('controls-open')));
$('close-controls').addEventListener('click',()=>toggleParameters(false,true));
$('controls-backdrop').addEventListener('click',()=>toggleParameters(false,true));
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&document.body.classList.contains('controls-open'))toggleParameters(false,true);
});
compactLayout.addEventListener('change',()=>toggleParameters(false));
toggleParameters(false);

updateControls();
loadWords();

export {state,model,recalculate,loadWords,modelFrequencies,softmaxData,effectiveMultiplier,loadAttentionKeys};
