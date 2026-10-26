import {decodeBF16, matvec, rmsNorm} from './math.js';

export const MODEL = 'Qwen/Qwen3-0.6B';
export const REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca';
const REMOTE = `https://huggingface.co/${MODEL}/resolve/${REVISION}`;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function kvHeadForQuery(head, config) {
  if (!Number.isInteger(head) || head < 0 || head >= config.num_attention_heads) {
    throw new RangeError(`Query head must be an integer from 0 to ${config.num_attention_heads - 1}.`);
  }
  const groupSize = config.num_attention_heads / config.num_key_value_heads;
  if (!Number.isInteger(groupSize) || groupSize < 1) {
    throw new Error('Unsupported grouped-query attention configuration.');
  }
  return Math.floor(head / groupSize);
}

// The checkpoint's actual tokenizer: NFC → Qwen Split regex → byte-level BPE.
// The only regex translation is Oniguruma's case-insensitive contraction group.
export class QwenTokenizer {
  constructor(data) {
    this.vocab = data.model.vocab;
    this.idToToken = new Map(Object.entries(this.vocab).map(([token, id]) => [id, token]));
    this.ranks = new Map(data.model.merges.map((pair, i) => [Array.isArray(pair) ? pair.join(' ') : pair, i]));
    const byteValues = [...Array.from({length: 94}, (_, i) => i + 33), ...Array.from({length: 12}, (_, i) => i + 161), ...Array.from({length: 82}, (_, i) => i + 174)];
    const unicodeValues = [...byteValues];
    let extra = 0;
    for (let byte = 0; byte < 256; byte++) {
      if (!byteValues.includes(byte)) {
        byteValues.push(byte);
        unicodeValues.push(256 + extra++);
      }
    }
    this.byteToChar = new Map(byteValues.map((byte, i) => [byte, String.fromCodePoint(unicodeValues[i])]));
    this.charToByte = new Map([...this.byteToChar].map(([byte, char]) => [char, byte]));
    const regex = data.pre_tokenizer.pretokenizers[0].pattern.Regex.replace("(?i:'s|'t|'re|'ve|'m|'ll|'d)", "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])");
    this.pattern = new RegExp(regex, 'gu');
    this.specials = data.added_tokens.filter(token => token.special).sort((a, b) => b.content.length - a.content.length);
    this.memo = new Map();
  }

  bpe(text) {
    if (this.memo.has(text)) return this.memo.get(text);
    let pieces = Array.from(text);
    while (pieces.length > 1) {
      let best = Infinity, index = -1;
      for (let i = 0; i < pieces.length - 1; i++) {
        const rank = this.ranks.get(`${pieces[i]} ${pieces[i + 1]}`);
        if (rank !== undefined && rank < best) { best = rank; index = i; }
      }
      if (index < 0) break;
      const first = pieces[index], second = pieces[index + 1];
      const merged = [];
      for (let i = 0; i < pieces.length; i++) {
        if (pieces[i] === first && pieces[i + 1] === second) { merged.push(first + second); i++; }
        else merged.push(pieces[i]);
      }
      pieces = merged;
    }
    const result = pieces.map(piece => {
      const id = this.vocab[piece];
      if (id === undefined) throw new Error('Token not found in the Qwen vocabulary.');
      return id;
    });
    this.memo.set(text, result);
    return result;
  }

  encode(text) {
    text = text.normalize('NFC');
    // Special-token strings are never silently converted into normal text.
    const escaped = this.specials.map(token => token.content.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const chunks = text.split(new RegExp(`(${escaped.join('|')})`, 'u'));
    const ids = [];
    for (const chunk of chunks) {
      const special = this.specials.find(token => token.content === chunk);
      if (special) { ids.push(special.id); continue; }
      for (const match of chunk.matchAll(this.pattern)) {
        const bytes = Array.from(encoder.encode(match[0]), byte => this.byteToChar.get(byte)).join('');
        ids.push(...this.bpe(bytes));
      }
    }
    return ids;
  }

  decode(id) {
    const special = this.specials.find(token => token.id === id);
    if (special) return special.content;
    const token = this.idToToken.get(id);
    return decoder.decode(Uint8Array.from(Array.from(token), char => this.charToByte.get(char)));
  }

  label(id) {
    return this.decode(id).replaceAll(' ', '␣').replaceAll('\n', '↵').replaceAll('\t', '⇥');
  }
}

export class QwenModel {
  constructor(onProgress = () => {}) {
    this.onProgress = onProgress;
    this.downloaded = 0;
    this.requests = new Map();
    this.embeddings = new Map();
    this.heads = new Map();
  }

  async fetchFile(name, start, end) {
    const key = `${name}:${start ?? ''}:${end ?? ''}`;
    if (this.requests.has(key)) return this.requests.get(key);
    const promise = this.fetchUncached(name, start, end);
    this.requests.set(key, promise);
    try { return await promise; }
    catch (error) { this.requests.delete(key); throw error; }
  }

  async fetchUncached(name, start, end) {
    const ranged = start !== undefined;
    const url = `/model/${name}${ranged ? `?start=${start}&end=${end}` : ''}`;
    const cacheKey = new Request(`${location.origin}/__qwen_cache/${REVISION}/${name}${ranged ? `?start=${start}&end=${end}` : ''}`);
    let cache;
    try {
      cache = await caches.open(`qwen-rope-${REVISION}`);
      const cached = await cache.match(cacheKey);
      if (cached) return await cached.arrayBuffer();
    } catch { /* Browser storage can be unavailable; network still works. */ }
    let response = await fetch(url, {signal: AbortSignal.timeout(60000)});
    // A static host can serve the same site without Python, using HF's CORS API.
    if (response.status === 404 || (response.ok && response.headers.get('content-type')?.includes('text/html'))) {
      response = await fetch(`${REMOTE}/${name}${ranged ? `?range=${start}-${end}` : ''}`, {
        headers: ranged ? {Range: `bytes=${start}-${end}`} : {}, signal: AbortSignal.timeout(60000),
      });
      if (ranged && response.status !== 206) throw new Error('The model host did not honor the byte range. Run server.py.');
    }
    if (!response.ok) throw new Error(`Unable to load ${name} (${response.status}). Check your connection and select Retry.`);
    const buffer = await response.arrayBuffer();
    if (ranged && buffer.byteLength !== end - start + 1) throw new Error('Incomplete checkpoint range. Select Retry.');
    this.downloaded += buffer.byteLength;
    this.onProgress(this.downloaded);
    if (cache) {
      try { await cache.put(cacheKey, new Response(buffer)); } catch { /* Optional cache. */ }
    }
    return buffer;
  }

  async initialize() {
    const [config, tokenizer, headerSize] = await Promise.all([
      this.fetchFile('config.json'), this.fetchFile('tokenizer.json'), this.fetchFile('model.safetensors', 0, 7),
    ]);
    this.config = JSON.parse(decoder.decode(config));
    this.tokenizer = new QwenTokenizer(JSON.parse(decoder.decode(tokenizer)));
    this.headerLength = Number(new DataView(headerSize).getBigUint64(0, true));
    if (this.headerLength > 1024 * 1024) throw new Error('Unexpected checkpoint header.');
    this.header = JSON.parse(decoder.decode(await this.fetchFile('model.safetensors', 8, 7 + this.headerLength)));
    const prefix = 'model.layers.0.';
    [this.inputNorm, this.qNorm, this.kNorm] = await Promise.all([
      this.tensor(`${prefix}input_layernorm.weight`),
      this.tensor(`${prefix}self_attn.q_norm.weight`),
      this.tensor(`${prefix}self_attn.k_norm.weight`),
    ]);
    return this;
  }

  async headWeights(head) {
    const kvHead = kvHeadForQuery(head, this.config);
    if (!this.heads.has(head)) {
      const dimension = this.config.head_dim;
      const prefix = 'model.layers.0.self_attn.';
      const promise = Promise.all([
        this.tensor(`${prefix}q_proj.weight`, head * dimension, dimension),
        this.tensor(`${prefix}k_proj.weight`, kvHead * dimension, dimension),
      ]).then(([wq, wk]) => ({wq, wk, kvHead}));
      this.heads.set(head, promise);
      promise.catch(() => this.heads.delete(head));
    }
    return this.heads.get(head);
  }

  async tensor(name, rowStart, rowCount = 1) {
    const info = this.header[name];
    if (!info || info.dtype !== 'BF16') throw new Error(`Unexpected tensor format: ${name}.`);
    const rowBytes = info.shape.length > 1 ? info.shape[1] * 2 : 2;
    const relativeStart = rowStart === undefined ? info.data_offsets[0] : info.data_offsets[0] + rowStart * rowBytes;
    const relativeEnd = rowStart === undefined ? info.data_offsets[1] : relativeStart + rowCount * rowBytes;
    const offset = 8 + this.headerLength;
    return decodeBF16(await this.fetchFile('model.safetensors', offset + relativeStart, offset + relativeEnd - 1));
  }

  async embedding(tokenId) {
    if (!this.embeddings.has(tokenId)) {
      const promise = this.tensor('model.embed_tokens.weight', tokenId);
      this.embeddings.set(tokenId, promise);
      promise.catch(() => this.embeddings.delete(tokenId));
    }
    return this.embeddings.get(tokenId);
  }

  async vectors(queryId, keyId, head = 0) {
    const [queryEmbedding, keyEmbedding, {wq, wk, kvHead}] = await Promise.all([
      this.embedding(queryId), this.embedding(keyId), this.headWeights(head),
    ]);
    const epsilon = this.config.rms_norm_eps;
    const q = rmsNorm(matvec(wq, rmsNorm(queryEmbedding, this.inputNorm, epsilon)), this.qNorm, epsilon);
    const k = rmsNorm(matvec(wk, rmsNorm(keyEmbedding, this.inputNorm, epsilon)), this.kNorm, epsilon);
    return {q, k, head, kvHead};
  }

  async keyVectors(tokenIds, head = 0) {
    const [embeddings, {wk, kvHead}] = await Promise.all([
      Promise.all(tokenIds.map(id => this.embedding(id))), this.headWeights(head),
    ]);
    const epsilon = this.config.rms_norm_eps;
    const keys = embeddings.map(embedding => rmsNorm(matvec(wk, rmsNorm(embedding, this.inputNorm, epsilon)), this.kNorm, epsilon));
    return {keys, head, kvHead};
  }
}
