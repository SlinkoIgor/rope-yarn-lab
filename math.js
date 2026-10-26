// Qwen uses the split-half convention: pair i is (x[i], x[i + d/2]).
export const TAU = 2 * Math.PI;
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
export const norm = a => Math.sqrt(dot(a, a));
export const cosine = (a, b) => {
  const denominator = norm(a) * norm(b);
  return denominator > 0 ? dot(a, b) / denominator : 0;
};

export function rmsNorm(x, weight, epsilon = 1e-6) {
  const scale = 1 / Math.sqrt(dot(x, x) / x.length + epsilon);
  return Float64Array.from(x, (value, i) => value * scale * weight[i]);
}

export function matvec(matrix, x) {
  const rows = matrix.length / x.length;
  const result = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let col = 0; col < x.length; col++) sum += matrix[row * x.length + col] * x[col];
    result[row] = sum;
  }
  return result;
}

export function frequencies(dimension, base = 1e6) {
  return Float64Array.from({length: dimension / 2}, (_, i) => base ** (-2 * i / dimension));
}

// YaRN's linear ramp is over frequency indices, with the original floor/ceil
// correction boundaries (beta_fast=32, beta_slow=1), as in Transformers.
export function yarnFrequencies(dimension, base, factor, originalLength, betaFast = 32, betaSlow = 1) {
  const original = frequencies(dimension, base);
  const correction = rotations => dimension * Math.log(originalLength / (rotations * TAU)) / (2 * Math.log(base));
  const low = Math.max(0, Math.floor(correction(betaFast)));
  const high = Math.min(dimension - 1, Math.ceil(correction(betaSlow)));
  const width = high === low ? 0.001 : high - low;
  const mix = Float64Array.from(original, (_, i) => 1 - clamp((i - low) / width, 0, 1));
  const scaled = Float64Array.from(original, (frequency, i) => frequency * (mix[i] + (1 - mix[i]) / factor));
  return {original, scaled, mix, low, high};
}

export const yarnAmplitude = (factor, coefficient = 0.1) => factor <= 1 ? 1 : 1 + coefficient * Math.log(factor);

export function softmax(logits, multiplier = 1) {
  if (!logits.length) return new Float64Array();
  if (!Number.isFinite(multiplier) || multiplier < 0) throw new RangeError('Softmax scale must be finite and nonnegative.');
  if (multiplier === 0) return new Float64Array(logits.length).fill(1 / logits.length);
  const maximum = Math.max(...logits);
  // Subtract before scaling to avoid overflow, including at large multipliers.
  const weights = Float64Array.from(logits, value => Math.exp((value - maximum) * multiplier));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return Float64Array.from(weights, value => value / total);
}

export function entropy(probabilities) {
  return -probabilities.reduce((sum, value) => value > 0 ? sum + value * Math.log2(value) : sum, 0);
}

export function rotate(vector, position, frequency) {
  const half = vector.length / 2;
  const result = new Float64Array(vector.length);
  for (let i = 0; i < half; i++) {
    const angle = position * frequency[i];
    const c = Math.cos(angle), s = Math.sin(angle);
    result[i] = vector[i] * c - vector[i + half] * s;
    result[i + half] = vector[i] * s + vector[i + half] * c;
  }
  return result;
}

export function pairContribution(q, k, i, delta, frequency) {
  const half = q.length / 2;
  const a = q[i] * k[i] + q[i + half] * k[i + half];
  const b = q[i] * k[i + half] - q[i + half] * k[i];
  return a * Math.cos(delta * frequency) + b * Math.sin(delta * frequency);
}

export function scoreAt(q, k, delta, frequency) {
  let result = 0;
  for (let i = 0; i < frequency.length; i++) result += pairContribution(q, k, i, delta, frequency[i]);
  return result;
}

export function decodeBF16(buffer) {
  const source = new DataView(buffer);
  const values = new Float32Array(buffer.byteLength / 2);
  const bits = new Uint32Array(values.buffer);
  for (let i = 0; i < values.length; i++) bits[i] = source.getUint16(i * 2, true) * 65536;
  return values;
}
