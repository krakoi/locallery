import type { Config } from './config';

export async function discoverEmbeddingServer(config: Config) {
  const get = async (url: string) => {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(config.embedding.timeout_seconds * 1000),
    });
    if (!response.ok) {
      throw new Error(`llama-server metadata returned ${response.status}`);
    }
    return response.json();
  };
  const [models, props] = await Promise.all([
    get(`${config.embedding.base_url}/models`),
    get(`${config.embedding.base_url.slice(0, -3)}/props`),
  ]);
  if (
    !Array.isArray(models?.data) ||
    models.data.length !== 1 ||
    typeof models.data[0]?.id !== 'string'
  ) {
    throw new Error(
      'Expected a single loaded embedding model from llama-server /v1/models',
    );
  }
  if (typeof props?.model_path !== 'string' || !props.model_path) {
    throw new Error('llama-server /props did not report model_path');
  }
  if (!Number.isInteger(props.total_slots) || props.total_slots < 1) {
    throw new Error('llama-server /props did not report a valid total_slots');
  }
  config.embedding.model = models.data[0].id;
  config.embedding.revision = JSON.stringify([
    props.model_path,
    models.data[0].meta ?? null,
  ]);
  config.embedding.concurrency = props.total_slots;
  console.log(
    `Embedding model: ${config.embedding.model} · ${props.total_slots} server slots`,
  );
}

export function normalize(values: ArrayLike<number>, dimensions = 768) {
  if (values.length !== dimensions) {
    throw new Error(
      `Expected ${dimensions} embedding dimensions; received ${values.length}`,
    );
  }

  const result = new Float32Array(dimensions);
  let norm = 0;

  for (let n = 0; n < dimensions; n++) {
    const v = values[n];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new Error('Embedding contains invalid values');
    }
    result[n] = v;
    norm += v * v;
  }

  if (!Number.isFinite(norm) || norm <= 0) {
    throw new Error('Embedding has zero or invalid norm');
  }

  const scale = 1 / Math.sqrt(norm);

  for (let n = 0; n < dimensions; n++) {
    result[n] *= scale;
  }
  return result;
}

export async function embed(config: Config, query?: string, cache?: string) {
  const content: Record<string, unknown>[] = [];

  if (query?.trim()) {
    content.push({
      type: 'text',
      text: `task: search result | query: ${query.trim()}`,
    });
  }

  if (cache) {
    content.push({
      type: 'image_url',
      image_url: {
        url: `data:image/jpeg;base64,${Buffer.from(await Bun.file(cache).arrayBuffer()).toString('base64')}`,
      },
    });
  }

  if (!content.length) {
    throw new Error('A query or image is required');
  }

  const response = await fetch(`${config.embedding.base_url}/embeddings`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: config.embedding.model,
      encoding_format: 'float',
      input: [{ content }],
    }),
    signal: AbortSignal.timeout(config.embedding.timeout_seconds * 1000),
  });
  if (!response.ok) {
    throw new Error(
      `llama-server returned ${response.status}: ${(await response.text()).slice(0, 300)}`,
    );
  }

  const result = (await response.json()) as {
    data?: { embedding?: unknown }[];
  } | null;

  const values = result?.data?.[0]?.embedding;
  if (!Array.isArray(values)) {
    throw new Error('llama-server did not return a float embedding');
  }
  return normalize(values);
}
