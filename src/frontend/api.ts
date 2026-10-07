export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    '/api' + path,
    body === undefined
      ? undefined
      : {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || 'Request failed');
  }
  return result;
}
