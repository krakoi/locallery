import { test, expect } from 'bun:test';
import { remember, readHistory, clearHistory } from '../src/frontend/history';

test('browser history deduplicates query, reference and scope and keeps 50 entries', () => {
  const values = new Map<string, string>();
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k: string) => values.get(k) || null,
      setItem: (k: string, v: string) => values.set(k, v),
      removeItem: (k: string) => values.delete(k),
    },
  });
  try {
    let history = remember([], { query: 'cats' });
    history = remember(history, { query: 'cats' });
    expect(history.length).toBe(1);
    history = remember(history, { query: 'cats', folderId: 'different' });
    expect(history.length).toBe(2);
    for (let i = 0; i < 55; i++) {
      history = remember(history, { query: `query-${i}` });
    }
    expect(history.length).toBe(50);
    expect(readHistory()[0].query).toBe('query-54');
    clearHistory();
    expect(readHistory()).toEqual([]);
  } finally {
    if (previous) {
      Object.defineProperty(globalThis, 'localStorage', previous);
    } else {
      Reflect.deleteProperty(globalThis, 'localStorage');
    }
  }
});
