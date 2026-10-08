import js from '@eslint/js';
import { defineConfig } from 'eslint/config';
import prettier from 'eslint-config-prettier';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';
import svelteConfig from './svelte.config.js';

export default defineConfig(
  {
    ignores: [
      'node_modules/**',
      '.venv/**',
      '.locallery/**',
      'dist/**',
      'data/**',
      '.test-artifacts/**',
      '.benchmark-results/**',
      '.agents/**',
      '.codex/**',
      'test/**',
    ],
  },
  js.configs.recommended,
  ts.configs.recommended,
  svelte.configs.recommended,
  {
    files: ['src/frontend/**/*.{ts,svelte}'],
    languageOptions: { globals: globals.browser },
  },
  {
    files: [
      'scripts/**/*.ts',
      'tests/**/*.ts',
      '*.js',
      '*.ts',
    ],
    languageOptions: { globals: { ...globals.node, Bun: 'readonly' } },
  },
  {
    files: ['**/*.svelte'],
    languageOptions: {
      parserOptions: { parser: ts.parser, svelteConfig },
    },
  },
  prettier,
  svelte.configs.prettier,
  {
    rules: {
      curly: ['error', 'all'],
    },
  },
  {
    files: ['tests/**/*.ts'],
    // Fixtures inspect partial database rows and deliberately malformed JSON.
    rules: { '@typescript-eslint/no-explicit-any': 'off' },
  },
);
