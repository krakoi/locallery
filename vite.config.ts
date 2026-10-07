import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: { outDir: 'dist' },
  server: {
    proxy: {
      '/api': {
        target: process.env.LOCALLERY_BACKEND || 'http://127.0.0.1:3000',
        changeOrigin: true,
      },
    },
  },
});
