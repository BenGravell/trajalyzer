import { defineConfig, loadEnv } from 'vite';
import wasm from 'vite-plugin-wasm';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  return {
    base: './',
    plugins: [react(), wasm()],
    optimizeDeps: { exclude: ['@rerun-io/web-viewer'] },
    publicDir: env.VITE_CORPUS_BASE_URL ? false : 'public',
    build: { target: 'esnext' },
  };
});
