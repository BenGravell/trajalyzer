import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import react from '@vitejs/plugin-react';

export default defineConfig({base: './', plugins: [react(), wasm()], build: {target: 'esnext'}});
