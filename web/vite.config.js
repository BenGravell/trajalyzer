import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  base: './',
  plugins: [
    react(),
    wasm(),
    viteStaticCopy({
      targets: [
        {
          src: '../data/corpus/{manifest.json,source.json,index.csv,model-meta.json,exemplars.bin}',
          dest: 'corpus',
          rename: { stripBase: true },
        },
        {
          src: '../data/corpus/shards/*.bin',
          dest: 'corpus/shards',
          rename: { stripBase: true },
        },
        {
          src: [
            '../data/{actions,clusters,distances,exemplars,focused-recordings,heldout,trajectories}.bin',
            '../data/{clusters,cohorts,focused-recordings}.json',
            '../data/{all,center,negative,positive}.rrd',
          ],
          dest: '.',
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  build: { target: 'esnext' },
});
