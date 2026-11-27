import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { Slider } from '@astryxdesign/core/Slider';
import { Selector } from '@astryxdesign/core/Selector';
import FallbackViewer from './FallbackViewer.jsx';
import ActionPlots from './ActionPlots.jsx';

const base = new URL(
  import.meta.env.VITE_CORPUS_BASE_URL || `${import.meta.env.BASE_URL}corpus/`,
  document.baseURI,
);
const groups = [
  { value: 'all', label: 'All motion' },
  { value: 'center', label: 'Near center' },
  { value: 'positive', label: 'Positive lateral' },
  { value: 'negative', label: 'Negative lateral' },
];
const MAX_SHARDS = 6;
const MAX_VISIBLE = 768;

function selectShards(manifest, cohort, cluster, page) {
  const eligible = manifest.shards.filter(
    (s) =>
      (cohort === 'all' || s.cohort === cohort) &&
      (cluster === 'all' || s.cluster === Number(cluster)),
  );
  // Spread requests over the corpus, preserving diversity across partitions.
  const pages = Math.max(1, Math.ceil(eligible.length / MAX_SHARDS));
  const chosen = Array.from({ length: MAX_SHARDS }, (_, i) => eligible[i * pages + page]).filter(
    Boolean,
  );
  return { count: eligible.reduce((sum, s) => sum + s.count, 0), chosen, pages };
}

async function fetchPaths(shards, signal) {
  const results = new Array(shards.length);
  let cursor = 0;
  await Promise.all(
    Array.from({ length: Math.min(6, shards.length) }, async () => {
      while (cursor < shards.length) {
        const slot = cursor++;
        const shard = shards[slot];
        const response = await fetch(new URL(shard.path, base), { signal });
        if (!response.ok) throw Error(`Shard unavailable: ${shard.path}`);
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== shard.count * 208) throw Error(`Invalid shard: ${shard.path}`);
        const view = new DataView(buffer);
        results[slot] = Array.from({ length: shard.count }, (_, index) => {
          const offset = index * 208;
          const path = Array.from({ length: 50 }, (_, t) => [
            view.getInt16(offset + 4 * t, true) / 100,
            view.getInt16(offset + 4 * t + 2, true) / 100,
          ]);
          return {
            path,
            id: view.getUint32(offset + 200, true),
            distance: view.getFloat32(offset + 204, true),
            cluster: shard.cluster,
            shard,
            index,
          };
        });
      }
    }),
  );
  const rows = results.flat();
  if (rows.length <= MAX_VISIBLE) return rows;
  return Array.from(
    { length: MAX_VISIBLE },
    (_, i) => rows[Math.floor(((i + 0.5) * rows.length) / MAX_VISIBLE)],
  );
}

export default function CorpusExplorer({ onDiagnostics }) {
  const [manifest, setManifest] = useState(null);
  const [cohort, setCohort] = useState('all');
  const [cluster, setCluster] = useState('all');
  const [page, setPage] = useState(0);
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(0);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [series, setSeries] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const actionCache = useRef(new Map());

  useEffect(() => {
    const controller = new AbortController();
    fetch(new URL('manifest.json', base), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw Error('Corpus manifest unavailable');
        return r.json();
      })
      .then((data) => {
        if (data.version !== 1 || data.frames !== 50) throw Error('Unsupported corpus format');
        setManifest(data);
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e.message);
      });
    return () => controller.abort();
  }, []);
  const selection = useMemo(
    () => manifest && selectShards(manifest, cohort, cluster, page),
    [manifest, cohort, cluster, page],
  );
  useEffect(() => {
    if (!selection) return;
    const controller = new AbortController();
    setLoading(true);
    setRows([]);
    setSelected(0);
    fetchPaths(selection.chosen, controller.signal)
      .then(setRows)
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selection]);
  const chosen = rows[selected];
  useEffect(() => {
    if (!chosen) {
      setSeries(null);
      return;
    }
    const key = chosen.shard.path;
    const cached = actionCache.current.get(key);
    const controller = new AbortController();
    const load = cached
      ? Promise.resolve(cached)
      : fetch(new URL(key.replace(/\.bin$/, '.actions.bin'), base), { signal: controller.signal })
          .then((r) => {
            if (!r.ok) throw Error('Action shard unavailable');
            return r.arrayBuffer();
          })
          .then((buffer) => {
            if (buffer.byteLength !== chosen.shard.count * 400) throw Error('Invalid action shard');
            actionCache.current.set(key, buffer);
            if (actionCache.current.size > 8)
              actionCache.current.delete(actionCache.current.keys().next().value);
            return buffer;
          });
    setSeries(null);
    load
      .then((buffer) => {
        if (controller.signal.aborted) return;
        const view = new DataView(buffer);
        setSeries(
          Array.from({ length: 50 }, (_, t) => [
            view.getFloat32(chosen.index * 400 + t * 8, true),
            view.getFloat32(chosen.index * 400 + t * 8 + 4, true),
          ]),
        );
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e.message);
      });
    return () => controller.abort();
  }, [chosen]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setTime((t) => {
          if (t >= 49) {
            setPlaying(false);
            return 49;
          }
          return t + 1;
        }),
      100,
    );
    return () => clearInterval(timer);
  }, [playing]);
  const clusters = useMemo(
    () =>
      manifest ? [...new Set(manifest.shards.map((s) => s.cluster))].sort((a, b) => a - b) : [],
    [manifest],
  );
  const palette = useMemo(
    () =>
      Array.from(
        { length: Math.max(1, ...clusters.map((c) => c + 1)) },
        (_, i) => `hsl(${Math.round((i * 137.508) % 360)} 70% 65%)`,
      ),
    [clusters],
  );
  const paths = useMemo(() => rows.map((r) => r.path), [rows]);
  const labels = useMemo(() => rows.map((r) => r.cluster), [rows]);
  return (
    <article className="app">
      <header className="masthead">
        <section>
          <p className="eyebrow">TRAJECTORY MANIFOLD / CORPUS</p>
          <h1>Trajalyzer</h1>
          <p className="subhead">Indexed motion, progressively loaded</p>
        </section>
        <section className="header-actions">
          <Button label="Rerun diagnostics" onClick={onDiagnostics} />
        </section>
      </header>
      <section className="workspace">
        <aside className="sidebar" aria-label="Corpus controls">
          <section className="intro">
            <p className="eyebrow">CORPUS</p>
            <h2>{manifest?.count.toLocaleString() ?? '—'} windows</h2>
            <p>
              Browse the indexed collection. Only a bounded, diverse working set enters the browser.
            </p>
          </section>
          <section className="metric-row">
            <article>
              <strong>{selection?.count.toLocaleString() ?? '—'}</strong>
              <small>matching windows</small>
            </article>
            <article>
              <strong>{rows.length}</strong>
              <small>loaded for display</small>
            </article>
          </section>
          <section className="control-group">
            <p className="eyebrow">FILTER</p>
            <Selector
              label="Lateral group"
              value={cohort}
              options={groups}
              onChange={(value) => {
                setPage(0);
                setCohort(value || 'all');
              }}
              presentation="adaptive"
            />
            <Selector
              label="Exemplar"
              value={cluster}
              options={[
                { value: 'all', label: 'All exemplars' },
                ...clusters.map((c) => ({ value: String(c), label: `Exemplar ${c + 1}` })),
              ]}
              onChange={(value) => {
                setPage(0);
                setCluster(value || 'all');
              }}
              presentation="adaptive"
            />
            <Slider
              label="Corpus page"
              value={page + 1}
              min={1}
              max={selection?.pages ?? 1}
              step={1}
              onChange={(value) => setPage(value - 1)}
              formatValue={(value) => `${value} / ${selection?.pages ?? 1}`}
              valueDisplay="text"
              isDisabled={!selection || selection.pages < 2}
            />
            <Slider
              label="Inspect loaded path"
              value={selected + 1}
              min={1}
              max={Math.max(1, rows.length)}
              step={1}
              onChange={(v) => setSelected(v - 1)}
              formatValue={(v) => `${v} / ${rows.length}`}
              valueDisplay="text"
              isDisabled={!rows.length}
            />
          </section>
          <section className="control-group">
            <p className="eyebrow">PLAYBACK</p>
            <Button
              label={playing ? 'Pause' : 'Play'}
              onClick={() => {
                if (time === 49) setTime(0);
                setPlaying(!playing);
              }}
              isDisabled={!rows.length}
            />
            <Slider
              label="Elapsed time"
              value={time}
              min={0}
              max={49}
              step={1}
              onChange={setTime}
              formatValue={(v) => `${(v / 10).toFixed(1)} s`}
              valueDisplay="text"
            />
          </section>
          <section className="control-group inspector">
            <p className="eyebrow">INSPECTOR</p>
            <p>
              Trajectory #{chosen?.id ?? '—'} · exemplar {chosen ? chosen.cluster + 1 : '—'}
            </p>
            <p className="supporting">
              {chosen
                ? `Feature distance ${chosen.distance.toFixed(2)}`
                : 'Choose a displayed path.'}
            </p>
            <p className="supporting">
              {loading
                ? 'Loading indexed shards…'
                : 'Drag or zoom to explore; tap a point to inspect its actions.'}
            </p>
          </section>
        </aside>
        <section className="viewer-pane" aria-label="Corpus visualization">
          <header className="viewer-heading">
            <strong>Corpus overview</strong>
            <small>
              {rows.length} displayed / {selection?.count.toLocaleString() ?? '—'} matching · 0–4.9
              s
            </small>
          </header>
          <FallbackViewer
            paths={paths}
            time={time}
            selectedIndex={selected}
            labels={labels}
            palette={palette}
            onSelect={(index) => setSelected(index)}
          />
          <ActionPlots series={series} time={time} onSeek={setTime} sampleIndex={chosen?.id ?? 0} />
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>
      </section>
      <footer>
        Static indexed shards can be hosted on a CDN; the GitHub Pages app fetches only the active
        working set.
      </footer>
    </article>
  );
}
