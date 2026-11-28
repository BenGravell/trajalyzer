import { useCallback, useEffect, useState } from 'react';
import { Slider } from '@astryxdesign/core/Slider';
import { Selector } from '@astryxdesign/core/Selector';
import RerunViewer from './RerunViewer.jsx';
import TrajectoryVisibility from './TrajectoryVisibility.jsx';

const corpusBase = new URL(
  import.meta.env.VITE_CORPUS_BASE_URL || `${import.meta.env.BASE_URL}corpus/`,
  document.baseURI,
);
const base = new URL('recordings/', corpusBase);

async function fetchJson(name, signal) {
  const response = await fetch(new URL(name, base), { signal });
  if (!response.ok) throw Error(`Recording index unavailable (${response.status})`);
  return response.json();
}

export default function CorpusExplorer() {
  const [info, setInfo] = useState(null);
  const [group, setGroup] = useState('all');
  const [page, setPage] = useState(0);
  const [visibility, setVisibility] = useState('all');
  const [selected, setSelected] = useState(0);
  const [loaded, setLoaded] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    fetchJson('index.json', signal)
      .then(setInfo)
      .catch((reason) => {
        if (!signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, []);

  const selection = info?.groups[group];
  const pageAsset = selection?.pages[page];
  const recording = loaded?.pageAsset === pageAsset ? loaded?.recording : null;

  useEffect(() => {
    if (!pageAsset) return;
    const controller = new AbortController();
    const { signal } = controller;
    setError('');
    fetchJson(pageAsset, signal)
      .then((metadata) => {
        if (signal.aborted) return;
        setSelected(0);
        setLoaded({
          pageAsset,
          recording: {
            url: new URL(metadata.recording, base).href,
            rows: metadata.rows.map(([id, cluster, distance]) => ({ id, cluster, distance })),
          },
        });
      })
      .catch((reason) => {
        if (!signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
  }, [pageAsset]);

  const rows = recording?.rows;
  const chosen = rows?.[selected];
  const onSelect = useCallback(
    (id) => {
      const index = rows?.findIndex((row) => row.id === id) ?? -1;
      if (index >= 0) setSelected(index);
    },
    [rows],
  );
  const visibleCount =
    rows?.filter((row) => {
      if (visibility === 'selected') return row === chosen;
      if (visibility === 'exemplar') return row.cluster === chosen?.cluster;
      return true;
    }).length ?? 0;
  const duration = info ? (info.frames - 1) / info.sampleRate : 0;
  const source = info?.source;
  const groups = info ? Object.keys(info.groups).filter((key) => key !== 'all') : [];

  return (
    <article className="app">
      <header className="masthead">
        <section>
          <p className="eyebrow">TRAJECTORY MANIFOLD / CORPUS</p>
          <h1>Trajalyzer</h1>
          <p className="subhead">{info?.dataset ?? 'Loading corpus…'}</p>
        </section>
      </header>
      <section className="workspace">
        <aside className="sidebar" aria-label="Corpus controls">
          <section className="intro">
            <p className="eyebrow">CORPUS</p>
            <h2>{info?.count.toLocaleString() ?? '—'} windows</h2>
            {info && (
              <p>
                {source?.track_scope === 'av' ? 'AV ego' : 'Vehicle'} tracks · {info.frames} samples
                at {info.sampleRate} Hz
              </p>
            )}
          </section>
          <section className="metric-row">
            <article>
              <strong>{selection?.count.toLocaleString() ?? '—'}</strong>
              <small>matching windows</small>
            </article>
            <article>
              <strong>{rows?.length ?? '—'}</strong>
              <small>paths in this recording</small>
            </article>
          </section>
          <section className="control-group">
            <p className="eyebrow">FILTER</p>
            {groups.length > 0 && (
              <Selector
                label="Corpus exemplar"
                options={[
                  { value: 'all', label: 'All exemplars' },
                  ...groups.map((key) => ({ value: key, label: `Exemplar ${Number(key) + 1}` })),
                ]}
                onChange={(value) => {
                  setPage(0);
                  setGroup(value || 'all');
                }}
                presentation="adaptive"
                value={group}
              />
            )}
            <TrajectoryVisibility value={visibility} onChange={setVisibility} />
            {selection?.pages.length > 1 && (
              <Slider
                label="Corpus page"
                value={page + 1}
                min={1}
                max={selection.pages.length}
                step={1}
                onChange={(value) => setPage(value - 1)}
                formatValue={(value) => `${value} / ${selection.pages.length}`}
                valueDisplay="text"
              />
            )}
            <Slider
              label="Inspect loaded path"
              value={selected + 1}
              min={1}
              max={Math.max(1, rows?.length ?? 0)}
              step={1}
              onChange={(value) => setSelected(value - 1)}
              formatValue={(value) => `${value} / ${rows?.length ?? 0}`}
              valueDisplay="text"
              isDisabled={!rows || rows.length < 2}
            />
          </section>
          <section className="control-group inspector">
            <p className="eyebrow">INSPECTOR</p>
            {chosen && (
              <>
                <p>
                  Trajectory #{chosen.id} · exemplar {chosen.cluster + 1}
                </p>
                <p className="supporting">Feature distance {chosen.distance.toFixed(2)}</p>
              </>
            )}
            <p className="supporting">
              Select a path, vehicle marker, or action trace to inspect its trajectory.
            </p>
          </section>
        </aside>
        <section className="viewer-pane" aria-label="Corpus visualization">
          <header className="viewer-heading">
            <strong>Corpus overview</strong>
            <small>
              {visibleCount} visible · 0–{duration.toFixed(1)} s
            </small>
          </header>
          <RerunViewer
            url={recording?.url}
            endNanos={Math.round(duration * 1e9)}
            sampleRate={info?.sampleRate}
            visibility={visibility}
            chosen={chosen}
            rows={rows}
            onSelect={onSelect}
          />
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>
      </section>
      {source?.dataset === 'Argoverse 2 Motion Forecasting' && (
        <footer>
          AV2 data © 2021 Argo AI, LLC ·{' '}
          <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> ·{' '}
          {!source.complete_split ? 'Preview · ' : 'Complete training split · '}
          {source.scenarios} source scenarios · {info.trainingWindows ?? '—'} windows used for
          exemplar fitting.
        </footer>
      )}
    </article>
  );
}
