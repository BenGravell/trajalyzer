import { useEffect, useMemo, useRef, useState } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';
import { Button } from '@astryxdesign/core/Button';
import { Slider } from '@astryxdesign/core/Slider';
import { Selector } from '@astryxdesign/core/Selector';
import { Badge } from '@astryxdesign/core/Badge';
import FallbackViewer from './FallbackViewer.jsx';
import ActionPlots from './ActionPlots.jsx';

const cohorts = [
  { value: 'all', label: 'All sampled paths', detail: 'The 500-window display sample' },
  { value: 'center', label: 'Near center', detail: 'Final lateral displacement within ±3 m' },
  { value: 'positive', label: 'Positive lateral', detail: 'Final lateral displacement above +3 m' },
  { value: 'negative', label: 'Negative lateral', detail: 'Final lateral displacement below −3 m' },
];
const visibilityOptions = [
  { value: 'all', label: 'All paths in group' },
  { value: 'exemplar', label: 'Selected exemplar group' },
  { value: 'selected', label: 'Selected path only' },
];
const url = (name) => new URL(`${import.meta.env.BASE_URL}${name}.rrd`, document.baseURI).href;
const hasWebGL2 = () => {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
};

function DiagnosticApp({ onCorpus }) {
  const host = useRef(null),
    viewer = useRef(null),
    recordingId = useRef(null),
    timeRef = useRef(0);
  const [cohort, setCohort] = useState('all'),
    [summary, setSummary] = useState(null);
  const [visibility, setVisibility] = useState('all');
  const [ready, setReady] = useState(false),
    [error, setError] = useState('');
  const [playing, setPlaying] = useState(false),
    [time, setTime] = useState(0);
  const [selection, setSelection] = useState('Tap a path or position to inspect it.');
  const [focus, setFocus] = useState(false);
  const [mode, setMode] = useState(hasWebGL2() ? 'rerun' : 'canvas'),
    [paths, setPaths] = useState(null);
  const [actions, setActions] = useState(null),
    [selectedSample, setSelectedSample] = useState(2);
  const [clusters, setClusters] = useState(null);
  const showRerun = mode === 'rerun' && visibility === 'all';

  useEffect(() => {
    timeRef.current = time;
  }, [time]);

  useEffect(() => {
    fetch(url('cohorts').replace('.rrd', '.json'))
      .then((r) => {
        if (!r.ok) throw Error('Summary unavailable');
        return r.json();
      })
      .then(setSummary)
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (paths) return;
    fetch(new URL(`${import.meta.env.BASE_URL}trajectories.bin`, document.baseURI))
      .then((r) => {
        if (!r.ok) throw Error('Trajectory data unavailable');
        return r.arrayBuffer();
      })
      .then((buffer) => {
        const data = new Int16Array(buffer);
        if (data.length % 100) throw Error('Invalid trajectory sample');
        const count = data.length / 100;
        setPaths(
          Array.from({ length: count }, (_, i) =>
            Array.from({ length: 50 }, (_, t) => [
              data[i * 100 + t * 2] / 100,
              data[i * 100 + t * 2 + 1] / 100,
            ]),
          ),
        );
      })
      .catch((e) => setError(e.message));
  }, [paths]);

  useEffect(() => {
    fetch(new URL(`${import.meta.env.BASE_URL}actions.bin`, document.baseURI))
      .then((r) => {
        if (!r.ok) throw Error('Inferred actions unavailable');
        return r.arrayBuffer();
      })
      .then((buffer) => {
        const data = new Float32Array(buffer);
        if (data.length % 100) throw Error('Invalid action sample');
        setActions(
          Array.from({ length: data.length / 100 }, (_, i) =>
            Array.from({ length: 50 }, (_, t) => [
              data[i * 100 + t * 2],
              data[i * 100 + t * 2 + 1],
            ]),
          ),
        );
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const asset = (name) => new URL(`${import.meta.env.BASE_URL}${name}`, document.baseURI);
    Promise.all(
      ['clusters.json', 'clusters.bin', 'distances.bin', 'heldout.bin', 'exemplars.bin'].map(
        (name) =>
          fetch(asset(name)).then((r) => {
            if (!r.ok) throw Error(`Missing ${name}`);
            return name.endsWith('.json') ? r.json() : r.arrayBuffer();
          }),
      ),
    )
      .then(([info, idsBuffer, distanceBuffer, heldoutBuffer, exemplarBuffer]) => {
        const raw = new Int16Array(exemplarBuffer);
        if (raw.length !== info.count * 100) throw Error('Invalid exemplar sample');
        setClusters({
          info,
          ids: new Uint8Array(idsBuffer),
          distances: new Float32Array(distanceBuffer),
          heldout: new Uint8Array(heldoutBuffer),
          exemplars: Array.from({ length: info.count }, (_, i) =>
            Array.from({ length: 50 }, (_, t) => [
              raw[i * 100 + t * 2] / 100,
              raw[i * 100 + t * 2 + 1] / 100,
            ]),
          ),
        });
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!showRerun) setReady(!!paths);
  }, [showRerun, paths, cohort]);

  useEffect(() => {
    if (showRerun || !playing) return;
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
  }, [showRerun, playing]);

  useEffect(() => {
    if (!showRerun || !paths) return;
    let disposed = false;
    const instance = new WebViewer();
    viewer.current = instance;
    recordingId.current = null;
    setReady(false);
    setError('');
    setPlaying(false);
    setSelection('Tap a path or position to inspect it.');
    const recordingIndices = paths
      ?.map((path, i) => ({ path, i }))
      .filter(
        ({ path }) =>
          cohort === 'all' ||
          (cohort === 'center'
            ? Math.abs(path[49][1]) <= 3
            : cohort === 'positive'
              ? path[49][1] > 3
              : path[49][1] < -3),
      )
      .map(({ i }) => i);
    const remove = [
      instance.on('recording_open', (e) => {
        if (disposed) return;
        recordingId.current = e.recording_id;
        instance.set_active_timeline(e.recording_id, 'elapsed');
        instance.set_current_time(
          e.recording_id,
          'elapsed',
          Math.round(timeRef.current * 100_000_000),
        );
        instance.override_panel_state('blueprint', 'hidden');
        setReady(true);
      }),
      instance.on('time_update', (e) => {
        if (!disposed) setTime(Math.min(49, Math.max(0, e.time / 100_000_000)));
      }),
      instance.on('play', () => {
        if (!disposed) setPlaying(true);
      }),
      instance.on('pause', () => {
        if (!disposed) setPlaying(false);
      }),
      instance.on('selection_change', (e) => {
        if (disposed) return;
        const item = e.items.find(
          (x) =>
            x.type === 'entity' &&
            Number.isInteger(x.instance_id) &&
            x.entity_path?.includes('trajectories/'),
        );
        if (item && recordingIndices?.[item.instance_id] != null)
          setSelectedSample(recordingIndices[item.instance_id]);
      }),
    ];
    instance.start(url(cohort), host.current, null).catch(() => {
      if (!disposed) {
        instance.stop();
        setMode('canvas');
      }
    });
    return () => {
      disposed = true;
      remove.forEach((fn) => fn());
      instance.stop();
      recordingId.current = null;
      if (viewer.current === instance) viewer.current = null;
    };
  }, [cohort, showRerun, paths]);

  const seek = (value) => {
    setTime(value);
    if (recordingId.current)
      viewer.current?.set_current_time(
        recordingId.current,
        'elapsed',
        Math.round(value * 100_000_000),
      );
  };
  const toggle = () => {
    if (showRerun) {
      if (!recordingId.current) return;
      viewer.current?.set_playing(recordingId.current, !playing);
    } else if (time >= 49) setTime(0);
    setPlaying(!playing);
  };
  const active = cohorts.find((x) => x.value === cohort);
  const cohortIndices = useMemo(
    () =>
      paths
        ?.map((path, i) => ({ path, i }))
        .filter(
          ({ path }) =>
            cohort === 'all' ||
            (cohort === 'center'
              ? Math.abs(path[49][1]) <= 3
              : cohort === 'positive'
                ? path[49][1] > 3
                : path[49][1] < -3),
        )
        .map(({ i }) => i),
    [paths, cohort],
  );
  const selectedCluster = clusters?.ids[selectedSample];
  const visibleIndices = useMemo(
    () =>
      cohortIndices?.filter(
        (i) =>
          visibility === 'all' ||
          (visibility === 'selected' ? i === selectedSample : clusters?.ids[i] === selectedCluster),
      ),
    [cohortIndices, visibility, selectedSample, selectedCluster, clusters],
  );
  const visiblePaths = useMemo(() => visibleIndices?.map((i) => paths[i]), [visibleIndices, paths]);
  const visibleLabels = useMemo(
    () => visibleIndices?.map((i) => clusters?.ids[i]),
    [visibleIndices, clusters],
  );
  const localIndex = Math.max(0, visibleIndices?.indexOf(selectedSample) ?? 0);
  const sampleIndex = selectedSample;
  const traces = useMemo(
    () =>
      visibleIndices?.flatMap((i) =>
        actions?.[i]
          ? [
              {
                id: i,
                values: actions[i],
                color: clusters?.info.palette[clusters.ids[i]] ?? '#84baff',
              },
            ]
          : [],
      ) ?? [],
    [visibleIndices, actions, clusters],
  );
  const clusterId = clusters?.ids[sampleIndex];
  const changeCohort = (value) => {
    if (value) {
      setCohort(value);
      const first = paths?.findIndex(
        (path) =>
          value === 'all' ||
          (value === 'center'
            ? Math.abs(path[49][1]) <= 3
            : value === 'positive'
              ? path[49][1] > 3
              : path[49][1] < -3),
      );
      if (first >= 0) setSelectedSample(first);
      setSelection('Choose a path or tap one in the view.');
    }
  };
  return (
    <article className={`app ${focus ? 'focus' : ''}`}>
      <header className="masthead">
        <section>
          <p className="eyebrow">TRAJECTORY MANIFOLD / EXPLORE</p>
          <h1>Trajalyzer</h1>
          <p className="subhead">Real vehicle motion in a shared ego-relative frame</p>
        </section>
        <section className="header-actions">
          <Badge label={ready ? 'Recording ready' : 'Loading recording'} />
          <Button label="Explore corpus" onClick={onCorpus} />
          <Button label={focus ? 'Show controls' : 'Focus view'} onClick={() => setFocus(!focus)} />
        </section>
      </header>
      <section className="workspace">
        <aside className="sidebar" aria-label="Trajectory controls">
          <section className="intro">
            <p className="eyebrow">DATASET</p>
            <h2>INTERACTION</h2>
            <p>
              Traffic-camera intersection · five-second windows sampled at 10 Hz.{' '}
              {summary?.eligible_windows?.toLocaleString() ?? '—'} windows sustain at least 1 m/s;
              500 shown.
            </p>
          </section>
          <section className="metric-row">
            <article>
              <strong>{summary?.source_windows?.toLocaleString() ?? '—'}</strong>
              <small>valid windows loaded</small>
            </article>
            <article>
              <strong>{visibleIndices?.length ?? '—'}</strong>
              <small>paths in view</small>
            </article>
          </section>
          <section className="control-group">
            <p className="eyebrow">EXPLORE</p>
            <Selector
              label="Trajectory group"
              value={cohort}
              options={cohorts}
              onChange={changeCohort}
              presentation="adaptive"
            />
            <Selector
              label="Show trajectories"
              value={visibility}
              options={visibilityOptions}
              onChange={(value) => {
                setVisibility(value || 'all');
                setPlaying(false);
              }}
              presentation="adaptive"
            />
            <p className="supporting">
              {active.detail}. This threshold is descriptive, not a maneuver label.
            </p>
            <Slider
              label="Inspect path"
              value={localIndex + 1}
              min={1}
              max={visibleIndices?.length ?? 1}
              step={1}
              onChange={(value) => setSelectedSample(visibleIndices[value - 1])}
              formatValue={(value) => `${value} / ${visibleIndices?.length ?? '—'}`}
              valueDisplay="text"
              isDisabled={!visibleIndices?.length}
            />
          </section>
          <section className="control-group">
            <p className="eyebrow">PLAYBACK</p>
            <Button
              label={playing ? 'Pause trajectories' : 'Play trajectories'}
              variant="primary"
              isDisabled={!ready}
              onClick={toggle}
            />
            <Slider
              label="Elapsed time"
              value={time}
              min={0}
              max={49}
              step={1}
              onChange={seek}
              formatValue={(value) => `${(value / 10).toFixed(1)} s`}
              valueDisplay="text"
              isDisabled={!ready}
            />
            <p className="supporting">
              Use this control{showRerun ? ' or Rerun’s timeline' : ''} to scrub through 50
              positions.
            </p>
          </section>
          <section className="control-group inspector">
            <p className="eyebrow">INSPECTOR</p>
            <p>
              Sample path {sampleIndex + 1} · {(time / 10).toFixed(1)} s
            </p>
            <p
              className="cluster-tag"
              style={{ '--tag-color': clusters?.info.palette[clusterId] ?? 'currentColor' }}
            >
              Exemplar {clusterId == null ? '—' : clusterId + 1} / {clusters?.info.count ?? '—'}
            </p>
            <p className="supporting">
              {clusters
                ? `${clusters.heldout[sampleIndex] ? 'Held-out track · test assignment' : 'Training track · assigned to exemplar'} · feature distance ${clusters.distances[sampleIndex].toFixed(2)}`
                : 'Loading exemplar assignment…'}
            </p>
            <p className="supporting">{selection}</p>
            <p className="supporting">
              {showRerun
                ? 'Select a path in Rerun or use the path slider.'
                : 'Drag and zoom the canvas. Tap a point or use the path slider. Focused visibility uses the canvas view.'}
            </p>
          </section>
        </aside>
        <section className="viewer-pane" aria-label="Trajectory visualization">
          <header className="viewer-heading">
            <strong>{active.label}</strong>
            <small>{visibleIndices?.length ?? '—'} visible paths · metres · 0–4.9 seconds</small>
          </header>
          {showRerun ? (
            <section className="viewer-host" ref={host} />
          ) : (
            <FallbackViewer
              paths={visiblePaths}
              time={time}
              selectedIndex={localIndex}
              labels={visibleLabels}
              palette={clusters?.info.palette}
              exemplarPath={clusters?.exemplars[clusterId]}
              onSelect={(index, detail) => {
                setSelectedSample(visibleIndices[index]);
                setSelection(detail);
              }}
            />
          )}
          <ActionPlots trajectories={traces} selectedId={sampleIndex} time={time} onSeek={seek} />
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>
      </section>
      <footer>
        Source:{' '}
        <a href="https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset">
          INTERACTION dataset
        </a>{' '}
        · CC BY 4.0. One intersection subset; overlapping windows may share tracks.{' '}
        <a href="https://github.com/BenGravell/trajalyzer">Repository</a>.
      </footer>
    </article>
  );
}

export default DiagnosticApp;
