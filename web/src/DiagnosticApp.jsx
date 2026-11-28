import { useEffect, useMemo, useRef, useState } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';
import { Button } from '@astryxdesign/core/Button';
import { Slider } from '@astryxdesign/core/Slider';
import { Selector } from '@astryxdesign/core/Selector';
import { Badge } from '@astryxdesign/core/Badge';
import FallbackViewer from './FallbackViewer.jsx';
import ActionPlots from './ActionPlots.jsx';
import TrajectoryVisibility from './TrajectoryVisibility.jsx';

const cohorts = [
  { value: 'all', label: 'All sampled paths', detail: 'The 500-window display sample' },
  { value: 'center', label: 'Near center', detail: 'Final lateral displacement within ±3 m' },
  { value: 'positive', label: 'Positive lateral', detail: 'Final lateral displacement above +3 m' },
  { value: 'negative', label: 'Negative lateral', detail: 'Final lateral displacement below −3 m' },
];
const asset = (name) => new URL(`${import.meta.env.BASE_URL}${name}`, document.baseURI).href;
const frameNanoseconds = 100_000_000;

let focusedRecordings;
async function loadFocusedRecording(key) {
  focusedRecordings ??= Promise.all(
    ['focused-recordings.json', 'focused-recordings.bin'].map(async (name) => {
      const response = await fetch(asset(name));
      if (!response.ok) throw Error('Focused recording unavailable');
      if (name.endsWith('.json')) return response.json();
      const decoded = response.body.pipeThrough(new DecompressionStream('gzip'));
      return new Response(decoded).arrayBuffer();
    }),
  ).catch((error) => {
    focusedRecordings = undefined;
    throw error;
  });
  const [index, buffer] = await focusedRecordings;
  const [offset, length] = index[key] ?? [];
  if (
    !Number.isInteger(offset) ||
    !Number.isInteger(length) ||
    length <= 0 ||
    offset < 0 ||
    offset + length > buffer.byteLength
  )
    throw Error('Invalid focused recording');
  return new Uint8Array(buffer, offset, length);
}

function inCohort(path, cohort) {
  const lateral = path[49][1];
  if (cohort === 'center') return Math.abs(lateral) <= 3;
  if (cohort === 'positive') return lateral > 3;
  if (cohort === 'negative') return lateral < -3;
  return true;
}
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
  const [mode, setMode] = useState(() => (hasWebGL2() ? 'rerun' : 'canvas')),
    [paths, setPaths] = useState(null);
  const [actions, setActions] = useState(null),
    [selectedSample, setSelectedSample] = useState(2);
  const [clusters, setClusters] = useState(null);
  const showRerun = mode === 'rerun';

  useEffect(() => {
    fetch(asset('cohorts.json'))
      .then((r) => {
        if (!r.ok) throw Error('Summary unavailable');
        return r.json();
      })
      .then(setSummary)
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (paths) return;
    fetch(asset('trajectories.bin'))
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
    fetch(asset('actions.bin'))
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

  const cohortIndices = useMemo(
    () => paths?.flatMap((path, i) => (inCohort(path, cohort) ? [i] : [])),
    [paths, cohort],
  );
  const sampleIndex = cohortIndices?.includes(selectedSample) ? selectedSample : cohortIndices?.[0];
  const clusterId = clusters?.ids[sampleIndex];
  const focusedPath = visibility === 'selected' ? sampleIndex : undefined;
  const focusedExemplar = visibility === 'exemplar' ? clusterId : undefined;
  const visibleIndices = useMemo(() => {
    if (visibility === 'all') return cohortIndices;
    if (visibility === 'selected') return focusedPath == null ? [] : [focusedPath];
    return cohortIndices?.filter(
      (i) => focusedExemplar != null && clusters.ids[i] === focusedExemplar,
    );
  }, [cohortIndices, visibility, focusedPath, focusedExemplar, clusters]);
  const recordingKey =
    focusedPath != null
      ? `path-${focusedPath}`
      : focusedExemplar != null
        ? `${cohort}-exemplar-${focusedExemplar}`
        : null;
  const canShowRerun = showRerun && !!visibleIndices?.length;
  const playbackReady = showRerun ? ready && canShowRerun : !!visibleIndices?.length;

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
    if (!canShowRerun) return;
    let disposed = false;
    const instance = new WebViewer();
    viewer.current = instance;
    recordingId.current = null;
    setReady(false);
    setPlaying(false);
    const remove = [
      instance.on('recording_open', (e) => {
        if (disposed) return;
        recordingId.current = e.recording_id;
        instance.set_active_timeline(e.recording_id, 'elapsed');
        instance.set_current_time(
          e.recording_id,
          'elapsed',
          Math.round(timeRef.current * frameNanoseconds),
        );
        instance.set_playing(e.recording_id, false);
        instance.override_panel_state('blueprint', 'hidden');
        setReady(true);
      }),
      instance.on('time_update', (e) => {
        if (disposed || instance.get_active_timeline(e.recording_id) !== 'elapsed') return;
        timeRef.current = Math.min(49, Math.max(0, e.time / frameNanoseconds));
        setTime(timeRef.current);
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
            /^\/?trajectories\/(paths|vehicles)$/.test(x.entity_path) &&
            (Number.isInteger(x.instance_id) || visibleIndices.length === 1),
        );
        if (item && visibleIndices[item.instance_id ?? 0] != null)
          setSelectedSample(visibleIndices[item.instance_id ?? 0]);
      }),
    ];
    const start = async () => {
      const data = recordingKey ? await loadFocusedRecording(recordingKey) : null;
      if (disposed) return;
      await instance.start(data ? null : asset(`${cohort}.rrd`), host.current, {
        width: '100%',
        height: '100%',
      });
      if (data && !disposed) {
        const channel = instance.open_channel();
        channel.send_rrd(data);
        channel.close();
      }
    };
    start().catch(() => {
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
  }, [canShowRerun, cohort, recordingKey, visibleIndices]);

  const seek = (value) => {
    timeRef.current = value;
    setTime(value);
    if (recordingId.current)
      viewer.current?.set_current_time(
        recordingId.current,
        'elapsed',
        Math.round(value * frameNanoseconds),
      );
  };
  const toggle = () => {
    if (showRerun) {
      if (!recordingId.current) return;
      viewer.current?.set_playing(recordingId.current, !playing);
    } else if (time >= 49) seek(0);
    setPlaying(!playing);
  };
  const active = cohorts.find((x) => x.value === cohort);
  const visiblePaths = useMemo(() => visibleIndices?.map((i) => paths[i]), [visibleIndices, paths]);
  const visibleLabels = useMemo(
    () => visibleIndices?.map((i) => clusters?.ids[i]),
    [visibleIndices, clusters],
  );
  const localIndex = visibleIndices?.indexOf(sampleIndex) ?? -1;
  const cohortIndex = cohortIndices?.indexOf(sampleIndex) ?? -1;
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
  const changeCohort = (value) => {
    if (value) {
      setCohort(value);
      if (paths?.[sampleIndex] && !inCohort(paths[sampleIndex], value))
        setSelectedSample(paths.findIndex((path) => inCohort(path, value)));
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
          <Badge label={playbackReady ? 'Recording ready' : 'Loading recording'} />
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
            <TrajectoryVisibility
              value={visibility}
              onChange={(value) => {
                setVisibility(value);
                setPlaying(false);
              }}
            />
            <p className="supporting">
              {active.detail}. This threshold is descriptive, not a maneuver label.
            </p>
            <Slider
              label="Inspect path"
              value={Math.max(1, cohortIndex + 1)}
              min={1}
              max={Math.max(1, cohortIndices?.length ?? 0)}
              step={1}
              onChange={(value) => setSelectedSample(cohortIndices[value - 1])}
              formatValue={(value) => `${value} / ${cohortIndices?.length ?? '—'}`}
              valueDisplay="text"
              isDisabled={!cohortIndices?.length}
            />
          </section>
          <section className="control-group">
            <p className="eyebrow">PLAYBACK</p>
            <Button
              label={playing ? 'Pause trajectories' : 'Play trajectories'}
              variant="primary"
              isDisabled={!playbackReady}
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
              isDisabled={!playbackReady}
            />
            <p className="supporting">
              Use this control{showRerun ? ' or Rerun’s timeline' : ''} to scrub through 50
              positions.
            </p>
          </section>
          <section className="control-group inspector">
            <p className="eyebrow">INSPECTOR</p>
            <p>
              Sample path {sampleIndex == null ? '—' : sampleIndex + 1} · {(time / 10).toFixed(1)} s
            </p>
            <p
              className="cluster-tag"
              style={{ '--tag-color': clusters?.info.palette[clusterId] ?? 'currentColor' }}
            >
              Exemplar {clusterId == null ? '—' : clusterId + 1} / {clusters?.info.count ?? '—'}
            </p>
            <p className="supporting">
              {clusterId != null
                ? `${clusters.heldout[sampleIndex] ? 'Held-out track · test assignment' : 'Training track · assigned to exemplar'} · feature distance ${clusters.distances[sampleIndex].toFixed(2)}`
                : 'Loading exemplar assignment…'}
            </p>
            <p className="supporting">{selection}</p>
            <p className="supporting">
              {showRerun
                ? 'Select a path in Rerun or use the path slider.'
                : 'Drag and zoom the canvas. Tap a point or use the path slider.'}
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
              time={Math.round(time)}
              selectedIndex={localIndex}
              labels={visibleLabels}
              palette={clusters?.info.palette}
              exemplarPath={visibility === 'selected' ? undefined : clusters?.exemplars[clusterId]}
              onSelect={(index, detail) => {
                setSelectedSample(visibleIndices[index]);
                setSelection(detail);
              }}
            />
          )}
          <ActionPlots
            trajectories={traces}
            selectedId={sampleIndex}
            selectedLabel={sampleIndex == null ? undefined : sampleIndex + 1}
            time={time}
            onSeek={seek}
          />
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
