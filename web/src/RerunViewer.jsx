import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { Slider } from '@astryxdesign/core/Slider';
import { WebViewer } from '@rerun-io/web-viewer';

const elapsed = 'elapsed';
const metrics = new Set(['/acceleration', '/curvature']);

async function loadRecording(viewer, url, endNanos, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw Error(`Recording unavailable (${response.status})`);
  const bytes = await new Response(
    response.body.pipeThrough(new DecompressionStream('gzip')),
  ).arrayBuffer();
  signal.throwIfAborted();

  const channel = viewer.openChannel();
  try {
    channel.sendRrd(new Uint8Array(bytes));
  } finally {
    channel.close();
  }

  // The RRD is decoded asynchronously after it crosses into the viewer.
  const deadline = performance.now() + 30_000;
  while (performance.now() < deadline) {
    signal.throwIfAborted();
    const state = await viewer.getState();
    const recording = state.recordings.find((item) => item.storeId === state.activeStoreId);
    const timeline = recording?.timelines.find((item) => item.name === elapsed);
    if (
      timeline?.timeRange?.end === BigInt(endNanos) &&
      state.views.length >= 3 &&
      !state.loading.length
    ) {
      return recording.storeId;
    }
    await new Promise((resolve) => setTimeout(resolve, 35));
  }
  throw Error('Rerun did not finish opening the recording');
}

function sameBatch(current, next) {
  return JSON.stringify([current].flat()) === JSON.stringify([next].flat());
}

async function updateViews(viewer, recording, { visibility, chosen, groups, compact }) {
  const target = { kind: 'recording', recording };
  const { blueprint } = await viewer.getBlueprint(target);
  const focused = visibility !== 'all' && chosen;
  const selectedOnly = visibility === 'selected' && chosen;
  const selectedPath = chosen && `+ /motion/paths/${chosen.cluster}/${chosen.id}`;
  let changed = false;
  for (const [path, entry] of Object.entries(blueprint)) {
    const container = entry.ContainerBlueprint;
    if (
      container?.contents?.length === 2 &&
      container.contents.every((child) =>
        metrics.has(blueprint[`/${child}`]?.ViewBlueprint?.space_origin),
      )
    ) {
      const kind = compact ? 'Vertical' : 'Horizontal';
      if (container.container_kind !== kind) {
        container.container_kind = kind;
        changed = true;
      }
    }
    const origin = entry.ViewBlueprint?.space_origin;
    if (origin !== '/motion' && !metrics.has(origin)) continue;
    const contents = blueprint[`${path}/ViewContents`].ViewContents;
    let query;
    if (origin === '/motion') {
      query = ['+ /motion/origin'];
      if (!selectedOnly) {
        query.push(focused ? `+ /motion/batches/${chosen.cluster}` : '+ /motion/batches/**');
        if (chosen) query.push(`+ /motion/exemplars/${chosen.cluster}`);
      }
      if (selectedPath) query.push(selectedPath);
    } else {
      query = [focused ? `+ ${origin}/${chosen.cluster}` : `+ ${origin}/**`];
      const overridePrefix = `${path}/ViewContents/overrides${origin}/`;
      for (const [overridePath, override] of Object.entries(blueprint)) {
        if (!overridePath.startsWith(overridePrefix) || !override.SeriesLines) continue;
        const cluster = Number(overridePath.slice(overridePrefix.length).split('/')[0]);
        const ids = groups.get(cluster) ?? [];
        const chosenGroup = chosen?.cluster === cluster;
        const visible = selectedOnly && chosenGroup ? ids.map((id) => id === chosen.id) : true;
        const widths = chosenGroup ? ids.map((id) => (id === chosen.id ? 2.5 : 1)) : 1;
        const style = override.SeriesLines;
        if (!sameBatch(style.visible_series, visible) || !sameBatch(style.widths, widths)) {
          style.visible_series = visible;
          style.widths = widths;
          changed = true;
        }
      }
    }
    if (!sameBatch(contents.query, query)) {
      contents.query = query;
      changed = true;
    }
  }
  if (changed) await viewer.setBlueprint(blueprint, target);
}

function CompactPlayback({ viewer, recording, endNanos, sampleRate, onError }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = Math.round((endNanos * sampleRate) / 1e9);

  useEffect(() => {
    let disposed = false;
    const remove = [
      viewer.on('time_update', ({ time }) => setFrame(Math.round((time * sampleRate) / 1e9))),
      viewer.on('play', () => setPlaying(true)),
      viewer.on('pause', () => setPlaying(false)),
    ];
    viewer
      .getState()
      .then(({ recordings }) => {
        if (disposed) return;
        const active = recordings.find(({ storeId }) => storeId === recording);
        setFrame(Math.round((Number(active?.currentTime?.time ?? 0) * sampleRate) / 1e9));
        setPlaying(active?.playState === 'playing');
      })
      .catch((reason) => {
        if (!disposed) onError(reason.message);
      });
    return () => {
      disposed = true;
      remove.forEach((unsubscribe) => unsubscribe());
    };
  }, [viewer, recording, sampleRate, onError]);

  const setTime = (options) =>
    viewer.setTime({ recording, timeline: elapsed, ...options }).catch((reason) => {
      onError(reason.message);
    });

  return (
    <section className="compact-playback" aria-label="Playback">
      <Button
        label={playing ? 'Pause' : 'Play'}
        onClick={() =>
          setTime({
            playState: playing ? 'paused' : 'playing',
            durationNanos: !playing && frame === last ? 0 : undefined,
          })
        }
      />
      <section className="compact-scrubber">
        <Slider
          label="Elapsed time"
          value={frame}
          min={0}
          max={last}
          step={1}
          onChange={(next) => {
            setFrame(next);
            setTime({ durationNanos: Math.round((next * 1e9) / sampleRate) });
          }}
          formatValue={(value) => `${(value / sampleRate).toFixed(1)} s`}
          valueDisplay="text"
        />
      </section>
    </section>
  );
}

export default function RerunViewer({
  url,
  endNanos,
  sampleRate,
  visibility,
  chosen,
  rows,
  onSelect,
}) {
  const host = useRef(null);
  const [viewer, setViewer] = useState(null);
  const [recording, setRecording] = useState(null);
  const [error, setError] = useState('');
  const [focusError, setFocusError] = useState('');
  const [compact, setCompact] = useState(() => matchMedia('(max-width: 750px)').matches);
  const loadingQueue = useRef(Promise.resolve());
  const focusQueue = useRef(Promise.resolve());
  const pathId = chosen?.id;
  const cluster = chosen?.cluster;
  const groups = useMemo(() => {
    const grouped = new Map();
    for (const row of rows ?? []) {
      if (!grouped.has(row.cluster)) grouped.set(row.cluster, []);
      grouped.get(row.cluster).push(row.id);
    }
    return grouped;
  }, [rows]);

  useEffect(() => {
    const media = matchMedia('(max-width: 750px)');
    const resize = () => setCompact(media.matches);
    media.addEventListener('change', resize);
    return () => media.removeEventListener('change', resize);
  }, []);

  useEffect(() => {
    viewer?.overridePanelState('time', compact ? 'hidden' : null);
  }, [viewer, compact]);

  useEffect(() => {
    let disposed = false;
    const instance = new WebViewer();
    instance
      .start(null, host.current, {
        width: '100%',
        height: '100%',
        hideWelcomeScreen: true,
        checkForUpdatesOnStartup: false,
      })
      .then(() => {
        if (disposed) return;
        instance.overridePanelState('top', 'hidden');
        instance.overridePanelState('blueprint', 'hidden');
        instance.overridePanelState('selection', 'hidden');
        setViewer(instance);
      })
      .catch((reason) => {
        if (!disposed) setError(`Rerun could not start. ${reason.message}`);
      });
    return () => {
      disposed = true;
      instance.stop();
    };
  }, []);

  useEffect(() => {
    if (!viewer || !url) return;
    const controller = new AbortController();
    const { signal } = controller;
    setRecording(null);
    setError('');
    setFocusError('');
    loadingQueue.current = loadingQueue.current.then(async () => {
      try {
        if (signal.aborted) return;
        await focusQueue.current;
        if (signal.aborted) return;
        const { recordings } = await viewer.getState();
        if (recordings.length) await viewer.close('all');
        const id = await loadRecording(viewer, url, endNanos, signal);
        await viewer.setTime({
          recording: id,
          timeline: elapsed,
          durationNanos: 0,
          playState: 'paused',
        });
        if (!signal.aborted) setRecording({ id, url });
      } catch (reason) {
        if (!signal.aborted) setError(reason.message);
      }
    });
    return () => controller.abort();
  }, [viewer, url, endNanos]);

  useEffect(() => {
    if (!viewer || !recording || recording.url !== url) return;
    setFocusError('');
    let stale = false;
    focusQueue.current = focusQueue.current.then(async () => {
      if (stale) return;
      try {
        await updateViews(viewer, recording.id, {
          visibility,
          chosen: pathId == null ? null : { id: pathId, cluster },
          groups,
          compact,
        });
      } catch (reason) {
        if (!stale) setFocusError(`Could not update the visible trajectories. ${reason.message}`);
      }
    });
    return () => {
      stale = true;
    };
  }, [viewer, recording, url, visibility, pathId, cluster, groups, compact]);

  useEffect(() => {
    if (!viewer || !recording || recording.url !== url) return;
    return viewer.on('selection_change', (event) => {
      for (const item of event.items) {
        if (item.type !== 'entity') continue;
        const path = /^\/?motion\/paths\/\d+\/(\d+)$/.exec(item.entityPath);
        const batch = /^\/?(?:motion\/batches|acceleration|curvature)\/(\d+)$/.exec(
          item.entityPath,
        );
        const ids = batch ? groups.get(Number(batch[1])) : null;
        const id = path ? Number(path[1]) : ids?.[item.instanceId ?? (ids.length === 1 ? 0 : -1)];
        if (id != null) {
          onSelect(id);
          break;
        }
      }
    });
  }, [viewer, recording, url, onSelect, groups]);

  const ready = Boolean(url && recording?.url === url);
  return (
    <section aria-label="Rerun visualization" aria-busy={!ready && !error} className="viewer-host">
      <section className="rerun-viewport">
        <section className="rerun-surface" ref={host} />
      </section>
      {compact && ready && (
        <CompactPlayback
          key={recording.id}
          viewer={viewer}
          recording={recording.id}
          endNanos={endNanos}
          sampleRate={sampleRate}
          onError={setError}
        />
      )}
      {!ready && !error && <p className="viewer-status">Opening recording…</p>}
      {(error || focusError) && (
        <p className="error" role="alert">
          {error || focusError}
        </p>
      )}
    </section>
  );
}
