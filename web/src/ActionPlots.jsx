import { useEffect, useMemo, useRef } from 'react';

const plots = [
  { index: 0, title: 'Acceleration', unit: 'm/s²', color: '#f2af72', floor: 1 },
  { index: 1, title: 'Curvature', unit: 'm⁻¹', color: '#84baff', floor: 0.1 },
];
const left = 42,
  right = 574,
  top = 16,
  bottom = 104;

function linePath(values, index, xAt, yAt) {
  let path = '',
    open = false;
  values?.forEach((row, i) => {
    const value = row[index];
    if (Number.isFinite(value)) {
      path += `${open ? 'L' : 'M'}${xAt(i).toFixed(1)},${yAt(value).toFixed(1)} `;
      open = true;
    } else open = false;
  });
  return path;
}

function Plot({ trajectories, selected, time, onSeek, config, frames }) {
  const canvas = useRef(null);
  const last = frames - 1;
  const xAt = (i) => left + (i * (right - left)) / last;
  const extent = useMemo(() => {
    let maximum = config.floor;
    for (const trace of trajectories) {
      for (const row of trace.values) {
        if (Number.isFinite(row[config.index]))
          maximum = Math.max(maximum, Math.abs(row[config.index]));
      }
    }
    return maximum;
  }, [trajectories, config]);
  const yAt = (value) => (top + bottom) / 2 - ((value / extent) * (bottom - top)) / 2;
  useEffect(() => {
    const element = canvas.current;
    const canvasX = (i) => left + (i * (right - left)) / last;
    const canvasY = (value) => (top + bottom) / 2 - ((value / extent) * (bottom - top)) / 2;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    element.width = 600 * dpr;
    element.height = 120 * dpr;
    const ctx = element.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, 600, 120);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1.1;
    for (const trace of trajectories) {
      if (trace.id === selected?.id) continue;
      ctx.strokeStyle = trace.color;
      ctx.beginPath();
      let open = false;
      trace.values.forEach((row, i) => {
        const value = row[config.index];
        if (Number.isFinite(value)) {
          if (open) ctx.lineTo(canvasX(i), canvasY(value));
          else ctx.moveTo(canvasX(i), canvasY(value));
          open = true;
        } else open = false;
      });
      ctx.stroke();
    }
    ctx.restore();
  }, [trajectories, selected, config, last, extent]);
  const values = selected?.values;
  const current = values?.[Math.min(last, Math.round(time))]?.[config.index];
  const label = Number.isFinite(current)
    ? `${current.toFixed(config.index ? 3 : 2)} ${config.unit}`
    : 'Select a visible path';
  const scrub = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const pixel = ((event.clientX - rect.left) / rect.width) * 600;
    const fraction = Math.max(0, Math.min(1, (pixel - left) / (right - left)));
    onSeek(Math.round(fraction * last));
  };
  return (
    <section className="action-plot">
      <header>
        <strong>{config.title}</strong>
        <output>{label}</output>
      </header>
      <section className="plot-surface">
        <canvas ref={canvas} aria-hidden="true" />
        <svg
          viewBox="0 0 600 120"
          preserveAspectRatio="none"
          role="slider"
          tabIndex="0"
          aria-label={`${config.title} time cursor`}
          aria-valuemin="0"
          aria-valuemax={last}
          aria-valuenow={Math.round(time)}
          aria-valuetext={`${(time / 10).toFixed(1)} seconds`}
          onKeyDown={(e) => {
            if (e.key === 'ArrowRight') onSeek(Math.min(last, Math.round(time) + 1));
            if (e.key === 'ArrowLeft') onSeek(Math.max(0, Math.round(time) - 1));
          }}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            scrub(e);
          }}
          onPointerMove={(e) => {
            if (e.buttons) scrub(e);
          }}
        >
          <line className="plot-grid" x1={left} x2={right} y1={yAt(0)} y2={yAt(0)} />
          <line className="plot-grid faint" x1={left} x2={right} y1={top} y2={top} />
          <line className="plot-grid faint" x1={left} x2={right} y1={bottom} y2={bottom} />
          <text x="3" y="19">
            +{extent.toFixed(config.index ? 2 : 1)}
          </text>
          <text x="9" y="65">
            0
          </text>
          <text x="3" y="107">
            −{extent.toFixed(config.index ? 2 : 1)}
          </text>
          <path
            d={linePath(values, config.index, xAt, yAt)}
            fill="none"
            stroke={selected?.color ?? config.color}
            strokeWidth="2.4"
            vectorEffect="non-scaling-stroke"
          />
          <line className="plot-cursor" x1={xAt(time)} x2={xAt(time)} y1={top} y2={bottom} />
          <text x={left} y="118">
            0 s
          </text>
          <text x={right - 25} y="118">
            {(last / 10).toFixed(1)} s
          </text>
        </svg>
      </section>
    </section>
  );
}

export default function ActionPlots({
  trajectories = [],
  selectedId,
  selectedLabel,
  time,
  onSeek,
  frames = 50,
}) {
  const selected = trajectories.find((trace) => trace.id === selectedId);
  const valid = selected?.values.filter((row) => Number.isFinite(row[1])).length ?? 0;
  return (
    <section className="diagnostics" aria-label="Inferred action diagnostics">
      <header className="diagnostics-heading">
        <strong>
          {trajectories.length} visible action traces
          {selected && ` · selected path ${selectedLabel ?? selectedId}`}
        </strong>
        <small>
          {valid}/{frames} selected valid · local cubic derivative, 0.7 s window · moving paths ≥1
          m/s
        </small>
      </header>
      <section className="plot-grid-layout">
        {plots.map((config) => (
          <Plot
            key={config.index}
            config={config}
            trajectories={trajectories}
            selected={selected}
            time={time}
            onSeek={onSeek}
            frames={frames}
          />
        ))}
      </section>
    </section>
  );
}
