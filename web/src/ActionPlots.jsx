const plots = [
  { index: 0, title: 'Acceleration', unit: 'm/s²', color: '#f2af72', floor: 1 },
  { index: 1, title: 'Curvature', unit: 'm⁻¹', color: '#84baff', floor: 0.1 },
];
const left = 42,
  right = 574,
  top = 16,
  bottom = 104;
const xAt = (i) => left + (i * (right - left)) / 49;

function Plot({ series, time, onSeek, config }) {
  const values = series?.map((row) => row[config.index]) ?? [];
  const finite = values.filter(Number.isFinite);
  const extent = Math.max(config.floor, ...finite.map(Math.abs));
  const yAt = (value) => (top + bottom) / 2 - ((value / extent) * (bottom - top)) / 2;
  let path = '',
    open = false;
  values.forEach((value, i) => {
    if (Number.isFinite(value)) {
      path += `${open ? 'L' : 'M'}${xAt(i).toFixed(1)},${yAt(value).toFixed(1)} `;
      open = true;
    } else open = false;
  });
  const current = values[Math.min(49, Math.round(time))];
  const label = Number.isFinite(current)
    ? `${current.toFixed(config.index ? 3 : 2)} ${config.unit}`
    : 'undefined at low speed';
  const scrub = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const pixel = ((event.clientX - rect.left) / rect.width) * 600;
    const fraction = Math.max(0, Math.min(1, (pixel - left) / (right - left)));
    onSeek(Math.round(fraction * 49));
  };
  return (
    <section className="action-plot">
      <header>
        <strong>{config.title}</strong>
        <output>{label}</output>
      </header>
      <svg
        viewBox="0 0 600 120"
        preserveAspectRatio="none"
        role="slider"
        tabIndex="0"
        aria-label={`${config.title} time cursor`}
        aria-valuemin="0"
        aria-valuemax="49"
        aria-valuenow={Math.round(time)}
        aria-valuetext={`${(time / 10).toFixed(1)} seconds`}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') onSeek(Math.min(49, Math.round(time) + 1));
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
          d={path}
          fill="none"
          stroke={config.color}
          strokeWidth="2.2"
          vectorEffect="non-scaling-stroke"
        />
        <line className="plot-cursor" x1={xAt(time)} x2={xAt(time)} y1={top} y2={bottom} />
        <text x={left} y="118">
          0 s
        </text>
        <text x={right - 25} y="118">
          4.9 s
        </text>
      </svg>
    </section>
  );
}

export default function ActionPlots({ series, time, onSeek, sampleIndex }) {
  const valid = series?.filter((row) => Number.isFinite(row[1])).length ?? 0;
  return (
    <section className="diagnostics" aria-label="Inferred action diagnostics">
      <header className="diagnostics-heading">
        <strong>Inferred actions · sample path {sampleIndex + 1}</strong>
        <small>{valid}/50 valid · local cubic derivative, 0.7 s window · moving paths ≥1 m/s</small>
      </header>
      <section className="plot-grid-layout">
        {plots.map((config) => (
          <Plot key={config.index} config={config} series={series} time={time} onSeek={onSeek} />
        ))}
      </section>
    </section>
  );
}
