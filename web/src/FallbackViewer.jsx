import { useCallback, useEffect, useRef, useState } from 'react';

export default function FallbackViewer({
  paths,
  time,
  selectedIndex,
  labels,
  palette,
  exemplarPath,
  onSelect,
}) {
  const canvas = useRef(null);
  const [size, setSize] = useState([1, 1]);
  const [view, setView] = useState({ scale: 1, dx: 0, dy: 0 });
  const drag = useRef(null);
  const bounds = useRef(null);

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize([width, height]);
    });
    observer.observe(canvas.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!paths?.length) return;
    const all = paths.flat();
    const xs = all.map((p) => p[0]),
      ys = all.map((p) => p[1]);
    const minX = Math.min(...xs),
      maxX = Math.max(...xs);
    const minY = Math.min(...ys),
      maxY = Math.max(...ys);
    bounds.current = { minX, maxX, minY, maxY };
    setView({ scale: 1, dx: 0, dy: 0 });
  }, [paths]);

  const project = useCallback(
    (point, width, height) => {
      const b = bounds.current;
      const spanX = Math.max(1, b.maxX - b.minX),
        spanY = Math.max(1, b.maxY - b.minY);
      const unit = Math.min((width - 64) / spanX, (height - 64) / spanY) * view.scale;
      return [
        (point[0] - (b.minX + b.maxX) / 2) * unit + width / 2 + view.dx,
        -(point[1] - (b.minY + b.maxY) / 2) * unit + height / 2 + view.dy,
      ];
    },
    [view],
  );

  useEffect(() => {
    if (!paths?.length || !bounds.current) return;
    const element = canvas.current,
      [width, height] = size;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    element.width = Math.max(1, Math.round(width * dpr));
    element.height = Math.max(1, Math.round(height * dpr));
    const ctx = element.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#101722';
    ctx.fillRect(0, 0, width, height);
    const origin = project([0, 0], width, height);
    ctx.strokeStyle = '#34465a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, origin[1]);
    ctx.lineTo(width, origin[1]);
    ctx.moveTo(origin[0], 0);
    ctx.lineTo(origin[0], height);
    ctx.stroke();
    for (const [index, path] of paths.entries()) {
      const lateral = path[path.length - 1][1];
      const color =
        labels && palette
          ? palette[labels[index]]
          : lateral > 3
            ? '#ffad66'
            : lateral < -3
              ? '#83b9ff'
              : '#7de2c0';
      ctx.strokeStyle = color;
      ctx.globalAlpha = index === selectedIndex ? 1 : 0.22;
      ctx.lineWidth = index === selectedIndex ? 3 : 1.3;
      ctx.beginPath();
      for (let i = 0; i <= time && i < path.length; i++) {
        const [x, y] = project(path[i], width, height);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 0.85;
      const [x, y] = project(path[Math.min(time, path.length - 1)], width, height);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, index === selectedIndex ? 5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (exemplarPath) {
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = '#ffffff';
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = 2;
      ctx.beginPath();
      exemplarPath.forEach((point, i) => {
        if (i > time) return;
        const [x, y] = project(point, width, height);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  }, [paths, time, selectedIndex, labels, palette, exemplarPath, size, project]);

  const select = (event) => {
    if (drag.current?.moved || !paths?.length) return;
    const rect = canvas.current.getBoundingClientRect();
    const x = event.clientX - rect.left,
      y = event.clientY - rect.top;
    let best = Infinity,
      index = -1;
    paths.forEach((path, i) => {
      const [px, py] = project(path[Math.min(time, path.length - 1)], size[0], size[1]);
      const distance = Math.hypot(x - px, y - py);
      if (distance < best) {
        best = distance;
        index = i;
      }
    });
    if (best < 24 && index >= 0) {
      const [px, py] = paths[index][time];
      onSelect(index, `${px.toFixed(1)} m forward, ${py.toFixed(1)} m lateral`);
    }
  };

  return (
    <div className="canvas-shell">
      <canvas
        ref={canvas}
        role="img"
        aria-label={`${paths?.length ?? 0} ego-relative vehicle trajectories at ${(time / 10).toFixed(1)} seconds`}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, moved: false };
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const dx = e.clientX - drag.current.x,
            dy = e.clientY - drag.current.y;
          if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true;
          setView((v) => ({ ...v, dx: v.dx + dx, dy: v.dy + dy }));
          drag.current.x = e.clientX;
          drag.current.y = e.clientY;
        }}
        onPointerUp={(e) => {
          select(e);
          drag.current = null;
        }}
        onWheel={(e) => {
          e.preventDefault();
          setView((v) => ({
            ...v,
            scale: Math.max(0.25, Math.min(8, v.scale * (e.deltaY < 0 ? 1.12 : 0.89))),
          }));
        }}
      />
      <span className="canvas-note">
        Color = nearest exemplar · white dashed = exemplar · drag to pan · tap to inspect
      </span>
    </div>
  );
}
