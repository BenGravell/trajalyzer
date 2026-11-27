//! Native CSV-to-corpus pipeline. Memory is bounded by one source CSV's tracks
//! plus 128 pending windows per cohort/exemplar bucket; the full corpus is never held.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const FRAMES: usize = 50;
const RATE: f64 = 10.0;
const STRIDE: f64 = 1.0;
const MIN_SPEED: f64 = 1.0;
const SHARD_SIZE: usize = 128;

#[derive(Deserialize, Serialize)]
struct Row {
    track_id: String,
    timestamp_ms: i64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    psi_rad: f64,
}
#[derive(Deserialize)]
struct Model {
    features: Vec<Vec<f32>>,
}
#[derive(Serialize)]
struct Shard {
    cohort: String,
    cluster: usize,
    count: usize,
    path: String,
}
#[derive(Serialize)]
struct Manifest<'a> {
    version: u8,
    count: u64,
    frames: usize,
    sample_rate_hz: u8,
    position_scale_m: f64,
    shards: &'a [Shard],
}
struct Window {
    positions: [[f32; 2]; FRAMES],
    actions: [[f32; 2]; FRAMES],
    distance: f32,
    id: u32,
}
struct Builder {
    root: PathBuf,
    pending: BTreeMap<(String, usize), Vec<Window>>,
    parts: BTreeMap<(String, usize), usize>,
    shards: Vec<Shard>,
    count: u64,
    index: BufWriter<File>,
}
impl Builder {
    fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(root.join("shards"))?;
        let mut index = BufWriter::new(File::create(root.join("index.csv"))?);
        writeln!(index, "trajectory_id,source_file,track_id,start_time_s,cohort,cluster")?;
        Ok(Self { root, pending: BTreeMap::new(), parts: BTreeMap::new(), shards: Vec::new(), count: 0, index })
    }
    fn add(&mut self, positions: [[f32; 2]; FRAMES], actions: [[f32; 2]; FRAMES],
           cluster: usize, distance: f32, source: &Path, track: &str, start: f64) -> Result<()> {
        let lateral = positions[FRAMES - 1][1];
        let cohort = if lateral > 3.0 { "positive" } else if lateral < -3.0 { "negative" } else { "center" };
        let id = u32::try_from(self.count)?;
        let source_name = source.file_name().ok_or("Missing source file name")?.to_string_lossy();
        // Source filenames and INTERACTION numeric track IDs are plain CSV tokens.
        writeln!(self.index, "{id},{source_name},{track},{start:.3},{cohort},{cluster}")?;
        let key = (cohort.to_owned(), cluster);
        let bucket = self.pending.entry(key.clone()).or_default();
        bucket.push(Window { positions, actions, distance, id });
        self.count += 1;
        if bucket.len() == SHARD_SIZE { self.flush(&key)?; }
        Ok(())
    }
    fn flush(&mut self, key: &(String, usize)) -> Result<()> {
        let Some(rows) = self.pending.get_mut(key) else { return Ok(()); };
        if rows.is_empty() { return Ok(()); }
        let part = self.parts.entry(key.clone()).or_insert(0);
        let stem = format!("shards/{}-{:04}-{:06}", key.0, key.1, *part);
        *part += 1;
        let mut paths = BufWriter::new(File::create(self.root.join(format!("{stem}.bin")))?);
        let mut actions = BufWriter::new(File::create(self.root.join(format!("{stem}.actions.bin")))?);
        for row in rows.iter() {
            for point in row.positions {
                for value in point {
                    let cm = (f64::from(value) * 100.0).round();
                    if !(-32767.0..=32767.0).contains(&cm) { return Err("Position exceeds int16 centimetres".into()); }
                    paths.write_all(&(cm as i16).to_le_bytes())?;
                }
            }
            paths.write_all(&row.id.to_le_bytes())?;
            paths.write_all(&row.distance.to_le_bytes())?;
            for pair in row.actions { for value in pair { actions.write_all(&value.to_le_bytes())?; } }
        }
        self.shards.push(Shard { cohort: key.0.clone(), cluster: key.1, count: rows.len(), path: format!("{stem}.bin") });
        rows.clear();
        Ok(())
    }
    fn finish(mut self) -> Result<()> {
        let keys: Vec<_> = self.pending.keys().cloned().collect();
        for key in &keys { self.flush(key)?; }
        self.index.flush()?;
        let manifest = Manifest { version: 1, count: self.count, frames: FRAMES, sample_rate_hz: 10, position_scale_m: 0.01, shards: &self.shards };
        serde_json::to_writer(File::create(self.root.join("manifest.json"))?, &manifest)?;
        println!("Indexed {} windows in {} shards", self.count, self.shards.len());
        Ok(())
    }
}

// Coefficients of a local least-squares cubic, computed once for this grid.
fn stencils() -> Vec<(usize, [f64; 7], [f64; 7])> {
    (0..FRAMES).map(|i| {
        let start = i.saturating_sub(3).min(FRAMES - 7);
        let mut powers = [[0.0; 4]; 7];
        for (j, row) in powers.iter_mut().enumerate() {
            let t = (start as f64 + j as f64 - i as f64) / RATE;
            *row = [1.0, t, t * t, t * t * t];
        }
        let mut normal = [[0.0; 8]; 4];
        for r in 0..4 { for c in 0..4 { normal[r][c] = powers.iter().map(|p| p[r] * p[c]).sum(); } normal[r][r + 4] = 1.0; }
        for k in 0..4 {
            let pivot = (k..4).max_by(|a, b| normal[*a][k].abs().total_cmp(&normal[*b][k].abs())).unwrap();
            normal.swap(k, pivot);
            let divisor = normal[k][k];
            for c in 0..8 { normal[k][c] /= divisor; }
            for r in 0..4 { if r != k {
                let factor = normal[r][k];
                for c in 0..8 { normal[r][c] -= factor * normal[k][c]; }
            }}
        }
        let mut first = [0.0; 7]; let mut second = [0.0; 7];
        for j in 0..7 {
            first[j] = (0..4).map(|k| normal[1][k + 4] * powers[j][k]).sum();
            second[j] = 2.0 * (0..4).map(|k| normal[2][k + 4] * powers[j][k]).sum::<f64>();
        }
        (start, first, second)
    }).collect()
}
fn actions_and_features(p: &[[f32; 2]; FRAMES], weights: &[(usize, [f64; 7], [f64; 7])]) -> Option<([[f32; 2]; FRAMES], Vec<f32>)> {
    let mut actions = [[0.0; 2]; FRAMES];
    let mut speeds = [0.0; FRAMES];
    for i in 0..FRAMES {
        let (start, first, second) = &weights[i];
        let mut v = [0.0; 2]; let mut a = [0.0; 2];
        for j in 0..7 { for axis in 0..2 {
            v[axis] += first[j] * f64::from(p[start + j][axis]);
            a[axis] += second[j] * f64::from(p[start + j][axis]);
        }}
        let speed = v[0].hypot(v[1]);
        if !speed.is_finite() || speed < MIN_SPEED { return None; }
        speeds[i] = speed;
        actions[i] = [((v[0] * a[0] + v[1] * a[1]) / speed) as f32,
                      ((v[0] * a[1] - v[1] * a[0]) / speed.powi(3)) as f32];
    }
    let mut feature = Vec::with_capacity(125);
    for axis in 0..2 { for i in (0..FRAMES).step_by(2) { feature.push(p[i][axis] / if axis == 0 {20.0} else {10.0}); } }
    for i in (0..FRAMES).step_by(2) { feature.push((speeds[i] / 8.0) as f32); }
    for axis in 0..2 { for i in (0..FRAMES).step_by(2) {
        let clipped = if axis == 0 { actions[i][axis].clamp(-5.0, 5.0) / 3.0 } else { actions[i][axis].clamp(-1.0, 1.0) / 0.2 };
        feature.push(clipped);
    }}
    Some((actions, feature))
}
fn assign(feature: &[f32], model: &Model) -> Result<(usize, f32)> {
    let (id, squared) = model.features.iter().enumerate().map(|(i, exemplar)| {
        let distance: f32 = feature.iter().zip(exemplar).map(|(x, y)| (x - y).powi(2)).sum();
        (i, distance)
    }).min_by(|a, b| a.1.total_cmp(&b.1)).ok_or("Empty exemplar model")?;
    Ok((id, squared.sqrt()))
}
fn interpolate(rows: &[Row], target: f64, cursor: &mut usize) -> [f64; 2] {
    while *cursor + 1 < rows.len() - 1 && rows[*cursor + 1].timestamp_ms as f64 / 1000.0 < target { *cursor += 1; }
    let left = &rows[*cursor]; let right = &rows[*cursor + 1];
    let t0 = left.timestamp_ms as f64 / 1000.0;
    let t1 = right.timestamp_ms as f64 / 1000.0;
    let alpha = ((target - t0) / (t1 - t0)).clamp(0.0, 1.0);
    [left.x + alpha * (right.x - left.x), left.y + alpha * (right.y - left.y)]
}
fn process_segment(rows: &[Row], source: &Path, track: &str, model: &Model, builder: &mut Builder, weights: &[(usize, [f64; 7], [f64; 7])]) -> Result<()> {
    if rows.len() < 2 { return Ok(()); }
    let first = rows[0].timestamp_ms as f64 / 1000.0;
    let last = rows[rows.len() - 1].timestamp_ms as f64 / 1000.0;
    let mut start = first;
    while start + (FRAMES - 1) as f64 / RATE <= last + 1e-9 {
        let mut cursor = 0;
        let mut world = [[0.0; 2]; FRAMES];
        for (i, point) in world.iter_mut().enumerate() { *point = interpolate(rows, start + i as f64 / RATE, &mut cursor); }
        let initial = rows.partition_point(|r| (r.timestamp_ms as f64 / 1000.0) < start).min(rows.len() - 1);
        let heading = if rows[initial].psi_rad.is_finite() { rows[initial].psi_rad } else { rows[initial].vy.atan2(rows[initial].vx) };
        if heading.is_finite() {
            let (s, c) = heading.sin_cos();
            let mut positions = [[0.0; 2]; FRAMES];
            for i in 0..FRAMES {
                let dx = world[i][0] - world[0][0]; let dy = world[i][1] - world[0][1];
                positions[i] = [(c * dx + s * dy) as f32, (-s * dx + c * dy) as f32];
            }
            if let Some((actions, features)) = actions_and_features(&positions, weights) {
                let (cluster, distance) = assign(&features, model)?;
                builder.add(positions, actions, cluster, distance, source, track, start)?;
            }
        }
        start += STRIDE;
    }
    Ok(())
}
fn process_partition(path: &Path, source: &Path, model: &Model, builder: &mut Builder, weights: &[(usize, [f64; 7], [f64; 7])]) -> Result<()> {
    let mut tracks: HashMap<String, Vec<Row>> = HashMap::new();
    for row in csv::Reader::from_path(path)?.deserialize::<Row>() {
        let row = row?;
        tracks.entry(row.track_id.clone()).or_default().push(row);
    }
    let mut tracks: Vec<_> = tracks.into_iter().collect();
    tracks.sort_by(|a, b| a.0.cmp(&b.0));
    for (track, mut rows) in tracks {
        rows.sort_by_key(|r| r.timestamp_ms);
        let mut segment = 0;
        for i in 1..=rows.len() {
            let cut = if i == rows.len() { true } else {
                let dt = (rows[i].timestamp_ms - rows[i - 1].timestamp_ms) as f64 / 1000.0;
                let speed = (rows[i].x - rows[i - 1].x).hypot(rows[i].y - rows[i - 1].y) / dt;
                dt <= 0.0 || dt > 0.3 || !speed.is_finite() || speed > 70.0
            };
            if cut {
                process_segment(&rows[segment..i], source, &track, model, builder, weights)?;
                segment = i;
            }
        }
    }
    Ok(())
}
fn process_csv(path: &Path, model: &Model, builder: &mut Builder, weights: &[(usize, [f64; 7], [f64; 7])]) -> Result<()> {
    // Hash partitioning bounds memory even if one CSV contains millions of rows.
    // All rows for a track land in the same temporary partition. Small files
    // need just one partition; large files target roughly 16 MiB of raw CSV.
    let partitions = (fs::metadata(path)?.len().div_ceil(16 * 1024 * 1024) as usize).clamp(1, 256);
    let stage = builder.root.join(".stage");
    fs::create_dir_all(&stage)?;
    let mut writers: Vec<_> = (0..partitions)
        .map(|i| csv::Writer::from_path(stage.join(format!("{i:02}.csv"))))
        .collect::<std::result::Result<_, _>>()?;
    for row in csv::Reader::from_path(path)?.deserialize::<Row>() {
        let row = row?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        row.track_id.hash(&mut hasher);
        writers[(hasher.finish() as usize) % partitions].serialize(row)?;
    }
    for writer in &mut writers { writer.flush()?; }
    drop(writers);
    for i in 0..partitions {
        let partition = stage.join(format!("{i:02}.csv"));
        process_partition(&partition, path, model, builder, weights)?;
        fs::remove_file(partition)?;
    }
    fs::remove_dir(stage)?;
    Ok(())
}
fn collect_csv(input: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if input.is_dir() {
        for entry in fs::read_dir(input)? { collect_csv(&entry?.path(), files)?; }
    } else if input.extension().and_then(|e| e.to_str()) == Some("csv") { files.push(input.to_path_buf()); }
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 { return Err("Usage: trajalyzer-corpus INPUT_CSV_OR_DIRECTORY EXEMPLARS_JSON OUTPUT_DIRECTORY".into()); }
    let model: Model = serde_json::from_reader(File::open(&args[2])?)?;
    if model.features.is_empty() || model.features.iter().any(|f| f.len() != 125) { return Err("Expected nonempty 125-dimensional exemplar features".into()); }
    let mut builder = Builder::new(PathBuf::from(&args[3]))?;
    let weights = stencils();
    let mut files = Vec::new();
    collect_csv(Path::new(&args[1]), &mut files)?;
    files.sort();
    if files.is_empty() { return Err("No CSV files found".into()); }
    for file in files { process_csv(&file, &model, &mut builder, &weights)?; }
    builder.finish()
}
fn main() { if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn straight_acceleration_has_zero_curvature() {
        let mut p = [[0.0; 2]; FRAMES];
        for (i, point) in p.iter_mut().enumerate() { let t = i as f32 / 10.0; point[0] = 5.0 * t + 0.5 * t * t; }
        let (a, f) = actions_and_features(&p, &stencils()).unwrap();
        assert_eq!(f.len(), 125);
        for pair in a { assert!((pair[0] - 1.0).abs() < 1e-3); assert!(pair[1].abs() < 1e-5); }
    }
    #[test]
    fn csv_pipeline_emits_addressable_shards() {
        let root = std::env::temp_dir().join(format!("trajalyzer-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let csv_path = root.join("vehicle_tracks_test.csv");
        let mut csv = File::create(&csv_path).unwrap();
        writeln!(csv, "track_id,timestamp_ms,x,y,vx,vy,psi_rad").unwrap();
        for i in 0..60 { writeln!(csv, "7,{},{},0,4,0,0", i * 100, i as f64 * 0.4).unwrap(); }
        drop(csv);
        let mut sample = [[0.0; 2]; FRAMES];
        for (i, point) in sample.iter_mut().enumerate() { point[0] = i as f32 * 0.4; }
        let weights = stencils();
        let (_, feature) = actions_and_features(&sample, &weights).unwrap();
        let model = Model { features: vec![feature] };
        let mut builder = Builder::new(root.join("out")).unwrap();
        process_csv(&csv_path, &model, &mut builder, &weights).unwrap();
        builder.finish().unwrap();
        let manifest: serde_json::Value = serde_json::from_reader(File::open(root.join("out/manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["count"], 2);
        assert_eq!(manifest["shards"][0]["count"], 2);
        let first = manifest["shards"][0]["path"].as_str().unwrap();
        assert_eq!(fs::metadata(root.join("out").join(first)).unwrap().len(), 416);
        fs::remove_dir_all(root).unwrap();
    }
}
