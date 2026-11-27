import {useEffect, useRef, useState} from 'react';
import {WebViewer} from '@rerun-io/web-viewer';
import {Button} from '@astryxdesign/core/Button';
import {Slider} from '@astryxdesign/core/Slider';
import {Selector} from '@astryxdesign/core/Selector';
import {Badge} from '@astryxdesign/core/Badge';

const cohorts = [
  {value:'all', label:'All sampled paths', detail:'The 500-window display sample'},
  {value:'center', label:'Near center', detail:'Final lateral displacement within ±3 m'},
  {value:'positive', label:'Positive lateral', detail:'Final lateral displacement above +3 m'},
  {value:'negative', label:'Negative lateral', detail:'Final lateral displacement below −3 m'},
];
const url = name => new URL(`${import.meta.env.BASE_URL}${name}.rrd`, document.baseURI).href;

export default function App() {
  const host = useRef(null), viewer = useRef(null), recordingId = useRef(null);
  const [cohort,setCohort]=useState('all'), [summary,setSummary]=useState(null);
  const [ready,setReady]=useState(false), [error,setError]=useState('');
  const [playing,setPlaying]=useState(false), [time,setTime]=useState(0);
  const [selection,setSelection]=useState('Tap a path or position to inspect it.');
  const [focus,setFocus]=useState(false);

  useEffect(()=>{
    fetch(url('cohorts').replace('.rrd','.json'))
      .then(r=>{if(!r.ok)throw Error('Summary unavailable');return r.json()})
      .then(setSummary).catch(e=>setError(e.message));
  },[]);

  useEffect(()=>{
    let disposed=false;
    const instance=new WebViewer();viewer.current=instance;recordingId.current=null;
    setReady(false);setError('');setPlaying(false);setTime(0);
    setSelection('Tap a path or position to inspect it.');
    const remove=[
      instance.on('recording_open',e=>{if(disposed)return;recordingId.current=e.recording_id;instance.set_active_timeline(e.recording_id,'elapsed');setReady(true)}),
      instance.on('time_update',e=>{if(!disposed)setTime(Math.min(49,Math.max(0,e.time/100_000_000)))}),
      instance.on('play',()=>{if(!disposed)setPlaying(true)}),
      instance.on('pause',()=>{if(!disposed)setPlaying(false)}),
      instance.on('selection_change',e=>{if(disposed)return;const item=e.items.find(x=>x.type==='entity');setSelection(item?`${item.entity_path}${item.instance_id==null?'':` · item ${item.instance_id}`}`:'Tap a path or position to inspect it.')}),
    ];
    instance.start(url(cohort),host.current,null).catch(e=>{if(!disposed)setError(`Rerun could not open the recording: ${e.message}`)});
    return ()=>{disposed=true;remove.forEach(fn=>fn());instance.stop();if(viewer.current===instance)viewer.current=null};
  },[cohort]);

  const seek=value=>{setTime(value);if(recordingId.current)viewer.current?.set_current_time(recordingId.current,'elapsed',Math.round(value*100_000_000))};
  const toggle=()=>{if(!recordingId.current)return;viewer.current?.set_playing(recordingId.current,!playing);setPlaying(!playing)};
  const active=cohorts.find(x=>x.value===cohort);
  return <article className={`app ${focus?'focus':''}`}>
    <header className="masthead"><section><p className="eyebrow">TRAJECTORY MANIFOLD / EXPLORE</p><h1>Trajalyzer</h1><p className="subhead">Real vehicle motion in a shared ego-relative frame</p></section><section className="header-actions"><Badge label={ready?'Recording ready':'Loading recording'}/><Button label={focus?'Show controls':'Focus view'} onClick={()=>setFocus(!focus)}/></section></header>
    <section className="workspace"><aside className="sidebar" aria-label="Trajectory controls">
      <section className="intro"><p className="eyebrow">DATASET</p><h2>INTERACTION</h2><p>Traffic-camera intersection · five-second windows sampled at 10 Hz.</p></section>
      <section className="metric-row"><article><strong>{summary?.source_windows?.toLocaleString()??'—'}</strong><small>valid windows loaded</small></article><article><strong>{summary?.counts?.[cohort]??'—'}</strong><small>paths in view</small></article></section>
      <section className="control-group"><p className="eyebrow">EXPLORE</p><Selector label="Trajectory group" value={cohort} options={cohorts} onChange={value=>value&&setCohort(value)} presentation="adaptive"/><p className="supporting">{active.detail}. This threshold is descriptive, not a maneuver label.</p></section>
      <section className="control-group"><p className="eyebrow">PLAYBACK</p><Button label={playing?'Pause trajectories':'Play trajectories'} variant="primary" isDisabled={!ready} onClick={toggle}/><Slider label="Elapsed time" value={time} min={0} max={49} step={1} onChange={seek} formatValue={value=>`${(value/10).toFixed(1)} s`} valueDisplay="text" isDisabled={!ready}/><p className="supporting">Use this control or Rerun’s timeline to scrub through 50 positions.</p></section>
      <section className="control-group inspector"><p className="eyebrow">INSPECTOR</p><p>{selection}</p><p className="supporting">Pan and zoom in Rerun. Select a point to see its instance and position.</p></section>
    </aside><section className="viewer-pane" aria-label="Rerun trajectory visualization"><header className="viewer-heading"><strong>{active.label}</strong><small>{summary?.counts?.[cohort]??'—'} paths · metres · 0–4.9 seconds</small></header><section className="viewer-host" ref={host}/>{error&&<p className="error" role="alert">{error}</p>}</section></section>
    <footer>Source: <a href="https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset">INTERACTION dataset</a> · CC BY 4.0. One intersection subset; overlapping windows may share tracks. <a href="https://github.com/BenGravell/trajalyzer">Repository</a>.</footer>
  </article>;
}
