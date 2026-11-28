import { Selector } from '@astryxdesign/core/Selector';

const options = [
  { value: 'all', label: 'All loaded paths' },
  { value: 'exemplar', label: 'Paths in selected exemplar' },
  { value: 'selected', label: 'Selected path only' },
];

export default function TrajectoryVisibility({ value, onChange }) {
  return (
    <Selector
      label="Show trajectories"
      value={value}
      options={options}
      onChange={(next) => onChange(next || 'all')}
      presentation="adaptive"
    />
  );
}
