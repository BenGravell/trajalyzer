import { lazy, Suspense, useState } from 'react';
import CorpusExplorer from './CorpusExplorer.jsx';

const DiagnosticApp = lazy(() => import('./DiagnosticApp.jsx'));

export default function App() {
  const [page, setPage] = useState('corpus');
  return page === 'corpus' ? (
    <CorpusExplorer onDiagnostics={() => setPage('diagnostics')} />
  ) : (
    <Suspense fallback={<p>Loading Rerun diagnostics…</p>}>
      <DiagnosticApp onCorpus={() => setPage('corpus')} />
    </Suspense>
  );
}
