import { lazy, Suspense } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Shell from './components/Shell';
import { Loading } from './components/bits';
import Home from './pages/Home';

const Test = lazy(() => import('./pages/Test'));
const Time = lazy(() => import('./pages/Time'));
const Week = lazy(() => import('./pages/Week'));
const Shabbos = lazy(() => import('./pages/Shabbos'));
const Metrics = lazy(() => import('./pages/Metrics'));
const Guests = lazy(() => import('./pages/Guests'));
const About = lazy(() => import('./pages/About'));

const NotFound = () => <div className="pt-24"><h1 className="c-h1">Nothing here</h1><p className="c-lede mt-5">That page does not exist. <Link to="/" className="c-link">Back to the ideas</Link>.</p></div>;

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Home />} />
          <Route path="test" element={<Test />} />
          <Route path="time" element={<Time />} />
          <Route path="week" element={<Week />} />
          <Route path="shabbos" element={<Shabbos />} />
          <Route path="metrics" element={<Metrics />} />
          <Route path="guests" element={<Guests />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
