import { useEffect } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom';
import SearchPage from './pages/SearchPage';
import AskPage from './pages/AskPage';
import LoadPage from './pages/LoadPage';

const tabClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded text-sm font-medium ${
    isActive ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-200'
  }`;

type Tab = 'search' | 'ask' | 'load';

/** Map the current pathname to which page should be visible. Anything
 *  outside the known set falls back to Search — same behavior as the
 *  prior `<Route path="*" />` catch-all. `/settings` is treated as
 *  `/load` here AND rewritten by the effect below so the URL stays
 *  canonical. */
function activeTab(pathname: string): Tab {
  if (pathname.startsWith('/ask')) return 'ask';
  if (pathname.startsWith('/load') || pathname.startsWith('/settings')) return 'load';
  return 'search';
}

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const active = activeTab(location.pathname);

  // Legacy `/settings` route — the panels were merged into `/load`.
  // We rewrite the URL (replace, not push) so bookmarks keep working
  // but back/forward doesn't bounce through the old path.
  useEffect(() => {
    if (location.pathname.startsWith('/settings')) {
      navigate('/load' + location.search + location.hash, { replace: true });
    }
  }, [location.pathname, location.search, location.hash, navigate]);

  return (
    <div className="h-full flex flex-col bg-gray-50">
      <header className="border-b bg-white">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-6">
          <Link to="/" className="text-lg font-semibold text-gray-900">
            EmailSearch
          </Link>
          <nav className="flex gap-2">
            <NavLink to="/" end className={tabClass}>
              Search
            </NavLink>
            <NavLink to="/ask" className={tabClass}>
              Ask
            </NavLink>
            <NavLink to="/load" className={tabClass}>
              Load
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <div className="max-w-screen-2xl mx-auto px-6 py-4">
          {/* All three pages stay mounted so per-page state (typed
              query, streamed hits, selected email, in-flight requests,
              filter dropdowns) survives tab switches. Inactive tabs
              are hidden via the HTML `hidden` attribute (display:none),
              which removes them from layout without tearing down their
              React subtree.

              Why not `<Routes>`? Routes unmounts the inactive element
              and re-mounts it on return with fresh state — so a user
              who ran a search, switched to Load, then switched back
              used to see an empty Search page. */}
          <div hidden={active !== 'search'}>
            <SearchPage />
          </div>
          <div hidden={active !== 'ask'}>
            <AskPage />
          </div>
          <div hidden={active !== 'load'}>
            <LoadPage />
          </div>
        </div>
      </main>
    </div>
  );
}
