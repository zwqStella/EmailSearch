import { Link, NavLink, Route, Routes } from 'react-router-dom';
import SearchPage from './pages/SearchPage';
import LoadPage from './pages/LoadPage';
import SettingsPage from './pages/SettingsPage';

const tabClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded text-sm font-medium ${
    isActive ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-200'
  }`;

export default function App() {
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
            <NavLink to="/load" className={tabClass}>
              Load
            </NavLink>
            <NavLink to="/settings" className={tabClass}>
              Settings
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 overflow-auto">
        <div className="max-w-screen-2xl mx-auto px-6 py-4">
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/load" element={<LoadPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
