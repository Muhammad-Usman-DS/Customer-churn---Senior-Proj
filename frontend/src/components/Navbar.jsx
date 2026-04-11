import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Predict" },
  { to: "/bulk", label: "Bulk Predict" },
  { to: "/history", label: "History" },
  { to: "/insights", label: "Insights" },
  { to: "/dataset", label: "Dataset" },
];

export default function Navbar() {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between h-16">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-slate-800 flex items-center justify-center">
            <span className="text-white text-xs font-bold">C</span>
          </div>
          <span className="font-semibold text-slate-800 text-sm tracking-tight">
            ChurnIQ
          </span>
        </div>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Built-by badge */}
        <a
          href="https://github.com/Muhammad-Usman-DS"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
        >
          Muhammad Usman
        </a>
      </div>
    </header>
  );
}
