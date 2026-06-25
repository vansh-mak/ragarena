import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/leaderboard", label: "Leaderboard" },
  { to: "/head-to-head", label: "Head to Head" },
  { to: "/cost-quality", label: "Cost vs Quality" },
  { to: "/rag-selector", label: "RAG Selector" },
  { to: "/latency", label: "Latency" },
];

export default function Layout() {
  return (
    <div style={{ display: "flex", minHeight: "100vh", fontFamily: "sans-serif" }}>
      <nav
        style={{
          width: 220,
          background: "#111827",
          color: "#f9fafb",
          display: "flex",
          flexDirection: "column",
          padding: "24px 0",
          flexShrink: 0,
        }}
      >
        <div style={{ padding: "0 20px 24px", fontSize: 18, fontWeight: 700, color: "#60a5fa" }}>
          RAGArena
        </div>

        <div style={{ flex: 1 }}>
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              style={({ isActive }) => ({
                display: "block",
                padding: "10px 20px",
                color: isActive ? "#60a5fa" : "#d1d5db",
                background: isActive ? "#1e3a5f" : "transparent",
                textDecoration: "none",
                fontSize: 14,
                borderLeft: isActive ? "3px solid #60a5fa" : "3px solid transparent",
              })}
            >
              {label}
            </NavLink>
          ))}
        </div>

        <div style={{ borderTop: "1px solid #374151", paddingTop: 16 }}>
          <NavLink
            to="/new-run"
            style={{ display: "block", padding: "10px 20px", color: "#d1d5db", textDecoration: "none", fontSize: 14 }}
          >
            + New Run
          </NavLink>
          <NavLink
            to="/settings"
            style={{ display: "block", padding: "10px 20px", color: "#d1d5db", textDecoration: "none", fontSize: 14 }}
          >
            Settings
          </NavLink>
        </div>
      </nav>

      <main style={{ flex: 1, padding: 32, background: "#f9fafb" }}>
        <Outlet />
      </main>
    </div>
  );
}
