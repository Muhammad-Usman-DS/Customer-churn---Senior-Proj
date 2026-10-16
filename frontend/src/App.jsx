import { BrowserRouter, Routes, Route, Outlet } from "react-router-dom";
import Navbar from "./components/Navbar";
import Predict from "./pages/Predict";
import BulkPredict from "./pages/BulkPredict";
import History from "./pages/History";
import Insights from "./pages/Insights";
import Dataset from "./pages/Dataset";
import SystemMap from "./pages/SystemMap";

function Contained() {
  return (
    <main className="max-w-6xl mx-auto px-6 py-8">
      <Outlet />
    </main>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-cream">
        <Navbar />
        <Routes>
          <Route element={<Contained />}>
            <Route path="/" element={<Predict />} />
            <Route path="/bulk" element={<BulkPredict />} />
            <Route path="/history" element={<History />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/dataset" element={<Dataset />} />
          </Route>
          {/* Full width: the map needs more room than the 6xl column */}
          <Route path="/system-map" element={<SystemMap />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
