import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Predict from "./pages/Predict";
import BulkPredict from "./pages/BulkPredict";
import History from "./pages/History";
import Insights from "./pages/Insights";
import Dataset from "./pages/Dataset";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-cream">
        <Navbar />
        <main className="max-w-6xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Predict />} />
            <Route path="/bulk" element={<BulkPredict />} />
            <Route path="/history" element={<History />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/dataset" element={<Dataset />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
