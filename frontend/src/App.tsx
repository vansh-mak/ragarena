import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RunProvider } from "./context/RunContext";
import Layout from "./components/Layout";
import Leaderboard from "./pages/Leaderboard";
import HeadToHead from "./pages/HeadToHead";
import CostQuality from "./pages/CostQuality";
import RAGSelector from "./pages/RAGSelector";
import Latency from "./pages/Latency";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RunProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/leaderboard" replace />} />
              <Route path="leaderboard" element={<Leaderboard />} />
              <Route path="head-to-head" element={<HeadToHead />} />
              <Route path="cost-quality" element={<CostQuality />} />
              <Route path="rag-selector" element={<RAGSelector />} />
              <Route path="latency" element={<Latency />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </RunProvider>
    </QueryClientProvider>
  );
}
