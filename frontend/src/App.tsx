import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router";

import AppLayout from "./layout/AppLayout";
import { ScrollToTop } from "./components/common/ScrollToTop";
import NotFound from "./pages/OtherPage/NotFound";

// Kerala Flood–Landslide Early Warning System
import EarlyWarningDashboard from "./pages/EarlyWarning/Dashboard";
import WarningsPage from "./pages/EarlyWarning/Warnings";
import AnalysisPage from "./pages/EarlyWarning/Analysis";
import ModelPerformancePage from "./pages/EarlyWarning/ModelPerformance";

/**
 * Routing for the Kerala Flood–Landslide Early Warning System.
 *
 * There is no authentication: the application has no sign-in, no protected
 * routes and no redirect to a login page. The root path goes straight to the
 * Early Warning dashboard, on first load and on refresh alike.
 */
export default function App() {
  return (
    <Router>
      <ScrollToTop />
      <Routes>
        <Route element={<AppLayout />}>
          <Route index path="/" element={<Navigate to="/early-warning" replace />} />
          <Route path="/early-warning" element={<EarlyWarningDashboard />} />
          <Route path="/warnings" element={<WarningsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/model-performance" element={<ModelPerformancePage />} />
        </Route>

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}
