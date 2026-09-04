import { BrowserRouter as Router, Routes, Route } from "react-router";

import { ScrollToTop } from "./components/common/ScrollToTop";
import NotFound from "./pages/OtherPage/NotFound";

// Kerala Flood–Landslide Early Warning System
import Landing from "./pages/Landing/Landing";
import EarlyWarningDashboard from "./pages/EarlyWarning/Dashboard";
import WarningsPage from "./pages/EarlyWarning/Warnings";
import AnalysisPage from "./pages/EarlyWarning/Analysis";
import ModelPerformancePage from "./pages/EarlyWarning/ModelPerformance";

/**
 * Routing for the Kerala Flood–Landslide Early Warning System.
 *
 * There is no authentication: the application has no sign-in, no protected
 * routes and no redirect to a login page. Every page renders its own CRT
 * terminal window (frame, path label, nav and status bar), so there is no
 * shared sidebar layout to wrap them in.
 */
export default function App() {
  return (
    <Router>
      <ScrollToTop />
      <Routes>
        {/* Every route below is a self-contained terminal window. */}
        <Route index path="/" element={<Landing />} />

        <Route path="/early-warning" element={<EarlyWarningDashboard />} />
        <Route path="/warnings" element={<WarningsPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/model-performance" element={<ModelPerformancePage />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}
