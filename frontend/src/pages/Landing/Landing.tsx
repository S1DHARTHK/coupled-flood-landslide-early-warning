/**
 * Public landing page for the Kerala Flood–Landslide Early Warning System.
 *
 * Retro CRT terminal treatment: the page is one fake terminal window on a
 * black canvas under a fixed scanline overlay, monospace throughout, with
 * phosphor green #39ff7a as the only accent and a single amber #ffd24a for
 * live warning status. The window frame itself is shared with the dashboard
 * (components/terminal/TerminalUI).
 */

import { Link } from "react-router";
import PageMeta from "../../components/common/PageMeta";
import { Prompt, TerminalWindow } from "../../components/terminal/TerminalUI";

const META_CHECKS = [
  "14 districts monitored // IST (UTC+5:30)",
  "6-hour forecast lead time",
  "TCDL coupled model v2 // online",
];

export default function Landing() {
  return (
    <>
      <PageMeta
        title="Kerala Flood–Landslide Early Warning System"
        description="Coupled flood and landslide early warning for Kerala: rainfall-driven hazard probabilities, TCDL warning status and lead-time analysis in one dashboard."
      />

      <TerminalWindow
        path="~/dev"
        maxWidthClass="max-w-[1120px]"
        status={
          <span className="flex items-center gap-2 text-[13px] text-[#ffd24a]">
            <span className="crt-blip h-2 w-2 rounded-full bg-[#ffd24a] shadow-[0_0_8px_rgba(255,210,74,0.8)]" />
            3 active warnings
          </span>
        }
      >
        <section className="px-5 py-10 sm:px-9 sm:py-14">
          <Prompt command="whoami --full" cwd="~/dev" />

          <h1 className="crt-name mt-6">
            KERALA EWS<span className="text-[#39ff7a]">_</span>
          </h1>

          <p className="mt-5 text-[15px] sm:text-[17px]">
            <span className="text-[#5f8d68]">&gt;</span>{" "}
            <span className="text-[#39ff7a] crt-glow">
              Coupled flood &amp; landslide early warning, one rainfall event at
              a time
            </span>
          </p>

          <p className="mt-6 max-w-[640px] text-[14px] leading-relaxed text-[#5f8d68] sm:text-[15px]">
            One storm, two hazards. The system reads rainfall, soil moisture and
            slope for every monitored location in Kerala, then couples{" "}
            <span className="text-[#39ff7a]">flood probability</span> with{" "}
            <span className="text-[#39ff7a]">landslide probability</span> instead
            of scoring them apart. What comes out is a{" "}
            <span className="text-[#39ff7a]">TCDL warning status</span> you can
            act on, hours before the slope moves.
          </p>

          <ul className="mt-7 flex flex-wrap gap-x-8 gap-y-3 text-[13px]">
            {META_CHECKS.map((check) => (
              <li key={check} className="text-[#5f8d68]">
                <span className="text-[#39ff7a] crt-glow-soft">[x]</span> {check}
              </li>
            ))}
          </ul>

          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link
              to="/early-warning"
              className="crt-btn-solid rounded-[5px] px-6 py-3 text-[14px] font-medium"
            >
              <span className="text-[#39ff7a]">$</span> ls ~/dashboard{" "}
              <span className="text-[#39ff7a]">-&gt;</span>
            </Link>
            <Link
              to="/warnings"
              className="crt-btn-ghost rounded-[5px] px-6 py-3 text-[14px]"
            >
              ./warnings --live
            </Link>
          </div>

          <p className="mt-8 text-[13px] text-[#1c7a3c]">
            <span className="text-[#5cf6ff]">ews</span>
            <span className="text-[#5f8d68]">~/dev</span>{" "}
            <span className="text-[#39ff7a] crt-glow-soft">$</span>{" "}
            <span className="crt-caret align-middle" />
          </p>
        </section>

        <div className="crt-rule" />

        <footer className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 text-[12px] sm:px-9">
          <p className="text-[#5f8d68]">
            <span className="text-[#39ff7a] crt-glow-soft">$</span> echo
            &quot;Kerala Flood&ndash;Landslide Early Warning System //
            capstone&quot;
          </p>
          <p className="text-[#1c7a3c]">
            model: tcdl@v2 &middot; districts: 14 &middot; uptime 99.98%
          </p>
        </footer>
      </TerminalWindow>
    </>
  );
}
