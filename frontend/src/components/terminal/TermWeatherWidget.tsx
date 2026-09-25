/**
 * Compact historical-weather widget for the dashboard header.
 *
 * Shows the observed weather for the day the dashboard is replaying (the
 * "as of" date), read from Open-Meteo's free Historical Weather API (no key,
 * no backend proxy) for a fixed Kerala location. It is display only: nothing
 * here feeds the models or TCDL.
 *
 * V1 scope: one fixed location, one day (daily mean temperature, feels-like
 * temperature and the day's WMO weather code). No location picker.
 */

import { useEffect, useState } from "react";
import { TERM } from "./termColors";

/** Fixed first-version location: Kochi, central Kerala coast. */
const SITE = {
  label: "Kochi, Kerala",
  latitude: 9.9312,
  longitude: 76.2673,
};

/** Historical (reanalysis) daily values for one YYYY-MM-DD day. */
function endpoint(date: string): string {
  return (
    `https://archive-api.open-meteo.com/v1/archive` +
    `?latitude=${SITE.latitude}&longitude=${SITE.longitude}` +
    `&start_date=${date}&end_date=${date}` +
    `&daily=weather_code,temperature_2m_mean,apparent_temperature_mean` +
    `&timezone=Asia%2FKolkata`
  );
}

interface OpenMeteoArchiveResponse {
  daily: {
    time: string[];
    weather_code: (number | null)[];
    temperature_2m_mean: (number | null)[];
    apparent_temperature_mean: (number | null)[];
  };
}

/** WMO weather code -> icon and short label for a compact console row. */
function condition(code: number): { icon: string; label: string } {
  if (code === 0) return { icon: "☀️", label: "clear" };
  if (code === 1) return { icon: "🌤️", label: "mainly clear" };
  if (code === 2) return { icon: "⛅", label: "partly cloudy" };
  if (code === 3) return { icon: "☁️", label: "overcast" };
  if (code === 45 || code === 48) return { icon: "🌫️", label: "fog" };
  if (code >= 51 && code <= 57) return { icon: "🌦️", label: "drizzle" };
  if (code >= 61 && code <= 67) return { icon: "🌧️", label: "rain" };
  if (code >= 71 && code <= 77) return { icon: "❄️", label: "snow" };
  if (code >= 80 && code <= 82) return { icon: "🌧️", label: "rain showers" };
  if (code >= 85 && code <= 86) return { icon: "🌨️", label: "snow showers" };
  if (code === 95) return { icon: "⛈️", label: "thunderstorm" };
  if (code >= 96) return { icon: "⛈️", label: "thunderstorm, hail" };
  return { icon: "🌡️", label: `code ${code}` };
}

/** YYYY-MM-DD -> DD/MM/YYYY for display. */
function displayDate(iso: string): string {
  const [y, m, d] = iso.split("-");
  return `${d}/${m}/${y}`;
}

interface DayWeather {
  code: number;
  temp: number;
  feelsLike: number;
}

export default function TermWeatherWidget({
  date,
}: {
  /** The replay ("as of") day shown by the dashboard, YYYY-MM-DD. */
  date?: string | null;
}) {
  const [data, setData] = useState<DayWeather | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!date) return;
    let cancelled = false;
    setData(null);
    setError(null);
    fetch(endpoint(date))
      .then((r) => {
        if (!r.ok) throw new Error(`open-meteo ${r.status}`);
        return r.json() as Promise<OpenMeteoArchiveResponse>;
      })
      .then((json) => {
        if (cancelled) return;
        const code = json.daily?.weather_code?.[0];
        const temp = json.daily?.temperature_2m_mean?.[0];
        const feelsLike = json.daily?.apparent_temperature_mean?.[0];
        if (code == null || temp == null || feelsLike == null) {
          setError("no data for this date");
          return;
        }
        setData({ code, temp, feelsLike });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "unreachable");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [date]);

  const c = data ? condition(data.code) : null;

  return (
    <div className="crt-panel w-full shrink-0 p-3 sm:w-[272px]">
      {/* Header: location + the historical date shown */}
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
          {SITE.label}
        </p>
        <span className="text-[9px] uppercase tracking-wide text-[#1c7a3c]">
          {date ? displayDate(date) : "—"}
        </span>
      </div>

      {!date ? (
        <p className="mt-3 text-[11px] text-[#2bbf5c]">
          waiting for date… <span className="crt-caret align-middle" />
        </p>
      ) : error ? (
        <p className="mt-3 text-[11px] text-[#5f8d68]">
          <span style={{ color: TERM.amber }}>[!]</span> weather unavailable (
          {error})
        </p>
      ) : !data || !c ? (
        <p className="mt-3 text-[11px] text-[#2bbf5c]">
          fetching weather… <span className="crt-caret align-middle" />
        </p>
      ) : (
        <div className="mt-2 flex items-end justify-between gap-3">
          <div className="flex items-end gap-2">
            <span className="text-[26px] leading-none" role="img" aria-label={c.label}>
              {c.icon}
            </span>
            <p
              className="text-[28px] font-bold leading-none text-[#eafff1]"
              style={{ textShadow: "0 0 12px rgba(57,255,122,0.35)" }}
            >
              {data.temp.toFixed(1)}
              <span className="text-[15px] text-[#5f8d68]">°C</span>
            </p>
          </div>
          <div className="pb-1 text-right text-[11px]">
            <p className="text-[#39ff7a] crt-glow-soft">{c.label}</p>
            <p className="text-[#5f8d68]">
              feels like{" "}
              <span style={{ color: TERM.ink }}>{data.feelsLike.toFixed(1)}°C</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
