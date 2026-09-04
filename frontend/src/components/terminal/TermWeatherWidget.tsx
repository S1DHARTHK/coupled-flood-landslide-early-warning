/**
 * Compact live-weather widget for the dashboard header.
 *
 * IMPORTANT — this is the one panel on the dashboard showing REAL data. Every
 * other number on the page is a synthetic model output. This widget reads live
 * observed weather from the free Open-Meteo API (no key, no backend proxy) for
 * a fixed Kerala location, and is labelled as such so the two can never be
 * confused. It is display only: nothing here feeds the models or TCDL.
 *
 * V1 scope: one fixed location, today's conditions, and a collapsed 5-day
 * forecast. No location picker and no geolocation.
 */

import { useEffect, useState } from "react";
import { TERM } from "./termColors";

/** Fixed first-version location: Kochi, central Kerala coast. */
const SITE = {
  label: "Kochi, Kerala",
  latitude: 9.9312,
  longitude: 76.2673,
};

const ENDPOINT =
  `https://api.open-meteo.com/v1/forecast` +
  `?latitude=${SITE.latitude}&longitude=${SITE.longitude}` +
  `&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m` +
  `&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum` +
  `&timezone=Asia%2FKolkata&forecast_days=5`;

interface OpenMeteoResponse {
  current: {
    time: string;
    temperature_2m: number;
    relative_humidity_2m: number;
    precipitation: number;
    weather_code: number;
    wind_speed_10m: number;
  };
  daily: {
    time: string[];
    weather_code: number[];
    temperature_2m_max: number[];
    temperature_2m_min: number[];
    precipitation_sum: number[];
  };
}

/** WMO weather codes, shortened to fit a compact console row. */
function condition(code: number): string {
  if (code === 0) return "clear";
  if (code === 1) return "mainly clear";
  if (code === 2) return "partly cloudy";
  if (code === 3) return "overcast";
  if (code === 45 || code === 48) return "fog";
  if (code >= 51 && code <= 57) return "drizzle";
  if (code >= 61 && code <= 67) return "rain";
  if (code >= 71 && code <= 77) return "snow";
  if (code >= 80 && code <= 82) return "rain showers";
  if (code >= 85 && code <= 86) return "snow showers";
  if (code === 95) return "thunderstorm";
  if (code >= 96) return "thunderstorm, hail";
  return `code ${code}`;
}

/** Three-letter weekday, or "today" for the first forecast row. */
function dayLabel(iso: string, index: number): string {
  if (index === 0) return "today";
  return new Date(`${iso}T00:00:00`)
    .toLocaleDateString("en-GB", { weekday: "short" })
    .toLowerCase();
}

export default function TermWeatherWidget() {
  const [data, setData] = useState<OpenMeteoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(ENDPOINT)
      .then((r) => {
        if (!r.ok) throw new Error(`open-meteo ${r.status}`);
        return r.json() as Promise<OpenMeteoResponse>;
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "unreachable");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const c = data?.current;

  return (
    <div className="crt-panel w-full shrink-0 p-3 sm:w-[272px]">
      {/* Header: location + the live/real-data marker */}
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
          {SITE.label}
        </p>
        <span className="text-[9px] uppercase tracking-wide text-[#1c7a3c]">
          live · open-meteo
        </span>
      </div>

      {error ? (
        <p className="mt-3 text-[11px] text-[#5f8d68]">
          <span style={{ color: TERM.amber }}>[!]</span> weather unavailable (
          {error})
        </p>
      ) : !c ? (
        <p className="mt-3 text-[11px] text-[#2bbf5c]">
          fetching weather… <span className="crt-caret align-middle" />
        </p>
      ) : (
        <>
          {/* Today */}
          <div className="mt-2 flex items-end justify-between gap-3">
            <p
              className="text-[28px] font-bold leading-none text-[#eafff1]"
              style={{ textShadow: "0 0 12px rgba(57,255,122,0.35)" }}
            >
              {c.temperature_2m.toFixed(1)}
              <span className="text-[15px] text-[#5f8d68]">°C</span>
            </p>
            <p className="pb-1 text-right text-[11px] text-[#39ff7a] crt-glow-soft">
              {condition(c.weather_code)}
            </p>
          </div>

          <dl className="mt-3 space-y-1 text-[11px]">
            {[
              ["humidity", `${Math.round(c.relative_humidity_2m)}%`, false],
              ["wind", `${c.wind_speed_10m.toFixed(0)} km/h`, false],
              [
                "precip",
                `${c.precipitation.toFixed(1)} mm`,
                c.precipitation > 0,
              ],
            ].map(([k, v, wet]) => (
              <div key={String(k)} className="flex justify-between">
                <dt className="text-[#5f8d68]">{k}</dt>
                <dd style={{ color: wet ? TERM.amber : TERM.ink }}>{String(v)}</dd>
              </div>
            ))}
          </dl>

          {/* Forecast toggle */}
          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="crt-chip mt-3 w-full px-2 py-1 text-[11px]"
          >
            forecast {open ? "[-]" : "[+]"}
          </button>

          {open && (
            <div className="mt-2 space-y-1 border-t border-[#0f2a12] pt-2">
              {data.daily.time.map((day, i) => {
                const wet = data.daily.precipitation_sum[i] > 0;
                return (
                  <div
                    key={day}
                    className="flex items-baseline justify-between gap-2 text-[11px]"
                  >
                    <span className="w-[42px] shrink-0 text-[#5f8d68]">
                      {dayLabel(day, i)}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[10px] text-[#3d6b47]">
                      {condition(data.daily.weather_code[i])}
                    </span>
                    <span className="shrink-0 tabular-nums text-[#cfe9d5]">
                      {Math.round(data.daily.temperature_2m_max[i])}°/
                      {Math.round(data.daily.temperature_2m_min[i])}°
                    </span>
                    <span
                      className="w-[46px] shrink-0 text-right tabular-nums text-[10px]"
                      style={{ color: wet ? TERM.amber : "#1c3a22" }}
                    >
                      {data.daily.precipitation_sum[i].toFixed(1)}mm
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
