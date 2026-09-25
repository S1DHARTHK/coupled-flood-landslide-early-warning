/**
 * KeralaMap -- reusable Kerala hazard map.
 *
 * Renders real Kerala district boundaries from GeoJSON (OpenStreetMap via
 * Open Data Kerala, ODbL) and places hazard markers from data passed in as
 * props. It hard-codes no ML result and computes no hazard: every value shown
 * comes from the `locations` prop, which the caller obtains from the backend.
 *
 * DISTRICT HANDLING
 * -----------------
 * District names shown on the map belong to the BOUNDARY POLYGONS (cartography
 * from OSM). A marker displays a district only if the backend supplied
 * `district` for that location -- the real pipeline does (one representative
 * point per district), the synthetic one does not. Nothing is inferred from
 * coordinates.
 *
 * Usage:  <KeralaMap locations={locations} />
 */

import { useEffect, useMemo, useRef } from "react";
import { GeoJSON, MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import "leaflet/dist/leaflet.css";

import keralaDistricts from "../../data/Kerala_districts.geojson";
import type { HazardLevel, MapLocation } from "../../types/hazard";
import { LEVEL_META, formatCoord, pct } from "../hazard/hazardUtils";

/** Kerala's approximate bounding box, used as the initial view. */
const KERALA_CENTER: [number, number] = [10.6, 76.3];
const KERALA_BOUNDS: [[number, number], [number, number]] = [
  [8.05, 74.7],
  [12.9, 77.6],
];

interface DistrictProps {
  district: string;
  name_ml?: string | null;
  wikidata?: string | null;
}

export interface KeralaMapProps {
  locations: MapLocation[];
  /** Which hazard drives the marker colour. */
  colorBy?: "coupled" | "flood" | "landslide";
  height?: number | string;
  showLegend?: boolean;
  onSelect?: (location: MapLocation) => void;
  selectedId?: string | null;
}

/** Circular marker whose colour encodes hazard level. */
function hazardIcon(level: HazardLevel, selected: boolean): L.DivIcon {
  const { hex } = LEVEL_META[level];
  const size = selected ? 26 : 20;
  return L.divIcon({
    className: "kerala-hazard-marker",
    html: `<span style="
        display:block;width:${size}px;height:${size}px;border-radius:9999px;
        background:${hex};border:2.5px solid #fff;
        box-shadow:0 0 0 2px ${hex}55, 0 2px 6px rgba(16,24,40,.35);
      "></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  });
}

/** Fits the view to Kerala plus every marker, so nothing sits off-screen. */
function FitToData({ locations }: { locations: MapLocation[] }) {
  const map = useMap();
  const done = useRef(false);
  useEffect(() => {
    if (done.current) return;
    const bounds = L.latLngBounds(KERALA_BOUNDS);
    locations.forEach((l) => bounds.extend([l.latitude, l.longitude]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [24, 24] });
      done.current = true;
    }
  }, [map, locations]);
  return null;
}

export default function KeralaMap({
  locations,
  colorBy = "coupled",
  height = 520,
  showLegend = true,
  onSelect,
  selectedId = null,
}: KeralaMapProps) {
  const districtStyle = useMemo(
    () => ({
      color: "#7d89b0",
      weight: 1,
      fillColor: "#98a2b3",
      fillOpacity: 0.08,
      opacity: 0.75,
    }),
    []
  );

  const onEachDistrict = (feature: Feature<Geometry, DistrictProps>, layer: L.Layer) => {
    const name = feature.properties?.district;
    if (!name) return;
    layer.bindTooltip(name, { sticky: true, direction: "auto", className: "text-xs" });
    layer.on({
      mouseover: (e) => (e.target as L.Path).setStyle({ fillOpacity: 0.22, weight: 2 }),
      mouseout: (e) => (e.target as L.Path).setStyle(districtStyle),
    });
  };

  const levelFor = (l: MapLocation): HazardLevel => {
    if (colorBy === "coupled") return l.level;
    const p = colorBy === "flood" ? l.floodProbability : l.landslideProbability;
    if (p === null) return "normal";
    if (p >= 0.75) return "critical";
    if (p >= 0.5) return "warning";
    if (p >= 0.3) return "watch";
    return "normal";
  };

  return (
    <div className="relative">
      <div
        className="overflow-hidden rounded-xl border border-gray-200 dark:border-gray-800"
        style={{ height }}
      >
        <MapContainer
          center={KERALA_CENTER}
          zoom={7}
          scrollWheelZoom
          style={{ height: "100%", width: "100%", background: "transparent" }}
          className="z-0"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <GeoJSON
            data={keralaDistricts as unknown as FeatureCollection}
            style={districtStyle}
            onEachFeature={onEachDistrict as never}
          />
          <FitToData locations={locations} />

          {locations.map((loc) => (
            <Marker
              key={loc.location_id}
              position={[loc.latitude, loc.longitude]}
              icon={hazardIcon(levelFor(loc), selectedId === loc.location_id)}
              eventHandlers={{ click: () => onSelect?.(loc) }}
            >
              <Popup minWidth={248}>
                <div className="space-y-2 text-[13px] leading-tight">
                  <div>
                    {/* District appears ONLY if the backend supplied it. */}
                    <p className="font-semibold text-gray-800">
                      {loc.district ? `${loc.district} district` : "Monitoring Point"}
                    </p>
                    <p className="text-gray-600">
                      Latitude: {loc.latitude.toFixed(3)}
                    </p>
                    <p className="text-gray-600">
                      Longitude: {loc.longitude.toFixed(3)}
                    </p>
                    {loc.date ? (
                      <p className="text-gray-500">As of {loc.date}</p>
                    ) : null}
                  </div>

                  <div className="grid grid-cols-3 gap-2 border-t border-gray-200 pt-2">
                    <div>
                      <p className="text-[11px] text-gray-500">Flood</p>
                      <p className="font-semibold text-gray-800">
                        {pct(loc.floodProbability, 0)}
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] text-gray-500">Landslide</p>
                      <p className="font-semibold text-gray-800">
                        {pct(loc.landslideProbability, 0)}
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] text-gray-500">Coupled</p>
                      <p className="font-semibold text-gray-800">
                        {pct(loc.coupledProbability, 0)}
                      </p>
                    </div>
                  </div>

                  <div className="border-t border-gray-200 pt-2">
                    <p className="text-[11px] text-gray-500">TCDL Status</p>
                    <p
                      className="font-semibold"
                      style={{ color: LEVEL_META[loc.level].hex }}
                    >
                      {loc.warningStatus.toUpperCase()}
                    </p>
                    <p className="text-[11px] text-gray-500 mt-1">Warning Type</p>
                    <p className="font-medium text-gray-800">
                      {loc.warningType.toUpperCase()}
                    </p>
                  </div>

                  {loc.triggeredRules.length > 0 && (
                    <div className="border-t border-gray-200 pt-2">
                      <p className="text-[11px] text-gray-500">Triggered Rules</p>
                      <ul className="mt-1 space-y-0.5">
                        {loc.triggeredRules.map((r) => (
                          <li key={r} className="font-mono text-[11px] text-gray-700">
                            • {r}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {showLegend && (
        <div className="pointer-events-none absolute bottom-3 left-3 z-[500] rounded-lg border border-gray-200 bg-white/95 px-3 py-2 shadow-sm backdrop-blur dark:border-gray-700 dark:bg-gray-900/95">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Hazard level
          </p>
          <div className="flex flex-col gap-1">
            {(Object.keys(LEVEL_META) as HazardLevel[]).map((lvl) => (
              <div key={lvl} className="flex items-center gap-2">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: LEVEL_META[lvl].hex }}
                />
                <span className="text-[11px] text-gray-600 dark:text-gray-300">
                  {LEVEL_META[lvl].label}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export { formatCoord };
