/**
 * Minimal weather / hazard line glyphs used purely for decoration.
 *
 * These are visual accents only -- they carry no data and change no behaviour.
 * All use currentColor so they inherit whatever text colour they sit in.
 */

type IconProps = { className?: string; size?: number };

export function DropletIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 3.2c3.2 3.9 5.6 7 5.6 9.9A5.6 5.6 0 0 1 12 18.7a5.6 5.6 0 0 1-5.6-5.6c0-2.9 2.4-6 5.6-9.9Z" />
    </svg>
  );
}

export function RainCloudIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M7 15.5A4 4 0 0 1 7.3 7.6a5 5 0 0 1 9.5 1.2A3.4 3.4 0 0 1 16.5 15.5H7Z" />
      <path d="M8 18.5 7 20.5M12 18.5 11 20.5M16 18.5 15 20.5" />
    </svg>
  );
}

export function MountainIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M3 19.5 9.5 8l4 6.5 2-3 5 8.5H3Z" />
      <path d="m8 12.5 1.5-2.5" />
    </svg>
  );
}

export function WaveIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M2 9c2 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 5-2" />
      <path d="M2 14c2 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 5-2" />
    </svg>
  );
}

export function ShieldIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 3 5 5.6v5.2c0 4.3 2.9 7.6 7 9.2 4.1-1.6 7-4.9 7-9.2V5.6L12 3Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

export function BellIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M6 9a6 6 0 0 1 12 0c0 5 2 6.5 2 6.5H4S6 14 6 9Z" />
      <path d="M10.2 19a2 2 0 0 0 3.6 0" />
    </svg>
  );
}

/**
 * Minimal radar / signal-pulse mark: a centre point with arcs radiating
 * outward. Reads as active monitoring rather than depicting weather, so it
 * suits a hazard-monitoring header without competing with the content.
 */
export function RadarPulseIcon({ className, size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      className={className}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="1.75" fill="currentColor" stroke="none" />
      <path d="M8.4 15.6a5.1 5.1 0 0 1 0-7.2" />
      <path d="M15.6 8.4a5.1 5.1 0 0 1 0 7.2" />
      <path d="M5.6 18.4a9.1 9.1 0 0 1 0-12.8" />
      <path d="M18.4 5.6a9.1 9.1 0 0 1 0 12.8" />
    </svg>
  );
}

/**
 * A few faint falling raindrops for the hero banner. Staggered so they
 * don't pulse in unison; respects prefers-reduced-motion via CSS.
 */
export function RainMotif({ className = "" }: { className?: string }) {
  const drops = [
    { left: "8%", delay: "0s", h: 14 },
    { left: "17%", delay: "0.7s", h: 10 },
    { left: "27%", delay: "0.3s", h: 16 },
    { left: "38%", delay: "1.1s", h: 12 },
    { left: "52%", delay: "0.5s", h: 14 },
    { left: "63%", delay: "0.15s", h: 10 },
    { left: "74%", delay: "0.9s", h: 16 },
    { left: "86%", delay: "0.45s", h: 12 },
  ];
  return (
    <div
      className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}
      aria-hidden="true"
    >
      {drops.map((d, i) => (
        <span
          key={i}
          className="ews-drop absolute top-2 block rounded-full bg-white/70"
          style={{ left: d.left, width: 1.5, height: d.h, animationDelay: d.delay }}
        />
      ))}
    </div>
  );
}
