/**
 * Allow importing .geojson files as typed JSON modules.
 * Handled at build time by the `geojson-loader` plugin in vite.config.ts.
 */
declare module "*.geojson" {
  import type { FeatureCollection } from "geojson";
  const value: FeatureCollection;
  export default value;
}
