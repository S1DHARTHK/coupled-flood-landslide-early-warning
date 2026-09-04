import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";
import { readFileSync } from "node:fs";

/**
 * Vite's built-in JSON handling only covers the `.json` extension, so a
 * `.geojson` import would otherwise be treated as a static asset URL.
 * This transforms .geojson files into JSON modules (used by KeralaMap).
 */
function geojsonLoader(): Plugin {
  return {
    name: "geojson-loader",
    enforce: "pre",
    transform(_code, id) {
      if (!id.endsWith(".geojson")) return null;
      const json = readFileSync(id.split("?")[0], "utf-8");
      return {
        code: `export default ${json};`,
        map: null,
      };
    },
  };
}

// https://vite.dev/config/
// https://vite.dev/config/
export default defineConfig({
  plugins: [
    geojsonLoader(),
    react(),
    svgr({
      svgrOptions: {
        icon: true,
        // This will transform your SVG to a React component
        exportType: "named",
        namedExport: "ReactComponent",
      },
    }),
  ],

  server: {
    // Honour PORT when the environment assigns one (e.g. a preview runner),
    // otherwise fall back to Vite's usual 5173.
    port: Number(process.env.PORT) || 5173,
    allowedHosts: ['.trycloudflare.com'],
  },
});