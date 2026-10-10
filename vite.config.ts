import { defineConfig } from "vite";
import { minify } from "html-minifier-terser";
import { cpSync } from "node:fs";

export default defineConfig({
  publicDir: false,
  plugins: [
    {
      name: "copy-public-directory",
      apply: "build",
      closeBundle() {
        cpSync("public", "dist/public", { recursive: true });
      },
    },
    {
      name: "strip-html-comments",
      apply: "build",
      transformIndexHtml: {
        order: "post",
        async handler(html) {
          return minify(html, {
            collapseWhitespace: true,
            removeComments: true,
            minifyCSS: true,
            minifyJS: true,
          });
        },
      },
    },
  ],
  build: {
    minify: "esbuild",
    cssMinify: "esbuild",
  },
  server: {
    port: 8080,
  },
});
