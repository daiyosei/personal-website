import { defineConfig } from "vite";
import { minify } from "html-minifier-terser";

export default defineConfig({
  plugins: [
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
