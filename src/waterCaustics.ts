import {
  Application,
  Color,
  Container,
  Mesh,
  MeshGeometry,
  RenderTexture,
  Shader,
  Texture,
  UniformGroup,
} from "pixi.js";

/**
 * Water caustics + refraction as a full-screen pass, adapted from the GLSL
 * sandbox shader by David Hoskins (original water turbulence by joltz0r), made
 * tileable.
 *
 * Each frame the complete underwater scene is rendered to a texture, then one
 * screen-space quad applies the caustic light and refraction to the composite.
 *
 * Optional overlay mode keeps the rendered layer's alpha when compositing over
 * an existing background.
 *
 * The Shadertoy `mainImage()` body was ported to PixiJS v8: `iTime` becomes the
 * `u_time` uniform, and `fragCoord / iResolution` becomes `v_uv` (0..1 over the
 * quad).
 */

export interface SandAppearance {
  color: number;
  /** Pixi 4×5 colour matrices, applied after the caustic light. */
  contrast: number[];
  levels: number[];
  fineNoise: number;
  coarseNoise: number;
  fineSeed: number;
  coarseSeed: number;
}

export interface CausticsOptions {
  sand: SandAppearance;
  /** Caustic light strength. */
  strength?: number;
  /** Refraction (bend) amount. Share it between passes to keep them aligned. */
  bend?: number;
  /** Water-column colour washed over the layer. */
  tint?: number;
  /** How much of `tint` to mix into the layer (0..1). */
  tintAmount?: number;
}

function createQuadGeometry(): MeshGeometry {
  return new MeshGeometry({
    positions: new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  });
}

const VERTEX_SRC = /* glsl */ `#version 300 es
in vec2 aPosition;
out vec2 v_uv;
void main() {
  v_uv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_scene;   // rocks, shadows, and fish over transparency
uniform float u_time;
uniform float u_strength;     // caustic light strength
uniform float u_bend;         // refraction (bend) amount
uniform vec3  u_tint;         // water-column colour
uniform float u_tintAmount;   // how much of it to mix into the layer
uniform vec3 u_sand;
uniform vec4 u_contrastR;
uniform vec4 u_contrastG;
uniform vec4 u_contrastB;
uniform vec3 u_contrastBias;
uniform vec4 u_levelsR;
uniform vec4 u_levelsG;
uniform vec4 u_levelsB;
uniform vec3 u_levelsBias;
uniform vec4 u_grain; // fine amount, coarse amount, fine seed, coarse seed

#define TAU 6.28318530718
#define MAX_ITER 3

vec3 grade(vec3 color, vec4 r, vec4 g, vec4 b, vec3 bias) {
  vec4 c = vec4(color, 1.0);
  return clamp(vec3(dot(r, c), dot(g, c), dot(b, c)) + bias, 0.0, 1.0);
}

// Same static grain as Pixi's NoiseFilter. It stays in screen pixels and is
// applied last, so refraction sampling cannot smooth away the sand texture.
float grain(float seed) {
  return fract(sin(dot(gl_FragCoord.xy * seed, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
}

void main() {
  float time = u_time * 0.16 + 23.0;
  vec2 p = mod(v_uv * TAU, TAU) - 250.0;
  vec2 i = p;
  float c = 1.0;
  float inten = 0.005;
  for (int n = 0; n < MAX_ITER; n++) {
    float t = time * (1.0 - (3.5 / float(n + 1)));
    i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
    c += 1.0 / length(vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten)));
  }
  c = 1.17 - pow(c / float(MAX_ITER), 1.4);
  float light = pow(abs(c), 3.0) * u_strength;
  vec2 bend = u_bend * vec2(
    sin(i.x + time) + cos(i.y + time),
    cos(i.x + time) + sin(i.y + time)
  );

  // Restore the original sand order: water wash, caustics, contrast/saturation,
  // levels, then the two grain passes. Use the same light for every material.
  vec3 sand = clamp(mix(u_sand, u_tint, u_tintAmount) + vec3(light), 0.0, 1.0);
  sand = grade(sand, u_contrastR, u_contrastG, u_contrastB, u_contrastBias);
  sand = grade(sand, u_levelsR, u_levelsG, u_levelsB, u_levelsBias);
  sand = clamp(sand + grain(u_grain.z) * u_grain.x, 0.0, 1.0);
  sand = clamp(sand + grain(u_grain.w) * u_grain.y, 0.0, 1.0);

  // The object texture is premultiplied. Its alpha lets the sand's grade and
  // grain remain visible around the objects without grading their colours.
  vec4 scene = texture(u_scene, v_uv + bend);
  vec3 layer = scene.a > 0.0 ? scene.rgb / scene.a : vec3(0.0);
  layer = mix(layer, u_tint, u_tintAmount * scene.a);

  vec3 objects = clamp(layer + vec3(light), 0.0, 1.0) * scene.a;
  outColor = vec4(objects + sand * (1.0 - scene.a), 1.0);
}`;

export class WaterCaustics {
  /** Full-screen result mesh; add it to the stage. */
  readonly result: Mesh<MeshGeometry, Shader>;

  private readonly app: Application;
  private readonly container: Container;
  private readonly uniforms: UniformGroup;
  private readonly shader: Shader;
  private sceneTex: RenderTexture;
  private w = 0;
  private h = 0;

  constructor(
    app: Application,
    container: Container,
    options: CausticsOptions,
  ) {
    this.app = app;
    this.container = container;
    const sand = options.sand;

    this.uniforms = new UniformGroup({
      u_time: { value: 0, type: "f32" },
      u_strength: { value: options.strength ?? 1.0, type: "f32" },
      u_bend: { value: options.bend ?? 0.005, type: "f32" },
      u_tint: {
        value: new Color(options.tint ?? 0x000000).toRgbArray(),
        type: "vec3<f32>",
      },
      u_tintAmount: { value: options.tintAmount ?? 0, type: "f32" },
      u_sand: { value: new Color(sand.color).toRgbArray(), type: "vec3<f32>" },
      u_contrastR: { value: sand.contrast.slice(0, 4), type: "vec4<f32>" },
      u_contrastG: { value: sand.contrast.slice(5, 9), type: "vec4<f32>" },
      u_contrastB: { value: sand.contrast.slice(10, 14), type: "vec4<f32>" },
      u_contrastBias: {
        value: [sand.contrast[4], sand.contrast[9], sand.contrast[14]],
        type: "vec3<f32>",
      },
      u_levelsR: { value: sand.levels.slice(0, 4), type: "vec4<f32>" },
      u_levelsG: { value: sand.levels.slice(5, 9), type: "vec4<f32>" },
      u_levelsB: { value: sand.levels.slice(10, 14), type: "vec4<f32>" },
      u_levelsBias: {
        value: [sand.levels[4], sand.levels[9], sand.levels[14]],
        type: "vec3<f32>",
      },
      u_grain: {
        value: [
          sand.fineNoise,
          sand.coarseNoise,
          sand.fineSeed,
          sand.coarseSeed,
        ],
        type: "vec4<f32>",
      },
    });

    this.shader = Shader.from({
      gl: {
        vertex: VERTEX_SRC,
        fragment: FRAG_SRC,
        preferredFragmentPrecision: "highp",
      },
      resources: {
        u_scene: Texture.EMPTY.source,
        uniforms: this.uniforms,
      },
    });

    this.result = new Mesh({
      geometry: createQuadGeometry(),
      shader: this.shader,
    });
    // The quad is expressed in clip space, so its AABB is tiny and would be
    // culled; disable culling for this full-screen pass.
    this.result.cullable = false;

    this.sceneTex = RenderTexture.create({ width: 1, height: 1 });
  }

  /** Render the scene and update the shader time. */
  update(time: number): void {
    const w = this.app.screen.width;
    const h = this.app.screen.height;

    if (w !== this.w || h !== this.h) {
      this.w = w;
      this.h = h;
      this.sceneTex.destroy(true);
      this.sceneTex = RenderTexture.create({ width: w, height: h });
      this.shader.resources.u_scene = this.sceneTex.source;
    }

    this.uniforms.uniforms.u_time = time;

    this.app.renderer.render({
      container: this.container,
      target: this.sceneTex,
      clear: true,
      clearColor: [0, 0, 0, 0],
    });
  }
}
