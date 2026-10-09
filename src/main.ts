import {
  Application,
  Assets,
  BlurFilter,
  ColorMatrixFilter,
  Container,
  Mesh,
  MeshGeometry,
  Sprite,
  Texture,
} from "pixi.js";
import { WaterCaustics, type SandAppearance } from "./waterCaustics";

// Spacing: big steps over the first 2/3 of the image (tail), small over the last 1/3 (body)
function pointX(
  i: number,
  n: number,
  W: number,
  splitX = 2 / 3,
  splitT = 0.4,
): number {
  const t = i / (n - 1);
  if (t < splitT) return (t / splitT) * splitX * W;
  return (splitX + ((t - splitT) / (1 - splitT)) * (1 - splitX)) * W;
}

// Shortest signed angle from `current` to `target` (in [-π, π])
function angleDiff(target: number, current: number): number {
  return Math.atan2(Math.sin(target - current), Math.cos(target - current));
}

// Rotate `from` toward `to` by at most `maxDelta` radians
function lerpAngle(from: number, to: number, maxDelta: number): number {
  const diff = angleDiff(to, from);
  if (Math.abs(diff) <= maxDelta) return to;
  return from + Math.sign(diff) * maxDelta;
}

/** Stable 32-bit PRNG used only for repeatable pond decoration placement. */
function createSeededRandom(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Fish geometry ---
// Cross-sections along the body. Shared by every fish; the live spine is stored
// in this many points and skinned as a triangle strip.
const SPINE_SLICES = 40;

// The tail fin ends ~30.5% along the texture (the peduncle waist sits at
// x≈78/256), so the wag taper is anchored there. Measured off the alpha channel,
// this matches the fish textures.
const TAIL_WAG_FRAC = 0.305;

/**
 * A fish texture plus everything derived purely from its pixel dimensions: where
 * each spine cross-section rests along the body, and the UVs/indices that map the
 * ribbon onto it. Fish sharing a skin share these buffers; only the live spine
 * positions are per-fish state.
 */
class FishSkin {
  readonly texture: Texture;
  readonly width: number;
  readonly height: number;
  readonly restX: Float32Array;
  readonly restLengths: Float32Array;
  readonly wagU: Float32Array;
  readonly wagTaper: Float32Array;
  readonly uvs: Float32Array;
  readonly indices: Uint32Array;

  constructor(texture: Texture) {
    this.texture = texture;
    this.width = texture.width;
    this.height = texture.height;

    this.restX = new Float32Array(SPINE_SLICES);
    this.restLengths = new Float32Array(SPINE_SLICES);
    this.wagU = new Float32Array(SPINE_SLICES);
    this.wagTaper = new Float32Array(SPINE_SLICES);
    for (let i = 0; i < SPINE_SLICES; i++) {
      this.restX[i] = pointX(i, SPINE_SLICES, this.width);
      this.wagU[i] = this.restX[i] / this.width;
      const t = this.wagU[i] / TAIL_WAG_FRAC;
      this.wagTaper[i] = this.wagU[i] < TAIL_WAG_FRAC ? 1 - t * t * (3 - 2 * t) : 0;
      if (i > 0) this.restLengths[i] = this.restX[i] - this.restX[i - 1];
    }

    this.uvs = new Float32Array(SPINE_SLICES * 4);
    this.indices = new Uint32Array((SPINE_SLICES - 1) * 6);
    for (let i = 0; i < SPINE_SLICES; i++) {
      this.uvs.set(
        [this.restX[i] / this.width, 0, this.restX[i] / this.width, 1],
        i * 4,
      );
    }
    for (let i = 0; i < SPINE_SLICES - 1; i++) {
      const a = i * 2;
      this.indices.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], i * 6);
    }
  }
}

// --- Pond planting ---
const ROCK_COUNT = 7;
const LILYPAD_COUNT = 14;
const LOTUS_COUNT = 8;

// --- Look tunables ---
const SAND_COLOR = 0x213626; // original sand colour before water wash and grade
const WATER_COLOR = 0x0a3520; // deep sea-green water column
const WATER_ALPHA = 0.3;

// Refraction (UV "bend") strength per pass. Kept deliberately light: the fish
// and sand should read as still shapes with only a faint shimmer, not a rolling
// wobble. The fish get a touch more than the floor since their flat colour makes
// the same amount imperceptible. Both passes sample the same turbulence field,
// so the two stay in step.
const FLOOR_BEND = 0.002;

// Fish shadow: a soft Gaussian-blurred copy of each fish, cast onto the sand and
// rocks below. It lives in its own layer between the rocks and the fish, so the
// fish always draw over their shadow.
//
// Tinted dark red rather than black purely to make it obvious while tuning the
// offset/blur; set to 0x000000 for the final look.
// const SHADOW_COLOR = 0xff0000;
const SHADOW_COLOR = 0x000000;
const SHADOW_ALPHA = 0.5;
const SHADOW_OFFSET_X = 20;
const SHADOW_OFFSET_Y = 10;
const SHADOW_BLUR = 5;

/**
 * Final grade applied on top of the caustic composite: two contrast passes and
 * two static grain passes. The grain seeds are fixed, so the noise never
 * crawls — it reads as fine sand grain over the whole frame rather than as
 * flickering static.
 */
function createSandAppearance(): SandAppearance {
  // Widen the range around mid grey and push saturation, so the darker water
  // greens stay rich instead of turning to mud.
  const contrast = new ColorMatrixFilter();
  contrast.contrast(0.4, false);
  contrast.saturate(0.3, true);

  // Second, gentler pass with a lower pivot: crushes the blacks and stretches
  // the midtones again, which is what really sells the contrast.
  const levels = new ColorMatrixFilter();
  levels.brightness(0.8, false);
  levels.contrast(0.25, true);

  // Two unrelated static grain fields stacked: they clump into coarse specks
  // instead of the even hiss a single noise pass gives you.
  const appearance: SandAppearance = {
    color: SAND_COLOR,
    contrast: [...contrast.matrix],
    levels: [...levels.matrix],
    fineNoise: 0.5,
    coarseNoise: 0.45,
    fineSeed: 0.731,
    coarseSeed: 2.194,
  };
  contrast.destroy();
  levels.destroy();
  return appearance;
}

interface ScatterOptions {
  /** Smallest and largest on-screen size, in px. */
  minSize: number;
  maxSize: number;
  /** Symmetric rotation limit, in radians. */
  maxRotation: number;
  /**
   * Extra clear space to keep between neighbours, in px. 0 allows overlap, which
   * is what lily pads want — they grow in clumps and read as one mass. Rocks pass
   * a positive gap so they never touch.
   */
  minGap: number;
}

/** One decoration variant: a texture plus the colour correction it needs. */
interface Variant {
  texture: Texture;
  /**
   * Multiplied into the texture colour, to push a decoration away from whatever
   * it sits on. Needed because the rock textures vary in brightness against the
   * value range against the sand.
   */
  tint?: number;
  /**
   * Radius of the visible silhouette as a fraction of the sprite size, used for
   * spacing. Defaults to 0.5 (half the box). The rock textures are transparent
   * around the edges, so they need a smaller value than the default to avoid
   * being spaced as if they filled their whole box.
   */
  radiusFrac?: number;
}

/** Relaxation passes used to push overlapping sprites apart. */
const SEPARATION_PASSES = 24;

/**
 * A scattered field of pond decorations (rocks, lily pads, lotus flowers).
 *
 * Placements are stored in normalized screen space and re-projected on resize,
 * so shrinking the window moves the decoration instead of re-scattering it. Each
 * sprite keeps a fixed pixel size rather than scaling with the viewport, which
 * keeps them in proportion with the fish (those are sized in px too).
 */
class Scatter {
  private readonly items: {
    sprite: Sprite;
    nx: number;
    ny: number;
    size: number;
    radiusFrac: number;
  }[] = [];
  private readonly minGap: number;
  // Nearby placement is an initial scatter operation. Re-running it during
  // resize would consume new random values and move every lotus independently
  // of its normalized position.
  private nearbyPlacementDone = false;

  constructor(
    layer: Container,
    variants: Variant[],
    count: number,
    options: ScatterOptions,
    random: () => number,
  ) {
    this.minGap = options.minGap;

    for (let i = 0; i < count; i++) {
      const variant = variants[Math.floor(random() * variants.length)];
      const sprite = new Sprite(variant.texture);
      // Anchor at the centre so rotation spins about the middle of the shape and
      // placement is by centre rather than by corner.
      sprite.anchor.set(0.5);
      sprite.rotation = (random() * 2 - 1) * options.maxRotation;
      if (variant.tint !== undefined) sprite.tint = variant.tint;

      const size =
        options.minSize + random() * (options.maxSize - options.minSize);
      sprite.width = size;
      sprite.height = size;

      this.items.push({
        sprite,
        nx: random(),
        ny: random(),
        size,
        radiusFrac: variant.radiusFrac ?? 0.5,
      });
      layer.addChild(sprite);
    }
  }

  /** Re-project every sprite onto the current viewport. */
  layout(
    width: number,
    height: number,
    near?: Scatter,
    random?: () => number,
  ): void {
    const minDim = Math.min(width, height);
    for (const item of this.items) {
      // Never let a decoration exceed the viewport.
      const size = Math.min(item.size, minDim);
      // Allow a quarter of it to hang off each edge: a pond is a window onto
      // something bigger, so decorations bleeding out of frame look natural.
      const bleed = size * 0.25;
      item.sprite.width = size;
      item.sprite.height = size;
      item.sprite.x = clamp(item.nx * width, -bleed, width + bleed);
      item.sprite.y = clamp(item.ny * height, -bleed, height + bleed);
    }

    if (this.minGap > 0) this.separate(minDim);
    if (near && random && !this.nearbyPlacementDone) {
      this.placeNear(width, height, near, random);
      this.nearbyPlacementDone = true;
    }
  }

  private placementCircles(
    minDim: number,
  ): { x: number; y: number; r: number }[] {
    return this.items.map((item) => ({
      x: item.sprite.x,
      y: item.sprite.y,
      r: this.radiusOf(item, minDim),
    }));
  }

  /** Place flowers around pads while keeping every visible silhouette apart. */
  private placeNear(
    width: number,
    height: number,
    near: Scatter,
    random: () => number,
  ): void {
    const minDim = Math.min(width, height);
    const occupied = near.placementCircles(minDim);
    const placed: { x: number; y: number; r: number }[] = [];

    for (const item of this.items) {
      const radius = this.radiusOf(item, minDim);
      let point: { x: number; y: number } | undefined;

      for (let attempt = 0; attempt < 2000 && !point; attempt++) {
        const anchor = occupied[Math.floor(random() * occupied.length)];
        const angle = random() * Math.PI * 2;
        const distance = anchor.r + radius + 8 + random() * 36;
        const x = anchor.x + Math.cos(angle) * distance;
        const y = anchor.y + Math.sin(angle) * distance;
        if (
          x < radius ||
          x > width - radius ||
          y < radius ||
          y > height - radius
        ) {
          continue;
        }

        const collides = [...occupied, ...placed].some(
          (other) => Math.hypot(x - other.x, y - other.y) < radius + other.r,
        );
        if (!collides) point = { x, y };
      }

      // Keep placement valid even on unusually small viewports with little free
      // space around the pads; the normal sized pond finds a nearby slot above.
      if (!point) {
        for (let attempt = 0; attempt < 4000 && !point; attempt++) {
          const x = radius + random() * Math.max(0, width - radius * 2);
          const y = radius + random() * Math.max(0, height - radius * 2);
          const collides = [...occupied, ...placed].some(
            (other) => Math.hypot(x - other.x, y - other.y) < radius + other.r,
          );
          if (!collides) point = { x, y };
        }
      }

      if (!point) point = { x: width / 2, y: height / 2 };
      item.nx = point.x / width;
      item.ny = point.y / height;
      item.sprite.position.set(point.x, point.y);
      placed.push({ ...point, r: radius });
    }
  }

  /**
   * Spacing radius of an item at the current viewport, i.e. half the clear space
   * it needs around its centre. Recomputed from the item's own fraction every
   * time rather than accumulated, so repeated resizes are stable.
   */
  private radiusOf(
    item: { size: number; radiusFrac: number },
    minDim: number,
  ): number {
    return Math.min(item.size, minDim) * item.radiusFrac + this.minGap / 2;
  }

  /**
   * Push overlapping sprites apart until none of them touch.
   *
   * Runs on every layout, not once at scatter time: the placements are stored in
   * normalized space, so a window resize can pull two rocks together again, and
   * this is what keeps the no-overlap guarantee true at every size.
   *
   * This is a few iterations of pairwise relaxation rather than a solver. It is
   * deterministic given the same positions, cheap at these counts, and good
   * enough for a decorative scatter.
   */
  private separate(minDim: number): void {
    const items = this.items;
    for (let pass = 0; pass < SEPARATION_PASSES; pass++) {
      let moved = false;

      for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
          const a = items[i];
          const b = items[j];
          const minDist = this.radiusOf(a, minDim) + this.radiusOf(b, minDim);

          let dx = b.sprite.x - a.sprite.x;
          let dy = b.sprite.y - a.sprite.y;
          let d = Math.hypot(dx, dy);

          if (d >= minDist) continue;

          // Exactly coincident centres give no direction to separate along, so
          // nudge along a deterministic axis instead of dividing by zero.
          if (d < 1e-4) {
            dx = 1;
            dy = 0;
            d = 1;
          }

          const push = (minDist - d) / 2;
          const ux = (dx / d) * push;
          const uy = (dy / d) * push;
          a.sprite.x -= ux;
          a.sprite.y -= uy;
          b.sprite.x += ux;
          b.sprite.y += uy;
          moved = true;
        }
      }

      if (!moved) break;
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

(async () => {
  const app = new Application();
  const pondElement = document.getElementById("pixi-container");
  if (!pondElement) throw new Error("Missing #pixi-container element");
  const isPhone =
    window.matchMedia("(max-width: 767px) and (pointer: coarse)").matches;
  const renderResolution = isPhone ? 1 : window.devicePixelRatio || 1;
  await app.init({
    background: "#000814",
    resizeTo: pondElement,
    // Reduce high-DPI backing-store cost on phones; keep native resolution elsewhere.
    resolution: renderResolution,
  });
  app.ticker.maxFPS = isPhone ? 24 : 60;
  pondElement.appendChild(app.canvas);
  const placementSeed = 1922;
  const placementRandom = createSeededRandom(placementSeed);

  // The pond is rendered in three layers so the grade can sit between them: the
  // floor takes the full grain/contrast treatment, the fish swim above it clean,
  // and the surface plants float on top of everything. The floor and the fish
  // both go through the same refraction pass.
  const rockLayer = new Container();
  const shadowLayer = new Container();
  const fishLayer = new Container();
  const underwaterLayer = new Container();
  const surfaceLayer = new Container();

  // Load every texture up front so nothing pops in mid-swim.
  const [
    rock1,
    rock2,
    rock3,
    lilypad,
    lilypad2,
    lotus,
    lotus2,
    ...fishTextures
  ] = await Promise.all([
    Assets.load<Texture>("/assets/rock1.png"),
    Assets.load<Texture>("/assets/rock2.png"),
    Assets.load<Texture>("/assets/rock3.png"),
    Assets.load<Texture>("/assets/lilypad.png"),
    Assets.load<Texture>("/assets/lilypad2.png"),
    Assets.load<Texture>("/assets/lotus.png"),
    Assets.load<Texture>("/assets/lotus2.png"),
    Assets.load<Texture>("/assets/fish1.png"),
    Assets.load<Texture>("/assets/fish2.png"),
    Assets.load<Texture>("/assets/fish3.png"),
    Assets.load<Texture>("/assets/fish4.png"),
  ]);

  // One skin per fish texture; each fish picks one at random.
  const skins = fishTextures.map((t) => new FishSkin(t));

  // Rocks rest on the bottom, so they go in their own layer above the graded
  // sand (the grain/contrast grade would otherwise be painted over them), but
  // still below the fish. They get their own refraction pass so they still
  // wobble with the water.
  //
  // The rocks need colour corrections to read against the sand. Rock1 and
  // rock3 are grayscale, while rock2 is a light tan; tint them down to sit
  // clearly below the floor's value.
  const rocks = new Scatter(
    rockLayer,
    [
      // The silhouette fills ~60% of its 256px box (measured off the alpha
      // channel), so space them by that rather than by the full box, otherwise
      // they end up further apart than they look.
      { texture: rock1, tint: 0x9aa39c, radiusFrac: 0.3 },
      { texture: rock2, tint: 0x6f7a63, radiusFrac: 0.29 },
      { texture: rock3, tint: 0x778477, radiusFrac: 0.3 },
    ],
    ROCK_COUNT,
    {
      minSize: 70,
      maxSize: 150,
      // Rocks are lit from a consistent direction in the texture, so only nudge
      // them off-axis instead of spinning them fully.
      maxRotation: 0.25,
      // Rocks are solid and opaque: keep them from touching or overlapping.
      minGap: 24,
    },
    placementRandom,
  );
  rocks.layout(app.screen.width, app.screen.height);

  // Lily pads and lotus flowers float on the surface, so they sit above the fish
  // and are drawn straight to the stage: crisp and static, with none of the
  // refraction or caustic light that the submerged layers get.
  // The pads and flowers are drawn raw, with none of the grade, so they are
  // knocked back toward the water colour a little — enough to sit in the same
  // pond as the graded floor instead of looking like stickers on top of it.
  const lilypads = new Scatter(
    surfaceLayer,
    // Each pad carries its own radiusFrac, measured off the alpha channel:
    // lilypad fills ~97% of its 256px box, lilypad2 only ~72% (it has a baked-in
    // drop shadow and transparent margins), so they need different spacing.
    [
      { texture: lilypad, tint: 0x8fb894, radiusFrac: 0.46 },
      { texture: lilypad2, tint: 0x8fb894, radiusFrac: 0.34 },
    ],
    LILYPAD_COUNT,
    {
      minSize: 95,
      maxSize: 185,
      maxRotation: Math.PI, // the pad's wedge notch makes full rotation natural
      minGap: 18, // keep the pads from overlapping each other
    },
    placementRandom,
  );
  const lotuses = new Scatter(
    surfaceLayer,
    [
      { texture: lotus, tint: 0xd6aeb2 },
      { texture: lotus2, tint: 0xd6aeb2 },
    ],
    LOTUS_COUNT,
    {
      minSize: 55,
      maxSize: 145,
      maxRotation: Math.PI, // radially symmetric; rotation is just variety
      minGap: 0,
    },
    placementRandom,
  );
  lilypads.layout(app.screen.width, app.screen.height);
  lotuses.layout(
    app.screen.width,
    app.screen.height,
    lilypads,
    placementRandom,
  );

  // Tunables
  const fishCount = 15;
  const onScreenLen = 120; // target fish length in screen px (any texture size)
  const wagAmpFrac = 0.08; // peak wag displacement, as a fraction of texture width
  const wagFreq = 6; // radians per unit of u
  const wagSpeed = 3;
  const wallMargin = 90; // how close to an edge before steering away

  class Fish {
    readonly mesh: Mesh;
    readonly shadow: Mesh;
    readonly geometry: MeshGeometry;
    readonly scale: number;
    readonly halfWidth: number;
    readonly turnRate: number;
    readonly speed: number;
    readonly wagPhase: number;
    readonly wagAmp: number;
    readonly wagScale: number;
    readonly segmentLengths: Float32Array;
    readonly spineX = new Float32Array(SPINE_SLICES);
    readonly spineY = new Float32Array(SPINE_SLICES);
    readonly directionX = new Float32Array(SPINE_SLICES);
    readonly directionY = new Float32Array(SPINE_SLICES);
    heading: number;
    targetX = 0;
    targetY = 0;

    constructor(
      private readonly skin: FishSkin,
      width: number,
      height: number,
    ) {
      const { restX } = skin;
      this.scale = (onScreenLen / skin.width) * (0.65 + Math.random() * 0.7); // ~78..162px long on screen
      this.halfWidth = (skin.height / 2) * this.scale;
      this.turnRate = 1.5 + Math.random() * 1.5; // rad/s
      this.speed = 40 + Math.random() * 60; // px/s
      this.wagPhase = Math.random() * Math.PI * 2;
      this.wagAmp = skin.width * wagAmpFrac;
      this.wagScale = Math.min(this.speed / 60, 1.4);
      this.segmentLengths = new Float32Array(SPINE_SLICES);
      for (let i = 1; i < SPINE_SLICES; i++) {
        this.segmentLengths[i] = skin.restLengths[i] * this.scale;
      }
      this.heading = Math.random() * Math.PI * 2;

      // Start somewhere in the tank with the body laid out straight behind the head
      const startX = wallMargin + Math.random() * (width - wallMargin * 2);
      const startY = wallMargin + Math.random() * (height - wallMargin * 2);
      const hx = Math.cos(this.heading);
      const hy = Math.sin(this.heading);
      for (let i = 0; i < SPINE_SLICES; i++) {
        const back = (skin.width - restX[i]) * this.scale;
        this.spineX[i] = startX - hx * back;
        this.spineY[i] = startY - hy * back;
      }
      this.pickTarget(width, height);

      const positions = new Float32Array(SPINE_SLICES * 4);
      for (let i = 0; i < SPINE_SLICES; i++) {
        positions.set(
          [
            restX[i] * this.scale,
            -this.halfWidth,
            restX[i] * this.scale,
            this.halfWidth,
          ],
          i * 4,
        );
      }
      this.geometry = new MeshGeometry({
        positions,
        uvs: skin.uvs,
        indices: skin.indices,
      });
      this.mesh = new Mesh({
        geometry: this.geometry,
        texture: skin.texture,
      });

      // Soft drop shadow: a darkened copy of the same deforming ribbon, offset a
      // little. It shares the fish's geometry, so it wags in lockstep with the
      // body with no extra per-frame work. The blur is applied once, at the
      // shadow layer level, not per fish.
      this.shadow = new Mesh({
        geometry: this.geometry,
        texture: skin.texture,
      });
      this.shadow.tint = SHADOW_COLOR;
      this.shadow.alpha = SHADOW_ALPHA;
      this.shadow.position.set(SHADOW_OFFSET_X, SHADOW_OFFSET_Y);
    }

    pickTarget(width: number, height: number) {
      this.targetX = wallMargin + Math.random() * (width - wallMargin * 2);
      this.targetY = wallMargin + Math.random() * (height - wallMargin * 2);
    }

    update(dt: number, time: number, width: number, height: number) {
      const last = SPINE_SLICES - 1;
      const headX = this.spineX[last];
      const headY = this.spineY[last];

      // Seek the current target
      let sdx = this.targetX - headX;
      let sdy = this.targetY - headY;
      const seekDist = Math.hypot(sdx, sdy) || 1;
      sdx /= seekDist;
      sdy /= seekDist;

      // Soft wall repulsion (steer away from the edges)
      let ax = 0;
      let ay = 0;
      if (headX < wallMargin) ax += 1 - headX / wallMargin;
      if (headX > width - wallMargin) ax -= 1 - (width - headX) / wallMargin;
      if (headY < wallMargin) ay += 1 - headY / wallMargin;
      if (headY > height - wallMargin) ay -= 1 - (height - headY) / wallMargin;

      const desired = Math.atan2(sdy + ay * 2, sdx + ax * 2);
      this.heading = lerpAngle(this.heading, desired, this.turnRate * dt);

      // Pick a fresh destination once we get close
      if (seekDist < 60) this.pickTarget(width, height);

      // Swim forward (the head leads)
      this.spineX[last] += Math.cos(this.heading) * this.speed * dt;
      this.spineY[last] += Math.sin(this.heading) * this.speed * dt;

      // Follow-the-leader: each slice trails the one in front at a fixed length
      for (let i = last - 1; i >= 0; i--) {
        const tx = this.spineX[i + 1] - this.spineX[i];
        const ty = this.spineY[i + 1] - this.spineY[i];
        const d = Math.hypot(tx, ty) || 1;
        const dx = tx / d;
        const dy = ty / d;
        this.directionX[i] = dx;
        this.directionY[i] = dy;
        const seg = this.segmentLengths[i + 1];
        this.spineX[i] = this.spineX[i + 1] - dx * seg;
        this.spineY[i] = this.spineY[i + 1] - dy * seg;
      }
      this.directionX[last] = this.directionX[last - 1];
      this.directionY[last] = this.directionY[last - 1];

      // Write the ribbon into the vertex buffer
      const data = this.geometry.positions;
      for (let i = 0; i < SPINE_SLICES; i++) {
        const tx = this.directionX[i];
        const ty = this.directionY[i];
        const ux = ty; // perpendicular ("up")
        const uy = -tx;

        // Tail wag: a traveling wave over the tail fin, tapering to zero at
        // the peduncle so the fin blends continuously into the static body
        let sway = 0;
        const u = this.skin.wagU[i];
        if (this.skin.wagTaper[i] !== 0) {
          sway =
            Math.sin(u * wagFreq + time * wagSpeed + this.wagPhase) *
            this.wagAmp *
            this.scale *
            this.skin.wagTaper[i] *
            this.wagScale;
        }

        const cx = this.spineX[i] + ux * sway;
        const cy = this.spineY[i] + uy * sway;

        data[i * 4 + 0] = cx + ux * this.halfWidth;
        data[i * 4 + 1] = cy + uy * this.halfWidth;
        data[i * 4 + 2] = cx - ux * this.halfWidth;
        data[i * 4 + 3] = cy - uy * this.halfWidth;
      }

      this.geometry.getBuffer("aPosition").update();
    }
  }

  const fish: Fish[] = [];
  for (let i = 0; i < fishCount; i++) {
    const skin = skins[Math.floor(Math.random() * skins.length)];
    const f = new Fish(skin, app.screen.width, app.screen.height);
    fish.push(f);
    fishLayer.addChild(f.mesh);
    shadowLayer.addChild(f.shadow);
  }

  // The shader draws the sand with its original light → grade → grain order.
  // The transparent object texture preserves coverage over that textured sand.
  underwaterLayer.addChild(rockLayer);

  // Fish shadows sit above the rocks and below the fish. A single blur on the
  // whole layer gives every shadow the same soft Gaussian fall-off in one pass.
  shadowLayer.filters = [new BlurFilter({ strength: SHADOW_BLUR, quality: 4 })];
  shadowLayer.filterArea = app.screen;
  underwaterLayer.addChild(shadowLayer);

  underwaterLayer.addChild(fishLayer);

  // Apply one caustic and refraction pass to the completed underwater scene, so
  // sand, rocks, shadows, and fish share the same light pattern.
  const underwaterPass = new WaterCaustics(app, underwaterLayer, {
    sand: createSandAppearance(),
    strength: 0.7,
    bend: FLOOR_BEND,
    tint: WATER_COLOR,
    tintAmount: WATER_ALPHA,
  });
  const underwaterComposite = new Container();
  underwaterComposite.addChild(underwaterPass.result);
  underwaterComposite.filterArea = app.screen;
  app.stage.addChild(underwaterComposite);

  // Floating plants last: they sit on the water, above the fish.
  app.stage.addChild(surfaceLayer);

  // Exposed for the headless render-checks: lets a test read the live sprite
  // transforms out of the running scene instead of inferring them from pixels.
  // Note the floor is rendered to an offscreen texture rather than parented to
  // the stage, so `floorLayer` has to be handed over explicitly — walking
  // `app.stage` alone would never find the rocks.
  (
    window as unknown as {
      __pixiScene: {
        app: Application;
        rockLayer: Container;
        fishLayer: Container;
        surfaceLayer: Container;
        underwaterLayer: Container;
        underwaterPass: WaterCaustics;
      };
    }
  ).__pixiScene = {
    app,
    rockLayer,
    fishLayer,
    surfaceLayer,
    underwaterLayer,
    underwaterPass,
  };

  let lastW = app.screen.width;
  let lastH = app.screen.height;

  app.ticker.add((ticker) => {
    const dt = Math.min(ticker.deltaMS / 1000, 0.1); // seconds, clamped
    const time = performance.now() / 1000;

    if (app.screen.width !== lastW || app.screen.height !== lastH) {
      lastW = app.screen.width;
      lastH = app.screen.height;
      rocks.layout(lastW, lastH);
      lilypads.layout(lastW, lastH);
      lotuses.layout(lastW, lastH, lilypads, placementRandom);
      shadowLayer.filterArea = app.screen;
    }

    for (const f of fish) {
      f.update(dt, time, app.screen.width, app.screen.height);
    }

    underwaterPass.update(time);
  });
})();
