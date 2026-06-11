"""WorldForge 2: a Minecraft style world generator with a top down map view.

High quality renderer (smooth water, antialiased coasts, hillshading,
terrain texture), threaded rendering with noise caching for smooth
interaction, and export to a playable Minecraft Java world
(1.18.2 Anvil format, opens in any newer version including 1.20+).

Usage:
    python world_forge.py                      start the GUI
    python world_forge.py --selftest out.png   render one map headless
    python world_forge.py --export-test DIR    export a small world headless
    python world_forge.py --smoke              open GUI once, render, close
"""

import math
import os
import random
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageTk

F32 = np.float32

APP_VERSION = "2.1.0"
APP_AUTHOR = "Mohammed Alanzi"
# update this after creating the repository on GitHub
GITHUB_URL = "https://github.com/YOUR-USERNAME/WorldForge"
MC_FORMAT = "Java Edition, 1.18.2 Anvil format\n(opens in 1.20, 1.21 and newer)"

# ============================================================ NBT writer ===

_TAG_IDS = {
    "end": 0, "byte": 1, "short": 2, "int": 3, "long": 4, "float": 5,
    "double": 6, "byte_array": 7, "string": 8, "list": 9, "compound": 10,
    "int_array": 11, "long_array": 12,
}


def _w_payload(buf, tag):
    t = tag[0]
    if t == "byte":
        buf += struct.pack(">b", int(tag[1]))
    elif t == "short":
        buf += struct.pack(">h", int(tag[1]))
    elif t == "int":
        buf += struct.pack(">i", int(tag[1]))
    elif t == "long":
        buf += struct.pack(">q", int(tag[1]))
    elif t == "float":
        buf += struct.pack(">f", float(tag[1]))
    elif t == "double":
        buf += struct.pack(">d", float(tag[1]))
    elif t == "string":
        raw = tag[1].encode("utf-8")
        buf += struct.pack(">H", len(raw)) + raw
    elif t == "byte_array":
        raw = bytes(tag[1])
        buf += struct.pack(">i", len(raw)) + raw
    elif t == "int_array":
        arr = np.asarray(tag[1], dtype=np.int32)
        buf += struct.pack(">i", arr.size) + arr.byteswap().tobytes()
    elif t == "long_array":
        arr = np.asarray(tag[1], dtype=np.uint64)
        buf += struct.pack(">i", arr.size) + arr.byteswap().tobytes()
    elif t == "list":
        etype, items = tag[1], tag[2]
        buf += bytes([_TAG_IDS[etype]]) + struct.pack(">i", len(items))
        for it in items:
            _w_payload(buf, it)
    elif t == "compound":
        for name, sub in tag[1].items():
            buf += bytes([_TAG_IDS[sub[0]]])
            raw = name.encode("utf-8")
            buf += struct.pack(">H", len(raw)) + raw
            _w_payload(buf, sub)
        buf += b"\x00"
    else:
        raise ValueError(t)


def nbt_bytes(root_compound_dict):
    buf = bytearray(b"\x0a\x00\x00")  # unnamed root compound
    _w_payload(buf, ("compound", root_compound_dict))
    return bytes(buf)

# ================================================================= noise ===

_GRAD = np.array(
    [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=F32,
)


class Noise:
    """Seeded 2D Perlin noise with fractal stacking, float32 vectorized."""

    def __init__(self, seed):
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self.perm = np.array(p + p, dtype=np.int64)
        self.gx = _GRAD[self.perm[:512] & 7, 0].astype(F32)
        self.gy = _GRAD[self.perm[:512] & 7, 1].astype(F32)

    def perlin(self, x, y):
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        xf = (x - xi).astype(F32)
        yf = (y - yi).astype(F32)
        xi &= 255
        yi &= 255

        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)

        perm = self.perm
        h_aa = perm[xi] + yi
        h_ba = perm[xi + 1] + yi

        def dot(h, dx, dy):
            return self.gx[h] * dx + self.gy[h] * dy

        n00 = dot(h_aa, xf, yf)
        n10 = dot(h_ba, xf - 1, yf)
        n01 = dot(h_aa + 1, xf, yf - 1)
        n11 = dot(h_ba + 1, xf - 1, yf - 1)

        x1 = n00 + u * (n10 - n00)
        x2 = n01 + u * (n11 - n01)
        return (x1 + v * (x2 - x1)) * F32(1.4142)

    def fbm(self, x, y, octaves=4, persistence=0.5, lacunarity=2.0,
            min_wavelength=0.0):
        """min_wavelength (in noise units) drops octaves too fine to see
        at the current sampling step, killing aliasing shimmer."""
        total = np.zeros(np.shape(x), dtype=F32)
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            if min_wavelength and 1.0 / freq < min_wavelength:
                break
            total += F32(amp) * self.perlin(x * F32(freq), y * F32(freq))
            norm += amp
            amp *= persistence
            freq *= lacunarity
        if norm == 0:
            return total
        return total / F32(norm)

# ================================================================ biomes ===

BIOMES = {
    0: ("Deep Ocean", (13, 35, 94), "minecraft:deep_ocean"),
    1: ("Ocean", (28, 64, 148), "minecraft:ocean"),
    2: ("Shallow Ocean", (60, 125, 195), "minecraft:lukewarm_ocean"),
    3: ("Frozen Ocean", (150, 172, 205), "minecraft:frozen_ocean"),
    4: ("Beach", (228, 214, 158), "minecraft:beach"),
    5: ("Snowy Beach", (222, 220, 202), "minecraft:snowy_beach"),
    6: ("River", (62, 120, 190), "minecraft:river"),
    7: ("Frozen River", (182, 210, 235), "minecraft:frozen_river"),
    8: ("Swamp", (80, 98, 58), "minecraft:swamp"),
    9: ("Plains", (126, 168, 76), "minecraft:plains"),
    10: ("Forest", (56, 112, 48), "minecraft:forest"),
    11: ("Dark Forest", (38, 84, 36), "minecraft:dark_forest"),
    12: ("Jungle", (40, 124, 30), "minecraft:jungle"),
    13: ("Desert", (224, 203, 134), "minecraft:desert"),
    14: ("Savanna", (180, 166, 92), "minecraft:savanna"),
    15: ("Badlands", (202, 116, 64), "minecraft:badlands"),
    16: ("Taiga", (86, 124, 90), "minecraft:taiga"),
    17: ("Snowy Plains", (234, 238, 243), "minecraft:snowy_plains"),
    18: ("Snowy Taiga", (172, 192, 180), "minecraft:snowy_taiga"),
    19: ("Mountains", (140, 136, 130), "minecraft:windswept_hills"),
    20: ("Snowy Peaks", (244, 247, 251), "minecraft:frozen_peaks"),
}

PALETTE_F = np.zeros((len(BIOMES), 3), dtype=F32)
for _b, (_n, _rgb, _mc) in BIOMES.items():
    PALETTE_F[_b] = _rgb

# per biome terrain texture strength (canopy, dunes, rock)
TEXTURE_F = np.array(
    [0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.03, 0.04, 0.09, 0.06,
     0.13, 0.16, 0.15, 0.06, 0.07, 0.08, 0.12, 0.04, 0.10, 0.11, 0.05],
    dtype=F32)

WATER_IDS = frozenset((0, 1, 2, 3, 6, 7))
VILLAGE_BIOMES = frozenset((9, 13, 14, 16, 17))

STRUCTURE_STYLES = {
    "Village": "#c8893c",
    "Desert Temple": "#e8c33d",
    "Jungle Temple": "#4d9a3a",
    "Witch Hut": "#9a6fd0",
    "Ocean Monument": "#37d4d4",
    "Woodland Mansion": "#3a6b33",
}

DEFAULTS = {
    "sea_level": 0.50,
    "mountain_level": 0.78,
    "terrain_scale": 900.0,
    "detail": 5,
    "temperature": 0.0,
    "moisture": 0.0,
    "rivers": 0.012,
}

PRESETS = {
    "Islands": {"sea_level": 0.58, "mountain_level": 0.84,
                "terrain_scale": 520, "detail": 6, "temperature": 0.06,
                "moisture": 0.05, "rivers": 0.006},
    "Mountain World": {"sea_level": 0.42, "mountain_level": 0.62,
                       "terrain_scale": 1100, "detail": 7,
                       "temperature": -0.05, "moisture": 0.0,
                       "rivers": 0.014},
    "Frozen World": {"sea_level": 0.50, "mountain_level": 0.76,
                     "terrain_scale": 900, "detail": 5,
                     "temperature": -0.40, "moisture": 0.05,
                     "rivers": 0.012},
    "Desert World": {"sea_level": 0.44, "mountain_level": 0.80,
                     "terrain_scale": 950, "detail": 5,
                     "temperature": 0.38, "moisture": -0.38,
                     "rivers": 0.004},
}

STRUCTURE_REGION = 320
SNOW_COL = np.array((235, 239, 244), dtype=F32)
SAND_COL = np.array((228, 214, 158), dtype=F32)
ICE_COL = np.array((205, 224, 240), dtype=F32)
W_SHALLOW = np.array((90, 168, 210), dtype=F32)
W_MID = np.array((38, 92, 178), dtype=F32)
W_DEEP = np.array((12, 32, 92), dtype=F32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp_col(c0, c1, w):
    return c0 + (c1 - c0) * w[..., None]

# ================================================================= world ===

_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4)))


class World:
    def __init__(self, seed, params=None):
        self.seed = int(seed)
        self.params = dict(DEFAULTS)
        if params:
            self.params.update(params)
        self.n_cont = Noise(self.seed)
        self.n_det = Noise(self.seed + 101)
        self.n_temp = Noise(self.seed + 202)
        self.n_moist = Noise(self.seed + 303)
        self.n_river = Noise(self.seed + 404)
        self.n_tex = Noise(self.seed + 505)
        self._cache = {}  # field cache: key -> dict of arrays

    # ---- field generation (cached; climate shifts and sea level are
    # ---- applied later, so those sliders reuse this cache and feel instant)

    def fields(self, x0, z0, step, w, h):
        p = self.params
        key = (round(x0, 3), round(z0, 3), round(step, 5), w, h,
               p["terrain_scale"], int(p["detail"]))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        s = p["terrain_scale"]
        det_scale = s / 6.0
        octs = int(p["detail"])
        min_wl = step * 2.0

        xs = (x0 + np.arange(w, dtype=F32) * F32(step))
        zs = (z0 + np.arange(h, dtype=F32) * F32(step))

        elev = np.empty((h, w), dtype=F32)
        river = np.empty((h, w), dtype=F32)
        tex = np.empty((h, w), dtype=F32)

        # screen space texture frequency so surface detail looks the same
        # at every zoom level
        tex_scale = max(3.5, 3.5 * step * 0.9)
        rs = s * 0.45

        def band(i):
            z_lo, z_hi = edges[i], edges[i + 1]
            x, z = np.meshgrid(xs, zs[z_lo:z_hi])
            cont = self.n_cont.fbm(x / F32(s), z / F32(s), octaves=4,
                                   min_wavelength=min_wl / s)
            det = self.n_det.fbm(x / F32(det_scale), z / F32(det_scale),
                                 octaves=octs,
                                 min_wavelength=min_wl / det_scale)
            e = 0.5 + 0.5 * (F32(0.74) * cont + F32(0.26) * det)
            e = np.clip(e, 0.0, 1.0)
            elev[z_lo:z_hi] = e
            rv = np.abs(self.n_river.fbm(x / F32(rs), z / F32(rs), octaves=3,
                                         min_wavelength=min_wl / rs))
            river[z_lo:z_hi] = rv * (F32(0.6) + e)
            tex[z_lo:z_hi] = self.n_tex.perlin(x / F32(tex_scale),
                                               z / F32(tex_scale))

        n_bands = min(_POOL._max_workers, max(1, h // 64))
        edges = np.linspace(0, h, n_bands + 1).astype(int)
        list(_POOL.map(band, range(n_bands)))

        # climate is low frequency: compute at quarter resolution, upsample
        ts = s * 1.6
        qw, qh = max(2, w // 4 + 2), max(2, h // 4 + 2)
        qxs = x0 + np.arange(qw, dtype=F32) * F32(step * w / qw)
        qzs = z0 + np.arange(qh, dtype=F32) * F32(step * h / qh)
        qx, qz = np.meshgrid(qxs, qzs)
        t_raw = 0.5 + 0.5 * self.n_temp.fbm(qx / F32(ts), qz / F32(ts), 3)
        m_raw = 0.5 + 0.5 * self.n_moist.fbm(qx / F32(ts), qz / F32(ts), 3)

        def up(a):
            return np.asarray(
                Image.fromarray(a, "F").resize((w, h), Image.BILINEAR),
                dtype=F32)

        out = {"elev": elev, "river": river, "tex": tex,
               "temp_raw": up(t_raw.astype(F32)),
               "moist_raw": up(m_raw.astype(F32)), "step": step}
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[key] = out
        return out

    def climate(self, f):
        p = self.params
        temp = (f["temp_raw"] + F32(p["temperature"])
                - np.maximum(F32(0), f["elev"] - F32(0.62)) * F32(0.9))
        moist = np.clip(f["moist_raw"] + F32(p["moisture"]), 0.0, 1.0)
        return temp, moist

    def classify(self, f):
        p = self.params
        sea, mt = p["sea_level"], p["mountain_level"]
        elev, river = f["elev"], f["river"]
        temp, moist = self.climate(f)
        cold = temp < 0.24
        hot = temp > 0.72

        idx = np.full(elev.shape, 9, dtype=np.uint8)            # plains
        idx[moist > 0.60] = 10                                  # forest
        idx[moist > 0.80] = 11                                  # dark forest
        idx[(temp < 0.45) & (moist > 0.50)] = 16                # taiga
        idx[cold] = 17                                          # snowy plains
        idx[cold & (moist > 0.55)] = 18                         # snowy taiga
        idx[hot] = 14                                           # savanna
        idx[hot & (moist > 0.62)] = 12                          # jungle
        idx[hot & (moist < 0.30)] = 13                          # desert
        idx[hot & (moist < 0.12)] = 15                          # badlands
        idx[(elev < sea + 0.05) & (moist > 0.74) & ~cold & ~hot] = 8

        idx[elev > mt] = 19                                     # mountains
        idx[(elev > mt + 0.07) | ((elev > mt) & cold)] = 20     # snowy peaks

        riv = (river < p["rivers"]) & (elev >= sea) & (elev <= mt)
        idx[riv] = 6
        idx[riv & cold] = 7

        beach = (elev >= sea) & (elev < sea + 0.012)
        idx[beach] = 4
        idx[beach & cold] = 5

        idx[elev < sea] = 1
        idx[(elev < sea) & (elev > sea - 0.035)] = 2
        idx[elev < sea - 0.10] = 0
        idx[(elev < sea) & cold] = 3
        return idx

    # ---- high quality top view shading

    def shade(self, f, idx):
        p = self.params
        sea, mt = F32(p["sea_level"]), F32(p["mountain_level"])
        elev, river, tex, step = f["elev"], f["river"], f["tex"], f["step"]
        temp, _moist = self.climate(f)

        col = PALETTE_F[idx].copy()
        land = (elev >= sea).astype(F32)

        # land gets brighter with altitude
        rel = np.clip((elev - sea) / (1.001 - sea), 0.0, 1.0)
        col *= (F32(0.86) + F32(0.30) * rel * land)[..., None]

        # per biome surface texture (tree canopy, dunes, rock)
        col *= (F32(1.0) + tex * TEXTURE_F[idx])[..., None]

        # soft snow transition instead of a hard biome edge
        snow_w = smoothstep(0.27, 0.21, temp) * land
        col = lerp_col(col, SNOW_COL, snow_w * F32(0.85))

        # sandy shore fade just above the waterline (warm regions only)
        sand_w = (smoothstep(sea + 0.030, sea + 0.010, elev) * land
                  * smoothstep(0.26, 0.32, temp))
        col = lerp_col(col, SAND_COL, sand_w * F32(0.7))

        # water with a depth gradient
        depth = np.clip((sea - elev) / F32(0.22), 0.0, 1.0) ** F32(0.8)
        wcol = np.where(
            depth[..., None] < 0.5,
            lerp_col(np.broadcast_to(W_SHALLOW, elev.shape + (3,)), W_MID,
                     np.clip(depth * 2, 0, 1)),
            lerp_col(np.broadcast_to(W_MID, elev.shape + (3,)), W_DEEP,
                     np.clip(depth * 2 - 1, 0, 1)))
        ice_w = smoothstep(0.24, 0.16, temp) * (1 - land)
        wcol = lerp_col(wcol, ICE_COL, ice_w * F32(0.85))
        wcol = wcol * (F32(1.0) + tex * F32(0.04))[..., None]

        # antialiased coastline: blend over a thin elevation band
        band = float(np.clip(0.0011 * step, 0.0006, 0.012))
        aa = smoothstep(sea - band, sea + band, elev)
        col = lerp_col(wcol, col, aa)

        # rivers, antialiased the same way, frozen ones icy
        thr = max(p["rivers"], 1e-5)
        rband = thr * 0.35 + 0.0006 * step
        rw = smoothstep(thr + rband, thr - rband, river)
        rw = rw * aa * (elev <= mt)
        rcol = W_SHALLOW + (W_MID - W_SHALLOW) * F32(0.45)
        rcol = np.broadcast_to(rcol, elev.shape + (3,)).copy()
        rcol = lerp_col(rcol, ICE_COL,
                        smoothstep(0.24, 0.16, temp) * F32(0.9))
        rcol = rcol * (F32(1.0) + tex * F32(0.04))[..., None]
        col = lerp_col(col, rcol, rw)

        # hillshading: sun from the northwest
        gz, gx = np.gradient(elev)
        vs = F32(170.0 / max(step, 1e-6))
        nx, nz = -gx * vs, -gz * vs
        inv = 1.0 / np.sqrt(nx * nx + nz * nz + 1.0)
        dot = (nx * F32(-0.45) + nz * F32(-0.55) + F32(0.703)) * inv
        shade = F32(0.55) + F32(0.62) * np.clip(dot, 0.0, 1.0)
        shade = shade * land + (1 - land) * (F32(0.92) + F32(0.08) * shade)
        col *= np.clip(shade, 0.4, 1.45)[..., None]

        return np.clip(col, 0, 255).astype(np.uint8)

    def render(self, width, height, cx, cz, zoom, res=1.0):
        w = max(2, int(width * res))
        h = max(2, int(height * res))
        step = 1.0 / (zoom * res)
        x0 = cx - (w / 2.0) * step
        z0 = cz - (h / 2.0) * step
        f = self.fields(x0, z0, step, w, h)
        idx = self.classify(f)
        rgb = self.shade(f, idx)
        img = Image.fromarray(rgb, "RGB")
        if res != 1.0:
            img = img.resize((width, height), Image.BILINEAR)
        return img

    # ---- point queries

    def point_data(self, x, z):
        ax = np.full((1, 1), x, dtype=F32)
        az = np.full((1, 1), z, dtype=F32)
        s = self.params["terrain_scale"]
        cont = self.n_cont.fbm(ax / F32(s), az / F32(s), 4)
        det = self.n_det.fbm(ax / F32(s / 6), az / F32(s / 6),
                             int(self.params["detail"]))
        elev = np.clip(0.5 + 0.5 * (0.74 * cont + 0.26 * det), 0, 1)
        ts = s * 1.6
        t_raw = 0.5 + 0.5 * self.n_temp.fbm(ax / F32(ts), az / F32(ts), 3)
        m_raw = 0.5 + 0.5 * self.n_moist.fbm(ax / F32(ts), az / F32(ts), 3)
        rs = s * 0.45
        rv = np.abs(self.n_river.fbm(ax / F32(rs), az / F32(rs), 3))
        rv = rv * (F32(0.6) + elev)
        f = {"elev": elev, "river": rv, "temp_raw": t_raw,
             "moist_raw": m_raw, "tex": np.zeros_like(elev), "step": 1.0}
        bid = int(self.classify(f)[0, 0])
        y = self.surface_y(float(elev[0, 0]))
        return bid, y, float(elev[0, 0])

    def surface_y(self, elev):
        return int(np.clip(
            62 + (elev - self.params["sea_level"]) * 150, -52, 300))

    def biome_at(self, x, z):
        bid, y, _ = self.point_data(x, z)
        return BIOMES[bid][0], y, bid

    # ---- deterministic structures

    def structures_in_rect(self, x0, z0, x1, z1, limit=400):
        out = []
        r0x, r1x = int(x0 // STRUCTURE_REGION), int(x1 // STRUCTURE_REGION)
        r0z, r1z = int(z0 // STRUCTURE_REGION), int(z1 // STRUCTURE_REGION)
        if (r1x - r0x + 1) * (r1z - r0z + 1) > 4000:
            return out
        for rx in range(r0x, r1x + 1):
            for rz in range(r0z, r1z + 1):
                k = (rx * 73856093) ^ (rz * 19349663) ^ (self.seed * 83492791)
                rng = random.Random(k & 0xFFFFFFFF)
                if rng.random() > 0.55:
                    continue
                sx = rx * STRUCTURE_REGION + rng.randrange(STRUCTURE_REGION)
                sz = rz * STRUCTURE_REGION + rng.randrange(STRUCTURE_REGION)
                _, _, bid = self.biome_at(sx, sz)
                roll = rng.random()
                kind = None
                if bid in VILLAGE_BIOMES:
                    kind = "Village"
                    if bid == 13 and roll < 0.35:
                        kind = "Desert Temple"
                elif bid == 12 and roll < 0.45:
                    kind = "Jungle Temple"
                elif bid == 8 and roll < 0.45:
                    kind = "Witch Hut"
                elif bid == 0 and roll < 0.22:
                    kind = "Ocean Monument"
                elif bid == 11 and roll < 0.18:
                    kind = "Woodland Mansion"
                if kind:
                    out.append((sx, sz, kind))
                    if len(out) >= limit:
                        return out
        return out

# ===================================================== Minecraft export ===

DATA_VERSION = 2975  # 1.18.2; newer versions auto upgrade on first load

BLOCK_DEFS = [
    ("minecraft:air", None),                       # 0
    ("minecraft:bedrock", None),                   # 1
    ("minecraft:stone", None),                     # 2
    ("minecraft:dirt", None),                      # 3
    ("minecraft:grass_block", None),               # 4
    ("minecraft:sand", None),                      # 5
    ("minecraft:sandstone", None),                 # 6
    ("minecraft:gravel", None),                    # 7
    ("minecraft:water", None),                     # 8
    ("minecraft:snow_block", None),                # 9
    ("minecraft:snow", None),                      # 10 (layer)
    ("minecraft:ice", None),                       # 11
    ("minecraft:red_sand", None),                  # 12
    ("minecraft:terracotta", None),                # 13
    ("minecraft:oak_log", None),                   # 14
    ("minecraft:oak_leaves", {"persistent": "true", "distance": "7"}),
    ("minecraft:spruce_log", None),                # 16
    ("minecraft:spruce_leaves", {"persistent": "true", "distance": "7"}),
    ("minecraft:jungle_log", None),                # 18
    ("minecraft:jungle_leaves", {"persistent": "true", "distance": "7"}),
    ("minecraft:dark_oak_log", None),              # 20
    ("minecraft:dark_oak_leaves", {"persistent": "true", "distance": "7"}),
]

(B_AIR, B_BEDROCK, B_STONE, B_DIRT, B_GRASS, B_SAND, B_SANDSTONE,
 B_GRAVEL, B_WATER, B_SNOWBLK, B_SNOW, B_ICE, B_REDSAND, B_TERRA,
 B_OAKLOG, B_OAKLEAF, B_SPRLOG, B_SPRLEAF, B_JUNLOG, B_JUNLEAF,
 B_DOAKLOG, B_DOAKLEAF) = range(22)

_PALETTE_TAGS = []
for _name, _props in BLOCK_DEFS:
    _entry = {"Name": ("string", _name)}
    if _props:
        _entry["Properties"] = ("compound", {
            k: ("string", v) for k, v in _props.items()})
    _PALETTE_TAGS.append(("compound", _entry))

WORLD_BOTTOM = -64
SEA_Y = 62  # top water block


def _pack_bits(vals, bits):
    epl = 64 // bits
    v = np.asarray(vals, dtype=np.uint64)
    pad = (-len(v)) % epl
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.uint64)])
    v = v.reshape(-1, epl)
    shifts = np.arange(epl, dtype=np.uint64) * np.uint64(bits)
    return np.bitwise_or.reduce(v << shifts, axis=1)


# trees: trunk block, leaf block, min height, max height
_TREE_KINDS = {
    "oak": (B_OAKLOG, B_OAKLEAF, 4, 6),
    "spruce": (B_SPRLOG, B_SPRLEAF, 6, 9),
    "jungle": (B_JUNLOG, B_JUNLEAF, 7, 11),
    "dark_oak": (B_DOAKLOG, B_DOAKLEAF, 5, 7),
}
# biome id -> (tree kind, trees per chunk)
_TREES_PER_BIOME = {
    9: ("oak", 1), 10: ("oak", 7), 11: ("dark_oak", 10), 12: ("jungle", 8),
    14: ("oak", 1), 16: ("spruce", 6), 18: ("spruce", 4), 8: ("oak", 3),
}


def _paint_tree(arr, lx, lz, y0, kind, rng):
    trunk, leaf, hmin, hmax = _TREE_KINDS[kind]
    h = rng.randint(hmin, hmax)
    top_i = y0 + h - WORLD_BOTTOM
    if top_i + 3 >= arr.shape[0]:
        return
    if kind == "spruce":
        for dy in range(2, h + 1):
            r = min(2, max(0, (h - dy + 1) // 2))
            if r == 0 and dy < h:
                continue
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) == 2 and abs(dz) == 2:
                        continue
                    i = y0 + dy - WORLD_BOTTOM
                    if arr[i, lz + dz, lx + dx] == B_AIR:
                        arr[i, lz + dz, lx + dx] = leaf
        if arr[top_i + 1, lz, lx] == B_AIR:
            arr[top_i + 1, lz, lx] = leaf
    else:
        for dy in range(h - 2, h + 1):
            r = 2 if dy < h else 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) == 2 and abs(dz) == 2:
                        continue
                    i = y0 + dy - WORLD_BOTTOM
                    if arr[i, lz + dz, lx + dx] == B_AIR:
                        arr[i, lz + dz, lx + dx] = leaf
        for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if arr[top_i + 1, lz + dz, lx + dx] == B_AIR:
                arr[top_i + 1, lz + dz, lx + dx] = leaf
    for dy in range(h):
        arr[y0 + dy - WORLD_BOTTOM, lz, lx] = trunk


def build_chunk_nbt(world, cx, cz, y_surf, idx, temp, with_trees):
    """y_surf, idx, temp: 16x16 arrays [z][x] for this chunk."""
    cold = temp < 0.24

    y_max = int(max(int(y_surf.max()), SEA_Y)) + 14
    height = y_max - WORLD_BOTTOM + 1
    arr = np.zeros((height, 16, 16), dtype=np.uint8)

    ys = (np.arange(height, dtype=np.int32) + WORLD_BOTTOM)[:, None, None]

    water_ids = np.isin(idx, (0, 1, 2, 3))
    river_ids = np.isin(idx, (6, 7))
    is_water = (water_ids | river_ids)[None, :, :]

    # rivers carve their bed a little below sea level
    surf2d = np.where(river_ids, np.minimum(y_surf, SEA_Y - 3), y_surf)
    surf = surf2d[None, :, :]
    arr[ys <= surf] = B_STONE

    # surface layers by biome
    top = np.full((16, 16), B_GRASS, dtype=np.uint8)
    under = np.full((16, 16), B_DIRT, dtype=np.uint8)
    sandy = np.isin(idx, (4, 13)) | water_ids
    top[sandy] = B_SAND
    under[sandy] = B_SANDSTONE
    deep_floor = idx == 0
    top[deep_floor] = B_GRAVEL
    under[deep_floor] = B_GRAVEL
    bad = idx == 15
    top[bad] = B_REDSAND
    under[bad] = B_TERRA
    rocky = idx == 19
    top[rocky] = B_STONE
    under[rocky] = B_STONE
    peaks = idx == 20
    top[peaks] = B_SNOWBLK
    under[peaks] = B_SNOWBLK
    snowy_beach = idx == 5
    top[snowy_beach] = B_SAND
    under[snowy_beach] = B_SAND
    top[river_ids] = B_GRAVEL
    under[river_ids] = B_GRAVEL

    t_mask = ys == surf
    u_mask = (ys < surf) & (ys >= surf - 3)
    arr = np.where(t_mask, top[None, :, :], arr)
    arr = np.where(u_mask, under[None, :, :], arr)
    arr[0] = B_BEDROCK

    # fill water up to sea level
    w_mask = is_water & (ys > surf) & (ys <= SEA_Y)
    arr = np.where(w_mask, np.uint8(B_WATER), arr)
    # ice sheet on frozen water
    icy = is_water & cold[None, :, :] & (ys == SEA_Y)
    arr = np.where(icy & (arr == B_WATER), np.uint8(B_ICE), arr)

    # snow layer on cold land
    snowy_land = (~(water_ids | river_ids)) & cold & (surf2d >= SEA_Y)
    lay = (ys == surf + 1) & snowy_land[None, :, :]
    arr = np.where(lay & (arr == B_AIR), np.uint8(B_SNOW), arr)

    if with_trees:
        bid = int(np.bincount(idx.ravel()).argmax())
        if bid in _TREES_PER_BIOME:
            rng = random.Random(
                (world.seed * 341873128712 + cx * 132897987541
                 + cz * 341873128713) & 0xFFFFFFFFFFFF)
            kind, n = _TREES_PER_BIOME[bid]
            if n == 1 and rng.random() > 0.45:
                n = 0
            for _ in range(n):
                lx = rng.randint(3, 12)
                lz = rng.randint(3, 12)
                if int(idx[lz, lx]) in WATER_IDS or idx[lz, lx] in (4, 5):
                    continue
                ys0 = int(surf2d[lz, lx])
                if ys0 < SEA_Y:
                    continue
                if arr[ys0 - WORLD_BOTTOM, lz, lx] not in (B_GRASS, B_DIRT):
                    continue
                _paint_tree(arr, lx, lz, ys0 + 1, kind, rng)

    # ---- sections (24, covering y -64 .. 319)
    sections = []
    air_pal = ("compound", {
        "palette": ("list", "compound", [_PALETTE_TAGS[B_AIR]])})
    uniq_b = np.unique(idx[::4, ::4])
    for sy in range(-4, 20):
        lo = sy * 16 - WORLD_BOTTOM
        sec = {"Y": ("byte", sy)}
        if lo >= height:
            sec["block_states"] = air_pal
        else:
            hi = min(lo + 16, height)
            sl = arr[lo:hi]
            if hi - lo < 16:
                sl = np.concatenate(
                    [sl, np.zeros((16 - (hi - lo), 16, 16), np.uint8)])
            uniq = np.unique(sl)
            pal = ("list", "compound", [_PALETTE_TAGS[u] for u in uniq])
            if len(uniq) == 1:
                sec["block_states"] = ("compound", {"palette": pal})
            else:
                remap = np.searchsorted(uniq, sl.ravel())
                bits = max(4, int(len(uniq) - 1).bit_length())
                sec["block_states"] = ("compound", {
                    "palette": pal,
                    "data": ("long_array", _pack_bits(remap, bits)),
                })
        # biomes: 4x4x4 cells per section, column biome everywhere
        if len(uniq_b) == 1:
            sec["biomes"] = ("compound", {
                "palette": ("list", "string",
                            [("string", BIOMES[int(uniq_b[0])][2])])})
        else:
            remapb = np.searchsorted(uniq_b,
                                     np.tile(idx[::4, ::4].ravel(), 4))
            bbits = max(1, int(len(uniq_b) - 1).bit_length())
            sec["biomes"] = ("compound", {
                "palette": ("list", "string",
                            [("string", BIOMES[int(u)][2])
                             for u in uniq_b]),
                "data": ("long_array", _pack_bits(remapb, bbits)),
            })
        sections.append(("compound", sec))

    # heightmaps: stored value = highest non air block index + 1
    nonair = arr != B_AIR
    hm = (height - 1 - np.argmax(nonair[::-1], axis=0)) + 1
    hm_packed = _pack_bits(hm.ravel(), 9)

    root = {
        "DataVersion": ("int", DATA_VERSION),
        "xPos": ("int", cx),
        "zPos": ("int", cz),
        "yPos": ("int", -4),
        "Status": ("string", "full"),
        "LastUpdate": ("long", 0),
        "InhabitedTime": ("long", 0),
        "sections": ("list", "compound", sections),
        "Heightmaps": ("compound", {
            "MOTION_BLOCKING": ("long_array", hm_packed),
            "WORLD_SURFACE": ("long_array", hm_packed),
            "OCEAN_FLOOR": ("long_array", hm_packed),
        }),
        "isLightOn": ("byte", 0),
        "block_entities": ("list", "end", []),
        "block_ticks": ("list", "end", []),
        "fluid_ticks": ("list", "end", []),
        "PostProcessing": ("list", "list",
                           [("list", "end", []) for _ in range(24)]),
        "structures": ("compound", {
            "References": ("compound", {}),
            "starts": ("compound", {}),
        }),
    }
    return nbt_bytes(root)


def write_region(path, chunks):
    """chunks: dict (cx & 31, cz & 31) -> chunk nbt bytes."""
    offsets = bytearray(4096)
    stamps = bytearray(4096)
    body = bytearray()
    sector = 2
    now = int(time.time())
    for (lx, lz), raw in chunks.items():
        comp = zlib.compress(raw, 6)
        payload = struct.pack(">iB", len(comp) + 1, 2) + comp
        payload += b"\x00" * ((-len(payload)) % 4096)
        count = len(payload) // 4096
        i = 4 * (lx + lz * 32)
        offsets[i:i + 4] = struct.pack(">I", (sector << 8) | count)
        stamps[i:i + 4] = struct.pack(">I", now)
        body += payload
        sector += count
    with open(path, "wb") as fh:
        fh.write(bytes(offsets) + bytes(stamps) + bytes(body))


def write_level_dat(path, world, name, spawn):
    seed = world.seed
    sx, sy, sz = spawn
    flat_layers = [
        ("compound", {"block": ("string", "minecraft:bedrock"),
                      "height": ("int", 1)}),
        ("compound", {"block": ("string", "minecraft:dirt"),
                      "height": ("int", 2)}),
        ("compound", {"block": ("string", "minecraft:grass_block"),
                      "height": ("int", 1)}),
    ]
    dims = {
        "minecraft:overworld": ("compound", {
            "type": ("string", "minecraft:overworld"),
            "generator": ("compound", {
                "type": ("string", "minecraft:flat"),
                "settings": ("compound", {
                    "biome": ("string", "minecraft:plains"),
                    "lakes": ("byte", 0),
                    "features": ("byte", 0),
                    "layers": ("list", "compound", flat_layers),
                }),
            }),
        }),
        "minecraft:the_nether": ("compound", {
            "type": ("string", "minecraft:the_nether"),
            "generator": ("compound", {
                "type": ("string", "minecraft:noise"),
                "seed": ("long", seed),
                "settings": ("string", "minecraft:nether"),
                "biome_source": ("compound", {
                    "type": ("string", "minecraft:multi_noise"),
                    "preset": ("string", "minecraft:nether"),
                }),
            }),
        }),
        "minecraft:the_end": ("compound", {
            "type": ("string", "minecraft:the_end"),
            "generator": ("compound", {
                "type": ("string", "minecraft:noise"),
                "seed": ("long", seed),
                "settings": ("string", "minecraft:end"),
                "biome_source": ("compound", {
                    "type": ("string", "minecraft:the_end"),
                    "seed": ("long", seed),
                }),
            }),
        }),
    }
    data = {
        "DataVersion": ("int", DATA_VERSION),
        "version": ("int", 19133),
        "Version": ("compound", {
            "Id": ("int", DATA_VERSION),
            "Name": ("string", "1.18.2"),
            "Series": ("string", "main"),
            "Snapshot": ("byte", 0),
        }),
        "LevelName": ("string", name),
        "GameType": ("int", 1),
        "Difficulty": ("byte", 1),
        "hardcore": ("byte", 0),
        "allowCommands": ("byte", 1),
        "initialized": ("byte", 1),
        "Time": ("long", 0),
        "DayTime": ("long", 1000),
        "LastPlayed": ("long", int(time.time() * 1000)),
        "SpawnX": ("int", sx),
        "SpawnY": ("int", sy),
        "SpawnZ": ("int", sz),
        "SpawnAngle": ("float", 0.0),
        "raining": ("byte", 0),
        "rainTime": ("int", 120000),
        "thundering": ("byte", 0),
        "thunderTime": ("int", 120000),
        "WanderingTraderSpawnChance": ("int", 25),
        "WanderingTraderSpawnDelay": ("int", 24000),
        "WorldGenSettings": ("compound", {
            "bonus_chest": ("byte", 0),
            "generate_features": ("byte", 0),
            "seed": ("long", seed),
            "dimensions": ("compound", dims),
        }),
    }
    import gzip
    with gzip.open(path, "wb") as fh:
        fh.write(nbt_bytes({"Data": ("compound", data)}))


def export_world(world, dest_dir, name, center_x, center_z, size,
                 with_trees=True, progress=None, cancel=None):
    """Write a playable Minecraft world folder. size = blocks per side."""
    half = size // 2
    c0x = (int(center_x) - half) >> 4
    c0z = (int(center_z) - half) >> 4
    n_side = size // 16
    c1x = c0x + n_side - 1
    c1z = c0z + n_side - 1

    os.makedirs(os.path.join(dest_dir, "region"), exist_ok=True)
    total = n_side * n_side
    done = 0

    for rx in range(c0x >> 5, (c1x >> 5) + 1):
        for rz in range(c0z >> 5, (c1z >> 5) + 1):
            cxs = range(max(c0x, rx * 32), min(c1x, rx * 32 + 31) + 1)
            czs = range(max(c0z, rz * 32), min(c1z, rz * 32 + 31) + 1)
            # batch all noise for this region slice at block resolution
            bx0 = cxs.start * 16
            bz0 = czs.start * 16
            bw = (cxs.stop - cxs.start) * 16
            bh = (czs.stop - czs.start) * 16
            f = world.fields(bx0, bz0, 1.0, bw, bh)
            idx = world.classify(f)
            temp, _ = world.climate(f)
            y_surf = np.clip(
                62 + (f["elev"] - world.params["sea_level"]) * 150,
                -52, 300).astype(np.int32)
            world._cache.clear()

            chunks = {}
            for cz_ in czs:
                if cancel is not None and cancel.is_set():
                    return None
                for cx_ in cxs:
                    ox = (cx_ - cxs.start) * 16
                    oz = (cz_ - czs.start) * 16
                    chunks[(cx_ & 31, cz_ & 31)] = build_chunk_nbt(
                        world, cx_, cz_,
                        y_surf[oz:oz + 16, ox:ox + 16],
                        idx[oz:oz + 16, ox:ox + 16],
                        temp[oz:oz + 16, ox:ox + 16],
                        with_trees)
                    done += 1
                if progress:
                    progress(done / total)
            write_region(os.path.join(
                dest_dir, "region", f"r.{rx}.{rz}.mca"), chunks)

    # find a land spawn near the center
    spawn = (int(center_x), 80, int(center_z))
    for r in range(0, max(half, 24), 24):
        found = False
        for ang in range(0, 360, 30):
            px = int(center_x + r * math.cos(math.radians(ang)))
            pz = int(center_z + r * math.sin(math.radians(ang)))
            bid, y, _ = world.point_data(px, pz)
            if bid not in WATER_IDS and y >= SEA_Y:
                spawn = (px, y + 2, pz)
                found = True
                break
        if found:
            break
    write_level_dat(os.path.join(dest_dir, "level.dat"), world, name, spawn)
    return spawn

def friendly_export_error(exc):
    import errno as _errno
    code = getattr(exc, "errno", None)
    if isinstance(exc, PermissionError):
        return ("No permission to write in that folder.\n\n"
                "Pick a different folder, for example Documents or "
                "Desktop. Folders like Program Files are protected by "
                "Windows.")
    if isinstance(exc, FileNotFoundError) or code in (_errno.EINVAL,
                                                      _errno.ENOENT):
        return ("The destination folder is not valid or no longer "
                "exists.\n\nUse the Change button and pick the folder "
                "again.")
    if code == _errno.ENOSPC:
        return ("The disk is full.\n\nFree some space or export a "
                "smaller world.")
    return (f"Unexpected error:\n{exc}\n\nCommon causes: no permission "
            "to write in the folder, the folder path is not valid, or "
            "the disk is full.")

# =================================================================== GUI ===

def run_gui(smoke=False, shot=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    ACCENT = "#3a7ebf"
    BG = "#101113"

    class RenderWorker(threading.Thread):
        """One background thread, latest job wins, result picked up by
        a polling loop on the UI thread."""

        def __init__(self):
            super().__init__(daemon=True)
            self.cond = threading.Condition()
            self.job = None
            self.result = None
            self.res_lock = threading.Lock()

        def submit(self, job):
            with self.cond:
                self.job = job
                self.cond.notify()

        def take_result(self):
            with self.res_lock:
                r, self.result = self.result, None
            return r

        def run(self):
            while True:
                with self.cond:
                    while self.job is None:
                        self.cond.wait()
                    job = self.job
                    self.job = None
                try:
                    world, w, h, cx, cz, zoom, res, gen = job
                    img = world.render(w, h, cx, cz, zoom, res=res)
                    with self.res_lock:
                        self.result = (img, cx, cz, zoom, res, gen)
                except Exception:
                    import traceback
                    traceback.print_exc()

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("WorldForge")
            self.geometry("1380x860")
            self.minsize(1020, 640)
            self.configure(fg_color=BG)

            self.seed_var = tk.StringVar(value=str(random.randrange(10 ** 8)))
            self.cx, self.cz = 0.0, 0.0
            self.zoom = 0.75
            self.world = None
            self.photo = None
            self.base_img = None      # last rendered PIL image
            self.base_view = None     # (cx, cz, zoom) of base_img
            self.gen = 0              # render generation counter
            self._drag = None
            self._full_job = None
            self._hover_t = 0.0
            self._last_res = 0.0
            self.show_structures = tk.BooleanVar(value=True)
            self.vars = {}

            self.worker = RenderWorker()
            self.worker.start()

            self._build_ui()
            self._rebuild_world()
            self.after(30, self._poll_results)
            self.after(80, self.request_render)

        # ---------- UI

        def _build_ui(self):
            side = ctk.CTkScrollableFrame(self, width=272,
                                          fg_color="#16171a",
                                          corner_radius=0)
            side.pack(side="left", fill="y")

            ctk.CTkLabel(side, text="WorldForge",
                         font=ctk.CTkFont(size=24, weight="bold")
                         ).pack(anchor="w", padx=14, pady=(14, 0))
            ctk.CTkLabel(side,
                         text=f"Minecraft world generator  v{APP_VERSION}",
                         font=ctk.CTkFont(size=12), text_color="#8a8d93"
                         ).pack(anchor="w", padx=14, pady=(0, 12))

            row = ctk.CTkFrame(side, fg_color="transparent")
            row.pack(fill="x", padx=12)
            ctk.CTkLabel(row, text="Seed").pack(side="left")
            ent = ctk.CTkEntry(row, textvariable=self.seed_var, width=128)
            ent.pack(side="left", padx=8)
            ent.bind("<Return>", lambda _e: self._rebuild_world(render=True))
            ctk.CTkButton(row, text="New", width=54,
                          command=self.randomize_seed).pack(side="left")

            def slider(label, key, lo, hi, fmt="{:.2f}", steps=None):
                head = ctk.CTkFrame(side, fg_color="transparent")
                head.pack(fill="x", padx=14, pady=(10, 0))
                ctk.CTkLabel(head, text=label,
                             font=ctk.CTkFont(size=12)).pack(side="left")
                val = ctk.CTkLabel(head, text="",
                                   font=ctk.CTkFont(size=12),
                                   text_color="#9fa3aa")
                val.pack(side="right")
                var = tk.DoubleVar(value=DEFAULTS[key])

                def on_move(_v):
                    val.configure(text=fmt.format(var.get()))
                    self._apply_params()
                    self.request_render(quick_only=True)

                s = ctk.CTkSlider(side, from_=lo, to=hi, variable=var,
                                  command=on_move,
                                  number_of_steps=steps or 200)
                s.pack(fill="x", padx=12, pady=(2, 0))
                s.bind("<ButtonRelease-1>",
                       lambda _e: self.request_render())
                val.configure(text=fmt.format(var.get()))
                self.vars[key] = var
                self._slider_labels[key] = (
                    lambda v=val, f=fmt, vr=var: v.configure(
                        text=f.format(vr.get())))

            self._slider_labels = {}
            slider("Sea level", "sea_level", 0.30, 0.70)
            slider("Mountains", "mountain_level", 0.60, 0.95)
            slider("Terrain scale", "terrain_scale", 250, 3000, "{:.0f}")
            slider("Detail", "detail", 2, 8, "{:.0f}", steps=6)
            slider("Temperature", "temperature", -0.45, 0.45, "{:+.2f}")
            slider("Moisture", "moisture", -0.45, 0.45, "{:+.2f}")
            slider("Rivers", "rivers", 0.0, 0.05, "{:.3f}")

            ctk.CTkLabel(side, text="Presets",
                         font=ctk.CTkFont(size=13, weight="bold")
                         ).pack(anchor="w", padx=14, pady=(12, 2))
            pres = ctk.CTkFrame(side, fg_color="transparent")
            pres.pack(fill="x", padx=12)
            pres.grid_columnconfigure((0, 1), weight=1, uniform="p")
            for i, pname in enumerate(PRESETS):
                ctk.CTkButton(
                    pres, text=pname, height=30,
                    fg_color="#26282d", hover_color="#33363c",
                    font=ctk.CTkFont(size=12),
                    command=lambda n=pname: self.apply_preset(n)
                ).grid(row=i // 2, column=i % 2, sticky="ew",
                       padx=2, pady=2)

            ctk.CTkSwitch(side, text="Show structures",
                          variable=self.show_structures,
                          command=self.request_render
                          ).pack(anchor="w", padx=14, pady=(14, 2))

            btns = ctk.CTkFrame(side, fg_color="transparent")
            btns.pack(fill="x", padx=12, pady=(10, 4))
            ctk.CTkButton(btns, text="Reset view", width=118,
                          fg_color="#26282d", hover_color="#33363c",
                          command=self.reset_view).pack(side="left")
            ctk.CTkButton(btns, text="Save image", width=118,
                          fg_color="#26282d", hover_color="#33363c",
                          command=self.export_png).pack(side="right")

            ctk.CTkButton(side, text="Export Minecraft world",
                          height=40, fg_color=ACCENT,
                          font=ctk.CTkFont(size=14, weight="bold"),
                          command=self.export_world_dialog
                          ).pack(fill="x", padx=12, pady=(6, 10))

            ctk.CTkLabel(side, text="Legend",
                         font=ctk.CTkFont(size=13, weight="bold")
                         ).pack(anchor="w", padx=14, pady=(4, 2))
            leg = tk.Canvas(side, width=246, height=176, bg="#16171a",
                            highlightthickness=0)
            leg.pack(padx=12)
            ids = [0, 1, 6, 4, 9, 10, 11, 12, 8, 13, 14, 15, 16, 17, 19, 20]
            for i, bid in enumerate(ids):
                name, rgb, _mc = BIOMES[bid]
                px = 6 + (i % 2) * 122
                py = 6 + (i // 2) * 21
                leg.create_rectangle(px, py, px + 13, py + 13,
                                     fill="#%02x%02x%02x" % rgb, outline="")
                leg.create_text(px + 19, py + 6, anchor="w", text=name,
                                fill="#c9ccd1", font=("Segoe UI", 9))
            st = tk.Canvas(side, width=246, height=70, bg="#16171a",
                           highlightthickness=0)
            st.pack(padx=12, pady=(4, 8))
            for i, (name, color) in enumerate(STRUCTURE_STYLES.items()):
                px = 6 + (i % 2) * 122
                py = 6 + (i // 2) * 21
                st.create_oval(px, py, px + 12, py + 12, fill=color,
                               outline="")
                st.create_text(px + 19, py + 6, anchor="w", text=name,
                               fill="#c9ccd1", font=("Segoe UI", 9))

            foot = ctk.CTkFrame(side, fg_color="transparent")
            foot.pack(fill="x", padx=12, pady=(0, 12))
            ctk.CTkLabel(foot, text="Drag to pan, scroll to zoom",
                         font=ctk.CTkFont(size=11), text_color="#6f7378"
                         ).pack(side="left", padx=(2, 0))
            ctk.CTkButton(foot, text="About", width=64, height=24,
                          fg_color="#26282d", hover_color="#33363c",
                          font=ctk.CTkFont(size=11),
                          command=self.show_about).pack(side="right")

            right = tk.Frame(self, bg=BG)
            right.pack(side="left", fill="both", expand=True)
            self.canvas = tk.Canvas(right, bg=BG, highlightthickness=0,
                                    cursor="fleur")
            self.canvas.pack(fill="both", expand=True)
            self.status = tk.Label(
                right, anchor="w", bg="#0c0d0f", fg="#9fa3aa",
                font=("Consolas", 10), padx=10, pady=4)
            self.status.pack(fill="x")

            self.canvas.bind("<Configure>", lambda _e: self.request_render())
            self.canvas.bind("<ButtonPress-1>", self._drag_start)
            self.canvas.bind("<B1-Motion>", self._drag_move)
            self.canvas.bind("<ButtonRelease-1>",
                             lambda _e: self.request_render())
            self.canvas.bind("<MouseWheel>", self._wheel)
            self.canvas.bind("<Motion>", self._hover)

        # ---------- params

        def _apply_params(self):
            if self.world:
                for k, v in self.vars.items():
                    self.world.params[k] = v.get()

        def _rebuild_world(self, render=False):
            try:
                seed = int(self.seed_var.get().strip())
            except ValueError:
                seed = abs(hash(self.seed_var.get())) % (10 ** 9)
                self.seed_var.set(str(seed))
            self.world = World(seed)
            self._apply_params()
            if render:
                self.request_render()

        def randomize_seed(self):
            self.seed_var.set(str(random.randrange(10 ** 8)))
            self._rebuild_world(render=True)

        def reset_view(self):
            self.cx, self.cz, self.zoom = 0.0, 0.0, 0.75
            self.request_render()

        def apply_preset(self, name):
            for key, value in PRESETS[name].items():
                self.vars[key].set(value)
                self._slider_labels[key]()
            self._apply_params()
            self.request_render()

        def show_about(self):
            dlg = ctk.CTkToplevel(self)
            dlg.title("About WorldForge")
            dlg.geometry("400x300")
            dlg.transient(self)
            dlg.grab_set()
            dlg.configure(fg_color="#16171a")
            ctk.CTkLabel(dlg, text="WorldForge",
                         font=ctk.CTkFont(size=26, weight="bold")
                         ).pack(pady=(24, 0))
            ctk.CTkLabel(dlg, text=f"Version {APP_VERSION}",
                         font=ctk.CTkFont(size=13),
                         text_color="#9fa3aa").pack(pady=(2, 14))
            ctk.CTkLabel(dlg, text="Minecraft world export format:",
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack()
            ctk.CTkLabel(dlg, text=MC_FORMAT,
                         font=ctk.CTkFont(size=12),
                         text_color="#9fa3aa", justify="center").pack()
            ctk.CTkLabel(dlg, text=f"Author: {APP_AUTHOR}",
                         font=ctk.CTkFont(size=12)).pack(pady=(14, 6))

            def open_github():
                import webbrowser
                webbrowser.open(GITHUB_URL)
            ctk.CTkButton(dlg, text="GitHub", fg_color=ACCENT,
                          width=130, command=open_github).pack(pady=4)
            ctk.CTkButton(dlg, text="Close", fg_color="#26282d",
                          hover_color="#33363c", width=130,
                          command=dlg.destroy).pack(pady=4)

        # ---------- interaction

        def _drag_start(self, e):
            self._drag = (e.x, e.y)

        def _drag_move(self, e):
            if not self._drag:
                return
            dx, dy = e.x - self._drag[0], e.y - self._drag[1]
            self._drag = (e.x, e.y)
            self.cx -= dx / self.zoom
            self.cz -= dy / self.zoom
            self._show_preview()
            self.request_render(quick_only=True)

        def _wheel(self, e):
            old = self.zoom
            factor = 1.25 if e.delta > 0 else 0.8
            self.zoom = min(16.0, max(0.03, self.zoom * factor))
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            self.cx += (e.x - w / 2) * (1 / old - 1 / self.zoom)
            self.cz += (e.y - h / 2) * (1 / old - 1 / self.zoom)
            self._show_preview()
            self.request_render(quick_only=True)

        def _hover(self, e):
            now = time.time()
            if now - self._hover_t < 0.06 or not self.world:
                return
            self._hover_t = now
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            x = self.cx + (e.x - w / 2) / self.zoom
            z = self.cz + (e.y - h / 2) / self.zoom
            name, y, _ = self.world.biome_at(x, z)
            self.status.config(
                text=f"X {int(x):>7}   Z {int(z):>7}   Y~{y:<4} {name:<16}"
                     f" | seed {self.world.seed}"
                     f" | zoom {self.zoom:.2f} px/block")

        # ---------- render pipeline

        def request_render(self, quick_only=False):
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 10 or h < 10 or not self.world:
                return
            self.gen += 1
            self.worker.submit((self.world, w, h, self.cx, self.cz,
                                self.zoom, 0.45, self.gen))
            self._schedule_full(60 if not quick_only else 320)

        def _schedule_full(self, delay):
            if self._full_job:
                self.after_cancel(self._full_job)
            self._full_job = self.after(delay, self._do_full)

        def _do_full(self):
            self._full_job = None
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 10 or h < 10 or not self.world:
                return
            self.gen += 1
            self.worker.submit((self.world, w, h, self.cx, self.cz,
                                self.zoom, 1.0, self.gen))

        def _poll_results(self):
            r = self.worker.take_result()
            if r:
                img, cx, cz, zoom, res, gen = r
                if gen >= self.gen - 1:  # drop stale frames
                    self.base_img = img
                    self.base_view = (cx, cz, zoom)
                    self._last_res = res
                    self._blit(img)
                    if res == 1.0:
                        self._draw_overlays()
            self.after(30, self._poll_results)

        def _blit(self, img):
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("map")
            self.canvas.delete("ovl")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo,
                                     tags="map")
            self.canvas.tag_lower("map")

        def _show_preview(self):
            """Instant feedback: reproject the last rendered image to the
            current view while the real render happens in the background."""
            if self.base_img is None:
                return
            bcx, bcz, bzoom = self.base_view
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            a = bzoom / self.zoom
            c = ((self.cx - bcx) * bzoom + self.base_img.width / 2
                 - a * w / 2)
            fy = ((self.cz - bcz) * bzoom + self.base_img.height / 2
                  - a * h / 2)
            try:
                img = self.base_img.transform(
                    (w, h), Image.AFFINE, (a, 0, c, 0, a, fy),
                    resample=Image.NEAREST, fillcolor=(10, 11, 13))
            except (ValueError, MemoryError):
                return
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("map")
            self.canvas.delete("ovl")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo,
                                     tags="map")
            self.canvas.tag_lower("map")

        def _draw_overlays(self):
            self.canvas.delete("ovl")
            if not self.show_structures.get():
                return
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            x0 = self.cx - w / 2 / self.zoom
            z0 = self.cz - h / 2 / self.zoom
            x1 = self.cx + w / 2 / self.zoom
            z1 = self.cz + h / 2 / self.zoom
            for sx, sz, kind in self.world.structures_in_rect(
                    x0, z0, x1, z1):
                px = (sx - x0) * self.zoom
                pz = (sz - z0) * self.zoom
                c = STRUCTURE_STYLES[kind]
                r = 5
                self.canvas.create_oval(px - r - 2, pz - r - 2, px + r + 2,
                                        pz + r + 2, fill="#101113",
                                        outline="", tags="ovl")
                if kind == "Village":
                    self.canvas.create_rectangle(
                        px - r + 1, pz - r + 2, px + r - 1, pz + r,
                        fill=c, outline="white", tags="ovl")
                    self.canvas.create_polygon(
                        px - r - 1, pz - r + 2, px, pz - r - 3,
                        px + r + 1, pz - r + 2, fill=c, outline="white",
                        tags="ovl")
                elif kind == "Ocean Monument":
                    self.canvas.create_oval(px - r, pz - r, px + r, pz + r,
                                            fill=c, outline="white",
                                            tags="ovl")
                else:
                    self.canvas.create_polygon(
                        px, pz - r - 1, px + r + 1, pz, px, pz + r + 1,
                        px - r - 1, pz, fill=c, outline="white", tags="ovl")

        # ---------- exports

        def export_png(self):
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                initialfile=f"world_{self.world.seed}.png")
            if not path:
                return
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            img = self.world.render(w * 2, h * 2, self.cx, self.cz,
                                    self.zoom * 2)
            img.save(path)
            messagebox.showinfo("WorldForge", f"Saved {path}")

        def export_world_dialog(self):
            dlg = ctk.CTkToplevel(self)
            dlg.title("Export Minecraft world")
            dlg.geometry("470x410")
            dlg.transient(self)
            dlg.grab_set()
            dlg.configure(fg_color="#16171a")

            ctk.CTkLabel(dlg, text="Export Minecraft world",
                         font=ctk.CTkFont(size=18, weight="bold")
                         ).pack(anchor="w", padx=18, pady=(16, 2))
            ctk.CTkLabel(
                dlg, text="Java Edition, 1.18.2 save format. Every newer\n"
                          "version (1.20, 1.21, ...) opens and upgrades it.",
                font=ctk.CTkFont(size=12), text_color="#8a8d93",
                justify="left").pack(anchor="w", padx=18)

            row1 = ctk.CTkFrame(dlg, fg_color="transparent")
            row1.pack(fill="x", padx=18, pady=(14, 4))
            ctk.CTkLabel(row1, text="World name").pack(side="left")
            name_var = tk.StringVar(value=f"WorldForge {self.world.seed}")
            ctk.CTkEntry(row1, textvariable=name_var, width=240
                         ).pack(side="right")

            row2 = ctk.CTkFrame(dlg, fg_color="transparent")
            row2.pack(fill="x", padx=18, pady=4)
            ctk.CTkLabel(row2, text="Size (blocks)").pack(side="left")
            size_var = tk.StringVar(value="1024 x 1024")
            size_warn = ctk.CTkLabel(dlg, text=" ",
                                     font=ctk.CTkFont(size=11),
                                     text_color="#d9a03c")

            def on_size(choice):
                big = int(choice.split(" ")[0]) >= 2048
                size_warn.configure(
                    text="Large worlds may take several minutes."
                         if big else " ")
            ctk.CTkOptionMenu(row2, variable=size_var, width=240,
                              values=["512 x 512", "1024 x 1024",
                                      "2048 x 2048", "3072 x 3072"],
                              command=on_size).pack(side="right")
            size_warn.pack(anchor="w", padx=18)

            trees_var = tk.BooleanVar(value=True)
            ctk.CTkSwitch(dlg, text="Generate trees", variable=trees_var
                          ).pack(anchor="w", padx=18, pady=8)

            saves = os.path.join(os.environ.get("APPDATA", ""),
                                 ".minecraft", "saves")
            dest_var = tk.StringVar(
                value=saves if os.path.isdir(saves)
                else os.path.expanduser("~"))
            row3 = ctk.CTkFrame(dlg, fg_color="transparent")
            row3.pack(fill="x", padx=18, pady=4)
            dest_lbl = ctk.CTkLabel(row3, text="", text_color="#8a8d93",
                                    font=ctk.CTkFont(size=11))
            dest_lbl.pack(side="left")

            def short(p):
                return p if len(p) < 46 else "..." + p[-43:]
            dest_lbl.configure(text=short(dest_var.get()))

            def pick():
                d = filedialog.askdirectory(initialdir=dest_var.get())
                if d:
                    dest_var.set(d)
                    dest_lbl.configure(text=short(d))
            ctk.CTkButton(row3, text="Change...", width=84,
                          fg_color="#26282d", hover_color="#33363c",
                          command=pick).pack(side="right")

            bar = ctk.CTkProgressBar(dlg)
            bar.set(0)
            bar.pack(fill="x", padx=18, pady=(12, 4))
            stat = ctk.CTkLabel(dlg, text="Centered on the current view",
                                font=ctk.CTkFont(size=11),
                                text_color="#8a8d93")
            stat.pack(anchor="w", padx=18)

            cancel_ev = threading.Event()
            state = {"running": False, "done": None, "frac": 0.0}

            def work(world_copy, dest, name, size, trees):
                try:
                    spawn = export_world(
                        world_copy, dest, name, self.cx, self.cz, size,
                        with_trees=trees,
                        progress=lambda fr: state.update(frac=fr),
                        cancel=cancel_ev)
                    state["done"] = ("ok", spawn)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    state["done"] = ("err", friendly_export_error(exc))

            def tick():
                if not dlg.winfo_exists():
                    return
                if state["running"]:
                    bar.set(state["frac"])
                    stat.configure(
                        text=f"Generating chunks... "
                             f"{int(state['frac'] * 100)}%")
                d = state["done"]
                if d:
                    state["done"] = None
                    state["running"] = False
                    if d[0] == "ok" and d[1] is not None:
                        bar.set(1.0)
                        messagebox.showinfo(
                            "WorldForge",
                            "World exported.\n\nOpen Minecraft, go to "
                            "Singleplayer and it is in the world list.",
                            parent=dlg)
                        dlg.destroy()
                        return
                    if d[0] == "err":
                        messagebox.showerror(
                            "WorldForge", f"Export failed.\n\n{d[1]}",
                            parent=dlg)
                dlg.after(100, tick)

            def start():
                if state["running"]:
                    return
                name = name_var.get().strip() or "WorldForge"
                size = int(size_var.get().split(" ")[0])
                safe = "".join(ch for ch in name
                               if ch.isalnum() or ch in " _-").strip()
                dest = os.path.join(dest_var.get(), safe or "world")
                base, i = dest, 2
                while os.path.exists(dest):
                    dest = f"{base} ({i})"
                    i += 1
                # private world copy keeps map browsing responsive
                wc = World(self.world.seed, dict(self.world.params))
                state["running"] = True
                threading.Thread(
                    target=work,
                    args=(wc, dest, name, size, trees_var.get()),
                    daemon=True).start()

            btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
            btn_row.pack(fill="x", padx=18, pady=12)
            ctk.CTkButton(btn_row, text="Export", fg_color=ACCENT,
                          width=110, command=start).pack(side="right")
            ctk.CTkButton(btn_row, text="Cancel", fg_color="#26282d",
                          hover_color="#33363c", width=90,
                          command=lambda: (cancel_ev.set(), dlg.destroy())
                          ).pack(side="right", padx=8)
            tick()

    app = App()
    if smoke or shot:
        if shot:
            app.geometry("1180x680+30+30")
        app.update()
        app.request_render()
        t0 = time.time()
        want_full = shot is not None
        while time.time() - t0 < 10:
            app.update()
            time.sleep(0.02)
            if app.base_img is not None and (
                    not want_full or app._last_res == 1.0):
                break
        ok = app.base_img is not None
        if shot and ok:
            app.lift()
            app.focus_force()
            for _ in range(20):
                app.update()
                time.sleep(0.03)
            from PIL import ImageGrab
            x, y = app.winfo_rootx(), app.winfo_rooty()
            w, h = app.winfo_width(), app.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
            print(f"screenshot saved: {shot}")
        app.destroy()
        print("smoke ok" if ok else "smoke FAILED: no render arrived")
        sys.exit(0 if ok else 1)
    app.mainloop()

# ================================================================== main ===

def selftest(out_path):
    world = World(12345)
    t0 = time.time()
    img = world.render(1280, 800, 0, 0, zoom=0.75)
    full_ms = (time.time() - t0) * 1000
    t0 = time.time()
    world.params["temperature"] = 0.1
    world.render(1280, 800, 0, 0, zoom=0.75)  # cache hit: reshade only
    cached_ms = (time.time() - t0) * 1000
    world.params["temperature"] = 0.0
    t0 = time.time()
    world.render(1280, 800, 50, 50, zoom=0.75, res=0.45)
    quick_ms = (time.time() - t0) * 1000
    img.save(out_path)
    n = len(world.structures_in_rect(-850, -530, 850, 530))
    name, y, _ = world.biome_at(0, 0)
    print(f"selftest ok: {out_path}")
    print(f"  full render {full_ms:.0f} ms, slider change {cached_ms:.0f} ms,"
          f" quick {quick_ms:.0f} ms")
    print(f"  {n} structures in view, spawn biome {name} (Y~{y})")


def export_test(dest):
    world = World(12345)
    t0 = time.time()
    spawn = export_world(world, dest, "ExportTest", 0, 0, 256,
                         with_trees=True)
    print(f"export ok: {dest} in {time.time() - t0:.1f} s, spawn {spawn}")


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        selftest(sys.argv[i + 1] if len(sys.argv) > i + 1 else "selftest.png")
        return
    if "--export-test" in sys.argv:
        i = sys.argv.index("--export-test")
        export_test(sys.argv[i + 1] if len(sys.argv) > i + 1 else "testworld")
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    shot = None
    if "--screenshot" in sys.argv:
        i = sys.argv.index("--screenshot")
        shot = sys.argv[i + 1] if len(sys.argv) > i + 1 else "screenshot.png"
    run_gui(smoke="--smoke" in sys.argv, shot=shot)


if __name__ == "__main__":
    main()
