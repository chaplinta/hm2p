# Camera and Lens Specifications

## Camera: Basler acA1300-200um (ace U)

| Parameter | Value |
|-----------|-------|
| Sensor | ON Semiconductor PYTHON 1300 |
| Sensor format | 1/2" |
| Sensor size | 6.14 × 4.92 mm |
| Resolution | 1280 × 1024 px (1.3 MP) |
| Pixel size | 4.8 × 4.8 µm |
| Max frame rate | 203 fps |
| Shutter | Global shutter (CMOS) |
| Bit depth | 10-bit |
| Interface | USB 3.0 |
| Mount | C-mount |

**Datasheet:** [Basler acA1300-200um](https://docs.baslerweb.com/aca1300-200um)

Recording at ~100 fps (Basler Pylon SDK, triggered by NI DAQ TTL pulses).

---

## Lenses

Two lenses were used across sessions (recorded in `experiments.csv` column `lens`):

### f4mm — Basler C125-0418-5M (sessions 1–19)

| Parameter | Value |
|-----------|-------|
| Focal length | 4 mm |
| Aperture | f/1.8 – f/22 |
| Max sensor format | 1/2.5" (covers 1/2") |
| Resolution | 5 MP rated |
| Mount | C-mount |
| Manufacturer | Basler (Fujinon design) |

Computed at 700mm working distance (1/2" sensor):

| Parameter | Value |
|-----------|-------|
| HFOV | 75.0° |
| VFOV | 63.2° |
| FOV at 700mm | 1074 × 861 mm |
| Scale | ~0.84 mm/px |
| Measured scale (meta.txt) | 0.811 mm/px |

Used for sessions: 20210823–20220804 (animals 1114353–1117646).

### f6mm — C-mount 6mm (sessions 20–26)

| Parameter | Value |
|-----------|-------|
| Focal length | 6 mm |
| Aperture | ~f/1.8 |
| Max sensor format | 1/2" |
| Mount | C-mount |

Computed at 700mm working distance (1/2" sensor):

| Parameter | Value |
|-----------|-------|
| HFOV | 54.2° |
| VFOV | 44.6° |
| FOV at 700mm | 716 × 574 mm |
| Scale | ~0.56 mm/px |

Used for sessions: 20221003–20221117 (animals 1118020–1118317).
The f6mm lens has a narrower FOV — the maze fills more of the frame,
giving higher spatial resolution per pixel.

---

## Mounting and Geometry

| Parameter | Value |
|-----------|-------|
| Mounting | Overhead, pointing straight down |
| Camera height | ~700 mm above maze floor |
| Camera position | Off-center (varies per session) |
| Lens distortion correction | Applied pre-recording (Basler Pylon) |
| Video format | Basler .camera → cropped .mp4 |
| Crop region | Per-session, stored in `meta.txt` |

## Maze Physical Dimensions

| Parameter | Value |
|-----------|-------|
| Overall footprint | ~500 × 347 mm (50 × 35 cm) |
| Grid | 7 columns × 5 rows (23 accessible cells) |
| Cell size | ~71 × 69 mm (~7 cm corridors) |
| Wall height | 120 mm (12 cm) above floor |
| Wall material | 9 mm clear acrylic (laser-cut) |
| Wall slot into floor | 10 mm tab |

Source: laser-cut SVG file (CorelDRAW `Tristan-maze-walls-large-2.cdr`).

The camera is **not centered** over the maze. The optical axis intersects
the maze floor at an offset from the maze center (estimated ~65mm lateral,
~32mm longitudinal for the sample session). This offset, combined with the
mouse's body height above the floor (2–5cm), causes **parallax displacement**
of keypoint positions. See `docs/perspective-correction-plan.md`.

---

## Session Lens Assignment

| Sessions | Lens | Animals | Fibre |
|----------|------|---------|-------|
| 20210823–20220804 (19 sessions) | f4mm | 1114353, 1114356, 1115464, 1115465, 1115816, 1116663, 1117217, 1116994, 1117646 | SFB/TFB |
| 20221003–20221117 (7 sessions) | f6mm | 1118020, 1118023, 1118018, 1117788, 1118213, 1118320, 1118317 | TFB |

The lens change happened between July 2022 (last f4mm session: 20220804)
and October 2022 (first f6mm session: 20221003).

---

## Implications for Analysis

1. **Scale differs between lens groups**: f4mm ≈ 0.81 mm/px, f6mm ≈ 0.56 mm/px.
   Position data must be converted using per-session `scale_mm_per_px` from `meta.txt`.

2. **Perspective correction** must use per-session camera center (from crop
   offset + original frame size), not a fixed value.

3. **The f6mm group is all Penk⁻CamKII+ or late Penk+**. This is a potential
   confound — any apparent celltype difference in spatial precision could be
   driven by the lens change. Report lens group alongside celltype in analyses.

Sources:
- [Basler acA1300-200um specs](https://docs.baslerweb.com/aca1300-200um)
- [Basler C125-0418-5M lens](https://www.baslerweb.com/en/shop/basler-lens-c125-0418-5m-f4mm/)
