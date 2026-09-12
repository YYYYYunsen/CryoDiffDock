#!/usr/bin/env python3
"""Recolor the current paper-figure exports with one of three deep-sea palettes.

Vector-first workflow:
  * SVG accents are recolored while neutral text, axes, and backgrounds stay put.
  * SVG-derived PDF and PNG companions are rendered from the recolored SVG.
  * PDF-only figures are rasterized, recolored while preserving grayscale/detail,
    and written to a new PNG plus a raster PDF. Source files are never modified.

Run from the repository root:
    python scripts/recolor_paper_figures.py
    python scripts/recolor_paper_figures.py --palette style3_blue_green
"""
from __future__ import annotations

import argparse
import base64
import colorsys
import csv
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Keep NumPy/OpenMP from creating worker threads. The batch renderer starts many
# short-lived SVG/PDF subprocesses, and single-threaded array work avoids macOS
# fork/shared-memory deadlocks while remaining fast for these figure-sized images.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("visualizations/paper_figures_deep_sea")

# User-provided palettes. Style 3 selects eight of the twelve swatches visible
# in the reference image: six saturated/mid-tone ocean colors and two pale tints.
PALETTES: dict[str, tuple[str, ...]] = {
    "style1_blue": ("#6CBADB", "#96C2D4", "#BAD2E1", "#D8E5F7", "#DBF1FA"),
    "style2_green": ("#519D78", "#88CF88", "#AADCA9"),
    "style3_blue_green": (
        "#045798", "#3492B2", "#58B8D1", "#6FC8CA",
        "#92C2A6", "#BCF4C5", "#ACEEFE", "#D6F6FF",
    ),
}


@dataclass(frozen=True)
class SourceGroup:
    label: str
    relative_dir: str
    pdf_only: bool = False


# These are the newest paper-ready export locations currently in the project.
# Update this list when a newer final export replaces one of these folders.
DEFAULT_SOURCES = (
    SourceGroup("figure1", "visualizations/figure1/final", pdf_only=True),
    SourceGroup("figure2", "visualizations/figure2_external/data/figure2_external_authoritative174_package_v2/figures"),
    SourceGroup("figure3", "visualizations/figure3_local/plots_final"),
    SourceGroup("figure4", "visualizations/figure4_uncertainty/data/figure4_pose_ensemble_reliability_authoritative174_v1/figures"),
    SourceGroup("figure5", "visualizations/figure5_cryoliganddb/plots_final_v9"),
    SourceGroup("figure6", "visualizations/figure6_global_cascade/figure6_global_cascade_authoritative_package_v3/figures"),
)

HEX_RE = re.compile(r"(?<![\w])#([0-9a-fA-F]{6})(?![\w])")
COLOR_PROPERTY_RE = re.compile(
    r"(?i)(?:(?:fill|stroke|stop-color|color)\s*:\s*|(?:fill|stroke|stop-color|color)\s*=\s*[\"'])(?P<color>#[0-9a-f]{6})"
)
EMBEDDED_IMAGE_RE = re.compile(
    r"(?is)((?:xlink:href|href)\s*=\s*([\"'])data:image/(?:png|jpeg);base64,)([^\"']+)(\2)"
)


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def is_chromatic(hex_color: str) -> bool:
    r, g, b = rgb(hex_color)
    hi, lo = max(r, g, b), min(r, g, b)
    # Leave black, gray strokes/text, and near-white paper/backgrounds alone.
    return (hi - lo) >= 22 and lo < 248


def hue_degrees(hex_color: str) -> float:
    r, g, b = (c / 255 for c in rgb(hex_color))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def hue_groups(colors: list[str], max_gap: float = 17.0) -> list[list[str]]:
    """Group nearby hues so line/fill tints of one series remain one color."""
    ordered = sorted(set(colors), key=hue_degrees)
    if not ordered:
        return []
    groups: list[list[str]] = [[ordered[0]]]
    for color in ordered[1:]:
        if hue_degrees(color) - hue_degrees(groups[-1][-1]) <= max_gap:
            groups[-1].append(color)
        else:
            groups.append([color])
    # Merge red hues that straddle 0/360 degrees.
    if len(groups) > 1:
        wrap_gap = hue_degrees(groups[0][0]) + 360 - hue_degrees(groups[-1][-1])
        if wrap_gap <= max_gap:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    groups.sort(key=lambda group: sum(hue_degrees(c) for c in group) / len(group))
    return groups


def hue_group_center(group: list[str]) -> float:
    angles = [math.radians(hue_degrees(color)) for color in group]
    x = sum(math.cos(angle) for angle in angles) / len(angles)
    y = sum(math.sin(angle) for angle in angles) / len(angles)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def collect_global_hue_centers(sources: list[Path]) -> list[float]:
    """Build one hue map across every figure so a category keeps its color."""
    all_colors: set[str] = set()
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        all_colors.update(
            match.group("color").upper()
            for match in COLOR_PROPERTY_RE.finditer(text)
            if is_chromatic(match.group("color"))
        )
    groups = hue_groups(list(all_colors))
    return [hue_group_center(group) for group in groups]


def closest_hue_index(hue: float, centers: list[float]) -> int:
    return min(range(len(centers)), key=lambda i: min(abs(hue - centers[i]), 360 - abs(hue - centers[i])))


def target_indices(source_count: int, palette_count: int) -> list[int]:
    # When a figure uses just a few series, use the most visible leading swatches.
    # If there are more source hues than target colors, distribute across the full
    # palette so no source category collapses onto the same target hue.
    if source_count <= palette_count:
        return list(range(source_count))
    if source_count == 1:
        return [0]
    return [round(i * (palette_count - 1) / (source_count - 1)) for i in range(source_count)]


def build_svg_map(text: str, palette: tuple[str, ...],
                  global_hue_centers: list[float] | None = None) -> tuple[dict[str, str], list[str]]:
    # Only inspect actual CSS paint properties. This avoids changing hex strings
    # embedded in labels, IDs, URLs, or metadata.
    found = [m.group("color").upper() for m in COLOR_PROPERTY_RE.finditer(text)]
    colors = sorted({c for c in found if is_chromatic(c)}, key=hue_degrees)
    mapping: dict[str, str] = {}
    group_log: list[str] = []
    if global_hue_centers:
        indices = target_indices(len(global_hue_centers), len(palette))
        grouped: dict[int, list[str]] = {}
        for source in colors:
            global_index = closest_hue_index(hue_degrees(source), global_hue_centers)
            palette_index = indices[global_index]
            mapping[source.upper()] = palette[palette_index].upper()
            grouped.setdefault(palette_index, []).append(source)
        group_log = [f"{'/'.join(sources)} -> {palette[index]}" for index, sources in sorted(grouped.items())]
    else:
        groups = hue_groups(colors)
        indices = target_indices(len(groups), len(palette))
        for group, index in zip(groups, indices):
            target = palette[index].upper()
            for source in group:
                mapping[source.upper()] = target
            group_log.append(f"{'/'.join(group)} -> {target}")
    return mapping, group_log


def recolor_svg(source: Path, destination: Path, palette: tuple[str, ...],
                global_hue_centers: list[float] | None = None) -> list[str]:
    text = source.read_text(encoding="utf-8")
    mapping, log = build_svg_map(text, palette, global_hue_centers)
    for old, new in mapping.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    embedded_logs: list[str] = []
    def recolor_embedded(match: re.Match[str]) -> str:
        try:
            raw = base64.b64decode(match.group(3))
            with Image.open(io.BytesIO(raw)) as embedded:
                recolored, image_log = recolor_raster(embedded, palette, global_hue_centers)
            buffer = io.BytesIO()
            recolored.save(buffer, format="PNG")
            embedded_logs.extend(image_log)
            return match.group(1) + base64.b64encode(buffer.getvalue()).decode("ascii") + match.group(4)
        except Exception as exc:
            embedded_logs.append(f"embedded image unchanged ({type(exc).__name__})")
            return match.group(0)

    text = EMBEDDED_IMAGE_RE.sub(recolor_embedded, text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return log + embedded_logs


def palette_arrays(palette: tuple[str, ...]) -> list[tuple[float, float, float]]:
    return [tuple(channel / 255 for channel in rgb(color)) for color in palette]


def recolor_raster(input_image: Image.Image, palette: tuple[str, ...],
                    global_hue_centers: list[float] | None = None) -> tuple[Image.Image, list[str]]:
    """Hue-map colored pixels, preserving each pixel's lightness and gray content."""
    import numpy as np

    image = input_image.convert("RGBA")
    arr = np.asarray(image).copy()
    rgb_arr = arr[:, :, :3].astype(np.float32) / 255.0
    r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    delta = high - low
    saturation = np.divide(delta, high, out=np.zeros_like(delta), where=high > 0)
    chromatic = (saturation >= 0.20) & (high < 0.985) & (arr[:, :, 3] > 0)

    if not np.any(chromatic):
        return image, ["no chromatic pixels found"]

    hue = np.zeros_like(high)
    nonzero = delta > 1e-6
    mask_r = nonzero & (high == r)
    mask_g = nonzero & (high == g)
    mask_b = nonzero & (high == b)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    hue *= 60

    # Build circular hue bands from pixels that actually occur in this artwork.
    bins = np.unique(np.floor(hue[chromatic]).astype(np.int16))
    if len(bins) == 1:
        centers = [float(bins[0]) + 0.5]
    else:
        gaps = np.diff(np.r_[bins, bins[0] + 360])
        split_points = [i for i, gap in enumerate(gaps) if gap > 18]
        if not split_points:
            centers = [float(np.median(hue[chromatic]))]
        else:
            raw_groups: list[list[int]] = []
            start = (split_points[0] + 1) % len(bins)
            current: list[int] = []
            idx = start
            while True:
                current.append(int(bins[idx]))
                next_idx = (idx + 1) % len(bins)
                if idx in split_points:
                    raw_groups.append(current)
                    current = []
                idx = next_idx
                if idx == start:
                    if current:
                        raw_groups.append(current)
                    break
            centers = []
            for band in raw_groups:
                angles = np.array(band, dtype=np.float32) * np.pi / 180
                centers.append(float((np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) + 360) % 360))
            centers.sort()

    palette_rgb = palette_arrays(palette)
    if global_hue_centers:
        global_indices = target_indices(len(global_hue_centers), len(palette_rgb))
        indices = [global_indices[closest_hue_index(center, global_hue_centers)] for center in centers]
    else:
        indices = target_indices(len(centers), len(palette_rgb))
    target_hues = []
    target_sats = []
    logs = []
    for center, palette_index in zip(centers, indices):
        pr, pg, pb = palette_rgb[palette_index]
        ph, ps, _ = colorsys.rgb_to_hsv(pr, pg, pb)
        target_hues.append(ph * 360)
        target_sats.append(ps)
        logs.append(f"hue {center:.0f}° -> {palette[palette_index]}")

    pixel_hues = hue[chromatic]
    distances = np.abs(pixel_hues[:, None] - np.asarray(centers)[None, :])
    distances = np.minimum(distances, 360 - distances)
    nearest = np.argmin(distances, axis=1)
    mapped_hue = np.asarray(target_hues, dtype=np.float32)[nearest] / 60.0
    mapped_sat = np.asarray(target_sats, dtype=np.float32)[nearest]
    value = high[chromatic]
    chroma = value * mapped_sat
    sector = np.floor(mapped_hue).astype(np.int16) % 6
    fraction = mapped_hue - np.floor(mapped_hue)
    x = chroma * (1 - np.abs((fraction * 2) - 1))
    zeros = np.zeros_like(chroma)
    choices = (
        (chroma, x, zeros), (x, chroma, zeros), (zeros, chroma, x),
        (zeros, x, chroma), (x, zeros, chroma), (chroma, zeros, x),
    )
    m = value - chroma
    for channel in range(3):
        vals = np.choose(sector, [choice[channel] for choice in choices]) + m
        arr[:, :, channel][chromatic] = np.clip(vals * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA"), logs


def run_checked(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def render_svg(svg_path: Path, pdf_path: Path, png_path: Path, rsvg: str, dpi: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked([rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)])
    run_checked([rsvg, "-d", str(dpi), "-p", str(dpi), "-f", "png", "-o", str(png_path), str(svg_path)])


def render_pdf_raster(pdf_path: Path, png_path: Path, gs: str, dpi: int) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="deep-sea-figure-") as temp_dir:
        rendered = Path(temp_dir) / "page.png"
        run_checked([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=png16m",
                     f"-r{dpi}", f"-sOutputFile={rendered}", str(pdf_path)])
        image = Image.open(rendered).convert("RGBA")
        image.save(png_path, dpi=(dpi, dpi))
        return image


def pdf_only_sources(group: SourceGroup, group_dir: Path) -> list[Path]:
    pdfs = sorted(group_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return []
    # Use the newest exported PDF rather than its smaller preview PNG.
    return [pdfs[0]]


def vector_sources(group_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(group_dir.rglob("*.svg")):
        if "slide_panels" in path.parts:
            continue
        paths.append(path)
    return paths


def write_palette_legend(output_root: Path) -> None:
    swatch_w, swatch_h, gap, margin = 150, 72, 24, 28
    width = 1740
    height = margin * 2 + 3 * (swatch_h + 48)
    canvas = Image.new("RGB", (width, height), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    names = ("Style 1 - Blue (5)", "Style 2 - Green (3)", "Style 3 - Blue-Green (8)")
    y = margin
    for (name, key), row in zip(zip(names, PALETTES), PALETTES.values()):
        draw.text((margin, y + 5), name, fill="#20262B")
        x = margin + 260
        for color in row:
            draw.rounded_rectangle((x, y, x + swatch_w, y + swatch_h), radius=14, fill=color, outline="#D5DDE2", width=2)
            draw.text((x + 12, y + swatch_h + 8), color, fill="#333333")
            x += swatch_w + gap
        y += swatch_h + 48
    canvas.save(output_root / "palette_legend.png")


def process_svg(source: Path, output_root: Path, group_label: str, relative_path: Path,
                palette_key: str, dpi: int, rsvg: str, manifest_rows: list[list[str]],
                global_hue_centers: list[float] | None = None) -> None:
    palette = PALETTES[palette_key]
    dest_base = output_root / palette_key / group_label / relative_path.with_suffix("")
    dest_svg = dest_base.with_suffix(".svg")
    dest_pdf = dest_base.with_suffix(".pdf")
    dest_png = dest_base.with_suffix(".png")
    log = recolor_svg(source, dest_svg, palette, global_hue_centers)
    render_svg(dest_svg, dest_pdf, dest_png, rsvg, dpi)
    manifest_rows.append([palette_key, group_label, display_path(source),
                          display_path(dest_svg), display_path(dest_pdf),
                          display_path(dest_png), "; ".join(log)])
    print(f"[{palette_key}] {group_label}/{relative_path}", flush=True)


def process_pdf(source: Path, output_root: Path, group_label: str, palette_key: str,
                dpi: int, gs: str, manifest_rows: list[list[str]],
                global_hue_centers: list[float] | None = None) -> None:
    palette = PALETTES[palette_key]
    dest_base = output_root / palette_key / group_label / source.stem
    dest_png = dest_base.with_suffix(".png")
    dest_pdf = dest_base.with_suffix(".pdf")
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    original = render_pdf_raster(source, dest_png, gs, dpi)
    recolored, log = recolor_raster(original, palette, global_hue_centers)
    recolored.convert("RGB").save(dest_png, dpi=(dpi, dpi), optimize=True)
    recolored.convert("RGB").save(dest_pdf, "PDF", resolution=dpi)
    manifest_rows.append([palette_key, group_label, display_path(source), "", 
                          display_path(dest_pdf), display_path(dest_png),
                          "; ".join(log)])
    print(f"[{palette_key}] {group_label}/{source.name}", flush=True)


def process_raster(source: Path, output_root: Path, group_label: str, relative_path: Path,
                   palette_key: str, manifest_rows: list[list[str]],
                   global_hue_centers: list[float] | None = None) -> None:
    dest = output_root / palette_key / group_label / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    palette = PALETTES[palette_key]
    with Image.open(source) as original:
        recolored, log = recolor_raster(original, palette, global_hue_centers)
        suffix = source.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            recolored.convert("RGB").save(dest, quality=95, optimize=True)
        else:
            recolored.save(dest)
    manifest_rows.append([palette_key, group_label, display_path(source), "",
                          "", display_path(dest), "; ".join(log)])
    print(f"[{palette_key}] {group_label}/{relative_path}", flush=True)


def collect_custom_sources(input_dir: Path) -> list[tuple[str, Path, Path]]:
    """Collect custom input files, preferring SVG/PDF over duplicate previews."""
    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in
                   {".svg", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    svg_stems = {p.with_suffix("") for p in files if p.suffix.lower() == ".svg"}
    pdf_stems = {p.with_suffix("") for p in files if p.suffix.lower() == ".pdf"}
    results = []
    for path in files:
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if path.with_suffix("") in svg_stems or (path.suffix.lower() != ".pdf" and path.with_suffix("") in pdf_stems):
                continue
        results.append(("figures", path, path.relative_to(input_dir)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palette", choices=[*PALETTES, "all"], default="all",
                        help="which palette to render (default: all three)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--input-dir", type=Path,
                        help="optional custom folder; recursively recolor SVGs and standalone raster/PDF files")
    parser.add_argument("--png-dpi", type=int, default=180, help="preview/render resolution (default: 180)")
    parser.add_argument("--list-palettes", action="store_true", help="print the palettes and exit")
    args = parser.parse_args()
    if args.list_palettes:
        for name, colors in PALETTES.items():
            print(f"{name}: {' '.join(colors)}")
        return 0
    if args.png_dpi < 72:
        parser.error("--png-dpi must be at least 72")

    output_root = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    palette_keys = list(PALETTES) if args.palette == "all" else [args.palette]
    rsvg = shutil.which("rsvg-convert")
    gs = shutil.which("gs")
    manifest_rows: list[list[str]] = []

    if args.input_dir:
        input_dir = args.input_dir.resolve() if args.input_dir.is_absolute() else (ROOT / args.input_dir).resolve()
        if not input_dir.is_dir():
            parser.error(f"input directory does not exist: {input_dir}")
        custom_sources = collect_custom_sources(input_dir)
        global_hue_centers = collect_global_hue_centers(
            [source for _, source, _ in custom_sources if source.suffix.lower() == ".svg"]
        )
        for group_label, source, relative in custom_sources:
            for palette_key in palette_keys:
                suffix = source.suffix.lower()
                if suffix == ".svg":
                    if not rsvg:
                        parser.error("rsvg-convert is required to render SVG-derived PDF/PNG exports")
                    process_svg(source, output_root, group_label, relative, palette_key, args.png_dpi, rsvg,
                                manifest_rows, global_hue_centers)
                elif suffix == ".pdf":
                    if not gs:
                        parser.error("Ghostscript (gs) is required to recolor PDF-only inputs")
                    process_pdf(source, output_root, group_label, palette_key, args.png_dpi, gs,
                                manifest_rows, global_hue_centers)
                else:
                    process_raster(source, output_root, group_label, relative, palette_key, manifest_rows,
                                   global_hue_centers)
    else:
        all_vector_sources: list[Path] = []
        for group in DEFAULT_SOURCES:
            if not group.pdf_only:
                source_dir = ROOT / group.relative_dir
                if source_dir.is_dir():
                    all_vector_sources.extend(vector_sources(source_dir))
        global_hue_centers = collect_global_hue_centers(all_vector_sources)
        for group in DEFAULT_SOURCES:
            group_dir = ROOT / group.relative_dir
            if not group_dir.is_dir():
                print(f"[skip] missing source folder: {group.relative_dir}", file=sys.stderr)
                continue
            if group.pdf_only:
                sources = pdf_only_sources(group, group_dir)
                if not sources:
                    print(f"[skip] no PDF in {group.relative_dir}", file=sys.stderr)
                    continue
                if not gs:
                    parser.error("Ghostscript (gs) is required to recolor PDF-only inputs")
                for source in sources:
                    for palette_key in palette_keys:
                        process_pdf(source, output_root, group.label, palette_key, args.png_dpi, gs,
                                    manifest_rows, global_hue_centers)
            else:
                sources = vector_sources(group_dir)
                if not sources:
                    print(f"[skip] no SVG files in {group.relative_dir}", file=sys.stderr)
                    continue
                if not rsvg:
                    parser.error("rsvg-convert is required to render SVG-derived PDF/PNG exports")
                for source in sources:
                    relative = source.relative_to(group_dir)
                    for palette_key in palette_keys:
                        process_svg(source, output_root, group.label, relative, palette_key, args.png_dpi,
                                    rsvg, manifest_rows, global_hue_centers)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["palette", "figure_group", "source", "output_svg", "output_pdf", "output_png", "mapped_colors"])
        writer.writerows(manifest_rows)
    write_palette_legend(output_root)
    print(f"Wrote {len(manifest_rows)} recolored assets to {output_root}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
