"""Atom-level Q-score (Pintilie et al., Nat. Methods 2020) for frozen poses.

This is a dependency-light re-implementation that follows the ChimeraX
``qscore`` bundle (tristanic/chimerax-qscore) as closely as practical:

* shells from 0 to 2.0 A in 0.1 A steps, 8 probe points per shell;
* a probe point is kept only if the query atom is its nearest heavy atom in
  the *whole* atomic model (protein context + ligand);
* shells with more than 8 accepted points are reduced to 8 by spherical
  k-means over 128 trial directions; shells with fewer keep all of them;
* the r = 0 value is the density at the atom centre, repeated 8 times;
* reference Gaussian ``A exp(-r^2 / 2 sigma^2) + B`` with sigma = 0.6 A,
  ``B = max(mean - sd, min)`` and ``A = min(mean + 10 sd, max) - B`` computed
  over the full map;
* Q is the Pearson (about-mean) correlation of the measured and reference
  profiles.

Sphere directions and k-means seeding differ from ChimeraX, so individual
atom values can differ in the second decimal; use the ChimeraX cross-check
in ``paahs_figure5_mode_qscore.py compare-chimerax`` before relying on the
absolute scale.  Coordinates are never moved or fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Sequence

import numpy as np
import pandas as pd

WATER_RESIDUES = {"HOH", "WAT", "DOD", "H2O", "SOL", "TIP", "TIP3"}
HYDROGEN_ELEMENTS = {"H", "D"}


@dataclass(frozen=True)
class DensityMap:
    """Orthogonal map in x, y, z index order with an absolute frame."""

    data: np.ndarray
    origin: np.ndarray
    voxel_size: np.ndarray
    frame_rule: str
    header_nstart_xyz: tuple
    header_origin_xyz: tuple
    mean: float
    sd: float
    minimum: float
    maximum: float

    def reference_bounds(self) -> tuple:
        """MapQ / ChimeraX reference Gaussian bounds (min_d, max_d)."""
        max_d = min(self.mean + 10.0 * self.sd, self.maximum)
        min_d = max(self.mean - self.sd, self.minimum)
        return float(min_d), float(max_d)

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """Trilinear interpolation; points outside the box return NaN."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        grid = (points - self.origin) / self.voxel_size
        shape = np.asarray(self.data.shape)
        inside = np.all((grid >= 0.0) & (grid <= shape - 1), axis=1)
        values = np.full(len(points), np.nan, dtype=np.float64)
        if not inside.any():
            return values
        g = grid[inside]
        lower = np.minimum(np.floor(g).astype(np.int64), shape - 2)
        lower = np.maximum(lower, 0)
        t = g - lower
        i, j, k = lower.T
        tx, ty, tz = t.T
        d = self.data
        c000 = d[i, j, k]
        c100 = d[i + 1, j, k]
        c010 = d[i, j + 1, k]
        c110 = d[i + 1, j + 1, k]
        c001 = d[i, j, k + 1]
        c101 = d[i + 1, j, k + 1]
        c011 = d[i, j + 1, k + 1]
        c111 = d[i + 1, j + 1, k + 1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        values[inside] = c0 * (1 - tz) + c1 * tz
        return values


def _read_ccp4(path: Path):
    import gemmi

    if path.suffix.lower() != ".gz":
        return gemmi.read_ccp4_map(str(path))
    with tempfile.NamedTemporaryFile(suffix=".map") as temporary:
        with gzip.open(path, "rb") as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        return gemmi.read_ccp4_map(str(temporary.name))


def read_density_map(path: Path, origin_mode: str = "auto") -> DensityMap:
    """Read a CCP4/MRC map without resampling or masking.

    ``origin_mode`` decides how the absolute frame is built:
    ``nstart`` uses NCSTART/NRSTART/NSSTART (gemmi and the PAAHS NCC code),
    ``origin`` uses the ORIGIN header words, ``sum`` adds both, and ``auto``
    uses whichever one is non-zero and refuses maps where both are non-zero.
    EMDB maps have ORIGIN = 0, so ``auto`` equals the PAAHS convention.
    """
    ccp4 = _read_ccp4(Path(path))
    header = [ccp4.header_i32(word) for word in range(1, 20)]
    nc, nr, ns = header[0:3]
    nstart_crs = header[4:7]
    mx, my, mz = header[7:10]
    mapc, mapr, maps = header[16:19]
    cell = ccp4.grid.unit_cell
    if any(abs(angle - 90.0) > 1e-3 for angle in (cell.alpha, cell.beta, cell.gamma)):
        raise ValueError(f"Non-orthogonal map cell is not supported: {path}")
    axis_of_crs = (mapc, mapr, maps)
    if sorted(axis_of_crs) != [1, 2, 3]:
        raise ValueError(f"Invalid MAPC/MAPR/MAPS {axis_of_crs}: {path}")

    raw = np.array(ccp4.grid, copy=True, dtype=np.float32)
    if raw.shape != (nc, nr, ns):
        raise ValueError(f"Unexpected raw grid shape {raw.shape} vs header {(nc, nr, ns)}")
    order = [axis_of_crs.index(axis + 1) for axis in range(3)]
    data = np.ascontiguousarray(np.transpose(raw, order))
    nstart_xyz = np.asarray([nstart_crs[order[axis]] for axis in range(3)], dtype=np.float64)
    voxel = np.asarray([cell.a / mx, cell.b / my, cell.c / mz], dtype=np.float64)
    header_origin = np.asarray([ccp4.header_float(word) for word in (50, 51, 52)], dtype=np.float64)

    has_nstart = bool(np.any(nstart_xyz != 0))
    has_origin = bool(np.any(np.abs(header_origin) > 1e-6))
    if origin_mode == "auto":
        if has_nstart and has_origin:
            raise ValueError(
                f"Both NSTART {nstart_xyz.tolist()} and ORIGIN {header_origin.tolist()} are "
                f"non-zero in {path}; choose --origin-mode explicitly"
            )
        origin_mode = "origin" if has_origin else "nstart"
    if origin_mode == "nstart":
        origin = nstart_xyz * voxel
    elif origin_mode == "origin":
        origin = header_origin.copy()
    elif origin_mode == "sum":
        origin = header_origin + nstart_xyz * voxel
    else:
        raise ValueError(f"Unknown origin_mode {origin_mode}")

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"Map has no finite values: {path}")
    return DensityMap(
        data=data,
        origin=origin,
        voxel_size=voxel,
        frame_rule=origin_mode,
        header_nstart_xyz=tuple(float(v) for v in nstart_xyz),
        header_origin_xyz=tuple(float(v) for v in header_origin),
        mean=float(finite.mean(dtype=np.float64)),
        sd=float(finite.std(dtype=np.float64)),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
    )


def read_sdf_heavy_atoms(path: Path) -> pd.DataFrame:
    """First molecule of a V2000/V3000 SDF: heavy-atom index, element, xyz.

    ``atom_index`` is the 1-based SDF atom index so atoms stay comparable
    across poses written from the same molecule template.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"Invalid SDF: {path}")
    counts = lines[3]
    records = []
    if "V3000" in counts:
        inside = False
        for line in lines[4:]:
            text = line[6:].strip() if line.startswith("M  V30") else None
            if text is None:
                if line.startswith("M  END"):
                    break
                continue
            if text.startswith("BEGIN ATOM"):
                inside = True
                continue
            if text.startswith("END ATOM"):
                break
            if inside:
                fields = text.split()
                records.append((int(fields[0]), fields[1], *map(float, fields[2:5])))
    else:
        atom_count = int(counts[:3])
        for offset, line in enumerate(lines[4:4 + atom_count], start=1):
            records.append((
                offset,
                line[31:34].strip(),
                float(line[0:10]),
                float(line[10:20]),
                float(line[20:30]),
            ))
    frame = pd.DataFrame(records, columns=["atom_index", "element", "x", "y", "z"])
    frame["element"] = frame.element.str.strip().str.capitalize()
    frame = frame[~frame.element.str.upper().isin(HYDROGEN_ELEMENTS)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No heavy atoms in {path}")
    return frame


def read_protein_heavy_atoms(path: Path) -> pd.DataFrame:
    """Heavy atoms of the first model, keeping HETATM and water records."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    rows = []
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                if atom.element.name.upper() in HYDROGEN_ELEMENTS:
                    continue
                rows.append((
                    chain.name,
                    residue.seqid.num,
                    residue.seqid.icode.strip(),
                    residue.name.strip().upper(),
                    residue.het_flag == "H",
                    atom.name.strip(),
                    atom.element.name,
                    atom.pos.x,
                    atom.pos.y,
                    atom.pos.z,
                ))
    frame = pd.DataFrame(rows, columns=[
        "chain", "resnum", "icode", "resname", "hetatm", "atom_name", "element", "x", "y", "z",
    ])
    if frame.empty:
        raise ValueError(f"No heavy atoms in {path}")
    return frame


def drop_target_ligand_copies(
    protein: pd.DataFrame,
    reference_coords: Optional[np.ndarray],
    distance: float = 1.5,
    min_fraction: float = 0.5,
) -> tuple:
    """Remove non-water HETATM residues that duplicate the target ligand.

    A residue is removed when at least ``min_fraction`` of its heavy atoms
    lie within ``distance`` A of the deposited ligand.  Returns
    ``(filtered_frame, removed_residue_labels)``.
    """
    if reference_coords is None or len(reference_coords) == 0:
        return protein, []
    from scipy.spatial import cKDTree

    tree = cKDTree(reference_coords)
    candidates = protein[protein.hetatm & ~protein.resname.isin(WATER_RESIDUES)]
    removed = []
    drop_index = []
    for key, residue in candidates.groupby(["chain", "resnum", "icode", "resname"], sort=False):
        nearest, _ = tree.query(residue[["x", "y", "z"]].to_numpy())
        if np.mean(nearest <= distance) >= min_fraction:
            removed.append(f"{key[0]}:{key[3]}{key[1]}{key[2]}")
            drop_index.extend(residue.index.tolist())
    return protein.drop(index=drop_index).reset_index(drop=True), removed


def unit_sphere_points(count: int) -> np.ndarray:
    """Deterministic, near-uniform unit vectors (Fibonacci lattice)."""
    if count == 1:
        return np.asarray([[0.0, 0.0, 1.0]])
    index = np.arange(count, dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([
        np.cos(azimuth) * np.sin(polar),
        np.sin(azimuth) * np.sin(polar),
        np.cos(polar),
    ], axis=1)


def spherical_kmeans_representatives(
    directions: np.ndarray, clusters: int, iterations: int, seed: int
) -> np.ndarray:
    """Pick ``clusters`` well-spread unit vectors; returns their row indices."""
    rng = np.random.default_rng(seed)
    centroids = directions[rng.choice(len(directions), size=clusters, replace=False)].copy()
    for _ in range(max(1, iterations)):
        labels = np.argmax(directions @ centroids.T, axis=1)
        for cluster in range(clusters):
            members = directions[labels == cluster]
            if len(members):
                mean = members.sum(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1e-12:
                    centroids[cluster] = mean / norm
    chosen = []
    similarity = directions @ centroids.T
    for cluster in range(clusters):
        order = np.argsort(-similarity[:, cluster])
        for candidate in order:
            if candidate not in chosen:
                chosen.append(int(candidate))
                break
    return np.asarray(chosen, dtype=np.int64)


@dataclass(frozen=True)
class QScoreParameters:
    reference_sigma: float = 0.6
    max_shell_radius: float = 2.0
    shell_step: float = 0.1
    points_per_shell: int = 8
    trial_points: int = 128
    quick_shell_limit: float = 0.7
    kmeans_iterations: int = 5
    random_seed: int = 1985

    def radii(self) -> np.ndarray:
        return np.arange(0.0, self.max_shell_radius + self.shell_step / 2.0, self.shell_step)


def atom_qscores(
    density: DensityMap,
    context_coords: np.ndarray,
    query_indices: Sequence[int],
    parameters: QScoreParameters = QScoreParameters(),
) -> pd.DataFrame:
    """Q-score of ``context_coords[query_indices]``.

    ``context_coords`` must contain every heavy atom of the single atomic
    model (protein context and the scored ligand), because probe points are
    assigned to their nearest atom in that model.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(context_coords, dtype=np.float64)
    tree = cKDTree(coords)
    radii = parameters.radii()
    min_d, max_d = density.reference_bounds()
    reference = (max_d - min_d) * np.exp(-0.5 * (radii / parameters.reference_sigma) ** 2) + min_d
    quick = unit_sphere_points(parameters.points_per_shell)
    trial = unit_sphere_points(parameters.trial_points)
    per_shell = parameters.points_per_shell

    rows = []
    for query in query_indices:
        centre = coords[query]
        neighbours = np.asarray(tree.query_ball_point(centre, parameters.max_shell_radius * 3.0))
        others = coords[neighbours[neighbours != query]]
        measured = [np.repeat(density.interpolate(centre[None, :]), per_shell)]
        expected = [np.repeat(reference[0], per_shell)]
        short_shells = 0
        empty_shells = 0
        for shell, radius in enumerate(radii[1:], start=1):
            accepted = None
            if radius < parameters.quick_shell_limit:
                points = centre + quick * radius
                if _owned_by_query(points, centre, others).all():
                    accepted = points
            if accepted is None:
                points = centre + trial * radius
                mask = _owned_by_query(points, centre, others)
                if mask.sum() > per_shell:
                    chosen = spherical_kmeans_representatives(
                        trial[mask], per_shell, parameters.kmeans_iterations,
                        parameters.random_seed + shell,
                    )
                    accepted = points[mask][chosen]
                else:
                    accepted = points[mask]
            if len(accepted) < per_shell:
                short_shells += 1
            if len(accepted) == 0:
                empty_shells += 1
                continue
            measured.append(density.interpolate(accepted))
            expected.append(np.repeat(reference[shell], len(accepted)))
        measured = np.concatenate(measured)
        expected = np.concatenate(expected)
        missing = int(np.isnan(measured).sum())
        q_value = np.nan
        if missing == 0:
            dm = measured - measured.mean()
            de = expected - expected.mean()
            denominator = np.sqrt(np.dot(dm, dm) * np.dot(de, de))
            if denominator > 0:
                q_value = float(np.dot(dm, de) / denominator)
        rows.append({
            "context_index": int(query),
            "qscore": q_value,
            "density_at_atom": float(measured[0]) if np.isfinite(measured[0]) else np.nan,
            "n_probe_points": int(len(measured)),
            "n_points_outside_map": missing,
            "n_short_shells": short_shells,
            "n_empty_shells": empty_shells,
        })
    return pd.DataFrame(rows)


def _owned_by_query(points: np.ndarray, centre: np.ndarray, others: np.ndarray) -> np.ndarray:
    """True where the query atom is the nearest atom to each probe point."""
    own = np.linalg.norm(points - centre, axis=1)
    if len(others) == 0:
        return np.ones(len(points), dtype=bool)
    distances = np.linalg.norm(points[:, None, :] - others[None, :, :], axis=2).min(axis=1)
    return own < distances
