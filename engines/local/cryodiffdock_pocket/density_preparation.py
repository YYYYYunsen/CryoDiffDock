"""Prepare Cartesian experimental-density crops for local docking."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import gzip
import shutil
import tempfile

import numpy as np


STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX",
}

NUCLEIC_ACID_RESIDUES = {
    "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DI",
    "ADE", "CYT", "GUA", "THY", "URA",
}


def _entity_type_name(residue) -> str:
    try:
        return str(residue.entity_type).split(".")[-1]
    except Exception:
        return "Unknown"


def polymer_category(residue) -> Optional[str]:
    """Classify polymer residues without treating ligands as mask atoms."""
    name = residue.name.strip().upper()
    if name in STANDARD_PROTEIN_RESIDUES:
        return "protein_polymer"
    if name in NUCLEIC_ACID_RESIDUES:
        return "nucleic_acid"
    # Deposited mmCIF entity typing is authoritative for unusual polymers.
    # HETATM-like residues must not be promoted to polymers using PDB-only
    # heuristics because that would accidentally mask the target ligand.
    if _entity_type_name(residue) == "Polymer" and residue.het_flag != "H":
        return "other_polymer"
    return None


def cartesian_axes(
    center: np.ndarray, box_size: float, spacing: float
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if box_size <= 0 or spacing <= 0:
        raise ValueError("box_size and spacing must be positive")
    center = np.asarray(center, dtype=np.float32).reshape(3)
    points_per_axis = max(2, int(np.ceil(float(box_size) / float(spacing))) + 1)
    covered_size = float(points_per_axis - 1) * float(spacing)
    origin = center - covered_size / 2.0
    axes = tuple(
        origin[axis] + np.arange(points_per_axis, dtype=np.float32) * float(spacing)
        for axis in range(3)
    )
    return origin.astype(np.float32), axes


def sample_map(map_path: Path, axes: Sequence[np.ndarray]) -> np.ndarray:
    """Sample a CCP4/MRC map on an absolute Cartesian grid."""
    import gemmi

    def sample_uncompressed(path: Path) -> np.ndarray:
        ccp4 = gemmi.read_ccp4_map(str(path))
        ccp4.setup(float("nan"))
        grid = ccp4.grid
        cell = grid.unit_cell
        x_axis, y_axis, z_axis = axes
        density = np.empty((len(x_axis), len(y_axis), len(z_axis)), dtype=np.float32)
        for i, x_value in enumerate(x_axis):
            for j, y_value in enumerate(y_axis):
                for k, z_value in enumerate(z_axis):
                    fractional = cell.fractionalize(
                        gemmi.Position(float(x_value), float(y_value), float(z_value))
                    )
                    try:
                        density[i, j, k] = grid.interpolate_value(fractional)
                    except (RuntimeError, ValueError):
                        density[i, j, k] = np.nan
        return density

    if map_path.suffix.lower() != ".gz":
        return sample_uncompressed(map_path)
    with tempfile.NamedTemporaryFile(suffix=".map") as temporary:
        with gzip.open(map_path, "rb") as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        return sample_uncompressed(Path(temporary.name))


def normalization(density: np.ndarray):
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise ValueError("Requested Cartesian crop does not overlap finite map density")
    center = float(np.median(finite))
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError(
            "Density crop has near-zero variance; check the map frame, pocket center, and box size"
        )
    clean = np.nan_to_num(density, nan=center, posinf=center, neginf=center)
    return clean.astype(np.float32), center, scale


def read_ligand_heavy_coordinates(path: Path) -> np.ndarray:
    """Read reference coordinates. Call only during oracle prep/evaluation."""
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        if path.suffix.lower() not in {".sdf", ".mol"}:
            raise
        lines = path.read_text().splitlines()
        if len(lines) < 5:
            raise ValueError(f"Invalid V2000 ligand file: {path}")
        try:
            atom_count = int(lines[3][:3])
            atom_lines = lines[4:4 + atom_count]
            coordinates = [
                (float(line[:10]), float(line[10:20]), float(line[20:30]))
                for line in atom_lines
                if line[31:34].strip().upper() not in {"H", "D"}
            ]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Could not parse V2000 ligand coordinates: {path}") from exc
        if not coordinates:
            raise ValueError(f"Reference ligand has no heavy atoms: {path}")
        return np.asarray(coordinates, dtype=np.float32)

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported reference ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read reference ligand coordinates: {path}")
    conformer = molecule.GetConformer()
    coordinates = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append((point.x, point.y, point.z))
    if not coordinates:
        raise ValueError(f"Reference ligand has no heavy atoms: {path}")
    return np.asarray(coordinates, dtype=np.float32)


def read_protein_heavy_coordinates(path: Path) -> np.ndarray:
    """Read polymer protein atoms while excluding ligand/water HETATM."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name.upper() not in STANDARD_PROTEIN_RESIDUES:
                    continue
                for atom in residue:
                    if atom.element.name in {"H", "D"}:
                        continue
                    coordinates.append((atom.pos.x, atom.pos.y, atom.pos.z))
    if not coordinates:
        raise ValueError(f"No protein heavy atoms found: {path}")
    return np.asarray(coordinates, dtype=np.float32)


def read_full_polymer_heavy_coordinates(path: Path):
    """Read every modeled polymer heavy atom from a complete PDB/mmCIF.

    Returns coordinates plus provenance counts.  Protein, nucleic-acid and
    other-polymer atoms are included; waters, ions and all non-polymers are
    deliberately retained in the density as possible hard negatives.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    try:
        structure.setup_entities()
        structure.add_entity_types(overwrite=False)
    except Exception:
        # Standard protein/nucleic residue-name fallbacks still work for PDBs
        # or incomplete mmCIF entity tables.
        pass
    if not len(structure):
        raise ValueError(f"Mask structure contains no model: {path}")

    coordinates = []
    category_counts = {
        "protein_polymer": 0,
        "nucleic_acid": 0,
        "other_polymer": 0,
    }
    for chain in structure[0]:
        for residue in chain:
            category = polymer_category(residue)
            if category is None:
                continue
            for atom in residue:
                if atom.element.name.upper() in {"H", "D", "T"}:
                    continue
                coordinates.append((atom.pos.x, atom.pos.y, atom.pos.z))
                category_counts[category] += 1
    if not coordinates:
        raise ValueError(f"No polymer heavy atoms found: {path}")
    return np.asarray(coordinates, dtype=np.float32), category_counts


def soft_protein_mask(
    axes: Sequence[np.ndarray],
    protein_coordinates: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """Return a smooth 0-to-1 protein-exclusion weight."""
    from scipy.spatial import cKDTree

    if inner_radius < 0 or outer_radius <= inner_radius:
        raise ValueError("Require 0 <= mask_inner < mask_outer")
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distances, _ = cKDTree(np.asarray(protein_coordinates)).query(mesh.reshape(-1, 3), k=1)
    fraction = np.clip(
        (distances - float(inner_radius)) / (float(outer_radius) - float(inner_radius)),
        0.0,
        1.0,
    )
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return smooth.reshape(mesh.shape[:-1]).astype(np.float32)


def hard_protein_mask(
    axes: Sequence[np.ndarray],
    protein_coordinates: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return a binary exclusion mask: zero within ``radius``, one outside."""
    from scipy.spatial import cKDTree

    if radius <= 0:
        raise ValueError("mask radius must be positive")
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distances, _ = cKDTree(np.asarray(protein_coordinates)).query(
        mesh.reshape(-1, 3), k=1
    )
    return (distances > float(radius)).reshape(mesh.shape[:-1]).astype(np.float32)


def apply_protein_mask(
    density: np.ndarray,
    mask_weight: np.ndarray,
    positive_only: bool = True,
) -> Tuple[np.ndarray, float]:
    """Suppress protein-proximal signal without using ligand shape."""
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise ValueError("Density contains no finite values")
    background = float(np.median(finite))
    signal = np.nan_to_num(density - background, nan=0.0, posinf=0.0, neginf=0.0)
    if positive_only:
        signal = np.maximum(signal, 0.0)
    return (background + signal * mask_weight).astype(np.float32), background


@dataclass(frozen=True)
class DensityComponentSegmentation:
    density: np.ndarray
    support: np.ndarray
    status: str
    selected_label: int
    num_components: int
    component_voxels: int
    support_voxels: int
    seed_distance_A: float
    retained_positive_signal_fraction: float


def segment_pocket_density_component(
    density: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    center: np.ndarray,
    low_z: float = 0.5,
    high_z: float = 1.5,
    seed_radius_A: float = 4.0,
    dilation_A: float = 2.0,
    max_seed_distance_A: float = 6.0,
    min_component_voxels: int = 16,
) -> DensityComponentSegmentation:
    """Select one density component using only a supplied pocket centre.

    A low threshold defines growable components and a high threshold identifies
    a reliable seed near the pocket centre.  The selected component is dilated
    in physical units to retain weak density tails.  No ligand atoms, candidate
    coordinates or reference RMSD are read by this operation.
    """
    from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

    density = np.asarray(density, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float32)
    if spacing.ndim == 0:
        spacing = np.repeat(float(spacing), 3)
    spacing = spacing.reshape(3)
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if density.ndim != 3 or min(density.shape) < 2:
        raise ValueError(f"density must be a non-trivial 3D array, got {density.shape}")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"invalid spacing: {spacing}")
    if not 0 <= low_z < high_z:
        raise ValueError("Require 0 <= low_z < high_z")
    if seed_radius_A <= 0 or dilation_A < 0 or max_seed_distance_A < 0:
        raise ValueError("seed radius must be positive and distances non-negative")
    if min_component_voxels <= 0:
        raise ValueError("min_component_voxels must be positive")

    finite = density[np.isfinite(density)]
    if not finite.size:
        raise ValueError("density contains no finite values")
    background = float(np.median(finite))
    scale = float(np.std(finite))
    clean = np.nan_to_num(density, nan=background, posinf=background, neginf=background)
    if not np.isfinite(scale) or scale <= 1e-6:
        support = np.ones(density.shape, dtype=bool)
        return DensityComponentSegmentation(
            clean, support, "fallback_near_zero_variance", 0, 0,
            int(support.sum()), int(support.sum()), float("nan"), 1.0,
        )

    z = (clean - background) / scale
    positive_signal = np.maximum(clean - background, 0.0)
    low_mask = z >= float(low_z)
    labels, num_components = label(
        low_mask, structure=generate_binary_structure(3, 3)
    )
    if num_components == 0:
        support = np.ones(density.shape, dtype=bool)
        return DensityComponentSegmentation(
            clean, support, "fallback_no_component", 0, 0,
            int(support.sum()), int(support.sum()), float("nan"), 1.0,
        )

    grid_axes = [
        origin[axis]
        + np.arange(density.shape[axis], dtype=np.float32) * spacing[axis]
        for axis in range(3)
    ]
    squared_distance = None
    for axis, values in enumerate(grid_axes):
        shape = [1, 1, 1]
        shape[axis] = len(values)
        term = (values.reshape(shape) - center[axis]) ** 2
        squared_distance = term if squared_distance is None else squared_distance + term
    distance_to_center = np.sqrt(squared_distance)
    seed_region = distance_to_center <= float(seed_radius_A)

    # Prefer the component containing the strongest high-confidence density
    # inside the seed sphere.  This selects one component rather than merging
    # every local blob, which would reintroduce nearby lipid/unmodeled density.
    seed_candidates = np.where(seed_region & (z >= float(high_z)) & (labels > 0))
    status = "selected_high_z_seed"
    selected_label = 0
    if seed_candidates[0].size:
        seed_values = z[seed_candidates]
        strongest = int(np.argmax(seed_values))
        index = tuple(axis[strongest] for axis in seed_candidates)
        selected_label = int(labels[index])
    else:
        local_candidates = np.where(seed_region & (labels > 0))
        if local_candidates[0].size:
            local_values = z[local_candidates]
            strongest = int(np.argmax(local_values))
            index = tuple(axis[strongest] for axis in local_candidates)
            selected_label = int(labels[index])
            status = "selected_low_z_seed"
        else:
            component_distances = []
            for component_label in range(1, int(num_components) + 1):
                component_distances.append((
                    float(distance_to_center[labels == component_label].min()),
                    component_label,
                ))
            seed_distance, selected_label = min(component_distances)
            status = "selected_nearest_component"
            if seed_distance > float(max_seed_distance_A):
                support = np.ones(density.shape, dtype=bool)
                return DensityComponentSegmentation(
                    clean, support, "fallback_component_too_far", 0,
                    int(num_components), int(support.sum()), int(support.sum()),
                    seed_distance, 1.0,
                )

    component = labels == selected_label
    component_voxels = int(component.sum())
    seed_distance = float(distance_to_center[component].min())
    if component_voxels < int(min_component_voxels):
        support = np.ones(density.shape, dtype=bool)
        return DensityComponentSegmentation(
            clean, support, "fallback_component_too_small", selected_label,
            int(num_components), component_voxels, int(support.sum()),
            seed_distance, 1.0,
        )

    if dilation_A > 0:
        support = distance_transform_edt(~component, sampling=spacing) <= float(dilation_A)
    else:
        support = component.copy()
    segmented = background + positive_signal * support
    total_signal = float(positive_signal.sum())
    retained = (
        float((positive_signal * support).sum()) / total_signal
        if total_signal > 1e-12 else 1.0
    )
    return DensityComponentSegmentation(
        segmented.astype(np.float32), support.astype(bool), status,
        selected_label, int(num_components), component_voxels,
        int(support.sum()), seed_distance, retained,
    )


def fraction_in_crop(
    coordinates: np.ndarray, origin: np.ndarray, shape, spacing: float
) -> float:
    upper = origin + (np.asarray(shape, dtype=np.float32) - 1.0) * float(spacing)
    inside = ((coordinates >= origin) & (coordinates <= upper)).all(axis=1)
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Model-density subtraction (difference / residual maps)
#
# Masking suppresses voxels near the protein.  It cannot separate the ligand
# from the protein where the two are in contact, and it leaves the protein's
# density shoulder beyond the mask radius inside the correlation.  Subtraction
# instead simulates the density of everything that *is* modelled and removes
# it, so that contact-region ligand signal survives.
#
# Synthetic probes (evaluation/probe_synthetic_gradient_conditions.py and
# evaluation/probe_unmodelled_density.py) indicate the rotational density
# gradient is dominated by density that *surrounds* the ligand: modelled
# protein costs most of it, whereas unmodelled blobs beside the ligand -- lipid
# tails, detergent, disordered loops -- cost almost nothing.  Subtraction is
# therefore worthwhile even though unmodelled density cannot be removed.
# ---------------------------------------------------------------------------

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT", "CB"}


@dataclass(frozen=True)
class ModelSubtraction:
    """Result of removing simulated model density from an experimental crop."""

    residual: np.ndarray
    sigma_A: float
    scale: float
    offset: float
    correlation: float
    atom_count: int
    excluded_residues: Tuple[str, ...]
    fit_voxels: int


def read_modelled_heavy_atoms(
    path: Path,
    centre: np.ndarray,
    ligand_exclusion_radius_A: float = 6.0,
    backbone_only: bool = False,
    exclude_resnames: Sequence[str] = (),
):
    """Read every modelled heavy atom except the docking target.

    Subtraction must never remove the target ligand's own density, so any
    non-polymer residue whose centroid falls within
    ``ligand_exclusion_radius_A`` of the crop centre is left in the map.  The
    bias is deliberate: failing to subtract a nearby lipid costs almost nothing,
    whereas subtracting the target is fatal.

    Polymer residues are always subtracted -- a protein residue near the centre
    lines the pocket, it is not the ligand.
    """
    import gemmi

    if ligand_exclusion_radius_A < 0:
        raise ValueError("ligand exclusion radius must be non-negative")
    centre = np.asarray(centre, dtype=np.float32).reshape(3)
    excluded_names = {name.strip().upper() for name in exclude_resnames if name.strip()}

    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    try:
        structure.setup_entities()
        structure.add_entity_types(overwrite=False)
    except Exception:
        pass
    if not len(structure):
        raise ValueError(f"Subtraction structure contains no model: {path}")

    coordinates, numbers, excluded = [], [], []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.upper()
            heavy = [a for a in residue if a.element.name not in {"H", "D"}]
            if not heavy:
                continue
            is_polymer = (
                name in STANDARD_PROTEIN_RESIDUES
                or name in NUCLEIC_ACID_RESIDUES
                or polymer_category(residue) is not None
            )
            if name in excluded_names:
                excluded.append(name)
                continue
            if not is_polymer:
                centroid = np.asarray(
                    [[a.pos.x, a.pos.y, a.pos.z] for a in heavy], dtype=np.float32
                ).mean(axis=0)
                if np.linalg.norm(centroid - centre) <= float(ligand_exclusion_radius_A):
                    excluded.append(name)
                    continue
            for atom in heavy:
                if backbone_only and is_polymer and atom.name.upper() not in BACKBONE_ATOM_NAMES:
                    continue
                coordinates.append((atom.pos.x, atom.pos.y, atom.pos.z))
                numbers.append(atom.element.atomic_number)

    if not coordinates:
        raise ValueError(f"No modelled heavy atoms left to subtract: {path}")
    return (
        np.asarray(coordinates, dtype=np.float32),
        np.asarray(numbers, dtype=np.float32),
        tuple(sorted(set(excluded))),
    )


def render_model_density(
    axes: Sequence[np.ndarray],
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    sigma_A: float,
    cutoff_sigma: float = 4.0,
) -> np.ndarray:
    """Render Z-weighted Gaussian density for model atoms onto the crop grid."""
    if sigma_A <= 0 or cutoff_sigma <= 0:
        raise ValueError("sigma and cutoff must be positive")
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(atomic_numbers, dtype=np.float32).reshape(-1)
    if weights.size != coordinates.shape[0]:
        raise ValueError("atomic number count must match coordinate count")

    shape = tuple(len(axis) for axis in axes)
    density = np.zeros(shape, dtype=np.float32)
    starts = np.asarray([axis[0] for axis in axes], dtype=np.float32)
    steps = np.asarray(
        [float(axis[1] - axis[0]) if len(axis) > 1 else 1.0 for axis in axes],
        dtype=np.float32,
    )
    cutoff = float(cutoff_sigma) * float(sigma_A)
    inv_two_sigma2 = 1.0 / (2.0 * float(sigma_A) ** 2)

    for point, weight in zip(coordinates, weights):
        lo = np.maximum(np.ceil((point - cutoff - starts) / steps), 0).astype(int)
        hi = np.minimum(
            np.floor((point + cutoff - starts) / steps).astype(int) + 1,
            np.asarray(shape, dtype=int),
        )
        if np.any(lo >= hi):
            continue
        local = [axes[k][lo[k]:hi[k]] - point[k] for k in range(3)]
        squared = (
            (local[0] ** 2)[:, None, None]
            + (local[1] ** 2)[None, :, None]
            + (local[2] ** 2)[None, None, :]
        )
        density[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (
            weight * np.exp(-squared * inv_two_sigma2)
        ).astype(np.float32)
    return density


def _fit_scale(experimental: np.ndarray, model: np.ndarray):
    """Least-squares ``experimental ~= scale * model + offset`` plus its correlation."""
    x = model.reshape(-1).astype(np.float64)
    y = experimental.reshape(-1).astype(np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float((xc * xc).sum())
    if denominator <= 1e-12:
        return 0.0, float(y.mean()), 0.0
    scale = float((xc * yc).sum() / denominator)
    offset = float(y.mean() - scale * x.mean())
    spread = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    correlation = float((xc * yc).sum() / spread) if spread > 1e-12 else 0.0
    return scale, offset, correlation


def subtract_model_density(
    sampled: np.ndarray,
    axes: Sequence[np.ndarray],
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    centre: np.ndarray,
    excluded_residues: Sequence[str] = (),
    sigma_candidates: Sequence[float] = (0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5),
    fit_exclusion_radius_A: float = 8.0,
    cutoff_sigma: float = 4.0,
) -> ModelSubtraction:
    """Fit and remove simulated model density from an experimental crop.

    The Gaussian width is not assumed: each candidate sigma is rendered and the
    one correlating best with the experimental map is kept, which also yields a
    per-map resolution proxy.  Scale and offset are fitted only *outside* a
    sphere around the crop centre, so the ligand's own density cannot bias the
    fit and then be subtracted away.
    """
    sampled = np.asarray(sampled, dtype=np.float32)
    if fit_exclusion_radius_A < 0:
        raise ValueError("fit exclusion radius must be non-negative")
    if not len(sigma_candidates):
        raise ValueError("at least one candidate sigma is required")

    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    centre = np.asarray(centre, dtype=np.float32).reshape(1, 1, 1, 3)
    outside = (
        np.linalg.norm(mesh - centre, axis=-1) > float(fit_exclusion_radius_A)
    ) & np.isfinite(sampled)
    if int(outside.sum()) < 64:
        raise ValueError(
            f"Only {int(outside.sum())} voxels available to fit the subtraction; "
            "reduce fit_exclusion_radius_A or enlarge the box"
        )

    finite = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    best = None
    for sigma in sigma_candidates:
        model = render_model_density(
            axes, coordinates, atomic_numbers, float(sigma), cutoff_sigma=cutoff_sigma
        )
        scale, offset, correlation = _fit_scale(finite[outside], model[outside])
        if best is None or correlation > best[0]:
            best = (correlation, float(sigma), scale, offset, model)

    correlation, sigma, scale, offset, model = best
    residual = (finite - (scale * model + offset)).astype(np.float32)
    return ModelSubtraction(
        residual=residual,
        sigma_A=float(sigma),
        scale=float(scale),
        offset=float(offset),
        correlation=float(correlation),
        atom_count=int(np.asarray(coordinates).reshape(-1, 3).shape[0]),
        excluded_residues=tuple(excluded_residues),
        fit_voxels=int(outside.sum()),
    )
