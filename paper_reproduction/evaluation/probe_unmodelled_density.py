#!/usr/bin/env python3
"""Synthetic probe: does unmodelled density (lipid, detergent, disordered loops)
destroy the benefit of subtracting the modelled protein?

Subtraction can only remove what is modelled.  In a membrane-protein map the
residual still contains lipid, detergent belt and disordered protein that no
model accounts for.  This probe adds such a blob at a controlled distance from
the ligand and measures what it costs the rotation gradient, then tests whether
connected-component segmentation of the residual recovers it.

As in probe_synthetic_gradient_conditions.py, only the gradient *direction*
matters, because compose_score discards the magnitude.
"""
import numpy as np
from scipy import ndimage

rng = np.random.default_rng(11)
N = 25
ax = np.linspace(-9., 9., N)
xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
pts = np.stack([xx, yy, zz], -1).reshape(-1, 3)
MAP_S = 1.5

def R_of(r):
    a = np.linalg.norm(r)
    if a < 1e-12: return np.eye(3)
    v = r/a; K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K

def render(c, w, s):
    d2 = ((pts[:,None,:]-c[None])**2).sum(-1)
    return (np.exp(-d2/(2*s**2))*w[None]).sum(1)

def coord_grad(c, w, s, target):
    d2 = ((pts[:,None,:]-c[None])**2).sum(-1)
    E = np.exp(-d2/(2*s**2))*w[None]
    m = E.sum(1); t = target
    dm = E[:,:,None]*((pts[:,None,:]-c[None])/s**2)
    mc, tc = m-m.mean(), t-t.mean()
    A = (mc*tc).mean(); vt = max((tc**2).mean(),1e-8); vm = max((mc**2).mean(),1e-8)
    B = np.sqrt(vm*vt)
    dA = (dm*tc[:,None,None]).mean(0)
    dB = 0.5/B*(2*(dm*mc[:,None,None]).mean(0))*vt
    return (dA*B - A*dB)/B**2

def acyl_chain(anchor, direction, n=12, spacing=1.3):
    """Elongated unmodelled blob, roughly a lipid tail."""
    d = direction/np.linalg.norm(direction)
    return anchor[None] + np.arange(n)[:,None]*spacing*d[None]

def keep_component_at_centre(residual, z_thresh=1.0):
    """Segment the residual and keep only the component covering the pocket centre."""
    vol = residual.reshape(N, N, N)
    z = (vol - vol.mean())/max(vol.std(), 1e-8)
    lab, n = ndimage.label(z >= z_thresh, structure=np.ones((3,3,3)))
    if n == 0: return residual
    c = N//2
    target_lab = lab[c, c, c]
    if target_lab == 0:                       # centre below threshold: take nearest component
        idx = np.argwhere(lab > 0)
        if len(idx) == 0: return residual
        target_lab = lab[tuple(idx[np.argmin(((idx-c)**2).sum(1))])]
    out = np.where(lab == target_lab, vol, vol.min())
    return out.reshape(-1)

N_LIG, N_PROT, N_TRIAL = 18, 150, 30
conds = ["difference map, nothing unmodelled",
         "  + unmodelled lipid, 8 A from ligand",
         "  + unmodelled lipid, 5 A from ligand",
         "  + unmodelled lipid, touching ligand (2.5 A)",
         "  + unmodelled lipid at 5 A, THEN segmentation",
         "  + 4 unmodelled lipid chains, adjacent",
         "  + unmodelled lipid SHELL surrounding ligand (worst case)",
         "(reference) soft-masked map, nothing subtracted"]
res = {c: [] for c in conds}

for _ in range(N_TRIAL):
    lig = rng.normal(scale=2.2, size=(N_LIG,3)); lig -= lig.mean(0)
    wl = rng.choice([6.,7.,8.,16.], size=N_LIG); wl = wl/max(wl.mean(),1.0)
    d = rng.normal(size=(N_PROT,3)); d /= np.linalg.norm(d,axis=1,keepdims=True)
    prot = d*rng.uniform(3.5, 9.0, size=(N_PROT,1))
    wp = rng.choice([6.,7.,8.], size=N_PROT); wp = wp/max(wp.mean(),1.0)
    lig_map, prot_map = render(lig,wl,MAP_S), render(prot,wp,MAP_S)
    noise = rng.normal(scale=0.05*lig_map.std(), size=lig_map.shape)
    lig_radius = np.linalg.norm(lig, axis=1).max()

    def lipid_at(gap):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        anchor = u*(lig_radius + gap)
        tail = acyl_chain(anchor, u)
        wt = np.full(len(tail), 6.0); wt = wt/max(wt.mean(),1.0)
        return render(tail, wt, MAP_S)

    base_resid = lig_map + prot_map + noise - prot_map
    heavy = sum(lipid_at(rng.uniform(2.5, 6.0)) for _ in range(4))
    ds = rng.normal(size=(120,3)); ds /= np.linalg.norm(ds,axis=1,keepdims=True)
    shell = ds*rng.uniform(3.5, 9.0, size=(120,1))
    ws = np.full(120, 6.0); ws = ws/max(ws.mean(),1.0)
    shell_map = render(shell, ws, MAP_S)
    l8, l5, l25 = lipid_at(8.0), lipid_at(5.0), lipid_at(2.5)
    full = lig_map + prot_map + noise + l5
    dist = np.sqrt(((pts[:,None,:]-prot[None])**2).sum(-1).min(1))
    frac = np.clip((dist-1.5)/1.0, 0.0, 1.0); soft = frac*frac*(3-2*frac)
    bg = float(np.median(full))

    targets = {
        conds[0]: base_resid,
        conds[1]: base_resid + l8,
        conds[2]: base_resid + l5,
        conds[3]: base_resid + l25,
        conds[4]: keep_component_at_centre(base_resid + l5),
        conds[5]: base_resid + heavy,
        conds[6]: base_resid + shell_map,
        conds[7]: bg + np.maximum(full-bg, 0.0)*soft,
    }
    rot_r = rng.normal(size=3); rot_r = rot_r/np.linalg.norm(rot_r)*np.deg2rad(25)
    pose = lig @ R_of(rot_r).T
    back = -rot_r/np.linalg.norm(rot_r)
    for c in conds:
        g = coord_grad(pose, wl, 1.5, targets[c])
        v = np.cross(pose-pose.mean(0), g).sum(0); n = np.linalg.norm(v)
        if n > 1e-12: res[c].append(float(v@back/n))

print(f"{N_TRIAL} trials | rotation error 25 deg, envelope 1.5 A")
print("mean cosine to the true rotation correction (fraction pointing the right way)\n")
for c in conds:
    a = np.array(res[c])
    print(f"{c:48s} {a.mean():+7.3f} ({(a>0).mean():.2f})")
