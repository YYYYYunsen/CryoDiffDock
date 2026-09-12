#!/usr/bin/env python3
"""Synthetic probe: how map *composition* affects the density gradient direction.

Complements ``evaluate_density_gradient_informativeness.py``, which measures
gradient quality on real crops.  This script instead isolates one variable that
real data cannot separate: what the target map contains.  Ligand and protein
density are synthesised at a chosen resolution, so the same ligand can be scored
against a clean ligand-only map, a realistic ligand+protein map, that map with
the protein masked out, and a difference map with an exact or a deliberately
wrong protein model.

Only the *direction* of the gradient is measured, because ``compose_score``
norm-matches the magnitude to the learned score and discards it.  The reported
number is the mean cosine between the gradient and the true correction, with the
fraction of trials pointing the right way in parentheses.

Runs in seconds on CPU and needs no data.  Numbers are qualitative: Gaussian
protein shells and uniform occupancy are idealisations.
"""
import numpy as np
rng = np.random.default_rng(7)
ax = np.linspace(-9., 9., 25)
xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
pts = np.stack([xx,yy,zz],-1).reshape(-1,3)
MAP_S = 1.5

def R_of(r):
    a = np.linalg.norm(r)
    if a < 1e-12: return np.eye(3)
    v = r/a; K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K

def render(c, w, s):
    d2 = ((pts[:,None,:]-c[None])**2).sum(-1)
    return (np.exp(-d2/(2*s**2))*w[None]).sum(1)

def ncc(m,t):
    mc,tc = m-m.mean(), t-t.mean()
    return (mc*tc).mean()/np.sqrt(max((mc**2).mean(),1e-8)*max((tc**2).mean(),1e-8))

def coord_grad(c, w, s, target, mask):
    d2 = ((pts[:,None,:]-c[None])**2).sum(-1)
    E = np.exp(-d2/(2*s**2))*w[None]
    base = E.sum(1)
    m, t = base[mask], target[mask]
    diff = (pts[:,None,:]-c[None])/s**2
    dm = (E[:,:,None]*diff)[mask]
    mc, tc = m-m.mean(), t-t.mean()
    A = (mc*tc).mean(); vt = max((tc**2).mean(),1e-8); vm = max((mc**2).mean(),1e-8)
    B = np.sqrt(vm*vt)
    dA = (dm*tc[:,None,None]).mean(0)
    dB = 0.5/B*(2*(dm*mc[:,None,None]).mean(0))*vt
    return (dA*B - A*dB)/B**2

N_LIG, N_PROT, N_TRIAL = 18, 150, 30
sigmas = [1.5, 3.0]
conds = ["clean ligand only",
         "ligand+protein, no mask",
         "ligand+protein, soft mask 1.5-2.5A (current default)",
         "ligand+protein, hard mask r=3.5A",
         "difference map, exact protein",
         "difference map, 30% sidechains wrong"]
res = {(c,s,k): [] for c in conds for s in sigmas for k in ("tr","rot")}

for _ in range(N_TRIAL):
    lig = rng.normal(scale=2.2, size=(N_LIG,3)); lig -= lig.mean(0)
    wl = rng.choice([6.,7.,8.,16.], size=N_LIG); wl = wl/max(wl.mean(),1.0)
    # protein shell: atoms 3.5-9 A from ligand centre, i.e. in contact
    d = rng.normal(size=(N_PROT,3)); d /= np.linalg.norm(d,axis=1,keepdims=True)
    prot = d*rng.uniform(3.5, 9.0, size=(N_PROT,1))
    wp = rng.choice([6.,7.,8.], size=N_PROT); wp = wp/max(wp.mean(),1.0)
    lig_map, prot_map = render(lig,wl,MAP_S), render(prot,wp,MAP_S)
    noise = rng.normal(scale=0.05*lig_map.std(), size=lig_map.shape)
    full = lig_map + prot_map + noise
    # imperfect protein model: 30% of atoms displaced ~1.5 A (wrong rotamer)
    bad = prot.copy(); sel = rng.random(N_PROT) < 0.30
    bad[sel] += rng.normal(scale=1.5, size=(sel.sum(),3))
    # Protein masking as the pipeline actually applies it: the *target* map is
    # down-weighted voxel-wise (apply_protein_mask), all voxels are retained,
    # and the rendered ligand model is never masked.
    dist = np.sqrt(((pts[:,None,:]-prot[None])**2).sum(-1).min(1))
    bg = float(np.median(full))
    def weighted(w):
        return bg + np.maximum(full - bg, 0.0) * w
    frac = np.clip((dist - 1.5)/(2.5 - 1.5), 0.0, 1.0)
    soft = frac*frac*(3.0 - 2.0*frac)            # smoothstep, matches soft_protein_mask
    hard = (dist > 3.5).astype(float)
    targets = {
        conds[0]: lig_map + noise,
        conds[1]: full,
        conds[2]: weighted(soft),
        conds[3]: weighted(hard),
        conds[4]: full - prot_map,
        conds[5]: full - render(bad,wp,MAP_S),
    }
    keep = np.ones(len(pts), bool)
    masks = {c: keep for c in conds}

    tr_shift = rng.normal(scale=1.0, size=3)
    rot_r = rng.normal(size=3); rot_r = rot_r/np.linalg.norm(rot_r)*np.deg2rad(25)
    pose_tr = lig + tr_shift
    pose_rot = lig @ R_of(rot_r).T
    for c in conds:
        tg, mk = targets[c], masks[c]
        for s in sigmas:
            g = coord_grad(pose_tr, wl, s, tg, mk)
            v = g.sum(0); n = np.linalg.norm(v)
            if n>1e-12: res[(c,s,"tr")].append(float(v@(-tr_shift/np.linalg.norm(tr_shift))/n))
            g = coord_grad(pose_rot, wl, s, tg, mk)
            v = np.cross(pose_rot-pose_rot.mean(0), g).sum(0); n = np.linalg.norm(v)
            if n>1e-12: res[(c,s,"rot")].append(float(v@(-rot_r/np.linalg.norm(rot_r))/n))

print(f"{N_TRIAL} trials | tr error ~1.0 A, rot error 25 deg | mean cos to true correction\n")
print(f"{'target condition':54s} {'sig':>4s} {'tr':>18s} {'rot':>18s}")
for c in conds:
    for s in sigmas:
        a, b = np.array(res[(c,s,'tr')]), np.array(res[(c,s,'rot')])
        print(f"{c:54s} {s:4.1f} {a.mean():+7.3f} ({(a>0).mean():.2f}) {b.mean():+7.3f} ({(b>0).mean():.2f})")
