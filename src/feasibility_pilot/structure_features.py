#!/usr/bin/env python3
"""Structure-only nucleotide annotations used by both analyses (no probing data are read here).

Definitions (frozen in plans/*.json; thresholds are module constants):

Canonical pair (cis Watson-Crick or G.U wobble) between chain residues i<j, j-i>=4:
  G-C : d(G N1, C N3) <= 3.8 and (d(G O6, C N4) <= 3.8 or d(G N2, C O2) <= 3.8)
  A-U : d(A N1, U N3) <= 3.8 and d(A N6, U O4) <= 3.8
  G-U : d(G N1, U O2) <= 3.8 and d(G O6, U N3) <= 3.8
  and C1'-C1' distance in [9.0, 12.0] A and base-plane angle <= 60 deg.
  (3.8 A / 60 deg chosen on structure only, before any probing data were read: recall 0.89 and
  precision 0.93 against the deposited RDAT secondary structures mapped onto 8 crystals.)
  Each residue keeps at most one canonical partner (smallest summed H-bond distance wins).
Helix: canonical pairs (i,j),(k,l) are in one helix when 0 < k-i <= 2 and 0 < j-l <= 2 (stacked
  pairs, tolerating a single-nucleotide bulge/mismatch on either strand); transitive closure.
Nested secondary structure: maximum-weight (pair count) subset of helices with no crossing pair
  (exhaustive search within each crossing-conflict component; ties -> earlier helices).
  Canonical pairs of non-selected (pseudoknot/kissing) helices are NOT secondary structure.
Loops (from the nested pairs): each unpaired residue belongs to the loop closed by its innermost
  enclosing nested pair (or the external loop). A loop is hairpin (no enclosed pair), internal/
  bulge (one directly enclosed pair) or multiloop (>=2). A loop unit = its unpaired residues +
  its closing pair + (internal loops) the enclosed pair.
Local relation local(i,j): |i-j| <= 3 in chain order; or (i,j) a nested canonical pair; or both in
  the same nested helix; or both in the same hairpin or internal/bulge loop unit.
Contacts: minimum heavy-atom distance <= 4.0 A (RNA chain residues only; protein, ions, water and
  symmetry mates are ignored).
Classes: ligand-contact L = any heavy atom within 4.0 A of a designated ligand heavy atom;
  tertiary-contact T = not L and has a contact to some j with not local(i,j) (this includes
  pseudoknot/kissing canonical pairs, junction/multiloop and loop-helix packing contacts);
  rest R = everything else.

Stacking (i,j): base-ring centroid distance <= 5.5 A, ring-plane angle <= 30 deg (or >= 150),
  vertical separation along the mean normal in [2.8, 4.5] A, horizontal offset <= 4.0 A
  (calibrated on structure only so that >=90% of same-strand neighbours inside nested helices are
  called stacked; 5-95% range there: centroid 3.5-5.1 A, vertical 3.0-3.7 A, offset 0.9-3.8 A).
Base-base hydrogen bond: any base N/O atom of i within 3.4 A of any base N/O atom of j (|i-j|>=2).
SASA: Shrake-Rupley, probe 1.4 A, 960 points, radii C 1.70 N 1.55 O 1.52 P 1.80 (S 1.80),
  computed on the whole deposited RNA chain plus designated ligand heavy atoms only.
"""
from __future__ import annotations

import itertools

import numpy as np

from pdbtools import BASE_ATOMS, BASE_POLAR, RING_ATOMS, parse_pdb, radius_of, rna_chain, shrake_rupley

HB = 3.8
C1C1 = (9.0, 12.0)
PAIR_ANGLE = 60.0
CONTACT = 4.0
LIGAND_CONTACT = 4.0
STACK_D, STACK_ANGLE, STACK_V, STACK_H = 5.5, 30.0, (2.8, 4.5), 4.0
BASE_HB = 3.4

CANONICAL_RULES = {
    ("G", "C"): ([("N1", "N3")], [("O6", "N4"), ("N2", "O2")]),
    ("A", "U"): ([("N1", "N3"), ("N6", "O4")], []),
    ("G", "U"): ([("N1", "O2"), ("O6", "N3")], []),
}


def load_structure(pdb_path, chain, ligand_ids):
    atoms, modres = parse_pdb(pdb_path)
    res = rna_chain(atoms, modres, chain)
    lig = [a for a in atoms if (a["resn"], a["chain"], a["resi"]) in {tuple(x) for x in ligand_ids}]
    found = {(a["resn"], a["chain"], a["resi"]) for a in lig}
    missing = [tuple(x) for x in ligand_ids if tuple(x) not in found]
    if missing:
        raise ValueError(f"Ligand residues not found in {pdb_path}: {missing}")
    return res, lig


def _xyz(r, name):
    a = r["atoms"].get(name)
    return None if a is None else a["xyz"]


def base_frame(r):
    pts = [r["atoms"][n]["xyz"] for n in RING_ATOMS[r["base"]] if n in r["atoms"]]
    if len(pts) < 5:
        return None, None
    pts = np.asarray(pts)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[2]


def _angle(n1, n2):
    cos = abs(float(np.dot(n1, n2)))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def canonical_pairs(res):
    frames = [base_frame(r) for r in res]
    cands = []
    for i, j in itertools.combinations(range(len(res)), 2):
        if j - i < 4:
            continue
        bi, bj = res[i]["base"], res[j]["base"]
        key, flip = ((bi, bj), False) if (bi, bj) in CANONICAL_RULES else (((bj, bi), True) if (bj, bi) in CANONICAL_RULES else (None, None))
        if key is None:
            continue
        a, b = (res[j], res[i]) if flip else (res[i], res[j])
        required, any_of = CANONICAL_RULES[key]
        total, ok = 0.0, True
        for x, y in required:
            p, q = _xyz(a, x), _xyz(b, y)
            if p is None or q is None:
                ok = False; break
            d = float(np.linalg.norm(p - q))
            if d > HB:
                ok = False; break
            total += d
        if not ok:
            continue
        if any_of:
            ds = [float(np.linalg.norm(_xyz(a, x) - _xyz(b, y))) for x, y in any_of if _xyz(a, x) is not None and _xyz(b, y) is not None]
            if not ds or min(ds) > HB:
                continue
            total += min(ds)
        c1 = float(np.linalg.norm(_xyz(res[i], "C1'") - _xyz(res[j], "C1'")))
        if not C1C1[0] <= c1 <= C1C1[1]:
            continue
        (ci, ni), (cj, nj) = frames[i], frames[j]
        if ni is None or nj is None or _angle(ni, nj) > PAIR_ANGLE:
            continue
        cands.append((total, i, j))
    used, pairs = set(), []
    for total, i, j in sorted(cands):
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j))
    return sorted(pairs)


def helices(pairs):
    pairs = sorted(pairs)
    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in itertools.combinations(range(len(pairs)), 2):
        (i, j), (k, l) = pairs[a], pairs[b]
        if (0 < k - i <= 2 and 0 < j - l <= 2) or (0 < i - k <= 2 and 0 < l - j <= 2):
            parent[find(a)] = find(b)
    groups = {}
    for idx, p in enumerate(pairs):
        groups.setdefault(find(idx), []).append(p)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def _cross(h1, h2):
    for i, j in h1:
        for k, l in h2:
            if i < k < j < l or k < i < l < j:
                return True
    return False


def nested_selection(hx):
    n = len(hx)
    conflict = {a: {b for b in range(n) if b != a and _cross(hx[a], hx[b])} for a in range(n)}
    selected = set(range(n))
    seen = set()
    for start in range(n):
        if start in seen or not conflict[start]:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x); stack.extend(conflict[x])
        seen |= comp
        comp = sorted(comp)
        if len(comp) > 18:
            raise ValueError("Crossing component too large for exhaustive search")
        best, best_key = None, None
        for mask in range(1 << len(comp)):
            sub = [comp[k] for k in range(len(comp)) if mask >> k & 1]
            if any(b in conflict[a] for a, b in itertools.combinations(sub, 2)):
                continue
            weight = sum(len(hx[a]) for a in sub)
            key = (weight, tuple(-a for a in sorted(sub)))
            if best_key is None or key > best_key:
                best, best_key = sub, key
        selected -= set(comp) - set(best)
    return sorted(selected)


def loop_units(n, nested_pairs):
    partner = {}
    for i, j in nested_pairs:
        partner[i], partner[j] = j, i
    closing = {}
    for x in range(n):
        if x in partner:
            continue
        enc = [(i, j) for i, j in nested_pairs if i < x < j]
        closing[x] = max(enc, key=lambda p: p[0]) if enc else None
    units = {}
    for x, c in closing.items():
        units.setdefault(c, set()).add(x)
    out = []
    for c, members in units.items():
        if c is None:
            out.append({"type": "external", "closing": None, "members": sorted(members)})
            continue
        i, j = c
        children = [(k, l) for k, l in nested_pairs if i < k < l < j and not any(i < a < k and l < b < j for a, b in nested_pairs)]
        kind = "hairpin" if not children else ("internal" if len(children) == 1 else "multiloop")
        unit = set(members) | {i, j}
        if kind == "internal":
            unit |= set(children[0])
        out.append({"type": kind, "closing": [i, j], "children": children, "members": sorted(members), "unit": sorted(unit)})
    # internal loops with zero unpaired residues do not occur (those pairs are stacked in a helix)
    return out


def residue_min_distances(res, extra_atoms=None):
    xyz, owner = [], []
    for k, r in enumerate(res):
        for a in r["atoms"].values():
            xyz.append(a["xyz"]); owner.append(k)
    xyz, owner = np.asarray(xyz, np.float64), np.asarray(owner)
    n = len(res)
    D = np.full((n, n), np.inf)
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    for k in range(n):
        m = owner == k
        d = np.sqrt(((xyz[m][:, None, :] - xyz[None, :, :]) ** 2).sum(-1)).min(0)
        D[k] = np.minimum.reduceat(d, starts)
    lig = None
    if extra_atoms:
        L = np.asarray([a["xyz"] for a in extra_atoms])
        dl = np.sqrt(((xyz[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min(1)
        lig = np.minimum.reduceat(dl, starts)
    return D, lig


def annotate(res, ligand_atoms, ignore=frozenset()):
    """ignore: chain indices of non-native (engineered) residues; they are never contact partners."""
    n = len(res)
    pairs = canonical_pairs(res)
    hx = helices(pairs)
    sel = nested_selection(hx)
    nested = sorted(p for h in sel for p in hx[h])
    pk = sorted(p for h in range(len(hx)) if h not in sel for p in hx[h])
    helix_of = {}
    for h in sel:
        for i, j in hx[h]:
            helix_of[i] = h; helix_of[j] = h
    loops = loop_units(n, nested)
    unit_of = {}
    for li, lp in enumerate(loops):
        if lp["type"] in ("hairpin", "internal"):
            for x in lp["unit"]:
                unit_of.setdefault(x, set()).add(li)
    nested_set = {tuple(p) for p in nested}
    D, dlig = residue_min_distances(res, ligand_atoms)

    def local(i, j):
        if abs(i - j) <= 3:
            return True
        if (min(i, j), max(i, j)) in nested_set:
            return True
        if i in helix_of and j in helix_of and helix_of[i] == helix_of[j]:
            return True
        if unit_of.get(i, set()) & unit_of.get(j, set()):
            return True
        return False

    out = []
    loop_type = {}
    for lp in loops:
        for x in lp["members"]:
            loop_type[x] = lp["type"]
    for i in range(n):
        tert = [j for j in range(n) if j != i and j not in ignore and D[i, j] <= CONTACT and not local(i, j)]
        is_l = dlig is not None and float(dlig[i]) <= LIGAND_CONTACT
        cls = "L" if is_l else ("T" if tert else "R")
        out.append({"chain_index": i, "resi": res[i]["resi"], "icode": res[i]["icode"], "base": res[i]["base"],
                    "class": cls, "ligand_min_distance": None if dlig is None else round(float(dlig[i]), 3),
                    "tertiary_partners_chain_index": tert,
                    "nested_helix": helix_of.get(i), "loop_type": loop_type.get(i, "paired"),
                    "pseudoknot_canonical": any(i in p for p in pk)})
    summary = {"n_residues": n, "n_canonical_pairs": len(pairs), "n_helices": len(hx),
               "n_nested_pairs": len(nested), "n_pseudoknot_pairs": len(pk),
               "nested_pairs_chain_index": nested, "pseudoknot_pairs_chain_index": pk,
               "class_counts": {c: sum(r["class"] == c for r in out) for c in "LTR"}}
    return out, summary


def stacking_and_hbonds(res):
    n = len(res)
    frames = [base_frame(r) for r in res]
    stacked = np.zeros((n, n), bool)
    for i, j in itertools.combinations(range(n), 2):
        (ci, ni), (cj, nj) = frames[i], frames[j]
        if ni is None or nj is None:
            continue
        v = cj - ci
        d = float(np.linalg.norm(v))
        if d > STACK_D:
            continue
        ang = _angle(ni, nj)
        if ang > STACK_ANGLE:
            continue
        nbar = ni + (nj if np.dot(ni, nj) >= 0 else -nj)
        nbar /= np.linalg.norm(nbar)
        vert = abs(float(np.dot(v, nbar)))
        horiz = float(np.sqrt(max(d * d - vert * vert, 0.0)))
        if STACK_V[0] <= vert <= STACK_V[1] and horiz <= STACK_H:
            stacked[i, j] = stacked[j, i] = True
    polar = []
    for r in res:
        polar.append(np.asarray([r["atoms"][a]["xyz"] for a in BASE_POLAR[r["base"]] if a in r["atoms"]]).reshape(-1, 3))
    hb = np.zeros((n, n), bool)
    for i, j in itertools.combinations(range(n), 2):
        if j - i < 2 or not len(polar[i]) or not len(polar[j]):
            continue
        d = np.sqrt(((polar[i][:, None] - polar[j][None]) ** 2).sum(-1)).min()
        if d <= BASE_HB:
            hb[i, j] = hb[j, i] = True
    return stacked, hb


def sasa_descriptors(res, ligand_atoms):
    xyz, rad, owner, names = [], [], [], []
    for k, r in enumerate(res):
        for nm, a in r["atoms"].items():
            xyz.append(a["xyz"]); rad.append(radius_of(a["element"])); owner.append(k); names.append(nm)
    for a in ligand_atoms:
        xyz.append(a["xyz"]); rad.append(radius_of(a["element"])); owner.append(-1); names.append(a["name"])
    s = shrake_rupley(np.asarray(xyz), np.asarray(rad))
    owner = np.asarray(owner)
    out = []
    for k, r in enumerate(res):
        m = np.flatnonzero(owner == k)
        per = {names[i]: float(s[i]) for i in m}
        dms_atom = {"A": "N1", "C": "N3"}.get(r["base"])
        out.append({"dms_atom": dms_atom, "dms_atom_sasa": per.get(dms_atom) if dms_atom else 0.0,
                    "o2p_sasa": per.get("O2'"), "base_sasa": sum(per.get(a, 0.0) for a in BASE_ATOMS[r["base"]]),
                    "missing_atoms": sorted(set(BASE_ATOMS[r["base"]] + ["O2'"]) - set(per))})
    return out
