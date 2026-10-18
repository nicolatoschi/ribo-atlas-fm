#!/usr/bin/env python3
"""Minimal, dependency-free PDB utilities for the feasibility pilot (numpy only).

* parse_pdb: ATOM/HETATM records, first model only, altloc blank or 'A' kept
  (first occurrence wins), hydrogens and waters dropped.
* rna_chain: ordered residues of one chain that carry a nucleotide backbone
  (P/C4'/C1'), with one-letter parent identity (MODRES records or a small table).
* local_align: Smith-Waterman (match +2, mismatch -1, gap -3) used only to place the
  crystal chain inside a probed construct; identity-matched columns are the mapping.
* shrake_rupley: per-atom SASA (probe 1.4 A, 960 points, Bondi-type radii).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PARENT = {"A": "A", "C": "C", "G": "G", "U": "U", "GTP": "G", "GDP": "G", "GMP": "G", "CCC": "C",
          "PSU": "U", "5BU": "U", "IU": "U", "OMG": "G", "OMC": "C", "OMU": "U", "A2M": "A", "1MA": "A",
          "H2U": "U", "5MU": "U", "4SU": "U", "M2G": "G", "2MG": "G", "7MG": "G", "5MC": "C", "YYG": "G",
          "ADP": "A", "ATP": "A", "UTP": "U", "CTP": "C", "UR3": "U", "AP7": "A", "6MZ": "A", "LCA": "A",
          "CBV": "C", "GN7": "G"}
WATER = {"HOH", "WAT", "DOD"}
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80}
DEFAULT_RADIUS = 1.80
BASE_ATOMS = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}
RING_ATOMS = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
}
BASE_POLAR = {  # base N/O atoms used for base-base hydrogen-bond detection
    "A": ["N1", "N3", "N6", "N7"], "G": ["N1", "N2", "N3", "O6", "N7"],
    "C": ["O2", "N3", "N4"], "U": ["O2", "N3", "O4"],
}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def element_of(name: str, elem_field: str) -> str:
    e = elem_field.strip().upper()
    if e:
        return e
    n = name.strip().upper()
    return n[0] if n else "X"


def parse_pdb(path):
    """Return list of atom dicts (first model; altloc ' ' or 'A'; no H, no water)."""
    atoms, seen = [], set()
    modres = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec == "MODRES":
            modres[(line[16].strip(), line[12:15].strip(), int(line[18:22]), line[22].strip())] = line[24:27].strip()
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        name = line[12:16].strip()
        resn = line[17:20].strip()
        chain = line[21].strip()
        resi = int(line[22:26])
        icode = line[26].strip()
        elem = element_of(name, line[76:78] if len(line) >= 78 else "")
        if elem in ("H", "D") or resn in WATER:
            continue
        key = (chain, resi, icode, name)
        if key in seen:
            continue
        seen.add(key)
        atoms.append({"record": rec.strip(), "name": name, "resn": resn, "chain": chain, "resi": resi,
                      "icode": icode, "element": elem,
                      "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                      "bfactor": float(line[60:66]) if line[60:66].strip() else float("nan")})
    return atoms, modres


def residues(atoms):
    """Group atoms by residue in file order."""
    order, groups = [], {}
    for a in atoms:
        k = (a["chain"], a["resi"], a["icode"])
        if k not in groups:
            groups[k] = {"chain": a["chain"], "resi": a["resi"], "icode": a["icode"], "resn": a["resn"], "atoms": {}}
            order.append(k)
        groups[k]["atoms"][a["name"]] = a
    return [groups[k] for k in order]


def rna_chain(atoms, modres, chain):
    out = []
    for r in residues([a for a in atoms if a["chain"] == chain]):
        names = r["atoms"]
        if not ("C4'" in names and "C1'" in names):
            continue
        parent = modres.get((chain, r["resn"], r["resi"], r["icode"])) or PARENT.get(r["resn"])
        if parent not in ("A", "C", "G", "U"):
            continue
        r = dict(r)
        r["base"] = parent
        out.append(r)
    return out


def local_align(a: str, b: str, match=2, mismatch=-1, gap=-3):
    """Smith-Waterman alignment of a (crystal) inside b (construct). Returns list of (i, j) aligned
    index pairs (0-based) and the alignment score."""
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1))
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (0.0, H[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch), H[i - 1, j] + gap, H[i, j - 1] + gap)
            k = int(np.argmax(opts))
            H[i, j], T[i, j] = opts[k], k
    i, j = (int(v) for v in np.unravel_index(int(np.argmax(H)), H.shape))
    score = float(H[i, j])
    pairs = []
    while i > 0 and j > 0 and T[i, j] != 0:
        t = T[i, j]
        if t == 1:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif t == 2:
            i -= 1
        else:
            j -= 1
    return pairs[::-1], score


def sphere_points(n=960):
    """Golden-spiral unit sphere points (deterministic)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def shrake_rupley(xyz, radii, probe=1.4, n_points=960, chunk=256):
    """Per-atom solvent-accessible surface area (A^2)."""
    xyz = np.asarray(xyz, float)
    R = np.asarray(radii, float) + probe
    pts = sphere_points(n_points)
    sasa = np.zeros(len(xyz))
    # neighbour lists via a coarse distance matrix in chunks
    maxr = R.max()
    for start in range(0, len(xyz), chunk):
        idx = np.arange(start, min(start + chunk, len(xyz)))
        d = np.sqrt(((xyz[idx, None, :] - xyz[None, :, :]) ** 2).sum(-1))
        for row, i in enumerate(idx):
            nb = np.flatnonzero((d[row] < R[i] + R + 1e-9) & (np.arange(len(xyz)) != i))
            surf = xyz[i] + R[i] * pts
            if len(nb):
                dd = ((surf[:, None, :] - xyz[nb][None, :, :]) ** 2).sum(-1)
                buried = (dd < (R[nb] ** 2)[None, :]).any(axis=1)
                frac = 1.0 - buried.mean()
            else:
                frac = 1.0
            sasa[i] = frac * 4 * np.pi * R[i] ** 2
    return sasa


def radius_of(element):
    return RADII.get(element, DEFAULT_RADIUS)
