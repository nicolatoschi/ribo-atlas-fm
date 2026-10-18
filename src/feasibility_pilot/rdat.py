#!/usr/bin/env python3
"""Small RDAT reader (v0.24 space-delimited and v0.3x tab-delimited files).

Index convention: construct index (1-based) = SEQPOS number - OFFSET. SEQPOS tokens may
carry a leading nucleotide letter (e.g. 'g-18'); the letter is checked against SEQUENCE.
Values 'NaN'/'nan' are missing. Reactivities are returned exactly as deposited.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


def _tokens(line):
    return line.replace("\t", " ").split()


def read_rdat(path, load_values=True):
    out = {"path": str(path), "annotation": [], "comments": [], "rows": {}}
    for raw in Path(path).read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        tok = _tokens(raw)
        tag, rest = tok[0], tok[1:]
        if tag == "NAME":
            out["name"] = " ".join(rest)
        elif tag == "SEQUENCE":
            out["sequence"] = "".join(rest)
        elif tag == "STRUCTURE":
            out["structure"] = "".join(rest)
        elif tag == "OFFSET":
            out["offset"] = int(rest[0])
        elif tag == "SEQPOS":
            nums, letters = [], []
            for t in rest:
                m = re.fullmatch(r"([A-Za-z]?)(-?\d+)", t)
                if not m:
                    raise ValueError(f"Bad SEQPOS token {t!r} in {path}")
                letters.append(m.group(1) or None)
                nums.append(int(m.group(2)))
            out["seqpos"], out["seqpos_letters"] = nums, letters
        elif tag == "ANNOTATION":
            out["annotation"] = rest
        elif tag == "COMMENT":
            out["comments"].append(" ".join(rest))
        elif tag.startswith("ANNOTATION_DATA:"):
            k = int(tag.split(":")[1])
            out["rows"].setdefault(k, {})["annotation"] = rest
        elif tag.startswith("REACTIVITY:") and load_values:
            k = int(tag.split(":")[1])
            out["rows"].setdefault(k, {})["reactivity"] = [_num(x) for x in rest]
        elif tag.startswith("REACTIVITY_ERROR:") and load_values:
            k = int(tag.split(":")[1])
            out["rows"].setdefault(k, {})["error"] = [_num(x) for x in rest]
    seq = out["sequence"]
    out["construct_index"] = [p - out["offset"] for p in out["seqpos"]]
    for ci, letter in zip(out["construct_index"], out["seqpos_letters"]):
        if not 1 <= ci <= len(seq):
            raise ValueError(f"SEQPOS outside sequence in {path}")
        if letter is not None and letter.upper() != seq[ci - 1].upper():
            raise ValueError(f"SEQPOS letter mismatch at {ci} in {path}")
    if load_values:
        for k, r in out["rows"].items():
            for f in ("reactivity", "error"):
                if f in r and len(r[f]) != len(out["seqpos"]):
                    raise ValueError(f"{path} row {k} {f} length {len(r[f])} != SEQPOS {len(out['seqpos'])}")
    return out


def _num(x):
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None
