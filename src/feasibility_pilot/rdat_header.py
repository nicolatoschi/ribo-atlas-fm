#!/usr/bin/env python3
"""Print RDAT header/annotation lines only (never REACTIVITY values) so that
construct/condition matching can be checked before any analysis plan is frozen."""
import sys
for path in sys.argv[1:]:
    print("=====", path)
    for line in open(path, encoding="utf-8", errors="replace"):
        tag = line.split(" ", 1)[0]
        if tag.startswith(("REACTIVITY", "SEQPOS", "XSEL", "TRACE", "READS", "SIGNAL_TO_NOISE")):
            n = len(line.split()) - 1
            print(f"{tag} <{n} values suppressed>")
        elif tag.startswith("ANNOTATION_DATA") or tag in ("NAME", "SEQUENCE", "STRUCTURE", "OFFSET", "ANNOTATION", "COMMENT", "RDAT_VERSION") or tag.startswith("SEQUENCE:") or tag.startswith("STRUCTURE:") or tag.startswith("MUTPOS"):
            print(line.rstrip()[:400])
