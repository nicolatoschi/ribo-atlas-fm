#!/usr/bin/env python3
"""Extract visible text (description/annotation/citation) from downloaded RMDB detail pages."""
import html, re, sys
for path in sys.argv[1:]:
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", " ", t)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", t))
    txt = re.sub(r"\s+", " ", txt)
    i = txt.find("Description")
    print("==", path.split("/")[-1].replace("_rmdb_detail.html", ""), "::", txt[max(0, i - 200): i + 900][:1100])
