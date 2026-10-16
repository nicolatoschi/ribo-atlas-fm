#!/bin/bash
# Fetch public RMDB RDAT release assets listed in download_list.tsv; verify size and published sha256.
# Never overwrites a verified file. Writes fetch_log.tsv (id, status, bytes, sha256).
set -u
DEST=${1:?destination dir}; LIST=${2:?download_list.tsv}
mkdir -p "$DEST"; LOG="$DEST/fetch_log.tsv"; : > "$LOG"
while IFS=$'\t' read -r id url sha bytes; do
  f="$DEST/$id.rdat"
  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then printf '%s\tPRESENT\t%s\t%s\n' "$id" "$bytes" "$sha" >> "$LOG"; continue; fi
  curl -fsSL --retry 3 --retry-delay 10 -o "$f.part" "$url" < /dev/null || { printf '%s\tDOWNLOAD_FAILED\t-\t-\n' "$id" >> "$LOG"; rm -f "$f.part"; continue; }
  got=$(stat -c %s "$f.part"); dig=$(sha256sum "$f.part" | cut -d' ' -f1)
  if [ "$got" = "$bytes" ] && [ "$dig" = "$sha" ]; then mv "$f.part" "$f"; chmod 444 "$f"; printf '%s\tOK\t%s\t%s\n' "$id" "$got" "$dig" >> "$LOG"
  else printf '%s\tHASH_MISMATCH\t%s\t%s\n' "$id" "$got" "$dig" >> "$LOG"; rm -f "$f.part"; fi
done < "$LIST"
touch "$DEST/FETCH_DONE"
