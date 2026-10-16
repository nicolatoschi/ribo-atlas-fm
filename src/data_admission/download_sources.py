"""Download only the eight pinned RNA-only RDAT assets and verify release SHA-256."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import time
import urllib.request

BASE = pathlib.Path(__file__).resolve().parent
SELECTION = json.loads((BASE / 'selected_sources.json').read_text())

def download(entry):
    key = entry['rmdb_id']
    target = BASE / 'raw' / (key + '.rdat')
    expected = entry['download_availability_evidence']['digest'].split(':', 1)[1]
    size = entry['download_availability_evidence']['bytes']
    result = {'rmdb_id': key, 'url': entry['rdat_URL'], 'expected_bytes': size,
              'expected_sha256': expected, 'started_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    partial = target.with_suffix('.rdat.partial')
    try:
        if target.exists():
            digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        else:
            h = hashlib.sha256()
            req = urllib.request.Request(entry['rdat_URL'], headers={'User-Agent':'RIBO-data-audit/1.0'})
            with urllib.request.urlopen(req, timeout=60) as source, partial.open('wb') as out:
                while True:
                    data = source.read(1024 * 1024)
                    if not data:
                        break
                    out.write(data)
                    h.update(data)
            digest = h.hexdigest()
            if digest != expected or partial.stat().st_size != size:
                raise RuntimeError(f'Integrity mismatch: {digest}, {partial.stat().st_size}')
            partial.replace(target)
        result.update({'actual_bytes': target.stat().st_size, 'actual_sha256': digest,
                       'sha256_verified': digest == expected, 'size_verified': target.stat().st_size == size})
        if not result['sha256_verified'] or not result['size_verified']:
            raise RuntimeError('Previously downloaded file mismatches pinned release metadata')
        result['status'] = 'verified'
    except Exception as error:
        result.update({'status': 'failed', 'error': str(error)})
    result['completed_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(result), flush=True)
    (BASE / (key + '_download.json')).write_text(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    (BASE / 'raw').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(download, SELECTION['entries']))
    manifest = {'selection': 'OK2TRN, OK1LIB, M2PK90, RFMREP DMS and 2A3',
                'excluded_source_group': 'VIRWIN',
                'pinned_repository_commit': SELECTION['pinned_repository_commit'],
                'assets': results}
    (BASE / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
