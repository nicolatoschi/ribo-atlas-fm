"""Acquire bounded, public original observations/coordinate sources; no fitting."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, re, urllib.request
BASE=Path(__file__).resolve().parent
COMMIT='0a267edb82a30119a0af3b3dfc8fcbf6dbf07357'
IDS=['ADDRSW_SHP_0001','CIDGMP_SHP_0001','GLYCFN_SHP_0001','TRP4P6_SHP_0001','5SRRNA_SHP_0001','TRNAPH_SHP_0001']
PDBS=['1Y26','1Y27','2G9C','3GO2','3MXH','3IWN','3MUV','3MUT','3P49','1GID','1EHZ']
def fetch(item):
    name,url=item;p=BASE/name;p.parent.mkdir(parents=True,exist_ok=True)
    try:
        cached=p.exists()
        if cached:data=p.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RNA-validation-source-audit/1.0'}),timeout=45) as r:data=r.read(20*1024*1024+1)
            assert len(data)<=20*1024*1024
            p.write_bytes(data)
        return dict(path=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),cached=cached)
    except Exception as e:return dict(path=name,url=url,error=str(e))
def batch(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:return list(p.map(fetch,items))
rows=batch([(f'sources/{i}.md',f'https://raw.githubusercontent.com/DasLab/rmdb.github.io/{COMMIT}/_entries/{i}.md') for i in IDS])
targets=[]
for i in IDS:
    text=(BASE/'sources'/f'{i}.md').read_text()
    targets.append((f'sources/{i}.rdat',re.search(r'^rdat:\s+(\S+)',text,re.M)[1]))
targets.extend([(f'sources/{p}.pdb',f'https://files.rcsb.org/download/{p}.pdb') for p in PDBS])
targets.extend([(f'sources/{p}_entry.json',f'https://data.rcsb.org/rest/v1/core/entry/{p}') for p in PDBS])
targets.append(('sources/PMC3172344.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3172344/fullTextXML'))
rows+=batch(targets)
manifest=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_commit=COMMIT,scope='Public source acquisition only, no model fitting or efficacy result',files=rows)
(BASE/'source_download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum(r.get('bytes',0) for r in rows),'errors':[r for r in rows if 'error'in r]},indent=2))
