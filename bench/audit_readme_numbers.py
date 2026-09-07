"""Re-derive every checkable figure in _to_send/README.txt from committed data.

Why this exists: on 5 Sep two numbers went into a write-up that were wrong --
"butter's worst case 4.12x -> 1.69x" conflated two clips, and "jitter moves
<0.002" was false for butter. Both were caught by re-deriving afterwards. This
does it as a gate instead, and it earned its keep on 7 Sep by catching a claim
that "not one frame anywhere falls below 0.80 IoU" when bilibili has sixteen.

Run it before the package goes anywhere:

    python bench/audit_readme_numbers.py     # expects FAILURES: 0
"""
import json, os, re, statistics as st
R='bench/results'
T='_to_send'
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

ok=[]; bad=[]
def check(label, claim, actual, tol=0):
    good = (abs(claim-actual)<=tol) if isinstance(claim,(int,float)) else (claim==actual)
    (ok if good else bad).append('%-46s claim=%s  actual=%s' % (label, claim, actual))

# --- frames / shots / gpu from the logs
LOGS=[R+'/run_2026-09-05b/full3.log', R+'/run_2026-09-06_full1917/render1917.log',
      R+'/run_2026-09-06_four/run_alpha.log', R+'/run_2026-09-06_four/run_all.log',
      R+'/run_2026-09-07_rest11/run_rest.log']
meta={}
for L in LOGS:
    if not os.path.exists(L): continue
    for line in open(L, encoding='utf-8', errors='replace'):
        m=re.match(r'^(\w+): (\d+)f @([\d.]+) (\d+) shot\(s\).*?bad_frames=(\d+).*?([\d.]+)s\s*$', line.strip())
        if m: meta[m.group(1)]=dict(frames=int(m.group(2)),shots=int(m.group(4)),bad=int(m.group(5)),secs=float(m.group(6)))
tot_f=sum(v['frames'] for v in meta.values()); tot_s=sum(v['shots'] for v in meta.values())
tot_h=sum(v['secs'] for v in meta.values())/3600.0
check('total frames', 9640, tot_f)
check('total shots', 56, tot_s)
check('total GPU hours', 6.8, round(tot_h,1), 0.05)
check('clips run', 18, len(meta))
check('bad_frames anywhere', 0, sum(v['bad'] for v in meta.values()))
check('clips with cuts', 9, sum(1 for v in meta.values() if v['shots']>1))

# per-clip frames / shots / gpu as printed in the README table
TABLE={'1917':(724,1,31),'butter':(382,9,17),'ipman':(501,7,20),'bilibili':(1372,13,57),
 'es2':(879,1,40),'dlh':(661,1,28),'dance3':(619,1,28),'asianboss2':(354,1,15),
 'codylexi':(490,1,20),'dance':(363,2,18),'dance2':(607,1,25),'eddie':(143,2,4),
 'interview':(642,4,27),'jensen':(509,1,20),'leo':(208,1,7),'microsoft':(567,2,21),
 'shakira':(369,4,17),'tryguys':(250,4,11)}
for c,(f,sh,mins) in TABLE.items():
    m=meta.get(c)
    if not m: bad.append('%-46s NO LOG ROW'%c); continue
    check('%s frames'%c, f, m['frames'])
    check('%s shots'%c, sh, m['shots'])
    check('%s GPU minutes'%c, mins, round(m['secs']/60.0), 1)

# --- solos table
SOL={}
for c in TABLE:
    p=T+'/results/%s.json'%c
    d=load(p)
    if d: SOL[c]=d
check('solos jsons present', 18, len(SOL))
WHO={'1917':(72,0.0498,146,0.0326),'ipman':(21,0.0171,2,0.0142),'butter':(13,0.0358,0,0.0),
     'dance':(11,0.0131,0,0.0),'bilibili':(6,0.0301,0,0.0)}
for c,(n1,m1,n2,m2) in WHO.items():
    d=SOL[c]
    check('%s v1-only frames'%c, n1, d['frames_with_v1_only'])
    check('%s v1-only max'%c, m1, round(max((p['v1_only'] for p in d['per_frame']),default=0.0),4), 0.0001)
    check('%s v2-only frames'%c, n2, d['frames_with_v2_only'])
    check('%s v2-only max'%c, m2, round(max((p['v2_only'] for p in d['per_frame']),default=0.0),4), 0.0001)
zero=[c for c in SOL if SOL[c]['frames_with_v1_only']==0 and SOL[c]['frames_with_v2_only']==0]
check('clips clean both ways', 13, len(zero))
check('frames in the clean clips', 6297, sum(SOL[c]['frames'] for c in zero))

# --- coverage flags
COVN={'interview':(369,642),'microsoft':(113,567),'tryguys':(69,250),'shakira':(57,369),'dance':(41,363)}
lowiou=0
for c in ['asianboss2','codylexi','dance','dance2','eddie','interview','jensen','leo','microsoft','shakira','tryguys']:
    d=load(R+'/run_2026-09-07_rest11/%s_cov.json'%c)
    rows=d['rows']
    lowiou+=sum(1 for r in rows if r[3]<0.80)
    if c in COVN:
        check('%s frames gap>1%%'%c, COVN[c][0], sum(1 for r in rows if r[4]>0.01))
        check('%s cov frames'%c, COVN[c][1], len(rows))
b=load(R+'/run_2026-09-06_four/bilibili_cov.json')
lowiou+=sum(1 for r in b if r[3]<0.80)
# recomputed over all eighteen
allcov={}
import glob
for f in glob.glob(T+'/results/*_cov.json'):
    c=os.path.basename(f).replace('_cov.json','')
    d=load(f); allcov[c]= d if isinstance(d,list) else d['rows']
check('cov jsons for all 18', 18, len(allcov))
check('total compared frames', 9463, sum(len(v) for v in allcov.values()))
low={c:sum(1 for r in v if r[3]<0.80) for c,v in allcov.items()}
check('frames below 0.80 IoU, all 18', 102, sum(low.values()))
check('  of which 1917', 86, low['1917'])
check('  of which bilibili', 16, low['bilibili'])
check('clips with zero such frames', 16, sum(1 for c in low if low[c]==0))
# the 5.5 before/after on the blunt measure
pre=load(R+'/run_2026-09-07_rest11/1917_pre55_cov.json')['rows']
check('1917 pre-5.5 frames IoU<0.80', 103, sum(1 for r in pre if r[3]<0.80))
check('1917 post frames IoU<0.80', 86, low['1917'])
check('1917 pre-5.5 gap>1%', 210, sum(1 for r in pre if r[4]>0.01))
check('1917 post gap>1%', 199, sum(1 for r in allcov['1917'] if r[4]>0.01))
check('1917 pre-5.5 mean IoU', 0.8934, round(st.mean(r[3] for r in pre),4), 0.0001)
check('1917 post mean IoU', 0.9023, round(st.mean(r[3] for r in allcov['1917']),4), 0.0001)
# structure
def struct(rows):
    g=[r[4] for r in rows]; m=st.median(g); srt=sorted(g,reverse=True); t=max(1,len(srt)//10)
    return round(max(g)/m,1), round(sum(srt[:t])/sum(g),2)
check('1917 max/median', 20.7, struct(allcov['1917'])[0], 0.15)
check('1917 worst-tenth share', 0.42, struct(allcov['1917'])[1], 0.01)
check('bilibili max/median', 17.1, struct(allcov['bilibili'])[0], 0.15)
check('gap>1% flagged clips', 9, sum(1 for c,v in allcov.items() if sum(1 for r in v if r[4]>0.01) > 5))
gb=[r[4] for r in b]; medb=st.median(gb)
check('bilibili max/median gap', 17.1, round(max(gb)/medb,1), 0.15)
sb=sorted(gb,reverse=True); check('bilibili worst-tenth share', 0.35, round(sum(sb[:len(sb)//10])/sum(gb),2), 0.01)

# --- code zip
import zipfile
z=zipfile.ZipFile(T+'/code/vbgr2_source.zip')
files=[x for x in z.namelist() if not x.endswith('/')]
check('source zip files', 79, len(files))
check('bench analysis scripts', 27, len([x for x in files if x.startswith('bench/') and x.endswith('.py')]))

print('PASS %d' % len(ok))
for l in bad: print('FAIL ' + l)
print('FAILURES: %d' % len(bad))

