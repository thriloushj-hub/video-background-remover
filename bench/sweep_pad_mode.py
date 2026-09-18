"""Seed-level screen of frame-edge pad variants, on ALL FOUR edges (5.10b, 17 Sep).

Why this exists
---------------
The 16 Sep CPU sweep measured the BOTTOM edge only, because that is where the
defect was first seen.  The 17 Sep GPU A/B then found the full replicate pad
breaking tryguys' RIGHT edge on 67 of 72 frames -- visible in the seed itself
(6 -> 33 px short), so a CPU pass could have caught it.  This is that CPU pass:
every arm, every edge, the product seeder path (YOLO boxes -> `_match` ->
`build_seed` -> `erode_dilate`), plus the integrity and interior controls.

Arms: shipped (pad 0); replicate 64 all edges; reflect-101 64 all edges;
replicate 64 bottom only.  The band arm is scored from the saved seeds
afterwards, and once it won it went into the seeder itself as
`maskrcnn_edge_pad_band` -- see `BAND_RESULT.md`.

It monkeypatches `MaskRCNNSeeder._instances` rather than adding config knobs,
deliberately: three of the four arms were expected to lose, and a rejected arm
should not leave a setting behind.  Reflect DID lose, spectacularly --
`evidence_reflect_swallows_the_room.jpg`.

Usage:  python bench/sweep_pad_mode.py CLIP.mp4 [CLIP.mp4 ...]
Writes: sweep_pad_mode.json, and every arm's seed per frame under sweep_seeds/.
"""
import sys, os, json, cv2, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'bench'))
import torch; torch.set_num_threads(os.cpu_count() or 4)
from vbgr.config import Config
from vbgr.pipeline import Pipeline
from vbgr.detect import select_person_boxes, held_props
from vbgr.seed import build_seed, MaskRCNNSeeder
from vbgr import video_io, refine
import ab_edge_pad_final as ab

ARMS = {  # name: (mode, top, bottom, left, right)
  'pad0':      None,
  'rep64':     (cv2.BORDER_REPLICATE, 64,64,64,64),
  'refl64':    (cv2.BORDER_REFLECT_101, 64,64,64,64),
  'rep64_b':   (cv2.BORDER_REPLICATE, 0,64,0,0),
}
def make_instances(spec):
    def _inst(self, frame_bgr, score_min=None, mask_thr=None):
        if spec is None:
            return self._instances_raw(frame_bgr, score_min, mask_thr)
        mode,t,b,l,r = spec; H,W = frame_bgr.shape[:2]
        pad = cv2.copyMakeBorder(frame_bgr,t,b,l,r,mode)
        m, bx = self._instances_raw(pad, score_min, mask_thr)
        m = np.asarray(m); bx = np.asarray(bx,float).reshape(-1,4)
        if len(m)==0: return m,bx
        m = m[:, t:t+H, l:l+W]; bx = bx - [l,t,l,t]
        bx[:,0::2]=np.clip(bx[:,0::2],0,W-1); bx[:,1::2]=np.clip(bx[:,1::2],0,H-1)
        keep=[i for i in range(len(m)) if m[i].any()]
        return m[keep], bx[keep]
    return _inst

clips=sys.argv[1:]; FR=[0,24,48]
os.makedirs('sweep_seeds', exist_ok=True)
cfg=Config(); cfg.detect.device='cpu'
pipe=Pipeline(cfg); sd=pipe.seeder; sd._device='cpu'
out={}
for clip in clips:
    name=os.path.basename(clip).split('.')[0]; fr=video_io.read_all(clip,None)
    out[name]={}
    for fi in FR:
        f=fr[fi]; H,W=f.shape[:2]
        people,props=pipe.detector.detect(f)
        kept,_=select_person_boxes(people,W,H,cfg.detect.person_rel_size_min,cfg.detect.box_score_ratio,cfg.detect.person_rel_area_min,cfg.detect.conf_abs_min)
        seeds={}
        for arm,spec in ARMS.items():
            MaskRCNNSeeder._instances = make_instances(spec)
            s=build_seed(f,sd,kept,held_props(props,kept),cfg.seed)
            seeds[arm]=refine.erode_dilate(s.mask,cfg.matting.r_erode,cfg.matting.r_dilate).astype(np.float32)
            seeds[arm+'_n']=sum(m is not None for m in sd.last_aligned)
        rec={}
        for arm in ARMS:
            if arm=='pad0': continue
            r={}
            for e in ab.EDGES:
                cols=ab.ref_columns([seeds['pad0'],seeds[arm]],e)
                s0=ab.edge_stats(seeds['pad0'],e,cols); s1=ab.edge_stats(seeds[arm],e,cols)
                if s0: r[e]=dict(n=s0['n'],p0=s0['edge_alpha'],a=s1['edge_alpha'],s0=s0['short_px'],s1=s1['short_px'],w90_0=s0['w90'],w90_1=s1['w90'])
            lost=ab.standalone_only(seeds['pad0'],seeds[arm]); new=ab.standalone_only(seeds[arm],seeds['pad0'])
            ch=ab.interior_churn(seeds[arm],seeds['pad0'])
            rec[arm]=dict(edges=r,lost=lost,new=new,churn=ch['flip'],nmatched=(seeds['pad0_n'],seeds[arm+'_n']))
        out[name][fi]=rec
        np.savez_compressed(f'sweep_seeds/{name}_{fi}.npz', frame=f, **{k:(v*255).astype(np.uint8) for k,v in seeds.items() if not k.endswith('_n')})
        print(name,fi,json.dumps({a:{e:(round(v['p0'],2),round(v['a'],2),v['s0'],v['s1']) for e,v in x['edges'].items()} for a,x in rec.items()}),flush=True)
    json.dump(out,open('sweep_pad_mode.json','w'),indent=1)
print('SWEEP_DONE')
