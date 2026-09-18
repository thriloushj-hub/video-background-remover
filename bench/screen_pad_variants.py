"""Second CPU screen (5.10b, 17 Sep): does a flat GREY pad, or a 32 px replicate pad,
avoid the side-edge retreat that pad 64 replicate caused?  All four edges,
7 clips x 3 frames, product seeder path.  Band versions scored from the same
seeds, so every arm is compared on identical detections."""
import sys, os, json, cv2, numpy as np, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'bench'))
torch.set_num_threads(os.cpu_count() or 4)
from vbgr.config import Config
from vbgr.pipeline import Pipeline
from vbgr.detect import select_person_boxes, held_props
from vbgr.seed import build_seed, MaskRCNNSeeder
from vbgr import video_io, refine
import ab_edge_pad_final as ab

P = 64
ARMS = {'pad0': None,
        'rep32': (cv2.BORDER_REPLICATE, None, 32),
        'rep64': (cv2.BORDER_REPLICATE, None, 64),
        'grey64': (cv2.BORDER_CONSTANT, (128,128,128), 64)}

def make(spec):
    def _inst(self, frame, score_min=None, mask_thr=None):
        if spec is None:
            return self._instances_raw(frame, score_min, mask_thr)
        mode, val, p = spec
        H, W = frame.shape[:2]
        img = (cv2.copyMakeBorder(frame,p,p,p,p,mode,value=val) if val
               else cv2.copyMakeBorder(frame,p,p,p,p,mode))
        m, b = self._instances_raw(img, score_min, mask_thr)
        m = np.asarray(m); b = np.asarray(b,float).reshape(-1,4)
        if len(m)==0: return m,b
        m = m[:, p:p+H, p:p+W]; b = b - p
        b[:,0::2]=np.clip(b[:,0::2],0,W-1); b[:,1::2]=np.clip(b[:,1::2],0,H-1)
        keep=[i for i in range(len(m)) if m[i].any()]
        return m[keep], b[keep]
    return _inst

clips = sys.argv[1:]
cfg = Config(); cfg.detect.device='cpu'
pipe = Pipeline(cfg); sd = pipe.seeder; sd._device='cpu'
rows = []
for clip in clips:
    name = os.path.basename(clip).split('.')[0].split('_')[0]
    fr = video_io.read_all(clip, None)
    for fi in (0,24,48):
        f = fr[fi]; H,W = f.shape[:2]
        people, props = pipe.detector.detect(f)
        kept,_ = select_person_boxes(people,W,H,cfg.detect.person_rel_size_min,
                                     cfg.detect.box_score_ratio,cfg.detect.person_rel_area_min,
                                     cfg.detect.conf_abs_min)
        S = {}
        for arm, spec in ARMS.items():
            MaskRCNNSeeder._instances = make(spec)
            s = build_seed(f, sd, kept, held_props(props,kept), cfg.seed)
            S[arm] = refine.erode_dilate(s.mask, cfg.matting.r_erode,
                                         cfg.matting.r_dilate).astype(np.float32)
        band = np.zeros((H,W), bool)
        for a in ('rep32','rep64','grey64'):
            p = 32 if a=='rep32' else 64
            bd = np.zeros((H,W),bool); bd[:p]=bd[-p:]=True; bd[:,:p]=bd[:,-p:]=True
            u = S['pad0'].copy(); u[bd] = np.maximum(S['pad0'], S[a])[bd]
            S['band_'+a] = u
        for arm in S:
            if arm=='pad0': continue
            for e in ab.EDGES:
                cols = ab.ref_columns([S['pad0'], S[arm]], e)
                s0 = ab.edge_stats(S['pad0'], e, cols); s1 = ab.edge_stats(S[arm], e, cols)
                if not s0: continue
                rows.append(dict(clip=name, f=fi, arm=arm, edge=e, n=s0['n'],
                                 p0=s0['edge_alpha'], a=s1['edge_alpha'],
                                 s0=s0['short_px'], s1=s1['short_px'],
                                 dw90=s1['w90']-s0['w90']))
            rows.append(dict(clip=name, f=fi, arm=arm, edge='_meta',
                             lost=ab.standalone_only(S['pad0'], S[arm]),
                             new=ab.standalone_only(S[arm], S['pad0']),
                             flip=ab.interior_churn(S[arm], S['pad0'])['flip']))
        print(name, fi, 'done', flush=True)
        json.dump(rows, open('pad_variant_screen2.json','w'))
print('SCREEN2_DONE')
