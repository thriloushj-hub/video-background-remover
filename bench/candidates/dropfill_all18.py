"""Apply refine.fill_dropouts (the 9 Oct pipeline stage) to the raw 2 Oct alphas of every clip on CPU, per shot
(same cut detector as the pipeline), memory-light (lazy video readers with a small cache), resumable per clip.
Writes <clip>/dropfill/<clip>_alpha.mp4 (raw soft alpha) + dropfill.json. finish C + green applied afterwards."""
import cv2, numpy as np, sys, os, time, json, subprocess
sys.path.insert(0, os.path.expanduser('~/mnt/vbgr2'))
from vbgr import refine, motion, shots
from vbgr.config import Config
R=os.path.expanduser('~/mnt/vbgr2/bench/results/run_2026-10-02_all18'); S=os.path.expanduser('~/mnt/vbgr2/bench_payload/src18')
cfg=Config(); BUDGET=float(os.environ.get('BUDGET','150')); T0=time.time()
clips=os.environ.get('ONLY','microsoft tryguys interview dance dance2 dance3 shakira butter ipman 1917 leo jensen eddie dlh es2 codylexi bilibili asianboss2').split()
class Lazy:
    """Sequential-only reader (no seeks: cv2 cannot seek exactly on these mp4s). Keeps the last `cap` frames."""
    def __init__(self,p,ch,lo,hi,cap=90):
        self.c=cv2.VideoCapture(p); self.ch=ch; self.lo=lo; self.n=hi-lo; self.cache={}; self.cap=cap; self.pos=0
        for _ in range(lo): self.c.read(); self.pos+=1
    def __len__(self): return self.n
    def __getitem__(self,i):
        if i<0 or i>=self.n: raise IndexError
        j=self.lo+i
        if j in self.cache: return self.cache[j]
        if j<self.pos: raise RuntimeError(f'backward access {j} < {self.pos} beyond cache')
        while self.pos<=j:
            r,f=self.c.read()
            if not r: raise IndexError
            v=(f[...,1].astype(np.float32)/255.0) if self.ch=='a' else f
            self.cache[self.pos]=v; self.pos+=1
            while len(self.cache)>self.cap: self.cache.pop(min(self.cache))
        return self.cache[j]
def small_frames(p):
    c=cv2.VideoCapture(p); out=[]
    while True:
        r,f=c.read()
        if not r: break
        out.append(cv2.resize(f,(320,180),interpolation=cv2.INTER_AREA))
    return out, c.get(cv2.CAP_PROP_FPS)
for c in clips:
    od=f'{R}/{c}/dropfill'; os.makedirs(od,exist_ok=True)
    if os.path.exists(f'{od}/{c}_alpha.mp4'): continue
    if time.time()-T0>BUDGET: print('pause before',c); sys.exit(0)
    sm,fps=small_frames(f'{S}/{c}.mp4'); n=len(sm)
    na=0; _c=cv2.VideoCapture(f'{R}/{c}/{c}_alpha.mp4')
    while _c.read()[0]: na+=1
    n=min(n,na)
    starts=shots.compute_scene_cuts(sm[:n],cfg.shots.cut_threshold,cfg.shots.hist_bins,cfg.shots.min_shot_len,cfg.shots.use_absdiff_guard,cfg.shots.absdiff_sigma) if n>cfg.shots.min_shot_len*2 else [0]
    del sm; rngs=shots.shot_ranges(starts,n); fl=motion.FlowEstimator(); log=[]
    parts=f'{od}/parts'; os.makedirs(parts,exist_ok=True); W=H=None
    step=60; order=[]
    for si,(a,b) in enumerate(rngs):
        for wi,w0 in enumerate(range(0,b-a,step)):
            pf=f'{parts}/s{si:02d}_w{wi:03d}'; order.append(pf)
            if os.path.exists(pf+'.ok'): continue
            if time.time()-T0>BUDGET: print('pause',c,'shot',si,'window',wi,flush=True); sys.exit(0)
            lo=max(0,w0-2); hi=min(b-a,w0+step+2)
            fr=Lazy(f'{S}/{c}.mp4','f',a+lo,a+hi); al=Lazy(f'{R}/{c}/{c}_alpha.mp4','a',a+lo,a+hi)
            o,ch,px=refine.fill_dropouts(fr,al,fl,still_px=cfg.refine.dropout_still_px,still_sigma=cfg.motion.flow_smooth_sigma,min_blob=cfg.refine.dropout_min_blob,warmup=(cfg.refine.dropout_warmup_frames if w0==0 else 0))
            H,W=o[0].shape; wr=cv2.VideoWriter(pf+'.tmp.avi',cv2.VideoWriter_fourcc(*'FFV1'),fps,(W,H))
            for x in o[w0-lo:w0-lo+min(step,b-a-w0)]: wr.write(np.repeat((x*255+0.5).astype(np.uint8)[...,None],3,2))
            wr.release(); os.replace(pf+'.tmp.avi',pf+'.avi'); open(pf+'.ok','w').write(json.dumps({'shot':[a,b],'window':[a+w0,a+min(w0+step,b-a)],'frames_changed':ch,'px_added':px}))
            del o,fr,al
        print(f'{c} shot {si+1}/{len(rngs)} f{a}-{b} {time.time()-T0:.0f}s',flush=True)
    lst=f'{parts}/list.txt'; open(lst,'w').write(''.join(f"file '{pf}.avi'\n" for pf in order))
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-f','concat','-safe','0','-i',lst,'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',f'{od}/{c}_alpha.mp4'],check=True)
    log=[json.load(open(pf+'.ok')) for pf in order]
    json.dump({'clip':c,'n':n,'shots':log},open(f'{od}/dropfill.json','w'),indent=1)
    print('done',c,n,'frames; changed',sum(x['frames_changed'] for x in log),'px',sum(x['px_added'] for x in log),flush=True)
print('ALL_DONE')
