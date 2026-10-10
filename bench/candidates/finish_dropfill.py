"""finish C on the dropfill alphas (9 Oct). Derived from finish_all18.py; reads <clip>/dropfill/<clip>_alpha.mp4, writes <clip>/finished/."""
"""Apply hard_round (C) on CPU to every clip in run_2026-10-02_all18, resumable in 400-frame chunks (device_bash kills jobs at call end).
Outputs <clip>/finished/<clip>_alpha_C.mp4, <clip>_green_C.mp4, <clip>_green_C_av.mp4."""
import cv2, numpy as np, subprocess, sys, os, time, glob
sys.path.insert(0, os.path.expanduser('~/mnt/vbgr2'))
from vbgr.refine import edge_finish
R=os.path.expanduser('~/mnt/vbgr2/bench/results/run_2026-10-02_all18'); DF=True; S=os.path.expanduser('~/mnt/vbgr2/bench_payload/src18')
G=np.array([152,255,120],np.uint8); CH=400; BUDGET=float(os.environ.get('BUDGET','140')); T0=time.time()
clips=open(f'{R}/dropfill_changed.txt').read().split()
def nframes(p):
    v=cv2.VideoCapture(p); n=int(v.get(cv2.CAP_PROP_FRAME_COUNT)); v.release(); return n
for c in clips:
    out=f'{R}/{c}/finished'; parts=f'{out}/parts'; os.makedirs(parts,exist_ok=True)
    if os.path.exists(f'{out}/{c}_green_C_av.mp4'): continue
    N=nframes(f'{R}/{c}/dropfill/{c}_alpha.mp4'); nch=(N+CH-1)//CH
    src=cv2.VideoCapture(f'{S}/{c}.mp4'); al=cv2.VideoCapture(f'{R}/{c}/dropfill/{c}_alpha.mp4'); fps=al.get(cv2.CAP_PROP_FPS); W=int(al.get(3)); H=int(al.get(4))
    for k in range(nch):
        if os.path.exists(f'{parts}/g_{k:02d}.ok'): continue
        if time.time()-T0>BUDGET: print(f'pause {c} chunk {k}/{nch}',flush=True); sys.exit(0)
        src.set(cv2.CAP_PROP_POS_FRAMES,k*CH); al.set(cv2.CAP_PROP_POS_FRAMES,k*CH)
        wa=cv2.VideoWriter(f'{parts}/a_{k:02d}.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); wg=cv2.VideoWriter(f'{parts}/g_{k:02d}.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
        for i in range(CH):
            r,f=src.read(); r2,fa=al.read()
            if not (r and r2): break
            a=edge_finish(fa[...,1].astype(np.float32)/255.0,'hard_round')
            wa.write(np.repeat((a*255+0.5).astype(np.uint8)[...,None],3,2)); wg.write((f*a[...,None]+G*(1-a[...,None])).astype(np.uint8))
        wa.release(); wg.release(); open(f'{parts}/g_{k:02d}.ok','w').close(); print(f'chunk {c} {k+1}/{nch}',flush=True)
    src.release(); al.release()
    for key,pre in (('alpha','a'),('green','g')):
        lst=f'{parts}/{pre}_list.txt'; open(lst,'w').write(''.join(f"file '{parts}/{pre}_{k:02d}.mp4'\n" for k in range(nch)))
        subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',lst,'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',f'{out}/{c}_{key}_C.mp4'],check=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i',f'{out}/{c}_green_C.mp4','-i',f'{S}/{c}.mp4','-map','0:v','-map','1:a?','-c:v','copy','-c:a','aac','-shortest',f'{out}/{c}_green_C_av.mp4'],check=True)
    print(f'done {c} {N}f',flush=True)
print('ALL_FINISHED')
