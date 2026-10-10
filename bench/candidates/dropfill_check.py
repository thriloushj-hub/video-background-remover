"""Per-frame missing/extra vs v1 (9x9-opened, hard 0.5) before/after dropfill, for the frames dropfill changed (+ context)."""
import cv2,numpy as np,sys,json
c=sys.argv[1]; R=f'bench/results/run_2026-10-02_all18/{c}'; V=f'compare_app/videos/{c}_v1.mp4'
G=np.array([152,255,120],np.float32); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
o=cv2.VideoCapture(f'{R}/{c}_alpha.mp4'); n_=cv2.VideoCapture(f'{R}/dropfill/{c}_alpha.mp4'); v=cv2.VideoCapture(V)
H=int(o.get(4)); vh=int(v.get(4)); top=(vh-H)//2
op=lambda m: int(cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,k).sum())
rows=[]; i=0; tm_o=tm_n=te_o=te_n=0
while True:
    r1,a=o.read(); r2,b=n_.read(); r3,f=v.read()
    if not(r1 and r2 and r3): break
    A=a[...,1]>127; B=b[...,1]>127; v1=(np.linalg.norm(f.astype(np.float32)-G,axis=2)>=60)[top:top+H]
    ch=int((A!=B).sum())
    mo,mn,eo,en=op(v1&~A),op(v1&~B),op(A&~v1),op(B&~v1); tm_o+=mo;tm_n+=mn;te_o+=eo;te_n+=en
    if ch>400: rows.append((i,ch,mo,mn,eo,en))
    i+=1
print(f'{c}: {i} frames. big-chunk missing vs v1 mean/frame {tm_o/i:.0f} -> {tm_n/i:.0f}; extra {te_o/i:.0f} -> {te_n/i:.0f}')
for r in rows: print('  f%4d changed %6d  missing %6d -> %6d  extra %6d -> %6d'%r)
