"""Look sheet: source | before | after | added(red) for given frames of a clip, cropped to the changed region."""
import cv2,numpy as np,sys
c=sys.argv[1]; idx=[int(x) for x in sys.argv[2:]]; R=f'bench/results/run_2026-10-02_all18/{c}'; S=f'bench_payload/src18/{c}.mp4'
def grab(p,fn):
    cap=cv2.VideoCapture(p); out={}
    for i in range(max(idx)+1):
        r,f=cap.read()
        if i in idx: out[i]=fn(f)
    return out
F=grab(S,lambda f:f); O=grab(f'{R}/{c}_alpha.mp4',lambda f:f[...,1]); N=grab(f'{R}/dropfill/{c}_alpha.mp4',lambda f:f[...,1])
G=np.array([152,255,120],np.float32); tiles=[]
for i in idx:
    d=(N[i].astype(int)-O[i].astype(int)); ys,xs=np.where(d>40)
    if len(xs)==0: print('no change f',i); continue
    H,W=O[i].shape; x0,x1=max(0,xs.min()-100),min(W,xs.max()+100); y0,y1=max(0,ys.min()-100),min(H,ys.max()+100)
    def comp(a): a=a[y0:y1,x0:x1,None].astype(np.float32)/255; return (F[i][y0:y1,x0:x1]*a+G*(1-a)).astype(np.uint8)
    src=F[i][y0:y1,x0:x1].copy(); over=src.copy(); over[d[y0:y1,x0:x1]>40]=(0,0,255)
    row=np.hstack([src,comp(O[i]),comp(N[i]),over]); h=340; row=cv2.resize(row,(int(row.shape[1]*h/row.shape[0]),h))
    cv2.putText(row,f'{c} f{i}: source | before | after | added=red',(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2); tiles.append(row)
w=max(t.shape[1] for t in tiles); tiles=[cv2.copyMakeBorder(t,0,0,0,w-t.shape[1],cv2.BORDER_CONSTANT) for t in tiles]
out=f'_to_delete/hand_edited_1004/check/{c}_dropfill.jpg'; cv2.imwrite(out,np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,85]); print(out)
