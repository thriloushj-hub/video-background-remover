import cv2, numpy as np, torch, os, json
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
torch.set_num_threads(os.cpu_count() or 4)
UP="/mnt/user-data/uploads/vbgr2/_to_delete/edgeprobe_2026-09-14"
det=maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
def people(img,thr=0.80):
    t=torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255)
    with torch.no_grad(): o=det([t])[0]
    s=(o["labels"]==1)&(o["scores"]>thr)
    return o["masks"][s,0].numpy(), o["boxes"][s].numpy().astype(float), o["scores"][s].numpy()

def edges_needed(b,H,W,frac):
    """Which frame edges need padding.  Tolerance must be MUCH larger than the
    box shortfall (2-11 px) -- gating at 12 px reads the very signal the bug
    corrupts, which is why the first asymmetric attempt padded nothing."""
    tx, ty = frac*W, frac*H
    L=T=R=B=False
    for x0,y0,x1,y1 in b:
        L|=x0<=tx; T|=y0<=ty; R|=x1>=W-1-tx; B|=y1>=H-1-ty
    return L,T,R,B

def arm(f,H,W,mode,P,frac=0.05):
    if mode=="pad0": return people(f)
    if mode=="sym": l=t_=r=bm=P
    else:
        _,b0,_=people(f)
        L,T,R,B=edges_needed(b0,H,W,frac)
        l,t_,r,bm=(P if L else 0),(P if T else 0),(P if R else 0),(P if B else 0)
    img=cv2.copyMakeBorder(f,t_,bm,l,r,cv2.BORDER_REPLICATE)
    m,b,sc=people(img)
    if len(m)==0: return m,b,sc
    m=m[:,t_:t_+H, l:l+W]; b=b-np.array([l,t_,l,t_],float)
    k=[i for i in range(len(m)) if (m[i]>0.5).sum()>500]
    return m[k],b[k],sc[k]

def body_cols(f,u,H):
    ref=u[H-100:H-80,:].mean(axis=0); cols=np.where(ref>0.95)[0]
    return np.array([c for c in cols
                     if np.linalg.norm(f[-1,c].astype(float)-f[H-40:H-20,c].astype(float).mean(0))<=45])

CLIPS=["microsoft_w249_321.mp4","ipman_w318_389.mp4","interview_w262_334.mp4",
       "tryguys_w69_141.mp4","1917_w4_76.mp4","codylexi_w4_76.mp4","jensen_w4_76.mp4"]
FR=(0,24,48); MIN_AREA=0.005; GATE=0.60
print(f"{'clip':<11}{'arm':<9}{'edge a':>8}{'absent':>8}{'realLOST':>10}{'realMERGE':>11}{'churn%':>8}")
tot={}
for clip in CLIPS:
    cap=cv2.VideoCapture(os.path.join(UP,clip)); frames=[]
    for i in range(49):
        ok,f=cap.read()
        if not ok: break
        if i in FR: frames.append(f)
    cap.release(); H,W=frames[0].shape[:2]
    bd=np.zeros((H,W),bool); bd[:64,:]=bd[-64:,:]=True; bd[:,:64]=bd[:,-64:]=True
    base_bins=[]
    for mode in ("pad0","sym","asym2"):
        e=[];ab=[];rl=0;rm=0;bins=[]
        for f in frames:
            m,b,sc=arm(f,H,W,mode,64)
            if len(m)==0: continue
            u=np.max(m,axis=0); bins.append(u>0.5)
            cols=body_cols(f,u,H)
            if len(cols)>=20:
                e.append(u[-1,cols].mean()); ab.append((u[-1,cols]<0.5).mean())
            if mode!="pad0":
                m0,b0,_=people(f)
                if len(b0)==0: continue
                tall=max(x[3]-x[1] for x in b0)
                B1=[(m[j]>0.5) for j in range(len(m))]
                for i2 in range(len(m0)):
                    A=(m0[i2]>0.5)
                    if A.sum()/(H*W)<MIN_AREA: continue
                    if (b0[i2][3]-b0[i2][1])/tall < GATE: continue   # bystander
                    iou=max((((A&x).sum()/max(1,(A|x).sum())) for x in B1), default=0)
                    cov=max((((A&x).sum()/max(1,A.sum())) for x in B1), default=0)
                    if iou<0.25 and cov<0.60: rl+=1
                    elif iou<0.25: rm+=1
        if mode=="pad0": base_bins=bins
        ch=0.0
        if mode!="pad0" and base_bins:
            ch=float(np.mean([((a^b)&~bd).sum()/max(1,a.sum())
                              for a,b in zip(base_bins,bins)]))*100
        print(f"{clip.split('_')[0] if mode=='pad0' else '':<11}{mode:<9}"
              f"{(np.mean(e) if e else 0):>8.3f}{(np.mean(ab) if ab else 1):>8.2f}"
              f"{rl:>10}{rm:>11}{ch:>8.2f}")
        tot.setdefault(mode,{"e":[], "l":0, "m":0, "c":[]})
        tot[mode]["e"].append(np.mean(e) if e else 0); tot[mode]["l"]+=rl
        tot[mode]["m"]+=rm; tot[mode]["c"].append(ch)
    print()
print("SET TOTALS")
for mode in ("sym","asym2"):
    t=tot[mode]
    print(f"  {mode:<7} mean edge alpha {np.mean(t['e']):.3f}  real LOST {t['l']}  "
          f"real MERGED {t['m']}  mean churn {np.mean(t['c']):.2f}%")
