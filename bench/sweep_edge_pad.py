import cv2, numpy as np, torch, json, os, sys, time
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
torch.set_num_threads(os.cpu_count() or 4)
UP="/mnt/user-data/uploads/vbgr2/_to_delete/edgeprobe_2026-09-14"
CLIPS=["microsoft_w249_321.mp4","ipman_w318_389.mp4","interview_w262_334.mp4",
       "tryguys_w69_141.mp4","1917_w4_76.mp4","codylexi_w4_76.mp4","jensen_w4_76.mp4"]
PADS=[0,16,32,64]; FRAMES=[0,24,48]
det=maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()

def detect(img):
    t=torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255)
    with torch.no_grad(): o=det([t])[0]
    s=(o["labels"]==1)&(o["scores"]>0.80)
    return o["masks"][s,0].numpy(), int(s.sum())

def stats(u,H,W):
    """u = soft union mask. Bottom-edge behaviour + dilation controls."""
    ref=u[H-100:H-80,:].mean(axis=0); cols=np.where(ref>0.95)[0]
    if len(cols)<20: return None
    b=u>0.5
    lows=[]
    for c in cols:
        idx=np.where(b[:,c])[0]
        lows.append(H-1-idx.max() if len(idx) else H)
    return dict(short=float(np.median(lows)), absent=float((u[-1,cols]<0.5).mean()),
                edge_alpha=float(u[-1,cols].mean()), w0=int(b[-1,:].sum()),
                w90=int(b[H-95:H-85,:].any(axis=0).sum()), area=float(b.mean()), n=len(cols))

out={}
for clip in CLIPS:
    p=os.path.join(UP,clip)
    if not os.path.exists(p): print("MISSING",clip, flush=True); continue
    cap=cv2.VideoCapture(p); fr=[]
    for i in range(max(FRAMES)+1):
        ok,f=cap.read()
        if not ok: break
        if i in FRAMES: fr.append(f)
    cap.release()
    if not fr: continue
    H,W=fr[0].shape[:2]; out[clip]={"shape":[W,H],"pads":{}}
    for P in PADS:
        acc=[]; nd=[]
        for f in fr:
            img=f if P==0 else cv2.copyMakeBorder(f,P,P,P,P,cv2.BORDER_REPLICATE)
            ms,k=detect(img); nd.append(k)
            if len(ms)==0: continue
            ms=ms if P==0 else ms[:,P:P+H,P:P+W]
            s=stats(np.max(ms,axis=0),H,W)
            if s: acc.append(s)
        if acc:
            agg={k:float(np.mean([a[k] for a in acc])) for k in acc[0]}
            agg["ndet"]=float(np.mean(nd))
            out[clip]["pads"][str(P)]=agg
        print(f"{clip} pad={P} done", flush=True)
    json.dump(out, open("/home/claude/sweep/out.json","w"), indent=1)
print("ALL DONE", flush=True)
