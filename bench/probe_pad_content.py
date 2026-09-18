"""Why does a 64 px replicate pad retreat tryguys' right edge?  (5.10b, 17 Sep)

Answer, from the table this writes (`pad_content_right_edge.txt`): it is the
CONTENT of the pad, not its geometry.  A replicate pad copies the last column
outward, so the jacket becomes a 64 px band of flat colour running off frame;
Mask R-CNN's soft mask then loses confidence and its boundary retreats INTO the
real picture, further the wider the smear (pad 64: 24-45 px short; pad 128: up
to 61).  A flat grey pad of the same width does not do it (0 px short).  Box
score stays 0.997-0.999 and the implied 28x28 mask cell is 18.4-20.4 px in every
arm, so neither detection confidence nor mask-grid scale explains it.

Arms: right-edge-only pad, widths 0/4/8/16/32/64/128, modes replicate and a
flat grey constant.  For each: the instance box IN PADDED COORDINATES (does the
detector think the person continues into the pad?), the 28x28 mask cell size
that box implies, and the SOFT mask profile at the right frame edge.
"""
import sys, os, cv2, numpy as np, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
torch.set_num_threads(os.cpu_count())
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from vbgr import video_io
det = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()

def run(img):
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255)
    with torch.no_grad(): o = det([t])[0]
    s = (o['labels']==1) & (o['scores']>0.80)
    return o['masks'][s,0].numpy(), o['boxes'][s].numpy(), o['scores'][s].numpy()

clip = sys.argv[1] if len(sys.argv) > 1 else 'edge/tryguys_w69_141.mp4'
frames=video_io.read_all(clip,None)
ROWS=slice(600,1000)          # the jacket rows at the right edge
for fi in (0,24,48):
    f=frames[fi]; H,W=f.shape[:2]
    print(f'== f{fi}')
    for mode,name in ((cv2.BORDER_REPLICATE,'replicate'), (None,'grey')):
        for p in (0,4,8,16,32,64,128):
            if p==0 and name=='grey': continue
            if p==0: img=f
            elif name=='grey': img=cv2.copyMakeBorder(f,0,0,0,p,cv2.BORDER_CONSTANT,value=(128,128,128))
            else: img=cv2.copyMakeBorder(f,0,0,0,p,mode)
            m,b,sc=run(img)
            # the instance whose box is rightmost = the man at the right edge
            i=int(np.argmax(b[:,2]))
            mm=m[i][:, :W]
            prof=[round(float(mm[ROWS, W-1-d].mean()),3) for d in (0,10,20,40,60)]
            reach=[]
            for y in range(ROWS.start,ROWS.stop,50):
                xs=np.where(mm[y]>0.5)[0]
                reach.append(W-1-int(xs.max()) if len(xs) else None)
            bw=b[i,2]-b[i,0]
            print(f'  {name:9s} pad {p:4d}  score {sc[i]:.3f}  box x2 {b[i,2]:7.1f} (frame ends {W-1})'
                  f'  box w {bw:6.1f}  cell {bw/28:5.1f}px   soft at d=0,10,20,40,60: {prof}   px short: {reach}')
