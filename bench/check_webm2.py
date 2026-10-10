# full per-frame check: webm alpha frame k vs matte frame k (and k-1, k+1), at 240x135
import subprocess,numpy as np,sys
def dec(args,n):
    b=subprocess.run(['ffmpeg','-v','error']+args+['-vsync','passthrough','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    return np.frombuffer(b,np.uint8).reshape(-1,135,240).astype(np.int16)
D='_to_send_2026-10-04/outputs'
for c in sys.argv[1:]:
    a=dec(['-c:v','libvpx-vp9','-i',f'{D}/{c}/{c}_transparent.webm','-vf','alphaextract,scale=240:135'],0)
    m=dec(['-i',f'{D}/{c}/{c}_matte.mp4','-vf','scale=240:135'],0)
    n=min(len(a),len(m)); bad=[]
    for k in range(1,n-1):
        d0=np.abs(a[k]-m[k]).mean(); dm=np.abs(a[k]-m[k-1]).mean(); dp=np.abs(a[k]-m[k+1]).mean()
        if d0>min(dm,dp)+0.3: bad.append(k)
    print(c,'webm',len(a),'matte',len(m),'misaligned frames',len(bad),bad[:8])
