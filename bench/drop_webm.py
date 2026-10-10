"""Pure script output only (Runbo, 4 Oct): no per-clip overrides.
Pairs matte and source by frame INDEX (alphamerge otherwise pairs by timestamp and drifts on dance/jensen).
Idempotent chunked transparent-webm builder for the 4 Oct drop (device shell kills jobs at ~3 min).
Each call encodes pending 600-frame chunks until ~140 s are used, then finalises any clip whose chunks are all done."""
import subprocess,os,time,json,glob
D='_to_send_2026-10-04'; R='bench/results/run_2026-10-02_all18'; S='bench_payload/src18'; T=f'{D}/_chunks'; os.makedirs(T,exist_ok=True)
CL='1917 asianboss2 bilibili butter codylexi dance dance2 dance3 dlh eddie es2 interview ipman jensen leo microsoft shakira tryguys'.split()
if os.environ.get('ONLY'): CL=os.environ['ONLY'].split()
def alpha(c):
    return f'{R}/{c}/finished/{c}_alpha_C.mp4'
def nfr(p): return int(subprocess.run(['ffprobe','-v','error','-count_packets','-select_streams','v:0','-show_entries','stream=nb_read_packets','-of','csv=p=0',p],capture_output=True,text=True).stdout.strip())
t0=time.time(); N=600
for c in CL:
    os.makedirs(f'{D}/outputs/{c}',exist_ok=True); out=f'{D}/outputs/{c}/{c}_transparent.webm'
    if os.path.exists(out): continue
    n=min(nfr(f'{S}/{c}.mp4'),nfr(alpha(c))); k=(n+N-1)//N
    for j in range(k):
        ch=f'{T}/{c}_{j:02d}.webm'
        if os.path.exists(ch): continue
        if time.time()-t0>55: print('PAUSE'); raise SystemExit
        a,b=j*N,min(n,(j+1)*N); r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','csv=p=0',f'{S}/{c}.mp4'],capture_output=True,text=True).stdout.strip()
        f=f"[0:v]trim=start_frame={a}:end_frame={b},settb=1/1000000,setpts=N/({r})/TB[s];[1:v]trim=start_frame={a}:end_frame={b},settb=1/1000000,setpts=N/({r})/TB,format=gray[m];[s][m]alphamerge,format=yuva420p[v]"
        subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',f'{S}/{c}.mp4','-i',alpha(c),'-filter_complex',f,'-map','[v]','-an','-frames:v',str(b-a),'-vsync','passthrough','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','8','-row-mt','1','-b:v','0','-crf','30','-auto-alt-ref','0',ch+'.tmp.webm'],check=True)
        os.replace(ch+'.tmp.webm',ch); print('chunk',c,j+1,'/',k,f'{time.time()-t0:.0f}s',flush=True)
    lst=f'{T}/{c}.txt'; open(lst,'w').write(''.join(f"file '{os.path.abspath(f'{T}/{c}_{j:02d}.webm')}'\n" for j in range(k)))
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-f','concat','-safe','0','-i',lst,'-i',f'{S}/{c}.mp4','-map','0:v','-map','1:a?','-c:v','copy','-c:a','libopus','-shortest',out+'.tmp.webm'],check=True)
    os.replace(out+'.tmp.webm',out); print('FINAL',c,n,'frames',flush=True)
print('ALL_DONE' if all(os.path.exists(f'{D}/outputs/{c}/{c}_transparent.webm') for c in CL) else 'MORE')
