# 18 Sep re-render, 5.10b defaults (edge_pad 64 + band) -- progress

Driven from the Cowork session while Thriloush is at school. One clip at a
time; after each: solos both directions, IoU floor, bottom-edge alpha vs v1,
zip downloaded and byte-verified in ~/Downloads before the next clip starts.
Code: vbgr_code.zip 282288 B, 226 tests passed on the VM, defaults verified
64/True there.

| # | clip | frames | render | solos v1/v2 | IoU mean/min (<0.80) | edge v1 vs v2 | zip verified |
|---|---|---|---|---|---|---|---|
| 1 | microsoft | 567 | 23 min, 2 shots, bad_frames 0 | 0 / 0 | 0.989 / 0.965 (0) | 0.946 / **0.930** | microsoft.zip 15032589 B |
| 2 | interview | 642 | 30 min, 4 shots, bad_frames 0 | 0 / 0 | 0.983 / 0.925 (0) | 0.961 / **0.889** | ON VM (download blocked, see note) |
| 3 | tryguys | 250 | 12 min, 4 shots, bad_frames 0 | 0 / 0 | 0.986 / 0.969 (0) | 0.981 / **0.924** | ON VM |
| 4 | ipman | 325 vs v1 (offset 160) | 20 min, 7 shots, bad_frames 0, reentries 2 | 34 / 6 frames at ~0.001 of frame | 0.960 / 0.697 (**2**) | 0.886 / **0.867** | ON VM -- FLAGGED: 2 frames <0.80 IoU in a blur burst (f394-399, 470); sheet in zip, needs an eye |

| 5 | dance | 363 (matches source) | re-rendered clean 20:19-20:39 after the first pass was an 18-frame partial (see note below); 2 shots, bad_frames 0, reentries 1 | 11 / 0 frames (0.04%) | 0.949 / 0.839 (0) | 0.058 / 0.057 (subject barely at edge) | ON VM 39981678 B |
| 6 | 1917 | 724 | done (driver, 17:33-18:06) | 73 / 145 frames (v2-only 0.3% of frame) | 0.910 / 0.542 (**71**) | 0.736 / 0.703 | ON VM 67647667 B |
| 7 | bilibili | 1372 | done (18:06-19:14) | 0 / 0 | 0.984 / 0.846 (0) | 0.854 / 0.830 | ON VM 58137571 B |
| 8 | butter | 382 | done (19:14-19:33) | 13 / 15 frames (<0.06%) | 0.951 / 0.814 (0) | 0.812 / 0.767 | ON VM 33645009 B |
| 9 | jensen | 509 | done (19:33-19:58) | 0 / 0 | 0.990 / 0.983 (0) | 0.786 / **0.821** | ON VM 15474662 B |
| 10 | leo | 208 | done (19:58-20:07) | 0 / 0 | 0.983 / 0.945 (0) | 0.961 / 0.938 | ON VM 8557532 B |

**Download note:** Chrome stopped honouring `files.download` after the first
zip of the session (known trap; a page reload did not clear it). Zips
accumulate in /content/rr_final and are re-pulled at the end; each is
byte-verified in ~/Downloads before being marked done.

Bottom-edge alpha is the defect metric: pre-fix v2 sat at 0.33-0.48 on this
clip's window against v1's ~0.95. Parity now.

**17:33 UTC restart:** the 15:45 driver start never took -- the run click was
lost when the notebook websocket dropped, the kernel sat idle from 15:43, and
Colab disconnected the UI for inactivity. Same VM on reconnect (clips 1-4
zips, sources, checkpoints all present; dance render from before the gap is
in rr_out awaiting postclip, which driver.sh does first). Driver re-started
17:33 UTC and CONFIRMED running this time (exec count assigned, cell state
running). ETA for clips 5-10: ~20:15 UTC. Lesson now baked in: after starting
the driver, verify the cell actually entered the running state.

**20:15 UTC, driver done (DRIVER_DONE seen):** clips 6-10 rendered and checked
clean except two things. (1) dance came out at 18 frames of 363 -- the render
from before the idle gap was a partial killed mid-write, and postclip/zip ran
on it without complaint. Caught only by counting frames against the source
(ffprobe: src 363, alpha 18, v1 matte 363). Partial deleted, re-render started
20:19 UTC. Frame-count-vs-source is now part of the accept bar for every clip.
(2) 1917 has 71 frames below 0.80 IoU vs v1 (min 0.542) and 145 frames where
only v2 finds subject pixels. Context that cuts the other way: v1's OWN 1917
render has documented full dropouts (f524, f695 contact sheets from 6 Sep), so
low IoU-vs-v1 on this clip partly measures v1's failures, not v2's. Needs
Thriloush's eye on the sheet in 1917.zip before it counts as pass or fail.
Pending his confirmation, not settled.

**20:55 UTC:** dance re-render clean -- 363/363 frames, mean IoU 0.949, min
0.839, 0 below 0.80, bad_frames 0; dance.zip 39981678 B on VM. All 10 clips
now rendered, checked and zipped in /content/rr_final. Download retry with a
real mouse click on the files.download run button ALSO failed -- the cell runs
but Chrome still swallows the download. Cause: files.download fires its anchor
click after the async kernel round-trip, so the user-gesture chain is broken
and Chrome's one-automatic-download-per-site block applies no matter how the
cell was started. Needs a human at the PC: allow automatic downloads for
colab.research.google.com in Chrome site settings (or click Allow on the
blocked-download prompt when the retry runs). Until then the VM is being kept
alive with a trivial cell each heartbeat so the zips survive. Clips 11-18
still blocked on the shrunken Chrome window (378x197) for source uploads.

**21:45-21:50 UTC:** Chrome on the PC became unreachable (closed or machine
slept) right after the 20:55 keep-alive, and the desktop-app link blipped too
(came back). Tried the built-in Claude browser as a fallback -- it is not
signed in to Google, and sign-ins are not something Claude does, so no path
to the notebook from there. Kernel has been idle since ~20:55; if Colab
reclaims the runtime (~90 min idle -> ~22:25 UTC) the 9 undownloaded zips die
with it and clips 2-10 need re-rendering (all check numbers survive here; the
visual sheets inside the zips would be lost). Retrying the link every ~15 min.

## 19 Sep -- fresh VM, full re-run of the 8 lost clips + clips 11-18

VM from 18 Sep was reclaimed overnight with 8 zips undownloaded (only
microsoft.zip and interview.zip, both byte-verified, reached ~/Downloads --
interview flushed at 22:14 when the tab was reopened). Fresh A100 set up
12:35-12:55 UTC: SETUP OK, IMPORTS OK, both checkpoints, code zip (282288 B,
commit 3517aa9), 226 tests pass, defaults verified 64/True. All 32 source and
v1-matte files re-uploaded as 9 MB pieces from the v1 delivery folder and
**md5-verified bit-exact on the VM (ALL_MD5_OK)** -- the leftover wave-4
pieces from the old plan did not match the delivery folder md5s (likely
remuxed copies) and were discarded rather than trusted. postclip.py rewritten
against the repo's real APIs (check_full_clip_solos.py CLI + the same
960-wide IoU/bottom-edge measurement; a FRAMECOUNT guard now runs FIRST and
hard-fails the clip if the alpha frame count differs from the source -- the
lesson from dance). Driver started 12:58 UTC, confirmed running, order:
tryguys leo jensen dance ipman butter 1917 bilibili dance2 dance3 codylexi
eddie shakira asianboss2 dlh es2. ETA ~19:30-20:30 UTC.

Downloads: still one-per-session (dltest: first file landed, second silently
blocked). Needs the one-time Chrome setting: on the Colab tab, allow
automatic downloads (blocked-download icon in the address bar, or Site
settings -> Automatic downloads -> Allow).

## 19 Sep run -- FINAL. All 18 clips rendered on 5.10b defaults and delivered.

Driver ran 12:58-19:59 UTC (7h01m), 16 clips, zero crashes. Every zip
downloaded to ~/Downloads and byte-verified against the VM (automatic
downloads allowed for the Colab site fixed the one-per-session block; batches
of 4 land fine now). microsoft.zip and interview.zip are yesterday's verified
builds of the same code; that completes 18/18.

| clip | frames (alpha/src) | mean/min IoU vs v1 | <0.80 | edge v1/v2 | solos v1/v2 | zip B |
|---|---|---|---|---|---|---|
| tryguys | 250/250 | 0.986/0.969 | 0 | 0.964/0.921 | 0/0 | 27308255 |
| leo | 208/208 | 0.983/0.946 | 0 | 0.318/0.323 | 0/0 | 11188788 |
| jensen | 509/509 | 0.990/0.983 | 0 | 0.252/**0.268** | 0/0 | 22089748 |
| dance | 363/363 | 0.950/0.839 | 0 | 0.033/0.033 | 11/0 | 55459853 |
| ipman | 325 cmp (src 501, offset 160) | 0.960/0.697 | **2** | 0.356/0.353 | 34/6 | 40879595 |
| butter | 382/382 | 0.951/0.815 | 0 | 0.505/0.482 | 13/15 | 47903735 |
| 1917 | 724/724 | 0.911/0.542 | **70** | 0.161/0.159 | 73/145 | 86555253 |
| bilibili | 1372/1372 | 0.984/0.846 | 0 | 0.174/0.171 | 0/0 | 73900433 |
| dance2 | 607/607 | 0.959/0.895 | 0 | 0.015/0.000 (2 edge frames) | 0/0 | 37043407 |
| dance3 | 619/619 | 0.958/0.936 | 0 | 0.054/0.057 | 0/0 | 74514665 |
| codylexi | 490/490 | 0.988/0.977 | 0 | 0.697/0.694 | 0/0 | 42112615 |
| eddie | 143/143 | 0.987/0.984 | 0 | 0.575/0.566 | 0/0 | 6043874 |
| shakira | 369/369 | 0.956/0.865 | 0 | 0.164/0.150 | 0/0 | 48106188 |
| asianboss2 | 354/354 | 0.990/0.987 | 0 | 0.440/0.430 | 0/0 | 32292130 |
| dlh | 660 cmp -- **FLAGGED framecount** | 0.977/0.896 | 0 | 0.094/0.092 | 0/0 | 29197272 |
| es2 | 879/879 | 0.985/0.971 | 0 | 0.353/0.350 | 0/0 | 102164056 |

Three items pending Thriloush's eye, none blocking:
1. **1917, 70 frames <0.80 (min 0.542), 145 frames v2-only.** Same shape as
   the 18 Sep run (71) and as the 5 Sep full-length finding: v1's own 1917
   render drops the subject in blur bursts (f524/f695 contact sheets), so
   IoU-vs-v1 partly scores v1's failures. Sheet in 1917.zip.
2. **ipman, 2 frames <0.80 (min 0.697), blur burst.** Same two frames as the
   18 Sep run. Sheet in ipman.zip.
3. **dlh framecount: source decodes 660 (ffprobe+cv2), pipeline read/emitted
   661** (its strict assert passed -- its reader sees one more frame in the
   same file; 1917's container/decode gap is the same family). Checks ran on
   the 660 comparable pairs, all clean. FLAG file inside dlh.zip.

Bottom-edge parity holds across the set (the 5.10 defect this re-render
exists to fix): edge_v2 within a point or two of edge_v1 everywhere, ahead on
jensen/leo/dance3. The edge numbers are mean bottom-row coverage over frames
where either side touches the bottom row -- comparison numbers, not the old
defect metric, and not directly comparable to the 18 Sep table's edge column
(postclip was rewritten 19 Sep against the repo APIs after the VM loss took
the original with it).
