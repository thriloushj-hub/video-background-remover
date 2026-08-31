"""Candidate discriminator for gate-dropped detections.

Question: can we separate a further-back CAST member (dance's fifth dancer,
which we should keep) from a background BYSTANDER (leo's man, 1917's soldiers,
which we correctly drop)?  Height ratio cannot: the right answer is not
monotonic in it (1917 0.634 drop, dance 0.549 keep, leo 0.473 drop).

Signals tested, all computed from the clip and the detector boxes only:
  persistence  - fraction of scan frames the candidate appears on
  motion       - mean frame-to-frame absdiff inside the box, per pixel
  motion_ratio - that, over the median of the KEPT subjects on the same frame
  depth_gap    - how much higher the candidate's feet sit than the kept median,
                 as a fraction of frame height (further back = higher feet)
"""
import json, os, glob, collections
import cv2, numpy as np
from pkg.detect import Detection, select_person_boxes

SCAN = list(range(0, 72, 8))


def frames(path):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release(); return out


def motion(FR, i, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = FR[0].shape
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    a = FR[max(0, i-2)][y1:y2, x1:x2].astype(np.float32)
    b = FR[min(len(FR)-1, i+2)][y1:y2, x1:x2].astype(np.float32)
    return float(np.abs(a-b).mean())


def link(tracks, box, thr=0.35):
    """Greedy link by centre distance relative to box size."""
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    h = box[3]-box[1]
    for t in tracks:
        px, py = t['c']
        if abs(cx-px) < thr*h*1.5 and abs(cy-py) < thr*h:
            return t
    return None


def main():
    r = json.load(open('sweep.json'))
    byframe = collections.defaultdict(list)
    for d in r: byframe[(d['clip'], d['f'])].append(d)
    DIM = {}
    for p in glob.glob('win/*.mp4'):
        c = os.path.basename(p).split('_w')[0]
        cap = cv2.VideoCapture(p); ok, f = cap.read(); cap.release()
        DIM[c] = (f.shape[1], f.shape[0], p)

    print(f"{'clip':<11}{'frames':>7}{'persist':>9}{'motion':>8}{'m_ratio':>9}"
          f"{'depth':>8}  box@mid")
    out = {}
    for clip in sorted(DIM):
        W, H, path = DIM[clip]
        FR = frames(path)
        tracks = []
        for i in SCAN:
            ds = byframe.get((clip, i), [])
            if not ds: continue
            dets = [Detection(tuple(d['box']), d['conf'], 0, 'person') for d in ds]
            kept, drop = select_person_boxes(dets, W, H, 0.60, 0.15, 0.20, 0.50)
            if not kept: continue
            kmot = np.median([motion(FR, i, d.box) for d in kept]) or 1e-6
            kfeet = np.median([d.box[3] for d in kept])
            for d in drop:
                if d.conf < 0.80: continue
                t = link(tracks, d.box)
                if t is None:
                    t = {'clip': clip, 'c': None, 'n': 0, 'mot': [], 'rat': [],
                         'dep': [], 'box': d.box, 'conf': []}
                    tracks.append(t)
                m = motion(FR, i, d.box)
                t['c'] = ((d.box[0]+d.box[2])/2, (d.box[1]+d.box[3])/2)
                t['n'] += 1
                t['mot'].append(m); t['rat'].append(m/kmot); t['conf'].append(d.conf)
                t['dep'].append((kfeet - d.box[3]) / H)
                if t['n'] == 2: t['box'] = d.box
        for t in tracks:
            if t['n'] < 2: continue
            print(f"{clip:<11}{t['n']:>7}{t['n']/len(SCAN):>9.2f}"
                  f"{np.mean(t['mot']):>8.2f}{np.mean(t['rat']):>9.2f}"
                  f"{np.mean(t['dep']):>8.3f}  {t['box']}")
            out.setdefault(clip, []).append(
                {'n': t['n'], 'persist': round(t['n']/len(SCAN), 2),
                 'motion': round(float(np.mean(t['mot'])), 2),
                 'motion_ratio': round(float(np.mean(t['rat'])), 2),
                 'depth_gap': round(float(np.mean(t['dep'])), 3),
                 'box': [int(v) for v in t['box']]})
    json.dump(out, open('persist.json', 'w'), indent=1)


if __name__ == '__main__':
    main()


# =====================================================================
# Second prototype: motion synchrony with the cast, with and without
# global camera-motion compensation.  See Subject_Selection_Signals.md.
# =====================================================================

"""Same synchrony test, with global (camera) motion removed first.

The uncompensated version scored 1917's background soldier at 0.871 -- but 1917
is a tracking shot, so every box moves together and the correlation is measuring
the camera, not the people.  Estimating the global shift per frame pair with
phase correlation and warping before differencing should leave only what the
PEOPLE did.
"""
import json, glob, collections
import cv2, numpy as np
from pkg.detect import Detection, select_person_boxes

STEP = 2
CASES = {
    'dance': ((1042, 483, 1160, 867), 'KEEP  fifth dancer'),
    'leo':   ((1822, 286, 1919, 630), 'drop  bystander'),
    '1917':  ((1558, 166, 1811, 710), 'drop  background soldier'),
    'butter':((1030, 198, 1157, 593), 'KEEP  entrant dancer'),
}


def load(path):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(cv2.resize(g, (g.shape[1]//2, g.shape[0]//2)))
    cap.release(); return out


def warp_to(a, b):
    """Shift b onto a using the global translation between them."""
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(b, M, (b.shape[1], b.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (dx, dy)


def energy(prev_w, cur, box, s=0.5):
    x1, y1, x2, y2 = [int(v*s) for v in box]
    H, W = cur.shape
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    return float(np.abs(cur[y1:y2, x1:x2] - prev_w[y1:y2, x1:x2]).mean())


def z(a):
    s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def main():
    r = json.load(open('sweep.json'))
    byframe = collections.defaultdict(list)
    for d in r: byframe[(d['clip'], d['f'])].append(d)
    print(f"{'clip':<8}{'case':<26}{'raw corr':>10}{'camera-comp':>13}{'cam shift px':>14}")
    res = {}
    for clip, (cbox, label) in CASES.items():
        path = glob.glob(f'win/{clip}_w*.mp4')[0]
        FR = load(path)
        W, H = FR[0].shape[1]*2, FR[0].shape[0]*2
        ds = byframe[(clip, 0)]
        dets = [Detection(tuple(d['box']), d['conf'], 0, 'person') for d in ds]
        kept, _ = select_person_boxes(dets, W, H, 0.60, 0.15, 0.20, 0.50)
        idx = list(range(STEP, len(FR), STEP))
        cand_r, cand_c, shifts = [], [], []
        ks_r = [[] for _ in kept]; ks_c = [[] for _ in kept]
        for i in idx:
            cur, prev = FR[i], FR[i-STEP]
            pw, (dx, dy) = warp_to(cur, prev)
            shifts.append((dx*dx+dy*dy)**0.5 * 2)
            cand_r.append(energy(prev, cur, cbox)); cand_c.append(energy(pw, cur, cbox))
            for j, k in enumerate(kept):
                ks_r[j].append(energy(prev, cur, k.box))
                ks_c[j].append(energy(pw, cur, k.box))
        for tag, cand, ks in (('raw', cand_r, ks_r), ('cc', cand_c, ks_c)):
            cast = np.mean(np.array(ks), axis=0)
            res[(clip, tag)] = float((z(np.array(cand)) * z(cast)).mean())
        print(f"{clip:<8}{label:<26}{res[(clip,'raw')]:>10.3f}"
              f"{res[(clip,'cc')]:>13.3f}{np.mean(shifts):>14.2f}")


if __name__ == '__main__':
    main()
