# run_2026-09-07_rest11 — the last eleven clips, and the whole set at full length

With this run **all eighteen clips have been through the shipping pipeline at
their full delivered length**: 9,640 frames, 56 shots, 6.8 GPU-hours across four
sessions, zero failures and `bad_frames=0` everywhere.

    asianboss2 354f/1shot  codylexi 490f/1  dance  363f/2  dance2 607f/1
    eddie      143f/2      interview 642f/4 jensen 509f/1  leo    208f/1
    microsoft  567f/2      shakira  369f/4  tryguys 250f/4

3h05m of A100 for the eleven, against a 3.2h estimate from measured frame
counts. `--mode alpha` only: these were run to answer an accuracy question and
the matte is what that needs.

## The shot counts are a result nobody could have had before

Every number this project produced before 5 Sep came from a hand-cut
**single-shot** 72-frame window, so the set looked like eighteen single-shot
clips. At full length **nine of the eighteen contain cuts** — and the surprises
are here, not in the clips that were expected to be complex: **interview 4,
shakira 4, tryguys 4, eddie 2, and `microsoft` — the static talking-head
control — has 2.** The pipeline re-seeds at all 56 and drops no subject.

## Thirteen of eighteen are clean, and clean means zero

`bench/check_full_clip_solos.py`, both directions, unchanged thresholds. Ten of
these eleven return **0 frames in each direction**; across the set that is
**13 of 18 at zero over 6,297 frames**, and **not one frame anywhere below 0.80
IoU** against v1.

The one exception here is `dance`: 11 frames, at most 0.0131 of frame, and it is
the **accepted** 3.3g limitation rather than a new defect — the fifth dancer the
subject-height gate drops at 0.549 against 0.60. `dance_sheet.jpg` boxes her.
Full length puts a number on a cost that had only ever been described: eleven
frames on one clip.

## A measurement that looked like a defect and is not

`bench/check_clip_coverage.py` is new here — the blunt companion to solos, added
because solos is blind by construction to a missing region that *touches* the
part being held, which is exactly how bilibili's boots nearly escaped.

It flags five clips with >1% of frame held by v1 and not v2 on many frames:
**interview 369/642, microsoft 113/567, tryguys 69/250, shakira 57/369, dance
41/363.** Read cold that is v2 losing something on more than half of interview.

**It is not, and three checks say so.**

1. **Structure.** The gap is distributed, not bursty — median 0.005–0.011 per
   clip with max/median 2–10x and 14–29% of the total in the worst tenth of
   frames. bilibili's real loss is **17.1x the median** with 35% in the worst
   tenth.
2. **No frame below 0.80 IoU** on any of the eighteen.
3. **The picture.** At the worst frame of each: v1 holding the sofa the four men
   sit on (tryguys), v1 holding the chair and lap (microsoft), v1 holding the
   motion-blur trail behind a gesturing hand (interview), and on shakira v1
   keying flying hair to full opacity where v2 gives it **real partial alpha**
   that a 0.5 threshold then discards.

So: v1 over-including, plus the cost of thresholding a real-alpha matte against
a chroma-keyed one at 0.5. Not a v2 defect, and not reported as one. Worth
recording that the flag fired on five clips and was wrong on all five — a
coverage number above a fixed fraction of frame is not a defect detector.

## The bilibili seed diagnosis (`seed_diag.txt`)

Run on the same VM before the batch. At each frame around the cut, the person
detector's box and the mask model's outline are dumped separately:

| frame | box bottom | mask bottom | short by |
|---|---|---|---|
| 1159 | 894 | **596** | **298 px** |
| 1165 | 982 | 933 | 49 px |
| 1170 | 1069 | 939 | 130 px |
| 1176 | 1065 | 1058 | 7 px (recovered) |

Detector confidence 0.930–0.938 throughout: **it sees the whole person, boots
included.** The box is right; the mask is short. So the box already matches an
instance, and the 5.5 recovery fix — retry the match at a lower score — returns
the same instance with the same truncated mask. **5.5 cannot fix this**, and
that is the useful half of the answer.

## Files

| | |
|---|---|
| `*_alpha.mp4` | the eleven mattes (gitignored; copies in `_to_send/outputs/full_length/`) |
| `side_by_side/` | source \| v1 \| v2 over one background, all eleven |
| `*_solos.json` | per-frame both-ways decomposition |
| `*_cov.json` | per-frame area, IoU and gap — the new coverage sweep |
| `dance_sheet.jpg` | the accepted fifth-dancer miss, boxed |
| `seed_diag.txt`, `seed_diag_f*.jpg` | the bilibili box-vs-mask dump |
| `run_rest.log` | the batch log, including every shot-detection line |

Scoring ran on the Windows device shell (~0.13 s/frame). The cloud container was
tried first and came back **~1.7 s/frame** on the same work — worth knowing
before choosing where to score a long clip.
