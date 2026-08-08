# Licensing — read this before you spend GPU hours

You said the commercial question is undecided. It is the single decision that
most changes what gets built, so here is the state of play.

## The engines

| Engine | Licence | Commercial use | Quality |
|---|---|---|---|
| **SAM2Matting** (ECCV 2026, Fudan) | CC BY-NC-SA 4.0 | **No** | Best tracking + matting decoupling; targets rapid motion |
| **MatAnyone 2** (CVPR 2026, NTU S-Lab) | NTU S-Lab License 1.0 | **No** | Best human hair/fabric edges |
| **RVM** (WACV 2022) | GPL-3.0 | Yes, under copyleft | Weak edges, no subject control |
| SAM 3 (Meta) | check the release terms for the specific weights you use | varies | segmentation only |
| YOLOv8 (Ultralytics) | AGPL-3.0, or a paid commercial licence | AGPL or paid | detection only |

Two things people commonly get wrong here:

* **CC BY-NC-SA 4.0 is not "non-commercial for the model only".** The
  ShareAlike term also means derivative works carry the same licence. You
  cannot fine-tune SAM2Matting on your own data and ship the result.
* **YOLOv8 is AGPL-3.0.** Even in the research path this is the component most
  likely to surprise you, because AGPL reaches network use. Ultralytics sells a
  commercial licence; alternatively swap the detector — the pipeline only needs
  person boxes, so RT-DETR (Apache-2.0) or a torchvision Faster R-CNN
  (BSD-3-Clause) drop in with a ~20-line change to `vbgr/detect.py`.

## What the code does about it

Every engine declares `commercial_ok`. Pass `--require-commercial` and the
pipeline refuses to construct a non-commercial engine rather than letting you
find out at ship time:

```bash
vbgr2 run -i clips/ --require-commercial
# PermissionError: engine 'SAM2Matting' is licensed 'CC BY-NC-SA 4.0'
# (non-commercial) but require_commercial=True.
```

`vbgr2 engines` prints the whole table.

## The three realistic paths

**1. Research / internal only.** Use `sam2matting` (SAM3 backbone). Nothing to
do. This is what `configs/default.yaml` is set to.

**2. Ship commercially, buy your way out.** Contact Henghui Ding
(henghui.ding@gmail.com, named in the SAM2Matting README) and NTU S-Lab about
commercial terms, and buy the Ultralytics licence. Fastest path to shipping the
quality you actually want. Cost unknown until you ask.

**3. Ship commercially, build your own.** The architecture in this repo is the
asset, not the weights. Everything outside `vbgr/engines/` is original and
MIT-licensed: the flow-adaptive trimap, the ReID re-prompting layer, the memory
gate, decontamination, the harness. To go fully commercial you would need to
train a matting head yourself on a permissively-licensed matting dataset and
pair it with a permissively-licensed tracker. That is a multi-month project,
not an afternoon — but it is tractable *because* the surrounding system already
exists and is measurable.

## My read

Decide before step 3 of the plan, not after. If the answer turns out to be
"commercial", most of the tuning work you would do on SAM2Matting is wasted —
but none of the harness, ReID, motion or decontamination work is, because those
sit outside the engine boundary by design. That is the main reason the engine
interface exists.

If it helps: option 2 is usually much cheaper than teams expect, and option 3
is usually much more expensive than they expect.
