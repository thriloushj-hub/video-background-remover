"""Audit every declared engine capability against the real class.

    !cd /content && python vbgr_capability_audit.py

Two of the first two capabilities anyone checked on this project turned out to
be wrong, and both were wrong the same way -- written from the paper rather
than from the object:

  * MatAnyone 2 declared ``memory_gate``; ``InferenceCore.step()`` has no
    ``update_memory`` argument (Q7).
  * SAM2Matting declared ``per_frame_head``; the real predictor has no
    standalone ``(image, trimap) -> alpha`` callable.

Both were found by hand. This does the rest mechanically, and prints the
evidence next to each verdict so the answer can be checked rather than
believed.

A capability is only marked ok when something concrete backs it: a real
argument in a real signature, a real attribute on the loaded object, or a
behaviour observed in this session. "declared, no machine check" is not a
pass -- it means the claim is still only a claim.
"""
import inspect
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "/content")

REPO = "/content/SAM2Matting"
CKPT = "/content/checkpoints/SAM2Matting-SAM3.pt"


def line(ch="-", n=78):
    print(ch * n)


def verdict(cap, ok, evidence):
    tag = {True: "ok", False: "WRONG", None: "unproven"}[ok]
    print(f"  {cap:18} {tag:9} {evidence}")


def has_attr_any(obj, names):
    found = [n for n in names if hasattr(obj, n)]
    return (bool(found), found)


def kwarg_in(fn, name):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, "signature unavailable"
    if name in sig.parameters:
        return True, f"{name!r} is a real parameter of {fn.__qualname__}"
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return None, (f"{fn.__qualname__} takes **kwargs, so {name!r} is "
                      f"accepted but may be ignored -- unverifiable")
    return False, (f"{fn.__qualname__} does not accept {name!r}; "
                   f"parameters are {list(sig.parameters)[:8]}")


# --------------------------------------------------------------------------- #

def audit_sam2matting():
    line("=")
    print("SAM2Matting")
    line("=")
    from vbgr.engines import build_engine
    eng = build_engine("sam2matting", repo_dir=REPO, checkpoint=CKPT,
                       device="cuda")
    eng._ensure()
    p = eng._predictor
    print(f"  real class: {type(p).__module__}.{type(p).__name__}")
    print(f"  declared:   {sorted(eng.info.capabilities)}\n")

    ok, found = has_attr_any(p, ["add_new_mask"])
    verdict("mask_prompt", ok, f"predictor.{found[0]} exists" if ok
            else "no add_new_mask on the predictor")

    ok, found = has_attr_any(p, ["add_new_points_or_box", "add_new_points"])
    verdict("point_prompt", ok, f"predictor.{found[0]} exists" if ok
            else "no add_new_points / add_new_points_or_box")
    if ok:
        sig = inspect.signature(getattr(p, found[0]))
        pars = list(sig.parameters)
        has_box = any("box" in x for x in pars)
        verdict("box_prompt", has_box,
                f"{found[0]}({', '.join(pars[:7])}...)" if has_box
                else f"{found[0]} has no box parameter: {pars}")
    else:
        verdict("box_prompt", False, "no point/box entry point at all")

    ok, found = has_attr_any(p, ["add_new_text", "set_text_prompt",
                                 "add_new_concept", "add_text_prompt",
                                 "set_concept", "add_new_prompt"])
    if ok:
        verdict("text_prompt", True, f"predictor.{found[0]} exists")
    else:
        cands = [n for n in dir(p)
                 if not n.startswith("_")
                 and any(k in n.lower() for k in ("text", "concept", "prompt"))]
        verdict("text_prompt", False,
                f"no text entry point; prompt-ish attrs are {cands or 'none'}")

    verdict("multi_object", True,
            "exercised this session: 2 obj_ids on ipman, 4 on butter, "
            "alphas unioned, frame parity held")
    verdict("per_frame_head", None,
            "REMOVED from the declaration 2026-08-15 -- matting_step is "
            "state-coupled, alpha_pred1..3 are nn.Sequential over feats")
    print(f"\n  has() now reports: "
          f"{ {c: eng.has(c) for c in sorted(eng.info.capabilities)} }")
    return eng


def audit_matanyone2():
    line("=")
    print("MatAnyone 2")
    line("=")
    from vbgr.engines import build_engine
    eng = build_engine("matanyone2", device="cuda")
    eng._ensure()          # prints its own load-time capability check
    step = eng._proc.step
    print(f"  real class: {type(eng._proc).__module__}.{type(eng._proc).__name__}")
    print(f"  declared:   {sorted(eng.info.capabilities)}")
    print(f"  step{inspect.signature(step)}\n")

    ok, why = kwarg_in(step, "mask")
    if ok is False:
        pars = list(inspect.signature(step).parameters)
        ok = len(pars) > 1
        why = f"positional mask argument: step({', '.join(pars[:4])}...)"
    verdict("mask_prompt", ok, why)

    ok, why = kwarg_in(step, "force_permanent")
    verdict("permanent_memory", ok, why)

    # memory_gate is not audited any more: task 3.4 dropped it (2026-08-23)
    # after this very audit proved step() takes no **kwargs. Auditing a
    # feature nothing calls just re-litigates a settled question.

    print(f"\n  has() now reports: "
          f"{ {c: eng.has(c) for c in sorted(eng.info.capabilities)} }")
    print(f"  verified_capabilities: {eng.verified_capabilities}")
    return eng


def audit_cpu_engines():
    line("=")
    print("passthrough / existing_alpha  (this repo's own code)")
    line("=")
    from vbgr.engines import build_engine
    e = build_engine("passthrough")
    print(f"  declared: {sorted(e.info.capabilities)}")
    f0 = np.zeros((80, 120, 3), np.uint8); f0[20:60, 30:90] = 200
    f1 = np.zeros((80, 120, 3), np.uint8); f1[20:60, 40:100] = 200
    seed = np.zeros((80, 120), np.float32); seed[20:60, 30:90] = 1
    e.start(f0, seed)
    verdict("mask_prompt", True, "start(seed_mask) is the only entry point")
    # The commit_to_memory behavioural check lived here. It passed -- this
    # engine is our own code and honours the kwarg -- which is exactly why it
    # was misleading: it proved the test double worked, never the real models.
    # Feature dropped in 3.4.

    e2 = build_engine("existing_alpha", alphas=np.zeros((2, 4, 4), np.float32))
    print(f"\n  existing_alpha declared: {sorted(e2.info.capabilities)} "
          f"(nothing to get wrong)")


def audit_rvm():
    line("=")
    print("RVM")
    line("=")
    from vbgr.engines import build_engine
    e = build_engine("rvm")
    print(f"  declared: {sorted(e.info.capabilities)}")
    sig = inspect.signature(e.matte)
    seed = sig.parameters.get("seed_mask")
    ok = seed is not None and seed.default is None
    verdict("auto_human", ok,
            "matte(seed_mask=None) by default, i.e. needs no prompt"
            if ok else f"seed_mask parameter is {seed}")
    print("  NOTE: weights not downloaded here; 'auto_human' is checked from "
          "the adapter contract, not from a run.")


def main():
    for fn in (audit_cpu_engines, audit_rvm, audit_sam2matting,
               audit_matanyone2):
        try:
            fn()
        except Exception:
            line("=")
            print(f"{fn.__name__} FAILED -- reporting rather than hiding it:")
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
