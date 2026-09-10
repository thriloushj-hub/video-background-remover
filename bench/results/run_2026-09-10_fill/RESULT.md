# 5.7b is WRONG: ranking by box fill does not fix the trailing boot (10 Sep)

Fresh A100, code at `0b154a7` (`193 passed` on the VM first, bilibili's md5
verified byte-identical to the delivery after reassembly). Sweep on bilibili
alone, `--k 4`, **38 s**.

## What was expected

`seed_tail_fraction` measures only how far DOWN a mask reaches, so a mask
holding a subject's leading boot and not the trailing one reaches the floor and
scores perfect. The prediction was that box FILL would see the missing boot and
pick a different frame.

## What the GPU said

Shot 11, frames 1159–1174, the shot whose trailing boot comes back at ~0.4:

    frame index      0        1        2        3
    tail          0.3326   0.0805   0.0707   0.0536
    fill          0.3325   0.3759   0.3908   0.4085

**Fill ranks the SAME frame best — index 3 — and by a margin of 0.076**, which
does not clear `lookahead_min_gain` (0.10). Ranked by fill, the whole clip
reports **moved 0** across its 13 shots, so shot 11 would go back to being
seeded from frame 0 and the boots would be missing *entirely* rather than soft.
The change was a regression, and it is reverted.

## What that means, and it is the useful half

A missing boot is a **small share of a standing person's bounding box**. Fill
moves by 0.076 across four frames where the tail moves by 0.279, so fill is the
blunter signal here, not the sharper one. The diagnosis was right — the seed is
deficient on the frame that gets chosen — but the remedy was wrong: the problem
is not *which frame is picked*, it is that **no frame in the look-ahead window
has a seed that holds both boots**. Frame 3 is merely the least deficient.

`seed_box_fill` / `worst_seed_fill` are kept, because fill is worth MEASURING —
the sweep now records it beside the tail on every shot, and that is what made
this answer cheap. It is just not what to steer on.

## Where to go next, in the order they are worth trying

1. **Retry the failing box at a lower mask threshold**, which is what 5.5 did
   for 1917: `masks_from_boxes` skips a box no instance matched, and here the
   instance matches but comes back truncated. Closest analogue in this codebase
   and the only one already proven on it.
2. **Widen the window.** Cheap to test — the sweep takes 38 s per clip — but the
   tails across the whole shot stay in the 0.05–0.12 band, so a complete frame
   may simply not exist inside this shot.
3. **Union the candidate seeds.** Attractive on paper and risky in practice: the
   subject is walking, so a union over three frames smears her by a stride.
