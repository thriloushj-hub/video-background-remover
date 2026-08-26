# ipman's output in this run is on a window that no longer exists

The 23 Aug run scored `ipman_w61_133.mp4` — source frames 61–133, which lies
entirely outside v1's matte coverage. That clip was replaced on 26 Aug by
`ipman_w318_389.mp4` and the v1 baseline was restored on the new window.

So this run's ipman alpha and contact sheet describe **different footage** from
the baseline now in `v1_window_baselines.json`. Scoring them together would
reproduce exactly the defect that was just fixed, and every metric would still
return a plausible number.

The folder is therefore renamed `ipman.STALE-WINDOW/`, which makes it invisible
to `bench/score_disagreement.py` (it looks for a directory named after the
clip) and says why to anyone who opens it.

Delete it, or leave it as the record. Do not rename it back. ipman gets real
numbers again from the next GPU run, which will use the new clip.

See `Ipman_Window_Void.md` in the vault.
