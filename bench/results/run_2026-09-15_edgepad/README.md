# 5.10 edge-pad sweep, 15 Sep 2026

Seven clips, three frames each (0/24/48), torchvision Mask R-CNN on CPU,
score>0.80. Clips decoded from their own mp4 in bench_payload -- NOT from the
committed JPEG frames (HANDOFF s7 entry 28: a q100 JPEG moved a Mask R-CNN
boundary by 229 px).

Columns: `short` = median px the binarised seed stops above the bottom edge,
over columns solidly foreground 80-100 px up. `absent` = fraction of those
columns with mask prob < 0.5 at the very bottom row. `edge a` = mean mask prob
there. `w0` = silhouette width at the bottom row. `w90` = width 90 px up (the
control: the body ABOVE the band must not move). `area%` = change in total
mask area vs pad 0. `ndet` = Mask R-CNN instances above 0.80.

| clip | pad | short | absent | edge a | w0 | w90 | w0/w90 | area% | ndet |
|---|---|---|---|---|---|---|---|---|---|
| microsoft | 0 | 17.0 | 1.00 | 0.172 | 0 | 951 | 0.000 | +0.00 | 2.0 |
| microsoft | 16 | 0.0 | 0.21 | 0.562 | 684 | 958 | 0.714 | +2.23 | 2.0 |
| microsoft | 32 | 0.0 | 0.01 | 0.716 | 891 | 955 | 0.933 | +0.77 | 2.0 |
| microsoft | 64 | 0.0 | 0.00 | 0.955 | 917 | 956 | 0.959 | +1.61 | 2.0 |
| ipman | 0 | 18.7 | 1.00 | 0.050 | 0 | 549 | 0.000 | +0.00 | 2.0 |
| ipman | 16 | 1.7 | 0.63 | 0.407 | 182 | 559 | 0.325 | +5.24 | 2.0 |
| ipman | 32 | 0.0 | 0.09 | 0.702 | 448 | 562 | 0.798 | +4.26 | 2.0 |
| ipman | 64 | 0.0 | 0.05 | 0.888 | 502 | 549 | 0.914 | +5.27 | 2.0 |
| interview | 0 | 16.7 | 1.00 | 0.233 | 0 | 1638 | 0.000 | +0.00 | 5.7 |
| interview | 16 | 2.0 | 0.54 | 0.441 | 613 | 1634 | 0.375 | +2.48 | 7.0 |
| interview | 32 | 0.0 | 0.20 | 0.612 | 1063 | 1631 | 0.651 | +2.65 | 6.7 |
| interview | 64 | 0.0 | 0.09 | 0.796 | 1375 | 1652 | 0.833 | +3.88 | 7.7 |
| tryguys | 0 | 9.0 | 1.00 | 0.331 | 0 | 1874 | 0.000 | +0.00 | 3.0 |
| tryguys | 16 | 0.0 | 0.10 | 0.595 | 1462 | 1886 | 0.775 | +1.93 | 3.0 |
| tryguys | 32 | 0.0 | 0.02 | 0.745 | 1747 | 1882 | 0.928 | +2.19 | 3.0 |
| tryguys | 64 | 0.0 | 0.00 | 0.895 | 1809 | 1879 | 0.963 | +1.16 | 3.0 |
| 1917 | 0 | 23.7 | 1.00 | 0.011 | 0 | 789 | 0.000 | +0.00 | 3.3 |
| 1917 | 16 | 3.0 | 0.40 | 0.460 | 414 | 772 | 0.536 | +4.72 | 3.7 |
| 1917 | 32 | 1.3 | 0.35 | 0.595 | 492 | 772 | 0.637 | +6.90 | 3.7 |
| 1917 | 64 | 0.0 | 0.07 | 0.788 | 673 | 795 | 0.847 | +6.49 | 3.3 |
| codylexi | 0 | 13.7 | 1.00 | 0.250 | 0 | 1455 | 0.000 | +0.00 | 2.0 |
| codylexi | 16 | 0.7 | 0.52 | 0.453 | 462 | 1437 | 0.322 | +0.85 | 2.0 |
| codylexi | 32 | 0.0 | 0.21 | 0.550 | 912 | 1439 | 0.634 | +1.24 | 2.0 |
| codylexi | 64 | 0.0 | 0.08 | 0.730 | 1145 | 1458 | 0.785 | +2.26 | 2.0 |
| jensen | 0 | 15.0 | 1.00 | 0.151 | 0 | 744 | 0.000 | +0.00 | 1.0 |
| jensen | 16 | 5.0 | 0.91 | 0.390 | 59 | 743 | 0.079 | +1.29 | 1.0 |
| jensen | 32 | 0.0 | 0.22 | 0.589 | 513 | 746 | 0.688 | +2.00 | 1.0 |
| jensen | 64 | 0.0 | 0.12 | 0.785 | 573 | 746 | 0.769 | +2.24 | 1.0 |
