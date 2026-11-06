# Benchmarks

`kernels` (our CUDA kernels) against `portable` (plain PyTorch, each model's tested precision) and the original authors'
code. One RTX 5060 Ti 16 GB, torch 2.14.1+cu130, CUDA 12.8 kernels (13.0 for four stereo families), cuDNN 9.24. Stereo and depth rows use real images
(KITTI 2015 `000000_10`, Middlebury `Motorcycle`) as uint8 CUDA tensors with default options; other tasks use a synthetic
input at the task's default size. Speeds are medians of warm runs (5 for most rows; each row's protocol is in `docs/data/speed.jsonl`); peak memory is torch's peak allocated / reserved GPU MiB (CUDA, weights included) over the first and the timed calls. `fastforks bench MODEL --size HxW` measures your GPU (it uses 10 runs).

Sizes are written height x width everywhere (`375x1242`, as in `fastforks bench --size`).

To choose a model, use the [overview](overview.md): one standard size per task, generated from the same data. The tables
below give every measured input, the startup times and, at the end, the same models at fixed sizes up to 12 MP.

Accuracy matches the original code. On ETH3D and Middlebury (the "Measured" tables on the model pages) both modes
stay within 0.15 bad-% / 0.025 EPE of the original with the same checkpoint for AANet, AANet+, CoEx, IGEV,
Selective-IGEV, LiteAnyStereo V2 M/H, DEFOM and FoundationStereo-L. S2M2 is better (up to 1 bad-% lower) because our
port uses the cost-volume lookup convention the released weights were trained with. CREStereo is 0.3 bad-% better on
Middlebury; the original run padded the input differently.

The accuracy columns on the model pages check parity (original code vs portable vs kernels) on public images; they are not a
ranking. Variants trained or fine-tuned on ETH3D / Middlebury are marked `†` in those columns (`?` where the training data
is not stated); to rank models use the official Middlebury and ETH3D test-set leaderboards.

<!-- tables:begin (tools/gen_docs.py, from docs/data/speed.jsonl) -->

### Stereo, KITTI 2015, 375x1242

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 65.7 | 55.4 | 55.9 | 32.6 | 1.7x | 1.7x | 1139 / 1210 | 469 / 518 | 287 / 380 |
| aanet-sceneflow | 75.4 | 47.9 | 45.9 | 30.6 | 1.6x | 1.5x | 363 / 428 | 310 / 366 | 208 / 264 |
| bridgedepth-rvc-pretrain | 169 | 121 | 114 | 94.7 | 1.3x | 1.2x | 1891 / 2006 | 1910 / 2006 | 1323 / 1446 |
| bridgedepth-sf | 164 | 118 | 111 | 92.1 | 1.3x | 1.2x | 1840 / 1970 | 1859 / 1954 | 1272 / 1394 |
| c-fast-foundationstereo | 203 | 143 | 97.7 | 50.9 | 2.8x | 1.9x | 1006 / 1138 | 733 / 914 | 438 / 554 |
| c-foundationstereo-s | 2499 | 1121 | 1074 | 431 | 2.6x | 2.5x | 2781 / 3204 | 3085 / 3450 | 1785 / 2036 |
| cas-gwcnet-c-sceneflow | 195 | 115 | 96.7 | 65.8 | 1.8x | 1.5x | 917 / 990 | 668 / 834 | 581 / 670 |
| cas-gwcnet-gc-sceneflow | 250 | 161 | 138 | 103 | 1.6x | 1.3x | 2634 / 2772 | 772 / 954 | 751 / 834 |
| cfnet-sceneflow | 194 | 155 | 112 | 89.8 | 1.7x | 1.3x | 2211 / 2584 | 1622 / 2016 | 1692 / 2116 |
| coex-sceneflow | 26.0 | 16.6 | 14.8 | 6.60 | 2.5x | 2.2x | 671 / 698 | 387 / 434 | 163 / 486 |
| crestereo-eth3d | 636 | 480 | 209 | 83.1 | 5.8x | 2.5x | 516 / 642 | 315 / 370 | 245 / 292 |
| defom-stereo-vitl-sceneflow | 947 | 509 | 404 | 184 | 2.8x | 2.2x | 2458 / 2610 | 2394 / 2538 | 1604 / 1846 |
| dlnr-middlebury | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| dlnr-sceneflow | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| fast-acvnet-generalization | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-kitti12 | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-kitti15 | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-plus-generalization | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-kitti12 | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-kitti15 | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-sceneflow | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-sceneflow | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-foundationstereo | 203 | 142 | 97.8 | 51.0 | 2.8x | 1.9x | 1052 / 1108 | 733 / 774 | 438 / 554 |
| foundationstereo-vit-l | 2801 | 1272 | 905 | 491 | 2.6x | 1.8x | 3614 / 3754 | 2154 / 2294 | 1588 / 1658 |
| foundationstereo-vit-s | 2139 | 1052 | 737 | 358 | 2.9x | 2.1x | 2388 / 2484 | 1542 / 1744 | 977 / 1090 |
| great-igev-eth3d | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-kitti | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-middlebury | 1055 | 705 | 492 | 175 | 4.0x | 2.8x | 10728 / 10974 | 1165 / 1298 | 1093 / 1184 |
| great-igev-sceneflow | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| gwcnet-g-sceneflow | 175 | 104 | 104 | 72.7 | 1.4x | 1.4x | 1622 / 1734 | 686 / 882 | 598 / 782 |
| gwcnet-gc-sceneflow | 183 | 107 | 107 | 78.3 | 1.4x | 1.4x | 1867 / 1976 | 687 / 982 | 567 / 642 |
| hitnet-eth3d | 63.1 | 46.0 | 22.4 | 7.41 | 6.2x | 3.0x | 746 / – | 341 / 510 | 208 / 728 |
| hitnet-middlebury-d400 | 177 | 130 | 52.6 | 17.0 | 7.7x | 3.1x | 2124 / – | 698 / 1074 | 425 / 1428 |
| igev-plusplus-eth3d | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-kitti12 | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-kitti15 | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-middlebury | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-sceneflow | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-stereo-eth3d | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti12 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti15 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-middlebury | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-sceneflow | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| las2-h | 51.1 | 36.4 | 37.5 | 14.9 | 2.4x | 2.5x | 352 / 386 | 387 / 430 | 311 / 528 |
| las2-l | 46.9 | 30.1 | 30.9 | 8.96 | 3.4x | 3.4x | 422 / 488 | 424 / 458 | 169 / 380 |
| las2-m | 23.9 | 20.0 | 23.2 | 5.28 | 3.8x | 4.4x | 360 / 404 | 360 / 386 | 136 / 332 |
| las2-s | 19.3 | 17.3 | 20.5 | 3.90 | 4.4x | 5.3x | 345 / 382 | 343 / 368 | 128 / 338 |
| mocha-v1-sceneflow | 648 | 451 | 321 | 165 | 2.7x | 1.9x | 1430 / 1608 | 668 / 750 | 657 / 750 |
| mocha-v2-sceneflow | 615 | 398 | 315 | 164 | 2.4x | 1.9x | 1411 / 1588 | 668 / 750 | 656 / 750 |
| monster-mix-all | 993 | 995 | 419 | 187 | 5.3x | 2.2x | 2733 / 2958 | 1262 / 1594 | 1282 / 1634 |
| monster-plusplus-mix-all | 1012 | 1009 | 432 | 195 | 5.2x | 2.2x | 2810 / 3018 | 1339 / 1618 | 1308 / 1660 |
| monster-sceneflow | 993 | 995 | 419 | 187 | 5.3x | 2.2x | 2733 / 2958 | 1262 / 1594 | 1282 / 1634 |
| nmrf-sceneflow | 122 | 78.1 | 58.1 | 51.5 | 1.5x | 1.1x | 587 / 634 | 548 / 586 | 364 / 406 |
| psmnet-sceneflow | 248 | 204 | 107 | 73.5 | 2.8x | 1.5x | 1950 / 1990 | 700 / 822 | 681 / 862 |
| raft-stereo-eth3d | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-middlebury | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-sceneflow | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| rt-monster-plusplus-mix-all | 91.8 | 91.3 | 37.4 | 31.4 | 2.9x | 1.2x | 1051 / 1106 | 892 / 926 | 899 / 1116 |
| s2m2-l | 597 | 283 | 288 | 120 | 2.4x | 2.4x | 1747 / 1998 | 1510 / 1718 | 831 / 988 |
| s2m2-m | 356 | 184 | 184 | 78.2 | 2.4x | 2.4x | 1080 / 1284 | 952 / 1126 | 491 / 542 |
| s2m2-s | 163 | 99.1 | 91.5 | 38.3 | 2.6x | 2.4x | 676 / 748 | 633 / 750 | 287 / 400 |
| s2m2-xl | 1063 | 529 | 536 | 212 | 2.5x | 2.5x | 3315 / 3484 | 2722 / 2976 | 1502 / 1614 |
| selective-igev-eth3d-finetune | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-eth3d-train | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-kitti12 | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-kitti15 | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-middlebury-finetune | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-middlebury-train | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-sceneflow | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| waft-stereo-b | 380 | 208 | 208 | 93.6 | 2.2x | 2.2x | 1674 / 1860 | 1610 / 1732 | 851 / 936 |
| waft-stereo-s | 234 | 128 | 129 | 65.2 | 2.0x | 2.0x | 1326 / 1480 | 1281 / 1356 | 489 / 566 |

### Stereo, Middlebury 2014 half, 994x1482

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 263 | 194 | 159 | 88.9 | 2.2x | 1.8x | 3652 / 3784 | 1142 / 1278 | 813 / 920 |
| aanet-sceneflow | 289 | 178 | 140 | 101 | 1.8x | 1.4x | 1000 / 1126 | 845 / 986 | 536 / 644 |
| bridgedepth-rvc-pretrain | 629 | 470 | 442 | 354 | 1.3x | 1.2x | 2903 / 3200 | 2968 / 3246 | 2361 / 2686 |
| bridgedepth-sf | 616 | 460 | 437 | 348 | 1.3x | 1.3x | 2852 / 3144 | 2916 / 3214 | 2310 / 2634 |
| c-fast-foundationstereo | 741 | 527 | 330 | 135 | 3.9x | 2.4x | 2892 / 2996 | 2109 / 2336 | 1196 / 1516 |
| c-foundationstereo-s | – | 3874 | 3646 | 1362 | 2.8x | 2.7x | 8338 / 12870 | 6312 / 6590 | 3107 / 4376 |
| cas-gwcnet-c-sceneflow | 656 | 389 | 327 | 225 | 1.7x | 1.5x | 2791 / 2970 | 2006 / 2474 | 1725 / 1930 |
| cas-gwcnet-gc-sceneflow | 862 | 567 | 472 | 358 | 1.6x | 1.3x | 8317 / 8730 | 2338 / 2794 | 2272 / 2514 |
| cfnet-sceneflow | 659 | 545 | 484 | 401 | 1.4x | 1.2x | 6765 / 7880 | 3441 / 4376 | 3662 / 4718 |
| coex-sceneflow | 94.9 | 60.0 | 49.9 | 23.5 | 2.6x | 2.1x | 2098 / 2160 | 1136 / 1254 | 495 / 886 |
| crestereo-eth3d | 1980 | 1597 | 732 | 229 | 7.0x | 3.2x | 1541 / 1752 | 884 / 1030 | 644 / 732 |
| defom-stereo-vitl-sceneflow | 4175 | 2816 | 1531 | 659 | 4.3x | 2.3x | 8639 / 8766 | 4362 / 4838 | 3154 / 3566 |
| dlnr-middlebury | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| dlnr-sceneflow | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| fast-acvnet-generalization | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-kitti12 | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-kitti15 | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-plus-generalization | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-kitti12 | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-kitti15 | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-sceneflow | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-sceneflow | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-foundationstereo | 741 | 527 | 330 | 135 | 3.9x | 2.4x | 2939 / 3306 | 2109 / 2256 | 1196 / 1576 |
| foundationstereo-vit-l | 9228 | 4237 | 2990 | 1513 | 2.8x | 2.0x | 8286 / 8750 | 3141 / 3436 | 2818 / 3300 |
| foundationstereo-vit-s | 7129 | 3609 | 2514 | 1120 | 3.2x | 2.2x | 6987 / 7280 | 2493 / 2606 | 2169 / 2470 |
| great-igev-eth3d | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-kitti | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-middlebury | – | – | 1785 | 579 | – | 3.1x | – | 1875 / 2078 | 1553 / 1744 |
| great-igev-sceneflow | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| gwcnet-g-sceneflow | 604 | 352 | 344 | 238 | 1.5x | 1.4x | 5105 / 5414 | 2081 / 2402 | 1799 / 2102 |
| gwcnet-gc-sceneflow | 632 | 365 | 353 | 257 | 1.4x | 1.4x | 5887 / 6214 | 2082 / 2402 | 1768 / 1962 |
| hitnet-eth3d | 234 | 182 | 67.8 | 23.2 | 7.9x | 2.9x | 2433 / – | 860 / 1090 | 647 / 1248 |
| hitnet-middlebury-d400 | 637 | 483 | 176 | 52.7 | 9.2x | 3.3x | 6831 / – | 1744 / 2414 | 1263 / 2368 |
| igev-plusplus-eth3d | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-kitti12 | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-kitti15 | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-middlebury | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-sceneflow | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-stereo-eth3d | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti12 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti15 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-middlebury | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-sceneflow | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| las2-h | 205 | 135 | 123 | 53.3 | 2.5x | 2.3x | 1022 / 1084 | 1050 / 1110 | 1021 / 1388 |
| las2-l | 154 | 90.8 | 80.3 | 29.9 | 3.0x | 2.7x | 1010 / 1082 | 1036 / 1098 | 397 / 600 |
| las2-m | 79.1 | 52.4 | 54.1 | 18.3 | 2.9x | 3.0x | 948 / 1000 | 972 / 1026 | 329 / 572 |
| las2-s | 62.0 | 43.5 | 46.9 | 14.3 | 3.0x | 3.3x | 933 / 978 | 955 / 988 | 320 / 558 |
| mocha-v1-sceneflow | 2148 | 1456 | 1057 | 455 | 3.2x | 2.3x | 3637 / 3982 | 1632 / 1770 | 1588 / 1730 |
| mocha-v2-sceneflow | 2054 | 1295 | 1033 | 448 | 2.9x | 2.3x | 3576 / 3982 | 1632 / 1770 | 1587 / 1790 |
| monster-mix-all | 4275 | 4277 | 1585 | 668 | 6.4x | 2.4x | 6382 / 6430 | 2336 / 2494 | 2356 / 2714 |
| monster-plusplus-mix-all | 4339 | 4334 | 1633 | 698 | 6.2x | 2.3x | 6382 / 6430 | 2583 / 2918 | 2438 / 2760 |
| monster-sceneflow | 4275 | 4277 | 1585 | 668 | 6.4x | 2.4x | 6382 / 6430 | 2336 / 2494 | 2356 / 2714 |
| nmrf-sceneflow | 445 | 313 | 216 | 196 | 1.6x | 1.1x | 1656 / 2054 | 1510 / 1552 | 1022 / 1112 |
| psmnet-sceneflow | 760 | 608 | 340 | 232 | 2.6x | 1.5x | 6015 / 6090 | 2078 / 2442 | 1786 / 2122 |
| raft-stereo-eth3d | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-middlebury | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-sceneflow | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| rt-monster-plusplus-mix-all | 252 | 251 | 113 | 94.1 | 2.7x | 1.2x | 2922 / 3044 | 2592 / 2646 | 2704 / 3976 |
| s2m2-l | 1934 | 1005 | 1015 | 376 | 2.7x | 2.7x | 3248 / 4090 | 3245 / 3838 | 1249 / 1568 |
| s2m2-m | 1242 | 666 | 670 | 248 | 2.7x | 2.7x | 2371 / 2920 | 2300 / 2786 | 876 / 1162 |
| s2m2-s | 582 | 350 | 348 | 113 | 3.1x | 3.1x | 1759 / 1986 | 1733 / 1930 | 686 / 820 |
| s2m2-xl | 3510 | 1796 | 1815 | 682 | 2.6x | 2.7x | 5606 / 6080 | 5228 / 6016 | 2092 / 2274 |
| selective-igev-eth3d-finetune | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-eth3d-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti12 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti15 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-finetune | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-sceneflow | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| waft-stereo-b | 1898 | 1140 | 1137 | 393 | 2.9x | 2.9x | 5743 / 5876 | 5775 / 5852 | 1907 / 2316 |
| waft-stereo-s | 1252 | 689 | 689 | 284 | 2.4x | 2.4x | 3314 / 3398 | 3293 / 3456 | 1113 / 1286 |

### Depth, synthetic, 518x770

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| da3-giant-1.1 | 406 | 224 | 81.5 | 44.2 | 5.1x | 1.8x | 8590 / 8718 | 2621 / 4608 | 2527 / 2710 |
| da3-large-1.1 | 205 | 163 | 36.9 | 16.7 | 9.7x | 2.2x | 3374 / 3472 | 975 / 1350 | 883 / 1090 |
| da3metric-large | 125 | 82.7 | 23.7 | 14.3 | 5.8x | 1.7x | 2996 / 3050 | 798 / 1328 | 794 / 868 |
| da3mono-large | 125 | 82.7 | 23.5 | 14.2 | 5.8x | 1.7x | 2997 / 3052 | 801 / 1328 | 797 / 868 |
| da3nested-giant-large-1.1 | 529 | 307 | 105 | 58.3 | 5.3x | 1.8x | 10332 / 10402 | 3259 / 5914 | 3171 / 3378 |
| depth-anything-v2-b | 88.8 | 53.8 | 18.8 | 10.6 | 5.1x | 1.8x | 928 / 1026 | 433 / 516 | 370 / 436 |
| depth-anything-v2-l | 240 | 136 | 55.0 | 31.6 | 4.3x | 1.7x | 2560 / 2654 | 1009 / 1352 | 904 / 1034 |
| depth-anything-v2-s | 45.5 | 38.1 | 7.73 | 4.45 | 8.6x | 1.7x | 373 / 418 | 232 / 326 | 173 / 248 |
| depth-pro | 1769 | 566 | 519 | 271 | 2.1x | 1.9x | 3822 / 4354 | 3289 / 3498 | 3650 / 3778 |
| marigold-depth-v1-1 | 962 | 429 | 408 | 384 | 1.1x | 1.1x | 3414 / 3580 | 2800 / 3760 | 2566 / 3762 |
| metric3d-vit-large | 630 | 439 | 200 | 158 | 2.8x | 1.3x | 3275 / 3370 | 1467 / 1696 | 1499 / 1688 |
| midas-dpt-beit-large-512 | 399 | 320 | 75.6 | 63.9 | 5.0x | 1.2x | 2759 / 2858 | 975 / 1418 | 986 / 1436 |
| moge-2-vitl-normal | 500 | 169 | 145 | 73.9 | 2.3x | 2.0x | 2326 / 2418 | 1645 / 1836 | 1481 / 1610 |
| moge-2-vits-normal | 160 | 81.2 | 61.3 | 32.9 | 2.5x | 1.9x | 1180 / 1288 | 1049 / 1336 | 867 / 1008 |
| moge-3-vitl | 3713 | – | 608 | 149 | – | 4.1x | – | 1992 / 2424 | 1506 / 1662 |
| unidepth-v2-vitl14 | 78.5 | 78.6 | 59.6 | 40.7 | 1.9x | 1.5x | 2820 / 3030 | 1053 / 1472 | 1061 / 1472 |
| unik3d-vitl | 123 | 123 | 99.2 | 58.2 | 2.1x | 1.7x | 3220 / 3530 | 1172 / 1492 | 1130 / 1266 |
| zoedepth-nk | 295 | 250 | 114 | 80.0 | 3.1x | 1.4x | 1965 / 2124 | 1950 / 2124 | 1658 / 1960 |

### Depth, Middlebury 2014 half, 994x1482

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| da3-giant-1.1 | 410 | 227 | 81.8 | 44.2 | 5.1x | 1.9x | 8590 / 8718 | 2638 / 4608 | 2628 / 2710 |
| da3-large-1.1 | 208 | 166 | 37.4 | 17.1 | 9.7x | 2.2x | 3374 / 3472 | 992 / 1350 | 985 / 1090 |
| da3metric-large | 128 | 85.1 | 24.1 | 14.7 | 5.8x | 1.6x | 2996 / 3050 | 815 / 1328 | 811 / 868 |
| da3mono-large | 127 | 85.2 | 23.9 | 14.6 | 5.8x | 1.6x | 2997 / 3052 | 818 / 1328 | 814 / 868 |
| da3nested-giant-large-1.1 | 533 | 308 | 108 | 58.2 | 5.3x | 1.9x | 10332 / 10402 | 3275 / 5914 | 3271 / 3378 |
| depth-anything-v2-b | 93.6 | 63.2 | 18.8 | 10.5 | 6.0x | 1.8x | 928 / 1026 | 450 / 516 | 387 / 436 |
| depth-anything-v2-l | 245 | 140 | 55.1 | 31.1 | 4.5x | 1.8x | 2560 / 2654 | 1026 / 1352 | 921 / 1034 |
| depth-anything-v2-s | 53.6 | 41.0 | 7.82 | 4.51 | 9.1x | 1.7x | 373 / 418 | 249 / 326 | 190 / 248 |
| depth-pro | 1776 | 569 | 518 | 271 | 2.1x | 1.9x | 3828 / 4294 | 3306 / 3498 | 3667 / 3778 |
| marigold-depth-v1-1 | 963 | 439 | 408 | 383 | 1.1x | 1.1x | 3414 / 3580 | 2817 / 3760 | 2583 / 3762 |
| metric3d-vit-large | 629 | 438 | 200 | 158 | 2.8x | 1.3x | 3275 / 3350 | 1484 / 1696 | 1516 / 1688 |
| midas-dpt-beit-large-512 | 386 | 333 | 78.8 | 63.9 | 5.2x | 1.2x | 2759 / 2898 | 992 / 1418 | 1003 / 1436 |
| moge-2-vitl-normal | 501 | 171 | 144 | 73.4 | 2.3x | 2.0x | 2339 / 2438 | 1661 / 1836 | 1498 / 1610 |
| moge-2-vits-normal | 162 | 82.6 | 64.2 | 29.7 | 2.8x | 2.2x | 1193 / 1268 | 1066 / 1336 | 884 / 1008 |
| moge-3-vitl | 32368 | – | 905 | 214 | – | 4.2x | – | 2157 / 2424 | 1690 / 1844 |
| unidepth-v2-vitl14 | 132 | 132 | 99.8 | 66.4 | 2.0x | 1.5x | 3230 / 3508 | 1208 / 1472 | 1211 / 1472 |
| unik3d-vitl | 134 | 134 | 108 | 62.9 | 2.1x | 1.7x | 3307 / 3570 | 1219 / 1492 | 1166 / 1266 |
| zoedepth-nk | 408 | 271 | 124 | 88.9 | 3.1x | 1.4x | 2040 / 2146 | 2037 / 2124 | 1771 / 1960 |

### Depth, synthetic, 504x504

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| da3mono-large | 124 | 52.4 | 35.5 | 21.3 | 2.5x | 1.7x | 2283 / 2330 | 861 / 1332 | 854 / 932 |

### Classification, synthetic, 224x224

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| alexnet.tv_in1k | 1.17 | 1.82 | 0.970 | 0.604 | 1.9x | 1.6x | 383 / 404 | 151 / 224 | 184 / 228 |
| convnextv2_base.fcmae_ft_in22k_in1k | 10.7 | 14.6 | 10.2 | 2.94 | 3.7x | 3.5x | 542 / 566 | 299 / 376 | 318 / 380 |
| deit_base_patch16_224.fb_in1k | 6.39 | 5.46 | 2.98 | 1.64 | 3.3x | 1.8x | 531 / 548 | 203 / 386 | 300 / 370 |
| densenet121.tv_in1k | 10.7 | 13.6 | 9.41 | 1.28 | 8.4x | 7.4x | 80 / 102 | 53 / 90 | 83 / 132 |
| dinov2_vitl14_lc | 22.4 | 10.8 | 5.43 | 4.93 | 2.2x | 1.1x | 1782 / 1806 | 623 / 1228 | 890 / 950 |
| efficientnet_b0.tv_in1k | 5.64 | 7.91 | 4.98 | 0.769 | 7.3x | 6.5x | 64 / 84 | 48 / 96 | 78 / 118 |
| efficientnet_b4.tv_in1k | 11.1 | 15.7 | 10.0 | 2.00 | 5.5x | 5.0x | 147 / 174 | 89 / 114 | 111 / 180 |
| efficientnet_v2_s.tv_in1k | 12.1 | 16.6 | 10.2 | 1.89 | 6.4x | 5.4x | 159 / 176 | 81 / 106 | 114 / 188 |
| efficientvit_b2.r288_in1k | 10.7 | 14.7 | 11.9 | 1.10 | 9.7x | 10.8x | 173 / 194 | 85 / 124 | 120 / 212 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 83.7 | 32.1 | 18.0 | 14.1 | 2.3x | 1.3x | 1819 / 1864 | 648 / 1206 | 1236 / 1386 |
| fastvit_sa24.apple_dist_in1k | 11.9 | 16.5 | 6.88 | 1.39 | 8.6x | 4.9x | 157 / 176 | 81 / 108 | 112 / 190 |
| inception_v3.tv_in1k | 7.88 | 10.3 | 5.21 | 1.30 | 6.1x | 4.0x | 185 / 206 | 120 / 166 | 150 / 228 |
| mobilenet_v2.tv2_in1k | 3.60 | 5.14 | 2.95 | 0.439 | 8.2x | 6.7x | 54 / 72 | 43 / 70 | 75 / 92 |
| mobilenet_v3_large.tv2_in1k | 4.33 | 6.20 | 3.80 | 0.545 | 7.9x | 7.0x | 65 / 80 | 46 / 76 | 78 / 114 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 2.19 | 2.96 | 1.77 | 0.370 | 5.9x | 4.8x | 58 / 74 | 44 / 72 | 76 / 114 |
| mobilenetv4_conv_medium.e500_r256_in1k | 5.78 | 7.77 | 3.91 | 0.641 | 9.0x | 6.1x | 90 / 106 | 54 / 86 | 87 / 124 |
| regnety_160.tv2_in1k | 12.1 | 12.2 | 8.34 | 1.86 | 6.5x | 4.5x | 512 / 524 | 208 / 250 | 229 / 316 |
| resnet152.tv2_in1k | 12.8 | 16.3 | 8.36 | 1.83 | 7.0x | 4.6x | 378 / 390 | 155 / 186 | 182 / 266 |
| resnet18.tv_in1k | 1.91 | 2.44 | 1.48 | 0.459 | 4.2x | 3.2x | 105 / 130 | 61 / 90 | 90 / 150 |
| resnet50.tv2_in1k | 4.74 | 5.84 | 3.15 | 0.826 | 5.7x | 3.8x | 180 / 192 | 89 / 142 | 116 / 162 |
| shufflenet_v2_x1_0.tv_in1k | 4.79 | 6.53 | 3.71 | 0.787 | 6.1x | 4.7x | 47 / 48 | 41 / 72 | 87 / 116 |
| squeezenet1_1.tv_in1k | 1.72 | 2.63 | 1.82 | 0.359 | 4.8x | 5.1x | 10 / 30 | 41 / 72 | 8 / 12 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 15.0 | 19.1 | 10.8 | 2.15 | 7.0x | 5.0x | 538 / 550 | 213 / 394 | 246 / 300 |
| vgg16.tv_in1k | 3.62 | 4.09 | 1.99 | 1.50 | 2.4x | 1.3x | 825 / 846 | 313 / 372 | 332 / 388 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 6.33 | 5.48 | 3.02 | 1.65 | 3.3x | 1.8x | 531 / 548 | 205 / 390 | 300 / 370 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 76.9 | 25.8 | 13.1 | 12.8 | 2.0x | 1.0x | 2495 / 2564 | 879 / 1708 | 1216 / 1346 |

### Segmentation, synthetic, 1024x2048

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| deeplabv3-r101-d8-ade20k | 108 | 54.8 | 47.9 | 20.1 | 2.7x | 2.4x | 2031 / 2162 | 453 / 532 | 362 / 518 |
| deeplabv3plus-r101-d8-ade20k | 104 | 53.4 | 47.1 | 18.0 | 3.0x | 2.6x | 1891 / 2024 | 413 / 492 | 365 / 560 |
| dinov2-b-linear-ade20k | 115 | 40.1 | 33.3 | 22.6 | 1.8x | 1.5x | 2063 / 2426 | 765 / 884 | 771 / 904 |
| dinov2-l-linear-ade20k | 332 | 95.7 | 83.9 | 56.8 | 1.7x | 1.5x | 3308 / 3726 | 1181 / 1244 | 1189 / 1244 |
| dinov2-s-linear-ade20k | 55.8 | 24.7 | 18.3 | 12.4 | 2.0x | 1.5x | 1694 / 2046 | 642 / 696 | 645 / 676 |
| eomt-l-dinov2-ade20k | 178 | 63.7 | 50.8 | 35.0 | 1.8x | 1.5x | 3827 / 4086 | 1515 / 1584 | 1515 / 1584 |
| eomt-l-dinov3-ade20k | 186 | 72.1 | 54.6 | 38.8 | 1.9x | 1.4x | 3822 / 4086 | 1514 / 1584 | 1514 / 1584 |
| fcn-r101-d8-ade20k | 94.2 | 47.0 | 40.7 | 16.6 | 2.8x | 2.4x | 1925 / 2102 | 417 / 492 | 294 / 478 |
| hrnet-w48-fcn-ade20k | 48.1 | 54.8 | 26.9 | 11.9 | 4.1x | 2.3x | 1915 / 2010 | 424 / 490 | 274 / 430 |
| mask2former-swinl-ade20k | 244 | 145 | 103 | 60.4 | 2.4x | 1.7x | 3572 / 3878 | 1164 / 1400 | 1164 / 1400 |
| oneformer-swinl-ade20k | 307 | 205 | 139 | 100.0 | 2.1x | 1.4x | 7395 / 7792 | 2054 / 2476 | 2054 / 2476 |
| pspnet-r101-d8-ade20k | 92.5 | 47.2 | 48.8 | 25.0 | 1.9x | 2.0x | 1922 / 2222 | 417 / 492 | 341 / 494 |
| sapiens-seg-0.3b-goliath | 466 | 173 | 132 | 92.6 | 1.9x | 1.4x | 2520 / 2724 | 1643 / 1762 | 1643 / 1762 |
| sapiens-seg-1b-goliath | 1189 | 354 | 275 | 185 | 1.9x | 1.5x | 5721 / 5906 | 3259 / 3384 | 3259 / 3382 |
| segformer-b0-ade20k | 17.2 | 17.4 | 11.4 | 3.48 | 5.0x | 3.3x | 1590 / 1766 | 339 / 376 | 148 / 238 |
| segformer-b5-ade20k | 132 | 72.0 | 38.7 | 25.3 | 2.8x | 1.5x | 2531 / 2620 | 546 / 590 | 407 / 534 |

### Segmentation, synthetic, 512x512

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| deeplabv3-r101-d8-ade20k | 50.3 | 27.8 | 22.5 | 9.86 | 2.8x | 2.3x | 807 / 942 | 301 / 334 | 274 / 540 |
| deeplabv3plus-r101-d8-ade20k | 47.4 | 25.5 | 21.3 | 8.70 | 2.9x | 2.4x | 667 / 784 | 270 / 316 | 251 / 562 |
| dinov2-b-linear-ade20k | 37.8 | 13.3 | 9.04 | 6.57 | 2.0x | 1.4x | 864 / 886 | 376 / 444 | 312 / 382 |
| dinov2-l-linear-ade20k | 109 | 33.6 | 25.9 | 18.3 | 1.8x | 1.4x | 2116 / 2186 | 792 / 842 | 730 / 804 |
| dinov2-s-linear-ade20k | 18.0 | 8.89 | 4.28 | 3.62 | 2.5x | 1.2x | 485 / 506 | 253 / 296 | 185 / 216 |
| eomt-l-dinov2-ade20k | 88.9 | 32.4 | 22.5 | 15.9 | 2.0x | 1.4x | 2377 / 2526 | 1230 / 1264 | 1230 / 1264 |
| eomt-l-dinov3-ade20k | 92.6 | 36.1 | 24.2 | 17.6 | 2.1x | 1.4x | 2372 / 2526 | 1228 / 1264 | 1228 / 1264 |
| fcn-r101-d8-ade20k | 42.5 | 23.6 | 18.8 | 8.12 | 2.9x | 2.3x | 701 / 802 | 242 / 294 | 223 / 500 |
| hrnet-w48-fcn-ade20k | 40.4 | 47.9 | 20.4 | 5.76 | 7.0x | 3.5x | 691 / 750 | 234 / 310 | 211 / 474 |
| mask2former-swinl-ade20k | 124 | 77.7 | 55.1 | 36.0 | 2.2x | 1.5x | 1963 / 2118 | 873 / 1016 | 873 / 996 |
| oneformer-swinl-ade20k | 149 | 95.6 | 63.1 | 43.8 | 2.2x | 1.4x | 2242 / 2370 | 948 / 1174 | 948 / 1034 |
| pspnet-r101-d8-ade20k | 41.9 | 23.6 | 20.6 | 10.2 | 2.3x | 2.0x | 698 / 862 | 253 / 314 | 246 / 516 |
| sapiens-seg-0.3b-goliath | 466 | 172 | 131 | 92.2 | 1.9x | 1.4x | 2499 / 2704 | 1643 / 1762 | 1643 / 1762 |
| sapiens-seg-1b-goliath | 1188 | 354 | 274 | 184 | 1.9x | 1.5x | 5700 / 5886 | 3259 / 3384 | 3259 / 3382 |
| segformer-b0-ade20k | 8.58 | 10.4 | 4.57 | 1.54 | 5.6x | 3.0x | 366 / 466 | 149 / 196 | 115 / 178 |
| segformer-b5-ade20k | 61.0 | 55.4 | 24.0 | 12.2 | 4.5x | 2.0x | 1518 / 1582 | 441 / 472 | 303 / 414 |

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| sam-vit-b | 168 | 98.2 | 48.3 | 37.0 | 2.7x | 1.3x | 1769 / 1802 | 378 / 420 | 377 / 420 |
| sam-vit-h | 734 | 306 | 214 | 143 | 2.1x | 1.5x | 5295 / 5404 | 1484 / 2524 | 1480 / 1524 |
| sam-vit-l | 406 | 197 | 116 | 81.5 | 2.4x | 1.4x | 3411 / 3464 | 833 / 1262 | 831 / 862 |
| sam2.1-hiera-base-plus | 107 | 47.2 | 32.0 | 25.1 | 1.9x | 1.3x | 823 / 870 | 412 / 464 | 426 / 524 |
| sam2.1-hiera-large | 242 | 86.4 | 68.0 | 48.8 | 1.8x | 1.4x | 1656 / 1724 | 742 / 768 | 760 / 828 |
| sam2.1-hiera-small | 74.8 | 41.8 | 18.8 | 15.6 | 2.7x | 1.2x | 618 / 662 | 318 / 366 | 330 / 386 |
| sam2.1-hiera-tiny | 66.3 | 38.0 | 16.6 | 13.9 | 2.7x | 1.2x | 577 / 636 | 304 / 382 | 316 / 362 |
| sam3 | 695 | 281 | 179 | 116 | 2.4x | 1.5x | 5255 / 5334 | 1922 / 2004 | 2019 / 2264 |

### Promptable segmentation, synthetic, 840x1297

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| sam-vit-b | – | – | 48.7 | 37.3 | – | 1.3x | – | 378 / 420 | 377 / 420 |
| sam-vit-h | – | – | 215 | 144 | – | 1.5x | – | 1484 / 2524 | 1480 / 1524 |
| sam-vit-l | – | – | 116 | 81.8 | – | 1.4x | – | 833 / 1262 | 831 / 862 |
| sam2.1-hiera-base-plus | – | – | 31.9 | 25.0 | – | 1.3x | – | 412 / 464 | 426 / 524 |
| sam2.1-hiera-large | – | – | 67.9 | 48.7 | – | 1.4x | – | 742 / 768 | 760 / 828 |
| sam2.1-hiera-small | – | – | 18.8 | 15.5 | – | 1.2x | – | 318 / 366 | 330 / 386 |
| sam2.1-hiera-tiny | – | – | 16.6 | 13.9 | – | 1.2x | – | 304 / 382 | 316 / 362 |
| sam3 | 695 | 282 | 180 | 116 | 2.4x | 1.6x | 5255 / 5332 | 1922 / 2004 | 2048 / 2264 |

### Detection, synthetic, 800x1344

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| detr-r101-coco | 78.9 | 80.8 | 18.7 | 13.7 | 5.8x | 1.4x | 531 / 650 | 316 / 368 | 254 / 328 |
| detr-r50-coco | 55.5 | 62.0 | 12.4 | 9.93 | 5.6x | 1.2x | 459 / 558 | 280 / 332 | 218 / 272 |
| dfine-l-coco | 37.1 | 44.9 | 16.2 | 8.70 | 4.3x | 1.9x | 278 / 350 | 152 / 172 | 179 / 300 |
| dfine-m-coco | 30.8 | 38.2 | 13.5 | 5.97 | 5.2x | 2.3x | 209 / 256 | 126 / 142 | 156 / 252 |
| dfine-n-coco | 24.3 | 29.1 | 10.3 | 2.72 | 8.9x | 3.8x | 92 / 126 | 88 / 116 | 124 / 202 |
| dfine-s-coco | 25.7 | 31.9 | 11.1 | 4.04 | 6.4x | 2.7x | 152 / 184 | 105 / 128 | 138 / 212 |
| dfine-x-coco | 44.9 | 54.6 | 18.5 | 14.1 | 3.2x | 1.3x | 476 / 536 | 227 / 258 | 237 / 364 |
| faster-rcnn-r50-fpn-3x-coco | 83.6 | 41.4 | 22.8 | 19.5 | 2.1x | 1.2x | 626 / 708 | 390 / 490 | 391 / 450 |
| mask-rcnn-r50-fpn-3x-coco | 85.4 | 43.5 | 24.6 | 21.0 | 2.1x | 1.2x | 604 / 650 | 352 / 440 | 353 / 400 |
| rtdetrv2-l-coco | 30.3 | 38.4 | 12.1 | 6.61 | 4.6x | 1.8x | 354 / 412 | 177 / 200 | 200 / 306 |
| rtdetrv2-m-coco | 17.1 | 20.9 | 10.4 | 5.27 | 3.2x | 2.0x | 254 / 270 | 156 / 170 | 177 / 274 |
| rtdetrv2-ms-coco | 24.7 | 22.5 | 9.60 | 5.46 | 4.1x | 1.8x | 309 / 358 | 166 / 182 | 189 / 306 |
| rtdetrv2-s-coco | 14.1 | 16.8 | 8.04 | 4.19 | 3.4x | 1.9x | 192 / 220 | 134 / 166 | 155 / 250 |
| rtdetrv2-x-coco | 36.3 | 46.8 | 14.6 | 9.24 | 3.9x | 1.6x | 589 / 622 | 244 / 278 | 262 / 384 |

### Features, synthetic, 224x224

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| dinov2-vitb14 | 6.90 | 4.50 | 2.92 | 1.48 | 3.0x | 2.0x | 530 / 548 | 205 / 226 | 237 / 290 |
| dinov2-vitb14-reg | 7.00 | 4.30 | 2.94 | 1.50 | 2.9x | 2.0x | 530 / 548 | 207 / 226 | 237 / 288 |
| dinov2-vitg14 | 86.4 | 33.9 | 18.9 | 13.2 | 2.6x | 1.4x | 6542 / 6548 | 2216 / 2230 | 2246 / 2318 |
| dinov2-vitg14-reg | 86.6 | 34.2 | 18.7 | 13.0 | 2.6x | 1.4x | 6542 / 6568 | 2217 / 2228 | 2222 / 4430 |
| dinov2-vitl14 | 22.8 | 9.80 | 6.12 | 4.59 | 2.1x | 1.3x | 1777 / 1784 | 622 / 646 | 655 / 714 |
| dinov2-vitl14-reg | 22.5 | 10.7 | 6.17 | 4.60 | 2.3x | 1.3x | 1777 / 1786 | 625 / 646 | 655 / 714 |
| dinov2-vits14 | 3.30 | 4.40 | 2.95 | 0.848 | 3.9x | 3.5x | 160 / 166 | 79 / 98 | 112 / 148 |
| dinov2-vits14-reg | 4.11 | 5.59 | 2.98 | 0.814 | 5.0x | 3.7x | 160 / 168 | 80 / 98 | 112 / 142 |
| dinov3-convnext-base | 7.38 | 8.24 | 5.71 | 2.18 | 3.4x | 2.6x | 535 / 552 | 208 / 232 | 200 / 214 |
| dinov3-convnext-large | 10.5 | 9.59 | 5.45 | 3.59 | 2.7x | 1.5x | 1163 / 1252 | 419 / 430 | 441 / 494 |
| dinov3-convnext-small | 5.97 | 8.09 | 5.42 | 1.99 | 3.0x | 2.7x | 328 / 348 | 134 / 150 | 127 / 152 |
| dinov3-convnext-tiny | 2.96 | 4.47 | 3.04 | 1.18 | 2.5x | 2.6x | 198 / 206 | 93 / 108 | 86 / 112 |
| dinov3-vitb16 | 6.95 | 9.33 | 4.86 | 1.40 | 5.0x | 3.5x | 527 / 550 | 201 / 206 | 201 / 252 |
| dinov3-vith16plus | 48.0 | 25.9 | 13.3 | 9.30 | 2.8x | 1.4x | 4849 / 4876 | 1645 / 1668 | 1644 / 1694 |
| dinov3-vitl16 | 19.2 | 17.7 | 9.13 | 3.28 | 5.4x | 2.8x | 1772 / 1812 | 617 / 646 | 617 / 674 |
| dinov3-vits16 | 6.47 | 9.16 | 4.67 | 0.903 | 7.2x | 5.2x | 158 / 166 | 77 / 80 | 77 / 106 |

### Multiview, synthetic, 336x518

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| mapanything-apache | 2875 | 1967 | 636 | 544 | 3.6x | 1.2x | 10227 / 10952 | 3697 / 4092 | 3697 / 4072 |
| vggt-1b | 2502 | 965 | 651 | 458 | 2.1x | 1.4x | 8688 / 9158 | 3691 / 4674 | 3702 / 4654 |
| vggt-omega-1b-512 | 2650 | 906 | 726 | 464 | 2.0x | 1.6x | 7495 / 7654 | 3421 / 3518 | 3422 / 3518 |

### Multiview, synthetic, 1536x1024

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| patchmatchnet | 248 | 220 | 202 | 29.6 | 7.4x | 6.8x | 1228 / 1272 | 1806 / 2096 | 740 / 856 |

### Multiview, synthetic, 448x672

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| mvsanywhere | 273 | 219 | 93.2 | 35.6 | 6.1x | 2.6x | 1504 / 1576 | 931 / 1226 | 761 / 884 |

### Startup and first call

load: from the load call until the model is on the GPU; first call: the whole first inference call, including one normal inference (see the speed table for the warm time). Seconds, at each task's standard size (another size is shown in brackets).

| model | load (s): original | portable | kernels | first call (s): original | portable | kernels |
|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 0.15 (375x1242) | 0.16 | 0.20† | 0.70 (375x1242) | 0.17† | 0.09† |
| aanet-sceneflow | 0.12 (375x1242) | 0.19 | 0.27† | 0.07 (375x1242) | 0.14† | 0.10† |
| bridgedepth-rvc-pretrain | 5.87 (375x1242) | 1.84 | 0.76† | 1.36 (375x1242) | 0.45† | 0.36† |
| bridgedepth-sf | 4.86 (375x1242) | 1.53 | 0.72† | 0.12 (375x1242) | 0.44† | 0.35† |
| c-fast-foundationstereo | 0.17 (375x1242) | 0.24 | 0.16† | 0.17 (375x1242) | 0.33† | 0.14† |
| c-foundationstereo-s | 1.67 (375x1242) | 0.62 | 0.40† | 2.73 (375x1242) | 3.65† | 1.36† |
| cas-gwcnet-c-sceneflow | 0.48 (375x1242) | 1.29 | 0.18† | 0.41 (375x1242) | 0.33† | 0.23† |
| cas-gwcnet-gc-sceneflow | 0.39 | 0.97 | 0.17† | 0.64 | 0.47† | 0.36† |
| cfnet-sceneflow | 0.55 | 0.94 | 0.21† | 0.56 | 0.48† | 0.40† |
| coex-sceneflow | – | 0.16 | 0.09† | 0.63 (375x1242) | 0.06† | 0.04† |
| crestereo-eth3d | 0.05 | 0.03 | 0.03† | 1.50 | 0.75† | 0.23† |
| defom-stereo-vitl-sceneflow | 4.24 | 2.31 | 1.18† | 2.83 | 1.53† | 0.66† |
| dlnr-middlebury | 0.83 (375x1242) | 0.63 | 0.26† | 1.71 (375x1242) | 0.40† | 0.21† |
| dlnr-sceneflow | 0.83 (375x1242) | 0.63 | 0.26† | 1.71 (375x1242) | 0.40† | 0.21† |
| fast-acvnet-generalization | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-kitti12 | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-kitti15 | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-plus-generalization | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-kitti12 | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-kitti15 | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-sceneflow | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-sceneflow | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-foundationstereo | 0.16 (375x1242) | 0.25 | 0.14† | 0.14 (375x1242) | 0.33† | 0.14† |
| foundationstereo-vit-l | 10.86 (375x1242) | 1.98 | 1.17† | 2.56 (375x1242) | 2.99† | 1.52† |
| foundationstereo-vit-s | 1.44 (375x1242) | 1.03 | 0.33† | 1.06 (375x1242) | 2.51† | 1.12† |
| great-igev-eth3d | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| great-igev-kitti | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| great-igev-middlebury | 0.39 (375x1242) | 3.91 | 0.61† | 4.59 (375x1242) | 1.81† | 0.59† |
| great-igev-sceneflow | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| gwcnet-g-sceneflow | 0.26 (375x1242) | 0.40 | 0.14† | 0.46 (375x1242) | 0.34† | 0.24† |
| gwcnet-gc-sceneflow | 0.29 (375x1242) | 0.39 | 0.44† | 0.12 (375x1242) | 0.35† | 0.26† |
| hitnet-eth3d | 0.31 | 0.09 | 0.07† | 2.06 | 0.08† | 0.03† |
| hitnet-middlebury-d400 | 0.57 | 0.13 | 0.10† | 3.51 | 0.18† | 0.05† |
| igev-plusplus-eth3d | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-kitti12 | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-kitti15 | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-middlebury | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-sceneflow | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-stereo-eth3d | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti12 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti15 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-middlebury | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-sceneflow | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| las2-h | 0.26 (375x1242) | 0.18 | 0.13† | 1.21 (375x1242) | 0.14† | 0.06† |
| las2-l | 0.39 (375x1242) | 0.23 | 0.18† | 0.04 (375x1242) | 0.08† | 0.03† |
| las2-m | 0.17 (375x1242) | 0.12 | 0.10† | 0.02 (375x1242) | 0.05† | 0.02† |
| las2-s | 0.11 (375x1242) | 0.09 | 0.07† | 0.02 (375x1242) | 0.05† | 0.01† |
| mocha-v1-sceneflow | 1.95 (375x1242) | 0.79 | 0.48† | 1.85 (375x1242) | 1.06† | 0.45† |
| mocha-v2-sceneflow | 2.37 (375x1242) | 1.30 | 0.45† | 0.40 (375x1242) | 1.03† | 0.45† |
| monster-mix-all | 5.85 (375x1242) | 4.83 | 0.95† | 1.26 (375x1242) | 1.59† | 0.67† |
| monster-plusplus-mix-all | 8.37 (375x1242) | 2.45 | 0.79† | 1.23 (375x1242) | 1.63† | 0.70† |
| monster-sceneflow | 5.85 (375x1242) | 4.83 | 0.95† | 1.26 (375x1242) | 1.59† | 0.67† |
| nmrf-sceneflow | 0.21 (375x1242) | 0.10 | 0.07† | 0.77 (375x1242) | 0.22† | 0.20† |
| psmnet-sceneflow | 0.20 (375x1242) | 0.10 | 0.08† | 0.53 (375x1242) | 0.35† | 0.23† |
| raft-stereo-eth3d | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| raft-stereo-middlebury | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| raft-stereo-sceneflow | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| rt-monster-plusplus-mix-all | 0.99 (375x1242) | 0.26 | 0.23† | 0.66 (375x1242) | 0.13† | 0.10† |
| s2m2-l | 1.38 | 0.99 | 0.58† | 1.01 | 1.01† | 0.37† |
| s2m2-m | 3.95 (375x1242) | 0.50 | 0.31† | 0.27 (375x1242) | 0.68† | 0.25† |
| s2m2-s | 0.22 (375x1242) | 0.22 | 0.20† | 0.14 (375x1242) | 0.35† | 0.11† |
| s2m2-xl | 4.50 (375x1242) | 2.73 | 0.90† | 0.53 (375x1242) | 1.81† | 0.68† |
| selective-igev-eth3d-finetune | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-eth3d-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti12 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti15 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-finetune | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-sceneflow | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| waft-stereo-b | 3.07 (375x1242) | 1.65 | 0.42† | 0.44 (375x1242) | 1.15† | 0.39† |
| waft-stereo-s | 1.60 (375x1242) | 0.58 | 0.30† | 0.14 (375x1242) | 0.69† | 0.28† |
| da3-giant-1.1 | 12.39 | 6.29 | 0.90† | 0.68 | 0.16 | 0.06 |
| da3-large-1.1 | 3.46 | 3.65 | 0.31† | 0.45 | 0.25 | 0.03 |
| da3metric-large | 2.70 | 0.35 | 0.30† | 0.28 | 0.05 | 0.02 |
| da3mono-large | 2.96 | 4.25 | 1.44† | 0.28 | 0.09 | 0.02 |
| da3nested-giant-large-1.1 | 28.67 | 13.25 | 2.73† | 1.63 | 0.55 | 0.07 |
| depth-anything-v2-b | 3.94 | 0.85 | 0.20† | 0.08 | 0.02 | 0.01 |
| depth-anything-v2-l | 5.66 | 2.39 | 0.91† | 0.15 | 0.08 | 0.06 |
| depth-anything-v2-s | 0.71 | 0.11 | 0.08† | 0.38 | 0.03 | 0.01 |
| depth-pro | 17.34 | 4.67 | 1.11† | 0.74 | 0.58 | 0.37 |
| marigold-depth-v1-1 | 1.32 | 8.20 | 1.10† | 0.60 | 0.56 | 0.52 |
| metric3d-vit-large | 9.96 | 3.45 | 0.31† | 0.88 | 0.23 | 0.16 |
| midas-dpt-beit-large-512 | 5.16 | 2.49 | 0.28† | 1.14 | 0.09 | 0.07 |
| moge-2-vitl-normal | 7.24 | 1.37 | 0.63† | 0.17 | 0.14 | 0.13 |
| moge-2-vits-normal | 0.88 | 0.35 | 0.11† | 1.50 | 0.08 | 0.10 |
| moge-3-vitl | 9.64 | 1.47 | 1.13 | 34.35 | 0.95 | 0.29 |
| unidepth-v2-vitl14 | 3.80 | 4.16 | 0.90† | 0.38 | 0.15 | 0.07 |
| unik3d-vitl | 7.28 | 4.16 | 0.84† | 0.57 | 0.14 | 0.07 |
| zoedepth-nk | 8.35 | 1.92 | 0.33† | 1.53 | 0.28 | 0.12 |
| alexnet.tv_in1k | 2.51 | 1.30 | 1.40 | 0.81 | 0.02 | 0.06 |
| convnextv2_base.fcmae_ft_in22k_in1k | 2.70 | 1.23 | 2.12 | 0.63 | 0.03 | 0.16 |
| deit_base_patch16_224.fb_in1k | 2.52 | 0.72 | 1.96 | 0.31 | 0.00 | 0.07 |
| densenet121.tv_in1k | 1.22 | 0.41 | 1.89 | 0.56 | 0.09 | 0.04 |
| dinov2_vitl14_lc | 5.80 | 3.34 | 3.94 | 0.32 | 0.02 | 0.10 |
| efficientnet_b0.tv_in1k | 1.25 | 0.17 | 1.53 | 2.86 | 0.38 | 0.06 |
| efficientnet_b4.tv_in1k | 0.41 | 0.37 | 1.94 | 1.72 | 0.06 | 0.07 |
| efficientnet_v2_s.tv_in1k | 1.48 | 0.35 | 1.90 | 0.97 | 0.04 | 0.05 |
| efficientvit_b2.r288_in1k | 1.84 | 0.40 | 1.92 | 1.89 | 0.04 | 0.05 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 6.98 | 2.80 | 3.61 | 0.39 | 0.02 | 0.10 |
| fastvit_sa24.apple_dist_in1k | 2.49 | 0.62 | 2.42 | 1.67 | 0.17 | 0.05 |
| inception_v3.tv_in1k | 1.62 | 0.28 | 1.86 | 0.40 | 0.04 | 0.05 |
| mobilenet_v2.tv2_in1k | 1.23 | 0.11 | 1.51 | 1.65 | 0.03 | 0.05 |
| mobilenet_v3_large.tv2_in1k | 1.25 | 0.34 | 1.46 | 1.97 | 0.03 | 0.05 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 1.34 | 0.06 | 1.70 | 1.50 | 0.01 | 0.05 |
| mobilenetv4_conv_medium.e500_r256_in1k | 1.56 | 0.36 | 1.89 | 2.29 | 0.04 | 0.05 |
| regnety_160.tv2_in1k | 3.07 | 1.08 | 2.22 | 0.48 | 0.04 | 0.06 |
| resnet152.tv2_in1k | 2.34 | 0.43 | 1.88 | 0.24 | 0.02 | 0.04 |
| resnet18.tv_in1k | 0.27 | 0.07 | 1.48 | 0.00 | 0.00 | 0.04 |
| resnet50.tv2_in1k | 0.51 | 0.16 | 1.48 | 0.01 | 0.00 | 0.04 |
| shufflenet_v2_x1_0.tv_in1k | 1.13 | 0.14 | 1.55 | 1.15 | 0.02 | 0.08 |
| squeezenet1_1.tv_in1k | 1.09 | 0.05 | 1.44 | 0.37 | 0.02 | 0.03 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 2.71 | 0.94 | 2.28 | 0.41 | 0.03 | 0.09 |
| vgg16.tv_in1k | 3.22 | 1.74 | 1.90 | 0.30 | 0.00 | 0.06 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 2.49 | 0.73 | 2.05 | 0.28 | 0.00 | 0.07 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 7.52 | 7.29 | 5.82 | 0.35 | 0.07 | 0.09 |
| deeplabv3-r101-d8-ade20k | 1.64 (512x512) | 0.73 | 0.91† | 0.03 (512x512) | 0.08 | 0.02† |
| deeplabv3plus-r101-d8-ade20k | 4.72 (512x512) | 0.62 | 0.75† | 0.66 (512x512) | 0.10 | 0.02† |
| dinov2-b-linear-ade20k | 12.01 (512x512) | 0.27 | 0.40† | 0.02 (512x512) | 0.10 | 0.07† |
| dinov2-l-linear-ade20k | 7.86 (512x512) | 0.81 | 0.92† | 0.38 (512x512) | 0.16 | 0.10† |
| dinov2-s-linear-ade20k | 4.02 (512x512) | 0.12 | 0.23† | 0.30 (512x512) | 0.09 | 0.06† |
| eomt-l-dinov2-ade20k | 4.80 (512x512) | 3.13 | 3.27† | 0.54 (512x512) | 0.05 | 0.04† |
| eomt-l-dinov3-ade20k | 1.83 (512x512) | 3.20 | 3.24† | 0.07 (512x512) | 0.08 | 0.04† |
| fcn-r101-d8-ade20k | 4.72 (512x512) | 0.53 | 0.91† | 0.39 (512x512) | 0.07 | 0.02† |
| hrnet-w48-fcn-ade20k | 1.48 (512x512) | 0.62 | 0.85† | 0.08 (512x512) | 0.07 | 0.01† |
| mask2former-swinl-ade20k | 4.86 (512x512) | 3.03 | 3.17† | 0.76 (512x512) | 0.13 | 0.07† |
| oneformer-swinl-ade20k | 5.08 (512x512) | 4.10 | 3.27† | 0.81 (512x512) | 0.15† | 0.11† |
| pspnet-r101-d8-ade20k | 1.35 (512x512) | 0.47 | 0.77† | 0.09 (512x512) | 0.08 | 0.03† |
| sapiens-seg-0.3b-goliath | 4.92 (512x512) | 1.22 | 0.96† | 1.02 (512x512) | 0.13† | 0.09† |
| sapiens-seg-1b-goliath | 11.33 (512x512) | 3.54 | 2.27† | 0.57 (512x512) | 0.27† | 0.18† |
| segformer-b0-ade20k | 4.83 (512x512) | 2.55 | 2.92 | 2.23 (512x512) | 0.08 | 0.01 |
| segformer-b5-ade20k | 1.24 (512x512) | 3.09 | 3.43 | 0.50 (512x512) | 0.08 | 0.03 |
| sam-vit-b | 0.75 | 0.46 | 0.17† | 0.18 | 0.05† | 0.04† |
| sam-vit-h | 4.61 | 2.45 | 0.89† | 0.38 | 0.21† | 0.14† |
| sam-vit-l | 1.80 | 1.21 | 0.43† | 0.27 | 0.12† | 0.08† |
| sam2.1-hiera-base-plus | 0.74 | 0.43 | 0.19† | 0.14 | 0.03† | 0.03† |
| sam2.1-hiera-large | 1.95 | 1.10 | 0.39† | 0.18 | 0.07† | 0.05† |
| sam2.1-hiera-small | 0.48 | 0.36 | 0.13† | 0.14 | 0.02† | 0.02† |
| sam2.1-hiera-tiny | 0.41 | 0.25 | 0.11† | 0.15 | 0.02† | 0.01† |
| sam3 | 1.12 (840x1297) | 2.95 | 0.70† | 0.96 (840x1297) | 0.18† | 0.12† |
| detr-r101-coco | 0.95 | 0.81 | 0.22† | 0.06 | 0.02 | 0.01 |
| detr-r50-coco | 2.55 | 0.70 | 0.28† | 0.83 | 0.44 | 0.03 |
| dfine-l-coco | 3.68 | 1.09 | 1.46 | 1.41 | 0.67 | 0.47 |
| dfine-m-coco | 3.63 | 1.04 | 1.20 | 1.55 | 0.69 | 0.56 |
| dfine-n-coco | 5.24 | 1.25 | 1.21 | 4.07 | 0.85 | 0.62 |
| dfine-s-coco | 2.99 | 1.05 | 1.16 | 1.44 | 0.71 | 0.52 |
| dfine-x-coco | 3.88 | 1.39 | 1.41 | 1.45 | 0.65 | 0.45 |
| faster-rcnn-r50-fpn-3x-coco | 0.73 | 0.53 | 0.14† | 0.04 | 0.25 | 0.04 |
| mask-rcnn-r50-fpn-3x-coco | 0.97 | 1.50 | 0.22† | 0.84 | 0.42 | 0.04 |
| rtdetrv2-l-coco | 0.66 | 0.50 | 0.25† | 0.04 | 0.01 | 0.02 |
| rtdetrv2-m-coco | 0.45 | 0.33 | 0.16† | 0.02 | 0.01 | 0.01 |
| rtdetrv2-ms-coco | 0.52 | 0.45 | 0.17† | 0.06 | 0.03 | 0.09 |
| rtdetrv2-s-coco | 1.32 | 0.31 | 0.46† | 0.42 | 0.10 | 0.53 |
| rtdetrv2-x-coco | 1.13 | 1.09 | 0.29† | 0.08 | 0.05 | 0.02 |
| dinov2-vitb14 | 1.51 | 0.27 | 0.19 | 0.01 | 0.20 | 0.19 |
| dinov2-vitb14-reg | 1.71 | 0.59 | 0.21 | 0.04 | 0.04 | 0.20 |
| dinov2-vitg14 | 19.29 | 4.37 | 2.22 | 0.05 | 0.10 | 0.16 |
| dinov2-vitg14-reg | 14.67 | 4.32 | 2.91† | 0.09 | 0.03 | 0.02 |
| dinov2-vitl14 | 7.06 | 0.57 | 0.57 | 0.01 | 0.15 | 0.18 |
| dinov2-vitl14-reg | 4.60 | 0.66 | 0.56 | 0.07 | 0.11 | 0.20 |
| dinov2-vits14 | 0.41 | 0.07 | 0.07 | 0.17 | 0.12 | 0.15 |
| dinov2-vits14-reg | 1.32 | 0.21 | 0.08 | 0.11 | 0.13 | 0.17 |
| dinov3-convnext-base | 0.42 | 0.43 | 0.19 | 0.01 | 0.27 | 0.10 |
| dinov3-convnext-large | 1.64 | 0.85 | 0.38 | 0.02 | 0.24 | 0.20 |
| dinov3-convnext-small | 0.27 | 0.25 | 0.11 | 0.01 | 0.33 | 0.07 |
| dinov3-convnext-tiny | 0.43 | 0.48 | 0.08 | 1.03 | 0.42 | 0.12 |
| dinov3-vitb16 | 0.68 | 0.38 | 0.16 | 0.01 | 0.20 | 0.29 |
| dinov3-vith16plus | 5.95 | 2.02 | 1.40 | 0.05 | 0.10 | 0.20 |
| dinov3-vitl16 | 2.37 | 0.54 | 0.52 | 0.02 | 0.18 | 0.25 |
| dinov3-vits16 | 0.48 | 0.07 | 0.06 | 0.95 | 0.27 | 0.26 |
| mapanything-apache | 6.12 (336x518) | 2.89 (336x518) | 0.57† (336x518) | 2.62 (336x518) | 0.66 (336x518) | 0.54 (336x518) |
| mvsanywhere | 3.21 (448x672) | 1.88 (448x672) | 0.52† (448x672) | 1.04 (448x672) | 0.22 (448x672) | 0.05 (448x672) |
| patchmatchnet | 0.04 (1536x1024) | 0.06 (1536x1024) | 0.05† (1536x1024) | 0.92 (1536x1024) | 0.25 (1536x1024) | 0.04 (1536x1024) |
| vggt-1b | 19.37 (336x518) | 4.98 (336x518) | 5.99† (336x518) | 1.29 (336x518) | 0.69 (336x518) | 0.46 (336x518) |
| vggt-omega-1b-512 | 16.31 (336x518) | 3.31 (336x518) | 2.27† (336x518) | 1.10 (336x518) | 0.73 (336x518) | 0.46 (336x518) |

† measured in the speed sweep after the portable load (or an earlier size) in the same process: file cache and kernels are warm, so the time is lower than a fresh start.
<!-- tables:end -->

## torch.compile and TensorRT

Portable with `model.compile()` (Inductor, default mode) and the fp16 TensorRT engine of the portable model
(`export_tensorrt`), one variant per family at the task's standard size, next to portable and `kernels`
([export](export.md) has the methods). The compile time is the cold first call (empty Inductor cache); the engine
runs `module()` on synthetic inputs of the shape the timed call gives the network (classifier crop, all views of a
scene; "only" marks engines of part of the call) and its memory is the device memory it holds. Rows measured by
`tools/speed_check.py run --modes compile tensorrt`; a failure is shown as `fails: reason`.

<!-- compile:begin (tools/gen_docs.py, from docs/data/speed_compile.jsonl) -->
| model | input | time: portable | compile | TensorRT fp16 | kernels | preparation: compile (cold first call) | ONNX + engine build | peak MiB (alloc / reserved): portable | compile | kernels | TensorRT (device) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-sceneflow | 994x1482 | 140 ms | 91.8 ms | 91.2 ms | 101 ms | 79.9 s | 211 s | 845 / 986 | 754 / 928 | 536 / 644 | 482 |
| bridgedepth-sf | 994x1482 | 437 ms | 325 ms | fails: TensorRT bug: Myelin internal error | 348 ms | 111 s | – | 2916 / 3214 | 2280 / 2308 | 2310 / 2634 | – |
| c-fast-foundationstereo | 994x1482 | 330 ms | 339 ms | 451 ms | 135 ms | 75.4 s | 240 s | 2109 / 2336 | 1649 / 1792 | 1196 / 1516 | 1252 |
| cas-gwcnet-c-sceneflow | 994x1482 | 327 ms | 295 ms | 220 ms | 225 ms | 61.7 s | 131 s | 2006 / 2474 | 2098 / 2448 | 1725 / 1930 | 1440 |
| cfnet-sceneflow | 994x1482 | 484 ms | 249 ms | 171 ms | 401 ms | 53.0 s | 139 s | 3441 / 4376 | 1026 / 1398 | 3662 / 4718 | 1238 |
| coex-sceneflow | 994x1482 | 49.9 ms | 29.4 ms | 20.6 ms | 23.5 ms | 22.4 s | 56.6 s | 1136 / 1254 | 351 / 452 | 495 / 886 | 360 |
| crestereo-eth3d | 994x1482 | 732 ms | 398 ms | 298 ms | 229 ms | 120 s | 224 s | 884 / 1030 | 695 / 748 | 644 / 732 | 584 |
| defom-stereo-vitl-sceneflow | 994x1482 | 1531 ms | 1002 ms | fails: TensorRT bug: Myelin internal error | 659 ms | 122 s | – | 4362 / 4838 | 3021 / 3292 | 3154 / 3566 | – |
| dlnr-sceneflow | 994x1482 | 398 ms | 301 ms | 241 ms | 214 ms | 69.0 s | 161 s | 2635 / 3142 | 1343 / 1672 | 1236 / 1606 | 1492 |
| fast-acvnet-sceneflow | 994x1482 | 54.7 ms | 36.2 ms | 36.7 ms | 38.2 ms | 41.8 s | 71.3 s | 664 / 736 | 584 / 698 | 739 / 1042 | 484 |
| foundationstereo-vit-s | 994x1482 | 2514 ms | 2044 ms | 1196 ms | 1120 ms | 212 s | 545 s | 2493 / 2606 | 2822 / 3646 | 2169 / 2470 | 2570 |
| great-igev-eth3d | 994x1482 | 1085 ms | 829 ms | 940 ms | 393 ms | 224 s | 685 s | 1402 / 1678 | 2831 / 3362 | 1192 / 1564 | 6018 |
| gwcnet-g-sceneflow | 994x1482 | 344 ms | 417 ms | 275 ms | 238 ms | 52.6 s | 66.0 s | 2081 / 2402 | 1527 / 1864 | 1799 / 2102 | 3548 |
| hitnet-eth3d | 994x1482 | 67.8 ms | 27.7 ms | 31.1 ms | 23.2 ms | 39.1 s | 136 s | 860 / 1090 | 719 / 728 | 647 / 1248 | 836 |
| igev-plusplus-eth3d | 994x1482 | 1081 ms | 716 ms | 494 ms | 393 ms | 159 s | 425 s | 1594 / 1840 | 952 / 1074 | 1389 / 2302 | 3632 |
| igev-stereo-eth3d | 994x1482 | 763 ms | 527 ms | 339 ms | 288 ms | 111 s | 213 s | 1237 / 1492 | 807 / 1052 | 1113 / 1374 | 1220 |
| las2-s | 994x1482 | 46.9 ms | 23.7 ms | 10.3 ms | 14.3 ms | 22.4 s | 86.9 s | 955 / 988 | 238 / 264 | 320 / 558 | 162 |
| mocha-v2-sceneflow | 994x1482 | 1033 ms | 713 ms | 437 ms | 448 ms | 148 s | 361 s | 1632 / 1770 | 1122 / 1308 | 1587 / 1790 | 1098 |
| monster-plusplus-mix-all | 994x1482 | 1633 ms | 1126 ms | 697 ms | 698 ms | 109 s | 416 s | 2583 / 2918 | 2328 / 2532 | 2438 / 2760 | 2642 |
| monster-sceneflow | 994x1482 | 1585 ms | 1096 ms | 671 ms | 668 ms | 103 s | 384 s | 2336 / 2494 | 2081 / 2272 | 2356 / 2714 | 2018 |
| nmrf-sceneflow | 994x1482 | 216 ms | 161 ms | 184 ms | 196 ms | 84.8 s | 142 s | 1510 / 1552 | 854 / 1104 | 1022 / 1112 | 826 |
| psmnet-sceneflow | 994x1482 | 340 ms | 440 ms | 669 ms | 232 ms | 60.0 s | 88.9 s | 2078 / 2442 | 1760 / 2082 | 1786 / 2122 | 1528 |
| raft-stereo-eth3d | 994x1482 | 746 ms | 506 ms | 333 ms | 273 ms | 99.9 s | 132 s | 1388 / 1492 | 815 / 988 | 765 / 802 | 826 |
| rt-monster-plusplus-mix-all | 994x1482 | 113 ms | 77.0 ms | 44.1 ms | 94.1 ms | 69.4 s | 150 s | 2592 / 2646 | 2750 / 2942 | 2704 / 3976 | 952 |
| s2m2-s | 994x1482 | 348 ms | 186 ms | 137 ms | 113 ms | 95.9 s | 363 s | 1733 / 1930 | 1579 / 1710 | 686 / 820 | 980 |
| selective-igev-eth3d-finetune | 994x1482 | 998 ms | 691 ms | 461 ms | 349 ms | 126 s | 256 s | 1238 / 1488 | 919 / 1088 | 1109 / 1390 | 1148 |
| waft-stereo-s | 994x1482 | 689 ms | 357 ms | 170 ms | 284 ms | 95.1 s | 152 s | 3293 / 3456 | 2024 / 2058 | 1113 / 1286 | 662 |
| da3mono-large | 994x1482 | 23.9 ms | 21.2 ms | 14.4 ms | 14.6 ms | 25.0 s | 52.3 s | 818 / 1328 | 792 / 828 | 814 / 868 | 754 |
| depth-anything-v2-s | 994x1482 | 7.82 ms | 6.65 ms | 4.46 ms | 4.51 ms | 15.7 s | 29.9 s | 249 / 326 | 201 / 224 | 190 / 248 | 186 |
| depth-pro | 994x1482 | 518 ms | 488 ms | 299 ms | 271 ms | 39.2 s | 163 s | 3306 / 3498 | 3053 / 3198 | 3667 / 3778 | 2926 |
| marigold-depth-v1-1 | 994x1482 | 408 ms | 342 ms | 172 ms | 383 ms | 127 s | 285 s | 2817 / 3760 | 2341 / 2418 | 2583 / 3762 | 2368 |
| metric3d-vit-large | 994x1482 | 200 ms | 149 ms | 91.4 ms | 158 ms | 54.7 s | 96.6 s | 1484 / 1696 | 1126 / 1148 | 1516 / 1688 | 1010 |
| midas-dpt-beit-large-512 | 994x1482 | 78.8 ms | 58.8 ms | 52.2 ms | 63.9 ms | 32.6 s | 53.5 s | 992 / 1418 | 1088 / 1116 | 1003 / 1436 | 800 |
| moge-2-vits-normal | 994x1482 | 64.2 ms | 43.9 ms | 24.5 ms | 29.7 ms | 124 s | 131 s | 1066 / 1336 | 836 / 992 | 884 / 1008 | 620 |
| unidepth-v2-vitl14 | 994x1482 | 99.8 ms | 96.4 ms | 64.3 ms | 66.4 ms | 31.1 s | 119 s | 1208 / 1472 | 1304 / 1392 | 1211 / 1472 | 1050 |
| unik3d-vitl | 994x1482 | 108 ms | 94.4 ms | 58.0 ms | 62.9 ms | 32.8 s | 105 s | 1219 / 1492 | 1258 / 1312 | 1166 / 1266 | 846 |
| zoedepth-nk | 994x1482 | 124 ms | 71.7 ms | 73.3 ms | 88.9 ms | 93.8 s | 138 s | 2037 / 2124 | 1813 / 1880 | 1771 / 1960 | 852 |
| alexnet.tv_in1k | 224x224 | 0.970 ms | 0.824 ms | 0.401 ms | 0.604 ms | 4.33 s | 13.4 s | 151 / 224 | 183 / 204 | 184 / 228 | 128 |
| convnextv2_base.fcmae_ft_in22k_in1k | 224x224 | 10.2 ms | 3.85 ms | 2.13 ms | 2.94 ms | 14.4 s | 35.1 s | 299 / 376 | 324 / 356 | 318 / 380 | 204 |
| deit_base_patch16_224.fb_in1k | 224x224 | 2.98 ms | 2.09 ms | 1.12 ms | 1.64 ms | 7.27 s | 28.3 s | 203 / 386 | 238 / 266 | 300 / 370 | 196 |
| densenet121.tv_in1k | 224x224 | 9.41 ms | 5.49 ms | 0.686 ms | 1.28 ms | 20.4 s | 33.6 s | 53 / 90 | 89 / 112 | 83 / 132 | 36 |
| dinov2_vitl14_lc | 224x224 | 5.43 ms | 5.09 ms | 4.22 ms | 4.93 ms | 12.1 s | 38.2 s | 623 / 1228 | 660 / 688 | 890 / 950 | 618 |
| efficientnet_b0.tv_in1k | 224x224 | 4.98 ms | 2.88 ms | 0.536 ms | 0.769 ms | 9.12 s | 25.8 s | 48 / 96 | 80 / 96 | 78 / 118 | 38 |
| efficientnet_v2_s.tv_in1k | 224x224 | 10.2 ms | 5.39 ms | 1.35 ms | 1.89 ms | 12.1 s | 32.0 s | 81 / 106 | 116 / 146 | 114 / 188 | 74 |
| efficientvit_b2.r288_in1k | 224x224 | 11.9 ms | 6.91 ms | 0.790 ms | 1.10 ms | 11.0 s | 35.3 s | 85 / 124 | 120 / 164 | 120 / 212 | 74 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 224x224 | 18.0 ms | 14.9 ms | 12.4 ms | 14.1 ms | 21.0 s | 43.5 s | 648 / 1206 | 674 / 684 | 1236 / 1386 | 642 |
| fastvit_sa24.apple_dist_in1k | 224x224 | 6.88 ms | 3.87 ms | 0.808 ms | 1.39 ms | 9.39 s | 44.9 s | 81 / 108 | 113 / 148 | 112 / 190 | 72 |
| inception_v3.tv_in1k | 224x224 | 5.21 ms | 3.03 ms | 0.685 ms | 1.30 ms | 8.61 s | 22.4 s | 120 / 166 | 152 / 184 | 150 / 228 | 70 |
| mobilenet_v2.tv2_in1k | 224x224 | 2.95 ms | 1.82 ms | 0.261 ms | 0.439 ms | 6.56 s | 16.3 s | 43 / 70 | 77 / 92 | 75 / 92 | 36 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 3.80 ms | 2.19 ms | 0.393 ms | 0.545 ms | 10.2 s | 36.5 s | 46 / 76 | 79 / 90 | 78 / 114 | 36 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 1.77 ms | 1.21 ms | 0.176 ms | 0.370 ms | 4.59 s | 14.1 s | 44 / 72 | 77 / 90 | 76 / 114 | 36 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 3.91 ms | 2.48 ms | 0.345 ms | 0.641 ms | 6.98 s | 19.6 s | 54 / 86 | 87 / 102 | 87 / 124 | 36 |
| regnety_160.tv2_in1k | 224x224 | 8.34 ms | 5.42 ms | 1.78 ms | 1.86 ms | 8.01 s | 28.1 s | 208 / 250 | 236 / 290 | 229 / 316 | 172 |
| resnet18.tv_in1k | 224x224 | 1.48 ms | 1.04 ms | 0.220 ms | 0.459 ms | 4.31 s | 14.0 s | 61 / 90 | 91 / 108 | 90 / 150 | 34 |
| resnet50.tv2_in1k | 224x224 | 3.15 ms | 2.10 ms | 0.466 ms | 0.826 ms | 8.31 s | 24.0 s | 89 / 142 | 119 / 138 | 116 / 162 | 70 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 3.71 ms | 1.98 ms | 0.329 ms | 0.787 ms | 7.73 s | 16.9 s | 41 / 72 | 71 / 88 | 87 / 116 | 32 |
| squeezenet1_1.tv_in1k | 224x224 | 1.82 ms | 1.13 ms | 0.214 ms | 0.359 ms | 4.49 s | 9.10 s | 41 / 72 | 39 / 48 | 8 / 12 | 34 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 224x224 | 10.8 ms | 4.19 ms | 1.48 ms | 2.15 ms | 26.9 s | 64.2 s | 213 / 394 | 243 / 270 | 246 / 300 | 202 |
| vgg16.tv_in1k | 224x224 | 1.99 ms | 1.86 ms | 1.17 ms | 1.50 ms | 4.08 s | 15.9 s | 313 / 372 | 342 / 368 | 332 / 388 | 302 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 224x224 | 3.02 ms | 2.10 ms | 1.14 ms | 1.65 ms | 7.26 s | 28.6 s | 205 / 390 | 238 / 266 | 300 / 370 | 196 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 224x224 | 13.1 ms | 13.3 ms | 11.4 ms | 12.8 ms | 14.2 s | 60.7 s | 879 / 1708 | 909 / 944 | 1216 / 1346 | 848 |
| dinov2-s-linear-ade20k | 1024x2048 | 18.3 ms | 20.7 ms | 40.3 ms | 12.4 ms | 10.6 s | 36.2 s | 642 / 696 | 1331 / 1376 | 645 / 676 | 1794 |
| eomt-l-dinov2-ade20k | 1024x2048 | 50.8 ms | 49.4 ms | 54.3 ms | 35.0 ms | 16.5 s | 52.9 s | 1515 / 1584 | 1585 / 1624 | 1515 / 1584 | 1798 |
| hrnet-w48-fcn-ade20k | 1024x2048 | 26.9 ms | 23.9 ms | 20.9 ms | 11.9 ms | 35.7 s | 29.6 s | 424 / 490 | 421 / 448 | 274 / 430 | 888 |
| mask2former-swinl-ade20k | 1024x2048 | 103 ms | fails: Inductor: Triton kernel does not compile (torch bug) | 55.6 ms | 60.4 ms | – | 129 s | 1164 / 1400 | – | 1164 / 1400 | 1816 |
| oneformer-swinl-ade20k | 1024x2048 | 139 ms | fails: Inductor: Triton kernel does not compile (torch bug) | 83.3 ms | 100.0 ms | – | 140 s | 2054 / 2476 | – | 2054 / 2476 | 3258 |
| sapiens-seg-0.3b-goliath | 1024x2048 | 132 ms | 118 ms | 156 ms | 92.6 ms | 12.5 s | 52.9 s | 1643 / 1762 | 1373 / 1462 | 1643 / 1762 | 1564 |
| segformer-b0-ade20k | 1024x2048 | 11.4 ms | 9.27 ms | 5.51 ms | 3.48 ms | 12.3 s | 60.9 s | 339 / 376 | 336 / 356 | 148 / 238 | 792 |
| sam-vit-b | 1024x1024 | 48.3 ms | 40.9 ms | 27.3 ms (image encoder only) | 37.0 ms | 14.5 s | 36.4 s | 378 / 420 | 362 / 400 | 377 / 420 | 322 |
| sam2.1-hiera-tiny | 1024x1024 | 16.6 ms | 14.2 ms | 8.56 ms (image encoder only) | 13.9 ms | 13.4 s | 34.1 s | 304 / 382 | 270 / 322 | 316 / 362 | 266 |
| sam3 | 1024x1024 | 179 ms | 168 ms | 89.5 ms (image encoder only) | 116 ms | 20.0 s | 54.2 s | 1922 / 2004 | 1910 / 2004 | 2019 / 2264 | 1122 |
| detr-r50-coco | 800x1344 | 12.4 ms | 9.27 ms | 4.40 ms | 9.93 ms | 21.7 s | 33.4 s | 280 / 332 | 233 / 292 | 218 / 272 | 182 |
| dfine-n-coco | 800x1344 | 10.3 ms | 5.43 ms | 0.942 ms | 2.72 ms | 26.7 s | 51.5 s | 88 / 116 | 101 / 118 | 124 / 202 | 56 |
| faster-rcnn-r50-fpn-3x-coco | 800x1344 | 22.8 ms | 18.9 ms | 5.35 ms (backbone + FPN only) | 19.5 ms | 26.0 s | 15.6 s | 390 / 490 | 413 / 438 | 391 / 450 | 258 |
| rtdetrv2-s-coco | 800x1344 | 8.04 ms | 4.22 ms | 1.81 ms | 4.19 ms | 37.0 s | 42.2 s | 134 / 166 | 152 / 182 | 155 / 250 | 98 |
| dinov2-vitb14 | 224x224 | 2.92 ms | 2.20 ms | 1.30 ms | 1.48 ms | 9.52 s | 25.2 s | 205 / 226 | 240 / 266 | 237 / 290 | 198 |
| dinov3-convnext-tiny | 224x224 | 3.04 ms | 1.96 ms | 0.681 ms | 1.18 ms | 7.76 s | 22.8 s | 93 / 108 | 123 / 148 | 86 / 112 | 72 |
| dinov3-vits16 | 224x224 | 4.67 ms | 2.18 ms | 0.564 ms | 0.903 ms | 8.81 s | 25.0 s | 77 / 80 | 109 / 138 | 77 / 106 | 64 |
| mapanything-apache | 336x518 | 636 ms | 596 ms | 367 ms | 544 ms | 46.2 s | 137 s | 3697 / 4092 | 3347 / 3792 | 3697 / 4072 | 3880 |
| mvsanywhere | 448x672 | 93.2 ms | 70.8 ms | fails: TensorRT: NaN output (onnxruntime on the same ONNX is right) | 35.6 ms | 153 s | – | 931 / 1226 | 818 / 860 | 761 / 884 | – |
| patchmatchnet | 1536x1024 | 202 ms | 175 ms | 51.0 ms | 29.6 ms | 29.4 s | 82.1 s | 1806 / 2096 | 1798 / 1888 | 740 / 856 | 844 |
| vggt-1b | 336x518 | 651 ms | 557 ms | 424 ms | 458 ms | 74.3 s | 199 s | 3691 / 4674 | 3611 / 3856 | 3702 / 4654 | 3642 |
| vggt-omega-1b-512 | 336x518 | 726 ms | 589 ms | – | 464 ms | 65.7 s | – | 3421 / 3518 | 3082 / 3154 | 3422 / 3518 | – |
<!-- compile:end -->

## Original code

The original columns are the authors' code with the same checkpoint and settings, in fp16 where the code supports it and
the output stays finite, else fp32 (marked `(fp32)`). Original fp32 is the same code at its default fp32 precision,
where that differs. The kernels speed-up is against the faster of original fp16 and fp32. Notes per row are on the model
pages. A dash means not measured or no runnable original. Where kernels was not timed at the original's size, the
overview shows the speed-up against portable instead.

## Fixed-size speed and peak memory, all models

Every model, `portable` and `kernels`, on a synthetic input of the size in the column header, median of 10 warm runs
(3 for calls over 2 s). Cells: median ms / peak allocated MiB / peak reserved MiB, weights included. OOM = out of memory
on the 16 GB GPU. Fixed-input models (classification, DINO features) have their own table at the end, at their native size. The 12 MP column probes the largest size
that fits. Depth-anything-v2, midas-dpt-beit-large-512 and zoedepth-nk resize to their working resolution: time depends on the aspect ratio, not the input pixels (12 MP is 4:3, so it is faster than 1080p at 16:9). Measured 2026-10/11 on main at the time; the fixed-input table and the GREAT-IGEV and IGEV++ rows are from 2026-11-03, and the other columns have not been re-measured since.

<!-- sizes:begin (tools/bench_sizes.py docs) -->
Cells: median ms / peak allocated MiB / peak reserved MiB (incl. weights). OOM = out of memory; SKIPPED =
a smaller size already failed; TIMEOUT = the call did not finish in a 170 s GPU-lock step; - = not run (the
size is outside the model's range).

| model | mode | 480x640 | 720x1280 | 1080x1920 | 12 MP 3000x4000 |
|---|---|---|---|---|---|
| aanet-plus-sceneflow | portable | 58.4 ms / 1109 / 1292 | 170.2 ms / 3400 / 3858 | 389.0 ms / 7214 / 8226 | SKIPPED |
| aanet-plus-sceneflow | kernels | 28.8 ms / 178 / 244 | 53.9 ms / 501 / 696 | 128.5 ms / 1049 / 1430 | 793.2 ms / 5752 / 8230 |
| aanet-sceneflow | portable | 40.1 ms / 300 / 340 | 94.2 ms / 781 / 830 | 231.2 ms / 1711 / 1820 | 1591.3 ms / 9603 / 12836 |
| aanet-sceneflow | kernels | 25.9 ms / 148 / 196 | 59.2 ms / 344 / 602 | 152.4 ms / 728 / 1252 | 967.9 ms / 3967 / 6854 |
| bridgedepth-rvc-pretrain | portable | 76.7 ms / 1727 / 1798 | 247.2 ms / 2359 / 2628 | 676.5 ms / 3562 / 4102 | 7918.7 ms (n=3) / 13620 / 14590 |
| bridgedepth-rvc-pretrain | kernels | 62.2 ms / 1147 / 1266 | 200.9 ms / 1764 / 2096 | 534.7 ms / 2939 / 3514 | 5573.1 ms (n=3) / 12950 / 13592 |
| bridgedepth-sf | portable | 73.2 ms / 1676 / 1748 | 242.8 ms / 2308 / 2560 | 668.4 ms / 3511 / 4032 | 7849.6 ms (n=3) / 13569 / 14540 |
| bridgedepth-sf | kernels | 59.1 ms / 1096 / 1214 | 196.3 ms / 1714 / 2044 | 526.9 ms / 2888 / 3462 | 5498.6 ms (n=3) / 12898 / 13540 |
| c-fast-foundationstereo | portable | 64.6 ms / 600 / 784 | 193.1 ms / 1626 / 2144 | 464.9 ms / 3485 / 4540 | OOM |
| c-fast-foundationstereo | kernels | 38.5 ms / 306 / 386 | 85.9 ms / 762 / 960 | 182.0 ms / 1582 / 2060 | 1072.8 ms / 8699 / 11372 |
| c-foundationstereo-s | portable | 681.8 ms / 2075 / 3130 | 2218.0 ms (n=3) / 4355 / 6278 | 4704.5 ms (n=3) / 8457 / 12000 | SKIPPED |
| c-foundationstereo-s | kernels | 291.3 ms / 1262 / 1430 | 838.6 ms / 2380 / 3312 | 1826.4 ms / 4137 / 6164 | SKIPPED |
| fast-foundationstereo | portable | 64.6 ms / 600 / 784 | 193.1 ms / 1626 / 2144 | 464.8 ms / 3485 / 4540 | OOM |
| fast-foundationstereo | kernels | 38.7 ms / 306 / 386 | 85.8 ms / 762 / 960 | 181.9 ms / 1582 / 2060 | 1073.0 ms / 8699 / 11372 |
| cas-gwcnet-c-sceneflow | portable | 61.3 ms / 465 / 738 | 195.5 ms / 1386 / 2160 | 452.8 ms / 3046 / 4720 | OOM |
| cas-gwcnet-c-sceneflow | kernels | 44.8 ms / 442 / 568 | 135.5 ms / 1242 / 1586 | 316.1 ms / 2692 / 3408 | OOM |
| cas-gwcnet-gc-sceneflow | portable | 86.7 ms / 533 / 1166 | 281.6 ms / 1588 / 3308 | 647.4 ms / 3494 / 7262 | OOM |
| cas-gwcnet-gc-sceneflow | kernels | 66.9 ms / 510 / 956 | 213.7 ms / 1449 / 2606 | 493.8 ms / 3147 / 5634 | OOM |
| cfnet-sceneflow | portable | 73.6 ms / 1070 / 1588 | 284.0 ms / 2640 / 4342 | 758.8 ms / 4508 / 5956 | OOM |
| cfnet-sceneflow | kernels | 59.3 ms / 1118 / 1618 | 229.9 ms / 2778 / 4362 | 634.1 ms / 4801 / 5976 | OOM |
| coex-sceneflow | portable | 11.2 ms / 225 / 312 | 28.5 ms / 671 / 890 | 70.0 ms / 1485 / 1916 | 554.7 ms / 8660 / 10964 |
| coex-sceneflow | kernels | 4.3 ms / 103 / 322 | 13.7 ms / 256 / 400 | 32.9 ms / 559 / 804 | 211.1 ms / 3190 / 4410 |
| crestereo-eth3d | portable | 142.5 ms / 214 / 284 | 427.8 ms / 556 / 740 | 1022.9 ms / 1172 / 1568 | 6466.7 ms (n=3) / 6508 / 8990 |
| crestereo-eth3d | kernels | 61.8 ms / 216 / 292 | 145.2 ms / 552 / 792 | 311.2 ms / 1162 / 1638 | 1767.7 ms / 6427 / 9362 |
| defom-stereo-vitl-sceneflow | portable | 263.9 ms / 2070 / 2368 | 860.4 ms / 3248 / 4044 | 2169.2 ms (n=3) / 5378 / 7082 | OOM |
| defom-stereo-vitl-sceneflow | kernels | 120.5 ms / 1357 / 1702 | 376.5 ms / 2275 / 3190 | 936.5 ms / 2991 / 3578 | 5950.0 ms (n=3) / 11923 / 15526 |
| dlnr-middlebury | portable | 81.8 ms / 623 / 774 | 235.4 ms / 1651 / 2134 | 548.9 ms / 3617 / 4726 | OOM |
| dlnr-middlebury | kernels | 52.8 ms / 372 / 470 | 126.7 ms / 815 / 1070 | 297.2 ms / 1617 / 1958 | 1775.1 ms / 8572 / 12080 |
| dlnr-sceneflow | portable | 81.8 ms / 623 / 774 | 235.2 ms / 1651 / 2134 | 549.1 ms / 3617 / 4726 | OOM |
| dlnr-sceneflow | kernels | 52.9 ms / 372 / 470 | 126.8 ms / 815 / 1070 | 297.4 ms / 1617 / 1958 | 1775.1 ms / 8572 / 12080 |
| fast-acvnet-generalization | portable | 13.9 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.3 ms / 890 / 1418 | 492.1 ms / 5077 / 8074 |
| fast-acvnet-generalization | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.7 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-kitti12 | portable | 14.0 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-kitti12 | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-kitti15 | portable | 13.9 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-kitti15 | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-plus-generalization | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.6 ms / 5054 / 8076 |
| fast-acvnet-plus-generalization | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.5 ms / 5743 / 7342 |
| fast-acvnet-plus-kitti12 | portable | 14.4 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.3 ms / 5054 / 8076 |
| fast-acvnet-plus-kitti12 | kernels | 7.7 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.4 ms / 5743 / 7342 |
| foundationstereo-vit-l | portable | 577.7 ms / 1654 / 2216 | 1846.4 ms / 2676 / 3370 | 3776.7 ms (n=3) / 3570 / 5048 | OOM |
| foundationstereo-vit-l | kernels | 320.0 ms / 1296 / 1642 | 961.0 ms / 2232 / 2800 | 1788.5 ms / 3254 / 4506 | 9521.9 ms (n=3) / 12653 / 14398 |
| foundationstereo-vit-s | portable | 481.4 ms / 1042 / 1352 | 1508.2 ms / 2047 / 2678 | 3424.1 ms (n=3) / 2900 / 4592 | OOM |
| foundationstereo-vit-s | kernels | 245.6 ms / 685 / 862 | 687.1 ms / 1604 / 2008 | 1499.8 ms / 2585 / 3440 | 8960.9 ms (n=3) / 12058 / 13800 |
| gwcnet-g-sceneflow | portable | 69.2 ms / 418 / 586 | 219.1 ms / 1254 / 1776 | 502.0 ms / 2764 / 3874 | 2911.8 ms (n=3) / 13647 / 15104 |
| gwcnet-g-sceneflow | kernels | 50.9 ms / 393 / 526 | 156.0 ms / 1112 / 1590 | 358.5 ms / 2412 / 3464 | OOM |
| gwcnet-gc-sceneflow | portable | 71.2 ms / 418 / 442 | 224.5 ms / 1255 / 1688 | 514.1 ms / 2766 / 3680 | 2981.7 ms (n=3) / 13924 / 14002 |
| gwcnet-gc-sceneflow | kernels | 54.9 ms / 394 / 468 | 167.5 ms / 1114 / 1534 | 385.9 ms / 2415 / 3300 | OOM |
| hitnet-eth3d | portable | 18.1 ms / 221 / 336 | 40.2 ms / 659 / 1004 | 92.6 ms / 1134 / 2438 | 579.4 ms / 6558 / 11110 |
| hitnet-eth3d | kernels | 5.2 ms / 140 / 470 | 14.5 ms / 392 / 482 | 30.8 ms / 836 / 1004 | 181.7 ms / 4814 / 5758 |
| hitnet-middlebury-d400 | portable | 35.8 ms / 602 / 838 | 108.8 ms / 1067 / 2222 | 251.7 ms / 2392 / 3814 | OOM |
| hitnet-middlebury-d400 | kernels | 11.7 ms / 266 / 966 | 33.1 ms / 759 / 978 | 74.2 ms / 1695 / 2248 | 426.2 ms / 9604 / 12664 |
| igev-stereo-eth3d | portable | 148.4 ms / 280 / 352 | 442.8 ms / 768 / 1034 | 1089.2 ms / 1689 / 2272 | 6940.4 ms (n=3) / 10547 / 14300 |
| igev-stereo-eth3d | kernels | 70.2 ms / 263 / 372 | 177.6 ms / 698 / 1000 | 391.1 ms / 1506 / 2044 | 2328.3 ms (n=3) / 6315 / 8190 |
| igev-stereo-kitti12 | portable | 148.6 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.5 ms / 1689 / 2272 | 6944.6 ms (n=3) / 10547 / 14300 |
| igev-stereo-kitti12 | kernels | 70.1 ms / 263 / 372 | 177.7 ms / 698 / 1000 | 391.5 ms / 1506 / 2044 | 2330.4 ms (n=3) / 6315 / 8190 |
| igev-stereo-kitti15 | portable | 148.3 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.7 ms / 1689 / 2272 | 6944.1 ms (n=3) / 10547 / 14300 |
| igev-stereo-kitti15 | kernels | 70.3 ms / 263 / 372 | 177.8 ms / 698 / 1000 | 391.3 ms / 1506 / 2044 | 2330.4 ms (n=3) / 6315 / 8190 |
| igev-stereo-middlebury | portable | 148.4 ms / 280 / 352 | 442.4 ms / 768 / 1034 | 1089.3 ms / 1689 / 2272 | 6939.3 ms (n=3) / 10547 / 14300 |
| igev-stereo-middlebury | kernels | 70.2 ms / 263 / 372 | 177.9 ms / 698 / 1000 | 391.5 ms / 1506 / 2044 | 2328.7 ms (n=3) / 6315 / 8190 |
| igev-stereo-sceneflow | portable | 148.6 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.8 ms / 1689 / 2272 | 6943.4 ms (n=3) / 10547 / 14300 |
| igev-stereo-sceneflow | kernels | 70.2 ms / 263 / 372 | 178.2 ms / 698 / 1000 | 392.0 ms / 1506 / 2044 | 2334.4 ms (n=3) / 6315 / 8190 |
| las2-h | portable | 29.8 ms / 269 / 328 | 69.1 ms / 668 / 904 | 174.8 ms / 1386 / 2168 | 1408.7 ms / 8708 / 12018 |
| las2-h | kernels | 10.2 ms / 218 / 596 | 28.5 ms / 604 / 1636 | 73.6 ms / 1344 / 1648 | 512.5 ms / 9695 / 12606 |
| las2-l | portable | 25.5 ms / 286 / 328 | 51.1 ms / 653 / 700 | 110.1 ms / 1317 / 1404 | 824.0 ms / 7201 / 7598 |
| las2-l | kernels | 6.6 ms / 129 / 270 | 16.8 ms / 241 / 676 | 38.6 ms / 422 / 588 | 254.6 ms / 2174 / 2896 |
| las2-m | portable | 18.9 ms / 219 / 252 | 35.4 ms / 586 / 648 | 73.9 ms / 1249 / 1358 | 602.8 ms / 7135 / 7584 |
| las2-m | kernels | 4.0 ms / 96 / 258 | 9.2 ms / 187 / 618 | 21.5 ms / 346 / 516 | 146.5 ms / 1870 / 2684 |
| las2-s | portable | 16.1 ms / 202 / 234 | 31.0 ms / 569 / 632 | 64.4 ms / 1232 / 1342 | 532.1 ms / 7119 / 7566 |
| las2-s | kernels | 3.0 ms / 88 / 246 | 6.9 ms / 179 / 602 | 16.6 ms / 338 / 500 | 109.3 ms / 1859 / 2598 |
| mocha-v1-sceneflow | portable | 231.1 ms / 500 / 662 | 628.3 ms / 1078 / 1438 | 1484.8 ms / 2164 / 2688 | 10361.4 ms (n=3) / 12466 / 15520 |
| mocha-v1-sceneflow | kernels | 135.4 ms / 495 / 694 | 317.0 ms / 1055 / 1338 | 718.5 ms / 2092 / 2484 | 4396.8 ms (n=3) / 11664 / 14958 |
| mocha-v2-sceneflow | portable | 227.0 ms / 500 / 662 | 613.9 ms / 1078 / 1438 | 1451.9 ms / 2164 / 2688 | 10135.5 ms (n=3) / 12464 / 15164 |
| mocha-v2-sceneflow | kernels | 133.6 ms / 494 / 654 | 312.9 ms / 1054 / 1456 | 709.3 ms / 2091 / 2482 | 4316.6 ms (n=3) / 11664 / 14404 |
| monster-mix-all | portable | 284.1 ms / 1089 / 1184 | 894.5 ms / 1732 / 2154 | 2291.3 ms (n=3) / 2935 / 3668 | OOM |
| monster-mix-all | kernels | 127.2 ms / 1112 / 1250 | 379.5 ms / 1752 / 2288 | 956.8 ms / 2939 / 3460 | OOM |
| monster-sceneflow | portable | 284.1 ms / 1089 / 1184 | 894.5 ms / 1732 / 2154 | 2290.9 ms (n=3) / 2935 / 3668 | OOM |
| monster-sceneflow | kernels | 127.1 ms / 1112 / 1250 | 379.4 ms / 1752 / 2288 | 956.9 ms / 2939 / 3460 | OOM |
| monster-plusplus-mix-all | portable | 294.3 ms / 1138 / 1246 | 928.8 ms / 1880 / 2318 | 2370.0 ms (n=3) / 3269 / 3812 | OOM |
| monster-plusplus-mix-all | kernels | 132.9 ms / 1130 / 1270 | 396.3 ms / 1804 / 2344 | 998.1 ms / 3051 / 3724 | OOM |
| nmrf-sceneflow | portable | 35.9 ms / 390 / 454 | 124.3 ms / 953 / 1094 | 315.1 ms / 2040 / 2402 | 2021.2 ms (n=3) / 11263 / 13666 |
| nmrf-sceneflow | kernels | 31.7 ms / 247 / 298 | 112.1 ms / 626 / 800 | 286.8 ms / 1356 / 1814 | 1834.3 ms / 7543 / 10866 |
| psmnet-sceneflow | portable | 69.0 ms / 426 / 606 | 206.2 ms / 1256 / 1780 | 479.2 ms / 2834 / 3966 | OOM |
| psmnet-sceneflow | kernels | 49.6 ms / 415 / 518 | 140.3 ms / 1108 / 1414 | 329.2 ms / 2459 / 3104 | OOM |
| raft-stereo-eth3d | portable | 143.6 ms / 320 / 376 | 433.7 ms / 864 / 1058 | 1036.0 ms / 1852 / 2258 | 6289.1 ms (n=3) / 10406 / 12814 |
| raft-stereo-eth3d | kernels | 57.8 ms / 201 / 214 | 166.5 ms / 569 / 590 | 370.4 ms / 1233 / 1264 | 2138.5 ms (n=3) / 7295 / 7436 |
| raft-stereo-middlebury | portable | 143.7 ms / 320 / 376 | 433.7 ms / 864 / 1058 | 1035.8 ms / 1852 / 2258 | 6287.7 ms (n=3) / 10406 / 12814 |
| raft-stereo-middlebury | kernels | 57.8 ms / 201 / 214 | 165.9 ms / 569 / 590 | 370.5 ms / 1233 / 1264 | 2139.9 ms (n=3) / 7295 / 7436 |
| raft-stereo-sceneflow | portable | 143.7 ms / 320 / 376 | 433.6 ms / 864 / 1058 | 1035.8 ms / 1852 / 2258 | 6286.9 ms (n=3) / 10406 / 12814 |
| raft-stereo-sceneflow | kernels | 57.6 ms / 201 / 214 | 166.0 ms / 569 / 590 | 370.4 ms / 1233 / 1264 | 2138.4 ms (n=3) / 7295 / 7436 |
| rt-monster-plusplus-mix-all | portable | 31.2 ms / 609 / 720 | 67.1 ms / 1625 / 1968 | 168.0 ms / 3478 / 4148 | OOM |
| rt-monster-plusplus-mix-all | kernels | 18.3 ms / 630 / 1934 | 57.5 ms / 1660 / 1998 | 140.1 ms / 3545 / 4188 | OOM |
| s2m2-l | portable | 182.7 ms / 1304 / 1460 | 618.6 ms / 2499 / 2956 | 1459.4 ms / 4660 / 5680 | OOM |
| s2m2-l | kernels | 83.3 ms / 719 / 782 | 228.8 ms / 988 / 1224 | 534.8 ms / 1530 / 2250 | 4189.4 ms (n=3) / 6355 / 9246 |
| s2m2-m | portable | 114.5 ms / 780 / 904 | 402.5 ms / 1690 / 1954 | 973.5 ms / 3330 / 3896 | OOM |
| s2m2-m | kernels | 55.3 ms / 395 / 492 | 153.2 ms / 655 / 940 | 355.5 ms / 1119 / 1760 | 2793.3 ms (n=3) / 5151 / 7876 |
| s2m2-s | portable | 56.0 ms / 456 / 536 | 202.3 ms / 1141 / 1450 | 511.1 ms / 2427 / 3054 | OOM |
| s2m2-s | kernels | 32.7 ms / 212 / 294 | 70.2 ms / 452 / 692 | 161.9 ms / 887 / 1534 | 1203.3 ms / 4637 / 7166 |
| s2m2-xl | portable | 338.0 ms / 2476 / 2710 | 1112.0 ms / 4248 / 4842 | 2594.0 ms (n=3) / 7447 / 8684 | SKIPPED |
| s2m2-xl | kernels | 140.8 ms / 1441 / 1644 | 414.8 ms / 1748 / 2040 | 956.5 ms / 2493 / 3446 | 6933.0 ms (n=3) / 9200 / 11578 |
| selective-igev-eth3d-finetune | portable | 190.5 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.2 ms / 1690 / 2030 | 8918.7 ms (n=3) / 10549 / 14320 |
| selective-igev-eth3d-finetune | kernels | 82.5 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 472.5 ms / 1502 / 2046 | 2806.7 ms (n=3) / 6928 / 8800 |
| selective-igev-eth3d-train | portable | 190.5 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.4 ms / 1690 / 2030 | 8919.8 ms (n=3) / 10549 / 14320 |
| selective-igev-eth3d-train | kernels | 82.4 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 472.6 ms / 1502 / 2046 | 2807.0 ms (n=3) / 6928 / 8800 |
| selective-igev-kitti12 | portable | 190.6 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.5 ms / 1690 / 2030 | 8917.7 ms (n=3) / 10549 / 14320 |
| selective-igev-kitti12 | kernels | 82.3 ms / 260 / 368 | 212.9 ms / 693 / 1010 | 471.9 ms / 1502 / 2046 | 2803.4 ms (n=3) / 6928 / 8800 |
| selective-igev-kitti15 | portable | 190.6 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.8 ms / 1690 / 2030 | 8918.8 ms (n=3) / 10549 / 14320 |
| selective-igev-kitti15 | kernels | 82.5 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 471.9 ms / 1502 / 2046 | 2803.9 ms (n=3) / 6928 / 8800 |
| selective-igev-middlebury-finetune | portable | 190.6 ms / 282 / 362 | 575.6 ms / 769 / 1036 | 1420.8 ms / 1690 / 2030 | 8923.1 ms (n=3) / 10549 / 14320 |
| selective-igev-middlebury-finetune | kernels | 82.3 ms / 260 / 368 | 212.6 ms / 693 / 1010 | 472.1 ms / 1502 / 2046 | 2804.4 ms (n=3) / 6928 / 8800 |
| waft-stereo-b | portable | SKIPPED | SKIPPED | SKIPPED | SKIPPED |
| waft-stereo-b | kernels | 62.9 ms / 676 / 908 | 214.9 ms / 1310 / 1880 | 607.1 ms / 2523 / 3712 | OOM |
| waft-stereo-s | portable | 83.4 ms / 941 / 1146 | 357.7 ms / 2165 / 3356 | 1234.1 ms / 5368 / 8448 | SKIPPED |
| waft-stereo-s | kernels | 54.4 ms / 381 / 516 | 148.0 ms / 760 / 1120 | 441.0 ms / 1473 / 2150 | 6423.1 ms (n=3) / 7556 / 11364 |
| da3-giant-1.1 | portable | 112.6 ms / 5032 / 5332 | 90.1 ms / 4930 / 5150 | 90.3 ms / 4943 / 5150 | 118.5 ms / 5206 / 5524 |
| da3-giant-1.1 | kernels | 49.5 ms / 2630 / 3032 | 36.0 ms / 2562 / 2854 | 36.7 ms / 2575 / 2886 | 54.5 ms / 3011 / 3386 |
| da3-large-1.1 | portable | 45.6 ms / 1733 / 1992 | 37.8 ms / 1639 / 1814 | 38.8 ms / 1651 / 1826 | 51.5 ms / 1938 / 2214 |
| da3-large-1.1 | kernels | 18.3 ms / 967 / 1218 | 14.2 ms / 901 / 1068 | 14.8 ms / 914 / 1108 | 24.1 ms / 1354 / 1608 |
| da3metric-large | portable | 32.3 ms / 1542 / 1652 | 25.4 ms / 1488 / 1576 | 26.0 ms / 1502 / 1600 | 38.1 ms / 1896 / 2184 |
| da3metric-large | kernels | 15.7 ms / 805 / 880 | 12.1 ms / 782 / 838 | 12.8 ms / 795 / 858 | 21.5 ms / 1266 / 1570 |
| da3mono-large | portable | 32.2 ms / 1542 / 1652 | 25.2 ms / 1488 / 1576 | 25.9 ms / 1502 / 1600 | 38.0 ms / 1896 / 2184 |
| da3mono-large | kernels | 15.6 ms / 805 / 880 | 12.0 ms / 782 / 838 | 12.6 ms / 795 / 858 | 21.3 ms / 1266 / 1570 |
| da3nested-giant-large-1.1 | portable | 145.3 ms / 6309 / 6624 | 114.9 ms / 6206 / 6432 | 115.5 ms / 6218 / 6440 | 151.5 ms / 6482 / 6810 |
| da3nested-giant-large-1.1 | kernels | 65.6 ms / 3276 / 3658 | 48.0 ms / 3206 / 3490 | 48.7 ms / 3220 / 3528 | 71.2 ms / 3658 / 4168 |
| depth-anything-v2-b | portable | 18.8 ms / 643 / 776 | 26.2 ms / 733 / 876 | 26.3 ms / 745 / 908 | 20.0 ms / 787 / 924 |
| depth-anything-v2-b | kernels | 9.5 ms / 325 / 408 | 13.6 ms / 376 / 466 | 13.7 ms / 390 / 486 | 10.7 ms / 606 / 622 |
| depth-anything-v2-l | portable | 55.9 ms / 1752 / 1946 | 78.6 ms / 1915 / 2170 | 78.8 ms / 1928 / 2180 | 57.1 ms / 1885 / 2216 |
| depth-anything-v2-l | kernels | 28.3 ms / 875 / 1000 | 38.7 ms / 965 / 1104 | 38.8 ms / 977 / 1112 | 29.3 ms / 1058 / 1174 |
| depth-anything-v2-s | portable | 8.1 ms / 262 / 326 | 10.9 ms / 314 / 376 | 11.0 ms / 329 / 422 | 9.5 ms / 509 / 536 |
| depth-anything-v2-s | kernels | 4.0 ms / 129 / 166 | 5.7 ms / 164 / 218 | 5.8 ms / 177 / 214 | 5.1 ms / 462 / 480 |
| depth-pro | portable | 558.6 ms / 3317 / 4044 | 558.7 ms / 3323 / 4080 | 559.0 ms / 3337 / 4084 | 562.3 ms / 3451 / 4120 |
| depth-pro | kernels | 272.9 ms / 3621 / 4372 | 272.1 ms / 3628 / 4388 | 271.8 ms / 3641 / 4386 | 274.9 ms / 3754 / 4500 |
| marigold-depth-v1-1 | portable | 455.3 ms / 2898 / 3650 | 351.7 ms / 2648 / 3306 | 351.8 ms / 2661 / 3378 | 458.1 ms / 3033 / 3844 |
| marigold-depth-v1-1 | kernels | 433.0 ms / 2636 / 3464 | 324.3 ms / 2452 / 3070 | 324.9 ms / 2465 / 3142 | 435.6 ms / 2770 / 3552 |
| metric3d-vit-large | portable | 207.0 ms / 2219 / 2422 | 207.0 ms / 2228 / 2440 | 207.3 ms / 2242 / 2428 | 211.9 ms / 2401 / 2926 |
| metric3d-vit-large | kernels | 158.4 ms / 1471 / 1796 | 158.4 ms / 1478 / 1812 | 158.8 ms / 1492 / 1822 | 162.9 ms / 1616 / 2022 |
| midas-dpt-beit-large-512 | portable | 79.3 ms / 1729 / 1880 | 141.1 ms / 1870 / 2052 | 141.2 ms / 1883 / 2080 | 81.6 ms / 1863 / 1990 |
| midas-dpt-beit-large-512 | kernels | 52.1 ms / 922 / 1048 | 96.0 ms / 1012 / 1220 | 96.0 ms / 1025 / 1280 | 53.1 ms / 1077 / 1178 |
| moge-2-vitl-normal | portable | 175.5 ms / 2275 / 2508 | 178.8 ms / 2286 / 2518 | 179.2 ms / 2299 / 2528 | 191.5 ms / 2410 / 2562 |
| moge-2-vitl-normal | kernels | 74.3 ms / 1435 / 1572 | 77.5 ms / 1444 / 1588 | 76.0 ms / 1459 / 1644 | 90.5 ms / 1755 / 2174 |
| moge-2-vits-normal | portable | 88.6 ms / 1123 / 1418 | 85.4 ms / 1132 / 1392 | 86.7 ms / 1146 / 1446 | 98.6 ms / 1257 / 1464 |
| moge-2-vits-normal | kernels | 33.9 ms / 818 / 970 | 33.1 ms / 826 / 976 | 31.0 ms / 839 / 1010 | 46.9 ms / 1144 / 1566 |
| moge-3-vitl | portable | 657.4 ms / 2580 / 3070 | 633.4 ms / 2577 / 2792 | 620.0 ms / 2588 / 2804 | 655.1 ms / 2701 / 3066 |
| moge-3-vitl | kernels | 386.2 ms / 1570 / 1836 | 374.2 ms / 1557 / 1840 | 362.3 ms / 1566 / 1890 | 390.6 ms / 1838 / 2300 |
| unidepth-v2-vitl14 | portable | 61.2 ms / 1993 / 2266 | 130.9 ms / 2561 / 3154 | 131.1 ms / 2575 / 3150 | 134.4 ms / 2674 / 3266 |
| unidepth-v2-vitl14 | kernels | 32.5 ms / 1011 / 1110 | 69.1 ms / 1214 / 1378 | 69.4 ms / 1227 / 1430 | 72.6 ms / 1334 / 1706 |
| unik3d-vitl | portable | 115.1 ms / 2101 / 2474 | 124.2 ms / 2153 / 2570 | 124.8 ms / 2166 / 2582 | 128.7 ms / 2271 / 2486 |
| unik3d-vitl | kernels | 57.9 ms / 1076 / 1202 | 62.6 ms / 1101 / 1272 | 63.2 ms / 1115 / 1240 | 65.9 ms / 1226 / 1512 |
| zoedepth-nk | portable | 107.5 ms / 1857 / 1966 | 147.2 ms / 2046 / 2184 | 147.6 ms / 2107 / 2220 | 113.0 ms / 2451 / 2658 |
| zoedepth-nk | kernels | 74.9 ms / 1548 / 1802 | 105.6 ms / 1847 / 2138 | 106.2 ms / 1907 / 2230 | 80.3 ms / 2142 / 2486 |
| deeplabv3-r101-d8-ade20k | portable | 33.5 ms / 550 / 784 | 45.5 ms / 978 / 1282 | 52.4 ms / 1650 / 1966 | 103.8 ms / 7442 / 7698 |
| deeplabv3-r101-d8-ade20k | kernels | 20.0 ms / 583 / 718 | 28.5 ms / 1012 / 1196 | 35.5 ms / 1684 / 1880 | 90.5 ms / 7476 / 7708 |
| deeplabv3plus-r101-d8-ade20k | portable | 35.4 ms / 513 / 766 | 48.8 ms / 944 / 1272 | 55.8 ms / 1616 / 1944 | 105.7 ms / 7396 / 7644 |
| deeplabv3plus-r101-d8-ade20k | kernels | 23.8 ms / 546 / 716 | 34.0 ms / 977 / 1222 | 41.0 ms / 1649 / 1902 | 94.4 ms / 7429 / 7668 |
| fcn-r101-d8-ade20k | portable | 28.6 ms / 515 / 720 | 38.9 ms / 943 / 1230 | 46.0 ms / 1614 / 1894 | 99.2 ms / 7407 / 7662 |
| fcn-r101-d8-ade20k | kernels | 17.6 ms / 549 / 692 | 25.5 ms / 977 / 1154 | 32.5 ms / 1649 / 1854 | 88.2 ms / 7441 / 7682 |
| hrnet-w48-fcn-ade20k | portable | 31.0 ms / 523 / 696 | 36.1 ms / 954 / 1194 | 42.8 ms / 1626 / 1836 | 101.5 ms / 7406 / 7632 |
| hrnet-w48-fcn-ade20k | kernels | 23.0 ms / 523 / 652 | 27.7 ms / 955 / 1114 | 35.1 ms / 1627 / 1824 | 93.3 ms / 7406 / 7694 |
| pspnet-r101-d8-ade20k | portable | 31.8 ms / 513 / 810 | 46.8 ms / 941 / 1296 | 53.8 ms / 1613 / 1976 | 102.4 ms / 7406 / 7632 |
| pspnet-r101-d8-ade20k | kernels | 20.5 ms / 548 / 722 | 32.6 ms / 975 / 1178 | 39.5 ms / 1647 / 1838 | 91.2 ms / 7439 / 7668 |
| dinov2-b-linear-ade20k | portable | 22.5 ms / 922 / 1040 | 35.5 ms / 1217 / 1398 | 40.0 ms / 1230 / 1416 | 151.7 ms / 1289 / 1960 |
| dinov2-b-linear-ade20k | kernels | 18.3 ms / 929 / 1080 | 28.4 ms / 1224 / 1438 | 34.1 ms / 1237 / 1456 | 147.9 ms / 1296 / 2000 |
| dinov2-l-linear-ade20k | portable | 56.5 ms / 1330 / 1524 | 86.3 ms / 1624 / 1876 | 91.1 ms / 1638 / 1890 | 184.9 ms / 1697 / 2266 |
| dinov2-l-linear-ade20k | kernels | 39.6 ms / 1338 / 1526 | 62.8 ms / 1632 / 1884 | 67.7 ms / 1645 / 1894 | 168.5 ms / 1704 / 2264 |
| dinov2-s-linear-ade20k | portable | 12.6 ms / 792 / 834 | 20.1 ms / 1088 / 1154 | 25.1 ms / 1101 / 1168 | 141.7 ms / 1160 / 1522 |
| dinov2-s-linear-ade20k | kernels | 11.5 ms / 796 / 854 | 18.5 ms / 1092 / 1168 | 23.5 ms / 1105 / 1174 | 140.4 ms / 1163 / 1774 |
| eomt-l-dinov2-ade20k | portable | 50.3 ms / 1917 / 2484 | 55.4 ms / 2412 / 3506 | 63.2 ms / 3417 / 4834 | 129.2 ms / 14890 / 15016 |
| eomt-l-dinov2-ade20k | kernels | 38.5 ms / 1917 / 2484 | 43.8 ms / 2412 / 3506 | 51.5 ms / 3417 / 4834 | 117.4 ms / 14890 / 15016 |
| eomt-l-dinov3-ade20k | portable | 54.2 ms / 1916 / 2486 | 59.2 ms / 2411 / 3502 | 67.0 ms / 3415 / 4836 | 133.1 ms / 14889 / 15016 |
| eomt-l-dinov3-ade20k | kernels | 42.5 ms / 1916 / 2486 | 47.7 ms / 2411 / 3502 | 55.3 ms / 3415 / 4836 | 121.2 ms / 14889 / 15016 |
| mask2former-swinl-ade20k | portable | 71.9 ms / 1194 / 1912 | 96.1 ms / 1698 / 3182 | 104.1 ms / 2985 / 4500 | 152.8 ms / 14457 / 15480 |
| mask2former-swinl-ade20k | kernels | 45.3 ms / 1194 / 1816 | 60.5 ms / 1699 / 3056 | 68.5 ms / 2986 / 4374 | 126.0 ms / 14457 / 15344 |
| oneformer-swinl-ade20k | portable | 86.2 ms / 1295 / 1940 | 120.8 ms / 2220 / 3960 | 138.9 ms / 3772 / 6702 | OOM |
| oneformer-swinl-ade20k | kernels | 59.0 ms / 1295 / 1858 | 85.4 ms / 2222 / 3854 | 103.3 ms / 3772 / 6642 | OOM |
| sapiens-seg-0.3b-goliath | portable | 131.3 ms / 1621 / 1960 | 132.2 ms / 1627 / 1964 | 133.7 ms / 1641 / 1950 | 146.0 ms / 2254 / 3302 |
| sapiens-seg-0.3b-goliath | kernels | 92.6 ms / 1621 / 1960 | 93.5 ms / 1627 / 1964 | 95.2 ms / 1641 / 1950 | 107.5 ms / 2254 / 3302 |
| sapiens-seg-1b-goliath | portable | 274.9 ms / 3238 / 3718 | 275.7 ms / 3245 / 3734 | 277.0 ms / 3259 / 3730 | 289.7 ms / 3869 / 5076 |
| sapiens-seg-1b-goliath | kernels | 185.7 ms / 3238 / 3718 | 186.6 ms / 3245 / 3734 | 187.6 ms / 3259 / 3730 | 199.5 ms / 3869 / 5076 |
| segformer-b0-ade20k | portable | 7.5 ms / 237 / 290 | 11.9 ms / 454 / 518 | 18.5 ms / 798 / 864 | 74.7 ms / 3888 / 4048 |
| segformer-b0-ade20k | kernels | 7.3 ms / 237 / 290 | 11.6 ms / 454 / 518 | 18.2 ms / 798 / 864 | 74.5 ms / 3888 / 4048 |
| segformer-b5-ade20k | portable | 28.3 ms / 469 / 668 | 36.7 ms / 707 / 958 | 43.4 ms / 1050 / 1304 | 96.1 ms / 4063 / 4380 |
| segformer-b5-ade20k | kernels | 26.6 ms / 470 / 668 | 33.7 ms / 708 / 958 | 40.4 ms / 1051 / 1304 | 94.4 ms / 4064 / 4380 |
| sam-vit-b | portable | 48.6 ms / 367 / 482 | 48.6 ms / 374 / 482 | 49.0 ms / 387 / 484 | - |
| sam-vit-b | kernels | 37.3 ms / 364 / 424 | 37.4 ms / 370 / 404 | 38.1 ms / 384 / 432 | - |
| sam-vit-h | portable | 216.1 ms / 1490 / 1882 | 215.4 ms / 1496 / 1902 | 215.7 ms / 1510 / 1916 | - |
| sam-vit-h | kernels | 143.6 ms / 1459 / 1712 | 144.4 ms / 1466 / 1704 | 145.1 ms / 1479 / 1714 | - |
| sam-vit-l | portable | 117.2 ms / 814 / 1052 | 116.5 ms / 820 / 1060 | 117.5 ms / 834 / 1048 | - |
| sam-vit-l | kernels | 82.0 ms / 812 / 930 | 81.6 ms / 818 / 946 | 83.2 ms / 833 / 922 | - |
| sam2.1-hiera-base-plus | portable | 32.0 ms / 395 / 540 | 31.9 ms / 402 / 558 | 32.1 ms / 415 / 568 | - |
| sam2.1-hiera-base-plus | kernels | 25.1 ms / 408 / 472 | 25.1 ms / 415 / 490 | 25.3 ms / 428 / 472 | - |
| sam2.1-hiera-large | portable | 68.2 ms / 724 / 936 | 68.6 ms / 732 / 948 | 68.3 ms / 745 / 954 | - |
| sam2.1-hiera-large | kernels | 48.7 ms / 742 / 824 | 48.7 ms / 750 / 836 | 49.0 ms / 763 / 842 | - |
| sam2.1-hiera-small | portable | 18.7 ms / 298 / 420 | 19.1 ms / 305 / 422 | 18.9 ms / 319 / 426 | - |
| sam2.1-hiera-small | kernels | 15.4 ms / 310 / 356 | 15.5 ms / 317 / 358 | 15.6 ms / 331 / 346 | - |
| sam2.1-hiera-tiny | portable | 16.7 ms / 285 / 410 | 16.6 ms / 292 / 410 | 16.8 ms / 305 / 426 | - |
| sam2.1-hiera-tiny | kernels | 13.8 ms / 296 / 346 | 13.8 ms / 304 / 346 | 14.0 ms / 317 / 346 | - |
| sam3 | portable | 179.8 ms / 1928 / 2196 | 181.5 ms / 1935 / 2208 | 182.6 ms / 1948 / 2192 | - |
| sam3 | kernels | 116.3 ms / 2052 / 2460 | 116.5 ms / 2059 / 2420 | 116.9 ms / 2072 / 2490 | - |
| detr-r101-coco | portable | 14.4 ms / 278 / 332 | 17.6 ms / 307 / 356 | 18.3 ms / 320 / 416 | 23.8 ms / 636 / 676 |
| detr-r101-coco | kernels | 10.5 ms / 228 / 278 | 13.0 ms / 248 / 294 | 13.3 ms / 262 / 354 | 19.8 ms / 637 / 678 |
| detr-r50-coco | portable | 9.5 ms / 241 / 296 | 11.6 ms / 270 / 320 | 12.2 ms / 283 / 380 | 18.7 ms / 599 / 640 |
| detr-r50-coco | kernels | 7.7 ms / 191 / 240 | 8.9 ms / 211 / 256 | 9.8 ms / 225 / 316 | 16.7 ms / 600 / 640 |
| faster-rcnn-r50-fpn-3x-coco | portable | 18.5 ms / 309 / 420 | 21.5 ms / 338 / 426 | 22.2 ms / 352 / 454 | 27.2 ms / 602 / 734 |
| faster-rcnn-r50-fpn-3x-coco | kernels | 16.1 ms / 310 / 422 | 18.7 ms / 339 / 460 | 18.4 ms / 353 / 488 | 24.6 ms / 602 / 736 |
| mask-rcnn-r50-fpn-3x-coco | portable | 18.1 ms / 314 / 422 | 23.1 ms / 348 / 440 | 25.1 ms / 413 / 536 | 27.1 ms / 607 / 736 |
| mask-rcnn-r50-fpn-3x-coco | kernels | 15.6 ms / 315 / 422 | 19.6 ms / 349 / 472 | 22.3 ms / 414 / 568 | 24.6 ms / 607 / 736 |
| rtdetrv2-l-coco | portable | 24.1 ms / 171 / 202 | 15.4 ms / 176 / 204 | 13.2 ms / 204 / 226 | 18.2 ms / 574 / 698 |
| rtdetrv2-l-coco | kernels | 6.4 ms / 205 / 328 | 6.5 ms / 213 / 338 | 7.1 ms / 240 / 338 | 16.4 ms / 610 / 758 |
| rtdetrv2-m-coco | portable | 9.4 ms / 147 / 178 | 11.9 ms / 155 / 202 | 12.8 ms / 181 / 216 | 16.1 ms / 552 / 688 |
| rtdetrv2-m-coco | kernels | 5.1 ms / 182 / 312 | 5.3 ms / 189 / 324 | 5.8 ms / 216 / 336 | 15.1 ms / 586 / 752 |
| rtdetrv2-ms-coco | portable | 9.9 ms / 160 / 190 | 9.5 ms / 166 / 200 | 9.7 ms / 193 / 228 | 16.6 ms / 563 / 700 |
| rtdetrv2-ms-coco | kernels | 5.3 ms / 195 / 314 | 5.4 ms / 202 / 326 | 6.0 ms / 229 / 338 | 15.2 ms / 600 / 760 |
| rtdetrv2-s-coco | portable | 7.7 ms / 128 / 154 | 17.8 ms / 135 / 164 | 10.2 ms / 160 / 192 | 14.8 ms / 530 / 664 |
| rtdetrv2-s-coco | kernels | 4.0 ms / 161 / 288 | 4.1 ms / 168 / 300 | 4.7 ms / 194 / 312 | 14.0 ms / 564 / 728 |
| rtdetrv2-x-coco | portable | 15.4 ms / 237 / 264 | 23.5 ms / 244 / 274 | 18.4 ms / 266 / 302 | 22.8 ms / 636 / 774 |
| rtdetrv2-x-coco | kernels | 9.0 ms / 272 / 390 | 9.2 ms / 279 / 402 | 9.8 ms / 302 / 414 | 19.0 ms / 673 / 836 |
| dinov2-vitb14 | portable | 10.3 ms / 231 / 242 | 39.8 ms / 295 / 342 | 137.6 ms / 417 / 442 | 3232.4 ms (n=3) / 1460 / 1562 |
| dinov2-vitb14 | kernels | 7.5 ms / 243 / 262 | 28.0 ms / 324 / 362 | 94.7 ms / 480 / 542 | 2147.8 ms (n=3) / 1821 / 2002 |
| dinov2-vitb14-reg | portable | 10.1 ms / 238 / 254 | 39.9 ms / 303 / 368 | 140.2 ms / 423 / 544 | 3244.0 ms (n=3) / 1467 / 2244 |
| dinov2-vitb14-reg | kernels | 7.6 ms / 254 / 374 | 28.1 ms / 337 / 452 | 94.7 ms / 492 / 704 | 2147.6 ms (n=3) / 1833 / 2404 |
| dinov2-vitg14 | portable | 101.5 ms / 2273 / 2304 | 375.9 ms / 2410 / 2464 | 1169.8 ms / 2673 / 2764 | 23035.0 ms (n=3) / 4929 / 5404 |
| dinov2-vitg14 | kernels | 69.7 ms / 2296 / 2322 | 257.4 ms / 2469 / 2522 | 797.8 ms / 2801 / 2924 | 15248.1 ms (n=3) / 5650 / 6264 |
| dinov2-vitg14-reg | portable | 101.8 ms / 2294 / 2454 | 377.1 ms / 2432 / 2656 | 1184.7 ms / 2694 / 2984 | 23131.1 ms (n=3) / 4951 / 6020 |
| dinov2-vitg14-reg | kernels | 70.3 ms / 2298 / 2634 | 257.6 ms / 2472 / 2864 | 802.9 ms / 2803 / 3352 | 15244.2 ms (n=3) / 5654 / 6720 |
| dinov2-vitl14 | portable | 29.8 ms / 656 / 682 | 118.1 ms / 736 / 782 | 395.5 ms / 889 / 902 | 8770.0 ms (n=3) / 2205 / 2482 |
| dinov2-vitl14 | kernels | 19.6 ms / 671 / 702 | 77.1 ms / 775 / 802 | 252.1 ms / 974 / 1042 | 5331.0 ms (n=3) / 2686 / 2862 |
| dinov3-convnext-base | portable | 8.3 ms / 253 / 290 | 22.8 ms / 357 / 410 | 55.2 ms / 565 / 650 | 323.2 ms / 2310 / 2670 |
| dinov3-convnext-base | kernels | 6.5 ms / 279 / 360 | 15.1 ms / 372 / 572 | 33.5 ms / 502 / 548 | 190.9 ms / 1914 / 2048 |
| dinov3-convnext-large | portable | 14.8 ms / 484 / 510 | 42.5 ms / 633 / 766 | 99.9 ms / 932 / 1048 | 587.2 ms / 3435 / 4008 |
| dinov3-convnext-large | kernels | 11.0 ms / 505 / 642 | 26.7 ms / 635 / 950 | 61.0 ms / 837 / 926 | 357.8 ms / 2887 / 2946 |
| dinov3-convnext-small | portable | 5.9 ms / 169 / 188 | 15.6 ms / 250 / 266 | 37.9 ms / 413 / 486 | 220.9 ms / 1778 / 2106 |
| dinov3-convnext-small | kernels | 4.5 ms / 197 / 276 | 10.6 ms / 272 / 410 | 23.3 ms / 366 / 406 | 133.7 ms / 1474 / 1526 |
| dinov3-convnext-tiny | portable | 3.6 ms / 127 / 148 | 9.8 ms / 209 / 226 | 23.9 ms / 372 / 446 | 136.6 ms / 1737 / 2066 |
| dinov3-convnext-tiny | kernels | 2.7 ms / 156 / 236 | 6.5 ms / 231 / 368 | 14.6 ms / 324 / 366 | 82.1 ms / 1433 / 1484 |
| dinov3-vitb16 | portable | 8.8 ms / 224 / 242 | 31.6 ms / 281 / 304 | 102.4 ms / 389 / 464 | 2039.3 ms (n=3) / 1308 / 1502 |
| dinov3-vitb16 | kernels | 5.3 ms / 233 / 326 | 19.4 ms / 297 / 324 | 64.4 ms / 424 / 462 | 1322.5 ms / 1515 / 1702 |
| dinov3-vith16plus | portable | 60.7 ms / 1689 / 1724 | 218.8 ms / 1795 / 1846 | 636.2 ms / 1996 / 2106 | 10005.0 ms (n=3) / 3712 / 3944 |
| dinov3-vith16plus | kernels | 38.6 ms / 1690 / 1786 | 138.3 ms / 1789 / 1806 | 393.4 ms / 1984 / 2024 | 6021.9 ms (n=3) / 3643 / 3804 |
| dinov3-vitl16 | portable | 25.9 ms / 645 / 662 | 95.3 ms / 715 / 764 | 298.3 ms / 847 / 924 | 5549.3 ms (n=3) / 1975 / 2142 |
| dinov3-vitl16 | kernels | 15.6 ms / 656 / 726 | 54.9 ms / 738 / 764 | 174.5 ms / 899 / 922 | 3262.4 ms (n=3) / 2274 / 2402 |
| dinov3-vits16 | portable | 4.8 ms / 92 / 104 | 12.9 ms / 130 / 158 | 44.4 ms / 201 / 238 | 990.9 ms / 807 / 1036 |
| dinov3-vits16 | kernels | 2.3 ms / 97 / 168 | 8.6 ms / 145 / 242 | 29.7 ms / 213 / 236 | 644.8 ms / 876 / 1076 |
| fast-acvnet-plus-kitti15 | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.7 ms / 5054 / 8076 |
| fast-acvnet-plus-kitti15 | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.8 ms / 5743 / 7342 |
| fast-acvnet-plus-sceneflow | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.7 ms / 5054 / 8076 |
| fast-acvnet-plus-sceneflow | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.1 ms / 1006 / 1226 | 422.7 ms / 5743 / 7342 |
| fast-acvnet-sceneflow | portable | 14.0 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-sceneflow | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.6 ms / 5445 / 7618 |
| selective-igev-middlebury-train | portable | 190.7 ms / 282 / 362 | 575.4 ms / 769 / 1036 | 1420.5 ms / 1690 / 2030 | 8921.4 ms (n=3) / 10549 / 14320 |
| selective-igev-middlebury-train | kernels | 82.5 ms / 260 / 368 | 213.1 ms / 693 / 1010 | 472.4 ms / 1502 / 2046 | 2804.8 ms (n=3) / 6928 / 8800 |
| selective-igev-sceneflow | portable | 190.6 ms / 282 / 362 | 575.4 ms / 769 / 1036 | 1420.3 ms / 1690 / 2030 | 8917.6 ms (n=3) / 10549 / 14320 |
| selective-igev-sceneflow | kernels | 82.5 ms / 260 / 368 | 213.5 ms / 693 / 1010 | 473.3 ms / 1502 / 2046 | 2808.4 ms (n=3) / 6928 / 8800 |
| dinov2-vitl14-reg | portable | 30.1 ms / 656 / 682 | 118.2 ms / 736 / 782 | 396.1 ms / 889 / 982 | 8778.1 ms (n=3) / 2205 / 2622 |
| dinov2-vitl14-reg | kernels | 19.6 ms / 671 / 702 | 77.1 ms / 775 / 802 | 252.5 ms / 974 / 1042 | 5326.8 ms (n=3) / 2686 / 2982 |
| dinov2-vits14 | portable | 4.3 ms / 95 / 116 | 16.8 ms / 133 / 176 | 61.4 ms / 207 / 256 | 1574.5 ms / 843 / 976 |
| dinov2-vits14 | kernels | 3.3 ms / 100 / 116 | 12.0 ms / 148 / 176 | 39.0 ms / 239 / 276 | 957.1 ms / 1023 / 1116 |
| dinov2-vits14-reg | portable | 4.3 ms / 96 / 108 | 16.9 ms / 135 / 164 | 61.3 ms / 210 / 248 | 1582.1 ms / 846 / 1240 |
| dinov2-vits14-reg | kernels | 3.3 ms / 106 / 148 | 12.0 ms / 153 / 176 | 39.0 ms / 244 / 308 | 960.9 ms / 1029 / 1344 |
| vggt-1b | portable | 793.4 ms / 3868 / 5308 | 561.3 ms / 3710 / 4930 | 566.5 ms / 3920 / 5178 | 849.5 ms / 8219 / 10306 |
| vggt-1b | kernels | 559.0 ms / 3880 / 5474 | 398.6 ms / 3721 / 5156 | 404.0 ms / 3931 / 5390 | 615.0 ms / 8232 / 10518 |
| vggt-omega-1b-512 | portable | 722.8 ms / 3449 / 3866 | 723.2 ms / 3559 / 4038 | 725.7 ms / 3771 / 4366 | 757.0 ms / 5594 / 6894 |
| vggt-omega-1b-512 | kernels | 461.8 ms / 3449 / 3796 | 461.1 ms / 3560 / 3968 | 463.9 ms / 3771 / 4332 | 493.0 ms / 5594 / 6894 |
| mapanything-apache | portable | 680.9 ms / 3914 / 5848 | 493.1 ms / 3635 / 5196 | 498.1 ms / 3846 / 5442 | 734.6 ms / 7240 / 9442 |
| mapanything-apache | kernels | 569.7 ms / 3914 / 5846 | 415.3 ms / 3635 / 5196 | 419.9 ms / 3846 / 5442 | 622.3 ms / 7240 / 9440 |
| mvsanywhere | portable | 107.5 ms / 969 / 1500 | 75.8 ms / 954 / 1398 | 77.4 ms / 1166 / 1756 | 124.0 ms / 3110 / 4342 |
| mvsanywhere | kernels | 40.6 ms / 769 / 1006 | 31.0 ms / 838 / 1068 | 32.5 ms / 1049 / 1414 | 57.1 ms / 2908 / 4006 |
| patchmatchnet | portable | 43.1 ms / 439 / 522 | 154.4 ms / 1242 / 1534 | 408.6 ms / 2753 / 3346 | OOM |
| patchmatchnet | kernels | 9.2 ms / 253 / 328 | 24.4 ms / 693 / 942 | 56.7 ms / 1520 / 2016 | 378.6 ms / 8628 / 11332 |
| great-igev-eth3d | portable | 206.7 ms / 681 / 1030 | 627.7 ms / 1216 / 2262 | 1532.6 ms / 1688 / 2780 | 13067.4 ms (n=3) / 11364 / 15106 |
| great-igev-eth3d | kernels | 99.0 ms / 385 / 476 | 241.2 ms / 1041 / 1338 | 544.2 ms / 1506 / 2354 | 3753.5 ms (n=3) / 11736 / 15500 |
| great-igev-kitti | portable | 217.1 ms / 681 / 1030 | 628.2 ms / 1216 / 2262 | 1532.7 ms / 1688 / 2780 | 13065.3 ms (n=3) / 11364 / 15106 |
| great-igev-kitti | kernels | 93.2 ms / 385 / 476 | 240.9 ms / 1041 / 1338 | 544.6 ms / 1506 / 2356 | 3755.4 ms (n=3) / 11736 / 15500 |
| great-igev-middlebury | portable | 328.5 ms / 1082 / 1888 | 1025.5 ms / 1344 / 2262 | 2554.0 ms (n=3) / 2549 / 4266 | OOM |
| great-igev-middlebury | kernels | 131.4 ms / 942 / 1258 | 356.0 ms / 1279 / 1906 | 804.4 ms / 1825 / 2656 | 6324.4 ms (n=3) / 11920 / 15500 |
| great-igev-sceneflow | portable | 200.9 ms / 681 / 1030 | 624.1 ms / 1216 / 2262 | 1531.8 ms / 1688 / 2780 | 13071.3 ms (n=3) / 11364 / 15106 |
| great-igev-sceneflow | kernels | 93.5 ms / 385 / 476 | 241.1 ms / 1041 / 1338 | 545.3 ms / 1506 / 2354 | 3759.3 ms (n=3) / 11736 / 15500 |
| igev-plusplus-eth3d | portable | 201.5 ms / 354 / 614 | 606.5 ms / 987 / 1662 | 1534.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-eth3d | kernels | 88.7 ms / 357 / 1092 | 241.2 ms / 830 / 1210 | 534.3 ms / 1799 / 2464 | 3135.8 ms (n=3) / 8419 / 9752 |
| igev-plusplus-kitti12 | portable | 201.4 ms / 354 / 614 | 607.2 ms / 987 / 1662 | 1535.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-kitti12 | kernels | 89.1 ms / 357 / 1092 | 244.8 ms / 830 / 1210 | 535.9 ms / 1799 / 2464 | 3143.3 ms (n=3) / 8419 / 9750 |
| igev-plusplus-kitti15 | portable | 202.7 ms / 354 / 614 | 606.9 ms / 987 / 1662 | 1535.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-kitti15 | kernels | 88.7 ms / 357 / 1092 | 240.2 ms / 830 / 1210 | 534.9 ms / 1799 / 2464 | 3141.8 ms (n=3) / 8419 / 9752 |
| igev-plusplus-middlebury | portable | 201.2 ms / 354 / 612 | 606.8 ms / 987 / 1662 | 1535.6 ms / 2172 / 3496 | OOM |
| igev-plusplus-middlebury | kernels | 88.7 ms / 357 / 1092 | 240.5 ms / 830 / 1210 | 535.0 ms / 1799 / 2464 | 3140.6 ms (n=3) / 8419 / 9752 |
| igev-plusplus-sceneflow | portable | 201.0 ms / 354 / 612 | 606.2 ms / 987 / 1662 | 1534.9 ms / 2172 / 3496 | OOM |
| igev-plusplus-sceneflow | kernels | 88.6 ms / 357 / 1092 | 240.6 ms / 830 / 1210 | 535.4 ms / 1799 / 2464 | 3143.9 ms (n=3) / 8419 / 9752 |

Fixed-input models (classification, features), run at their one native size, 224x224:

| model | mode | 224x224 |
|---|---|---|
| alexnet.tv_in1k | portable | 1.0 ms / 152 / 272 |
| alexnet.tv_in1k | kernels | 0.7 ms / 185 / 310 |
| convnextv2_base.fcmae_ft_in22k_in1k | portable | 10.2 ms / 300 / 370 |
| convnextv2_base.fcmae_ft_in22k_in1k | kernels | 3.0 ms / 334 / 450 |
| deit_base_patch16_224.fb_in1k | portable | 2.9 ms / 214 / 338 |
| deit_base_patch16_224.fb_in1k | kernels | 1.8 ms / 316 / 418 |
| densenet121.tv_in1k | portable | 9.5 ms / 53 / 84 |
| densenet121.tv_in1k | kernels | 1.6 ms / 86 / 168 |
| dinov2_vitl14_lc | portable | 5.4 ms / 625 / 724 |
| dinov2_vitl14_lc | kernels | 5.2 ms / 897 / 1040 |
| efficientnet_b0.tv_in1k | portable | 5.0 ms / 48 / 84 |
| efficientnet_b0.tv_in1k | kernels | 0.9 ms / 81 / 164 |
| efficientnet_b4.tv_in1k | portable | 9.9 ms / 92 / 120 |
| efficientnet_b4.tv_in1k | kernels | 2.2 ms / 125 / 200 |
| efficientnet_v2_s.tv_in1k | portable | 10.0 ms / 82 / 132 |
| efficientnet_v2_s.tv_in1k | kernels | 2.1 ms / 116 / 210 |
| efficientvit_b2.r288_in1k | portable | 11.9 ms / 87 / 116 |
| efficientvit_b2.r288_in1k | kernels | 1.2 ms / 122 / 200 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | portable | 18.0 ms / 666 / 864 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | kernels | 14.0 ms / 1296 / 1456 |
| fastvit_sa24.apple_dist_in1k | portable | 6.9 ms / 81 / 118 |
| fastvit_sa24.apple_dist_in1k | kernels | 1.5 ms / 113 / 194 |
| inception_v3.tv_in1k | portable | 5.2 ms / 123 / 226 |
| inception_v3.tv_in1k | kernels | 1.6 ms / 154 / 306 |
| mobilenet_v2.tv2_in1k | portable | 3.0 ms / 43 / 62 |
| mobilenet_v2.tv2_in1k | kernels | 0.5 ms / 75 / 98 |
| mobilenet_v3_large.tv2_in1k | portable | 3.6 ms / 46 / 66 |
| mobilenet_v3_large.tv2_in1k | kernels | 0.6 ms / 78 / 104 |
| mobilenetv1_100.ra4_e3600_r224_in1k | portable | 1.8 ms / 44 / 62 |
| mobilenetv1_100.ra4_e3600_r224_in1k | kernels | 0.4 ms / 75 / 98 |
| mobilenetv4_conv_medium.e500_r256_in1k | portable | 3.8 ms / 55 / 74 |
| mobilenetv4_conv_medium.e500_r256_in1k | kernels | 0.8 ms / 87 / 112 |
| regnety_160.tv2_in1k | portable | 8.3 ms / 211 / 524 |
| regnety_160.tv2_in1k | kernels | 2.0 ms / 238 / 610 |
| resnet152.tv2_in1k | portable | 8.4 ms / 157 / 192 |
| resnet152.tv2_in1k | kernels | 2.2 ms / 188 / 270 |
| resnet18.tv_in1k | portable | 1.4 ms / 60 / 68 |
| resnet18.tv_in1k | kernels | 0.6 ms / 91 / 144 |
| resnet50.tv2_in1k | portable | 3.1 ms / 91 / 110 |
| resnet50.tv2_in1k | kernels | 1.0 ms / 121 / 190 |
| shufflenet_v2_x1_0.tv_in1k | portable | 3.7 ms / 39 / 58 |
| shufflenet_v2_x1_0.tv_in1k | kernels | 0.8 ms / 87 / 102 |
| squeezenet1_1.tv_in1k | portable | 1.8 ms / 7 / 28 |
| squeezenet1_1.tv_in1k | kernels | 0.4 ms / 8 / 14 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | portable | 10.6 ms / 218 / 312 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | kernels | 2.5 ms / 261 / 412 |
| vgg16.tv_in1k | portable | 2.0 ms / 313 / 338 |
| vgg16.tv_in1k | kernels | 1.6 ms / 343 / 412 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | portable | 2.9 ms / 214 / 338 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | kernels | 1.8 ms / 316 / 418 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | portable | 13.1 ms / 901 / 1050 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | kernels | 12.9 ms / 1254 / 1446 |
<!-- sizes:end -->

## Apple silicon (MPS)

`portable` on an M1 Pro (16 GB, 16-core GPU, torch 2.14, macOS 27), one variant per family, on the same
`fastforks bench` input as the tables above (480x640; classification and features at their native crop). Cells: median
ms of 5 warm calls (2 for calls over 2 s) / the process's peak footprint in MiB (Activity Monitor's "Memory": model and
activations, which share the system RAM on a Mac). float16 is measured only where the card offers it on MPS (n/a = it does not). The first
call at a new size takes 0.5-4 s longer. Not measured: the gated DINOv3, DINOv3-ConvNeXt, SAM 3 and VGGT-Omega
weights. Reproduce (`bench` prints the Metal driver's memory: the GPU buffers only, not the whole process):

    fastforks bench MODEL --device mps --size 480x640 --views 4 [--dtype float16]

Accuracy against the CPU (float32, Middlebury ArtL for stereo): MPS float32 outputs match to 1e-3 px mean or better
(most families 1e-5), except MonSter++ (0.006 px) and NMRF (0.03 px on non-occluded pixels, about 0.7 px in
occlusions, where its local-maximum test flips on last-bit differences in flat cost curves; the error against ground
truth is unchanged). float16 moves stereo outputs by 0.003-0.07 px mean. Depth: float32 matches the CPU to 1e-4 of the
median depth, float16 moves it by 0.02-0.12 %.

| task | model | float32 | float16 |
|---|---|---|---|
| classification | `alexnet.tv_in1k` | 5.3 ms / 1590 | 4.5 ms / 1583 |
| classification | `convnextv2_base.fcmae_ft_in22k_in1k` | 36.6 ms / 2317 | n/a |
| classification | `deit_base_patch16_224.fb_in1k` | 18.1 ms / 1330 | n/a |
| classification | `densenet121.tv_in1k` | 26.5 ms / 727 | n/a |
| classification | `dinov2_vitl14_lc` | 65.8 ms / 4435 | n/a |
| classification | `efficientnet_b0.tv_in1k` | 12.9 ms / 487 | n/a |
| classification | `efficientnet_v2_s.tv_in1k` | 34.7 ms / 633 | n/a |
| classification | `efficientvit_b2.r288_in1k` | 20.6 ms / 1710 | n/a |
| classification | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 266 ms / 4108 | 236 ms / 4106 |
| classification | `fastvit_sa24.apple_dist_in1k` | 18.0 ms / 669 | n/a |
| classification | `inception_v3.tv_in1k` | 18.2 ms / 766 | n/a |
| classification | `mobilenet_v2.tv2_in1k` | 7.1 ms / 444 | n/a |
| classification | `mobilenet_v3_large.tv2_in1k` | 10.5 ms / 481 | n/a |
| classification | `mobilenetv1_100.ra4_e3600_r224_in1k` | 5.1 ms / 456 | n/a |
| classification | `mobilenetv4_conv_medium.e500_r256_in1k` | 10.2 ms / 601 | n/a |
| classification | `regnety_160.tv2_in1k` | 40.0 ms / 2025 | 33.8 ms / 2038 |
| classification | `resnet152.tv2_in1k` | 25.9 ms / 1101 | n/a |
| classification | `resnet50.tv2_in1k` | 12.9 ms / 660 | n/a |
| classification | `shufflenet_v2_x1_0.tv_in1k` | 7.1 ms / 386 | n/a |
| classification | `squeezenet1_1.tv_in1k` | 4.4 ms / 359 | 4.3 ms / 356 |
| classification | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 29.0 ms / 2283 | n/a |
| classification | `vgg16.tv_in1k` | 14.5 ms / 2453 | 12.5 ms / 2459 |
| classification | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 18.1 ms / 1331 | n/a |
| classification | `vit_so400m_patch14_siglip_378.webli_ft_in1k` | 216 ms / 5563 | 197 ms / 5560 |
| features | `dinov2-vitb14` | 130 ms / 1964 | 116 ms / 819 |
| segmentation | `deeplabv3-r101-d8-ade20k` | 415 ms / 2133 | 354 ms / 2035 |
| segmentation | `dinov2-b-linear-ade20k` | 267 ms / 1974 |  |
| segmentation | `eomt-l-dinov2-ade20k` | 615 ms / 6298 | 548 ms / 4257 |
| segmentation | `mask2former-swinl-ade20k` | 587 ms / 5599 | 639 ms / 4473 |
| segmentation | `oneformer-swinl-ade20k` | 661 ms / 6174 | 697 ms / 4512 |
| segmentation | `sapiens-seg-0.3b-goliath` | 4287 ms / 4522 | 2791 ms / 2813 |
| segmentation | `segformer-b0-ade20k` | 64.7 ms / 1522 |  |
| detection | `faster-rcnn-r50-fpn-3x-coco` | 189 ms / 1932 |  |
| promptable_segmentation | `sam-vit-b` | 635 ms / 4350 |  |
| promptable_segmentation | `sam2.1-hiera-base-plus` | 428 ms / 3951 |  |
| multiview | `mapanything-apache` | 3507 ms / 8525 | 3138 ms / 5482 |
| multiview | `mvsanywhere` | 770 ms / 2507 |  |
| multiview | `patchmatchnet` | 184 ms / 1379 |  |
| multiview | `vggt-1b` | 4072 ms / 9555 |  |
| depth | `da3-giant-1.1` | 939 ms / 6565 | 935 ms / 6567 |
| depth | `da3-large-1.1` | 368 ms / 3480 | 373 ms / 2438 |
| depth | `depth-anything-v2-b` | 199 ms / 2011 | n/a |
| depth | `depth-anything-v2-l` | 588 ms / 3952 | n/a |
| depth | `depth-pro` | 5598 ms / 11224 | 5117 ms / 6086 |
| depth | `marigold-depth-v1-1` | 3451 ms / 9757 | 2882 ms / 6701 |
| depth | `metric3d-vit-large` | 1811 ms / 3445 | n/a |
| depth | `midas-dpt-beit-large-512` | 581 ms / 3385 | 590 ms / 2351 |
| depth | `moge-2-vitl-normal` | 1487 ms / 3892 | n/a |
| depth | `moge-3-vitl` | 3975 ms / 6330 | n/a |
| depth | `unidepth-v2-vitl14` | 611 ms / 4003 | n/a |
| depth | `unik3d-vitl` | 1151 ms / 3986 | n/a |
| depth | `zoedepth-nk` | 734 ms / 3499 | 709 ms / 2454 |
| stereo | `aanet-plus-sceneflow` | 345 ms / 2623 | n/a |
| stereo | `bridgedepth-rvc-pretrain` | 726 ms / 4004 | n/a |
| stereo | `c-fast-foundationstereo` | 1169 ms / 2696 | n/a |
| stereo | `cas-gwcnet-c-sceneflow` | 1369 ms / 2579 | 1335 ms / 1533 |
| stereo | `cfnet-sceneflow` | 1829 ms / 5992 | 1775 ms / 5589 |
| stereo | `coex-sceneflow` | 196 ms / 1387 | 187 ms / 1448 |
| stereo | `crestereo-eth3d` | 954 ms / 1629 | 788 ms / 1626 |
| stereo | `defom-stereo-vitl-sceneflow` | 1863 ms / 5088 | n/a |
| stereo | `dlnr-middlebury` | 591 ms / 1981 | 537 ms / 1932 |
| stereo | `fast-acvnet-generalization` | 242 ms / 1432 | n/a |
| stereo | `foundationstereo-vit-l` | 8814 ms / 8229 | 6265 ms / 4485 |
| stereo | `great-igev-eth3d` | 5446 ms / 3559 | 2176 ms / 2845 |
| stereo | `gwcnet-g-sceneflow` | 1185 ms / 2531 | n/a |
| stereo | `hitnet-eth3d` | 117 ms / 1408 | 124 ms / 1408 |
| stereo | `igev-plusplus-eth3d` | 2283 ms / 2960 | 2154 ms / 1790 |
| stereo | `igev-stereo-eth3d` | 1441 ms / 1671 | n/a |
| stereo | `las2-h` | 278 ms / 1564 | n/a |
| stereo | `mocha-v1-sceneflow` | 2066 ms / 2486 | 1916 ms / 2399 |
| stereo | `monster-mix-all` | 2512 ms / 4176 | 2275 ms / 3494 |
| stereo | `monster-plusplus-mix-all` | 2707 ms / 4212 | 2467 ms / 3493 |
| stereo | `nmrf-sceneflow` | 305 ms / 2430 | n/a |
| stereo | `psmnet-sceneflow` | 1092 ms / 2448 | n/a |
| stereo | `raft-stereo-eth3d` | 953 ms / 1548 | 866 ms / 1507 |
| stereo | `rt-monster-plusplus-mix-all` | 148 ms / 1916 | n/a |
| stereo | `s2m2-l` | 1300 ms / 3959 | 1198 ms / 3955 |
| stereo | `selective-igev-eth3d-finetune` | 1845 ms / 1704 | 1666 ms / 1703 |
| stereo | `waft-stereo-b` | 740 ms / 3362 | n/a |

RAFT-Stereo `kernels` on MPS (Metal ConvGRU and correlation lookup, float16) against float16 `portable`, interleaved
runs in separate processes: 1.08x at 480x640, 1.12x at 992x1440, about 30 MiB less. Its convolutions already run
near the M1 Pro's peak, so fusing the elementwise steps gains little.

`kernels` on MPS (float16; Metal ConvGRUs, correlation / geometry lookups, cost volumes, warp costs, deformable
sampling, window attention) against the fastest validated `portable` dtype, 384x1248, each run in its own process (median of 5, peak footprint):

| model | `portable` | `kernels` | speed-up |
| --- | --- | --- | --- |
| `raft-stereo-sceneflow` | 1273 ms / 2604 MiB (float16) | 1192 ms / 2538 MiB | 1.07x |
| `igev-stereo-sceneflow` | 2090 ms / 2752 MiB (float32) | 1354 ms / 1665 MiB | 1.54x |
| `selective-igev-sceneflow` | 2292 ms / 1727 MiB (float16) | 1878 ms / 1704 MiB | 1.22x |
| `great-igev-sceneflow` | 3366 ms / 3416 MiB (float16) | 2869 ms / 3463 MiB | 1.17x |
| `igev-plusplus-sceneflow` | 2852 ms / 1783 MiB (float16) | 2054 ms / 1803 MiB | 1.39x |
| `mocha-v1-sceneflow` | 2734 ms / 2403 MiB (float16) | 2277 ms / 2382 MiB | 1.20x |
| `monster-sceneflow` | 3344 ms / 3491 MiB (float16) | 2756 ms / 4093 MiB | 1.21x |
| `monster-plusplus-mix-all` | 3518 ms / 3490 MiB (float16) | 2928 ms / 3492 MiB | 1.20x |
| `cfnet-sceneflow` | 2228 ms / 5463 MiB (float16) | 758 ms / 2931 MiB | 2.94x |
| `las2-h` | 418 ms / 2625 MiB (float32) | 266 ms / 1642 MiB | 1.57x |
| `las2-s` | 94 ms / 1501 MiB (float32) | 85 ms / 1542 MiB | 1.11x |
| `hitnet-eth3d` | 206 ms / 2444 MiB (float32) | 74 ms / 1409 MiB | 2.80x |
| `hitnet-middlebury-d400` | 448 ms / 4539 MiB (float32) | 175 ms / 1454 MiB | 2.57x |
| `aanet-sceneflow` | 319 ms / 1484 MiB (float32) | 246 ms / 1466 MiB | 1.30x |
| `aanet-plus-sceneflow` | 457 ms / 2650 MiB (float32) | 410 ms / 1585 MiB | 1.11x |
| `crestereo-eth3d` | 1265 ms / 1597 MiB (float16) | 1104 ms / 1642 MiB | 1.15x |
| `defom-stereo-vitl-sceneflow` | 3505 ms / 6120 MiB (float32) | 2627 ms / 4069 MiB | 1.33x |
| `foundationstereo-vit-s` | 7825 ms / 4304 MiB (float16) | 7383 ms / 4284 MiB | 1.06x |
| `nmrf-sceneflow` | 455 ms / 2408 MiB (float32) | 365 ms / 1425 MiB | 1.24x |
| `bridgedepth-sf` | 1065 ms / 4145 MiB (float32) | 1012 ms / 4093 MiB | 1.05x |

Most of the IGEV family's gain is the lookup (the portable one gathers from each pyramid level separately); the fused
ConvGRUs add about 5 % (Selective-IGEV and MoCha get the lookup only: a selective GRU and an LSTM). IGEV-Stereo
`portable` is validated in float32 only on MPS, hence its larger factor (as DEFOM-Stereo's: fused lookup and ConvGRUs
in float16). CFNet's gain is the warped volumes of its refinement stages built by one kernel (and 2.5 GB less memory);
HITNet's is the warp costs of its tile hypotheses, which `portable` builds from many small ops. FoundationStereo's
Metal geometry lookup gains little: its hourglass and transformer dominate. NMRF and BridgeDepth get their window
attention with relative-position embeddings as one kernel (2.5-4x faster than the float16 module); BridgeDepth's time is
mostly its ViT and alignment layers. Both modes, in every dtype, replace the transposed
convolutions MPS runs slowly: a stride-2 one as one convolution per output parity (four `conv2d` or eight `conv3d`
calls; MPS has no float16 3-D kernel at all), a 2-D one whose kernel equals its stride (ViT necks, DPT heads) as a
1 x 1 convolution and `pixel_shuffle`. End to end at 384x1248 that makes CAS-GwcNet float16 2.3x, CoEx float16 1.9x,
GwcNet 1.3x and PSMNet 1.15x faster, and Sapiens 1.9x at 1024x768.
