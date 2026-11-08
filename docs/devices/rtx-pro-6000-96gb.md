# NVIDIA RTX PRO 6000

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed/rtx-pro-6000-96gb.jsonl) -->

| | |
|---|---|
| device | NVIDIA RTX PRO 6000 |
| backend | cuda |
| memory | 96 GB GDDR7 |
| hardware | Blackwell Server Edition, sm_120, 600 W |
| software | torch 2.14.1+cu130, cuda 13.0, driver 610.57.04 |
| torch (rows) | 2.14.1+cu130 |
| measured | 2026-11-08 to 2026-11-08 |
| rows | 348 in `docs/data/speed/rtx-pro-6000-96gb.jsonl` |

A survey, not a regression floor: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 3 warm-up, peak reset, median of 3 rounds x median of 5.

## Stereo

### Stereo, Middlebury 2014 half, 994x1482

| model | time (ms): portable fp16 | portable bf16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 46.3 | – | 22.1 | 2.1x | 1139 / 1276 | 778 / 978 |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 36.8 | – | 20.9 | 1.8x | 843 / 982 | 502 / 620 |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 71.5 | – | 68.8 | 1.0x | 2957 / 3204 | 2347 / 2642 |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 71.6 | – | 68.8 | 1.0x | 2906 / 3192 | 2295 / 2592 |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 72.8 | – | 37.3 | 2.0x | 2107 / 2456 | 1193 / 1536 |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | 747 | – | 319 | 2.3x | 6309 / 6450 | 3102 / 4248 |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 61.0 | – | 46.7 | 1.3x | 1972 / 2448 | 1690 / 1890 |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 93.6 | – | 78.8 | 1.2x | 2303 / 2768 | 2237 / 2452 |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 89.5 | – | 76.4 | 1.2x | 3406 / 4336 | 3628 / 4598 |
| [`coex-sceneflow`](../models/stereo.md#coex) | 14.9 | – | 6.43 | 2.3x | 1102 / 1194 | 420 / 514 |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 128 | – | 60.9 | 2.1x | 810 / 966 | 640 / 708 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | – | 266 | 146 | 1.8x | 4357 / 4972 | 3148 / 3542 |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 76.4 | – | 48.6 | 1.6x | 2632 / 3216 | 1234 / 1342 |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 76.4 | – | 48.6 | 1.6x | 2632 / 3216 | 1234 / 1342 |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 15.4 | – | 9.33 | 1.6x | 662 / 736 | 709 / 794 |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | 15.4 | – | 9.58 | 1.6x | 662 / 736 | 709 / 794 |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | 15.4 | – | 9.34 | 1.6x | 662 / 736 | 709 / 794 |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | 21.6 | – | 11.7 | 1.8x | 659 / 736 | 748 / 976 |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | 21.6 | – | 11.5 | 1.9x | 659 / 736 | 748 / 976 |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | 21.6 | – | 11.6 | 1.9x | 659 / 736 | 748 / 976 |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | 21.6 | – | 11.8 | 1.8x | 659 / 736 | 748 / 976 |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | 15.4 | – | 9.55 | 1.6x | 662 / 736 | 709 / 794 |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 72.7 | – | 36.8 | 2.0x | 2107 / 2456 | 1193 / 1536 |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | 496 | – | 337 | 1.5x | 3138 / 3436 | 2815 / 3180 |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | 426 | – | 270 | 1.6x | 2490 / 2786 | 2167 / 2610 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | 191 | – | 97.2 | 2.0x | 1399 / 1598 | 1189 / 1602 |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | 191 | – | 97.4 | 2.0x | 1399 / 1598 | 1189 / 1602 |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | 338 | – | 143 | 2.4x | 1872 / 2078 | 1551 / 1702 |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | 191 | – | 98.3 | 1.9x | 1399 / 1598 | 1189 / 1602 |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 102 | – | 50.5 | 2.0x | 2046 / 2140 | 1764 / 1942 |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | 104 | – | 56.7 | 1.8x | 2047 / 2142 | 1765 / 1942 |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 14.4 | – | 6.22 | 2.3x | 857 / 1068 | 632 / 770 |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | 40.1 | – | 14.3 | 2.8x | 1709 / 2174 | 1216 / 1514 |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 185 | – | 104 | 1.8x | 1591 / 1876 | 1327 / 1594 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 185 | – | 104 | 1.8x | 1591 / 1916 | 1327 / 1574 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 185 | – | 104 | 1.8x | 1591 / 1876 | 1327 / 1754 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 185 | – | 104 | 1.8x | 1591 / 1876 | 1327 / 1754 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 185 | – | 104 | 1.8x | 1591 / 1876 | 1327 / 1574 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 128 | – | 72.6 | 1.8x | 1234 / 1452 | 1110 / 1332 |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 128 | – | 72.2 | 1.8x | 1234 / 1452 | 1110 / 1512 |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 128 | – | 72.3 | 1.8x | 1234 / 1452 | 1110 / 1512 |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 128 | – | 72.6 | 1.8x | 1234 / 1452 | 1110 / 1512 |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 128 | – | 72.6 | 1.8x | 1234 / 1452 | 1110 / 1512 |
| [`las2-h`](../models/stereo.md#las2) | 28.8 | – | 15.5 | 1.8x | 1047 / 1186 | 951 / 1040 |
| [`las2-l`](../models/stereo.md#las2) | 22.1 | – | 8.96 | 2.5x | 1002 / 1054 | 332 / 422 |
| [`las2-m`](../models/stereo.md#las2) | 17.0 | – | 6.91 | 2.5x | 937 / 982 | 263 / 376 |
| [`las2-s`](../models/stereo.md#las2) | 14.5 | – | 4.99 | 2.9x | 921 / 966 | 255 / 380 |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 189 | – | 116 | 1.6x | 1630 / 1760 | 1585 / 1688 |
| [`mocha-v2-sceneflow`](../models/stereo.md#mocha_stereo) | 185 | – | 115 | 1.6x | 1629 / 1866 | 1585 / 1788 |
| [`monster-mix-all`](../models/stereo.md#monster) | 275 | – | 164 | 1.7x | 2334 / 2614 | 2350 / 2574 |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 285 | – | 167 | 1.7x | 2580 / 2734 | 2432 / 2794 |
| [`monster-sceneflow`](../models/stereo.md#monster) | 275 | – | 164 | 1.7x | 2334 / 2614 | 2350 / 2574 |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 43.1 | – | 37.7 | 1.1x | 1481 / 1630 | 993 / 1110 |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 93.1 | – | 48.9 | 1.9x | 2043 / 2422 | 1752 / 2080 |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 148 | – | 63.6 | 2.3x | 1385 / 1490 | 730 / 758 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 148 | – | 63.6 | 2.3x | 1385 / 1490 | 730 / 758 |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 148 | – | 63.6 | 2.3x | 1385 / 1490 | 730 / 758 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 25.3 | – | 23.6 | 1.1x | 2589 / 2658 | 2641 / 3080 |
| [`s2m2-l`](../models/stereo.md#s2m2) | 204 | – | 91.7 | 2.2x | 3242 / 3816 | 1247 / 1510 |
| [`s2m2-m`](../models/stereo.md#s2m2) | 134 | – | 61.1 | 2.2x | 2297 / 2826 | 874 / 1122 |
| [`s2m2-s`](../models/stereo.md#s2m2) | 71.2 | – | 29.0 | 2.5x | 1730 / 2008 | 683 / 902 |
| [`s2m2-xl`](../models/stereo.md#s2m2) | 352 | – | 168 | 2.1x | 5226 / 6094 | 2089 / 2334 |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 170 | – | 89.3 | 1.9x | 1235 / 1448 | 1106 / 1370 |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | 170 | – | 89.3 | 1.9x | 1235 / 1448 | 1106 / 1370 |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | 170 | – | 89.2 | 1.9x | 1235 / 1448 | 1106 / 1368 |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | 170 | – | 89.1 | 1.9x | 1235 / 1448 | 1106 / 1370 |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | 170 | – | 89.4 | 1.9x | 1235 / 1448 | 1106 / 1368 |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | 170 | – | 89.2 | 1.9x | 1235 / 1488 | 1106 / 1368 |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 170 | – | 89.2 | 1.9x | 1235 / 1488 | 1106 / 1350 |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | 257 | – | 85.7 | 3.0x | 5773 / 5850 | 1987 / 2272 |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | 159 | – | 60.1 | 2.6x | 3290 / 3454 | 1130 / 1242 |

## Depth

### Depth, Middlebury 2014 half, 994x1482

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 19.6 | 14.2 | 1.4x | 2633 / 2728 | 2592 / 2690 |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 11.2 | 6.59 | 1.7x | 987 / 1090 | 948 / 1030 |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 6.29 | 6.04 | 1.0x | 811 / 868 | 807 / 848 |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 5.94 | 5.81 | 1.0x | 811 / 868 | 807 / 848 |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 26.2 | 19.7 | 1.3x | 3271 / 3374 | 3235 / 3338 |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 4.37 | 4.00 | 1.1x | 445 / 514 | 350 / 396 |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 9.81 | 8.61 | 1.1x | 1022 / 1112 | 917 / 1034 |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 2.23 | 2.02 | 1.1x | 244 / 288 | 153 / 208 |
| [`depth-pro`](../models/depth.md#depth_pro) | 77.9 | 68.9 | 1.1x | 3301 / 3438 | 3630 / 3718 |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 99.7 | 74.8 | 1.3x | 2780 / 2940 | 2355 / 2500 |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 41.3 | 38.9 | 1.1x | 1447 / 1598 | 1479 / 1628 |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 17.8 | 17.8 | 1.0x | 955 / 1078 | 970 / 1076 |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 27.4 | 24.1 | 1.1x | 1607 / 1674 | 1443 / 1550 |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 15.4 | 11.0 | 1.4x | 1011 / 1132 | 829 / 908 |
| [`moge-3-vitl`](../models/depth.md#moge) | 147 | 61.5 | 2.4x | 1682 / 1964 | 1557 / 1744 |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 18.4 | 18.8 | 1.0x | 1153 / 1232 | 1156 / 1232 |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 20.3 | 19.1 | 1.1x | 1165 / 1232 | 1111 / 1166 |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 29.5 | 19.2 | 1.5x | 1983 / 2064 | 1711 / 1900 |

## Classification

### Classification, synthetic, 224x224

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 0.340 | 0.240 | 1.4x | 151 / 184 | 184 / 228 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 4.03 | 2.83 | 1.4x | 299 / 316 | 318 / 380 |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 1.11 | 1.02 | 1.1x | 203 / 226 | 300 / 368 |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 3.64 | 1.31 | 2.8x | 53 / 70 | 83 / 132 |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 2.22 | 2.31 | 1.0x | 623 / 648 | 889 / 950 |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 1.89 | 0.736 | 2.6x | 48 / 54 | 78 / 118 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 3.67 | 1.61 | 2.3x | 89 / 114 | 109 / 178 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 3.55 | 1.62 | 2.2x | 81 / 126 | 113 / 188 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 5.37 | 0.926 | 5.8x | 85 / 144 | 120 / 210 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 6.80 | 5.46 | 1.2x | 651 / 664 | 1233 / 1406 |
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 2.88 | 1.31 | 2.2x | 81 / 128 | 111 / 188 |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 1.77 | 1.26 | 1.4x | 120 / 164 | 149 / 228 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 1.16 | 0.346 | 3.4x | 43 / 50 | 74 / 92 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 1.46 | 0.489 | 3.0x | 46 / 70 | 78 / 114 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 0.725 | 0.253 | 2.9x | 44 / 70 | 75 / 114 |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 1.46 | 0.592 | 2.5x | 54 / 82 | 87 / 124 |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 3.34 | 1.46 | 2.3x | 208 / 250 | 228 / 316 |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 3.02 | 1.82 | 1.7x | 153 / 200 | 182 / 266 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 0.550 | 0.370 | 1.5x | 59 / 88 | 89 / 150 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 1.20 | 0.666 | 1.8x | 87 / 98 | 116 / 164 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 1.37 | 0.811 | 1.7x | 39 / 48 | 87 / 116 |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 0.655 | 0.269 | 2.4x | 7 / 28 | 8 / 12 |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 4.06 | 1.91 | 2.1x | 211 / 230 | 246 / 298 |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 0.739 | 0.622 | 1.2x | 311 / 328 | 331 / 388 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 1.16 | 1.02 | 1.1x | 203 / 226 | 300 / 368 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 4.80 | 5.17 | 0.9x | 877 / 904 | 1214 / 1346 |

## Segmentation

### Segmentation, synthetic, 1024x2048

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 8.61 | 6.03 | 1.4x | 450 / 512 | 396 / 514 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 9.28 | 5.21 | 1.8x | 410 / 472 | 355 / 456 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 7.70 | 6.86 | 1.1x | 762 / 864 | 768 / 884 |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 14.7 | 13.8 | 1.1x | 1178 / 1244 | 1186 / 1264 |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 5.11 | 3.90 | 1.3x | 639 / 676 | 642 / 676 |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 9.87 | 8.51 | 1.2x | 1512 / 1604 | 1512 / 1604 |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 10.9 | 9.54 | 1.1x | 1511 / 1584 | 1511 / 1584 |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 7.82 | 5.21 | 1.5x | 414 / 472 | 288 / 374 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 8.45 | 4.64 | 1.8x | 421 / 450 | 265 / 344 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 24.9 | 16.9 | 1.5x | 1155 / 1338 | 1155 / 1398 |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 37.7 | 31.4 | 1.2x | 2046 / 2474 | 2046 / 2454 |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 12.7 | 10.3 | 1.2x | 414 / 452 | 359 / 474 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 24.4 | 25.7 | 1.0x | 1916 / 2042 | 1916 / 2042 |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 45.5 | 46.1 | 1.0x | 3532 / 3664 | 3532 / 3662 |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 3.18 | 1.24 | 2.6x | 336 / 356 | 137 / 218 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 10.4 | 8.22 | 1.3x | 543 / 590 | 360 / 492 |

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 11.2 | 10.3 | 1.1x | 365 / 400 | 365 / 420 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 38.7 | 35.1 | 1.1x | 1472 / 1524 | 1467 / 1544 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 21.6 | 19.6 | 1.1x | 821 / 882 | 818 / 880 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 7.78 | 7.85 | 1.0x | 400 / 444 | 414 / 464 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 13.8 | 14.4 | 1.0x | 730 / 768 | 748 / 768 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 5.43 | 5.58 | 1.0x | 305 / 346 | 317 / 346 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 5.15 | 5.11 | 1.0x | 292 / 322 | 304 / 342 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 33.4 | 29.5 | 1.1x | 1910 / 1984 | 1974 / 2224 |

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`detr-r101-coco`](../models/detection.md#detr) | 4.76 | 3.52 | 1.4x | 316 / 370 | 257 / 290 |
| [`detr-r50-coco`](../models/detection.md#detr) | 3.32 | 3.14 | 1.1x | 280 / 332 | 221 / 252 |
| [`dfine-l-coco`](../models/detection.md#dfine) | 7.21 | 4.23 | 1.7x | 152 / 172 | 186 / 294 |
| [`dfine-m-coco`](../models/detection.md#dfine) | 5.64 | 3.30 | 1.7x | 126 / 142 | 163 / 282 |
| [`dfine-n-coco`](../models/detection.md#dfine) | 4.38 | 1.89 | 2.3x | 88 / 116 | 126 / 200 |
| [`dfine-s-coco`](../models/detection.md#dfine) | 4.74 | 2.37 | 2.0x | 105 / 128 | 143 / 230 |
| [`dfine-x-coco`](../models/detection.md#dfine) | 7.23 | 6.05 | 1.2x | 227 / 258 | 245 / 378 |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 6.42 | 5.86 | 1.1x | 342 / 400 | 346 / 380 |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 6.82 | 6.55 | 1.0x | 352 / 440 | 356 / 420 |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 5.07 | 3.31 | 1.5x | 177 / 200 | 207 / 322 |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 4.37 | 2.68 | 1.6x | 156 / 170 | 180 / 272 |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 4.00 | 2.57 | 1.6x | 166 / 182 | 196 / 302 |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 3.55 | 2.21 | 1.6x | 134 / 146 | 158 / 248 |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 6.43 | 4.07 | 1.6x | 243 / 278 | 270 / 380 |

## Features

### Features, synthetic, 224x224

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`dinov2-vitb14`](../models/features.md#dinov2) | 1.15 | 0.985 | 1.2x | 204 / 226 | 237 / 288 |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 1.11 | 0.987 | 1.1x | 206 / 226 | 237 / 288 |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 5.24 | 4.95 | 1.1x | 2215 / 2228 | 2245 / 2290 |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 5.29 | 4.94 | 1.1x | 2217 / 2228 | 2245 / 2290 |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 2.13 | 2.00 | 1.1x | 622 / 644 | 654 / 708 |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 2.19 | 2.00 | 1.1x | 624 / 646 | 654 / 708 |
| [`dinov2-vits14`](../models/features.md#dinov2) | 1.07 | 0.645 | 1.7x | 78 / 98 | 111 / 142 |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 1.08 | 0.639 | 1.7x | 79 / 98 | 111 / 142 |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 1.78 | 2.03 | 0.9x | 208 / 232 | 174 / 214 |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 1.88 | 2.54 | 0.7x | 419 / 430 | 441 / 492 |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 1.81 | 1.78 | 1.0x | 133 / 148 | 100 / 152 |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 1.07 | 0.973 | 1.1x | 92 / 108 | 59 / 110 |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 1.93 | 0.865 | 2.2x | 200 / 206 | 201 / 250 |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 5.27 | 3.56 | 1.5x | 1644 / 1668 | 1644 / 1692 |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 3.56 | 1.99 | 1.8x | 616 / 646 | 617 / 670 |
| [`dinov3-vits16`](../models/features.md#dinov3) | 1.87 | 0.654 | 2.9x | 76 / 78 | 77 / 104 |

## Multiview

### Multiview, synthetic, 336x518

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`mapanything-apache`](../models/multiview.md#mapanything) | 96.5 | 97.6 | 1.0x | 3647 / 3992 | 3647 / 4030 |
| [`vggt-1b`](../models/multiview.md#vggt) | 104 | 106 | 1.0x | 3643 / 3874 | 3654 / 4194 |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | 110 | 87.6 | 1.3x | 3404 / 3556 | 3404 / 3534 |

### Multiview, synthetic, 448x672

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | 23.6 | 12.5 | 1.9x | 871 / 988 | 1134 / 1180 |

### Multiview, synthetic, 1536x1024

| model | time (ms): portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 39.6 | 7.95 | 5.0x | 1714 / 1892 | 648 / 772 |
<!-- device:end -->
