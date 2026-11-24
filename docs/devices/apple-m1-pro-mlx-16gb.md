# Apple M1 Pro 16 GB (MLX)

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed/apple-m1-pro-mlx-16gb.jsonl) -->

| | |
|---|---|
| device | Apple M1 Pro 16 GB (MLX) |
| backend | mlx |
| memory | 16 GB unified |
| hardware | 16-core GPU; the families with an MLX port (device="mlx") |
| software | macOS 27, mlx 0.32.4, torch 2.14 |
| torch (rows) | 2.14.1 |
| measured | 2026-11-19 to 2026-11-24 |
| rows | 1506 in `docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 1898 | 1367 | – | – | – | 2989 | – |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 1724 | 1695 | – | – | – | 2131 | – |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 4192 | 3179 | unsupported | – | – | 2555 | – |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 4973 | 3977 | – | – | – | 2650 | – |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 4565 | 2737 | – | – | – | 2858 | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | 29962 | 18737 | – | – | – | 4010 | – |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 6066 | 2934 | – | – | – | 3733 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 4595 | 3841 | unsupported | – | – | 4754 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 4485 | 3178 | unsupported | – | – | 3669 | – |
| [`coex-sceneflow`](../models/stereo.md#coex) | 765 | 608 | – | – | – | 3187 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 6118 | 4297 | 5212 | 3558 | 1.2x | 1557 | 1863 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 13339 | 10693 | – | – | – | 5189 | – |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 3213 | 2052 | unsupported | – | – | 2609 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 4221 | 3762 | unsupported | – | – | 3937 | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 505 | 475 | unsupported | – | – | 1918 | – |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | 521 | 457 | unsupported | – | – | 1808 | – |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | 506 | 455 | unsupported | – | – | 1831 | – |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | 587 | 541 | unsupported | – | – | 1812 | – |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | 582 | 564 | unsupported | – | – | 1814 | – |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | 580 | 562 | unsupported | – | – | 1810 | – |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | 584 | 540 | unsupported | – | – | 1813 | – |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | 523 | 459 | unsupported | – | – | 1934 | – |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 4326 | 2875 | unsupported | – | – | 2829 | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | – | 25847 | – | – | – | 4941 | – |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | 40293 | 20504 | – | 17814 | 1.2x | 4943 | 4613 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | 11467 | 7613 | unsupported | – | – | 5336 | – |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | 11481 | 7588 | unsupported | – | – | 5294 | – |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | 23063 | 14680 | – | – | – | 5607 | – |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | 12477 | 8028 | – | – | – | 5560 | – |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 3676 | 2450 | – | – | – | 2169 | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | 4102 | 2933 | unsupported | – | – | 2876 | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 419 | 289 | unsupported | – | – | 1157 | – |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | 1616 | 686 | unsupported | – | – | 1891 | – |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 14411 | 9128 | 10129 | 5478 | 1.7x | 2749 | 2863 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 15055 | 8568 | 10166 | 5297 | 1.6x | 2754 | 2873 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 15502 | 8456 | 10077 | 5268 | 1.6x | 2749 | 2894 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 15680 | 7451 | 10176 | 5290 | 1.4x | 2749 | 2872 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 14770 | 10068 | 10647 | 6039 | 1.7x | 2993 | 2957 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 4658 | 3628 | unsupported | – | – | 3428 | – |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 4594 | 3654 | unsupported | – | – | 3414 | – |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 4609 | 3674 | unsupported | – | – | 3419 | – |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 4622 | 3649 | unsupported | – | – | 3439 | – |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 4609 | 3641 | unsupported | – | – | 3423 | – |
| [`las2-h`](../models/stereo.md#las2) | 888 | 537 | – | – | – | 1347 | – |
| [`las2-l`](../models/stereo.md#las2) | 477 | 355 | – | – | – | 1103 | – |
| [`las2-m`](../models/stereo.md#las2) | 252 | 190 | – | – | – | 1061 | – |
| [`las2-s`](../models/stereo.md#las2) | 290 | 230 | unsupported | – | – | 988 | – |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 10109 | 6508 | – | – | – | 2182 | – |
| [`mocha-v2-sceneflow`](../models/stereo.md#mocha_stereo) | 9856 | 6354 | – | – | – | 2187 | – |
| [`monster-mix-all`](../models/stereo.md#monster) | 13393 | 10082 | unsupported | – | – | 2915 | – |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 13670 | 10647 | unsupported | – | – | 3033 | – |
| [`monster-sceneflow`](../models/stereo.md#monster) | 13099 | 10064 | unsupported | – | – | 2941 | – |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 2224 | 1438 | unsupported | – | – | 2241 | – |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 3638 | 2597 | unsupported | – | – | 2941 | – |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 5143 | 4154 | 4521 | 2772 | 1.5x | 3714 | 4233 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 7097 | 5704 | – | – | – | 4315 | – |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 5336 | 4163 | 4662 | 2798 | 1.5x | 3867 | 4613 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 891 | 605 | 852 | 511 | 1.2x | 1453 | 1591 |
| [`s2m2-l`](../models/stereo.md#s2m2) | 8523 | 5464 | – | – | – | 5135 | – |
| [`s2m2-m`](../models/stereo.md#s2m2) | 6278 | 4085 | – | – | – | 3490 | – |
| [`s2m2-s`](../models/stereo.md#s2m2) | 4036 | 1835 | unsupported | – | – | 2779 | – |
| [`s2m2-xl`](../models/stereo.md#s2m2) | 13001 | 10264 | – | – | – | 8376 | – |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 10974 | 6535 | 9963 | 5218 | 1.3x | 1798 | 2522 |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | 10935 | 6465 | 9762 | 5353 | 1.2x | 1782 | 2710 |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | 10862 | 6358 | 10046 | 5422 | 1.2x | 1786 | 2711 |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | 10823 | 6467 | 9783 | 5238 | 1.2x | 1786 | 2531 |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | 10883 | 6343 | 10115 | 5222 | 1.2x | 1781 | 2717 |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | 10894 | 6450 | 10062 | 5266 | 1.2x | 1787 | 2523 |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 11924 | 6895 | 10715 | 6296 | 1.1x | 1814 | 2735 |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | 7982 | 5040 | – | – | – | 2731 | – |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | 5474 | 3509 | unsupported | – | – | 1835 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Depth

### Depth, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`video-depth-anything-metric-s`](../models/depth.md#video_depth_anything) | 1246 | 872 | 2021 |
| [`video-depth-anything-s`](../models/depth.md#video_depth_anything) | 1300 | 896 | 2027 |

### Depth, synthetic, 756x1008

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`prompt-depth-anything-s`](../models/depth.md#prompt_depth_anything) | 250 | 244 | 1168 |

### Depth, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | portable bf16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`any2full-b`](../models/depth.md#any2full) | 379 | 379 | – | – | – | – | 1583 | – |
| [`any2full-l`](../models/depth.md#any2full) | 864 | 909 | – | – | – | – | 3827 | – |
| [`any2full-s`](../models/depth.md#any2full) | 169 | 160 | – | – | – | – | 1012 | – |
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 827 | 647 | – | – | – | – | 3454 | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 317 | 268 | – | – | – | – | 1763 | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 286 | 231 | – | – | – | – | 1555 | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 288 | 231 | – | – | – | – | 1557 | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 1138 | 878 | – | – | – | – | 4123 | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 229 | 182 | – | 214 | 168 | 1.1x | 1026 | 1029 |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 669 | 522 | – | 628 | 501 | 1.0x | 1896 | 1768 |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | 237 | 184 | – | 216 | 170 | 1.1x | 1017 | 1027 |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | 700 | 527 | – | 641 | 581 | 0.9x | 1831 | 1737 |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | 96.4 | 79.8 | – | 106 | 68.5 | 1.2x | 737 | 710 |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | 291 | 188 | – | 232 | 198 | 0.9x | 1021 | 996 |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | 661 | 527 | – | 720 | 587 | 0.9x | 1924 | 1684 |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | 117 | 78.5 | – | 87.1 | 67.4 | 1.2x | 735 | 707 |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 93.0 | 73.4 | – | 88.2 | 66.3 | 1.1x | 727 | 698 |
| [`depth-pro`](../models/depth.md#depth_pro) | 7541 | 4653 | – | – | – | – | 5490 | – |
| [`depthfm-v1`](../models/depth.md#depthfm) | 75642 | 39312 | – | – | – | – | 8118 | – |
| [`distill-any-depth-b`](../models/depth.md#depth_anything_v2) | 231 | 184 | – | 216 | 170 | 1.1x | 1027 | 1020 |
| [`distill-any-depth-l`](../models/depth.md#depth_anything_v2) | 685 | 529 | – | 640 | 544 | 1.0x | 1847 | 1769 |
| [`distill-any-depth-s`](../models/depth.md#depth_anything_v2) | 94.2 | 74.5 | – | 85.2 | 66.9 | 1.1x | 728 | 698 |
| [`lingbot-depth`](../models/depth.md#lingbot_depth) | 4367 | 4577 | – | unsupported | – | – | 3930 | – |
| [`lingbot-depth-dc`](../models/depth.md#lingbot_depth) | 4414 | 5338 | – | – | – | – | 3849 | – |
| [`lotus-normal-d-v1-1`](../models/depth.md#lotus) | 2872 | 1973 | – | unsupported | – | – | 5670 | – |
| [`lotus-normal-g-v1-1`](../models/depth.md#lotus) | 3063 | 2005 | – | – | – | – | 5674 | – |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 4442 | 2868 | – | – | – | – | 4172 | – |
| [`marigold-normals-lcm-v0-1`](../models/depth.md#marigold_normals) | 2981 | 1772 | – | – | – | – | 4212 | – |
| [`marigold-v2-depth`](../models/depth.md#marigold_v2) | – | – | 51319 | – | – | – | 2251 | – |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 2586 | 2337 | – | – | – | – | 2943 | – |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 861 | 668 | – | – | – | – | 1994 | – |
| [`moge-2-vitb-normal`](../models/depth.md#moge) | 595 | 449 | – | unsupported | – | – | 1731 | – |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 1350 | 1027 | – | – | – | – | 2482 | – |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 372 | 271 | – | unsupported | – | – | 1449 | – |
| [`moge-3-vitl`](../models/depth.md#moge) | 2867 | 2573 | – | – | – | – | 3084 | – |
| [`omni-dc`](../models/depth.md#omni_dc) | 4033 | 3873 | – | – | – | – | 5387 | – |
| [`pointdit-b-512`](../models/depth.md#pointdit) | 343 | 356 | – | – | – | – | 1695 | – |
| [`pointdit-l-512`](../models/depth.md#pointdit) | 1047 | 1231 | – | – | – | – | 4003 | – |
| [`prompt-depth-anything-l`](../models/depth.md#prompt_depth_anything) | 1345 | 1022 | – | – | – | – | 2862 | – |
| [`stablenormal-turbo`](../models/depth.md#stablenormal) | 3589 | 2299 | – | unsupported | – | – | 8683 | – |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 1254 | 1173 | – | – | – | – | 1825 | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1230 | 947 | – | – | – | – | 1830 | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 957 | 759 | – | – | – | – | 1955 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Classification

### Classification, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 3.71 | 2.85 | – | – | – | 1024 | – |
| [`coatnet_0_rw_224.sw_in1k`](../models/classification.md#coatnet) | 12.0 | 9.66 | 11.3 | 9.65 | 1.0x | 735 | 745 |
| [`coatnet_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 20.2 | 16.0 | 22.2 | 16.0 | 1.0x | 966 | 970 |
| [`coatnet_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 30.4 | 23.5 | 32.0 | 23.0 | 1.0x | 1288 | 1311 |
| [`coatnet_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 9.29 | 7.62 | 9.35 | 7.88 | 1.0x | 600 | 605 |
| [`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 24.8 | 16.3 | 22.0 | 16.3 | 1.0x | 968 | 969 |
| [`coatnet_rmlp_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 21.2 | 16.1 | 21.1 | 16.1 | 1.0x | 968 | 975 |
| [`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 30.9 | 23.3 | 32.3 | 23.3 | 1.0x | 1306 | 1324 |
| [`coatnet_rmlp_2_rw_224.sw_in1k`](../models/classification.md#coatnet) | 30.3 | 23.2 | 30.6 | 23.0 | 1.0x | 1303 | 1322 |
| [`coatnet_rmlp_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 9.96 | 8.28 | 9.82 | 8.05 | 1.0x | 622 | 625 |
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 19.2 | 16.2 | 19.6 | 16.5 | 1.0x | 1212 | 1212 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 19.3 | 16.9 | 19.7 | 16.2 | 1.0x | 1212 | 1212 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 39.0 | 34.1 | 39.2 | 35.2 | 1.0x | 1210 | 1212 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 33.8 | 29.2 | 34.9 | 29.0 | 1.0x | 2341 | 2340 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 34.1 | 29.1 | 34.0 | 29.7 | 1.0x | 2340 | 2340 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 73.3 | 63.4 | 73.3 | 63.5 | 1.0x | 2340 | 2341 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 14.1 | 12.4 | 14.2 | 12.3 | 1.0x | 812 | 813 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 14.2 | 12.4 | 14.2 | 12.2 | 1.0x | 813 | 812 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 27.9 | 25.5 | 28.0 | 25.0 | 1.0x | 813 | 811 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 8.45 | 7.26 | 8.45 | 7.46 | 1.0x | 642 | 646 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 8.41 | 7.51 | 8.48 | 7.52 | 1.0x | 644 | 646 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 17.4 | 15.2 | 17.3 | 15.3 | 1.0x | 714 | 707 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 22.2 | 19.7 | 22.1 | 19.7 | 1.0x | 1190 | 1192 |
| [`davit_base.msft_in1k`](../models/classification.md#davit) | 22.6 | 18.9 | – | – | – | 1338 | – |
| [`davit_small.msft_in1k`](../models/classification.md#davit) | 16.9 | 14.4 | – | – | – | 952 | – |
| [`davit_tiny.msft_in1k`](../models/classification.md#davit) | 9.62 | 8.23 | – | – | – | 715 | – |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 16.6 | 14.4 | – | – | – | 1254 | – |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 12.0 | 9.93 | – | – | – | 414 | – |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 66.4 | 57.6 | unsupported | – | – | 3981 | – |
| [`efficientformer_l1.snap_dist_in1k`](../models/classification.md#efficientformer) | 5.70 | 5.22 | – | – | – | 534 | – |
| [`efficientformer_l3.snap_dist_in1k`](../models/classification.md#efficientformer) | 9.99 | 8.92 | – | – | – | 779 | – |
| [`efficientformer_l7.snap_dist_in1k`](../models/classification.md#efficientformer) | 19.1 | 17.2 | – | – | – | 1357 | – |
| [`efficientformerv2_l.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 13.5 | 11.7 | – | – | – | 718 | – |
| [`efficientformerv2_s0.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 4.78 | 4.22 | – | – | – | 421 | – |
| [`efficientformerv2_s1.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 6.41 | 5.66 | – | – | – | 466 | – |
| [`efficientformerv2_s2.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 9.48 | 8.50 | – | – | – | 551 | – |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 4.84 | 4.39 | 4.77 | 4.52 | 1.0x | 345 | 344 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 18.1 | 16.8 | 17.8 | 16.0 | 1.1x | 645 | 649 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 19.0 | 17.1 | 19.1 | 16.7 | 1.0x | 553 | 552 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 14.0 | 10.2 | – | – | – | 735 | – |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 245 | 199 | – | – | – | 3399 | – |
| [`falconsai-nsfw-vit`](../models/classification.md#nsfw_vit) | 16.4 | 14.0 | – | – | – | 992 | – |
| [`focalnet_base_lrf.ms_in1k`](../models/classification.md#focalnet) | 30.8 | 23.4 | – | – | – | 1395 | – |
| [`focalnet_base_srf.ms_in1k`](../models/classification.md#focalnet) | 27.8 | 22.4 | – | – | – | 1381 | – |
| [`focalnet_small_lrf.ms_in1k`](../models/classification.md#focalnet) | 20.9 | 18.0 | – | – | – | 977 | – |
| [`focalnet_small_srf.ms_in1k`](../models/classification.md#focalnet) | 19.6 | 16.9 | – | – | – | 967 | – |
| [`focalnet_tiny_lrf.ms_in1k`](../models/classification.md#focalnet) | 11.9 | 10.2 | – | – | – | 732 | – |
| [`focalnet_tiny_srf.ms_in1k`](../models/classification.md#focalnet) | 11.0 | 9.53 | – | – | – | 720 | – |
| [`hgnetv2_b0.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 3.43 | 2.96 | – | – | – | 456 | – |
| [`hgnetv2_b1.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 3.92 | 3.45 | – | – | – | 463 | – |
| [`hgnetv2_b2.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 5.44 | 5.48 | – | – | – | 599 | – |
| [`hgnetv2_b3.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 6.64 | 7.85 | – | – | – | 712 | – |
| [`hgnetv2_b4.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 7.58 | 6.27 | – | – | – | 710 | – |
| [`hgnetv2_b5.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 12.9 | 11.1 | – | – | – | 1063 | – |
| [`hgnetv2_b6.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 27.1 | 21.8 | – | – | – | 1542 | – |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 13.5 | 11.2 | – | – | – | 642 | – |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 3.47 | 2.78 | 3.25 | 2.80 | 1.0x | 319 | 318 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 3.20 | 3.31 | 3.31 | 4.51 | 1.0x | 339 | 338 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 2.85 | 2.29 | – | – | – | 367 | – |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 5.67 | 4.65 | – | – | – | 507 | – |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 19.7 | 18.9 | 19.9 | 18.9 | 1.0x | 1047 | 1046 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 4.72 | 4.31 | – | – | – | 384 | – |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 4.65 | 4.30 | – | – | – | 384 | – |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 5.61 | 5.20 | – | – | – | 410 | – |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 5.78 | 5.29 | – | – | – | 414 | – |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 5.06 | 4.38 | – | – | – | 425 | – |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 5.11 | 4.41 | – | – | – | 425 | – |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 8.98 | 12.1 | – | – | – | 548 | – |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 8.86 | 6.88 | – | – | – | 506 | – |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 12.1 | 11.1 | – | – | – | 630 | – |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 12.1 | 10.7 | – | – | – | 628 | – |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 20.5 | 17.1 | 20.2 | 17.0 | 1.0x | 729 | 728 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 5.26 | 3.77 | 4.37 | 3.75 | 1.0x | 383 | 382 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 7.94 | 6.77 | 7.92 | 6.76 | 1.0x | 503 | 500 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 3.47 | 3.27 | – | – | – | 309 | – |
| [`shvit_s1.in1k`](../models/classification.md#shvit) | 4.93 | 4.43 | – | – | – | 439 | – |
| [`shvit_s2.in1k`](../models/classification.md#shvit) | 5.76 | 5.11 | – | – | – | 509 | – |
| [`shvit_s3.in1k`](../models/classification.md#shvit) | 6.58 | 5.69 | – | – | – | 538 | – |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 2.34 | 1.96 | – | – | – | 297 | – |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 5.95 | 3.15 | – | – | – | 342 | – |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 5.08 | 2.74 | – | – | – | 348 | – |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 5.28 | 3.31 | – | – | – | 375 | – |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 5.93 | 4.37 | – | – | – | 397 | – |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 22.3 | 19.2 | – | – | – | 1366 | – |
| [`tiny_vit_11m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 5.95 | 5.22 | – | – | – | 516 | – |
| [`tiny_vit_21m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 8.75 | 7.44 | – | – | – | 639 | – |
| [`tiny_vit_5m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 5.05 | 4.64 | – | – | – | 451 | – |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 17.2 | 13.8 | – | – | – | 2521 | – |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 16.7 | 14.3 | – | – | – | 1252 | – |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 211 | 179 | 211 | 179 | 1.0x | 4547 | 4547 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Classification, synthetic, 256x256

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 10.3 | 8.69 | 10.3 | 8.70 | 1.0x | 623 | 623 |
| [`fastvit_t12.apple_dist_in1k`](../models/classification.md#fastvit) | 5.34 | 4.57 | 5.15 | 4.56 | 1.0x | 449 | 448 |
| [`fastvit_t8.apple_dist_in1k`](../models/classification.md#fastvit) | 4.26 | 3.64 | 4.28 | 3.69 | 1.0x | 404 | 405 |
| [`shvit_s4.in1k`](../models/classification.md#shvit) | 8.22 | 7.82 | – | – | – | 572 | – |

### Classification, synthetic, 384x384

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 82.2 | 56.8 | 84.1 | 57.5 | 1.0x | 1543 | 1560 |
| [`tiny_vit_21m_384.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 19.8 | 15.4 | – | – | – | 779 | – |

### Classification, synthetic, 448x448

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`wd-convnext-tagger-v3`](../models/classification.md#wd_tagger) | 57.6 | 47.3 | 59.1 | 47.0 | 1.0x | 1519 | 1521 |
| [`wd-eva02-large-tagger-v3`](../models/classification.md#wd_tagger) | 254 | 206 | 261 | 207 | 1.0x | 3502 | 3506 |
| [`wd-vit-large-tagger-v3`](../models/classification.md#wd_tagger) | 207 | 183 | 211 | 179 | 1.0x | 3502 | 3518 |
| [`wd-vit-tagger-v3`](../models/classification.md#wd_tagger) | 56.5 | 45.6 | 56.6 | 46.2 | 1.0x | 1342 | 1344 |

### Classification, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`tiny_vit_21m_512.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 39.5 | 26.9 | 973 |

## Segmentation

### Segmentation, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`bisenet-face-parsing`](../models/segmentation.md#bisenet) | 21.0 | 17.6 | – | – | – | 607 | – |
| [`segformer-b2-clothes`](../models/segmentation.md#segformer) | 65.4 | 42.4 | 63.4 | 40.6 | 1.0x | 724 | 760 |
| [`segformer-b5-face-parsing`](../models/segmentation.md#segformer) | 137 | 101 | – | – | – | 963 | – |

### Segmentation, synthetic, 1024x2048

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 668 | 442 | 675 | 441 | 1.0x | 1404 | 1426 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 613 | 444 | 505 | 370 | 1.2x | 1618 | 1358 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 312 | 256 | unsupported | – | – | 1372 | – |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 937 | 761 | unsupported | – | – | 2223 | – |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 132 | 112 | unsupported | – | – | 1002 | – |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 667 | 574 | – | – | – | 3661 | – |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 765 | 599 | – | – | – | 3724 | – |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 490 | 353 | 490 | 358 | 1.0x | 1357 | 1373 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 282 | 220 | – | – | – | 1112 | – |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 1033 | 850 | – | – | – | 3614 | – |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 1741 | 2077 | – | – | – | 6005 | – |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 510 | 380 | 511 | 375 | 1.0x | 1441 | 1450 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 2664 | 2392 | – | – | – | 3921 | – |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 5247 | 4110 | – | – | – | 6692 | – |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 94.8 | 78.9 | 91.9 | 77.5 | 1.0x | 776 | 764 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 470 | 357 | – | – | – | 1382 | – |
| [`segnext-mscan-b-ade20k`](../models/segmentation.md#segnext) | 301 | 264 | – | – | – | 894 | – |
| [`segnext-mscan-l-ade20k`](../models/segmentation.md#segnext) | 489 | 424 | – | – | – | 974 | – |
| [`segnext-mscan-s-ade20k`](../models/segmentation.md#segnext) | 178 | 153 | 106 | 84.9 | 1.8x | 828 | 820 |
| [`segnext-mscan-t-ade20k`](../models/segmentation.md#segnext) | 133 | 117 | 85.1 | 67.7 | 1.7x | 769 | 754 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matting

### Matting, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ben2`](../models/matting.md#ben2) | 1524 | 850 | unsupported | – | – | 2863 | – |
| [`birefnet`](../models/matting.md#birefnet) | 1625 | 1048 | unsupported | – | – | 2364 | – |
| [`birefnet-lite`](../models/matting.md#birefnet) | 640 | 392 | – | – | – | 1824 | – |
| [`inspyrenet`](../models/matting.md#inspyrenet) | 1121 | 679 | unsupported | – | – | 1741 | – |
| [`inspyrenet-fast`](../models/matting.md#inspyrenet) | 129 | 95.6 | – | – | – | 1060 | – |
| [`isnet`](../models/matting.md#isnet) | 197 | 145 | 197 | 121 | 1.2x | 1200 | 1254 |
| [`isnet-general-use`](../models/matting.md#isnet) | 211 | 145 | 209 | 122 | 1.2x | 1217 | 1328 |
| [`matanyone`](../models/matting.md#matanyone) | 5111 | 3495 | – | – | – | 3609 | – |
| [`rmbg-2.0`](../models/matting.md#birefnet) | 1643 | 1050 | – | – | – | 2508 | – |
| [`u2net`](../models/matting.md#u2net) | 82.5 | 62.9 | unsupported | – | – | 1156 | – |
| [`u2net-human-seg`](../models/matting.md#u2net) | 78.6 | 62.7 | unsupported | – | – | 972 | – |
| [`u2netp`](../models/matting.md#u2net) | 35.0 | 28.3 | – | – | – | 574 | – |
| [`vitmatte-b`](../models/matting.md#vitmatte) | 2370 | 1652 | – | – | – | 1741 | – |
| [`vitmatte-s`](../models/matting.md#vitmatte) | 1181 | 816 | unsupported | – | – | 1622 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Edges

### Edges, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`anyline`](../models/edges.md#anyline) | 223 | 197 | 219 | 181 | 1.1x | 1776 | 1604 |
| [`hed`](../models/edges.md#hed) | 74.9 | 65.2 | unsupported | – | – | 1174 | – |
| [`lineart`](../models/edges.md#lineart) | 134 | 108 | unsupported | – | – | 1099 | – |
| [`lineart-anime`](../models/edges.md#lineart_anime) | 51.8 | 44.5 | unsupported | – | – | 982 | – |
| [`lineart-coarse`](../models/edges.md#lineart) | 98.1 | 78.9 | unsupported | – | – | 933 | – |
| [`mlsd`](../models/edges.md#mlsd) | 46.3 | 35.8 | unsupported | – | – | 676 | – |
| [`pidinet`](../models/edges.md#pidinet) | 196 | 173 | 107 | 75.1 | 2.3x | 728 | 732 |
| [`teed`](../models/edges.md#teed) | 19.6 | 19.2 | 17.5 | 14.1 | 1.4x | 492 | 523 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`clipseg`](../models/promptable_segmentation.md#clipseg) | 33.6 | 38.5 | – | 938 |
| [`efficientvit-sam-l0`](../models/promptable_segmentation.md#efficientvit_sam) | 113 | – | – | 1102 |
| [`efficientvit-sam-l1`](../models/promptable_segmentation.md#efficientvit_sam) | 101 | – | – | 1127 |
| [`efficientvit-sam-l2`](../models/promptable_segmentation.md#efficientvit_sam) | 120 | – | – | 1253 |
| [`efficientvit-sam-xl0`](../models/promptable_segmentation.md#efficientvit_sam) | 268 | – | – | 1968 |
| [`efficientvit-sam-xl1`](../models/promptable_segmentation.md#efficientvit_sam) | 435 | – | – | 2595 |
| [`insid3-vitl16`](../models/promptable_segmentation.md#insid3) | 3216 | 2512 | – | 2227 |
| [`mobile-sam`](../models/promptable_segmentation.md#mobile_sam) | 113 | 83.1 | unsupported | 807 |
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | 648 | 493 | – | 1464 |
| [`sam-hq-vit-h`](../models/promptable_segmentation.md#sam_hq) | 2462 | 1901 | unsupported | 2602 |
| [`sam-hq-vit-l`](../models/promptable_segmentation.md#sam_hq) | 1350 | 1051 | – | 1900 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 600 | 441 | – | 1150 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 2413 | 1884 | unsupported | 2387 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1282 | 1010 | – | 1644 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 352 | 282 | – | 1099 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 713 | 597 | – | 1460 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 204 | 164 | – | 951 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 176 | 141 | – | 933 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 2062 | 1657 | unsupported | 2768 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`deim-l-coco`](../models/detection.md#dfine) | 73.6 | 59.7 | – | – | – | 872 | – |
| [`deim-m-coco`](../models/detection.md#dfine) | 58.4 | 46.9 | – | – | – | 809 | – |
| [`deim-n-coco`](../models/detection.md#dfine) | 23.3 | 21.0 | – | – | – | 602 | – |
| [`deim-s-coco`](../models/detection.md#dfine) | 35.8 | 30.8 | – | – | – | 674 | – |
| [`deim-x-coco`](../models/detection.md#dfine) | 120 | 94.8 | – | – | – | 1117 | – |
| [`detr-r101-coco`](../models/detection.md#detr) | 181 | 144 | – | – | – | 1397 | – |
| [`detr-r50-coco`](../models/detection.md#detr) | 118 | 97.2 | unsupported | – | – | 1414 | – |
| [`dfine-l-coco`](../models/detection.md#dfine) | 74.4 | 59.9 | – | – | – | 881 | – |
| [`dfine-m-coco`](../models/detection.md#dfine) | 58.4 | 46.6 | – | – | – | 821 | – |
| [`dfine-n-coco`](../models/detection.md#dfine) | 25.7 | 22.3 | unsupported | – | – | 624 | – |
| [`dfine-s-coco`](../models/detection.md#dfine) | 37.1 | 30.1 | – | – | – | 673 | – |
| [`dfine-x-coco`](../models/detection.md#dfine) | 120 | 94.6 | – | – | – | 1133 | – |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 410 | 367 | – | – | – | 1694 | – |
| [`grounding-dino-base`](../models/detection.md#grounding_dino) | 1189 | 889 | unsupported | – | – | 1687 | – |
| [`grounding-dino-tiny`](../models/detection.md#grounding_dino) | 933 | 753 | unsupported | – | – | 1499 | – |
| [`lw-detr-large-coco`](../models/detection.md#lwdetr) | 70.8 | 64.6 | – | – | – | 891 | – |
| [`lw-detr-medium-coco`](../models/detection.md#lwdetr) | 47.4 | 48.0 | – | – | – | 773 | – |
| [`lw-detr-small-coco`](../models/detection.md#lwdetr) | 33.3 | 33.6 | – | – | – | 736 | – |
| [`lw-detr-tiny-coco`](../models/detection.md#lwdetr) | 37.7 | 34.9 | unsupported | – | – | 935 | – |
| [`lw-detr-xlarge-coco`](../models/detection.md#lwdetr) | 123 | 117 | – | – | – | 1077 | – |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 462 | 411 | unsupported | – | – | 1533 | – |
| [`mm-grounding-dino-base`](../models/detection.md#grounding_dino) | 1254 | 916 | – | – | – | 1719 | – |
| [`mm-grounding-dino-tiny`](../models/detection.md#grounding_dino) | 935 | 766 | – | – | – | 1507 | – |
| [`owlv2-b16`](../models/detection.md#owlv2) | 390 | 294 | unsupported | – | – | 991 | – |
| [`owlv2-l14`](../models/detection.md#owlv2) | 1885 | 1423 | unsupported | – | – | 1755 | – |
| [`rf-detr-l-coco`](../models/detection.md#rfdetr) | 56.4 | 46.9 | unsupported | – | – | 731 | – |
| [`rf-detr-m-coco`](../models/detection.md#rfdetr) | 39.8 | 33.3 | unsupported | – | – | 740 | – |
| [`rf-detr-n-coco`](../models/detection.md#rfdetr) | 21.0 | 19.0 | unsupported | – | – | 657 | – |
| [`rf-detr-s-coco`](../models/detection.md#rfdetr) | 32.6 | 28.0 | unsupported | – | – | 679 | – |
| [`rf-detr-seg-2xl-coco`](../models/detection.md#rfdetr) | 181 | 137 | unsupported | – | – | 936 | – |
| [`rf-detr-seg-l-coco`](../models/detection.md#rfdetr) | 65.1 | 50.6 | unsupported | – | – | 814 | – |
| [`rf-detr-seg-m-coco`](../models/detection.md#rfdetr) | 50.9 | 39.9 | unsupported | – | – | 821 | – |
| [`rf-detr-seg-n-coco`](../models/detection.md#rfdetr) | 27.8 | 23.8 | unsupported | – | – | 730 | – |
| [`rf-detr-seg-s-coco`](../models/detection.md#rfdetr) | 38.0 | 30.2 | unsupported | – | – | 729 | – |
| [`rf-detr-seg-xl-coco`](../models/detection.md#rfdetr) | 117 | 84.7 | unsupported | – | – | 859 | – |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 88.9 | 67.8 | – | – | – | 815 | – |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 64.8 | 50.0 | – | – | – | 811 | – |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 71.8 | 57.4 | – | – | – | 791 | – |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 48.1 | 39.5 | unsupported | – | – | 904 | – |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 139 | 118 | – | – | – | 953 | – |
| [`rtmdet-l-coco`](../models/detection.md#rtmdet) | 98.7 | 74.9 | – | – | – | 928 | – |
| [`rtmdet-m-coco`](../models/detection.md#rtmdet) | 60.2 | 50.1 | – | – | – | 921 | – |
| [`rtmdet-m-person`](../models/detection.md#rtmdet) | 60.1 | 47.7 | – | – | – | 880 | – |
| [`rtmdet-nano-person`](../models/detection.md#rtmdet) | 9.26 | 8.87 | – | – | – | 417 | – |
| [`rtmdet-s-coco`](../models/detection.md#rtmdet) | 34.5 | 26.1 | – | – | – | 720 | – |
| [`rtmdet-tiny-coco`](../models/detection.md#rtmdet) | 32.4 | 24.4 | unsupported | – | – | 618 | – |
| [`rtmdet-x-coco`](../models/detection.md#rtmdet) | 163 | 126 | – | – | – | 1219 | – |
| [`yolox-l`](../models/detection.md#yolox) | 92.9 | 69.6 | – | – | – | 851 | – |
| [`yolox-m`](../models/detection.md#yolox) | 51.8 | 44.6 | – | – | – | 736 | – |
| [`yolox-s`](../models/detection.md#yolox) | 28.8 | 23.0 | – | – | – | 615 | – |
| [`yolox-tiny`](../models/detection.md#yolox) | 15.1 | 13.4 | 14.5 | 12.6 | 1.1x | 454 | 470 |
| [`yolox-x`](../models/detection.md#yolox) | 152 | 121 | – | – | – | 1229 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Ocr

### Ocr, synthetic, 32x128

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`parseq`](../models/ocr.md#parseq) | 20.1 | 17.9 | 495 |
| [`parseq-tiny`](../models/ocr.md#parseq) | 8.10 | 7.46 | 407 |

### Ocr, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`ppocrv5-mobile`](../models/ocr.md#ppocrv5) | 1251 | 869 | 1610 |
| [`ppocrv5-server`](../models/ocr.md#ppocrv5) | 2691 | 1904 | 2666 |

## Pose

### Pose, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`dwpose-l`](../models/pose.md#rtmpose) | 105 | 81.3 | – | 1157 |
| [`dwpose-l-384`](../models/pose.md#rtmpose) | 217 | 165 | – | 1207 |
| [`dwpose-m`](../models/pose.md#rtmpose) | 69.8 | 54.5 | – | 968 |
| [`dwpose-s`](../models/pose.md#rtmpose) | 40.3 | 34.4 | – | 793 |
| [`dwpose-t`](../models/pose.md#rtmpose) | 38.6 | 28.6 | – | 761 |
| [`hrnet-w32`](../models/pose.md#hrnet) | 159 | 117 | unsupported | 911 |
| [`hrnet-w32-384`](../models/pose.md#hrnet) | 327 | 237 | – | 1104 |
| [`hrnet-w48`](../models/pose.md#hrnet) | 285 | 222 | – | 897 |
| [`hrnet-w48-384`](../models/pose.md#hrnet) | 597 | 449 | – | 1058 |
| [`retinaface-mobilenet0.25`](../models/pose.md#retinaface) | 23.8 | 19.8 | – | 621 |
| [`retinaface-resnet50`](../models/pose.md#retinaface) | 263 | 185 | unsupported | 1165 |
| [`rf-detr-keypoint-preview`](../models/pose.md#rfdetr_keypoint) | 81.4 | 67.0 | unsupported | 915 |
| [`rtmo-l`](../models/pose.md#rtmo) | 88.9 | 71.9 | unsupported | 1171 |
| [`rtmo-m`](../models/pose.md#rtmo) | 53.2 | 45.0 | unsupported | 999 |
| [`rtmo-s`](../models/pose.md#rtmo) | 33.8 | 25.6 | unsupported | 763 |
| [`rtmo-t`](../models/pose.md#rtmo) | 16.2 | 14.4 | unsupported | 596 |
| [`rtmpose-l`](../models/pose.md#rtmpose) | 98.4 | 75.1 | – | 1094 |
| [`rtmpose-l-384`](../models/pose.md#rtmpose) | 205 | 150 | – | 1157 |
| [`rtmpose-l-halpe26`](../models/pose.md#rtmpose) | 98.8 | 75.5 | – | 1096 |
| [`rtmpose-l-halpe26-384`](../models/pose.md#rtmpose) | 205 | 151 | – | 1161 |
| [`rtmpose-m`](../models/pose.md#rtmpose) | 64.3 | 50.7 | – | 933 |
| [`rtmpose-m-384`](../models/pose.md#rtmpose) | 142 | 108 | – | 1047 |
| [`rtmpose-m-ap10k`](../models/pose.md#rtmpose) | 85.0 | 65.4 | – | 937 |
| [`rtmpose-m-face6`](../models/pose.md#rtmpose) | 88.0 | 67.7 | – | 966 |
| [`rtmpose-m-halpe26`](../models/pose.md#rtmpose) | 65.3 | 51.1 | – | 939 |
| [`rtmpose-m-halpe26-384`](../models/pose.md#rtmpose) | 143 | 106 | – | 1057 |
| [`rtmpose-m-hand5`](../models/pose.md#rtmpose) | 84.6 | 65.1 | – | 940 |
| [`rtmpose-s`](../models/pose.md#rtmpose) | 34.4 | 30.5 | – | 730 |
| [`rtmpose-s-face6`](../models/pose.md#rtmpose) | 45.4 | 39.6 | – | 805 |
| [`rtmpose-s-halpe26`](../models/pose.md#rtmpose) | 40.7 | 31.0 | – | 733 |
| [`rtmpose-t`](../models/pose.md#rtmpose) | 30.3 | 25.2 | unsupported | 709 |
| [`rtmpose-t-face6`](../models/pose.md#rtmpose) | 38.4 | 33.1 | – | 798 |
| [`rtmpose-t-halpe26`](../models/pose.md#rtmpose) | 33.8 | 25.2 | – | 716 |
| [`rtmpose-x-halpe26-384`](../models/pose.md#rtmpose) | 367 | 257 | – | 1675 |
| [`rtmw-l`](../models/pose.md#rtmpose) | 157 | 128 | – | 1331 |
| [`rtmw-l-384`](../models/pose.md#rtmpose) | 348 | 280 | – | 1379 |
| [`rtmw-m`](../models/pose.md#rtmpose) | 110 | 87.2 | – | 1037 |
| [`rtmw-x`](../models/pose.md#rtmpose) | 293 | 236 | – | 1736 |
| [`rtmw-x-384`](../models/pose.md#rtmpose) | 613 | 472 | – | 2040 |
| [`vitpose-base`](../models/pose.md#vitpose) | 178 | 158 | – | 831 |
| [`vitpose-base-simple`](../models/pose.md#vitpose) | 202 | 169 | unsupported | 1080 |
| [`vitpose-plus-base`](../models/pose.md#vitpose) | 179 | 158 | – | 876 |
| [`vitpose-plus-huge`](../models/pose.md#vitpose) | 1005 | 877 | unsupported | 3175 |
| [`vitpose-plus-large`](../models/pose.md#vitpose) | 504 | 452 | unsupported | 1775 |
| [`vitpose-plus-small`](../models/pose.md#vitpose) | 84.6 | 72.3 | – | 712 |
| [`yunet-n`](../models/pose.md#yunet) | 30.2 | 22.3 | unsupported | 717 |
| [`yunet-s`](../models/pose.md#yunet) | 21.9 | 17.7 | – | 703 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Features

### Features, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](../models/features.md#c_radio) | 16.9 | 16.4 | – | – | – | 764 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | 190 | 158 | – | – | – | 4960 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | 109 | 93.5 | – | – | – | 2983 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | 55.1 | 47.9 | – | – | – | 1635 | – |
| [`c-radio-v4-h`](../models/features.md#c_radio) | 104 | 88.0 | unsupported | – | – | 2947 | – |
| [`c-radio-v4-so400m`](../models/features.md#c_radio) | 71.5 | 62.2 | unsupported | – | – | 2077 | – |
| [`clip-b16-laion2b`](../models/features.md#clip) | 16.5 | 14.2 | – | – | – | 1245 | – |
| [`clip-b16-openai`](../models/features.md#clip) | 18.8 | 15.3 | – | – | – | 1478 | – |
| [`clip-h14-378-dfn5b`](../models/features.md#clip) | 126 | 105 | unsupported | – | – | 5248 | – |
| [`clip-h14-dfn5b`](../models/features.md#clip) | 126 | 105 | unsupported | – | – | 5255 | – |
| [`clip-h14-laion2b`](../models/features.md#clip) | 125 | 105 | unsupported | – | – | 5245 | – |
| [`clip-l14-336-openai`](../models/features.md#clip) | 63.0 | 54.5 | – | – | – | 2964 | – |
| [`clip-l14-datacompxl`](../models/features.md#clip) | 63.1 | 54.6 | – | – | – | 2962 | – |
| [`clip-l14-dfn2b`](../models/features.md#clip) | 62.8 | 54.4 | – | – | – | 2961 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | 63.0 | 54.5 | – | – | – | 2958 | – |
| [`clip-l14-openai`](../models/features.md#clip) | 62.9 | 54.7 | unsupported | – | – | 3425 | – |
| [`convnext_base.fb_in22k`](../models/features.md#convnext_features) | 20.8 | 18.1 | 20.6 | 20.4 | 0.9x | 1368 | 1366 |
| [`convnext_large.fb_in22k`](../models/features.md#convnext_features) | 35.5 | 33.8 | 36.4 | 29.2 | 1.2x | 2575 | 2574 |
| [`convnext_small.fb_in22k`](../models/features.md#convnext_features) | 14.3 | 12.1 | 14.8 | 12.2 | 1.0x | 931 | 930 |
| [`convnext_tiny.fb_in22k`](../models/features.md#convnext_features) | 8.54 | 7.50 | 8.27 | 7.47 | 1.0x | 760 | 761 |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 20.5 | 18.6 | unsupported | – | – | 607 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 20.6 | 18.0 | unsupported | – | – | 601 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 227 | 193 | unsupported | – | – | 2634 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 234 | 203 | unsupported | – | – | 2630 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 64.3 | 55.5 | unsupported | – | – | 1011 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 75.2 | 58.2 | unsupported | – | – | 1029 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 8.24 | 7.21 | unsupported | – | – | 398 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 8.29 | 7.22 | unsupported | – | – | 398 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 19.6 | 17.1 | 21.4 | 16.0 | 1.1x | 512 | 512 |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 35.4 | 29.9 | 35.2 | 34.1 | 0.9x | 822 | 824 |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 14.6 | 11.9 | 14.0 | 12.1 | 1.0x | 388 | 387 |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 8.01 | 7.23 | 8.51 | 8.19 | 0.9x | 315 | 314 |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 17.1 | 15.0 | unsupported | – | – | 576 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 155 | 120 | unsupported | – | – | 2174 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 53.3 | 45.8 | unsupported | – | – | 1017 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 7.32 | 6.53 | unsupported | – | – | 385 | – |
| [`pe-core-b16-224`](../models/features.md#perception_encoder) | 18.0 | 22.9 | – | – | – | 1509 | – |
| [`pe-core-g14-448`](../models/features.md#perception_encoder) | 296 | 363 | unsupported | – | – | 14856 | – |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | 67.0 | 85.5 | – | – | – | 3712 | – |
| [`pe-core-s16-384`](../models/features.md#perception_encoder) | 7.59 | 9.64 | – | – | – | 708 | – |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | 4.75 | 5.11 | – | – | – | 456 | – |
| [`pe-spatial-b16-512`](../models/features.md#perception_encoder) | 17.6 | 21.9 | – | – | – | 1431 | – |
| [`pe-spatial-g14-448`](../models/features.md#perception_encoder) | 291 | 375 | unsupported | – | – | 14815 | – |
| [`pe-spatial-l14-448`](../models/features.md#perception_encoder) | 66.3 | 80.1 | – | – | – | 3629 | – |
| [`pe-spatial-s16-512`](../models/features.md#perception_encoder) | 7.38 | 8.21 | – | – | – | 682 | – |
| [`pe-spatial-t16-512`](../models/features.md#perception_encoder) | 4.31 | 4.42 | – | – | – | 452 | – |
| [`siglip2-b16-256`](../models/features.md#siglip2) | 19.5 | 16.6 | – | – | – | 1306 | – |
| [`siglip2-b16-512`](../models/features.md#siglip2) | 17.1 | 14.7 | – | – | – | 1322 | – |
| [`siglip2-g16-256`](../models/features.md#siglip2) | 183 | 150 | unsupported | – | – | 9161 | – |
| [`siglip2-g16-384`](../models/features.md#siglip2) | 191 | 154 | unsupported | – | – | 9166 | – |
| [`siglip2-l16-256`](../models/features.md#siglip2) | 51.8 | 44.5 | – | – | – | 3010 | – |
| [`siglip2-l16-512`](../models/features.md#siglip2) | 52.1 | 44.7 | – | – | – | 3017 | – |
| [`siglip2-so400m14-224`](../models/features.md#siglip2) | 74.6 | 65.7 | unsupported | – | – | 3796 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | 72.4 | 66.4 | – | – | – | 3825 | – |
| [`siglip2-so400m16-256`](../models/features.md#siglip2) | 78.8 | 65.2 | unsupported | – | – | 3801 | – |
| [`siglip2-so400m16-512`](../models/features.md#siglip2) | 75.0 | 64.0 | unsupported | – | – | 3806 | – |
| [`tipsv2-b14`](../models/features.md#tipsv2) | 20.6 | 17.8 | unsupported | – | – | 598 | – |
| [`tipsv2-l14`](../models/features.md#tipsv2) | 68.5 | 72.9 | – | – | – | 1665 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Retrieval

### Retrieval, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`boq-dinov2`](../models/retrieval.md#boq) | 52.1 | 55.3 | 910 |
| [`megaloc`](../models/retrieval.md#megaloc) | 55.3 | 87.3 | 1420 |

## Calibration

### Calibration, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`anycalib-pinhole`](../models/calibration.md#anycalib) | 342 | 439 | 2089 |
| [`geocalib-pinhole`](../models/calibration.md#geocalib) | 416 | 352 | 1013 |

## Multiview

### Multiview, synthetic, 256x256

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`resplat-base-re10k`](../models/multiview.md#resplat) | 19441 | 14649 | 5639 |

### Multiview, synthetic, 256x448

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`resplat-small-dl3dv`](../models/multiview.md#resplat) | 2804 | 2113 | 1760 |

### Multiview, synthetic, 336x518

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`anysplat`](../models/multiview.md#anysplat) | 6109 | 4824 | – | – | – | 6370 | – |
| [`depthsplat-small-re10k`](../models/multiview.md#depthsplat) | 820 | 709 | – | – | – | 1716 | – |
| [`mapanything-apache`](../models/multiview.md#mapanything) | 9055 | 5993 | – | – | – | 5514 | – |
| [`mvsplat-re10k`](../models/multiview.md#mvsplat) | 1363 | 1145 | 1375 | 973 | 1.2x | 2342 | 2848 |
| [`pi3`](../models/multiview.md#pi3) | 10305 | 7752 | – | – | – | 3102 | – |
| [`vggt-1b`](../models/multiview.md#vggt) | 7805 | 5659 | – | – | – | 5158 | – |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | 7746 | 5853 | – | – | – | 4289 | – |

### Multiview, synthetic, 448x672

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | 619 | 467 | 1467 |

### Multiview, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`lingbot-map`](../models/multiview.md#lingbot_map) | 9970 | 7261 | 5438 |

### Multiview, synthetic, 1536x1024

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 1612 | 920 | 2247 |

## Flow

### Flow, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`flowformer-small-chairs`](../models/flow.md#flowformer) | 19772 | 13009 | unsupported | 1920 |
| [`flowformer-small-sintel`](../models/flow.md#flowformer) | 19787 | 13044 | unsupported | 1920 |
| [`flowformer-small-things`](../models/flow.md#flowformer) | 19854 | 13011 | unsupported | 1915 |
| [`flowseek-t`](../models/flow.md#flowseek) | 3246 | 2160 | unsupported | 5781 |
| [`flowseek-t-things`](../models/flow.md#flowseek) | 2523 | 1760 | unsupported | 5769 |
| [`flowseek-t-things-notartan`](../models/flow.md#flowseek) | 2544 | 1760 | unsupported | 5766 |
| [`gmflow-scale1-mixdata`](../models/flow.md#gmflow) | 6050 | 4340 | unsupported | 12170 |
| [`gmflow-scale1-things`](../models/flow.md#gmflow) | 5909 | 3871 | unsupported | 9100 |
| [`gmflow-scale2-mixdata`](../models/flow.md#gmflow) | 12373 | 8276 | – | 11508 |
| [`gmflow-scale2-regrefine6-kitti`](../models/flow.md#gmflow) | 16667 | 11251 | – | 11324 |
| [`gmflow-scale2-regrefine6-mixdata`](../models/flow.md#gmflow) | 16974 | 11296 | – | 10699 |
| [`gmflow-scale2-regrefine6-sintelft`](../models/flow.md#gmflow) | 16766 | 10975 | – | 12341 |
| [`gmflow-scale2-regrefine6-things`](../models/flow.md#gmflow) | 16556 | 11280 | – | 10551 |
| [`gmflow-scale2-sintel`](../models/flow.md#gmflow) | 13073 | 8283 | – | 11046 |
| [`gmflow-scale2-things`](../models/flow.md#gmflow) | 13356 | 8261 | – | 11819 |
| [`megaflow`](../models/flow.md#megaflow) | 13158 | 8480 | – | 6561 |
| [`memflow`](../models/flow.md#memflow) | 9908 | 6997 | unsupported | 8345 |
| [`memflow-kitti`](../models/flow.md#memflow) | 9851 | 6988 | – | 7586 |
| [`memflow-sintel`](../models/flow.md#memflow) | 9860 | 6990 | – | 8075 |
| [`memfof`](../models/flow.md#memfof) | 3807 | 3317 | unsupported | 1986 |
| [`memfof-kitti`](../models/flow.md#memfof) | 19707 | 16074 | – | 3652 |
| [`memfof-sintel`](../models/flow.md#memfof) | 19915 | 16049 | – | 3643 |
| [`memfof-spring`](../models/flow.md#memfof) | 3774 | 3319 | unsupported | 1986 |
| [`neuflow-v2`](../models/flow.md#neuflow_v2) | 951 | 949 | unsupported | 9200 |
| [`neuflow-v2-sintel`](../models/flow.md#neuflow_v2) | 1021 | 952 | unsupported | 9201 |
| [`neuflow-v2-things`](../models/flow.md#neuflow_v2) | 975 | 946 | unsupported | 9170 |
| [`raft-large`](../models/flow.md#raft) | 1552 | 1028 | unsupported | 2069 |
| [`raft-large-kitti`](../models/flow.md#raft) | 1443 | 1094 | unsupported | 2056 |
| [`raft-large-things`](../models/flow.md#raft) | 1635 | 1127 | unsupported | 2036 |
| [`raft-small`](../models/flow.md#raft) | 450 | 332 | unsupported | 1645 |
| [`recover-cx`](../models/flow.md#recover) | 1064 | 758 | unsupported | 1297 |
| [`recover-mn`](../models/flow.md#recover) | 471 | 316 | unsupported | 1181 |
| [`recover-rn`](../models/flow.md#recover) | 624 | 429 | unsupported | 1224 |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 2003 | 1387 | unsupported | 5600 |
| [`sea-raft-m-kitti`](../models/flow.md#sea_raft) | 1958 | 1410 | unsupported | 5622 |
| [`sea-raft-m-spring`](../models/flow.md#sea_raft) | 369 | 298 | – | 2051 |
| [`sea-raft-m-things`](../models/flow.md#sea_raft) | 2396 | 1428 | unsupported | 5620 |
| [`sea-raft-s`](../models/flow.md#sea_raft) | 1584 | 1069 | – | 5642 |
| [`sea-raft-s-kitti`](../models/flow.md#sea_raft) | 1585 | 1083 | – | 5640 |
| [`sea-raft-s-spring`](../models/flow.md#sea_raft) | 251 | 198 | – | 1817 |
| [`waft-a1`](../models/flow.md#waft) | 8587 | 6080 | unsupported | 2583 |
| [`waft-a1-kitti`](../models/flow.md#waft) | 8585 | 6078 | unsupported | 2564 |
| [`waft-a1-sintel`](../models/flow.md#waft) | 8590 | 6076 | unsupported | 2564 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matching

### Matching, synthetic, 768x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`aliked-flexglue`](../models/matching.md#flexglue) | 1224 | 1100 | – | – | – | 1387 | – |
| [`aliked-lightglue`](../models/matching.md#lightglue) | 448 | 455 | unsupported | – | – | 2407 | – |
| [`dedode-v2-b`](../models/matching.md#dedode) | 2389 | 1814 | unsupported | – | – | 3622 | – |
| [`dedode-v2-g`](../models/matching.md#dedode) | 4240 | 3222 | – | – | – | 4226 | – |
| [`disk-lightglue`](../models/matching.md#lightglue) | 886 | 786 | unsupported | – | – | 2058 | – |
| [`loftr-indoor`](../models/matching.md#loftr) | 1806 | 1549 | unsupported | – | – | 3675 | – |
| [`loftr-outdoor`](../models/matching.md#loftr) | 1788 | 1526 | unsupported | – | – | 3703 | – |
| [`loma-b`](../models/matching.md#loma) | 3617 | 2760 | – | – | – | 4561 | – |
| [`loma-b128`](../models/matching.md#loma) | 1546 | 1085 | – | – | – | 1902 | – |
| [`loma-b64`](../models/matching.md#loma) | 1228 | 828 | – | – | – | 1630 | – |
| [`loma-g`](../models/matching.md#loma) | 4152 | 3230 | unsupported | – | – | 5159 | – |
| [`loma-l`](../models/matching.md#loma) | 3706 | 2810 | – | – | – | 4929 | – |
| [`loma-r`](../models/matching.md#loma) | 3622 | 2708 | unsupported | – | – | 4549 | – |
| [`romav1`](../models/matching.md#romav1) | 3733 | 3135 | – | – | – | 2460 | – |
| [`romav2`](../models/matching.md#romav2) | 6727 | 6078 | 3669 | 3633 | 1.7x | 2849 | 3120 |
| [`xfeat`](../models/matching.md#xfeat) | 164 | – | – | – | – | 538 | – |
| [`xfeat-lighterglue`](../models/matching.md#xfeat) | 524 | – | – | – | – | 1002 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Tracking

### Tracking, synthetic, 480x854

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`alltracker`](../models/tracking.md#alltracker) | 3151 | 2288 | – | 2232 |
| [`alltracker-tiny`](../models/tracking.md#alltracker) | 2909 | 2174 | – | 2245 |
| [`bootstapir`](../models/tracking.md#tapir) | 1881 | – | – | 2497 |
| [`cotracker3-offline`](../models/tracking.md#cotracker) | 1595 | 1455 | – | 3219 |
| [`cotracker3-online`](../models/tracking.md#cotracker) | 1638 | 1491 | – | 2453 |
| [`locotrack-b`](../models/tracking.md#locotrack) | 1453 | 1113 | – | 1597 |
| [`locotrack-s`](../models/tracking.md#locotrack) | 1233 | 961 | – | 1575 |
| [`tapir`](../models/tracking.md#tapir) | 1592 | – | – | 1717 |
| [`tapnext`](../models/tracking.md#tapnext) | 4033 | 3724 | – | 2420 |
| [`tapnext-pp`](../models/tracking.md#tapnext) | 4035 | 3746 | – | 1677 |
| [`track-on-r`](../models/tracking.md#track_on) | 11174 | 8822 | unsupported | 3089 |
| [`track-on2`](../models/tracking.md#track_on) | 10543 | 8827 | unsupported | 3090 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Interpolation

### Interpolation, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ema-vfi`](../models/interpolation.md#ema_vfi) | 3362 | 2255 | – | – | – | 2770 | – |
| [`ema-vfi-small`](../models/interpolation.md#ema_vfi) | 980 | 666 | unsupported | – | – | 1587 | – |
| [`ema-vfi-small-t`](../models/interpolation.md#ema_vfi) | 1019 | 876 | – | – | – | 2460 | – |
| [`ema-vfi-t`](../models/interpolation.md#ema_vfi) | 3305 | 2222 | – | – | – | 2700 | – |
| [`film`](../models/interpolation.md#film) | 9227 | 6590 | – | – | – | 9495 | – |
| [`gimm-vfi-r`](../models/interpolation.md#gimm_vfi) | 58742 | 30451 | – | – | – | 12185 | – |
| [`gimm-vfi-r-p`](../models/interpolation.md#gimm_vfi) | 39847 | 16157 | unsupported | – | – | 10310 | – |
| [`rife-4.25`](../models/interpolation.md#rife) | 256 | 211 | 223 | 180 | 1.2x | 1680 | 1655 |
| [`rife-4.25-lite`](../models/interpolation.md#rife) | 200 | 202 | – | – | – | 1590 | – |
| [`rife-4.26`](../models/interpolation.md#rife) | 235 | 160 | 226 | 137 | 1.2x | 1265 | 1379 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Restoration

### Restoration, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | portable bf16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`big-lama`](../models/restoration.md#lama) | 347 | 225 | – | unsupported | – | – | 990 | – |
| [`codeformer`](../models/restoration.md#codeformer) | 484 | 285 | – | 481 | 239 | 1.2x | 1597 | 1637 |
| [`dat-2-x4`](../models/restoration.md#dat) | 12210 | 15757 | – | – | – | – | 5878 | – |
| [`dat-light-x4`](../models/restoration.md#dat) | 2678 | 1870 | – | – | – | – | 2425 | – |
| [`dat-x4`](../models/restoration.md#dat) | 24476 | 9358 | – | – | – | – | 3305 | – |
| [`drct-l-x4`](../models/restoration.md#drct) | 29775 | 18644 | – | – | – | – | 5279 | – |
| [`drct-x4`](../models/restoration.md#drct) | 15300 | 10754 | – | – | – | – | 5003 | – |
| [`gfpgan-v1.4`](../models/restoration.md#gfpgan) | 258 | 159 | – | unsupported | – | – | 1108 | – |
| [`hat-s-x4`](../models/restoration.md#hat) | 15572 | 10010 | – | – | – | – | 3396 | – |
| [`hat-x4`](../models/restoration.md#hat) | 17231 | 11318 | – | – | – | – | 3674 | – |
| [`nafnet-gopro`](../models/restoration.md#nafnet) | 524 | 359 | – | 522 | 358 | 1.0x | 1187 | 1124 |
| [`nafnet-sidd`](../models/restoration.md#nafnet) | 457 | 277 | – | – | – | – | 980 | – |
| [`real-hat-gan-x4`](../models/restoration.md#hat) | 24682 | 15831 | – | – | – | – | 5687 | – |
| [`realesr-general-x4v3`](../models/restoration.md#compact) | 333 | 246 | – | 177 | 104 | 2.4x | 737 | 1061 |
| [`realesrgan-x2plus`](../models/restoration.md#realesrgan) | 1599 | 1199 | – | 1667 | 567 | 2.1x | 974 | 1222 |
| [`realesrgan-x4plus`](../models/restoration.md#realesrgan) | 7548 | 5321 | – | 7380 | 2184 | 2.4x | 1390 | 2452 |
| [`realesrgan-x4plus-anime-6b`](../models/restoration.md#realesrgan) | 2700 | 1617 | – | 2038 | 849 | 1.9x | 1319 | 2106 |
| [`realplksr-nomoswebphoto-x4`](../models/restoration.md#realplksr) | 2800 | 1991 | – | 3408 | 1910 (golden fail) | 0.6x | 953 | 1736 |
| [`restormer-deblur`](../models/restoration.md#restormer) | 2577 | 1467 | – | – | – | – | 1317 | – |
| [`restormer-derain`](../models/restoration.md#restormer) | 3040 | 2815 | – | – | – | – | 2668 | – |
| [`restormer-sidd`](../models/restoration.md#restormer) | 2793 | 2427 | – | – | – | – | 2672 | – |
| [`seedvr2-3b`](../models/restoration.md#seedvr2) | – | 14023 | 14307 | – | – | – | 5178 | – |
| [`seedvr2-7b`](../models/restoration.md#seedvr2) | – | 32426 | 25365 | – | – | – | 6547 | – |
| [`span-x2`](../models/restoration.md#span) | 180 | 134 | – | 171 | 76.5 | 1.7x | 753 | 1073 |
| [`span-x4`](../models/restoration.md#span) | 190 | 149 | – | 179 | 86.0 | 1.7x | 827 | 1210 |
| [`swin2sr-x2`](../models/restoration.md#swin2sr) | 8277 | 5144 | – | – | – | – | 1980 | – |
| [`swin2sr-x4`](../models/restoration.md#swin2sr) | 8163 | 4902 | – | – | – | – | 3313 | – |
| [`swin2sr-x4-real`](../models/restoration.md#swin2sr) | 8813 | 5522 | – | – | – | – | 2268 | – |
| [`swinir-l-x4-real`](../models/restoration.md#swinir) | 14252 | 8691 | – | – | – | – | 2437 | – |
| [`swinir-m-x2`](../models/restoration.md#swinir) | 6857 | 4312 | – | – | – | – | 2423 | – |
| [`swinir-m-x4`](../models/restoration.md#swinir) | 6887 | 4339 | – | – | – | – | 3281 | – |
| [`swinir-m-x4-real`](../models/restoration.md#swinir) | 7297 | 4668 | – | – | – | – | 2243 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Volume

### Volume, synthetic, 192x192

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`nnfoundation-vit`](../models/volume.md#nnfoundation) | 18038 | 13037 | 2469 |

### Volume, synthetic, 200x200

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`nnfoundation-cnn`](../models/volume.md#nnfoundation) | 4271 | 3520 | 8041 |
| [`nninteractive`](../models/volume.md#nninteractive) | 4005 | 3797 | 5587 |
| [`totalsegmentator-total`](../models/volume.md#totalsegmentator) | 42521 | 33823 | 3402 |
| [`totalsegmentator-total-3mm`](../models/volume.md#totalsegmentator) | 828 | 740 | 3246 |
<!-- device:end -->
