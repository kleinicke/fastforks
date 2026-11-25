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
| measured | 2026-11-19 to 2026-11-25 |
| rows | 1707 in `docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 925 | 687 | – | – | – | 2259 | – |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 692 | 492 | – | – | – | 1362 | – |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 4192 | 3179 | unsupported | – | – | 2555 | – |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 4973 | 3977 | – | – | – | 2650 | – |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 4565 | 2737 | – | – | – | 2858 | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | 29962 | 18737 | – | – | – | 4010 | – |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 6066 | 2934 | – | – | – | 3733 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 4595 | 3841 | unsupported | – | – | 4754 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 4485 | 3178 | unsupported | – | – | 3669 | – |
| [`coex-sceneflow`](../models/stereo.md#coex) | 282 | 215 | – | – | – | 947 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 6118 | 4297 | 5212 | 3558 | 1.2x | 1557 | 1863 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 13339 | 10693 | – | – | – | 5189 | – |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 2234 | 1711 | unsupported | – | – | 2841 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 2276 | 1662 | unsupported | – | – | 2844 | – |
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
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 11268 | 9128 | 10129 | 5478 | 1.7x | 2749 | 2863 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 11383 | 8568 | 10166 | 5297 | 1.6x | 2754 | 2873 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 11467 | 8456 | 10077 | 5268 | 1.6x | 2749 | 2894 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 11338 | 7451 | 10176 | 5290 | 1.4x | 2749 | 2872 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 11310 | 10068 | 10647 | 6039 | 1.7x | 2993 | 2957 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 4658 | 3628 | unsupported | – | – | 3428 | – |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 4594 | 3654 | unsupported | – | – | 3414 | – |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 4609 | 3674 | unsupported | – | – | 3419 | – |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 4622 | 3649 | unsupported | – | – | 3439 | – |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 4607 | 3598 | unsupported | – | – | 3421 | – |
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
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 4966 | 3239 | 4194 | 2361 | 1.4x | 3296 | 3875 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 4944 | 3249 | 4193 | 2370 | 1.4x | 3286 | 3898 |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 4958 | 3250 | 4162 | 2361 | 1.4x | 3284 | 3874 |
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

### Depth, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | portable bf16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`any2full-b`](../models/depth.md#any2full) | 379 | 379 | – | – | – | – | 1583 | – |
| [`any2full-l`](../models/depth.md#any2full) | 864 | 909 | – | – | – | – | 3827 | – |
| [`any2full-s`](../models/depth.md#any2full) | 138 | 113 | – | unsupported | – | – | 819 | – |
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 751 | 606 | – | unsupported | – | – | 3231 | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 273 | 226 | – | unsupported | – | – | 1594 | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 275 | 227 | – | unsupported | – | – | 1561 | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 275 | 227 | – | unsupported | – | – | 1563 | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 1037 | 840 | – | unsupported | – | – | 3895 | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 228 | 184 | – | 214 | 169 | 1.1x | 1026 | 1026 |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 671 | 518 | – | 630 | 498 | 1.0x | 1893 | 1759 |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | 237 | 184 | – | 216 | 170 | 1.1x | 1017 | 1027 |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | 700 | 527 | – | 641 | 581 | 0.9x | 1831 | 1737 |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | 96.4 | 79.8 | – | 106 | 68.5 | 1.2x | 737 | 710 |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | 291 | 188 | – | 232 | 198 | 0.9x | 1021 | 996 |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | 661 | 527 | – | 720 | 587 | 0.9x | 1924 | 1684 |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | 117 | 78.5 | – | 87.1 | 67.4 | 1.2x | 735 | 707 |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 92.7 | 73.2 | – | 87.0 | 66.2 | 1.1x | 729 | 700 |
| [`depth-pro`](../models/depth.md#depth_pro) | 5310 | 4271 | – | unsupported | – | – | 4905 | – |
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
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 907 | 785 | – | unsupported | – | – | 2025 | – |
| [`moge-2-vitb-normal`](../models/depth.md#moge) | 579 | 443 | – | unsupported | – | – | 1389 | – |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 1339 | 1036 | – | unsupported | – | – | 1950 | – |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 372 | 282 | – | unsupported | – | – | 1250 | – |
| [`moge-3-vitl`](../models/depth.md#moge) | 2879 | 2543 | – | unsupported | – | – | 2566 | – |
| [`omni-dc`](../models/depth.md#omni_dc) | 3609 | 3352 | – | unsupported | – | – | 5336 | – |
| [`pointdit-b-512`](../models/depth.md#pointdit) | 343 | 356 | – | – | – | – | 1695 | – |
| [`pointdit-l-512`](../models/depth.md#pointdit) | 1047 | 1231 | – | – | – | – | 4003 | – |
| [`prompt-depth-anything-l`](../models/depth.md#prompt_depth_anything) | 1340 | 1008 | – | unsupported | – | – | 2710 | – |
| [`prompt-depth-anything-s`](../models/depth.md#prompt_depth_anything) | 189 | 148 | – | unsupported | – | – | 971 | – |
| [`stablenormal-turbo`](../models/depth.md#stablenormal) | 3589 | 2299 | – | unsupported | – | – | 8683 | – |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 1254 | 1173 | – | – | – | – | 1825 | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1266 | 935 | – | unsupported | – | – | 1948 | – |
| [`video-depth-anything-metric-s`](../models/depth.md#video_depth_anything) | 1474 | 893 | – | unsupported | – | – | 2978 | – |
| [`video-depth-anything-s`](../models/depth.md#video_depth_anything) | 1472 | 900 | – | unsupported | – | – | 2980 | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 1009 | 837 | – | unsupported | – | – | 1919 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Classification

### Classification, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 3.88 | 3.09 | unsupported | – | – | 772 | – |
| [`coatnet_0_rw_224.sw_in1k`](../models/classification.md#coatnet) | 9.59 | 8.47 | 9.54 | 8.50 | 1.0x | 515 | 513 |
| [`coatnet_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 16.0 | 14.1 | 15.6 | 13.7 | 1.0x | 617 | 613 |
| [`coatnet_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 23.0 | 18.6 | 24.5 | 18.8 | 1.0x | 758 | 753 |
| [`coatnet_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 7.78 | 7.06 | 7.33 | 7.11 | 1.0x | 449 | 455 |
| [`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 15.8 | 14.1 | 16.2 | 14.1 | 1.0x | 617 | 613 |
| [`coatnet_rmlp_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 16.1 | 13.5 | 16.7 | 13.5 | 1.0x | 612 | 612 |
| [`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 24.2 | 18.4 | 23.4 | 19.5 | 0.9x | 777 | 776 |
| [`coatnet_rmlp_2_rw_224.sw_in1k`](../models/classification.md#coatnet) | 23.1 | 18.9 | 23.4 | 19.1 | 1.0x | 783 | 763 |
| [`coatnet_rmlp_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 8.05 | 7.95 | 8.38 | 7.07 | 1.1x | 458 | 457 |
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 24.8 | 17.4 | 21.1 | 16.2 | 1.1x | 648 | 646 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 20.2 | 16.4 | 20.1 | 17.2 | 1.0x | 647 | 651 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 39.7 | 34.0 | 45.0 | 37.5 | 0.9x | 746 | 750 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 35.2 | 28.7 | 33.7 | 28.6 | 1.0x | 969 | 966 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 33.7 | 28.8 | 34.1 | 28.7 | 1.0x | 963 | 965 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 72.9 | 63.0 | 73.2 | 62.8 | 1.0x | 1037 | 1037 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 14.5 | 12.3 | 14.5 | 12.3 | 1.0x | 523 | 522 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 14.4 | 12.4 | 14.1 | 12.7 | 1.0x | 521 | 521 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 28.3 | 25.4 | 30.8 | 25.0 | 1.0x | 628 | 625 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 8.98 | 8.00 | 9.08 | 7.97 | 1.0x | 447 | 450 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 9.82 | 8.22 | 9.44 | 8.11 | 1.0x | 449 | 446 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 16.9 | 15.8 | 18.6 | 15.9 | 1.0x | 521 | 518 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 25.8 | 21.3 | 23.2 | 20.6 | 1.0x | 773 | 772 |
| [`davit_base.msft_in1k`](../models/classification.md#davit) | 23.5 | 19.4 | 24.0 | 20.1 | 1.0x | 662 | 663 |
| [`davit_small.msft_in1k`](../models/classification.md#davit) | 18.8 | 16.1 | 18.7 | 15.1 | 1.1x | 570 | 572 |
| [`davit_tiny.msft_in1k`](../models/classification.md#davit) | 11.8 | 9.05 | 12.8 | 8.87 | 1.0x | 498 | 498 |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 17.7 | 14.5 | 18.3 | 15.0 | 1.0x | 592 | 587 |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 11.3 | 9.71 | unsupported | – | – | 416 | – |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 65.0 | 56.1 | unsupported | – | – | 1108 | – |
| [`efficientformer_l1.snap_dist_in1k`](../models/classification.md#efficientformer) | 6.90 | 6.87 | 4.92 | 5.89 | 1.4x | 444 | 477 |
| [`efficientformer_l3.snap_dist_in1k`](../models/classification.md#efficientformer) | 9.25 | 8.64 | 9.85 | 8.33 | 1.0x | 543 | 541 |
| [`efficientformer_l7.snap_dist_in1k`](../models/classification.md#efficientformer) | 18.6 | 15.1 | 18.5 | 14.9 | 1.0x | 750 | 748 |
| [`efficientformerv2_l.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 10.9 | 11.6 | 10.8 | 11.5 | 1.0x | 555 | 561 |
| [`efficientformerv2_s0.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 4.13 | 4.07 | 4.32 | 4.15 | 1.0x | 371 | 371 |
| [`efficientformerv2_s1.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 5.97 | 6.18 | 6.08 | 7.28 | 1.0x | 393 | 391 |
| [`efficientformerv2_s2.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 7.34 | 7.56 | 7.20 | 7.73 | 1.0x | 453 | 451 |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 4.17 | 3.97 | 4.41 | 4.12 | 1.0x | 345 | 345 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 15.9 | 15.1 | 16.0 | 14.2 | 1.1x | 606 | 598 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 16.3 | 14.9 | 16.5 | 14.5 | 1.0x | 500 | 505 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 16.8 | 9.51 | 15.9 | 10.7 | 0.9x | 555 | 553 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 251 | 207 | 253 | 205 | 1.0x | 1163 | 1133 |
| [`falconsai-nsfw-vit`](../models/classification.md#nsfw_vit) | 16.5 | 14.0 | 17.2 | 14.2 | 1.0x | 664 | 665 |
| [`focalnet_base_lrf.ms_in1k`](../models/classification.md#focalnet) | 23.8 | 21.1 | 24.4 | 21.5 | 1.0x | 1222 | 1223 |
| [`focalnet_base_srf.ms_in1k`](../models/classification.md#focalnet) | 23.3 | 19.7 | 22.4 | 18.9 | 1.0x | 1217 | 1220 |
| [`focalnet_small_lrf.ms_in1k`](../models/classification.md#focalnet) | 18.2 | 16.3 | 19.3 | 16.8 | 1.0x | 859 | 859 |
| [`focalnet_small_srf.ms_in1k`](../models/classification.md#focalnet) | 17.2 | 15.9 | 16.3 | 15.4 | 1.0x | 854 | 858 |
| [`focalnet_tiny_lrf.ms_in1k`](../models/classification.md#focalnet) | 11.1 | 10.5 | 10.6 | 9.56 | 1.1x | 691 | 696 |
| [`focalnet_tiny_srf.ms_in1k`](../models/classification.md#focalnet) | 9.27 | 8.19 | 9.21 | 9.55 | 0.9x | 686 | 761 |
| [`hgnetv2_b0.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 2.59 | 2.47 | 2.98 | 2.53 | 1.0x | 416 | 416 |
| [`hgnetv2_b1.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 3.03 | 3.12 | 3.67 | 2.95 | 1.0x | 426 | 424 |
| [`hgnetv2_b2.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 4.17 | 4.94 | 4.39 | 3.95 | 1.1x | 502 | 508 |
| [`hgnetv2_b3.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 5.03 | 5.16 | 5.66 | 4.45 | 1.1x | 591 | 589 |
| [`hgnetv2_b4.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 6.36 | 5.77 | 6.34 | 5.78 | 1.0x | 644 | 645 |
| [`hgnetv2_b5.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 12.4 | 10.5 | 11.8 | 10.2 | 1.0x | 926 | 923 |
| [`hgnetv2_b6.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 23.0 | 20.9 | 27.0 | 21.3 | 1.0x | 1289 | 1289 |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 13.3 | 10.7 | 14.0 | 11.6 | 0.9x | 548 | 591 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 2.92 | 2.82 | 3.01 | 2.50 | 1.1x | 324 | 322 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 3.09 | 2.98 | 3.23 | 3.14 | 0.9x | 333 | 332 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 2.46 | 2.39 | unsupported | – | – | 357 | – |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 4.35 | 3.91 | 7.44 | 3.96 | 1.0x | 433 | 437 |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 19.1 | 18.1 | 19.8 | 18.2 | 1.0x | 766 | 763 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 4.41 | 4.03 | unsupported | – | – | 374 | – |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 4.56 | 4.65 | unsupported | – | – | 390 | – |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 4.87 | 5.90 | unsupported | – | – | 412 | – |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 4.90 | 4.79 | unsupported | – | – | 390 | – |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 4.90 | 4.32 | unsupported | – | – | 397 | – |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 4.57 | 4.38 | unsupported | – | – | 399 | – |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 7.34 | 8.33 | unsupported | – | – | 499 | – |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 7.30 | 6.66 | unsupported | – | – | 455 | – |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 11.7 | 10.1 | unsupported | – | – | 545 | – |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 11.7 | 10.3 | unsupported | – | – | 545 | – |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 19.4 | 16.3 | 19.2 | 16.4 | 1.0x | 550 | 551 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 4.78 | 4.08 | 4.62 | 4.24 | 1.0x | 356 | 353 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 8.12 | 7.10 | 8.24 | 6.84 | 1.0x | 442 | 444 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 3.39 | 3.29 | unsupported | – | – | 319 | – |
| [`shvit_s1.in1k`](../models/classification.md#shvit) | 6.86 | 3.74 | 4.03 | 3.97 | 0.9x | 389 | 391 |
| [`shvit_s2.in1k`](../models/classification.md#shvit) | 4.74 | 4.01 | 4.25 | 3.87 | 1.0x | 412 | 413 |
| [`shvit_s3.in1k`](../models/classification.md#shvit) | 4.96 | 4.77 | 5.17 | 4.52 | 1.1x | 431 | 432 |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 2.19 | 2.50 | unsupported | – | – | 314 | – |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 3.15 | 2.84 | unsupported | – | – | 341 | – |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 5.75 | 10.3 | unsupported | – | – | 356 | – |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 4.45 | 6.36 | unsupported | – | – | 374 | – |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 5.55 | 5.07 | unsupported | – | – | 361 | – |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 24.1 | 19.5 | 23.9 | 19.6 | 1.0x | 690 | 686 |
| [`tiny_vit_11m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 8.84 | 7.82 | 7.82 | 7.06 | 1.1x | 435 | 431 |
| [`tiny_vit_21m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 10.2 | 8.48 | 9.88 | 8.38 | 1.0x | 477 | 475 |
| [`tiny_vit_5m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 6.59 | 5.89 | 6.08 | 4.58 | 1.3x | 406 | 411 |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 17.4 | 13.9 | unsupported | – | – | 2003 | – |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 17.5 | 14.8 | 18.1 | 14.4 | 1.0x | 588 | 588 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 215 | 184 | 215 | 181 | 1.0x | 1298 | 1279 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Classification, synthetic, 256x256

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 9.08 | 7.48 | 8.65 | 7.53 | 1.0x | 556 | 556 |
| [`fastvit_t12.apple_dist_in1k`](../models/classification.md#fastvit) | 4.43 | 3.85 | 4.32 | 4.26 | 0.9x | 434 | 434 |
| [`fastvit_t8.apple_dist_in1k`](../models/classification.md#fastvit) | 3.78 | 3.49 | 4.16 | 3.34 | 1.0x | 401 | 403 |
| [`shvit_s4.in1k`](../models/classification.md#shvit) | 6.28 | 5.77 | 6.09 | 5.45 | 1.1x | 448 | 451 |

### Classification, synthetic, 384x384

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 65.5 | 47.2 | 63.3 | 47.2 | 1.0x | 1141 | 1132 |
| [`tiny_vit_21m_384.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 19.9 | 14.9 | 20.4 | 14.7 | 1.0x | 640 | 641 |

### Classification, synthetic, 448x448

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`wd-convnext-tagger-v3`](../models/classification.md#wd_tagger) | 56.1 | 45.8 | 55.6 | 45.6 | 1.0x | 817 | 821 |
| [`wd-eva02-large-tagger-v3`](../models/classification.md#wd_tagger) | 249 | 201 | 256 | 229 | 0.9x | 1275 | 1276 |
| [`wd-vit-large-tagger-v3`](../models/classification.md#wd_tagger) | 208 | 181 | 207 | 181 | 1.0x | 1100 | 1124 |
| [`wd-vit-tagger-v3`](../models/classification.md#wd_tagger) | 56.6 | 46.0 | 55.7 | 45.2 | 1.0x | 639 | 636 |

### Classification, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`tiny_vit_21m_512.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 38.8 | 27.0 | 38.5 | 26.4 | 1.0x | 838 | 840 |

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
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 622 | 445 | 601 | 444 | 1.0x | 1550 | 1542 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 479 | 361 | 478 | 359 | 1.0x | 1533 | 1552 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 313 | 256 | unsupported | – | – | 1356 | – |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 941 | 760 | unsupported | – | – | 2178 | – |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 135 | 115 | unsupported | – | – | 1201 | – |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 667 | 574 | – | – | – | 3661 | – |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 765 | 599 | – | – | – | 3724 | – |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 510 | 382 | 528 | 366 | 1.0x | 1332 | 1391 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 267 | 241 | 262 | 234 | 1.0x | 1258 | 1500 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 692 | 607 | unsupported | – | – | 2149 | – |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 924 | 824 | unsupported | – | – | 2587 | – |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 547 | 430 | 498 | 376 | 1.1x | 1542 | 1553 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 1485 | 1214 | unsupported | – | – | 2967 | – |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 3594 | 2679 | unsupported | – | – | 4541 | – |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 68.3 | 57.6 | 67.6 | 57.1 | 1.0x | 734 | 725 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 351 | 270 | 351 | 270 | 1.0x | 1283 | 1282 |
| [`segnext-mscan-b-ade20k`](../models/segmentation.md#segnext) | 126 | 119 | 126 | 119 | 1.0x | 947 | 940 |
| [`segnext-mscan-l-ade20k`](../models/segmentation.md#segnext) | 197 | 181 | 196 | 186 | 1.0x | 1140 | 1145 |
| [`segnext-mscan-s-ade20k`](../models/segmentation.md#segnext) | 87.1 | 81.4 | 87.1 | 81.6 | 1.0x | 828 | 830 |
| [`segnext-mscan-t-ade20k`](../models/segmentation.md#segnext) | 73.5 | 69.7 | 73.6 | 69.9 | 1.0x | 745 | 746 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matting

### Matting, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ben2`](../models/matting.md#ben2) | 1524 | 850 | unsupported | – | – | 2863 | – |
| [`birefnet`](../models/matting.md#birefnet) | 1625 | 1048 | unsupported | – | – | 2364 | – |
| [`birefnet-lite`](../models/matting.md#birefnet) | 618 | 382 | unsupported | – | – | 1810 | – |
| [`inspyrenet`](../models/matting.md#inspyrenet) | 1121 | 679 | unsupported | – | – | 1741 | – |
| [`inspyrenet-fast`](../models/matting.md#inspyrenet) | 129 | 95.6 | – | – | – | 1060 | – |
| [`isnet`](../models/matting.md#isnet) | 197 | 145 | 197 | 121 | 1.2x | 1200 | 1254 |
| [`isnet-general-use`](../models/matting.md#isnet) | 211 | 145 | 209 | 122 | 1.2x | 1217 | 1328 |
| [`matanyone`](../models/matting.md#matanyone) | 4891 | 3513 | unsupported | – | – | 3562 | – |
| [`rmbg-2.0`](../models/matting.md#birefnet) | 1643 | 1050 | – | – | – | 2508 | – |
| [`u2net`](../models/matting.md#u2net) | 78.3 | 62.3 | unsupported | – | – | 992 | – |
| [`u2net-human-seg`](../models/matting.md#u2net) | 78.6 | 62.7 | unsupported | – | – | 972 | – |
| [`u2netp`](../models/matting.md#u2net) | 35.0 | 28.3 | – | – | – | 574 | – |
| [`vitmatte-b`](../models/matting.md#vitmatte) | 2370 | 1652 | – | – | – | 1741 | – |
| [`vitmatte-s`](../models/matting.md#vitmatte) | 1181 | 816 | unsupported | – | – | 1622 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Edges

### Edges, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`anyline`](../models/edges.md#anyline) | 55.4 | 43.8 | 55.4 | 39.8 | 1.1x | 729 | 731 |
| [`hed`](../models/edges.md#hed) | 74.9 | 65.2 | unsupported | – | – | 1174 | – |
| [`lineart`](../models/edges.md#lineart) | 78.6 | 51.6 | 77.5 | 51.5 | 1.0x | 882 | 883 |
| [`lineart-anime`](../models/edges.md#lineart_anime) | 51.8 | 44.5 | unsupported | – | – | 982 | – |
| [`lineart-coarse`](../models/edges.md#lineart) | 80.0 | 51.8 | 81.0 | 52.4 | 1.0x | 867 | 842 |
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
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | 500 | 398 | – | 1374 |
| [`sam-hq-vit-h`](../models/promptable_segmentation.md#sam_hq) | 2318 | 1919 | unsupported | 2466 |
| [`sam-hq-vit-l`](../models/promptable_segmentation.md#sam_hq) | 1091 | 899 | – | 1884 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 409 | 352 | – | 1069 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 1968 | 1589 | unsupported | 2350 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1051 | 864 | – | 1639 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 298 | 263 | unsupported | 1112 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 643 | 584 | unsupported | 1512 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 182 | 159 | unsupported | 998 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 156 | 137 | unsupported | 1023 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 1961 | 1573 | unsupported | 2775 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`deim-l-coco`](../models/detection.md#dfine) | 73.6 | 59.7 | – | – | – | 872 | – |
| [`deim-m-coco`](../models/detection.md#dfine) | 58.4 | 46.9 | – | – | – | 809 | – |
| [`deim-n-coco`](../models/detection.md#dfine) | 25.7 | 21.7 | unsupported | – | – | 575 | – |
| [`deim-s-coco`](../models/detection.md#dfine) | 35.8 | 30.8 | – | – | – | 674 | – |
| [`deim-x-coco`](../models/detection.md#dfine) | 120 | 94.8 | – | – | – | 1117 | – |
| [`detr-r101-coco`](../models/detection.md#detr) | 181 | 144 | – | – | – | 1397 | – |
| [`detr-r50-coco`](../models/detection.md#detr) | 108 | 90.6 | unsupported | – | – | 1397 | – |
| [`dfine-l-coco`](../models/detection.md#dfine) | 74.4 | 59.9 | – | – | – | 881 | – |
| [`dfine-m-coco`](../models/detection.md#dfine) | 58.4 | 46.6 | – | – | – | 821 | – |
| [`dfine-n-coco`](../models/detection.md#dfine) | 23.9 | 21.6 | unsupported | – | – | 578 | – |
| [`dfine-s-coco`](../models/detection.md#dfine) | 34.6 | 29.5 | unsupported | – | – | 652 | – |
| [`dfine-x-coco`](../models/detection.md#dfine) | 120 | 94.6 | – | – | – | 1133 | – |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 156 | 126 | unsupported | – | – | 1156 | – |
| [`grounding-dino-base`](../models/detection.md#grounding_dino) | 1189 | 889 | unsupported | – | – | 1687 | – |
| [`grounding-dino-tiny`](../models/detection.md#grounding_dino) | 933 | 753 | unsupported | – | – | 1499 | – |
| [`lw-detr-large-coco`](../models/detection.md#lwdetr) | 70.8 | 64.6 | – | – | – | 891 | – |
| [`lw-detr-medium-coco`](../models/detection.md#lwdetr) | 47.4 | 48.0 | – | – | – | 773 | – |
| [`lw-detr-small-coco`](../models/detection.md#lwdetr) | 33.3 | 33.6 | – | – | – | 736 | – |
| [`lw-detr-tiny-coco`](../models/detection.md#lwdetr) | 37.7 | 34.9 | unsupported | – | – | 935 | – |
| [`lw-detr-xlarge-coco`](../models/detection.md#lwdetr) | 123 | 117 | – | – | – | 1077 | – |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 181 | 144 | unsupported | – | – | 1174 | – |
| [`mm-grounding-dino-base`](../models/detection.md#grounding_dino) | 1254 | 916 | – | – | – | 1719 | – |
| [`mm-grounding-dino-tiny`](../models/detection.md#grounding_dino) | 935 | 766 | – | – | – | 1507 | – |
| [`owlv2-b16`](../models/detection.md#owlv2) | 383 | 290 | unsupported | – | – | 1046 | – |
| [`owlv2-l14`](../models/detection.md#owlv2) | 1890 | 1426 | unsupported | – | – | 1793 | – |
| [`rf-detr-l-coco`](../models/detection.md#rfdetr) | 56.4 | 50.7 | unsupported | – | – | 726 | – |
| [`rf-detr-m-coco`](../models/detection.md#rfdetr) | 40.2 | 34.8 | unsupported | – | – | 697 | – |
| [`rf-detr-n-coco`](../models/detection.md#rfdetr) | 20.9 | 19.1 | unsupported | – | – | 594 | – |
| [`rf-detr-s-coco`](../models/detection.md#rfdetr) | 33.1 | 28.6 | unsupported | – | – | 656 | – |
| [`rf-detr-seg-2xl-coco`](../models/detection.md#rfdetr) | 180 | 142 | unsupported | – | – | 994 | – |
| [`rf-detr-seg-l-coco`](../models/detection.md#rfdetr) | 67.0 | 53.0 | unsupported | – | – | 738 | – |
| [`rf-detr-seg-m-coco`](../models/detection.md#rfdetr) | 50.5 | 41.3 | unsupported | – | – | 733 | – |
| [`rf-detr-seg-n-coco`](../models/detection.md#rfdetr) | 29.6 | 24.5 | unsupported | – | – | 628 | – |
| [`rf-detr-seg-s-coco`](../models/detection.md#rfdetr) | 37.7 | 31.1 | unsupported | – | – | 668 | – |
| [`rf-detr-seg-xl-coco`](../models/detection.md#rfdetr) | 114 | 90.2 | unsupported | – | – | 840 | – |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 88.9 | 67.8 | – | – | – | 815 | – |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 64.8 | 50.0 | – | – | – | 811 | – |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 71.8 | 57.4 | – | – | – | 791 | – |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 46.1 | 37.1 | unsupported | – | – | 768 | – |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 139 | 118 | – | – | – | 953 | – |
| [`rtmdet-l-coco`](../models/detection.md#rtmdet) | 77.2 | 63.9 | unsupported | – | – | 1096 | – |
| [`rtmdet-m-coco`](../models/detection.md#rtmdet) | 51.2 | 43.7 | unsupported | – | – | 958 | – |
| [`rtmdet-m-person`](../models/detection.md#rtmdet) | 51.1 | 41.5 | unsupported | – | – | 830 | – |
| [`rtmdet-nano-person`](../models/detection.md#rtmdet) | 7.44 | 6.69 | unsupported | – | – | 367 | – |
| [`rtmdet-s-coco`](../models/detection.md#rtmdet) | 24.3 | 22.3 | unsupported | – | – | 635 | – |
| [`rtmdet-tiny-coco`](../models/detection.md#rtmdet) | 20.7 | 19.7 | unsupported | – | – | 518 | – |
| [`rtmdet-x-coco`](../models/detection.md#rtmdet) | 138 | 113 | unsupported | – | – | 1478 | – |
| [`yolox-l`](../models/detection.md#yolox) | 69.1 | 58.4 | 60.4 | 53.9 | 1.1x | 1021 | 1104 |
| [`yolox-m`](../models/detection.md#yolox) | 41.2 | 36.2 | 35.7 | 30.3 | 1.2x | 704 | 799 |
| [`yolox-s`](../models/detection.md#yolox) | 21.0 | 17.6 | 19.7 | 16.1 | 1.1x | 539 | 543 |
| [`yolox-tiny`](../models/detection.md#yolox) | 10.4 | 9.83 | 9.93 | 9.10 | 1.1x | 417 | 427 |
| [`yolox-x`](../models/detection.md#yolox) | 119 | 103 | 101 | 93.9 | 1.1x | 1350 | 1522 |

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
| [`dwpose-l`](../models/pose.md#rtmpose) | 83.0 | 72.2 | – | 1144 |
| [`dwpose-l-384`](../models/pose.md#rtmpose) | 156 | 137 | – | 1232 |
| [`dwpose-m`](../models/pose.md#rtmpose) | 57.8 | 53.4 | – | 915 |
| [`dwpose-s`](../models/pose.md#rtmpose) | 30.6 | 29.6 | – | 647 |
| [`dwpose-t`](../models/pose.md#rtmpose) | 28.7 | 27.7 | – | 591 |
| [`hrnet-w32`](../models/pose.md#hrnet) | 145 | 109 | unsupported | 860 |
| [`hrnet-w32-384`](../models/pose.md#hrnet) | 327 | 237 | – | 1104 |
| [`hrnet-w48`](../models/pose.md#hrnet) | 285 | 222 | – | 897 |
| [`hrnet-w48-384`](../models/pose.md#hrnet) | 597 | 449 | – | 1058 |
| [`retinaface-mobilenet0.25`](../models/pose.md#retinaface) | 23.8 | 19.8 | – | 621 |
| [`retinaface-resnet50`](../models/pose.md#retinaface) | 263 | 185 | unsupported | 1165 |
| [`rf-detr-keypoint-preview`](../models/pose.md#rfdetr_keypoint) | 74.5 | 64.0 | unsupported | 978 |
| [`rtmo-l`](../models/pose.md#rtmo) | 62.0 | 55.0 | unsupported | 1188 |
| [`rtmo-m`](../models/pose.md#rtmo) | 38.7 | 35.1 | unsupported | 917 |
| [`rtmo-s`](../models/pose.md#rtmo) | 21.2 | 19.0 | unsupported | 607 |
| [`rtmo-t`](../models/pose.md#rtmo) | 12.3 | 11.8 | unsupported | 486 |
| [`rtmpose-l`](../models/pose.md#rtmpose) | 71.1 | 61.9 | – | 1054 |
| [`rtmpose-l-384`](../models/pose.md#rtmpose) | 138 | 118 | – | 1159 |
| [`rtmpose-l-halpe26`](../models/pose.md#rtmpose) | 70.5 | 62.5 | – | 1077 |
| [`rtmpose-l-halpe26-384`](../models/pose.md#rtmpose) | 139 | 120 | – | 1164 |
| [`rtmpose-m`](../models/pose.md#rtmpose) | 50.5 | 46.1 | – | 810 |
| [`rtmpose-m-384`](../models/pose.md#rtmpose) | 106 | 89.9 | – | 870 |
| [`rtmpose-m-ap10k`](../models/pose.md#rtmpose) | 63.9 | 57.2 | – | 853 |
| [`rtmpose-m-face6`](../models/pose.md#rtmpose) | 70.8 | 63.6 | – | 899 |
| [`rtmpose-m-halpe26`](../models/pose.md#rtmpose) | 50.4 | 46.5 | – | 816 |
| [`rtmpose-m-halpe26-384`](../models/pose.md#rtmpose) | 103 | 90.4 | – | 874 |
| [`rtmpose-m-hand5`](../models/pose.md#rtmpose) | 64.5 | 57.4 | – | 859 |
| [`rtmpose-s`](../models/pose.md#rtmpose) | 25.3 | 24.1 | – | 580 |
| [`rtmpose-s-face6`](../models/pose.md#rtmpose) | 36.1 | 34.2 | – | 709 |
| [`rtmpose-s-halpe26`](../models/pose.md#rtmpose) | 25.2 | 24.3 | – | 589 |
| [`rtmpose-t`](../models/pose.md#rtmpose) | 27.7 | 23.1 | unsupported | 521 |
| [`rtmpose-t-face6`](../models/pose.md#rtmpose) | 33.3 | 31.9 | – | 602 |
| [`rtmpose-t-halpe26`](../models/pose.md#rtmpose) | 24.3 | 23.7 | – | 527 |
| [`rtmpose-x-halpe26-384`](../models/pose.md#rtmpose) | 283 | 230 | – | 1414 |
| [`rtmw-l`](../models/pose.md#rtmpose) | 134 | 126 | – | 1258 |
| [`rtmw-l-384`](../models/pose.md#rtmpose) | 283 | 246 | – | 1343 |
| [`rtmw-m`](../models/pose.md#rtmpose) | 98.9 | 88.6 | – | 983 |
| [`rtmw-x`](../models/pose.md#rtmpose) | 235 | 205 | – | 1605 |
| [`rtmw-x-384`](../models/pose.md#rtmpose) | 531 | 418 | – | 1713 |
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
| [`c-radio-v3-b`](../models/features.md#c_radio) | 16.7 | 15.0 | unsupported | – | – | 518 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | 180 | 151 | unsupported | – | – | 2590 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | 104 | 88.6 | unsupported | – | – | 1617 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | 52.6 | 45.6 | unsupported | – | – | 932 | – |
| [`c-radio-v4-h`](../models/features.md#c_radio) | 104 | 88.6 | unsupported | – | – | 1619 | – |
| [`c-radio-v4-so400m`](../models/features.md#c_radio) | 71.6 | 62.5 | unsupported | – | – | 1166 | – |
| [`clip-b16-laion2b`](../models/features.md#clip) | 17.2 | 15.0 | unsupported | – | – | 680 | – |
| [`clip-b16-openai`](../models/features.md#clip) | 16.9 | 15.3 | unsupported | – | – | 673 | – |
| [`clip-h14-378-dfn5b`](../models/features.md#clip) | 141 | 108 | unsupported | – | – | 1707 | – |
| [`clip-h14-dfn5b`](../models/features.md#clip) | 131 | 108 | unsupported | – | – | 1704 | – |
| [`clip-h14-laion2b`](../models/features.md#clip) | 130 | 108 | unsupported | – | – | 1736 | – |
| [`clip-l14-336-openai`](../models/features.md#clip) | 65.4 | 56.2 | unsupported | – | – | 1133 | – |
| [`clip-l14-datacompxl`](../models/features.md#clip) | 65.2 | 56.2 | unsupported | – | – | 1112 | – |
| [`clip-l14-dfn2b`](../models/features.md#clip) | 65.1 | 57.2 | unsupported | – | – | 1116 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | 65.2 | 56.1 | unsupported | – | – | 1116 | – |
| [`clip-l14-openai`](../models/features.md#clip) | 65.3 | 55.6 | unsupported | – | – | 1151 | – |
| [`convnext_base.fb_in22k`](../models/features.md#convnext_features) | 21.1 | 17.8 | 20.3 | 16.4 | 1.1x | 961 | 958 |
| [`convnext_large.fb_in22k`](../models/features.md#convnext_features) | 35.0 | 30.2 | 34.0 | 28.5 | 1.1x | 1711 | 1710 |
| [`convnext_small.fb_in22k`](../models/features.md#convnext_features) | 14.7 | 12.1 | 14.1 | 12.1 | 1.0x | 691 | 691 |
| [`convnext_tiny.fb_in22k`](../models/features.md#convnext_features) | 8.47 | 7.59 | 9.87 | 8.62 | 0.9x | 605 | 605 |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 20.6 | 18.0 | unsupported | – | – | 439 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 20.7 | 17.9 | unsupported | – | – | 442 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 225 | 189 | unsupported | – | – | 2527 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 225 | 189 | unsupported | – | – | 2529 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 64.6 | 55.6 | unsupported | – | – | 889 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 64.8 | 55.5 | unsupported | – | – | 876 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 8.21 | 7.01 | unsupported | – | – | 268 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 8.27 | 7.15 | unsupported | – | – | 269 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 19.3 | 15.7 | 19.3 | 15.7 | 1.0x | 530 | 529 |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 34.2 | 28.2 | 33.2 | 28.2 | 1.0x | 841 | 842 |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 13.8 | 11.8 | 14.4 | 11.9 | 1.0x | 407 | 408 |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 9.13 | 7.23 | 8.38 | 7.71 | 0.9x | 333 | 331 |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 17.6 | 15.1 | unsupported | – | – | 438 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 139 | 116 | unsupported | – | – | 1949 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 54.7 | 46.4 | unsupported | – | – | 909 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 7.33 | 6.41 | unsupported | – | – | 261 | – |
| [`pe-core-b16-224`](../models/features.md#perception_encoder) | 18.6 | 23.3 | unsupported | – | – | 901 | – |
| [`pe-core-g14-448`](../models/features.md#perception_encoder) | 3316 | 379 | unsupported | – | – | 4584 | – |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | 68.7 | 86.5 | unsupported | – | – | 2067 | – |
| [`pe-core-s16-384`](../models/features.md#perception_encoder) | 8.33 | 10.5 | unsupported | – | – | 561 | – |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | 7.95 | 4.98 | unsupported | – | – | 431 | – |
| [`pe-spatial-b16-512`](../models/features.md#perception_encoder) | 17.8 | 22.6 | unsupported | – | – | 915 | – |
| [`pe-spatial-g14-448`](../models/features.md#perception_encoder) | 14121 | 377 | unsupported | – | – | 4376 | – |
| [`pe-spatial-l14-448`](../models/features.md#perception_encoder) | 69.1 | 84.2 | unsupported | – | – | 2035 | – |
| [`pe-spatial-s16-512`](../models/features.md#perception_encoder) | 8.09 | 9.28 | unsupported | – | – | 550 | – |
| [`pe-spatial-t16-512`](../models/features.md#perception_encoder) | 6.40 | 5.02 | unsupported | – | – | 430 | – |
| [`siglip2-b16-256`](../models/features.md#siglip2) | 17.1 | 15.2 | unsupported | – | – | 707 | – |
| [`siglip2-b16-512`](../models/features.md#siglip2) | 17.3 | 15.1 | unsupported | – | – | 713 | – |
| [`siglip2-g16-256`](../models/features.md#siglip2) | 187 | 154 | unsupported | – | – | 2791 | – |
| [`siglip2-g16-384`](../models/features.md#siglip2) | 187 | 154 | unsupported | – | – | 2794 | – |
| [`siglip2-l16-256`](../models/features.md#siglip2) | 53.2 | 45.8 | unsupported | – | – | 1133 | – |
| [`siglip2-l16-512`](../models/features.md#siglip2) | 53.5 | 45.5 | unsupported | – | – | 1148 | – |
| [`siglip2-so400m14-224`](../models/features.md#siglip2) | 79.1 | 65.1 | unsupported | – | – | 1337 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | 75.9 | 65.1 | unsupported | – | – | 1365 | – |
| [`siglip2-so400m16-256`](../models/features.md#siglip2) | 74.2 | 64.3 | unsupported | – | – | 1324 | – |
| [`siglip2-so400m16-512`](../models/features.md#siglip2) | 75.0 | 64.6 | unsupported | – | – | 1320 | – |
| [`tipsv2-b14`](../models/features.md#tipsv2) | 20.6 | 18.0 | unsupported | – | – | 433 | – |
| [`tipsv2-l14`](../models/features.md#tipsv2) | 64.5 | 55.4 | unsupported | – | – | 877 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Retrieval

### Retrieval, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`boq-dinov2`](../models/retrieval.md#boq) | 52.1 | 55.3 | 910 |
| [`megaloc`](../models/retrieval.md#megaloc) | 55.3 | 87.3 | 1420 |

## Calibration

### Calibration, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`anycalib-pinhole`](../models/calibration.md#anycalib) | 310 | 295 | unsupported | 1287 |
| [`geocalib-pinhole`](../models/calibration.md#geocalib) | 314 | 276 | unsupported | 961 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

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
| [`flowformer-small-chairs`](../models/flow.md#flowformer) | 20537 | 13243 | unsupported | 1889 |
| [`flowformer-small-sintel`](../models/flow.md#flowformer) | 20007 | 12987 | unsupported | 2056 |
| [`flowformer-small-things`](../models/flow.md#flowformer) | 20971 | 13811 | unsupported | 1888 |
| [`flowseek-t`](../models/flow.md#flowseek) | 3246 | 2160 | unsupported | 5781 |
| [`flowseek-t-things`](../models/flow.md#flowseek) | 2523 | 1760 | unsupported | 5769 |
| [`flowseek-t-things-notartan`](../models/flow.md#flowseek) | 2544 | 1760 | unsupported | 5766 |
| [`gmflow-scale1-mixdata`](../models/flow.md#gmflow) | 2482 | 1820 | unsupported | 1699 |
| [`gmflow-scale1-things`](../models/flow.md#gmflow) | 2627 | 1934 | unsupported | 1698 |
| [`gmflow-scale2-mixdata`](../models/flow.md#gmflow) | 5997 | 4132 | unsupported | 2559 |
| [`gmflow-scale2-regrefine6-kitti`](../models/flow.md#gmflow) | 9845 | 6520 | unsupported | 2791 |
| [`gmflow-scale2-regrefine6-mixdata`](../models/flow.md#gmflow) | 9804 | 6693 | unsupported | 2789 |
| [`gmflow-scale2-regrefine6-sintelft`](../models/flow.md#gmflow) | 9787 | 6491 | unsupported | 2788 |
| [`gmflow-scale2-regrefine6-things`](../models/flow.md#gmflow) | 9869 | 6487 | unsupported | 2834 |
| [`gmflow-scale2-sintel`](../models/flow.md#gmflow) | 5979 | 4129 | unsupported | 2494 |
| [`gmflow-scale2-things`](../models/flow.md#gmflow) | 5990 | 4167 | unsupported | 2554 |
| [`megaflow`](../models/flow.md#megaflow) | 13158 | 8480 | – | 6561 |
| [`memflow`](../models/flow.md#memflow) | 9908 | 6997 | unsupported | 8345 |
| [`memflow-kitti`](../models/flow.md#memflow) | 9851 | 6988 | – | 7586 |
| [`memflow-sintel`](../models/flow.md#memflow) | 9860 | 6990 | – | 8075 |
| [`memfof`](../models/flow.md#memfof) | 3807 | 3317 | unsupported | 1986 |
| [`memfof-kitti`](../models/flow.md#memfof) | 19707 | 16074 | – | 3652 |
| [`memfof-sintel`](../models/flow.md#memfof) | 19915 | 16049 | – | 3643 |
| [`memfof-spring`](../models/flow.md#memfof) | 3774 | 3319 | unsupported | 1986 |
| [`neuflow-v2`](../models/flow.md#neuflow_v2) | 781 | 790 | unsupported | 9253 |
| [`neuflow-v2-sintel`](../models/flow.md#neuflow_v2) | 979 | 960 | unsupported | 9271 |
| [`neuflow-v2-things`](../models/flow.md#neuflow_v2) | 941 | 951 | unsupported | 9267 |
| [`raft-large`](../models/flow.md#raft) | 1348 | 916 | unsupported | 1829 |
| [`raft-large-kitti`](../models/flow.md#raft) | 1358 | 913 | unsupported | 1832 |
| [`raft-large-things`](../models/flow.md#raft) | 1344 | 916 | unsupported | 1828 |
| [`raft-small`](../models/flow.md#raft) | 403 | 294 | unsupported | 1092 |
| [`recover-cx`](../models/flow.md#recover) | 1064 | 758 | unsupported | 1297 |
| [`recover-mn`](../models/flow.md#recover) | 471 | 316 | unsupported | 1181 |
| [`recover-rn`](../models/flow.md#recover) | 624 | 429 | unsupported | 1224 |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 1296 | 928 | unsupported | 2002 |
| [`sea-raft-m-kitti`](../models/flow.md#sea_raft) | 1297 | 928 | unsupported | 2146 |
| [`sea-raft-m-spring`](../models/flow.md#sea_raft) | 319 | 265 | unsupported | 1023 |
| [`sea-raft-m-things`](../models/flow.md#sea_raft) | 1296 | 928 | unsupported | 2125 |
| [`sea-raft-s`](../models/flow.md#sea_raft) | 832 | 617 | unsupported | 2019 |
| [`sea-raft-s-kitti`](../models/flow.md#sea_raft) | 828 | 621 | unsupported | 2021 |
| [`sea-raft-s-spring`](../models/flow.md#sea_raft) | 218 | 167 | unsupported | 985 |
| [`waft-a1`](../models/flow.md#waft) | 8587 | 6080 | unsupported | 2583 |
| [`waft-a1-kitti`](../models/flow.md#waft) | 8585 | 6078 | unsupported | 2564 |
| [`waft-a1-sintel`](../models/flow.md#waft) | 8590 | 6076 | unsupported | 2564 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matching

### Matching, synthetic, 768x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`aliked-flexglue`](../models/matching.md#flexglue) | 938 | 992 | unsupported | – | – | 1644 | – |
| [`aliked-lightglue`](../models/matching.md#lightglue) | 101 | 102 | unsupported | – | – | 1105 | – |
| [`dedode-v2-b`](../models/matching.md#dedode) | 2389 | 1814 | unsupported | – | – | 3622 | – |
| [`dedode-v2-g`](../models/matching.md#dedode) | 4240 | 3222 | – | – | – | 4226 | – |
| [`disk-lightglue`](../models/matching.md#lightglue) | 775 | 697 | unsupported | – | – | 1729 | – |
| [`loftr-indoor`](../models/matching.md#loftr) | 1101 | 442 | unsupported | – | – | 2129 | – |
| [`loftr-outdoor`](../models/matching.md#loftr) | 997 | 483 | unsupported | – | – | 2228 | – |
| [`loma-b`](../models/matching.md#loma) | 3617 | 2760 | – | – | – | 4561 | – |
| [`loma-b128`](../models/matching.md#loma) | 1546 | 1085 | – | – | – | 1902 | – |
| [`loma-b64`](../models/matching.md#loma) | 1228 | 828 | – | – | – | 1630 | – |
| [`loma-g`](../models/matching.md#loma) | 4152 | 3230 | unsupported | – | – | 5159 | – |
| [`loma-l`](../models/matching.md#loma) | 3706 | 2810 | – | – | – | 4929 | – |
| [`loma-r`](../models/matching.md#loma) | 3622 | 2708 | unsupported | – | – | 4549 | – |
| [`romav1`](../models/matching.md#romav1) | 3733 | 3135 | – | – | – | 2460 | – |
| [`romav2`](../models/matching.md#romav2) | 5903 | 5563 | 3388 | 3368 | 1.7x | 2949 | 3128 |
| [`xfeat`](../models/matching.md#xfeat) | 29.9 | – | unsupported | – | – | 642 | – |
| [`xfeat-lighterglue`](../models/matching.md#xfeat) | 135 | – | unsupported | – | – | 816 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Tracking

### Tracking, synthetic, 480x854

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`alltracker`](../models/tracking.md#alltracker) | 3151 | 2288 | – | 2232 |
| [`alltracker-tiny`](../models/tracking.md#alltracker) | 2909 | 2174 | – | 2245 |
| [`bootstapir`](../models/tracking.md#tapir) | 2003 | – | unsupported | 1925 |
| [`cotracker3-offline`](../models/tracking.md#cotracker) | 1595 | 1455 | – | 3219 |
| [`cotracker3-online`](../models/tracking.md#cotracker) | 1638 | 1491 | – | 2453 |
| [`locotrack-b`](../models/tracking.md#locotrack) | 1453 | 1113 | – | 1597 |
| [`locotrack-s`](../models/tracking.md#locotrack) | 1233 | 961 | – | 1575 |
| [`tapir`](../models/tracking.md#tapir) | 1319 | – | unsupported | 1601 |
| [`tapnext`](../models/tracking.md#tapnext) | 4033 | 3724 | – | 2420 |
| [`tapnext-pp`](../models/tracking.md#tapnext) | 4035 | 3746 | – | 1677 |
| [`track-on-r`](../models/tracking.md#track_on) | 11174 | 8822 | unsupported | 3089 |
| [`track-on2`](../models/tracking.md#track_on) | 10543 | 8827 | unsupported | 3090 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Interpolation

### Interpolation, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`ema-vfi`](../models/interpolation.md#ema_vfi) | 1741 | 1320 | – | 3386 |
| [`ema-vfi-small`](../models/interpolation.md#ema_vfi) | 538 | 426 | unsupported | 1930 |
| [`ema-vfi-small-t`](../models/interpolation.md#ema_vfi) | 539 | 427 | – | 1932 |
| [`ema-vfi-t`](../models/interpolation.md#ema_vfi) | 1713 | 1325 | – | 3463 |
| [`film`](../models/interpolation.md#film) | 9227 | 6590 | – | 9495 |
| [`gimm-vfi-r`](../models/interpolation.md#gimm_vfi) | 58742 | 30451 | – | 12185 |
| [`gimm-vfi-r-p`](../models/interpolation.md#gimm_vfi) | 39847 | 16157 | unsupported | 10310 |
| [`rife-4.25`](../models/interpolation.md#rife) | 91.8 | 78.5 | – | 921 |
| [`rife-4.25-lite`](../models/interpolation.md#rife) | 73.3 | 63.3 | – | 935 |
| [`rife-4.26`](../models/interpolation.md#rife) | 92.2 | 78.3 | – | 923 |

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
| [`nafnet-gopro`](../models/restoration.md#nafnet) | 458 | 293 | – | 463 | 293 | 1.0x | 1359 | 1359 |
| [`nafnet-sidd`](../models/restoration.md#nafnet) | 383 | 232 | – | 380 | 228 | 1.0x | 1395 | 1395 |
| [`real-hat-gan-x4`](../models/restoration.md#hat) | 24682 | 15831 | – | – | – | – | 5687 | – |
| [`realesr-general-x4v3`](../models/restoration.md#compact) | 333 | 246 | – | 177 | 104 | 2.4x | 737 | 1061 |
| [`realesrgan-x2plus`](../models/restoration.md#realesrgan) | 1599 | 1199 | – | 1667 | 567 | 2.1x | 974 | 1222 |
| [`realesrgan-x4plus`](../models/restoration.md#realesrgan) | 7548 | 5321 | – | 7380 | 2184 | 2.4x | 1390 | 2452 |
| [`realesrgan-x4plus-anime-6b`](../models/restoration.md#realesrgan) | 2700 | 1617 | – | 2038 | 849 | 1.9x | 1319 | 2106 |
| [`realplksr-nomoswebphoto-x4`](../models/restoration.md#realplksr) | 2191 | 1500 | – | 2122 | 1020 | 1.5x | 1215 | 1923 |
| [`restormer-deblur`](../models/restoration.md#restormer) | 809 | 554 | – | unsupported | – | – | 2195 | – |
| [`restormer-derain`](../models/restoration.md#restormer) | 806 | 552 | – | unsupported | – | – | 2262 | – |
| [`restormer-sidd`](../models/restoration.md#restormer) | 796 | 551 | – | unsupported | – | – | 2264 | – |
| [`seedvr2-3b`](../models/restoration.md#seedvr2) | – | 14023 | 14307 | – | – | – | 5178 | – |
| [`seedvr2-7b`](../models/restoration.md#seedvr2) | – | 32426 | 25365 | – | – | – | 6547 | – |
| [`span-x2`](../models/restoration.md#span) | 180 | 134 | – | 171 | 76.5 | 1.7x | 753 | 1073 |
| [`span-x4`](../models/restoration.md#span) | 190 | 149 | – | 179 | 86.0 | 1.7x | 827 | 1210 |
| [`swin2sr-x2`](../models/restoration.md#swin2sr) | 8277 | 5144 | – | – | – | – | 1980 | – |
| [`swin2sr-x4`](../models/restoration.md#swin2sr) | 8163 | 4902 | – | – | – | – | 3313 | – |
| [`swin2sr-x4-real`](../models/restoration.md#swin2sr) | 8813 | 5522 | – | – | – | – | 2268 | – |
| [`swinir-l-x4-real`](../models/restoration.md#swinir) | 14252 | 8691 | – | – | – | – | 2437 | – |
| [`swinir-m-x2`](../models/restoration.md#swinir) | 6857 | 4312 | – | – | – | – | 2423 | – |
| [`swinir-m-x4`](../models/restoration.md#swinir) | 6706 | 4217 | – | unsupported | – | – | 3278 | – |
| [`swinir-m-x4-real`](../models/restoration.md#swinir) | 7297 | 4668 | – | – | – | – | 2243 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Volume

### Volume, synthetic, 192x192

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`nnfoundation-vit`](../models/volume.md#nnfoundation) | 17526 | 13240 | unsupported | 2519 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Volume, synthetic, 200x200

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`nnfoundation-cnn`](../models/volume.md#nnfoundation) | 4271 | 3520 | 8041 |
| [`nninteractive`](../models/volume.md#nninteractive) | 4005 | 3797 | 5587 |
| [`totalsegmentator-total`](../models/volume.md#totalsegmentator) | 25905 | 19980 | 5853 |
| [`totalsegmentator-total-3mm`](../models/volume.md#totalsegmentator) | 709 | 495 | 4254 |
<!-- device:end -->
