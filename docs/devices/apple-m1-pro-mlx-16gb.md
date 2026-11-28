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
| measured | 2026-11-19 to 2026-11-28 |
| rows | 1740 in `docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` |

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
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 4902 | 2459 | – | – | – | 4587 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 5559 | 3039 | unsupported | – | – | 7831 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 2960 | 2104 | unsupported | – | – | 3950 | – |
| [`coex-sceneflow`](../models/stereo.md#coex) | 282 | 215 | – | – | – | 947 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 6118 | 4297 | 5212 | 3558 | 1.2x | 1557 | 1863 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 13339 | 10693 | – | – | – | 5189 | – |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 2234 | 1711 | unsupported | – | – | 2841 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 2276 | 1662 | unsupported | – | – | 2844 | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 383 | 334 | unsupported | – | – | 1202 | – |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | 387 | 338 | unsupported | – | – | 1148 | – |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | 388 | 339 | unsupported | – | – | 1149 | – |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | 462 | 412 | unsupported | – | – | 1392 | – |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | 455 | 408 | unsupported | – | – | 1394 | – |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | 456 | 408 | unsupported | – | – | 1400 | – |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | 455 | 408 | unsupported | – | – | 1395 | – |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | 382 | 338 | unsupported | – | – | 1217 | – |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 4326 | 2875 | unsupported | – | – | 2829 | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | – | 25847 | – | – | – | 4941 | – |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | 40293 | 20504 | – | 17814 | 1.2x | 4943 | 4613 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | 7350 | 5732 | unsupported | – | – | 2890 | – |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | 7301 | 5730 | unsupported | – | – | 2864 | – |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | 15126 | 12474 | – | – | – | 4619 | – |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | 7503 | 5741 | – | – | – | 2865 | – |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 2777 | 1769 | – | – | – | 2965 | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | 3083 | 1991 | unsupported | – | – | 3015 | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 335 | 256 | unsupported | – | – | 1158 | – |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | 817 | 624 | unsupported | – | – | 1628 | – |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 7874 | 5853 | 7868 | 5726 | 1.0x | 2953 | 2959 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 7956 | 5685 | 7821 | 5705 | 1.0x | 3016 | 2963 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 7788 | 5768 | 7862 | 5701 | 1.0x | 2962 | 2962 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 8011 | 5700 | 8065 | 5726 | 1.0x | 2920 | 2930 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 7888 | 5846 | 7876 | 5724 | 1.0x | 2949 | 2963 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 4851 | 3306 | unsupported | – | – | 2551 | – |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 4798 | 3296 | unsupported | – | – | 2559 | – |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 4872 | 3303 | unsupported | – | – | 2570 | – |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 4833 | 3301 | unsupported | – | – | 2555 | – |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 4866 | 3301 | unsupported | – | – | 2558 | – |
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
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 3042 | 2111 | unsupported | – | – | 2602 | – |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 5141 | 3193 | 4194 | 2361 | 1.4x | 2719 | 3875 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 5151 | 3214 | 4193 | 2370 | 1.4x | 2746 | 3898 |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 5151 | 3169 | 4162 | 2361 | 1.3x | 2741 | 3874 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 891 | 605 | 852 | 511 | 1.2x | 1453 | 1591 |
| [`s2m2-l`](../models/stereo.md#s2m2) | 5554 | 4233 | – | – | – | 4688 | – |
| [`s2m2-m`](../models/stereo.md#s2m2) | 3361 | 2588 | – | – | – | 3796 | – |
| [`s2m2-s`](../models/stereo.md#s2m2) | 1810 | 1379 | unsupported | – | – | 3047 | – |
| [`s2m2-xl`](../models/stereo.md#s2m2) | 10391 | 8122 | – | – | – | 6420 | – |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 5420 | 3757 | 5448 | 3751 | 1.0x | 2776 | 2767 |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | 5405 | 3753 | 5386 | 3755 | 1.0x | 2777 | 2768 |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | 5443 | 3745 | 5363 | 3757 | 1.0x | 2764 | 2781 |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | 5404 | 3751 | 5369 | 3756 | 1.0x | 2774 | 2770 |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | 5453 | 3747 | 5378 | 3756 | 1.0x | 2932 | 2772 |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | 5410 | 3765 | 5381 | 3758 | 1.0x | 2779 | 2768 |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 5425 | 3754 | 5363 | 3741 | 1.0x | 2781 | 2772 |
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
| [`coatnet_0_rw_224.sw_in1k`](../models/classification.md#coatnet) | 9.98 | 9.25 | 9.54 | 8.50 | 1.1x | 501 | 513 |
| [`coatnet_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 16.5 | 14.0 | 15.6 | 13.7 | 1.0x | 582 | 613 |
| [`coatnet_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 23.5 | 19.3 | 24.5 | 18.8 | 1.0x | 691 | 753 |
| [`coatnet_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 8.18 | 6.93 | 7.33 | 7.11 | 1.0x | 440 | 455 |
| [`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 16.6 | 14.3 | 16.2 | 14.1 | 1.0x | 578 | 613 |
| [`coatnet_rmlp_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 16.4 | 14.7 | 16.7 | 13.5 | 1.1x | 570 | 612 |
| [`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 25.5 | 19.2 | 23.4 | 19.5 | 1.0x | 710 | 776 |
| [`coatnet_rmlp_2_rw_224.sw_in1k`](../models/classification.md#coatnet) | 23.4 | 19.8 | 23.4 | 19.1 | 1.0x | 727 | 763 |
| [`coatnet_rmlp_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 8.21 | 7.07 | 8.38 | 7.07 | 1.0x | 450 | 457 |
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 17.0 | 15.1 | 21.1 | 16.2 | 0.9x | 609 | 646 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 17.1 | 15.1 | 20.1 | 17.2 | 0.9x | 608 | 651 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 36.5 | 31.1 | 45.0 | 37.5 | 0.8x | 685 | 750 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 32.3 | 27.3 | 33.7 | 28.6 | 1.0x | 931 | 966 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 32.2 | 27.3 | 34.1 | 28.7 | 1.0x | 933 | 965 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 69.3 | 60.0 | 73.2 | 62.8 | 1.0x | 1005 | 1037 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 12.4 | 10.9 | 14.5 | 12.3 | 0.9x | 499 | 522 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 12.3 | 11.1 | 14.1 | 12.7 | 0.9x | 498 | 521 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 24.4 | 21.7 | 30.8 | 25.0 | 0.9x | 552 | 625 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 7.26 | 6.44 | 9.08 | 7.97 | 0.8x | 442 | 450 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 7.22 | 6.46 | 9.44 | 8.11 | 0.8x | 442 | 446 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 14.6 | 12.9 | 18.6 | 15.9 | 0.8x | 496 | 518 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 19.8 | 18.0 | 23.2 | 20.6 | 0.9x | 673 | 772 |
| [`davit_base.msft_in1k`](../models/classification.md#davit) | 21.6 | 19.6 | 22.2 | 19.9 | 1.0x | 660 | 732 |
| [`davit_small.msft_in1k`](../models/classification.md#davit) | 17.1 | 15.7 | 17.6 | 16.0 | 1.0x | 531 | 586 |
| [`davit_tiny.msft_in1k`](../models/classification.md#davit) | 10.4 | 9.66 | 10.6 | 9.84 | 1.0x | 481 | 496 |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 17.0 | 14.8 | unsupported | – | – | 402 | – |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 11.3 | 9.71 | unsupported | – | – | 416 | – |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 65.0 | 56.1 | unsupported | – | – | 1108 | – |
| [`efficientformer_l1.snap_dist_in1k`](../models/classification.md#efficientformer) | 4.62 | 4.35 | 4.25 | 3.96 | 1.1x | 397 | 405 |
| [`efficientformer_l3.snap_dist_in1k`](../models/classification.md#efficientformer) | 7.49 | 6.68 | 7.98 | 6.92 | 1.0x | 474 | 483 |
| [`efficientformer_l7.snap_dist_in1k`](../models/classification.md#efficientformer) | 15.7 | 13.2 | 16.0 | 13.4 | 1.0x | 644 | 664 |
| [`efficientformerv2_l.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 10.2 | 9.78 | 10.8 | 11.5 | 0.9x | 450 | 561 |
| [`efficientformerv2_s0.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 4.22 | 3.77 | 4.32 | 4.15 | 0.9x | 349 | 371 |
| [`efficientformerv2_s1.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 4.79 | 4.41 | 6.08 | 7.28 | 0.7x | 358 | 391 |
| [`efficientformerv2_s2.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 7.21 | 6.63 | 7.20 | 7.73 | 0.9x | 390 | 451 |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 3.98 | 4.70 | 4.41 | 4.12 | 1.0x | 355 | 345 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 16.5 | 14.5 | 16.0 | 14.2 | 1.0x | 539 | 598 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 16.4 | 14.7 | 16.5 | 14.5 | 1.0x | 480 | 505 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 8.93 | 8.12 | 9.19 | 7.21 | 1.1x | 528 | 539 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 243 | 195 | unsupported | – | – | 930 | – |
| [`falconsai-nsfw-vit`](../models/classification.md#nsfw_vit) | 16.3 | 14.1 | unsupported | – | – | 398 | – |
| [`focalnet_base_lrf.ms_in1k`](../models/classification.md#focalnet) | 23.1 | 19.9 | 24.4 | 21.5 | 0.9x | 633 | 1223 |
| [`focalnet_base_srf.ms_in1k`](../models/classification.md#focalnet) | 23.3 | 19.1 | 22.4 | 18.9 | 1.0x | 627 | 1220 |
| [`focalnet_small_lrf.ms_in1k`](../models/classification.md#focalnet) | 17.7 | 16.1 | 19.3 | 16.8 | 1.0x | 511 | 859 |
| [`focalnet_small_srf.ms_in1k`](../models/classification.md#focalnet) | 16.6 | 15.3 | 16.3 | 15.4 | 1.0x | 501 | 858 |
| [`focalnet_tiny_lrf.ms_in1k`](../models/classification.md#focalnet) | 10.5 | 9.54 | 10.6 | 9.56 | 1.0x | 458 | 696 |
| [`focalnet_tiny_srf.ms_in1k`](../models/classification.md#focalnet) | 9.77 | 9.11 | 9.21 | 9.55 | 1.0x | 457 | 761 |
| [`hgnetv2_b0.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 3.09 | 2.84 | 2.98 | 2.53 | 1.1x | 381 | 416 |
| [`hgnetv2_b1.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 2.92 | 2.71 | 3.67 | 2.95 | 0.9x | 388 | 424 |
| [`hgnetv2_b2.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 4.08 | 3.71 | 4.39 | 3.95 | 0.9x | 431 | 508 |
| [`hgnetv2_b3.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 4.98 | 4.47 | 5.66 | 4.45 | 1.0x | 467 | 589 |
| [`hgnetv2_b4.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 6.31 | 5.71 | 6.34 | 5.78 | 1.0x | 494 | 645 |
| [`hgnetv2_b5.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 11.5 | 10.1 | 11.8 | 10.2 | 1.0x | 615 | 923 |
| [`hgnetv2_b6.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 24.3 | 19.9 | 27.0 | 21.3 | 0.9x | 792 | 1289 |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 10.5 | 8.45 | 9.27 | 8.41 | 1.0x | 490 | 489 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 3.23 | 2.46 | 3.01 | 2.50 | 1.0x | 327 | 322 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 3.27 | 3.00 | 3.23 | 3.14 | 1.0x | 333 | 332 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 3.18 | 2.62 | unsupported | – | – | 353 | – |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 3.90 | 3.70 | 3.89 | 3.58 | 1.0x | 389 | 389 |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 19.6 | 18.3 | 19.8 | 18.2 | 1.0x | 747 | 763 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 4.51 | 5.65 | unsupported | – | – | 385 | – |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 5.20 | 4.40 | unsupported | – | – | 364 | – |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 4.99 | 4.84 | unsupported | – | – | 383 | – |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 6.20 | 4.45 | unsupported | – | – | 379 | – |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 4.66 | 4.25 | unsupported | – | – | 392 | – |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 5.05 | 4.36 | unsupported | – | – | 387 | – |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 6.90 | 6.46 | unsupported | – | – | 431 | – |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 7.00 | 6.53 | unsupported | – | – | 432 | – |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 11.3 | 10.0 | unsupported | – | – | 491 | – |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 11.4 | 10.5 | unsupported | – | – | 494 | – |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 18.2 | 16.1 | 19.2 | 16.4 | 1.0x | 551 | 551 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 4.83 | 4.05 | 4.62 | 4.24 | 1.0x | 359 | 353 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 7.43 | 6.49 | 8.24 | 6.84 | 0.9x | 439 | 444 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 2.98 | 3.01 | unsupported | – | – | 322 | – |
| [`shvit_s1.in1k`](../models/classification.md#shvit) | 4.27 | 4.54 | 3.88 | 3.41 | 1.3x | 359 | 352 |
| [`shvit_s2.in1k`](../models/classification.md#shvit) | 4.44 | 3.97 | 4.18 | 3.94 | 1.0x | 365 | 367 |
| [`shvit_s3.in1k`](../models/classification.md#shvit) | 5.13 | 4.52 | 4.94 | 4.45 | 1.0x | 379 | 377 |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 2.19 | 2.50 | unsupported | – | – | 314 | – |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 3.19 | 2.93 | unsupported | – | – | 339 | – |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 2.59 | 3.63 | unsupported | – | – | 353 | – |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 3.19 | 2.66 | unsupported | – | – | 349 | – |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 4.04 | 3.46 | unsupported | – | – | 358 | – |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 23.1 | 19.7 | 21.5 | 19.2 | 1.0x | 678 | 741 |
| [`tiny_vit_11m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 5.44 | 5.02 | 5.78 | 5.32 | 0.9x | 403 | 407 |
| [`tiny_vit_21m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 7.86 | 7.13 | 8.29 | 7.41 | 1.0x | 450 | 459 |
| [`tiny_vit_5m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 4.68 | 4.24 | 6.97 | 5.31 | 0.8x | 369 | 374 |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 17.4 | 13.9 | unsupported | – | – | 2003 | – |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 16.7 | 14.6 | unsupported | – | – | 400 | – |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 215 | 183 | unsupported | – | – | 1108 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Classification, synthetic, 256x256

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 7.70 | 7.09 | 7.56 | 6.88 | 1.0x | 513 | 528 |
| [`fastvit_t12.apple_dist_in1k`](../models/classification.md#fastvit) | 4.33 | 4.03 | 5.02 | 4.71 | 0.9x | 403 | 408 |
| [`fastvit_t8.apple_dist_in1k`](../models/classification.md#fastvit) | 3.76 | 3.16 | 3.65 | 3.47 | 0.9x | 366 | 369 |
| [`shvit_s4.in1k`](../models/classification.md#shvit) | 6.03 | 5.54 | 6.08 | 5.45 | 1.0x | 382 | 387 |

### Classification, synthetic, 384x384

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 59.1 | 45.3 | 63.3 | 47.2 | 1.0x | 1101 | 1132 |
| [`tiny_vit_21m_384.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 19.0 | 14.6 | 18.9 | 15.9 | 0.9x | 643 | 697 |

### Classification, synthetic, 448x448

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`wd-convnext-tagger-v3`](../models/classification.md#wd_tagger) | 51.4 | 43.4 | unsupported | 794 |
| [`wd-eva02-large-tagger-v3`](../models/classification.md#wd_tagger) | 242 | 195 | unsupported | 948 |
| [`wd-vit-large-tagger-v3`](../models/classification.md#wd_tagger) | 204 | 179 | unsupported | 905 |
| [`wd-vit-tagger-v3`](../models/classification.md#wd_tagger) | 54.2 | 45.2 | unsupported | 447 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Classification, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`tiny_vit_21m_512.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 33.4 | 25.3 | 41.5 | 29.1 | 0.9x | 918 | 919 |

## Segmentation

### Segmentation, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`bisenet-face-parsing`](../models/segmentation.md#bisenet) | 16.1 | 12.3 | unsupported | – | – | 371 | – |
| [`segformer-b2-clothes`](../models/segmentation.md#segformer) | 42.3 | 31.8 | 60.5 | 53.8 | 0.6x | 686 | 982 |
| [`segformer-b5-face-parsing`](../models/segmentation.md#segformer) | 91.4 | 72.9 | 108 | 94.1 | 0.8x | 900 | 1200 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

### Segmentation, synthetic, 1024x2048

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 723 | 500 | 601 | 444 | 1.1x | 1446 | 1542 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 460 | 358 | 478 | 359 | 1.0x | 1444 | 1552 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 313 | 256 | unsupported | – | – | 1356 | – |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 941 | 760 | unsupported | – | – | 2178 | – |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 135 | 115 | unsupported | – | – | 1201 | – |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 571 | 474 | – | – | – | 2317 | – |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 583 | 484 | – | – | – | 2357 | – |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 477 | 362 | 528 | 366 | 1.0x | 1359 | 1391 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 243 | 204 | 262 | 234 | 0.9x | 1054 | 1500 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 772 | 649 | unsupported | – | – | 2170 | – |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 972 | 865 | unsupported | – | – | 2137 | – |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 482 | 378 | 498 | 376 | 1.0x | 1530 | 1553 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 1553 | 1308 | unsupported | – | – | 2950 | – |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 3552 | 2683 | unsupported | – | – | 4522 | – |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 64.3 | 55.8 | 76.9 | 69.6 | 0.8x | 735 | 917 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 332 | 269 | 396 | 348 | 0.8x | 1179 | 2048 |
| [`segnext-mscan-b-ade20k`](../models/segmentation.md#segnext) | 137 | 118 | 126 | 119 | 1.0x | 861 | 940 |
| [`segnext-mscan-l-ade20k`](../models/segmentation.md#segnext) | 207 | 179 | 196 | 186 | 1.0x | 1029 | 1145 |
| [`segnext-mscan-s-ade20k`](../models/segmentation.md#segnext) | 95.7 | 82.8 | 87.1 | 81.6 | 1.0x | 751 | 830 |
| [`segnext-mscan-t-ade20k`](../models/segmentation.md#segnext) | 77.4 | 71.5 | 73.6 | 69.9 | 1.0x | 581 | 746 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matting

### Matting, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ben2`](../models/matting.md#ben2) | 1004 | 806 | unsupported | – | – | 2345 | – |
| [`birefnet`](../models/matting.md#birefnet) | 975 | 877 | unsupported | – | – | 1668 | – |
| [`birefnet-lite`](../models/matting.md#birefnet) | 351 | 274 | unsupported | – | – | 1027 | – |
| [`inspyrenet`](../models/matting.md#inspyrenet) | 1121 | 679 | unsupported | – | – | 1741 | – |
| [`inspyrenet-fast`](../models/matting.md#inspyrenet) | 129 | 95.6 | – | – | – | 1060 | – |
| [`isnet`](../models/matting.md#isnet) | 156 | 118 | 153 | 106 | 1.1x | 1005 | 1118 |
| [`isnet-general-use`](../models/matting.md#isnet) | 153 | 129 | 167 | 96.8 | 1.3x | 1016 | 1168 |
| [`matanyone`](../models/matting.md#matanyone) | 4891 | 3513 | unsupported | – | – | 3562 | – |
| [`rmbg-2.0`](../models/matting.md#birefnet) | 975 | 899 | unsupported | – | – | 1667 | – |
| [`u2net`](../models/matting.md#u2net) | 69.6 | 57.8 | unsupported | – | – | 723 | – |
| [`u2net-human-seg`](../models/matting.md#u2net) | 68.8 | 59.3 | unsupported | – | – | 709 | – |
| [`u2netp`](../models/matting.md#u2net) | 33.2 | 27.9 | unsupported | – | – | 504 | – |
| [`vitmatte-b`](../models/matting.md#vitmatte) | 1344 | 1075 | unsupported | – | – | 1246 | – |
| [`vitmatte-s`](../models/matting.md#vitmatte) | 754 | 570 | unsupported | – | – | 1073 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Edges

### Edges, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`anyline`](../models/edges.md#anyline) | 55.4 | 43.8 | 55.4 | 39.8 | 1.1x | 729 | 731 |
| [`lineart`](../models/edges.md#lineart) | 78.6 | 51.6 | 77.5 | 51.5 | 1.0x | 882 | 883 |
| [`lineart-coarse`](../models/edges.md#lineart) | 80.0 | 51.8 | 81.0 | 52.4 | 1.0x | 867 | 842 |
| [`pidinet`](../models/edges.md#pidinet) | 196 | 173 | 107 | 75.1 | 2.3x | 728 | 732 |

### Edges, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`hed`](../models/edges.md#hed) | 116 | 94.3 | unsupported | – | – | 1393 | – |
| [`lineart-anime`](../models/edges.md#lineart_anime) | 87.5 | 65.3 | unsupported | – | – | 1377 | – |
| [`mlsd`](../models/edges.md#mlsd) | 62.7 | 43.5 | unsupported | – | – | 890 | – |
| [`teed`](../models/edges.md#teed) | 19.0 | 16.5 | 23.8 | 16.6 | 1.0x | 534 | 535 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`clipseg`](../models/promptable_segmentation.md#clipseg) | 33.6 | 38.5 | – | 938 |
| [`efficientvit-sam-l0`](../models/promptable_segmentation.md#efficientvit_sam) | 59.8 | – | unsupported | 893 |
| [`efficientvit-sam-l1`](../models/promptable_segmentation.md#efficientvit_sam) | 68.7 | – | unsupported | 1090 |
| [`efficientvit-sam-l2`](../models/promptable_segmentation.md#efficientvit_sam) | 83.0 | – | unsupported | 1247 |
| [`efficientvit-sam-xl0`](../models/promptable_segmentation.md#efficientvit_sam) | 219 | – | unsupported | 2075 |
| [`efficientvit-sam-xl1`](../models/promptable_segmentation.md#efficientvit_sam) | 380 | – | unsupported | 2983 |
| [`insid3-vitl16`](../models/promptable_segmentation.md#insid3) | 3216 | 2512 | – | 2227 |
| [`mobile-sam`](../models/promptable_segmentation.md#mobile_sam) | 113 | 83.1 | unsupported | 807 |
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | 500 | 398 | – | 1374 |
| [`sam-hq-vit-h`](../models/promptable_segmentation.md#sam_hq) | 2318 | 1919 | unsupported | 2466 |
| [`sam-hq-vit-l`](../models/promptable_segmentation.md#sam_hq) | 1091 | 899 | – | 1884 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 409 | 352 | – | 1069 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 1968 | 1589 | unsupported | 2350 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1051 | 864 | – | 1639 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 272 | 229 | unsupported | 911 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 597 | 516 | unsupported | 1243 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 169 | 133 | unsupported | 798 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 145 | 115 | unsupported | 785 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 1894 | 1547 | unsupported | 2582 |

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
| [`detr-r101-coco`](../models/detection.md#detr) | 148 | 120 | – | – | – | 994 | – |
| [`detr-r50-coco`](../models/detection.md#detr) | 102 | 79.8 | unsupported | – | – | 954 | – |
| [`dfine-l-coco`](../models/detection.md#dfine) | 63.8 | 52.2 | unsupported | – | – | 785 | – |
| [`dfine-m-coco`](../models/detection.md#dfine) | 42.1 | 37.0 | unsupported | – | – | 663 | – |
| [`dfine-n-coco`](../models/detection.md#dfine) | 18.3 | 17.0 | unsupported | – | – | 406 | – |
| [`dfine-s-coco`](../models/detection.md#dfine) | 30.7 | 29.4 | unsupported | – | – | 512 | – |
| [`dfine-x-coco`](../models/detection.md#dfine) | 99.4 | 77.9 | unsupported | – | – | 1220 | – |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 146 | 123 | unsupported | – | – | 1123 | – |
| [`grounding-dino-base`](../models/detection.md#grounding_dino) | 689 | 606 | unsupported | – | – | 1645 | – |
| [`grounding-dino-tiny`](../models/detection.md#grounding_dino) | 463 | 387 | unsupported | – | – | 1525 | – |
| [`lw-detr-large-coco`](../models/detection.md#lwdetr) | 63.2 | 60.2 | unsupported | – | – | 1005 | – |
| [`lw-detr-medium-coco`](../models/detection.md#lwdetr) | 45.3 | 40.7 | unsupported | – | – | 730 | – |
| [`lw-detr-small-coco`](../models/detection.md#lwdetr) | 30.8 | 28.4 | unsupported | – | – | 661 | – |
| [`lw-detr-tiny-coco`](../models/detection.md#lwdetr) | 22.8 | 20.0 | unsupported | – | – | 590 | – |
| [`lw-detr-xlarge-coco`](../models/detection.md#lwdetr) | 111 | 105 | unsupported | – | – | 1440 | – |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 166 | 147 | unsupported | – | – | 1135 | – |
| [`mm-grounding-dino-base`](../models/detection.md#grounding_dino) | 1254 | 916 | – | – | – | 1719 | – |
| [`mm-grounding-dino-tiny`](../models/detection.md#grounding_dino) | 935 | 766 | – | – | – | 1507 | – |
| [`owlv2-b16`](../models/detection.md#owlv2) | 374 | 291 | unsupported | – | – | 885 | – |
| [`owlv2-l14`](../models/detection.md#owlv2) | 1879 | 1428 | unsupported | – | – | 1463 | – |
| [`rf-detr-l-coco`](../models/detection.md#rfdetr) | 56.5 | 48.1 | unsupported | – | – | 690 | – |
| [`rf-detr-m-coco`](../models/detection.md#rfdetr) | 39.7 | 35.0 | unsupported | – | – | 673 | – |
| [`rf-detr-n-coco`](../models/detection.md#rfdetr) | 21.2 | 19.4 | unsupported | – | – | 593 | – |
| [`rf-detr-s-coco`](../models/detection.md#rfdetr) | 31.9 | 28.8 | unsupported | – | – | 639 | – |
| [`rf-detr-seg-2xl-coco`](../models/detection.md#rfdetr) | 180 | 136 | unsupported | – | – | 1002 | – |
| [`rf-detr-seg-l-coco`](../models/detection.md#rfdetr) | 64.5 | 52.7 | unsupported | – | – | 743 | – |
| [`rf-detr-seg-m-coco`](../models/detection.md#rfdetr) | 49.0 | 41.5 | unsupported | – | – | 721 | – |
| [`rf-detr-seg-n-coco`](../models/detection.md#rfdetr) | 27.5 | 24.0 | unsupported | – | – | 626 | – |
| [`rf-detr-seg-s-coco`](../models/detection.md#rfdetr) | 36.1 | 31.3 | unsupported | – | – | 667 | – |
| [`rf-detr-seg-xl-coco`](../models/detection.md#rfdetr) | 108 | 85.1 | unsupported | – | – | 840 | – |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 65.0 | 55.1 | unsupported | – | – | 831 | – |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 53.1 | 44.4 | unsupported | – | – | 713 | – |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 53.9 | 46.2 | unsupported | – | – | 788 | – |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 39.2 | 33.2 | unsupported | – | – | 631 | – |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 104 | 88.6 | unsupported | – | – | 1115 | – |
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
| [`hrnet-w32`](../models/pose.md#hrnet) | 118 | 97.0 | unsupported | 691 |
| [`hrnet-w32-384`](../models/pose.md#hrnet) | 251 | 205 | – | 881 |
| [`hrnet-w48`](../models/pose.md#hrnet) | 243 | 193 | – | 746 |
| [`hrnet-w48-384`](../models/pose.md#hrnet) | 488 | 416 | – | 894 |
| [`retinaface-mobilenet0.25`](../models/pose.md#retinaface) | 22.7 | 15.8 | – | 497 |
| [`retinaface-resnet50`](../models/pose.md#retinaface) | 173 | 133 | unsupported | 936 |
| [`rf-detr-keypoint-preview`](../models/pose.md#rfdetr_keypoint) | 76.4 | 65.6 | unsupported | 869 |
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
| [`vitpose-base`](../models/pose.md#vitpose) | 176 | 159 | – | 740 |
| [`vitpose-base-simple`](../models/pose.md#vitpose) | 170 | 155 | unsupported | 733 |
| [`vitpose-plus-base`](../models/pose.md#vitpose) | 176 | 160 | – | 749 |
| [`vitpose-plus-huge`](../models/pose.md#vitpose) | 1017 | 895 | unsupported | 1858 |
| [`vitpose-plus-large`](../models/pose.md#vitpose) | 510 | 460 | unsupported | 1184 |
| [`vitpose-plus-small`](../models/pose.md#vitpose) | 85.3 | 74.4 | – | 601 |
| [`yunet-n`](../models/pose.md#yunet) | 18.8 | 15.6 | unsupported | 579 |
| [`yunet-s`](../models/pose.md#yunet) | 17.3 | 13.8 | – | 550 |

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
| [`clip-b16-laion2b`](../models/features.md#clip) | 16.7 | 14.6 | unsupported | – | – | 524 | – |
| [`clip-b16-openai`](../models/features.md#clip) | 16.7 | 14.4 | unsupported | – | – | 527 | – |
| [`clip-h14-378-dfn5b`](../models/features.md#clip) | 128 | 108 | unsupported | – | – | 1661 | – |
| [`clip-h14-dfn5b`](../models/features.md#clip) | 128 | 108 | unsupported | – | – | 1657 | – |
| [`clip-h14-laion2b`](../models/features.md#clip) | 128 | 108 | unsupported | – | – | 1658 | – |
| [`clip-l14-336-openai`](../models/features.md#clip) | 64.5 | 55.5 | unsupported | – | – | 984 | – |
| [`clip-l14-datacompxl`](../models/features.md#clip) | 64.5 | 55.5 | unsupported | – | – | 981 | – |
| [`clip-l14-dfn2b`](../models/features.md#clip) | 64.3 | 57.4 | unsupported | – | – | 987 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | 64.7 | 55.7 | unsupported | – | – | 982 | – |
| [`clip-l14-openai`](../models/features.md#clip) | 64.4 | 55.6 | unsupported | – | – | 982 | – |
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
| [`pe-core-b16-224`](../models/features.md#perception_encoder) | 17.6 | 15.2 | unsupported | – | – | 607 | – |
| [`pe-core-g14-448`](../models/features.md#perception_encoder) | 293 | 250 | unsupported | – | – | 4082 | – |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | 66.9 | 57.3 | unsupported | – | – | 1080 | – |
| [`pe-core-s16-384`](../models/features.md#perception_encoder) | 7.57 | 6.67 | unsupported | – | – | 410 | – |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | 4.11 | 3.85 | unsupported | – | – | 343 | – |
| [`pe-spatial-b16-512`](../models/features.md#perception_encoder) | 17.1 | 14.8 | unsupported | – | – | 599 | – |
| [`pe-spatial-g14-448`](../models/features.md#perception_encoder) | 292 | 249 | unsupported | – | – | 4030 | – |
| [`pe-spatial-l14-448`](../models/features.md#perception_encoder) | 66.3 | 56.6 | unsupported | – | – | 1048 | – |
| [`pe-spatial-s16-512`](../models/features.md#perception_encoder) | 7.34 | 6.42 | unsupported | – | – | 408 | – |
| [`pe-spatial-t16-512`](../models/features.md#perception_encoder) | 4.20 | 4.00 | unsupported | – | – | 343 | – |
| [`siglip2-b16-256`](../models/features.md#siglip2) | 16.8 | 14.9 | unsupported | – | – | 537 | – |
| [`siglip2-b16-512`](../models/features.md#siglip2) | 16.8 | 14.5 | unsupported | – | – | 541 | – |
| [`siglip2-g16-256`](../models/features.md#siglip2) | 185 | 155 | unsupported | – | – | 2677 | – |
| [`siglip2-g16-384`](../models/features.md#siglip2) | 185 | 155 | unsupported | – | – | 2687 | – |
| [`siglip2-l16-256`](../models/features.md#siglip2) | 52.5 | 45.2 | unsupported | – | – | 1000 | – |
| [`siglip2-l16-512`](../models/features.md#siglip2) | 53.7 | 46.9 | unsupported | – | – | 1024 | – |
| [`siglip2-so400m14-224`](../models/features.md#siglip2) | 74.0 | 63.9 | unsupported | – | – | 1246 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | 74.2 | 64.6 | unsupported | – | – | 1251 | – |
| [`siglip2-so400m16-256`](../models/features.md#siglip2) | 73.1 | 64.1 | unsupported | – | – | 1266 | – |
| [`siglip2-so400m16-512`](../models/features.md#siglip2) | 73.3 | 63.9 | unsupported | – | – | 1275 | – |
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
| [`flowformer-small-chairs`](../models/flow.md#flowformer) | 16916 | 11994 | unsupported | 1894 |
| [`flowformer-small-sintel`](../models/flow.md#flowformer) | 16914 | 12003 | unsupported | 1926 |
| [`flowformer-small-things`](../models/flow.md#flowformer) | 16907 | 12001 | unsupported | 1948 |
| [`flowseek-t`](../models/flow.md#flowseek) | 1449 | 1280 | unsupported | 3777 |
| [`flowseek-t-things`](../models/flow.md#flowseek) | 1461 | 1280 | unsupported | 3793 |
| [`flowseek-t-things-notartan`](../models/flow.md#flowseek) | 1451 | 1281 | unsupported | 3771 |
| [`gmflow-scale1-mixdata`](../models/flow.md#gmflow) | 2494 | 1827 | unsupported | 1191 |
| [`gmflow-scale1-things`](../models/flow.md#gmflow) | 2387 | 1690 | unsupported | 1150 |
| [`gmflow-scale2-mixdata`](../models/flow.md#gmflow) | 5350 | 3600 | unsupported | 2843 |
| [`gmflow-scale2-regrefine6-kitti`](../models/flow.md#gmflow) | 6554 | 4596 | unsupported | 2846 |
| [`gmflow-scale2-regrefine6-mixdata`](../models/flow.md#gmflow) | 6569 | 4563 | unsupported | 2841 |
| [`gmflow-scale2-regrefine6-sintelft`](../models/flow.md#gmflow) | 6541 | 4569 | unsupported | 2846 |
| [`gmflow-scale2-regrefine6-things`](../models/flow.md#gmflow) | 6656 | 4616 | unsupported | 2842 |
| [`gmflow-scale2-sintel`](../models/flow.md#gmflow) | 5201 | 3509 | unsupported | 2839 |
| [`gmflow-scale2-things`](../models/flow.md#gmflow) | 5250 | 3540 | unsupported | 2841 |
| [`megaflow`](../models/flow.md#megaflow) | 7161 | 5841 | – | 4129 |
| [`memflow`](../models/flow.md#memflow) | 8103 | 5579 | unsupported | 3649 |
| [`memflow-kitti`](../models/flow.md#memflow) | 8082 | 5576 | unsupported | 3647 |
| [`memflow-sintel`](../models/flow.md#memflow) | 8024 | 5606 | unsupported | 3634 |
| [`memfof`](../models/flow.md#memfof) | 3554 | 2983 | unsupported | 1944 |
| [`memfof-kitti`](../models/flow.md#memfof) | 18468 | 14923 | unsupported | 3430 |
| [`memfof-sintel`](../models/flow.md#memfof) | 18650 | 14816 | unsupported | 3430 |
| [`memfof-spring`](../models/flow.md#memfof) | 3624 | 2983 | unsupported | 1957 |
| [`neuflow-v2`](../models/flow.md#neuflow_v2) | 441 | 413 | unsupported | 1900 |
| [`neuflow-v2-sintel`](../models/flow.md#neuflow_v2) | 448 | 411 | unsupported | 1923 |
| [`neuflow-v2-things`](../models/flow.md#neuflow_v2) | 441 | 411 | unsupported | 1938 |
| [`raft-large`](../models/flow.md#raft) | 1348 | 916 | unsupported | 1829 |
| [`raft-large-kitti`](../models/flow.md#raft) | 1358 | 913 | unsupported | 1832 |
| [`raft-large-things`](../models/flow.md#raft) | 1344 | 916 | unsupported | 1828 |
| [`raft-small`](../models/flow.md#raft) | 403 | 294 | unsupported | 1092 |
| [`recover-cx`](../models/flow.md#recover) | 957 | 721 | unsupported | 1229 |
| [`recover-mn`](../models/flow.md#recover) | 335 | 259 | unsupported | 867 |
| [`recover-rn`](../models/flow.md#recover) | 616 | 411 | unsupported | 1034 |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 1296 | 928 | unsupported | 2002 |
| [`sea-raft-m-kitti`](../models/flow.md#sea_raft) | 1297 | 928 | unsupported | 2146 |
| [`sea-raft-m-spring`](../models/flow.md#sea_raft) | 319 | 265 | unsupported | 1023 |
| [`sea-raft-m-things`](../models/flow.md#sea_raft) | 1296 | 928 | unsupported | 2125 |
| [`sea-raft-s`](../models/flow.md#sea_raft) | 832 | 617 | unsupported | 2019 |
| [`sea-raft-s-kitti`](../models/flow.md#sea_raft) | 828 | 621 | unsupported | 2021 |
| [`sea-raft-s-spring`](../models/flow.md#sea_raft) | 218 | 167 | unsupported | 985 |
| [`waft-a1`](../models/flow.md#waft) | 7307 | 5377 | unsupported | 2217 |
| [`waft-a1-kitti`](../models/flow.md#waft) | 7256 | 5364 | unsupported | 2366 |
| [`waft-a1-sintel`](../models/flow.md#waft) | 7278 | 5366 | unsupported | 2255 |

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
| [`loftr-indoor`](../models/matching.md#loftr) | 773 | 485 | unsupported | – | – | 1663 | – |
| [`loftr-outdoor`](../models/matching.md#loftr) | 728 | 478 | unsupported | – | – | 1662 | – |
| [`loma-b`](../models/matching.md#loma) | 3246 | 2375 | unsupported | – | – | 2819 | – |
| [`loma-b128`](../models/matching.md#loma) | 1446 | 990 | unsupported | – | – | 1409 | – |
| [`loma-b64`](../models/matching.md#loma) | 1204 | 830 | unsupported | – | – | 1330 | – |
| [`loma-g`](../models/matching.md#loma) | 4322 | 3201 | unsupported | – | – | 2984 | – |
| [`loma-l`](../models/matching.md#loma) | 3944 | 2793 | unsupported | – | – | 2820 | – |
| [`loma-r`](../models/matching.md#loma) | 3327 | 2401 | unsupported | – | – | 2818 | – |
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
| [`big-lama`](../models/restoration.md#lama) | 247 | 193 | – | unsupported | – | – | 916 | – |
| [`codeformer`](../models/restoration.md#codeformer) | 484 | 285 | – | 481 | 239 | 1.2x | 1597 | 1637 |
| [`dat-2-x4`](../models/restoration.md#dat) | 8587 | 5231 | – | unsupported | – | – | 3643 | – |
| [`dat-light-x4`](../models/restoration.md#dat) | 1761 | 1031 | – | unsupported | – | – | 2751 | – |
| [`dat-x4`](../models/restoration.md#dat) | 10244 | 5785 | – | unsupported | – | – | 3572 | – |
| [`drct-l-x4`](../models/restoration.md#drct) | 12178 | 8199 | – | unsupported | – | – | 2349 | – |
| [`drct-x4`](../models/restoration.md#drct) | 6303 | 4229 | – | unsupported | – | – | 2306 | – |
| [`gfpgan-v1.4`](../models/restoration.md#gfpgan) | 258 | 159 | – | unsupported | – | – | 1108 | – |
| [`hat-s-x4`](../models/restoration.md#hat) | 8777 | 6491 | – | unsupported | – | – | 3108 | – |
| [`hat-x4`](../models/restoration.md#hat) | 11091 | 7478 | – | unsupported | – | – | 3470 | – |
| [`nafnet-gopro`](../models/restoration.md#nafnet) | 284 | 202 | – | unsupported | – | – | 1295 | – |
| [`nafnet-sidd`](../models/restoration.md#nafnet) | 260 | 187 | – | unsupported | – | – | 1252 | – |
| [`real-hat-gan-x4`](../models/restoration.md#hat) | 11457 | 7445 | – | unsupported | – | – | 3416 | – |
| [`realesr-general-x4v3`](../models/restoration.md#compact) | 236 | 195 | – | 165 | 96.3 | 2.0x | 516 | 968 |
| [`realesrgan-x2plus`](../models/restoration.md#realesrgan) | 1488 | 1057 | – | 1506 | 567 | 1.9x | 628 | 1130 |
| [`realesrgan-x4plus`](../models/restoration.md#realesrgan) | 5840 | 4195 | – | 5184 | 2047 | 2.0x | 1248 | 2337 |
| [`realesrgan-x4plus-anime-6b`](../models/restoration.md#realesrgan) | 1770 | 1304 | – | 1743 | 790 | 1.7x | 1187 | 1922 |
| [`realplksr-nomoswebphoto-x4`](../models/restoration.md#realplksr) | 2269 | 904 | – | 2150 | 904 | 1.0x | 1850 | 1849 |
| [`restormer-deblur`](../models/restoration.md#restormer) | 812 | 555 | – | unsupported | – | – | 2017 | – |
| [`restormer-derain`](../models/restoration.md#restormer) | 792 | 547 | – | unsupported | – | – | 2198 | – |
| [`restormer-sidd`](../models/restoration.md#restormer) | 789 | 541 | – | unsupported | – | – | 2218 | – |
| [`seedvr2-3b`](../models/restoration.md#seedvr2) | – | 14023 | 14307 | – | – | – | 5178 | – |
| [`seedvr2-7b`](../models/restoration.md#seedvr2) | – | 32426 | 25365 | – | – | – | 6547 | – |
| [`span-x2`](../models/restoration.md#span) | 123 | 103 | – | 117 | 62.7 | 1.6x | 640 | 830 |
| [`span-x4`](../models/restoration.md#span) | 132 | 110 | – | 125 | 67.3 | 1.6x | 746 | 927 |
| [`swin2sr-x2`](../models/restoration.md#swin2sr) | 5058 | 3730 | – | unsupported | – | – | 1524 | – |
| [`swin2sr-x4`](../models/restoration.md#swin2sr) | 5239 | 3966 | – | unsupported | – | – | 1677 | – |
| [`swin2sr-x4-real`](../models/restoration.md#swin2sr) | 5447 | 3883 | – | unsupported | – | – | 1597 | – |
| [`swinir-l-x4-real`](../models/restoration.md#swinir) | 9984 | 6640 | – | unsupported | – | – | 1521 | – |
| [`swinir-m-x2`](../models/restoration.md#swinir) | 4380 | 3118 | – | unsupported | – | – | 1523 | – |
| [`swinir-m-x4`](../models/restoration.md#swinir) | 5195 | 3492 | – | unsupported | – | – | 1687 | – |
| [`swinir-m-x4-real`](../models/restoration.md#swinir) | 4796 | 3360 | – | unsupported | – | – | 1478 | – |

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
