# Weights inventory and fetch, 2026-11-15/16

Command: `python results/raw/2026-11-15_weights_fetch/inventory.py` (registry.cards over all tasks, files.present per FileSpec), commit base origin/work/ports 6e48c99e. Downloads: library's own sha256-verified downloader (`python -m fastforks download <ref>`, FASTFORKS_NONCOMMERCIAL=1 for NC), one process at a time in `systemd-run --user -p MemoryMax=8G` units (ffweights, ffweights2). Raw: `results/raw/2026-11-15_weights_fetch/` (fetch.sh, fetch.log, inventory_before.json, missing_plan.json, moved_back.log).

## Result
- 634 distinct weight files (sha256) in all cards, 358 GB. Before this run 36 files / 68.1 GB were not cached (listed below, "cached before = no"); now **0 missing, 0 GB left**.
- Downloaded: 68.1 GB (rmbg-2.0, 0.9 GB, taken from the user's own gated copy in data/gated_models after sha256 check, hardlinked; the rest via the downloader).
- Dead URLs: none. Gated, not bypassed: DINOv3 and similar gated HF models are not in the missing list (their weights come from the user's gated copy / HF cache already present).
- All weights sit on the SSD (`~/.cache/fastforks/blobs/sha256`, no symlinks to the HDD; user decision 2026-11-15: weights stay fast to read).

## Decisions / problems
- pe-spatial-g14-448: first download failed the sha256 check. The card was verified correct (HF API revision f2cea21, sha db69c2e4..., 7407615928 B), so the download was corrupt; retried with the same command, ok.
- Space: weights first went partly to the HDD (7 files 46 GB, pe-core 7.5 GB, seedvr2-7b 16.5 GB) while / was at 16-23 GB. After the user's instruction (weights stay on SSD) other data was moved to the HDD instead (012 lf_cache, 190 GB, via an rsync that skipped the symlinks already pointing to the HDD copy), then all 9 weight files were copied back, cmp-verified, symlinks replaced (moved_back.log). / is now 190 GB free.
- 5 refs have a weights file listed as nnUNet `plans.json`/`dataset.json` of 0 MB: size is below 1 MB, shown as 0.

## Rows skipped or blocked for disk / weights: unblocked now?
Scope: the 31 refs whose weights were uncached before this run or were named in results/2026-11-15_gaps_remaining.md, gaps_solved.md, measurement_gaps.md, original_gaps.md, weights_not_cached.txt. "Row missing" is checked against docs/data/*.jsonl of origin/work/ports (portable, kernels, upstream, compile, tensorrt; upstream-default is only expected for refs with an upstream-default tier, so it is not counted). Nothing was measured by this task.

Unblocked now (weights cached) = yes for all rows below.

| ref | unblocked now | speed/memory rows still missing in docs/data | accuracy |
|---|---|---|---|
| bridgedepth-rvc-pretrain | yes | none | 12 eval rows |
| clip-h14-378-dfn5b | yes | none | 4 eval rows |
| clip-h14-dfn5b | yes | none | 4 eval rows |
| clip-l14-dfn2b | yes | none | 4 eval rows |
| da3-giant-1.1 | yes | none | 20 eval rows |
| da3nested-giant-large-1.1 | yes | none | 16 eval rows |
| depth-pro | yes | none | 16 eval rows |
| foundationstereo-vit-l | yes | none | 10 eval rows |
| megaflow | yes | none | 4 eval rows |
| monster-plusplus-mix-all | yes | none | 10 eval rows |
| monster-sceneflow | yes | none | 10 eval rows |
| nnfoundation-cnn | yes | none | no eval row |
| nnfoundation-vit | yes | none | no eval row |
| nninteractive | yes | none | no eval row |
| pe-core-g14-448 | yes | none | no eval row |
| pe-spatial-g14-448 | yes | none | 4 eval rows |
| pi3 | yes | none | 4 eval rows |
| pointdit-b-512 | yes | none | 16 eval rows |
| pointdit-l-512 | yes | compile, tensorrt | 16 eval rows |
| prompt-depth-anything-l | yes | none | no eval row |
| resplat-base-re10k | yes | kernels, upstream, compile, tensorrt | no eval row |
| rmbg-2.0 | yes | none | 4 eval rows |
| s2m2-m | yes | none | 12 eval rows |
| seedvr2-3b | yes | none | no eval row |
| seedvr2-7b | yes | none | no eval row |
| span-x2 | yes | none | 4 eval rows |
| span-x4 | yes | none | 4 eval rows |
| totalsegmentator-total | yes | none | 3 eval rows |
| wd-eva02-large-tagger-v3 | yes | none | no eval row |
| wd-vit-large-tagger-v3 | yes | none | no eval row |
| yolox-x | yes | none | 5 eval rows |

Notes for the next worker:
- Still missing speed rows: pointdit-l-512 (compile, tensorrt: weights were not cached), resplat-base-re10k (kernels, upstream, compile, tensorrt; weights public and cached, portable only timed out earlier; kernels needs the gsraster sm_120 fatbin).
- No accuracy row in docs/data: nnfoundation-cnn/-vit, nninteractive, pe-core-g14-448, prompt-depth-anything-l, resplat-base-re10k, seedvr2-3b/-7b, wd-eva02/vit-large-tagger-v3 (weights now present; check whether an eval is defined for the task before queuing; seedvr2-7b portable is OOM on 16 GB, recorded).
- Disk-full (Errno 122) retries named in gaps_solved.md (stablenormal-turbo, defom-stereo-vitl-sceneflow, dinov2-vitg14 TensorRT) and romav1 compile: rows exist now in docs/data (portable, kernels, compile, tensorrt present); only re-measure if they were recorded as errors.
- Not blocked by weights (still open for other reasons, see gaps_remaining.md): video-depth-anything-metric-s and tapnext-pp compile/TensorRT, memfof-sintel 1080x1920 OOM, c-foundationstereo-s Dynamo guard failure.

## All weight files
| ref | file | size MB | cached before | cached now | gated | note |
|---|---|---|---|---|---|---|
| aanet-plus-sceneflow@1 | aanet+_sceneflow.pth | 34.3 | yes | yes |  |  |
| aanet-sceneflow@1 | aanet_sceneflow.pth | 16.3 | yes | yes |  |  |
| alexnet.tv_in1k@1 | alexnet-owt-7be5be79.pth | 244.4 | yes | yes |  |  |
| aliked-flexglue@1 | aliked-n16.pth | 2.7 | yes | yes |  |  |
| aliked-flexglue@1 | flexglue.pth | 567.3 | yes | yes |  |  |
| aliked-lightglue@1 | aliked_lightglue.pth | 47.6 | yes | yes |  |  |
| alltracker-tiny@1 | alltracker_tiny.pth | 25.2 | yes | yes |  |  |
| alltracker@1 | alltracker.pth | 66.0 | yes | yes |  |  |
| any2full-b@1 | Any2Full_vitb.pth.tar | 801.2 | yes | yes |  |  |
| any2full-l@1 | Any2Full_vitl.pth.tar | 2069.0 | yes | yes |  |  |
| any2full-s@1 | Any2Full_vits.pth.tar | 204.1 | yes | yes |  |  |
| anycalib-pinhole@1 | anycalib_pinhole.pt | 1281.2 | yes | yes |  |  |
| anyline@1 | MTEED.pth | 0.2 | yes | yes |  |  |
| anysplat@1 | model.safetensors | 2944.9 | yes | yes |  |  |
| ben2@1 | ben2.safetensors | 380.6 | yes | yes |  |  |
| big-lama@1 | big-lama.ckpt | 381.4 | yes | yes |  |  |
| birefnet-lite@1 | birefnet_lite.safetensors | 177.6 | yes | yes |  |  |
| birefnet@1 | birefnet.safetensors | 444.5 | yes | yes |  |  |
| bisenet-face-parsing@1 | parsing_bisenet.pth | 53.3 | yes | yes |  |  |
| bootstapir@1 | bootstapir_checkpoint_v2.pt | 218.9 | yes | yes |  |  |
| boq-dinov2@1 | dinov2_12288.pth | 380.9 | yes | yes |  |  |
| bridgedepth-rvc-pretrain@1 | bridge_rvc_pretrain.pth | 1440.0 | yes | yes |  | nc |
| bridgedepth-sf@1 | bridge_sf.pth | 1386.1 | yes | yes |  | nc |
| c-fast-foundationstereo@1 | model_best_bp2_serialize.pth | 71.1 | yes | yes |  |  |
| c-foundationstereo-s@1 | trainable_foundation_stereo_small.pth | 537.8 | yes | yes |  |  |
| c-radio-v3-b@1 | c-radio-v3-b.safetensors | 393.0 | yes | yes |  |  |
| c-radio-v3-g@1 | c-radio-v3-g.safetensors | 4638.5 | yes | yes |  |  |
| c-radio-v3-h@1 | c-radio-v3-h.safetensors | 2606.6 | yes | yes |  |  |
| c-radio-v3-l@1 | c-radio-v3-l.safetensors | 1279.8 | yes | yes |  |  |
| c-radio-v4-h@1 | c-radio-v4-h.safetensors | 2606.6 | yes | yes |  |  |
| c-radio-v4-so400m@1 | c-radio-v4-so400m.safetensors | 1725.0 | yes | yes |  |  |
| cas-gwcnet-c-sceneflow@1 | casgwcnet-c-2.ckpt | 126.4 | yes | yes |  |  |
| cas-gwcnet-gc-sceneflow@1 | casgwcnet.ckpt | 131.7 | yes | yes |  |  |
| cfnet-sceneflow@1 | sceneflow_pretraining.ckpt | 277.0 | yes | yes |  |  |
| clip-b16-laion2b@1 | vit_base_patch16_clip_224.laion2b.safetensors | 344.8 | yes | yes |  |  |
| clip-b16-openai@1 | vit_base_patch16_clip_224.openai.safetensors | 598.5 | yes | yes |  |  |
| clip-h14-378-dfn5b@1 | vit_huge_patch14_clip_378.dfn5b.safetensors | 2530.8 | yes | yes |  | nc |
| clip-h14-dfn5b@1 | vit_huge_patch14_clip_224.dfn5b.safetensors | 2528.3 | yes | yes |  | nc |
| clip-h14-laion2b@1 | vit_huge_patch14_clip_224.laion2b.safetensors | 2528.3 | yes | yes |  |  |
| clip-l14-336-openai@1 | vit_large_patch14_clip_336.openai.safetensors | 1217.2 | yes | yes |  |  |
| clip-l14-datacompxl@1 | vit_large_patch14_clip_224.datacompxl.safetensors | 1215.9 | yes | yes |  |  |
| clip-l14-dfn2b@1 | vit_large_patch14_clip_224.dfn2b.safetensors | 1215.9 | yes | yes |  | nc |
| clip-l14-laion2b@1 | vit_large_patch14_clip_224.laion2b.safetensors | 1215.9 | yes | yes |  |  |
| clip-l14-openai@1 | vit_large_patch14_clip_224.openai.safetensors | 1710.5 | yes | yes |  |  |
| clipseg@1 | clip-bpe.merges.txt | 0.5 | yes | yes |  |  |
| clipseg@1 | clip-bpe.vocab.json | 1.1 | yes | yes |  |  |
| clipseg@1 | clipseg-rd64-refined.config.json | 0.0 | yes | yes |  |  |
| clipseg@1 | clipseg-rd64-refined.safetensors | 603.0 | yes | yes |  |  |
| coatnet_0_rw_224.sw_in1k@1 | coatnet_0_rw_224.sw_in1k.safetensors | 109.8 | yes | yes |  |  |
| coatnet_1_rw_224.sw_in1k@1 | coatnet_1_rw_224.sw_in1k.safetensors | 167.0 | yes | yes |  |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k@1 | coatnet_2_rw_224.sw_in12k_ft_in1k.safetensors | 295.6 | yes | yes |  |  |
| coatnet_nano_rw_224.sw_in1k@1 | coatnet_nano_rw_224.sw_in1k.safetensors | 60.6 | yes | yes |  |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k@1 | coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k.safetensors | 167.0 | yes | yes |  |  |
| coatnet_rmlp_1_rw_224.sw_in1k@1 | coatnet_rmlp_1_rw_224.sw_in1k.safetensors | 166.9 | yes | yes |  |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k@1 | coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k.safetensors | 295.7 | yes | yes |  |  |
| coatnet_rmlp_2_rw_224.sw_in1k@1 | coatnet_rmlp_2_rw_224.sw_in1k.safetensors | 295.7 | yes | yes |  |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k@1 | coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.safetensors | 295.7 | yes | yes |  |  |
| coatnet_rmlp_nano_rw_224.sw_in1k@1 | coatnet_rmlp_nano_rw_224.sw_in1k.safetensors | 60.7 | yes | yes |  |  |
| codeformer@1 | codeformer.pth | 376.6 | yes | yes |  | nc |
| coex-sceneflow@1 | sceneflow.ckpt | 33.2 | yes | yes |  |  |
| convnext_base.fb_in1k@1 | convnext_base.fb_in1k.safetensors | 354.4 | yes | yes |  |  |
| convnext_base.fb_in22k@1 | convnext_base.fb_in22k.safetensors | 439.8 | yes | yes |  |  |
| convnext_base.fb_in22k_ft_in1k@1 | convnext_base.fb_in22k_ft_in1k.safetensors | 354.4 | yes | yes |  |  |
| convnext_base.fb_in22k_ft_in1k_384@1 | convnext_base.fb_in22k_ft_in1k_384.safetensors | 354.4 | yes | yes |  |  |
| convnext_large.fb_in1k@1 | convnext_large.fb_in1k.safetensors | 791.1 | yes | yes |  |  |
| convnext_large.fb_in22k@1 | convnext_large.fb_in22k.safetensors | 919.2 | yes | yes |  |  |
| convnext_large.fb_in22k_ft_in1k@1 | convnext_large.fb_in22k_ft_in1k.safetensors | 791.1 | yes | yes |  |  |
| convnext_large.fb_in22k_ft_in1k_384@1 | convnext_large.fb_in22k_ft_in1k_384.safetensors | 791.1 | yes | yes |  |  |
| convnext_small.fb_in1k@1 | convnext_small.fb_in1k.safetensors | 200.9 | yes | yes |  |  |
| convnext_small.fb_in22k@1 | convnext_small.fb_in22k.safetensors | 265.0 | yes | yes |  |  |
| convnext_small.fb_in22k_ft_in1k@1 | convnext_small.fb_in22k_ft_in1k.safetensors | 200.9 | yes | yes |  |  |
| convnext_small.fb_in22k_ft_in1k_384@1 | convnext_small.fb_in22k_ft_in1k_384.safetensors | 200.9 | yes | yes |  |  |
| convnext_tiny.fb_in1k@1 | convnext_tiny.fb_in1k.safetensors | 114.4 | yes | yes |  |  |
| convnext_tiny.fb_in22k@1 | convnext_tiny.fb_in22k.safetensors | 178.5 | yes | yes |  |  |
| convnext_tiny.fb_in22k_ft_in1k@1 | convnext_tiny.fb_in22k_ft_in1k.safetensors | 114.4 | yes | yes |  |  |
| convnext_tiny.fb_in22k_ft_in1k_384@1 | convnext_tiny.fb_in22k_ft_in1k_384.safetensors | 114.4 | yes | yes |  |  |
| convnextv2_base.fcmae_ft_in22k_in1k@1 | convnextv2_base.fcmae_ft_in22k_in1k.safetensors | 354.9 | yes | yes |  | nc |
| cotracker3-offline@1 | scaled_offline.pth | 101.9 | yes | yes |  | nc |
| cotracker3-online@1 | scaled_online.pth | 101.7 | yes | yes |  | nc |
| crestereo-eth3d@1 | crestereo_eth3d.mge | 21.7 | yes | yes |  |  |
| da3-giant-1.1@1 | DA3-GIANT-1.1/model.safetensors | 5422.8 | no | yes |  | nc |
| da3-large-1.1@1 | DA3-LARGE-1.1/model.safetensors | 1643.8 | yes | yes |  | nc |
| da3metric-large@1 | DA3METRIC-LARGE/model.safetensors | 1336.7 | yes | yes |  |  |
| da3mono-large@1 | DA3MONO-LARGE/model.safetensors | 1336.7 | yes | yes |  |  |
| da3nested-giant-large-1.1@1 | DA3NESTED-GIANT-LARGE-1.1/model.safetensors | 6759.6 | no | yes |  | nc |
| dat-2-x4@1 | DAT_2_x4.pth | 140.3 | yes | yes |  |  |
| dat-light-x4@1 | DAT_light_x4.pth | 45.8 | yes | yes |  |  |
| dat-x4@1 | DAT_x4.pth | 154.7 | yes | yes |  |  |
| davit_base.msft_in1k@1 | davit_base.msft_in1k.safetensors | 351.9 | yes | yes |  |  |
| davit_small.msft_in1k@1 | davit_small.msft_in1k.safetensors | 199.0 | yes | yes |  |  |
| davit_tiny.msft_in1k@1 | davit_tiny.msft_in1k.safetensors | 113.5 | yes | yes |  |  |
| dedode-v2-b@1 | dedode_descriptor_B.pth | 54.3 | yes | yes |  |  |
| dedode-v2-b@1 | dedode_detector_L_v2.pth | 58.5 | yes | yes |  |  |
| dedode-v2-g@1 | dedode_descriptor_G.pth | 75.5 | yes | yes |  |  |
| deeplabv3-r101-d8-ade20k@1 | deeplabv3-r101-d8-ade20k.pth | 349.4 | yes | yes |  |  |
| deeplabv3plus-r101-d8-ade20k@1 | deeplabv3plus-r101-d8-ade20k.pth | 251.4 | yes | yes |  |  |
| defom-stereo-vitl-sceneflow@1 | defomstereo_vitl_sceneflow.pth | 1530.9 | yes | yes |  | nc |
| deim-l-coco@1 | deim_dfine_hgnetv2_l_coco.pth | 126.1 | yes | yes |  |  |
| deim-m-coco@1 | deim_dfine_hgnetv2_m_coco.pth | 79.1 | yes | yes |  |  |
| deim-n-coco@1 | deim_dfine_hgnetv2_n_coco.pth | 15.5 | yes | yes |  |  |
| deim-s-coco@1 | deim_dfine_hgnetv2_s_coco.pth | 41.9 | yes | yes |  |  |
| deim-x-coco@1 | deim_dfine_hgnetv2_x_coco.pth | 252.2 | yes | yes |  |  |
| deit_base_patch16_224.fb_in1k@1 | deit_base_patch16_224.fb_in1k.safetensors | 346.3 | yes | yes |  |  |
| densenet121.tv_in1k@1 | densenet121-a639ec97.pth | 32.3 | yes | yes |  |  |
| depth-anything-v2-b@1 | depth_anything_v2_vitb.pth | 390.0 | yes | yes |  | nc |
| depth-anything-v2-l@1 | depth_anything_v2_vitl.pth | 1341.4 | yes | yes |  | nc |
| depth-anything-v2-metric-hypersim-b@1 | depth_anything_v2_metric_hypersim_vitb.pth | 390.0 | yes | yes |  |  |
| depth-anything-v2-metric-hypersim-l@1 | depth_anything_v2_metric_hypersim_vitl.pth | 1341.4 | yes | yes |  |  |
| depth-anything-v2-metric-hypersim-s@1 | depth_anything_v2_metric_hypersim_vits.pth | 99.2 | yes | yes |  |  |
| depth-anything-v2-metric-vkitti-b@1 | depth_anything_v2_metric_vkitti_vitb.pth | 390.0 | yes | yes |  |  |
| depth-anything-v2-metric-vkitti-l@1 | depth_anything_v2_metric_vkitti_vitl.pth | 1341.4 | yes | yes |  |  |
| depth-anything-v2-metric-vkitti-s@1 | depth_anything_v2_metric_vkitti_vits.pth | 99.2 | yes | yes |  |  |
| depth-anything-v2-s@1 | depth_anything_v2_vits.pth | 99.2 | yes | yes |  |  |
| depth-pro@1 | depth_pro.pt | 1904.4 | no | yes |  | nc |
| depthfm-v1@1 | depthfm-v1.ckpt | 3464.4 | yes | yes |  |  |
| depthfm-v1@1 | vae/diffusion_pytorch_model.safetensors | 334.6 | yes | yes |  |  |
| depthsplat-small-re10k@1 | depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth | 153.7 | yes | yes |  |  |
| detr-r101-coco@1 | detr-r101-2c7b67e5.pth | 242.8 | yes | yes |  |  |
| detr-r50-coco@1 | detr-r50-e632da11.pth | 166.6 | yes | yes |  |  |
| dfine-l-coco@1 | dfine-large-coco.safetensors | 125.6 | yes | yes |  |  |
| dfine-m-coco@1 | dfine-medium-coco.safetensors | 78.7 | yes | yes |  |  |
| dfine-n-coco@1 | dfine-nano-coco.safetensors | 15.3 | yes | yes |  |  |
| dfine-s-coco@1 | dfine-small-coco.safetensors | 41.5 | yes | yes |  |  |
| dfine-x-coco@1 | dfine-xlarge-coco.safetensors | 251.6 | yes | yes |  |  |
| dinov2-b-linear-ade20k@1 | dinov2-b-linear-ade20k.backbone.pth | 346.4 | yes | yes |  |  |
| dinov2-b-linear-ade20k@1 | dinov2-b-linear-ade20k.head.pth | 1.4 | yes | yes |  |  |
| dinov2-l-linear-ade20k@1 | dinov2-l-linear-ade20k.backbone.pth | 1217.6 | yes | yes |  |  |
| dinov2-l-linear-ade20k@1 | dinov2-l-linear-ade20k.head.pth | 1.9 | yes | yes |  |  |
| dinov2-s-linear-ade20k@1 | dinov2-s-linear-ade20k.backbone.pth | 88.3 | yes | yes |  |  |
| dinov2-s-linear-ade20k@1 | dinov2-s-linear-ade20k.head.pth | 0.7 | yes | yes |  |  |
| dinov2-vitb14-reg@1 | dinov2_vitb14_reg4_pretrain.pth | 346.4 | yes | yes |  |  |
| dinov2-vitg14-reg@1 | dinov2_vitg14_reg4_pretrain.pth | 4546.1 | yes | yes |  |  |
| dinov2-vitg14@1 | dinov2_vitg14_pretrain.pth | 4546.1 | yes | yes |  |  |
| dinov2-vitl14-reg@1 | dinov2_vitl14_reg4_pretrain.pth | 1217.6 | yes | yes |  |  |
| dinov2-vits14-reg@1 | dinov2_vits14_reg4_pretrain.pth | 88.3 | yes | yes |  |  |
| dinov2_vitl14_lc@1 | dinov2_vitl14_linear_head.pth | 8.2 | yes | yes |  |  |
| dinov2_vitl14_lc@1 | vit_large_patch14_dinov2.lvd142m.safetensors | 1217.5 | yes | yes |  |  |
| dinov3-convnext-base@1 | dinov3-convnext-base-pretrain-lvd1689m/model.safetensors | 350.3 | yes | yes |  |  |
| dinov3-convnext-large@1 | dinov3-convnext-large-pretrain-lvd1689m/model.safetensors | 785.0 | yes | yes |  |  |
| dinov3-convnext-small@1 | dinov3-convnext-small-pretrain-lvd1689m/model.safetensors | 197.9 | yes | yes |  |  |
| dinov3-convnext-tiny@1 | dinov3-convnext-tiny-pretrain-lvd1689m/model.safetensors | 111.3 | yes | yes |  |  |
| dinov3-vith16plus@1 | dinov3-vith16plus-pretrain-lvd1689m/model.safetensors | 3362.4 | yes | yes |  |  |
| dinov3-vits16@1 | dinov3-vits16-pretrain-lvd1689m/model.safetensors | 86.4 | yes | yes |  |  |
| disk-lightglue@1 | depth-save.pth | 4.4 | yes | yes |  |  |
| disk-lightglue@1 | disk_lightglue.pth | 47.6 | yes | yes |  |  |
| distill-any-depth-b@1 | Distill-Any-Depth/base/model.safetensors | 389.9 | yes | yes |  |  |
| distill-any-depth-l@1 | Distill-Any-Depth/large/model.safetensors | 1341.3 | yes | yes |  |  |
| distill-any-depth-s@1 | Distill-Any-Depth/small/model.safetensors | 99.2 | yes | yes |  |  |
| dlnr-middlebury@1 | DLNR_Middlebury.pth | 229.8 | yes | yes |  |  |
| dlnr-sceneflow@1 | DLNR_SceneFlow.pth | 229.8 | yes | yes |  |  |
| drct-l-x4@1 | DRCT-L_X4.pth | 485.6 | yes | yes |  |  |
| drct-x4@1 | DRCT_X4.pth | 245.6 | yes | yes |  |  |
| dwpose-l-384@1 | dwpose-l-ucoco-384x288.pth | 134.7 | yes | yes |  |  |
| dwpose-l@1 | dwpose-l-ucoco-256x192.pth | 134.2 | yes | yes |  |  |
| dwpose-m@1 | dwpose-m-ucoco-256x192.pth | 72.0 | yes | yes |  |  |
| dwpose-s@1 | dwpose-s-ucoco-256x192.pth | 33.7 | yes | yes |  |  |
| dwpose-t@1 | dwpose-t-ucoco-256x192.pth | 22.2 | yes | yes |  |  |
| efficientformer_l1.snap_dist_in1k@1 | efficientformer_l1.snap_dist_in1k.safetensors | 49.3 | yes | yes |  |  |
| efficientformer_l3.snap_dist_in1k@1 | efficientformer_l3.snap_dist_in1k.safetensors | 126.0 | yes | yes |  |  |
| efficientformer_l7.snap_dist_in1k@1 | efficientformer_l7.snap_dist_in1k.safetensors | 329.5 | yes | yes |  |  |
| efficientformerv2_l.snap_dist_in1k@1 | efficientformerv2_l.snap_dist_in1k.safetensors | 106.2 | yes | yes |  |  |
| efficientformerv2_s0.snap_dist_in1k@1 | efficientformerv2_s0.snap_dist_in1k.safetensors | 14.7 | yes | yes |  |  |
| efficientformerv2_s1.snap_dist_in1k@1 | efficientformerv2_s1.snap_dist_in1k.safetensors | 25.1 | yes | yes |  |  |
| efficientformerv2_s2.snap_dist_in1k@1 | efficientformerv2_s2.snap_dist_in1k.safetensors | 51.4 | yes | yes |  |  |
| efficientnet_b0.tv_in1k@1 | efficientnet_b0_rwightman-7f5810bc.pth | 21.4 | yes | yes |  |  |
| efficientnet_b4.tv_in1k@1 | efficientnet_b4_rwightman-23ab8bcd.pth | 78.1 | yes | yes |  |  |
| efficientnet_v2_s.tv_in1k@1 | efficientnet_v2_s-dd5fe13b.pth | 86.7 | yes | yes |  |  |
| efficientvit-sam-l0@1 | efficientvit_sam_l0.pt | 139.4 | yes | yes |  |  |
| efficientvit-sam-l1@1 | efficientvit_sam_l1.pt | 190.9 | yes | yes |  |  |
| efficientvit-sam-l2@1 | efficientvit_sam_l2.pt | 245.7 | yes | yes |  |  |
| efficientvit-sam-xl0@1 | efficientvit_sam_xl0.pt | 468.2 | yes | yes |  |  |
| efficientvit-sam-xl1@1 | efficientvit_sam_xl1.pt | 814.0 | yes | yes |  |  |
| efficientvit_b2.r288_in1k@1 | efficientvit_b2.r288_in1k.safetensors | 97.5 | yes | yes |  |  |
| ema-vfi-small-t@1 | ema-vfi-ours_small_t.pkl | 58.5 | yes | yes |  |  |
| ema-vfi-small@1 | ema-vfi-ours_small.pkl | 58.5 | yes | yes |  |  |
| ema-vfi-t@1 | ema-vfi-ours_t.pkl | 263.8 | yes | yes |  |  |
| ema-vfi@1 | ema-vfi-ours.pkl | 263.8 | yes | yes |  |  |
| eomt-l-dinov2-ade20k@1 | eomt-l-dinov2-ade20k.config.json | 0.0 | yes | yes |  |  |
| eomt-l-dinov2-ade20k@1 | eomt-l-dinov2-ade20k.safetensors | 1264.1 | yes | yes |  |  |
| eomt-l-dinov3-ade20k@1 | eomt-l-dinov3-ade20k.config.json | 0.0 | yes | yes |  |  |
| eomt-l-dinov3-ade20k@1 | eomt-l-dinov3-ade20k.safetensors | 1259.8 | yes | yes |  |  |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k@1 | eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.safetensors | 1220.4 | yes | yes |  |  |
| falconsai-nsfw-vit@1 | falconsai-nsfw-vit.safetensors | 343.2 | yes | yes |  |  |
| fast-acvnet-generalization@1 | Fast-ACVNet/generalization.ckpt | 12.6 | yes | yes |  |  |
| fast-acvnet-kitti12@1 | Fast-ACVNet/kitti_2012.ckpt | 12.6 | yes | yes |  |  |
| fast-acvnet-kitti15@1 | Fast-ACVNet/kitti_2015.ckpt | 12.6 | yes | yes |  |  |
| fast-acvnet-plus-generalization@1 | Fast-ACVNet+/generalization.ckpt | 13.2 | yes | yes |  |  |
| fast-acvnet-plus-kitti12@1 | Fast-ACVNet+/kitti_2012.ckpt | 13.2 | yes | yes |  |  |
| fast-acvnet-plus-kitti15@1 | Fast-ACVNet+/kitti_2015.ckpt | 13.2 | yes | yes |  |  |
| fast-acvnet-plus-sceneflow@1 | Fast-ACVNet+/sceneflow.ckpt | 13.2 | yes | yes |  |  |
| fast-acvnet-sceneflow@1 | Fast-ACVNet/sceneflow.ckpt | 12.6 | yes | yes |  |  |
| fast-foundationstereo@1 | 23-36-37/model_best_bp2_serialize.pth | 71.1 | yes | yes |  | nc |
| faster-rcnn-r50-fpn-3x-coco@1 | faster_rcnn_R_50_FPN_3x.pkl | 167.3 | yes | yes |  |  |
| fastvit_sa24.apple_dist_in1k@1 | fastvit_sa24.apple_dist_in1k.safetensors | 86.6 | yes | yes |  |  |
| fastvit_t12.apple_dist_in1k@1 | fastvit_t12.apple_dist_in1k.safetensors | 30.4 | yes | yes |  |  |
| fastvit_t8.apple_dist_in1k@1 | fastvit_t8.apple_dist_in1k.safetensors | 16.3 | yes | yes |  |  |
| fcn-r101-d8-ade20k@1 | fcn-r101-d8-ade20k.pth | 274.9 | yes | yes |  |  |
| film@1 | film-net-style-fp32.pt | 137.9 | yes | yes |  |  |
| flowformer-small-chairs@1 | flowformer-small-chairs.pth | 69.0 | yes | yes |  |  |
| flowformer-small-sintel@1 | flowformer-small-sintel.pth | 69.0 | yes | yes |  |  |
| flowformer-small-things@1 | flowformer-small-things.pth | 69.0 | yes | yes |  |  |
| flowseek-t-things-notartan@1 | flowseek_T_CT.pth | 171.0 | yes | yes |  |  |
| flowseek-t-things@1 | flowseek_T_TartanCT.pth | 171.0 | yes | yes |  |  |
| flowseek-t@1 | flowseek_T_TartanCT_TSKH.pth | 171.0 | yes | yes |  |  |
| focalnet_base_lrf.ms_in1k@1 | focalnet_base_lrf.ms_in1k.safetensors | 355.0 | yes | yes |  |  |
| focalnet_base_srf.ms_in1k@1 | focalnet_base_srf.ms_in1k.safetensors | 352.6 | yes | yes |  |  |
| focalnet_small_lrf.ms_in1k@1 | focalnet_small_lrf.ms_in1k.safetensors | 201.4 | yes | yes |  |  |
| focalnet_small_srf.ms_in1k@1 | focalnet_small_srf.ms_in1k.safetensors | 199.6 | yes | yes |  |  |
| focalnet_tiny_lrf.ms_in1k@1 | focalnet_tiny_lrf.ms_in1k.safetensors | 114.6 | yes | yes |  |  |
| focalnet_tiny_srf.ms_in1k@1 | focalnet_tiny_srf.ms_in1k.safetensors | 113.7 | yes | yes |  |  |
| foundationstereo-vit-l@1 | 23-51-11/model_best_bp2.pth | 3298.5 | yes | yes |  | nc |
| foundationstereo-vit-s@1 | 11-33-40/model_best_bp2.pth | 787.7 | yes | yes |  | nc |
| geocalib-pinhole@1 | geocalib-pinhole.tar | 116.1 | yes | yes |  |  |
| gfpgan-v1.4@1 | GFPGANv1.4.pth | 348.6 | yes | yes |  |  |
| gimm-vfi-r-p@1 | gimmvfi_r_arb_lpips.pt | 79.3 | yes | yes |  | nc |
| gimm-vfi-r@1 | gimmvfi_r_arb.pt | 79.3 | yes | yes |  | nc |
| gmflow-scale1-mixdata@1 | gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth | 18.8 | yes | yes |  |  |
| gmflow-scale1-things@1 | gmflow-scale1-things-e9887eda.pth | 18.8 | yes | yes |  |  |
| gmflow-scale2-mixdata@1 | gmflow-scale2-mixdata-train320x576-9ff1c094.pth | 18.9 | yes | yes |  |  |
| gmflow-scale2-regrefine6-kitti@1 | gmflow-scale2-regrefine6-kitti15-25b554d7.pth | 29.5 | yes | yes |  |  |
| gmflow-scale2-regrefine6-mixdata@1 | gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth | 29.5 | yes | yes |  |  |
| gmflow-scale2-regrefine6-sintelft@1 | gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth | 29.5 | yes | yes |  |  |
| gmflow-scale2-regrefine6-things@1 | gmflow-scale2-regrefine6-things-776ed612.pth | 29.5 | yes | yes |  |  |
| gmflow-scale2-sintel@1 | gmflow-scale2-sintel-3ed1cf48.pth | 18.9 | yes | yes |  |  |
| gmflow-scale2-things@1 | gmflow-scale2-things-36579974.pth | 18.9 | yes | yes |  |  |
| great-igev-eth3d@1 | great-igev-eth3d-submit.pth | 58.4 | yes | yes |  |  |
| great-igev-kitti@1 | great-igev-kitti-submit.pth | 58.4 | yes | yes |  |  |
| great-igev-middlebury@1 | great-igev-middlebury-submit.pth | 58.4 | yes | yes |  |  |
| great-igev-sceneflow@1 | great-igev-sceneflow-submit.pth | 58.4 | yes | yes |  |  |
| grounding-dino-base@1 | bert-uncased.vocab.txt | 0.2 | yes | yes |  |  |
| grounding-dino-base@1 | gdino-base.config.json | 0.0 | yes | yes |  |  |
| grounding-dino-base@1 | gdino-base.model.safetensors | 933.4 | yes | yes |  |  |
| grounding-dino-tiny@1 | gdino-tiny.config.json | 0.0 | yes | yes |  |  |
| grounding-dino-tiny@1 | gdino-tiny.model.safetensors | 689.4 | yes | yes |  |  |
| gwcnet-g-sceneflow@1 | sceneflow/gwcnet-g/checkpoint_000015.ckpt | 297.8 | yes | yes |  |  |
| gwcnet-gc-sceneflow@1 | sceneflow/gwcnet-gc/checkpoint_000015.ckpt | 297.8 | yes | yes |  |  |
| hat-s-x4@1 | HAT-S_SRx4.pth | 81.1 | yes | yes |  |  |
| hat-x4@1 | HAT_SRx4_ImageNet-pretrain.pth | 85.1 | yes | yes |  |  |
| hed@1 | ControlNetHED.pth | 29.4 | yes | yes |  |  |
| hgnetv2_b0.ssld_stage2_ft_in1k@1 | hgnetv2_b0.ssld_stage2_ft_in1k.safetensors | 24.1 | yes | yes |  |  |
| hgnetv2_b1.ssld_stage2_ft_in1k@1 | hgnetv2_b1.ssld_stage2_ft_in1k.safetensors | 25.5 | yes | yes |  |  |
| hgnetv2_b2.ssld_stage2_ft_in1k@1 | hgnetv2_b2.ssld_stage2_ft_in1k.safetensors | 45.0 | yes | yes |  |  |
| hgnetv2_b3.ssld_stage2_ft_in1k@1 | hgnetv2_b3.ssld_stage2_ft_in1k.safetensors | 65.4 | yes | yes |  |  |
| hgnetv2_b4.ssld_stage2_ft_in1k@1 | hgnetv2_b4.ssld_stage2_ft_in1k.safetensors | 79.4 | yes | yes |  |  |
| hgnetv2_b5.ssld_stage2_ft_in1k@1 | hgnetv2_b5.ssld_stage2_ft_in1k.safetensors | 158.7 | yes | yes |  |  |
| hgnetv2_b6.ssld_stage2_ft_in1k@1 | hgnetv2_b6.ssld_stage2_ft_in1k.safetensors | 301.8 | yes | yes |  |  |
| hitnet-eth3d@1 | eth3d.pb | 440.1 | yes | yes |  |  |
| hitnet-middlebury-d400@1 | middlebury_d400.pb | 440.1 | yes | yes |  |  |
| hrnet-w32-384@1 | hrnet-w32-coco-384x288.pth | 122.1 | yes | yes |  |  |
| hrnet-w32@1 | hrnet-w32-coco-256x192.pth | 121.7 | yes | yes |  |  |
| hrnet-w48-384@1 | hrnet-w48-coco-384x288.pth | 267.9 | yes | yes |  |  |
| hrnet-w48-fcn-ade20k@1 | hrnet-w48-fcn-ade20k.pth | 264.5 | yes | yes |  |  |
| hrnet-w48@1 | hrnet-w48-coco-256x192.pth | 269.2 | yes | yes |  |  |
| igev-plusplus-eth3d@1 | eth3d.pth | 58.5 | yes | yes |  |  |
| igev-plusplus-kitti12@1 | kitti2012.pth | 58.5 | yes | yes |  |  |
| igev-plusplus-kitti15@1 | kitti2015.pth | 58.5 | yes | yes |  |  |
| igev-plusplus-middlebury@1 | middlebury.pth | 58.5 | yes | yes |  |  |
| igev-plusplus-sceneflow@1 | sceneflow.pth | 58.5 | yes | yes |  |  |
| igev-stereo-eth3d@1 | eth3d.pth | 50.8 | yes | yes |  |  |
| igev-stereo-kitti12@1 | kitti12.pth | 50.8 | yes | yes |  |  |
| igev-stereo-kitti15@1 | kitti15.pth | 50.8 | yes | yes |  |  |
| igev-stereo-middlebury@1 | middlebury.pth | 50.8 | yes | yes |  |  |
| igev-stereo-sceneflow@1 | sceneflow.pth | 50.8 | yes | yes |  |  |
| inception_v3.tv_in1k@1 | inception_v3_google-0cc3c7bd.pth | 108.9 | yes | yes |  |  |
| inspyrenet-fast@1 | ckpt_fast.pth | 367.5 | yes | yes |  |  |
| inspyrenet@1 | ckpt_base.pth | 367.5 | yes | yes |  |  |
| isnet-general-use@1 | isnet-general-use.pth | 176.6 | yes | yes |  |  |
| isnet@1 | isnet.pth | 176.6 | yes | yes |  |  |
| las2-h@1 | LAS2_H.pth | 46.7 | yes | yes |  |  |
| las2-l@1 | LAS2_L.pth | 109.3 | yes | yes |  |  |
| las2-m@1 | LAS2_M.pth | 41.9 | yes | yes |  |  |
| las2-s@1 | LAS2_S.pth | 24.4 | yes | yes |  |  |
| lineart-anime@1 | netG.pth | 217.6 | yes | yes |  |  |
| lineart-coarse@1 | sk_model2.pth | 17.2 | yes | yes |  |  |
| lineart@1 | sk_model.pth | 17.2 | yes | yes |  |  |
| lingbot-depth-dc@1 | lingbot-depth-postrain-dc-vitl14/model.pt | 1284.8 | yes | yes |  |  |
| lingbot-depth@1 | lingbot-depth-pretrain-vitl-14-v0.5/model.pt | 1284.8 | yes | yes |  |  |
| lingbot-map@1 | lingbot-map.pt | 4632.3 | yes | yes |  |  |
| locotrack-b@1 | locotrack_base.ckpt | 46.1 | yes | yes |  |  |
| locotrack-s@1 | locotrack_small.ckpt | 33.0 | yes | yes |  |  |
| loftr-indoor@1 | loftr_indoor_ds_new.ckpt | 46.4 | yes | yes |  |  |
| loftr-outdoor@1 | loftr_outdoor.ckpt | 46.3 | yes | yes |  |  |
| loma-b@1 | loma_B.pt | 757.9 | yes | yes |  |  |
| lotus-normal-d-v1-1@1 | text_encoder/model.safetensors | 680.8 | yes | yes |  |  |
| lotus-normal-d-v1-1@1 | unet-d/diffusion_pytorch_model.safetensors | 3470.3 | yes | yes |  |  |
| lotus-normal-d-v1-1@1 | vae/diffusion_pytorch_model.safetensors | 167.3 | yes | yes |  |  |
| lotus-normal-g-v1-1@1 | unet-g/diffusion_pytorch_model.safetensors | 3470.4 | yes | yes |  |  |
| lw-detr-large-coco@1 | lwdetr-large.config.json | 0.0 | yes | yes |  |  |
| lw-detr-large-coco@1 | lwdetr-large.model.safetensors | 187.4 | yes | yes |  |  |
| lw-detr-medium-coco@1 | lwdetr-medium.config.json | 0.0 | yes | yes |  |  |
| lw-detr-medium-coco@1 | lwdetr-medium.model.safetensors | 113.0 | yes | yes |  |  |
| lw-detr-small-coco@1 | lwdetr-small.config.json | 0.0 | yes | yes |  |  |
| lw-detr-small-coco@1 | lwdetr-small.model.safetensors | 58.3 | yes | yes |  |  |
| lw-detr-tiny-coco@1 | lwdetr-tiny.config.json | 0.0 | yes | yes |  |  |
| lw-detr-tiny-coco@1 | lwdetr-tiny.model.safetensors | 48.3 | yes | yes |  |  |
| lw-detr-xlarge-coco@1 | lwdetr-xlarge.config.json | 0.0 | yes | yes |  |  |
| lw-detr-xlarge-coco@1 | lwdetr-xlarge.model.safetensors | 472.3 | yes | yes |  |  |
| mapanything-apache@1 | model.safetensors | 4914.1 | yes | yes |  |  |
| marigold-depth-v1-1@1 | text_encoder/model.safetensors | 1361.6 | yes | yes |  |  |
| marigold-depth-v1-1@1 | unet/diffusion_pytorch_model.safetensors | 3463.8 | yes | yes |  |  |
| marigold-depth-v1-1@1 | vae/diffusion_pytorch_model.safetensors | 334.6 | yes | yes |  |  |
| marigold-normals-lcm-v0-1@1 | text_encoder/model.safetensors | 1361.6 | yes | yes |  |  |
| marigold-normals-lcm-v0-1@1 | unet/diffusion_pytorch_model.safetensors | 3463.8 | yes | yes |  |  |
| marigold-v2-depth@1 | marigold-v2-depth-log-stage2-nf4.safetensors | 8974.3 | yes | yes |  |  |
| mask-rcnn-r50-fpn-3x-coco@1 | mask_rcnn_R_50_FPN_3x.pkl | 177.8 | yes | yes |  |  |
| mask2former-swinl-ade20k@1 | mask2former-swinl-ade20k.config.json | 0.1 | yes | yes |  | nc |
| mask2former-swinl-ade20k@1 | mask2former-swinl-ade20k.safetensors | 866.1 | yes | yes |  | nc |
| matanyone@1 | matanyone.pth | 141.4 | yes | yes |  | nc |
| megaflow@1 | megaflow-flow.safetensors | 3745.0 | yes | yes |  |  |
| megaloc@1 | megaloc.safetensors | 914.6 | yes | yes |  |  |
| memflow-kitti@1 | MemFlowNet_kitti.pth | 25.4 | yes | yes |  |  |
| memflow-sintel@1 | MemFlowNet_sintel.pth | 25.4 | yes | yes |  |  |
| memflow@1 | MemFlowNet_things.pth | 25.4 | yes | yes |  |  |
| memfof-kitti@1 | MEMFOF-Tartan-T-TSKH-kitti.safetensors | 303.2 | yes | yes |  |  |
| memfof-sintel@1 | MEMFOF-Tartan-T-TSKH-sintel.safetensors | 303.2 | yes | yes |  |  |
| memfof-spring@1 | MEMFOF-Tartan-T-TSKH-spring.safetensors | 303.2 | yes | yes |  |  |
| memfof@1 | MEMFOF-Tartan-T-TSKH.safetensors | 303.2 | yes | yes |  |  |
| metric3d-vit-large@1 | metric_depth_vit_large_800k.pth | 1648.0 | yes | yes |  |  |
| midas-dpt-beit-large-512@1 | dpt_beit_large_512.pt | 1582.0 | yes | yes |  |  |
| mlsd@1 | mlsd_large_512_fp32.pth | 6.3 | yes | yes |  |  |
| mm-grounding-dino-base@1 | mmgdino-base.config.json | 0.0 | yes | yes |  |  |
| mm-grounding-dino-base@1 | mmgdino-base.model.safetensors | 936.1 | yes | yes |  |  |
| mm-grounding-dino-tiny@1 | mmgdino-tiny.config.json | 0.0 | yes | yes |  |  |
| mm-grounding-dino-tiny@1 | mmgdino-tiny.model.safetensors | 692.0 | yes | yes |  |  |
| mobile-sam@1 | mobile_sam.pt | 40.7 | yes | yes |  |  |
| mobilenet_v2.tv2_in1k@1 | mobilenet_v2-7ebf99e0.pth | 14.3 | yes | yes |  |  |
| mobilenet_v3_large.tv2_in1k@1 | mobilenet_v3_large-5c1a4163.pth | 22.1 | yes | yes |  |  |
| mobilenetv1_100.ra4_e3600_r224_in1k@1 | mobilenetv1_100.ra4_e3600_r224_in1k.safetensors | 17.0 | yes | yes |  |  |
| mobilenetv4_conv_medium.e500_r256_in1k@1 | mobilenetv4_conv_medium.e500_r256_in1k.safetensors | 39.2 | yes | yes |  |  |
| mocha-v1-sceneflow@1 | mocha-stereo.pth | 391.0 | yes | yes |  |  |
| mocha-v2-sceneflow@1 | mocha-V2.pth | 391.3 | yes | yes |  |  |
| moge-2-vitb-normal@1 | moge-2-vitb-normal/model.pt | 419.1 | yes | yes |  |  |
| moge-2-vitl-normal@1 | moge-2-vitl-normal/model.pt | 1323.8 | yes | yes |  |  |
| moge-2-vits-normal@1 | moge-2-vits-normal/model.pt | 140.6 | yes | yes |  |  |
| moge-3-vitl@1 | moge-3-vitl/model.pt | 1481.3 | yes | yes |  |  |
| monster-mix-all@1 | mix_all.pth | 1555.1 | yes | yes |  | nc |
| monster-plusplus-mix-all@1 | Mix_all_large.pth | 1555.1 | no | yes |  | nc |
| monster-sceneflow@1 | sceneflow.pth | 1555.2 | no | yes |  | nc |
| mvsanywhere@1 | mvsanywhere_hero.ckpt | 2420.6 | yes | yes |  | nc |
| mvsplat-re10k@1 | re10k.ckpt | 48.0 | yes | yes |  |  |
| nafnet-gopro@1 | NAFNet-GoPro-width64.pth | 271.8 | yes | yes |  |  |
| nafnet-sidd@1 | NAFNet-SIDD-width64.pth | 464.2 | yes | yes |  |  |
| neuflow-v2-sintel@1 | neuflow_sintel.pth | 36.2 | yes | yes |  |  |
| neuflow-v2-things@1 | neuflow_things.pth | 36.2 | yes | yes |  |  |
| neuflow-v2@1 | neuflow_mixed.pth | 36.2 | yes | yes |  |  |
| nmrf-sceneflow@1 | sceneflow.pth | 24.6 | yes | yes |  |  |
| nnfoundation-cnn@1 | nnFoundationCNN/checkpoint_final.pth | 409.6 | no | yes |  |  |
| nnfoundation-vit@1 | nnFoundationViT/checkpoint_final.pth | 2697.6 | no | yes |  |  |
| nninteractive@1 | nnInteractive_v1.0/fold_0/checkpoint_final.pth | 411.4 | no | yes |  | nc |
| nninteractive@1 | nnInteractive_v1.0/plans.json | 0.0 | no | yes |  | nc |
| omni-dc@1 | model.safetensors | 1683.1 | yes | yes |  | nc |
| oneformer-swinl-ade20k@1 | oneformer-swinl-ade20k.bin | 879.5 | yes | yes |  |  |
| oneformer-swinl-ade20k@1 | oneformer-swinl-ade20k.config.json | 0.1 | yes | yes |  |  |
| owlv2-b16@1 | owlv2-b16.config.json | 0.0 | yes | yes |  |  |
| owlv2-b16@1 | owlv2-b16.safetensors | 619.9 | yes | yes |  |  |
| owlv2-l14@1 | owlv2-l14.config.json | 0.0 | yes | yes |  |  |
| owlv2-l14@1 | owlv2-l14.safetensors | 1750.5 | yes | yes |  |  |
| parseq-tiny@1 | parseq_tiny-e7a21b54.pt | 24.1 | yes | yes |  |  |
| parseq@1 | parseq-bb5792a6.pt | 95.4 | yes | yes |  |  |
| patchmatchnet@1 | params_000007.ckpt | 2.8 | yes | yes |  |  |
| pe-core-b16-224@1 | vit_pe_core_base_patch16_224.fb.safetensors | 374.7 | yes | yes |  |  |
| pe-core-g14-448@1 | vit_pe_core_gigantic_patch14_448.fb.safetensors | 7528.2 | no | yes |  |  |
| pe-core-l14-336@1 | vit_pe_core_large_patch14_336.fb.safetensors | 1268.6 | yes | yes |  |  |
| pe-core-s16-384@1 | vit_pe_core_small_patch16_384.fb.safetensors | 95.1 | yes | yes |  |  |
| pe-core-t16-384@1 | vit_pe_core_tiny_patch16_384.fb.safetensors | 24.6 | yes | yes |  |  |
| pe-spatial-b16-512@1 | vit_pe_spatial_base_patch16_512.fb.safetensors | 345.7 | yes | yes |  |  |
| pe-spatial-g14-448@1 | vit_pe_spatial_gigantic_patch14_448.fb.safetensors | 7407.6 | no | NO |  |  |
| pe-spatial-l14-448@1 | vit_pe_spatial_large_patch14_448.fb.safetensors | 1215.9 | yes | yes |  |  |
| pe-spatial-s16-512@1 | vit_pe_spatial_small_patch16_512.fb.safetensors | 87.9 | yes | yes |  |  |
| pe-spatial-t16-512@1 | vit_pe_spatial_tiny_patch16_512.fb.safetensors | 22.7 | yes | yes |  |  |
| pi3@1 | model.safetensors | 3834.9 | yes | yes |  | nc |
| pidinet@1 | table5_pidinet.pth | 2.9 | yes | yes |  | nc |
| pointdit-b-512@1 | dinov3-vitb16-pretrain-lvd1689m/model.safetensors | 342.7 | yes | yes |  |  |
| pointdit-b-512@1 | pointditb-512-mixdata-nodinov3-1d42aacf.pth | 1097.9 | no | yes |  |  |
| pointdit-l-512@1 | dinov3-vitl16-pretrain-lvd1689m/model.safetensors | 1212.6 | yes | yes |  |  |
| pointdit-l-512@1 | pointditl-512-mixdata-nodinov3-240c1a4f.pth | 3746.8 | no | yes |  |  |
| ppocrv5-mobile@1 | PP-OCRv5_mobile_det/inference.json | 0.2 | yes | yes |  |  |
| ppocrv5-mobile@1 | PP-OCRv5_mobile_det/inference.pdiparams | 4.7 | yes | yes |  |  |
| ppocrv5-mobile@1 | PP-OCRv5_mobile_rec/config.json | 0.4 | yes | yes |  |  |
| ppocrv5-mobile@1 | PP-OCRv5_mobile_rec/inference.json | 0.2 | yes | yes |  |  |
| ppocrv5-mobile@1 | PP-OCRv5_mobile_rec/inference.pdiparams | 16.5 | yes | yes |  |  |
| ppocrv5-server@1 | PP-OCRv5_server_det/inference.json | 0.4 | yes | yes |  |  |
| ppocrv5-server@1 | PP-OCRv5_server_det/inference.pdiparams | 87.9 | yes | yes |  |  |
| ppocrv5-server@1 | PP-OCRv5_server_rec/config.json | 0.4 | yes | yes |  |  |
| ppocrv5-server@1 | PP-OCRv5_server_rec/inference.json | 0.3 | yes | yes |  |  |
| ppocrv5-server@1 | PP-OCRv5_server_rec/inference.pdiparams | 84.4 | yes | yes |  |  |
| prompt-depth-anything-l@1 | prompt_depth_anything_vitl.ckpt | 1360.4 | yes | yes |  |  |
| prompt-depth-anything-s@1 | prompt_depth_anything_vits.ckpt | 100.4 | yes | yes |  |  |
| psmnet-sceneflow@1 | pretrained_sceneflow_new.tar | 21.1 | yes | yes |  |  |
| pspnet-r101-d8-ade20k@1 | pspnet-r101-d8-ade20k.pth | 272.9 | yes | yes |  |  |
| raft-large-kitti@1 | raft_large_C_T_SKHT_K_V2-b5c70766.pth | 21.1 | yes | yes |  |  |
| raft-large-things@1 | raft_large_C_T_V2-1bb1363a.pth | 21.1 | yes | yes |  |  |
| raft-large@1 | raft_large_C_T_SKHT_V2-ff5fadd5.pth | 21.1 | yes | yes |  |  |
| raft-small@1 | raft_small_C_T_V2-01064c6d.pth | 4.0 | yes | yes |  |  |
| raft-stereo-eth3d@1 | raftstereo-eth3d.pth | 201.8 | yes | yes |  |  |
| raft-stereo-middlebury@1 | raftstereo-middlebury.pth | 201.8 | yes | yes |  |  |
| raft-stereo-sceneflow@1 | raftstereo-sceneflow.pth | 201.8 | yes | yes |  |  |
| real-hat-gan-x4@1 | Real_HAT_GAN_SRx4.pth | 170.3 | yes | yes |  |  |
| realesr-general-x4v3@1 | realesr-general-x4v3.pth | 4.9 | yes | yes |  |  |
| realesrgan-x2plus@1 | RealESRGAN_x2plus.pth | 67.1 | yes | yes |  |  |
| realesrgan-x4plus-anime-6b@1 | RealESRGAN_x4plus_anime_6B.pth | 17.9 | yes | yes |  |  |
| realesrgan-x4plus@1 | RealESRGAN_x4plus.pth | 67.0 | yes | yes |  |  |
| realplksr-nomoswebphoto-x4@1 | 4xNomosWebPhoto_RealPLKSR.safetensors | 29.6 | yes | yes |  |  |
| recover-cx@1 | recover_cx.pth | 276.6 | yes | yes |  |  |
| recover-mn@1 | recover_mn.pth | 276.6 | yes | yes |  |  |
| recover-rn@1 | recover_rn.pth | 276.6 | yes | yes |  |  |
| regnety_160.tv2_in1k@1 | regnet_y_16gf-3e4a00f9.pth | 335.0 | yes | yes |  |  |
| repvit_m0_9.dist_300e_in1k@1 | repvit_m0_9.dist_300e_in1k.safetensors | 22.2 | yes | yes |  |  |
| repvit_m0_9.dist_450e_in1k@1 | repvit_m0_9.dist_450e_in1k.safetensors | 22.2 | yes | yes |  |  |
| repvit_m1_0.dist_300e_in1k@1 | repvit_m1_0.dist_300e_in1k.safetensors | 29.5 | yes | yes |  |  |
| repvit_m1_0.dist_450e_in1k@1 | repvit_m1_0.dist_450e_in1k.safetensors | 29.5 | yes | yes |  |  |
| repvit_m1_1.dist_300e_in1k@1 | repvit_m1_1.dist_300e_in1k.safetensors | 35.5 | yes | yes |  |  |
| repvit_m1_1.dist_450e_in1k@1 | repvit_m1_1.dist_450e_in1k.safetensors | 35.5 | yes | yes |  |  |
| repvit_m1_5.dist_300e_in1k@1 | repvit_m1_5.dist_300e_in1k.safetensors | 59.1 | yes | yes |  |  |
| repvit_m1_5.dist_450e_in1k@1 | repvit_m1_5.dist_450e_in1k.safetensors | 59.1 | yes | yes |  |  |
| repvit_m2_3.dist_300e_in1k@1 | repvit_m2_3.dist_300e_in1k.safetensors | 95.5 | yes | yes |  |  |
| repvit_m2_3.dist_450e_in1k@1 | repvit_m2_3.dist_450e_in1k.safetensors | 95.5 | yes | yes |  |  |
| resnet152.tv2_in1k@1 | resnet152-f82ba261.pth | 241.7 | yes | yes |  |  |
| resnet18.tv_in1k@1 | resnet18-f37072fd.pth | 46.8 | yes | yes |  |  |
| resnet50.tv2_in1k@1 | resnet50-11ad3fa6.pth | 102.5 | yes | yes |  |  |
| resplat-base-re10k@1 | resplat-base-re10k-256x256-view2-b90d1b53.pth | 889.0 | no | yes |  | nc |
| resplat-small-dl3dv@1 | resplat-small-dl3dv-256x448-view8-548993fe.pth | 306.9 | yes | yes |  |  |
| restormer-deblur@1 | motion_deblurring.pth | 104.7 | yes | yes |  |  |
| restormer-derain@1 | deraining.pth | 104.7 | yes | yes |  |  |
| restormer-sidd@1 | real_denoising.pth | 104.6 | yes | yes |  |  |
| retinaface-mobilenet0.25@1 | detection_mobilenet0.25_Final.pth | 1.8 | yes | yes |  |  |
| retinaface-resnet50@1 | detection_Resnet50_Final.pth | 109.5 | yes | yes |  |  |
| rf-detr-keypoint-preview@1 | rf-detr-keypoint-preview-xlarge.pth | 163.7 | yes | yes |  |  |
| rf-detr-l-coco@1 | rf-detr-large-2026.pth | 136.0 | yes | yes |  |  |
| rf-detr-m-coco@1 | rf-detr-medium.pth | 405.0 | yes | yes |  |  |
| rf-detr-n-coco@1 | rf-detr-nano.pth | 366.3 | yes | yes |  |  |
| rf-detr-s-coco@1 | rf-detr-small.pth | 386.0 | yes | yes |  |  |
| rf-detr-seg-2xl-coco@1 | rf-detr-seg-2xl-ft.pth | 154.9 | yes | yes |  |  |
| rf-detr-seg-l-coco@1 | rf-detr-seg-l-ft.pth | 145.1 | yes | yes |  |  |
| rf-detr-seg-m-coco@1 | rf-detr-seg-m-ft.pth | 143.0 | yes | yes |  |  |
| rf-detr-seg-n-coco@1 | rf-detr-seg-n-ft.pth | 134.5 | yes | yes |  |  |
| rf-detr-seg-s-coco@1 | rf-detr-seg-s-ft.pth | 135.0 | yes | yes |  |  |
| rf-detr-seg-xl-coco@1 | rf-detr-seg-xl-ft.pth | 152.7 | yes | yes |  |  |
| rife-4.25-lite@1 | rife-4.25-lite-flownet.pkl | 22.8 | yes | yes |  |  |
| rife-4.25@1 | rife-4.25-flownet.pkl | 22.9 | yes | yes |  |  |
| rife-4.26@1 | rife-4.26-flownet.pkl | 22.9 | yes | yes |  |  |
| rmbg-2.0@1 | rmbg-2.0.safetensors | 884.9 | no | yes | yes (user copy in data/gated_models/rmbg2, sha ok, hardlinked) | nc |
| romav1@1 | roma_outdoor.pth | 445.6 | yes | yes |  |  |
| romav2@1 | romav2.0.1.pt | 1095.9 | yes | yes |  |  |
| rt-monster-plusplus-mix-all@1 | Zero_shot.pth | 171.5 | yes | yes |  | nc |
| rtdetrv2-l-coco@1 | rtdetrv2_r50vd_6x_coco_ema.pth | 172.5 | yes | yes |  |  |
| rtdetrv2-m-coco@1 | rtdetrv2_r34vd_120e_coco_ema.pth | 126.3 | yes | yes |  |  |
| rtdetrv2-ms-coco@1 | rtdetrv2_r50vd_m_7x_coco_ema.pth | 147.3 | yes | yes |  |  |
| rtdetrv2-s-coco@1 | rtdetrv2_r18vd_120e_coco_rerun_48.1.pth | 81.2 | yes | yes |  |  |
| rtdetrv2-x-coco@1 | rtdetrv2_r101vd_6x_coco_from_paddle.pth | 307.7 | yes | yes |  |  |
| rtmdet-l-coco@1 | rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth | 453.8 | yes | yes |  |  |
| rtmdet-m-coco@1 | rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth | 224.3 | yes | yes |  |  |
| rtmdet-m-person@1 | rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth | 99.0 | yes | yes |  |  |
| rtmdet-nano-person@1 | rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth | 4.2 | yes | yes |  |  |
| rtmdet-s-coco@1 | rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth | 91.5 | yes | yes |  |  |
| rtmdet-tiny-coco@1 | rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth | 57.5 | yes | yes |  |  |
| rtmdet-x-coco@1 | rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth | 395.6 | yes | yes |  |  |
| rtmo-l@1 | rtmo-l-body7-640x640.pth | 179.3 | yes | yes |  |  |
| rtmo-m@1 | rtmo-m-body7-640x640.pth | 90.5 | yes | yes |  |  |
| rtmo-s@1 | rtmo-s-body7-640x640.pth | 39.8 | yes | yes |  |  |
| rtmo-t@1 | rtmo-t-body7-416x416.pth | 27.6 | yes | yes |  |  |
| rtmpose-l-384@1 | rtmpose-l-body7-384x288.pth | 111.5 | yes | yes |  |  |
| rtmpose-l-halpe26-384@1 | rtmpose-l-halpe26-384x288.pth | 113.3 | yes | yes |  |  |
| rtmpose-l-halpe26@1 | rtmpose-l-halpe26-256x192.pth | 112.7 | yes | yes |  |  |
| rtmpose-l@1 | rtmpose-l-body7-256x192.pth | 110.9 | yes | yes |  |  |
| rtmpose-m-384@1 | rtmpose-m-body7-384x288.pth | 55.1 | yes | yes |  |  |
| rtmpose-m-ap10k@1 | rtmpose-m-ap10k-256x256.pth | 54.7 | yes | yes |  |  |
| rtmpose-m-face6@1 | rtmpose-m-face6-256x256.pth | 68.1 | yes | yes |  |  |
| rtmpose-m-halpe26-384@1 | rtmpose-m-halpe26-384x288.pth | 56.4 | yes | yes |  |  |
| rtmpose-m-halpe26@1 | rtmpose-m-halpe26-256x192.pth | 55.9 | yes | yes |  |  |
| rtmpose-m-hand5@1 | rtmpose-m-hand5-256x256.pth | 55.3 | yes | yes |  |  |
| rtmpose-m@1 | rtmpose-m-body7-256x192.pth | 54.6 | yes | yes |  |  |
| rtmpose-s-face6@1 | rtmpose-s-face6-256x256.pth | 31.1 | yes | yes |  |  |
| rtmpose-s-halpe26@1 | rtmpose-s-halpe26-256x192.pth | 22.9 | yes | yes |  |  |
| rtmpose-s@1 | rtmpose-s-body7-256x192.pth | 22.0 | yes | yes |  |  |
| rtmpose-t-face6@1 | rtmpose-t-face6-256x256.pth | 20.3 | yes | yes |  |  |
| rtmpose-t-halpe26@1 | rtmpose-t-halpe26-256x192.pth | 14.1 | yes | yes |  |  |
| rtmpose-t@1 | rtmpose-t-body7-256x192.pth | 13.5 | yes | yes |  |  |
| rtmpose-x-halpe26-384@1 | rtmpose-x-halpe26-384x288.pth | 200.4 | yes | yes |  |  |
| rtmw-l-384@1 | rtmw-l-cocktail14-384x288.pth | 230.3 | yes | yes |  |  |
| rtmw-l@1 | rtmw-l-cocktail14-256x192.pth | 229.7 | yes | yes |  |  |
| rtmw-m@1 | rtmw-m-cocktail14-256x192.pth | 129.8 | yes | yes |  |  |
| rtmw-x-384@1 | rtmw-x-cocktail14-384x288.pth | 369.7 | yes | yes |  |  |
| rtmw-x@1 | rtmw-x-cocktail14-256x192.pth | 369.1 | yes | yes |  |  |
| s2m2-l@1 | CH256NTR3.pth | 725.4 | yes | yes |  | nc |
| s2m2-m@1 | CH192NTR2.pth | 323.8 | yes | yes |  | nc |
| s2m2-s@1 | CH128NTR1.pth | 107.5 | yes | yes |  | nc |
| s2m2-xl@1 | CH384NTR3.pth | 1626.5 | yes | yes |  | nc |
| sam-hq-vit-b@1 | sam_hq_vit_b.pth | 379.3 | yes | yes |  |  |
| sam-hq-vit-h@1 | sam_hq_vit_h.pth | 2570.9 | yes | yes |  |  |
| sam-hq-vit-l@1 | sam_hq_vit_l.pth | 1254.9 | yes | yes |  |  |
| sam-vit-b@1 | sam_vit_b_01ec64.pth | 375.0 | yes | yes |  |  |
| sam-vit-h@1 | sam_vit_h_4b8939.pth | 2564.6 | yes | yes |  |  |
| sam-vit-l@1 | sam_vit_l_0b3195.pth | 1249.5 | yes | yes |  |  |
| sam2.1-hiera-base-plus@1 | sam2.1_hiera_base_plus.pt | 323.6 | yes | yes |  |  |
| sam2.1-hiera-large@1 | sam2.1_hiera_large.pt | 898.1 | yes | yes |  |  |
| sam2.1-hiera-small@1 | sam2.1_hiera_small.pt | 184.4 | yes | yes |  |  |
| sam2.1-hiera-tiny@1 | sam2.1_hiera_tiny.pt | 156.0 | yes | yes |  |  |
| sam3@1 | model.safetensors | 3439.9 | yes | yes |  |  |
| sam3@1 | vocab.json | 0.9 | yes | yes |  |  |
| sapiens-seg-0.3b-goliath@1 | sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2 | 1358.9 | yes | yes |  | nc |
| sapiens-seg-1b-goliath@1 | sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 | 4716.3 | yes | yes |  | nc |
| sea-raft-m-kitti@1 | Tartan-C-T-TSKH-kitti432x960-M.safetensors | 78.8 | yes | yes |  |  |
| sea-raft-m-spring@1 | Tartan-C-T-TSKH-spring540x960-M.safetensors | 78.8 | yes | yes |  |  |
| sea-raft-m-things@1 | Tartan-C-T432x960-M.safetensors | 78.8 | yes | yes |  |  |
| sea-raft-m@1 | Tartan-C-T-TSKH432x960-M.safetensors | 78.8 | yes | yes |  |  |
| sea-raft-s-kitti@1 | Tartan-C-T-TSKH-kitti432x960-S.safetensors | 35.6 | yes | yes |  |  |
| sea-raft-s-spring@1 | Tartan-C-T-TSKH-spring540x960-S.safetensors | 35.6 | yes | yes |  |  |
| sea-raft-s@1 | Tartan-C-T-TSKH432x960-S.safetensors | 35.6 | yes | yes |  |  |
| seedvr2-3b@1 | seedvr2-3b.safetensors | 6784.9 | no | yes |  |  |
| seedvr2-3b@1 | seedvr2-vae.safetensors | 501.3 | no | yes |  |  |
| seedvr2-7b@1 | seedvr2-7b.safetensors | 16482.6 | no | yes |  |  |
| segformer-b0-ade20k@1 | segformer-b0-ade20k.config.json | 0.0 | yes | yes |  | nc |
| segformer-b0-ade20k@1 | segformer-b0-ade20k.safetensors | 15.0 | yes | yes |  | nc |
| segformer-b2-clothes@1 | segformer-b2-clothes.config.json | 0.0 | yes | yes |  | nc |
| segformer-b2-clothes@1 | segformer-b2-clothes.safetensors | 109.5 | yes | yes |  | nc |
| segformer-b5-ade20k@1 | segformer-b5-ade20k.bin | 339.3 | yes | yes |  | nc |
| segformer-b5-ade20k@1 | segformer-b5-ade20k.config.json | 0.0 | yes | yes |  | nc |
| segformer-b5-face-parsing@1 | segformer-b5-face-parsing.config.json | 0.0 | yes | yes |  | nc |
| segformer-b5-face-parsing@1 | segformer-b5-face-parsing.safetensors | 338.6 | yes | yes |  | nc |
| segnext-mscan-b-ade20k@1 | segnext-mscan-b-ade20k.pth | 111.0 | yes | yes |  |  |
| segnext-mscan-l-ade20k@1 | segnext-mscan-l-ade20k.pth | 196.5 | yes | yes |  |  |
| segnext-mscan-s-ade20k@1 | segnext-mscan-s-ade20k.pth | 56.0 | yes | yes |  |  |
| segnext-mscan-t-ade20k@1 | segnext-mscan-t-ade20k.pth | 17.3 | yes | yes |  |  |
| selective-igev-eth3d-finetune@1 | eth3d_finetune.pth | 53.0 | yes | yes |  |  |
| selective-igev-eth3d-train@1 | eth3d_train.pth | 53.0 | yes | yes |  |  |
| selective-igev-kitti12@1 | kitti12.pth | 53.0 | yes | yes |  |  |
| selective-igev-kitti15@1 | kitti15.pth | 53.0 | yes | yes |  |  |
| selective-igev-middlebury-finetune@1 | middlebury_finetune.pth | 53.0 | yes | yes |  |  |
| selective-igev-middlebury-train@1 | middlebury_train.pth | 53.0 | yes | yes |  |  |
| selective-igev-sceneflow@1 | sceneflow.pth | 53.0 | yes | yes |  |  |
| shufflenet_v2_x1_0.tv_in1k@1 | shufflenetv2_x1-5666bf0f80.pth | 9.2 | yes | yes |  |  |
| shvit_s1.in1k@1 | shvit_s1.in1k.safetensors | 25.5 | yes | yes |  |  |
| shvit_s2.in1k@1 | shvit_s2.in1k.safetensors | 46.2 | yes | yes |  |  |
| shvit_s3.in1k@1 | shvit_s3.in1k.safetensors | 57.3 | yes | yes |  |  |
| shvit_s4.in1k@1 | shvit_s4.in1k.safetensors | 66.8 | yes | yes |  |  |
| siglip2-b16-256@1 | vit_base_patch16_siglip_256.v2_webli.safetensors | 371.7 | yes | yes |  |  |
| siglip2-b16-512@1 | vit_base_patch16_siglip_512.v2_webli.safetensors | 374.1 | yes | yes |  |  |
| siglip2-g16-256@1 | vit_giantopt_patch16_siglip_256.v2_webli.safetensors | 4652.7 | yes | yes |  |  |
| siglip2-g16-384@1 | vit_giantopt_patch16_siglip_384.v2_webli.safetensors | 4654.7 | yes | yes |  |  |
| siglip2-l16-256@1 | vit_large_patch16_siglip_256.v2_webli.safetensors | 1263.9 | yes | yes |  |  |
| siglip2-l16-512@1 | vit_large_patch16_siglip_512.v2_webli.safetensors | 1267.0 | yes | yes |  |  |
| siglip2-so400m14-224@1 | vit_so400m_patch14_siglip_224.v2_webli.safetensors | 1710.8 | yes | yes |  |  |
| siglip2-so400m14-378@1 | vit_so400m_patch14_siglip_378.v2_webli.safetensors | 1712.9 | yes | yes |  |  |
| siglip2-so400m16-256@1 | vit_so400m_patch16_siglip_256.v2_webli.safetensors | 1711.6 | yes | yes |  |  |
| siglip2-so400m16-512@1 | vit_so400m_patch16_siglip_512.v2_webli.safetensors | 1715.1 | yes | yes |  |  |
| span-x2@1 | spanx2_ch48.pth | 1326.8 | no | yes |  |  |
| span-x4@1 | spanx4_ch48.pth | 1326.8 | no | yes |  |  |
| squeezenet1_1.tv_in1k@1 | squeezenet1_1-b8a52dc0.pth | 5.0 | yes | yes |  |  |
| stablenormal-turbo@1 | controlnet/diffusion_pytorch_model.fp16.safetensors | 728.5 | yes | yes |  |  |
| stablenormal-turbo@1 | unet/diffusion_pytorch_model.fp16.safetensors | 1731.9 | yes | yes |  |  |
| starnet_s1.in1k@1 | starnet_s1.in1k.safetensors | 11.5 | yes | yes |  |  |
| starnet_s2.in1k@1 | starnet_s2.in1k.safetensors | 14.8 | yes | yes |  |  |
| starnet_s3.in1k@1 | starnet_s3.in1k.safetensors | 23.1 | yes | yes |  |  |
| starnet_s4.in1k@1 | starnet_s4.in1k.safetensors | 30.0 | yes | yes |  |  |
| swin2sr-x2@1 | Swin2SR_ClassicalSR_X2_64.pth | 68.8 | yes | yes |  |  |
| swin2sr-x4-real@1 | Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth | 68.6 | yes | yes |  |  |
| swin2sr-x4@1 | Swin2SR_ClassicalSR_X4_64.pth | 69.4 | yes | yes |  |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k@1 | swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors | 352.7 | yes | yes |  |  |
| swinir-l-x4-real@1 | 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth | 142.5 | yes | yes |  |  |
| swinir-m-x2@1 | 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth | 67.3 | yes | yes |  |  |
| swinir-m-x4-real@1 | 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth | 67.1 | yes | yes |  |  |
| swinir-m-x4@1 | 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth | 67.9 | yes | yes |  |  |
| tapir@1 | tapir_checkpoint_panning.pt | 124.3 | yes | yes |  |  |
| tapnext-pp@1 | tapnextpp_ckpt.pt | 2532.3 | yes | yes |  |  |
| tapnext@1 | bootstapnext_ckpt.npz | 777.0 | yes | yes |  |  |
| teed@1 | teed_5_model.pth | 0.2 | yes | yes |  |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k@1 | tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors | 44.1 | yes | yes |  |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k@1 | tiny_vit_21m_224.dist_in22k_ft_in1k.safetensors | 84.9 | yes | yes |  |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k@1 | tiny_vit_21m_384.dist_in22k_ft_in1k.safetensors | 85.0 | yes | yes |  |  |
| tiny_vit_21m_512.dist_in22k_ft_in1k@1 | tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors | 85.2 | yes | yes |  |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k@1 | tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors | 21.6 | yes | yes |  |  |
| tipsv2-b14@1 | tipsv2-b14/model.safetensors | 783.8 | yes | yes |  |  |
| tipsv2-l14@1 | tipsv2-l14/model.safetensors | 1951.8 | yes | yes |  |  |
| totalsegmentator-total-3mm@1 | Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset.json | 135.4 | yes | yes |  |  |
| totalsegmentator-total-3mm@1 | Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 135.4 | yes | yes |  |  |
| totalsegmentator-total-3mm@1 | Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json | 135.4 | yes | yes |  |  |
| totalsegmentator-total@1 | Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json | 233.7 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 233.7 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json | 233.7 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json | 234.1 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 234.1 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json | 234.1 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json | 234.2 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 234.2 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json | 234.2 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json | 233.6 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 233.6 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset294_TotalSegmentator_part4_muscles_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json | 233.6 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json | 234.0 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth | 234.0 | no | yes |  |  |
| totalsegmentator-total@1 | Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json | 234.0 | no | yes |  |  |
| track-on-r@1 | dinov3-vits16plus-pretrain-lvd1689m/model.safetensors | 114.8 | yes | yes |  |  |
| track-on-r@1 | track_on_r.pt | 93.9 | yes | yes |  |  |
| track-on2@1 | trackon2_dinov3_checkpoint.pt | 93.9 | yes | yes |  |  |
| u2net-human-seg@1 | u2net_human_seg.pth | 176.3 | yes | yes |  |  |
| u2net@1 | u2net.pth | 176.3 | yes | yes |  |  |
| u2netp@1 | u2netp.pth | 4.7 | yes | yes |  |  |
| unidepth-v2-vitl14@1 | model.safetensors | 1415.4 | yes | yes |  | nc |
| unik3d-vitl@1 | model.safetensors | 1435.4 | yes | yes |  | nc |
| vgg16.tv_in1k@1 | vgg16-397923af.pth | 553.4 | yes | yes |  |  |
| vggt-1b@1 | model.pt | 5026.9 | yes | yes |  | nc |
| vggt-omega-1b-512@1 | vggt_omega_1b_512.pt | 4576.7 | yes | yes |  | nc |
| video-depth-anything-metric-s@1 | metric_video_depth_anything_vits.pth | 116.4 | yes | yes |  |  |
| video-depth-anything-s@1 | video_depth_anything_vits.pth | 116.4 | yes | yes |  |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k@1 | vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors | 346.3 | yes | yes |  |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k@1 | vit_so400m_patch14_siglip_378.webli_ft_in1k.safetensors | 1717.5 | yes | yes |  |  |
| vitmatte-b@1 | vitmatte-b-composition-1k.bin | 386.8 | yes | yes |  |  |
| vitmatte-s@1 | vitmatte-s-composition-1k.safetensors | 103.3 | yes | yes |  |  |
| vitpose-base-simple@1 | vitpose-base-simple-85375373.safetensors | 343.7 | yes | yes |  |  |
| vitpose-base@1 | vitpose-base-cd9a4e6c.safetensors | 360.0 | yes | yes |  |  |
| vitpose-plus-base@1 | vitpose-plus-base-640225e4.safetensors | 501.6 | yes | yes |  |  |
| vitpose-plus-huge@1 | vitpose-plus-huge-0ecb49f1.safetensors | 3597.7 | yes | yes |  |  |
| vitpose-plus-large@1 | vitpose-plus-large-8b249f74.safetensors | 1737.7 | yes | yes |  |  |
| vitpose-plus-small@1 | vitpose-plus-small-f7bad8ed.safetensors | 132.6 | yes | yes |  |  |
| waft-a1-kitti@1 | waft-a1-tar-c-t-kitti.pth | 256.9 | yes | yes |  |  |
| waft-a1-sintel@1 | waft-a1-tar-c-t-sintel.pth | 256.9 | yes | yes |  |  |
| waft-a1@1 | waft-a1-tar-c-t.pth | 256.9 | yes | yes |  |  |
| waft-stereo-b@1 | DAv2B-4.pth | 1127.7 | yes | yes |  | nc |
| waft-stereo-s@1 | DAv2S-4.pth | 747.8 | yes | yes |  |  |
| wd-convnext-tagger-v3@1 | wd-convnext-tagger-v3.safetensors | 394.8 | yes | yes |  |  |
| wd-eva02-large-tagger-v3@1 | wd-eva02-large-tagger-v3.safetensors | 1260.8 | yes | yes |  |  |
| wd-vit-large-tagger-v3@1 | wd-vit-large-tagger-v3.safetensors | 1260.4 | yes | yes |  |  |
| wd-vit-tagger-v3@1 | wd-vit-tagger-v3.safetensors | 378.4 | yes | yes |  |  |
| xfeat-lighterglue@1 | xfeat-lighterglue.pt | 10.8 | yes | yes |  |  |
| xfeat@1 | xfeat.pt | 6.2 | yes | yes |  |  |
| yolox-l@1 | yolox_l.pth | 434.4 | yes | yes |  |  |
| yolox-m@1 | yolox_m.pth | 203.1 | yes | yes |  |  |
| yolox-s@1 | yolox_s.pth | 72.1 | yes | yes |  |  |
| yolox-tiny@1 | yolox_tiny.pth | 40.8 | yes | yes |  |  |
| yolox-x@1 | yolox_x.pth | 793.5 | no | yes |  |  |
| yunet-n@1 | yunet_n.pth | 0.7 | yes | yes |  |  |
| yunet-s@1 | yunet_s.pth | 0.6 | yes | yes |  |  |
| zoedepth-nk@1 | ZoeD_M12_NK.pt | 1448.6 | yes | yes |  |  |
