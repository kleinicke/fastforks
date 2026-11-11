# Originals on Apple MPS: which run out of the box (static audit)

Status (2026-11-06): read, not run. The authors' code at the pinned commits was read for anything that stops it on
MPS; nothing was adapted and nothing was measured. Verdicts:

- `works`: the authors' inference entry runs on MPS as shipped (a device flag that accepts `mps`, or device chosen
  from the inputs), with no CUDA-only op on the default path.
- `cuda-hardcoded`: `.cuda()`, `DEVICE='cuda'`, `torch.cuda.*` calls or a device flag without `mps`. Where only the
  entry script does this and the model code itself is device-agnostic, the note says so: those would run with a
  one-line device change, which is an adaptation and so out of scope.
- `cuda-extension`: a compiled CUDA extension on the default path, or not a PyTorch model at all.
- `unclear`: could not be decided statically.

Not counted as blockers: `torch.cuda.amp.autocast(...)` / `torch.autocast('cuda')` (they warn and disable themselves
without CUDA). `torch.load` without `map_location` on a GPU-saved checkpoint is a blocker and is listed.

## Stereo (27 families): works 0, cuda-hardcoded 23, cuda-extension 4

Model code itself hard-codes CUDA in psmnet, cfnet, coex; s2m2's inference helper calls CUDA events unconditionally;
las2's flag accepts only cpu / cuda (it silently runs on the CPU on a Mac); monster_plusplus and rt_monster_plusplus
load a Depth-Anything file from the authors' absolute `/home/lwj/...` path, so they fail on any machine as shipped.
The rest are device-agnostic models behind entry scripts that hard-code CUDA. crestereo (MegEngine) and hitnet
(TF1 frozen graph) are not torch models. Not verified statically: `torch.linalg.solve` on MPS (monster, test-mode
helper), 5D trilinear upsampling (cas_gwcnet).

| family | verdict | evidence (file:line at pin) | note |
|---|---|---|---|
| aanet | cuda-extension | nets/deform_conv/src/deform_conv_cuda_kernel.cu; nets/deform_conv/deform_conv.py:9 `from . import deform_conv_cuda`; nets/deform.py:3 imports it; nets/aggregation.py:5 imports nets.deform; inference.py:88 | Compiled deformable-conv CUDA ext imported at module import time (build via nets/deform_conv/build.sh), no pure-torch fallback; inference.py falls back to cpu but import fails first |
| bridgedepth | cuda-hardcoded | infer.py:115 `.cuda()`, infer.py:254 `model.to(torch.device("cuda"))`; demo.py:71,91; bridgedepth/stereo/NMP.py:219,236 `device=torch.device('cuda')` defaults | Model core is device-agnostic (NMP.py:263,510,742 pass device explicitly); only entry scripts hard-code CUDA. No CUDA ext found on forward path |
| c_foundationstereo | cuda-hardcoded | nvidia_tao_pytorch/cv/depth_net/model/stereo_depth/pl_stereo_model.py:418 `disp_pred.cuda().float()` in predict_step; scripts/inference.py:55 `Trainer(**trainer_kwargs)` (Lightning, GPU); fast_foundation_stereo.py:391 `AUTOCAST('cuda')` | TAO Toolkit Fast-FoundationStereo; Lightning/hydra/TAO runner stack. foundation_stereo/geometry.py:135 `device='cuda'` is in a mask helper (not clearly on default path). TensorRT only via separate tao-deploy. No hand-written CUDA kernels on the forward path |
| cas_gwcnet | cuda-hardcoded | CasStereoNet/save_disp.py:73 `model.cuda()`, :107 `.cuda()`; main.py:132 | Model code (CasStereoNet/models/*.py) uses `device=x.device`, device-agnostic; only scripts hard-code CUDA. No compiled ops; models use 4D grid_sample + 3D convs |
| cfnet | cuda-hardcoded | models/submodule.py:218-219 `xx.float().cuda()`, :230 `torch.autograd.Variable(torch.ones(x.size())).cuda()` (in warp helper, guarded by `if x.is_cuda` only for lines 218-219); save_disp.py:45,78 | Hard-coded `.cuda()` inside the model code (line 230 is unconditional); also in models/relu/submodule.py |
| coex | cuda-hardcoded | models/stereo/CoEx.py:125 `Variable(torch.FloatTensor(...).zero_()).cuda()`; models/stereo/submodules/spixel_utils/spixel.py:45,59,101 `.cuda()`; demo_torchscript.py:57 | Hard-coded `.cuda()` in model code (cost-volume build and superpixel utils). TensorRT/TorchScript demos are separate scripts |
| crestereo | cuda-extension | test.py:3 `import megengine as mge`; nets/crestereo.py, nets/corr.py:3 `import megengine.module as M` | NOT PyTorch: the original is MegEngine (zero `import torch`), so it has no MPS backend path at all. Label is closest-fit, not a CUDA kernel issue; strictly "not a torch model" |
| defom_stereo | cuda-hardcoded | demo.py:16 `DEVICE='cuda'`, :25 `torch.load(..., map_location='cuda')`; evaluate_stereo.py:413; core/defom_stereo.py:11 | Default `--corr_implementation` is pure-torch `reg` (evaluate_stereo.py:390); `reg_cuda` is optional. Model core uses `img.device` (core/defom_stereo.py:55). xformers is try/except (depth_anything_v2/dinov2_layers/attention.py:19-24), falls back. Entry hard-codes CUDA |
| dlnr | cuda-extension | evaluate_stereo.py:227 `--corr_implementation` default `"reg_cuda"` (choices reg/reg_cuda); sampler/setup.py:19 `CUDAExtension('corr_sampler')`; core/corr.py:6-9 import in try/except; evaluate_stereo.py:31,250 `.cuda()` | Default needs compiled `corr_sampler`; pure-torch `reg` exists but is not the default. No demo script, only evaluate_stereo (hard-coded `.cuda()` and datasets). nets/deform.py (deform conv) is not imported by core/dlnr.py |
| fast_acvnet | cuda-hardcoded | save_disp.py:48 `model.cuda()`, :84 `.cuda()` | Model code (models/*.py) is device-agnostic (`device=x.device`, submodule.py:85,166,203). Only scripts hard-code CUDA. 4D grid_sample only |
| foundationstereo | cuda-hardcoded | scripts/run_demo.py:62 `torch.load(ckpt_dir)` (no map_location), :66 `model.cuda()`, :80-81 `.cuda()`; scripts/run_demo_batch.py:37,45,247 | Model core (core/*.py) has no hard-coded device (only `torch.cuda.amp.autocast(enabled=False)` at core/submodule.py:390, core/geometry.py:73, which is inert off-CUDA). Entry scripts hard-code CUDA; torch.load needs map_location; TensorRT script separate |
| great_igev | cuda-hardcoded | save_disp_kitti.py:14, save_disp_eth3d.py:14, save_disp_middlebury.py:14 `DEVICE='cuda'`; torch.load without map_location (e.g. save_disp_kitti.py:39) | Model dirs (models/great_stereo, models/greaten_stereo) have no hard-coded devices; modules/backbones/depth_anything/depth_anything.py:401,487 even pick mps. xformers is try/except (dino_layers). Entry scripts hard-code CUDA |
| gwcnet | cuda-hardcoded | save_disp.py:45 `model.cuda()`, :77 `.cuda()`; main.py:67 | Model code (models/*.py) device-agnostic (models/submodule.py:25 `device=x.device`). Only scripts hard-code CUDA |
| hitnet | cuda-extension | hitnet/hitnet/predict.py:28 `import tensorflow.compat.v1 as tf`, :231 `tf.import_graph_def`; hitnet/hitnet/README.md "Performance" | NOT PyTorch: TF1 frozen-graph inference script only (no model code). Custom CUDA ops are mentioned but not released. Label is closest-fit; strictly "no torch model, cannot use torch MPS" |
| igev_plusplus | cuda-hardcoded | demo_imgs.py:20 and save_disp.py:19 `DEVICE='cuda'`; :31/:30 `torch.load(args.restore_ckpt)` without map_location; core/update.py:116 `torch.cuda.amp.autocast(enabled=False)` | Model core uses `.device` / `device=prob.device` (core/submodule.py:149,156). Pure-torch corr (3D conv + 4D grid_sample). Also core_rt (RT variant) same pattern. Entry scripts hard-code CUDA |
| igev_stereo | cuda-hardcoded | IGEV-Stereo/demo_imgs.py:3 and save_disp.py:19 `DEVICE='cuda'`, :26 `torch.load(args.restore_ckpt)` without map_location; core/igev_stereo.py:12 | Model core device-agnostic (core/igev_stereo.py:194, core/submodule.py:221, core/geometry.py:41). IGEV-MVS subdir not considered |
| las2 | cuda-hardcoded | demo.py:64 `--device` choices `["cpu","cuda"]`, :92 `torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")`; evaluate_stereo.py:224,241 same | Model code and `torch.load(..., map_location=device)` (demo.py:50) are device-agnostic, but the device flag cannot select mps, so on a Mac it silently runs on CPU. Works on MPS only if the device selection is edited |
| mocha_stereo | cuda-hardcoded | MoCha-Stereo/demo.py:3 `DEVICE='cuda'`, :26 `torch.load(args.restore_ckpt)` no map_location; MoCha-V2/motif.py:20 and MoCha-V2/MCCV.py:74 `torch.cdist(...).cuda()` | MoCha-Stereo model core (core/mocha_stereo.py, nets/) has no hard-coded device. The `.cuda()` calls are in the MoCha-V2 subdir. Entry script hard-codes CUDA |
| monster | cuda-hardcoded | save_disp.py:20 `DEVICE='cuda'`, :60 `torch.load(args.restore_ckpt)` no map_location; core/monster.py:294 loads `./pretrained/depth_anything_v2_{encoder}.pth`; :356,:419 `torch.autocast(device_type='cuda')`; core/monster.py:55-56 `torch.linalg.solve` | Model core device-agnostic. Needs a separately downloaded Depth-Anything-V2 weight file at `./pretrained/`. `linalg.solve` MPS support not verified (in a scale/shift helper at :420, on the test-mode path). xformers try/except |
| monster_plusplus | cuda-hardcoded | MonSter++/save_disp.py:20 `DEVICE='cuda'`, :60 `torch.load` no map_location, :96,:99 `torch.cuda.synchronize()`; MonSter++/core/monster.py:294 `torch.load('/home/lwj/monster/MonSter/checkpoints/depth_anything_v2_...')` | Model construction loads an absolute path on the authors' machine (`/home/lwj/...`), so it fails even on a CUDA box unless the file is placed there or the line is edited. Entry script hard-codes CUDA |
| nmrf | cuda-hardcoded | inference.py:147 `model.to(torch.device("cuda"))`, :148 `map_location="cuda"`; nmrf/models/NMP.py:245,260,861 `device=torch.device('cuda')` defaults | Default backbone is resnet (nmrf/config/default.py:30), which has no CUDA ext. The Swin variants (configs/*_swint.yaml) import nmrf/models/adaptor_modules.py:10 -> ops/functions/ms_deform_attn_func.py:11 `import MultiScaleDeformableAttention` (compiled `ops/src/cuda/*.cu`): that variant is `cuda-extension` |
| psmnet | cuda-hardcoded | models/submodule.py:48 `torch.Tensor(...).cuda()` in `disparityregression.__init__`; models/stackhourglass.py:110 `Variable(torch.FloatTensor(...).zero_()).cuda()`; submission.py:52,64 | Hard-coded `.cuda()` inside the model code. models/basic.py:66 uses `self.maxdisp/4` (float size, py2-era, breaks on py3 for the basic variant) |
| raft_stereo | cuda-hardcoded | demo.py:16 `DEVICE='cuda'`, :25 `torch.load(args.restore_ckpt)` no map_location, :28 `model.to(DEVICE)`; core/raft_stereo.py:11 | Default `--corr_implementation` is pure-torch `reg` (demo.py:67); `reg_cuda`/`alt_cuda` (sampler/*.cu) are optional and try/except-imported (core/corr.py:5-12). Model core uses `img.device` (core/raft_stereo.py:50). Entry hard-codes CUDA, no device flag |
| rt_monster_plusplus | cuda-hardcoded | RT-MonSter++/save_disp.py:20 `DEVICE='cuda'`, :60 `torch.load` no map_location, :98,:101 `torch.cuda.synchronize()`; RT-MonSter++/core/monster.py:292 `torch.load('/home/lwj/MonSter_kitti/checkpoint/depth_anything_v2_...')` | Same repo/pin as monster_plusplus (RT-MonSter++ subdir). Absolute `/home/lwj/...` path at model construction. TensorRT scripts (infer_trt.py) are separate |
| s2m2 | cuda-hardcoded | src/s2m2/core/utils/model_utils.py:77 `torch.cuda.Event(enable_timing=True)`, :83 `torch.cuda.synchronize()` unconditionally in `run_stereo_matching`; demo/visualize_2d_simple.py:14 picks `cuda:0` else `cpu` (never mps) | The model itself (src/s2m2/core/model/) is device-agnostic and `load_model(device=...)` takes any device; but the shipped inference helper unconditionally uses CUDA events, so MPS fails. Also `torch.load(ckpt_path, weights_only=True)` without map_location (:39) |
| selective_igev | cuda-hardcoded | Selective-IGEV/demo_imgs.py:17 `DEVICE="cuda"`, save_disp.py:17, :27 `torch.load` no map_location; core/igev_stereo.py:11 | Model core device-agnostic. Selective-RAFT subdir has the optional corr_sampler CUDA ext (Selective-RAFT/sampler/sampler_kernel.cu), not used by Selective-IGEV |
| waft_stereo | cuda-hardcoded | demo/demo.py:88 `model.to(torch.device("cuda"))`, :105-106 `.cuda()`, :109 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`; submission.py:104 `.cuda()` | Model core (model/) has no hard-coded device; demo loads checkpoint with map_location='cpu' (demo.py:90). xformers is try/except in thirdparty/DepthAnythingV2. Entry scripts hard-code CUDA |

## Other tasks (54 families): works 44, cuda-hardcoded 4, cuda-extension 4, unclear 2

- cuda-hardcoded: metric3d, patchmatchnet, sam3, sapiens (`.cuda()` / `device="cuda"` in the inference path).
- cuda-extension: unidepth_v2, unik3d (triton and xformers pinned, no macOS wheels), dinov2_linear, mmseg (compiled
  mmcv ops).
- unclear: depth_anything_3 (lists xformers; falls back to SDPA or not), rcnn (detectron2 must be built from source;
  RoIAlign / NMS on MPS untested).
- The `works` families are mostly torchvision / timm classifiers and the transformers models, plus most depth and
  multi-view originals, which take a device argument.

Evidence is weaker than for stereo: the six transformers families (dinov3, dinov3_convnext, eomt, mask2former,
oneformer, segformer) are judged from knowledge of the transformers code, not re-read at the pin, and the evidence
column for the torchvision, timm and depth rows is a summary, not file:line. A `works` here means "no blocker found",
not "ran".

| task | family | verdict | evidence | note |
|---|---|---|---|---|
| classification | alexnet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | densenet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | efficientnet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | efficientnetv2 | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | inception_v3 | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | mobilenetv2 | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | mobilenetv3 | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | regnet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | resnet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | shufflenetv2 | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | squeezenet | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | vgg | works | torchvision.models, device-agnostic (no cuda in model code) |  |
| classification | convnextv2 | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | deit | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | dinov2_linear_cls | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | efficientvit | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | eva02 | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | fastvit | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | mobilenetv1 | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | mobilenetv4 | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | siglip | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | swin | works | timm.models, device-agnostic; weights via hub with map_location |  |
| classification | vit | works | timm.models, device-agnostic; weights via hub with map_location |  |
| depth | depth_anything_3 | unclear | pyproject.toml deps list xformers | xformers has no macOS wheel; test needs whether the code falls back to SDPA when xformers is missing and a DA3 run on MPS |
| depth | depth_anything_v2 | works | depth_anything_v2/dpt.py: model code device-agnostic; run.py picks cuda/mps/cpu |  |
| depth | depth_pro | works | src/depth_pro: device arg in create_model_and_transforms |  |
| depth | marigold | works | marigold/*pipeline.py: device argument; script run.py picks mps if available |  |
| depth | metric3d | cuda-hardcoded | .cuda() in mono/ model and hubconf inference paths, no device argument | needs source edit |
| depth | midas | works | midas/model_loader.py and run.py take a device; no cuda in model code | DPT/BEiT via timm |
| depth | moge | works | moge/model: device argument on from_pretrained/infer; no custom ops |  |
| depth | unidepth_v2 | cuda-extension | requirements.txt: triton, xformers (no macOS wheels) | model code otherwise device-agnostic; pins block install |
| depth | unik3d | cuda-extension | requirements.txt: triton, xformers (no macOS wheels) | same stack as UniDepth |
| depth | zoedepth | works | zoedepth/models/builder, device-agnostic; torch.hub MiDaS backbone | old timm pin may need care |
| detection | detr | works | models/*.py device-agnostic; pure-PyTorch matcher, no custom ops |  |
| detection | rcnn | unclear | detectron2/layers uses torchvision roi_align/nms; detectron2._C not on forward path | detectron2 must be built from source; test needs build plus MPS run of RoIAlign and NMS |
| detection | rtdetrv2 | works | rtdetrv2_pytorch: deformable attention pure PyTorch with F.grid_sample (default) |  |
| features | dinov2 | works | dinov2/layers/attention.py: xformers optional (try/except, SDPA fallback) |  |
| features | dinov3 | works | transformers modeling_dinov3_vit.py, device-agnostic (judged from knowledge, not re-read) |  |
| features | dinov3_convnext | works | transformers modeling_dinov3_convnext.py, device-agnostic (judged from knowledge) |  |
| multiview | mapanything | works | mapanything/models: device argument, no custom ops |  |
| multiview | mvsanywhere | works | device-agnostic inference scripts with device selection |  |
| multiview | patchmatchnet | cuda-hardcoded | eval.py/models .cuda() in inference path | needs source edit |
| multiview | vggt | works | vggt/models: device-agnostic; demo uses device var, autocast cuda only warns |  |
| multiview | vggt_omega | works | same structure as vggt |  |
| promptable_segmentation | sam | works | segment_anything: SamPredictor takes device; no custom ops |  |
| promptable_segmentation | sam2 | works | build_sam.py device arg (default cuda but overridable); optional CUDAExtension sam2._C only for connected components, python fallback |  |
| promptable_segmentation | sam3 | cuda-hardcoded | sam3/model/decoder.py:301 device="cuda" in boxRPB init; sam3_tracker_base.py:663 .cuda(); sam3/sam/transformer.py:290 compute_cis device=cuda-if-available | image builder takes a device but decoder and tracker hard-code cuda; needs source edit |
| segmentation | dinov2_linear | cuda-extension | mmcv-full==1.5.0 and mmseg 0.27 pins (dinov2/eval/segmentation) | not installable on macOS arm64 |
| segmentation | eomt | works | transformers modeling_eomt.py, device-agnostic (judged from knowledge) |  |
| segmentation | mask2former | works | transformers: multi-scale deformable attention pure PyTorch via grid_sample (judged from knowledge) |  |
| segmentation | mmseg | cuda-extension | mmcv.ops imported at module top, mmcv<2.2 compiled ops | no pure-PyTorch default path |
| segmentation | oneformer | works | transformers: pure-PyTorch deformable attention (judged from knowledge) |  |
| segmentation | sapiens | cuda-hardcoded | lite/demo/vis_seg.py:34,48 and others: .cuda() and autocast(device_type="cuda"); lite models are torchscript/pt2 exports | demo scripts have no device argument; full stack needs mmcv |
| segmentation | segformer | works | transformers modeling_segformer.py, device-agnostic (judged from knowledge) |  |

## Measured on an M1 Pro 16 GB (2026-11-06)

The originals that run out of the box, timed with the survey protocol (first call, 4 warm-up, median of 5 synchronised
calls) on the same synthetic inputs as the portable rows: the torchvision and timm classifiers, SegFormer, Mask2Former
and DINOv3 (transformers). Random init, so speed only; installed releases (torch 2.14.1, torchvision 0.29.1, timm 1.0.30,
transformers 5.22.1), not the cards' pinned commits; the cards' test-time resize. 29 models x 2 tiers, 218 s, no
errors (DINOv3's portable has no MPS row: the card does not validate the family on MPS). The stereo, depth and multi-view originals need the authors' environments, which are not on this Mac.

- Portable is faster than the original for most CNNs (1.2-2.0x), the same for ViTs
  (DeiT, ViT, EVA-02, SigLIP, Mask2Former), slower only for AlexNet, SqueezeNet and VGG16 (0.8-0.9x).
- Autocast float16 makes the originals slower on MPS (median 1.14x the fp32 time; faster only for SegFormer and
  Mask2Former at 640x1280), while portable float16 (cast weights, no autocast) is faster (median 1.15x).

| model | original fp32 ms | original autocast fp16 ms | portable fp32 ms | portable fp16 ms | original / portable (fp32) |
|---|--:|--:|--:|--:|--:|
| alexnet.tv_in1k | 3.7 | 4.7 | 4.9 | 3.0 | 0.76x |
| densenet121.tv_in1k | 30.3 | 34.4 | 25.4 | - | 1.20x |
| efficientnet_b0.tv_in1k | 16.9 | 19.1 | 13.0 | - | 1.30x |
| efficientnet_b4.tv_in1k | 50.5 | 57.6 | 38.3 | - | 1.32x |
| efficientnet_v2_s.tv_in1k | 44.8 | 52.1 | 34.0 | - | 1.32x |
| inception_v3.tv_in1k | 26.4 | 29.5 | 17.4 | - | 1.52x |
| mobilenet_v2.tv2_in1k | 11.3 | 12.9 | 6.5 | - | 1.74x |
| mobilenet_v3_large.tv2_in1k | 12.8 | 14.7 | 8.5 | - | 1.49x |
| regnety_160.tv2_in1k | 45.5 | 51.7 | 38.9 | 33.1 | 1.17x |
| resnet152.tv2_in1k | 40.3 | 46.3 | 25.7 | - | 1.57x |
| resnet18.tv_in1k | 6.5 | 7.4 | 4.7 | - | 1.40x |
| resnet50.tv2_in1k | 16.1 | 18.4 | 11.2 | - | 1.43x |
| shufflenet_v2_x1_0.tv_in1k | 11.3 | 12.7 | 6.3 | - | 1.79x |
| squeezenet1_1.tv_in1k | 3.4 | 3.6 | 3.8 | 3.7 | 0.91x |
| vgg16.tv_in1k | 13.1 | 15.9 | 13.8 | 11.7 | 0.95x |
| convnextv2_base.fcmae_ft_in22k_in1k | 39.5 | 45.1 | 36.1 | - | 1.09x |
| deit_base_patch16_224.fb_in1k | 16.5 | 19.4 | 16.9 | - | 0.97x |
| efficientvit_b2.r288_in1k | 26.0 | 27.9 | 18.6 | - | 1.40x |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 266.0 | 284.9 | 267.3 | 234.8 | 1.00x |
| fastvit_sa24.apple_dist_in1k | 32.9 | 37.0 | 16.8 | - | 1.96x |
| mobilenetv1_100.ra4_e3600_r224_in1k | 6.9 | 8.1 | 4.3 | - | 1.62x |
| mobilenetv4_conv_medium.e500_r256_in1k | 17.1 | 19.8 | 9.6 | - | 1.79x |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 214.2 | 219.0 | 214.1 | 196.8 | 1.00x |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 35.9 | 39.5 | 28.2 | - | 1.27x |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 16.6 | 19.6 | 16.9 | - | 0.98x |
| segformer-b0-ade20k | 124.8 | 99.7 | 105.7 | - | 1.18x |
| segformer-b5-ade20k | 857.0 | 545.3 | 427.7 | - | 2.00x |
| mask2former-swinl-ade20k | 907.2 | 899.2 | 909.5 | 964.0 | 1.00x |
| dinov3-vitb16 | 22.3 | 26.8 | - | - | - |

## RF-DETR Keypoint Preview follow-up (2026-11-11)

The Apache preview has validated CPU/CUDA portable and custom CUDA paths. Portable MPS float32/float16 are
declared but unmeasured: compare the pinned original at both precisions before publishing Apple speed or accuracy.
The CUDA fastbb/graph path has no Metal counterpart. The whole-image decoder must preserve the original
uncertainty-fused person confidence, findability scores and 17-joint person schema.
