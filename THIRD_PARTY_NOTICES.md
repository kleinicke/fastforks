# Third-party notices

## CUTLASS (BSD-3-Clause)

The `s2m2_cutlass`, `fs_cutlass`, `ffs_cutlass`, `cutlass_conv`, `hitnet_conv`, `dpt_fused`, `fvm_conv` and `seg_conv` kernel libraries (`src/fastforks/kernels/{s2m2_cutlass,fs_cutlass,ffs_cutlass,cutlass_conv,hitnet_conv,dpt_fused,fvm_conv,seg_conv}/`) are compiled against the CUTLASS v4.7.1
headers (https://github.com/NVIDIA/cutlass); their prebuilt binaries therefore contain CUTLASS code. This notice is
published with every such binary.

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## CUDA runtime

The `.so` kernel libraries link the CUDA runtime statically (`-cudart static`), as permitted for applications by the
NVIDIA CUDA Toolkit EULA.

## Vendored model code

Each model directory under `src/fastforks/<task>/models/` that vendors upstream code contains that upstream licence file
verbatim (card field `files`); that licence covers the vendored code and our modifications of it. Families whose
network is the installed library's own module (torchvision, timm or transformers; card `files = []`, e.g. the
classification and several segmentation families) vendor no upstream code and so ship no licence file: the library is
a dependency that users install themselves under its own licence (torchvision BSD-3-Clause, timm and transformers
Apache-2.0), and the card's `[licence]` section records the code and weights licences.

## Kernel libraries derived from MIT / BSD code

The `raft` and `igev` kernel libraries (`src/fastforks/kernels/raft/`, `.../igev/`) re-implement layers of RAFT-Stereo,
IGEV-Stereo and DEFOM-Stereo (MIT) and the activation codes of WAFT-Stereo (BSD-3-Clause), e.g. the correlation lookup
and the ConvGRU, partly following the upstream CUDA sampler; the `raft_flow` library (`src/fastforks/kernels/raft_flow/`)
adapts the `raft` kernels to RAFT optical flow. As their notice requires, the notices of that original code follow.

```
MIT License
Copyright (c) 2021 Princeton Vision & Learning Lab   (RAFT-Stereo)
Copyright (c) 2023 Gangwei Xu                         (IGEV-Stereo)
Copyright (c) 2024 Insta360 Research Team             (DEFOM-Stereo)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

WAFT-Stereo (BSD-3-Clause, Copyright (c) 2026, Princeton Vision & Learning Lab): the licence text is in
`src/fastforks/stereo/models/waft_stereo/LICENSE`.

## Shared backbones

`src/fastforks/shared/mobilenetv2/` re-implements the module layout of timm's `mobilenetv2_100`
(https://github.com/huggingface/pytorch-image-models, Copyright 2019 Ross Wightman, Apache-2.0) so that checkpoints
trained with timm load unchanged. The Apache-2.0 licence text is in `src/fastforks/shared/mobilenetv2/LICENSE`.

`src/fastforks/shared/fasternet/` re-implements the layer layout of timm's `fasternet_t0` (Apache-2.0, text in its
`LICENSE`), which comes from FasterNet (https://github.com/JierunChen/FasterNet, Copyright (c) Microsoft Corporation,
MIT; the upstream repository has no LICENSE file, only per-file headers). The MIT text is appended to
`src/fastforks/shared/fasternet/LICENSE`.

## LiteAnyStereo V2 (H variant) GRU code

`src/fastforks/stereo/models/las2/net/liteanystereov2_H.py` contains `RaftConvGRU` and `SelectiveConvGRU`, taken from
RAFT-Stereo (https://github.com/princeton-vl/RAFT-Stereo, MIT, Copyright (c) 2021 Princeton Vision & Learning Lab; its
ConvGRU in turn follows RAFT, https://github.com/princeton-vl/RAFT, BSD-3-Clause, Copyright (c) 2020 princeton-vl) and
Selective-IGEV (https://github.com/Windsrain/Selective-Stereo, MIT, Copyright (c) 2024 Xianqi Wang). The three licence
texts are appended to `src/fastforks/stereo/models/las2/LICENSE` (the code carries no attribution header upstream; origin
identified by class names and structure).

## OpenCV resize arithmetic

`src/fastforks/detection/_cv2_resize.py` re-implements the 8-bit `INTER_LINEAR` path of OpenCV's `cv2.resize`
(https://github.com/opencv/opencv, `modules/imgproc/src/resize.cpp`, Apache-2.0, Copyright (C) 2000-2022 Intel
Corporation and the OpenCV contributors) in PyTorch integer arithmetic, so that RTMDet sees the pixels mmdetection's
test pipeline gives it; no OpenCV source is copied. The Apache-2.0 licence text is in
`src/fastforks/detection/models/rtmdet/LICENSE` (the vendored mmdetection code's licence file).

## EfficientViT-SAM prompt side

`src/fastforks/promptable_segmentation/models/efficientvit_sam/net/` is the EfficientViT-SAM code of
https://github.com/mit-han-lab/efficientvit (Apache-2.0, Copyright 2023 Han Cai); its `segment_anything/` folder is
that repository's copy of the prompt encoder, mask decoder and two-way transformer of Segment Anything
(https://github.com/facebookresearch/segment-anything, Apache-2.0, Copyright (c) Meta Platforms, Inc. and affiliates).
Both licence texts are in the family folder (`LICENSE`, `LICENSE-segment-anything`).

## MobileSAM

`src/fastforks/promptable_segmentation/models/mobile_sam/net/tiny_vit_sam.py` is the image encoder of
https://github.com/ChaoningZhang/MobileSAM (Apache-2.0), which is TinyViT of https://github.com/microsoft/Cream
(MIT, Copyright (c) Microsoft Corporation). Both licence texts are in the family folder (`LICENSE`,
`LICENSE-TinyViT`). Its prompt side is `src/fastforks/shared/sam_prompt/`: the prompt encoder, mask decoder and
two-way transformer of Segment Anything (https://github.com/facebookresearch/segment-anything, Apache-2.0, Copyright
(c) Meta Platforms, Inc. and affiliates), licence text in that folder (`LICENSE`).

## HQ-SAM

`src/fastforks/promptable_segmentation/models/sam_hq/net/` is the model code of https://github.com/SysCV/sam-hq
(Apache-2.0): Segment Anything (https://github.com/facebookresearch/segment-anything, Apache-2.0, Copyright (c) Meta
Platforms, Inc. and affiliates) as modified by the HQ-SAM team (image encoder that also returns its early features,
`MaskDecoderHQ`). The licence text is in the family folder (`LICENSE`).

## PyTorch LayerNorm (BSD-3-Clause)

The `sr_norm` library's 180- and 240-channel CUDA reductions follow the vectorized LayerNorm in PyTorch at
[5c4886908584029761b579af026dcfb627c84070](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/layer_norm_kernel.cu).

```text
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
