"""Measured Linux CUDA graph eligibility for selected workloads."""

from typing import TypeAlias

TensorSignature: TypeAlias = tuple[str, tuple[int, ...]]
MeasuredKey: TypeAlias = tuple[str, str, str, str, tuple[int, int] | tuple[TensorSignature, ...]]

_HostRuntime: TypeAlias = tuple[str, str, str, str | None, str | None, str | None, str | None, int]
_Resources: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]

_RTX_5060_TI = "NVIDIA GeForce RTX 5060 Ti"

# Torch version, GPU, allocator, allocator env aliases, Triton fusion, OMP threads, Torch threads.
_TORCH_214_NATIVE: _HostRuntime = ("2.14.1+cu130", _RTX_5060_TI, "native", None, None, None, "4", 1)
_TORCH_214_ASYNC: _HostRuntime = ("2.14.1+cu130", _RTX_5060_TI, "cudaMallocAsync", None, None, None, "4", 1)
_TORCH_214_ASYNC_FOUR_THREADS: _HostRuntime = (
    "2.14.1+cu130",
    _RTX_5060_TI,
    "cudaMallocAsync",
    None,
    None,
    None,
    "4",
    4,
)
_TORCH_214_ASYNC_TRITON_FUSION: _HostRuntime = (
    "2.14.1+cu130",
    _RTX_5060_TI,
    "cudaMallocAsync",
    None,
    None,
    "1",
    "4",
    1,
)
_TORCH_215_ASYNC: _HostRuntime = ("2.15.0+cu132", _RTX_5060_TI, "cudaMallocAsync", None, None, None, "4", 1)

# Resources retain ordered official checkpoint digests and sorted binary tags.
_FVM_BINARIES = (
    "fvm_conv@f1d064e539e1:sm_120",
    "fvm_extra@3208ce525a91:sm_120",
    "fvm_mb@720dd3f29bf7:sm_120",
    "fvm_r3@f54eb890e23c:sm_120",
)
_FVM_KERNELS = ("cudnn_hacc@cudnn92400", *_FVM_BINARIES)
_YUNET_KERNELS = ("box_nms@daa010d75e1a:sm_120", *_FVM_KERNELS)
_TAPNEXT_KERNELS = ("fvm_conv@f1d064e539e1:sm_120", "tapnext_state@be3777b468c9:sm_120")
_RIFE_KERNELS = ("cudnn_hacc@cudnn92400",)
_RAFT_CUDA_KERNELS = ("cutlass_conv@988138e042cb:sm_120", "raft_flow@639b3b16ea48:sm_120")
_RAFT_KERNELS = ("cudnn_hacc@cudnn92400", *_RAFT_CUDA_KERNELS)
_CRESTEREO_KERNELS = (
    "crestereo@4e3c00b2d395:sm_120",
    "cudnn_hacc@cudnn92400",
    "cutlass_conv@988138e042cb:sm_120",
    "fastbb@7d058322634d:sm_120",
    "igev@f075827d68c6:sm_120",
)
_DINOV3_CONVNEXT_KERNELS = ("fvm_conv@f1d064e539e1:sm_120", "fvm_seg@739693dff3b4:sm_120")
_IGEV_KERNELS = ("cudnn_hacc@cudnn92400", "cutlass_conv@988138e042cb:sm_120", "igev@f075827d68c6:sm_120")
_U2NET_KERNELS = (*_IGEV_KERNELS, "resize_cat@60f9273249a3:sm_120")

_MOBILENET_V2: _Resources = (_FVM_KERNELS, ("7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f",))
_COATNET_0: _Resources = (_FVM_KERNELS, ("efe01c8c139a4b0c360f508d10ae1217f7e3f3431fd85824438ed1110773366c",))
_YUNET_N: _Resources = (_YUNET_KERNELS, ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",))
_YUNET_S: _Resources = (_YUNET_KERNELS, ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",))
_RETINAFACE_MOBILENET: _Resources = (
    ("box_nms@daa010d75e1a:sm_120", *_FVM_KERNELS),
    ("2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220",),
)
_TAPNEXT: _Resources = (_TAPNEXT_KERNELS, ("19de6b1495555c2afb4b56e823a3f3c2fd7236eea05c53d98feab3cccd1b4a28",))
_TAPNEXT_PP: _Resources = (_TAPNEXT_KERNELS, ("cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da",))
_RIFE_425_LITE: _Resources = (_RIFE_KERNELS, ("81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1",))
_RIFE_425: _Resources = (_RIFE_KERNELS, ("6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",))
_RIFE_426: _Resources = (_RIFE_KERNELS, ("45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",))
_RAFT_SMALL_WEIGHTS = ("01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27",)
_RAFT_SMALL: _Resources = (_RAFT_KERNELS, _RAFT_SMALL_WEIGHTS)
_RAFT_LARGE: _Resources = (_RAFT_KERNELS, ("ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322",))
_CRESTEREO_ETH3D: _Resources = (
    _CRESTEREO_KERNELS,
    ("6a11b16a121d36077231ee65a793c7e60ba91d8fe7fe9475688fb5145c44596e",),
)
_RAFT_SMALL_ALTERNATE: _Resources = (
    ("cudnn_hacc@cudnn92600", *_RAFT_CUDA_KERNELS),
    _RAFT_SMALL_WEIGHTS,
)
_DINOV3_VITS16: _Resources = (
    ("dpt_fused@48e0065bb7cf:sm_120", "fastbb@7d058322634d:sm_120", "swiglu@25fe34a2d389:sm_120"),
    ("4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",),
)
_DINOV3_CONVNEXT_BASE: _Resources = (
    _DINOV3_CONVNEXT_KERNELS,
    ("ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc",),
)
_DINOV3_CONVNEXT_SMALL: _Resources = (
    _DINOV3_CONVNEXT_KERNELS,
    ("8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d",),
)
_DINOV3_CONVNEXT_TINY: _Resources = (
    _DINOV3_CONVNEXT_KERNELS,
    ("bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2",),
)
_SEGNEXT_S: _Resources = (
    (*_FVM_BINARIES, "seg_conv@8eec95783e28:sm_120", "seg_up@3e44e71200cd:sm_120"),
    ("ae9f7baa30817e62e36c9058ea3eeab0952baa3ce7ce52afef01423f68cbe1d8",),
)
_NEUFLOW_V2_SINTEL: _Resources = (
    ("raft_flow@639b3b16ea48:sm_120",),
    ("9bc12c9ef8298e3cca08a33a04bf99b63dc989e16808daca0c10ac5be4291eb1",),
)
_NEUFLOW_V2_THINGS: _Resources = (
    _NEUFLOW_V2_SINTEL[0],
    ("733d13b1b2202adefcc99bd1f0fceb89fc90da5479f9826fa3f17ff42c4bdbe0",),
)
_LAS2_S: _Resources = (
    (*_IGEV_KERNELS, "stereo_ops@899b54f4a755:sm_120"),
    ("dda5ad5495710466f7e5c5a2dff395223893ef68a17d57d88e7ea7dfbb9d08e1",),
)
_LAS2_M: _Resources = (_LAS2_S[0], ("c41b61efce9b36e6b8b38b580421cd20cc9e84a47f0f45361767281938f69696",))
_GWCNET_GC: _Resources = (_IGEV_KERNELS, ("95c9647f4896af1c9a3f2bb2e9989a569dde57c31b5a1b7830379b1d3434d13a",))
_BISENET_FACE_PARSING: _Resources = (
    _IGEV_KERNELS,
    ("468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",),
)
_PSMNET: _Resources = (_IGEV_KERNELS, ("eb3029f7ed5e62eae26dff42e3e5a44d47f014292ff680c7d76adca4749a9113",))
_S2M2_S: _Resources = (
    ("cudnn_hacc@cudnn92400", "s2m2@73d0d3fa70d2:sm_120", "s2m2_cutlass@3aec99821257:sm_120"),
    ("93f7177d8159d6666506df3abb0cdd75473614fab0e39f25e397acade0f9e241",),
)
_U2NETP: _Resources = (
    _U2NET_KERNELS,
    ("e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854",),
)
_U2NET: _Resources = (_U2NET_KERNELS, ("10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58",))
_NAFNET_GOPRO: _Resources = (
    ("cudnn_hacc@cudnn92400", "dpt_fused@48e0065bb7cf:sm_120", "nafnet_gate@95f8dd127200:sm_120"),
    ("329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",),
)
_HITNET_MIDDLEBURY: _Resources = (
    ("hitnet@45145b49dcb1:sm_120", "hitnet_conv@e5637e8a93cf:sm_120"),
    ("cc386fbb3181932e9ca9ac77f4a7bafc16f605ee490d5a0466b22f3df5e77fb0",),
)
_RTMPOSE_T: _Resources = (
    (*_FVM_KERNELS, "pose_warp@1589e39c4cb2:sm_120"),
    ("026a1439c5695f74f2e6e744427fc8190b57d630923eb772c1e9b4409a036f2f",),
)
_EFFICIENTFORMERV2_L: _Resources = (_FVM_KERNELS, ("a92de90055b116562f85322e596494829e54d86c56471652e3c07ed39957c92a",))
_SEGFORMER_B5_FACE: _Resources = (
    (*_FVM_KERNELS, "seg_up@3e44e71200cd:sm_120"),
    (
        "01d82e818569beda6aec804642e34ce187bed43326337d52ce370ff015964d68",
        "c2bec795a8c243db71bd95be538fd62559003566466c71237e45c99b920f4b62",
    ),
)


def _profiles(resources: _Resources, *runtimes: _HostRuntime) -> tuple[tuple[object, ...], ...]:
    """Expand measured host and model data into the existing lookup tuple."""
    return tuple((*runtime, *resources) for runtime in runtimes)


# MobileNetV2 1080p observed margins: -24.133, -7.468 ms.
# Receipt SHA-256 9e8a62ea5eb3f8c831011dedbad5ee94a820768b8fbc9a5f0c61c24fec022f85.
# yunet-n@1 f32-bchw-contiguous 720x1280, native, 1 CPU threads. Request-15 deltas: -1.088, -2.276 ms.
# Receipt SHA-256 e66f20f98e1f21e843775c4869bad2646621b6de20b253dc6f48533293fe8958. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 720x1280, native, 1 CPU threads. Request-15 deltas: -1.790, -4.368 ms.
# Receipt SHA-256 d6dad7e8d15d0aa86483a652cf03caad82aa435b41176b2765f80e6188d61842. NVIDIA driver 595.99.02.
# yunet-n@1 f32-bchw-contiguous 480x640, native, 1 CPU threads. Request-15 deltas: -6.263, -6.719 ms.
# Receipt SHA-256 fe48c53ab136c90edf62a5fb765ca8420ce1e975bcd93c0130d07ca6227841d6. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 480x640, native, 1 CPU threads. Request-15 deltas: -6.401, -7.014 ms.
# Receipt SHA-256 56fec9655513fb4eb3f9b25f5211c220bef2248df41eabd4e0fa3d8a122cba7d. NVIDIA driver 595.99.02.
# yunet-s@1 f32-bchw-contiguous 720x1280, native, 1 CPU threads. Request-15 deltas: -2.077, -1.159 ms.
# Receipt SHA-256 cf3373eaabb1dbe7232f89cbd0f9f97c66697e077866f3bb2d919fdb9d3c7f16. NVIDIA driver 595.99.02.
# yunet-s@1 f32-bchw-contiguous 480x640, native, 1 CPU threads. Request-15 deltas: -7.272, -8.492 ms.
# Receipt SHA-256 f9df516c284f0bdafca0a121139a738f66a987c6f89e083e6e88e339fb98cf38. NVIDIA driver 595.99.02.
# yunet-s@1 u8-chw-rgb-interleaved 480x640, native, 1 CPU threads. Request-15 deltas: -15.523, -7.813 ms.
# Receipt SHA-256 1b6f7c47e38e70eabf19c64d4f42a73f42a2e5f7db2ef8af2b442f555719ec5a. NVIDIA driver 595.99.02.
# yunet-n@1 f32-bchw-contiguous 480x640, cudaMallocAsync, 1 CPU threads. Request-15 deltas: -6.952, -9.328 ms.
# Receipt SHA-256 8215671c70d6c5ba856ba57ce4bd9d10b186ccf80d33f6fcfe253e232bee0300. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 480x640, cudaMallocAsync, 1 CPU threads. Request-15 deltas: -9.117, -6.515 ms.
# Receipt SHA-256 cd34f7c5fe7e7d4dd88cb83ec6796b629883c68256e11f8a382643437a5cc186. NVIDIA driver 595.99.02.
# yunet-n@1 f32-bchw-contiguous 480x640, cudaMallocAsync, 4 CPU threads. Request-15 deltas: -6.323, -7.486 ms.
# Receipt SHA-256 8b0be0566be0dc6e33e42635b34ed7e692085ff24e9cec46d4c9c24a48819e25. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 480x640, cudaMallocAsync, 4 CPU threads. Request-15 deltas: -43.711, -0.576 ms.
# Receipt SHA-256 2c6ed5fda8ac0a337ca8e5c651f61a8eaa83f67a0e397a861c236d9c88df8b7f. NVIDIA driver 595.99.02.
# yunet-s@1 f32-bchw-contiguous 480x640, cudaMallocAsync, 4 CPU threads. Request-15 deltas: -14.376, -8.743 ms.
# Receipt SHA-256 d979397d5056b431158fadd0bfc6efaa39ff82e6e8574484de64e5b1ea14d374. NVIDIA driver 595.99.02.
# yunet-s@1 u8-chw-rgb-interleaved 480x640, cudaMallocAsync, 4 CPU threads. Request-15 deltas: -7.934, -3.811 ms.
# Receipt SHA-256 cdd6af5d1fe8cff45e850a83ae2b4850152031965e8a113cce71cd58b1447490. NVIDIA driver 595.99.02.
APPROVED: dict[MeasuredKey, tuple[tuple[object, ...], ...]] = {
    ("mobilenet_v2.tv2_in1k@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): _profiles(
        _MOBILENET_V2,
        _TORCH_214_NATIVE,
        # cudaMallocAsync. Request-15 deltas: -7.501, -24.823 ms.
        # Receipt SHA-256 1d7d4f18f5b6041d1f19b7e373fc409290b7521243f54b81f0ea2168c7cbeb71.
        _TORCH_214_ASYNC,
    ),
    # CoAtNet-0 with cudaMallocAsync. Request-15 deltas: -57.078, -51.639 ms.
    # Receipt SHA-256 bb27f6ffe6a45231fab544e908d883a8df5494136baad5200b9acc3ee72f0a56.
    ("coatnet_0_rw_224.sw_in1k@1", "kernels", "float16", "f32-bchw-contiguous", (224, 224)): _profiles(
        _COATNET_0, _TORCH_214_ASYNC
    ),
    ("yunet-n@1", "kernels", "float16", "f32-bchw-contiguous", (480, 640)): _profiles(
        _YUNET_N,
        _TORCH_214_NATIVE,
        _TORCH_214_ASYNC,
        _TORCH_214_ASYNC_FOUR_THREADS,
    ),
    ("yunet-n@1", "kernels", "float16", "f32-bchw-contiguous", (720, 1280)): _profiles(_YUNET_N, _TORCH_214_NATIVE),
    ("yunet-n@1", "kernels", "float16", "u8-chw-rgb-interleaved", (480, 640)): _profiles(
        _YUNET_N,
        _TORCH_214_NATIVE,
        _TORCH_214_ASYNC,
        _TORCH_214_ASYNC_FOUR_THREADS,
    ),
    ("yunet-n@1", "kernels", "float16", "u8-chw-rgb-interleaved", (720, 1280)): _profiles(_YUNET_N, _TORCH_214_NATIVE),
    ("yunet-s@1", "kernels", "float16", "f32-bchw-contiguous", (480, 640)): _profiles(
        _YUNET_S,
        _TORCH_214_NATIVE,
        _TORCH_214_ASYNC_FOUR_THREADS,
    ),
    ("yunet-s@1", "kernels", "float16", "f32-bchw-contiguous", (720, 1280)): _profiles(_YUNET_S, _TORCH_214_NATIVE),
    ("yunet-s@1", "kernels", "float16", "u8-chw-rgb-interleaved", (480, 640)): _profiles(
        _YUNET_S,
        _TORCH_214_NATIVE,
        _TORCH_214_ASYNC_FOUR_THREADS,
    ),
    # RetinaFace MobileNet0.25, 480x640 interleaved RGB. Request-15 deltas below.
    ("retinaface-mobilenet0.25@1", "kernels", "float16", "u8-chw-rgb-interleaved", (480, 640)): _profiles(
        _RETINAFACE_MOBILENET,
        # Request-15 deltas: -24.433, -17.322 ms.
        # Receipt SHA-256 550f6d58cea4a694127f65be794165576172e08624755f70f80ce1908f9d6a60.
        _TORCH_214_NATIVE,
        # Request-15 deltas: -10.855, -18.194 ms.
        # Receipt SHA-256 81ba47811b133a0fb8371de9241cb14be2a4453fa6b8a68d25fa4f88a7f32f9b.
        _TORCH_214_ASYNC,
    ),
    # tapnext@1 48-frame 1080p, 256 queries. Request-15 deltas: -576.038, -544.690 ms.
    # Receipt SHA-256 6b90e52baeb980ec1cae8753b3bb28f050c4b2b28b2a4a37e99b8a534c33c1ae.
    (
        "tapnext@1",
        "kernels",
        "float16",
        "tracking-video-queries",
        (("f32-tchw-cuda-contiguous", (48, 3, 1080, 1920)), ("f32-q3-cuda-contiguous", (256, 3))),
    ): _profiles(_TAPNEXT, _TORCH_214_ASYNC_TRITON_FUSION, _TORCH_214_ASYNC),
    # tapnext-pp@1 48-frame 1080p, 256 queries. Request-15 deltas: -387.323, -658.914 ms.
    # Receipt SHA-256 fa7461ab8702aab6dd9b48c060540e56460434d15c06132927505f456a3fa93e.
    (
        "tapnext-pp@1",
        "kernels",
        "float16",
        "tracking-video-queries",
        (("f32-tchw-cuda-contiguous", (48, 3, 1080, 1920)), ("f32-q3-cuda-contiguous", (256, 3))),
    ): _profiles(_TAPNEXT_PP, _TORCH_214_ASYNC_TRITON_FUSION, _TORCH_214_ASYNC),
    # rife-4.25-lite@1 native 384x512 midpoint. Request-15 deltas: -73.780, -66.692 ms.
    # Receipt SHA-256 f1017a4099985e0b113885c84d5230da8373177d9e8c6dc414a205003c4272f7.
    (
        "rife-4.25-lite@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): _profiles(
        _RIFE_425_LITE,
        _TORCH_214_NATIVE,
        # cudaMallocAsync. Request-15 deltas: -71.544, -63.465 ms.
        # Receipt SHA-256 f354455ac995545dc0366179689012afac20e058ac6ffce66df00f3c6fff81b2.
        _TORCH_214_ASYNC,
    ),
    # rife-4.25@1 native 384x512 midpoint. Request-15 deltas: -72.846, -65.045 ms.
    # Receipt SHA-256 fdedd72f8e4247764a8056cfe3a9571e2beb61d68a7c1db671371c254b322083.
    (
        "rife-4.25@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): _profiles(
        _RIFE_425,
        _TORCH_214_NATIVE,
        # cudaMallocAsync. Request-15 deltas: -38.921, -62.616 ms.
        # Receipt SHA-256 701da9ddd9164476ec9da9562801d33e2b8001431b645c9e7882bad3b371b82b.
        _TORCH_214_ASYNC,
    ),
    # rife-4.26@1 native 384x512 midpoint. Request-15 deltas: -68.623, -64.815 ms.
    # Receipt SHA-256 40d2536f1757b35c83f3e4a6dc522f60635025b014c5dd8d07975362557d668a.
    (
        "rife-4.26@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): _profiles(
        _RIFE_426,
        _TORCH_214_NATIVE,
        # cudaMallocAsync. Request-15 deltas: -43.639, -59.427 ms.
        # Receipt SHA-256 cf934ce54c639c2dfa5936ee6d1972b2587fd62fc207ccc15111415f62ca6386.
        _TORCH_214_ASYNC,
    ),
    # RIFE 4.26, ComfyUI 0.46 two-frame RGB input. Public payoff by request 6-9, node by request 4.
    # Public receipt SHA-256 1de4dbc0abc6f428302b298694a5a6b0fe98262a6a09ea954ad770586322ed6a.
    # Public receipt SHA-256 add18d54d40f60fcba029ffadc42ffe1639d0f8f99cee22bf00408a43286eef1.
    # Node receipt SHA-256 accd3c63dfa0d10f1b83066bbd4629190dd30d48ff7345fa9f5cc12d392f071b.
    # Node receipt SHA-256 4beb281295e7d60323ac28fd6969c345979db2dfc30bf97fd06a80c0d142c4d6.
    (
        "rife-4.26@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-adjacent-rgb-pair", (1, 3, 384, 512)), ("f32-bchw-adjacent-rgb-pair", (1, 3, 384, 512))),
    ): _profiles(_RIFE_426, _TORCH_214_ASYNC_FOUR_THREADS),
    # RIFE 4.26, eight-frame ComfyUI window. All four pairs paid back on request 2.
    # Receipt SHA-256 7d65fc5bf1c1b1bf78616ae6bed0f4eaa2fca961344accf77061b88ddd780500.
    # Receipt SHA-256 66e897fe2fb5a70b4804ada18c99b222d91061f67e7c9d0d5b6353e6a600c783.
    (
        "rife-4.26@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-adjacent-rgb-pair", (7, 3, 384, 512)), ("f32-bchw-adjacent-rgb-pair", (7, 3, 384, 512))),
    ): _profiles(_RIFE_426, _TORCH_214_ASYNC_FOUR_THREADS),
    # raft-small@1 default policy. Verified comparison receipts follow.
    # SHA-256 7313aa80840ac7fc447c3e031271d68f446e1ba3ec3df116e4861f4b7d80b474.
    # SHA-256 2c490f272a41aae54a01ef8e907f6ba4baeb93a25da3a41d4a6a64faeff62d23.
    (
        "raft-small@1",
        "kernels",
        "float16",
        "flow-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): (
        *_profiles(_RAFT_SMALL_ALTERNATE, _TORCH_215_ASYNC),
        *_profiles(
            _RAFT_SMALL,
            _TORCH_214_NATIVE,
            # Primary cudaMallocAsync. Request-15 deltas: -39.235, -37.282 ms.
            # Receipt SHA-256 163a2a5d80df7ddebdd635d6e13c9439227673072a854afc52613cb45a9ba193.
            _TORCH_214_ASYNC,
        ),
    ),
    # raft-large@1 default policy. Verified comparison receipts follow.
    # SHA-256 c438b03bea927fe42736897fad11f28307297b330df0fcea4bf0b956f9e71595.
    # SHA-256 601b1c1c5f5d5e5c8e907cf4aa53552d3a9229e9dc7fcb8450eaeda129f2ab97.
    (
        "raft-large@1",
        "kernels",
        "float16",
        "flow-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_RAFT_LARGE, _TORCH_214_NATIVE),
    # crestereo-eth3d@1 default policy. Verified comparison receipts follow.
    # SHA-256 25ae4396d53dc5d9c59db06f280470c6b8b36f05ce255e9b48197e41b5aa7cd5.
    (
        "crestereo-eth3d@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_CRESTEREO_ETH3D, _TORCH_214_NATIVE),
    # DINOv3 ViT-S/16. Request-15 deltas: -18.471, -24.501 ms.
    # Receipt SHA-256 a5f13c681831cf0407ee9b1d2d6bd62c70699bdacc3fe5c59b6dff1145efa48c.
    ("dinov3-vits16@1", "kernels", "float16", "f32-bchw-contiguous", (224, 224)): _profiles(
        _DINOV3_VITS16,
        _TORCH_214_NATIVE,
        # cudaMallocAsync. Request-15 deltas: -14.922, -21.552 ms.
        # Receipt SHA-256 31359d935b3f67dd20248390a424e6d5a53117f9756c3a1841f028c24166fc29.
        _TORCH_214_ASYNC,
    ),
    # DINOv3 ConvNeXt Base. Request-15 deltas: -17.271, -15.698 ms.
    # Receipt SHA-256 3f03004acd403c87509e50ed50e0e0959d115dbfa4958bd046c7ea31b72cfb2b.
    ("dinov3-convnext-base@1", "kernels", "float16", "f32-bchw-contiguous", (224, 224)): _profiles(
        _DINOV3_CONVNEXT_BASE, _TORCH_214_NATIVE
    ),
    # DINOv3 ConvNeXt Small. Request-15 deltas: -17.824, -19.294 ms.
    # Receipt SHA-256 a70293a99b21e39f6cd78f304241fe6bdeafedeb16160a18b4c8d1a0390df78e.
    ("dinov3-convnext-small@1", "kernels", "float16", "f32-bchw-contiguous", (224, 224)): _profiles(
        _DINOV3_CONVNEXT_SMALL, _TORCH_214_NATIVE
    ),
    # DINOv3 ConvNeXt Tiny. Request-15 deltas: -11.659, -9.118 ms.
    # Receipt SHA-256 66a02143d0fe1ac677d78293842545d1ffeb529c78a837f2cc93771344e280f0.
    ("dinov3-convnext-tiny@1", "kernels", "float16", "f32-bchw-contiguous", (224, 224)): _profiles(
        _DINOV3_CONVNEXT_TINY, _TORCH_214_NATIVE
    ),
    # SegNeXt MSCAN-S. Request-15 deltas: -13.554, -23.325 ms.
    # Receipt SHA-256 37b289ec759650893d852131ed7744afae6ae9c7b562ce96c0051468df611922.
    ("segnext-mscan-s-ade20k@1", "kernels", "float16", "f32-bchw-contiguous", (512, 768)): _profiles(
        _SEGNEXT_S, _TORCH_214_NATIVE
    ),
    # NeuFlow v2 Sintel. Request-15 deltas: -110.079, -38.360 ms.
    # Receipt SHA-256 75b3bb15a5ee36eb4a7932c0b6ee4f6864d1e056e5ec01b4af608761f953c2ff.
    (
        "neuflow-v2-sintel@1",
        "kernels",
        "float16",
        "flow-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_NEUFLOW_V2_SINTEL, _TORCH_214_NATIVE),
    # LAS2 Small. Request-15 deltas: -101.184, -99.125 ms.
    # Receipt SHA-256 e190c59c24a9781f469d058efa77187bc9ab78303092b800a5427bf8acb2ae81.
    (
        "las2-s@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_LAS2_S, _TORCH_214_NATIVE),
    # PSMNet Scene Flow. Request-15 deltas: -44.002, -63.959 ms.
    # Receipt SHA-256 b2ca5bf8d7b355002e9adade682a7edd195b71aa742890b5511b1d4b7516216e.
    (
        "psmnet-sceneflow@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 256, 384)), ("f32-bchw-contiguous", (1, 3, 256, 384))),
    ): _profiles(_PSMNET, _TORCH_214_NATIVE),
    # cas-gwcnet-c-sceneflow@1 native 256x384. Request-15 deltas: -55.955, -50.881 ms.
    # Receipt SHA-256 18877052eff0fe6ae0a5246b8959e22ff8f159ed45371c603f55dce3f2f16c6b.
    (
        "cas-gwcnet-c-sceneflow@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 256, 384)), ("f32-bchw-contiguous", (1, 3, 256, 384))),
    ): _profiles(
        (
            ("cfnet_volume@705cd0d85dc1:sm_120", *_IGEV_KERNELS),
            ("9d30ee767111718e87728e09587263d0edecca76319e8cdbdd789946014ce85f",),
        ),
        _TORCH_214_NATIVE,
    ),
    # cas-gwcnet-gc-sceneflow@1 native 256x384. Request-15 deltas: -15.404, -21.287 ms.
    # Receipt SHA-256 9eba385d9549904bac308acc604c5f56d83188c91f95a11ead23b76ef2989ce8.
    (
        "cas-gwcnet-gc-sceneflow@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 256, 384)), ("f32-bchw-contiguous", (1, 3, 256, 384))),
    ): _profiles(
        (
            ("cfnet_volume@705cd0d85dc1:sm_120", *_IGEV_KERNELS),
            ("8a3638501cd5d4cb31d6ec523757134db967db8139715443d83f08913bffa8ab",),
        ),
        _TORCH_214_NATIVE,
    ),
    # las2-l@1 native 384x640. Request-15 deltas: -193.768, -188.643 ms.
    # Receipt SHA-256 49f095e9ce5dd2eba07018dbdd8567683079ba06bb4524225073b38b99a0e152.
    (
        "las2-l@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(
        (
            (*_IGEV_KERNELS, "stereo_ops@899b54f4a755:sm_120"),
            ("12f86f758b0aa6cd59c11d40f41e7d954121d7626e4719eff71d9dd22da36539",),
        ),
        _TORCH_214_NATIVE,
    ),
    # S2M2 Small. Request-15 deltas: -84.104, -94.813 ms.
    # Receipt SHA-256 b0e51da0f31a1d795de87ab926c9585999731b302f2673d3a32f72f1a201346f.
    (
        "s2m2-s@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_S2M2_S, _TORCH_214_NATIVE),
    # U2NetP. Request-15 deltas: -92.701, -88.629 ms.
    # Receipt SHA-256 c6cb651abf7a53c0525398406b9b3f49983680e271a5d6f1d8359e53dfd54b07.
    ("u2netp@1", "kernels", "float16", "f32-bchw-contiguous", (512, 768)): _profiles(_U2NETP, _TORCH_214_NATIVE),
    # U2NetP 1080p. Request-15 deltas: -58.907, -58.595 ms.
    # Receipt SHA-256 d7aa4fb7d54a1f1f5c4e5739e3c50efef91a2ecb28738cdfea173927f8413d2a.
    ("u2netp@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): _profiles(_U2NETP, _TORCH_214_NATIVE),
    # U2Net. Request-15 deltas: -39.335, -39.627 ms.
    # Receipt SHA-256 c14c19ac6ea92626198f17c526dbcc5f03591e436b2f8494ed301b49dddb365a.
    ("u2net@1", "kernels", "float16", "f32-bchw-contiguous", (512, 768)): _profiles(_U2NET, _TORCH_214_NATIVE),
    # EfficientFormerV2-L. Request-15 deltas: -124.620, -127.783 ms.
    # Receipt SHA-256 c378bf2ebe5d41ea90ec7c2210cf7ac90bbf4d7c89e6737c450a2cb040e37241.
    ("efficientformerv2_l.snap_dist_in1k@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): _profiles(
        _EFFICIENTFORMERV2_L, _TORCH_214_NATIVE
    ),
    # SegFormer B5 face parsing. Request-15 deltas: -101.861, -105.848 ms.
    # Receipt SHA-256 9da39708f0b18a57014e8bf764ef5652277cace64c330a919e0313a58caa649d.
    ("segformer-b5-face-parsing@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): _profiles(
        _SEGFORMER_B5_FACE, _TORCH_214_NATIVE
    ),
    # SegNeXt MSCAN-S 1080p. Request-15 deltas: -108.928, -14.113 ms.
    # Receipt SHA-256 d9da8dcc671d775f3c42c9ca03a8e9935e59d41981b25d13034b7b8318309a2e.
    ("segnext-mscan-s-ade20k@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): _profiles(
        _SEGNEXT_S, _TORCH_214_NATIVE
    ),
    # NAFNet GoPro. Request-15 deltas: -113.185, -106.786 ms.
    # Receipt SHA-256 f36b5f7d04d8037a136e062a473fa0e01c4efada8ed10f642202c41fd5171108.
    ("nafnet-gopro@1", "kernels", "float16", "f32-bchw-contiguous", (192, 256)): _profiles(
        _NAFNET_GOPRO, _TORCH_214_NATIVE
    ),
    # HitNet Middlebury with cudaMallocAsync. Request-15 deltas: -56.471, -26.822 ms.
    # Receipt SHA-256 3d90e0be4f6c89105dc84daf3e38c7ac632f2498aa6b597616a2ad40c2890978.
    (
        "hitnet-middlebury-d400@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_HITNET_MIDDLEBURY, _TORCH_214_ASYNC),
    # RTMPose-T, five person boxes. Request-15 deltas: -15.324, -10.711, -10.343, -12.267 ms.
    # Receipt SHA-256 c973de6f90dc63d64c6f0b89b4a2f584282a63cae54d6a51639bc11a41486a21.
    (
        "rtmpose-t@1",
        "kernels",
        "float16",
        "pose-boxes",
        (("f32-chw-contiguous", (3, 480, 640)), ("f32-p4-cuda-contiguous", (5, 4))),
    ): _profiles(_RTMPOSE_T, _TORCH_214_NATIVE),
    # NeuFlow v2 Things. Request-15 deltas: -95.982, -79.328 ms.
    # Receipt SHA-256 a4895ec1f3b4361f427dcef3a1f73103870c0b4ad44691690aa8026c03d7b3dc.
    (
        "neuflow-v2-things@1",
        "kernels",
        "float16",
        "flow-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_NEUFLOW_V2_THINGS, _TORCH_214_NATIVE),
    # GwcNet-gc Scene Flow. Request-15 deltas: -51.541, -45.335 ms.
    # Receipt SHA-256 b5587a77286a23c3272088b6d6e4053df82657b737609d671ef664b36c311078.
    (
        "gwcnet-gc-sceneflow@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 256, 384)), ("f32-bchw-contiguous", (1, 3, 256, 384))),
    ): _profiles(_GWCNET_GC, _TORCH_214_NATIVE),
    # LAS2 Medium. Request-15 deltas: -112.657, -147.702 ms.
    # Receipt SHA-256 ce2b928f8d4fefc9c58b832f67627f082d768c3df9e2ca9e220fcbd899d97465.
    (
        "las2-m@1",
        "kernels",
        "float16",
        "stereo-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 640)), ("f32-bchw-contiguous", (1, 3, 384, 640))),
    ): _profiles(_LAS2_M, _TORCH_214_NATIVE),
    # BiSeNet face parsing. Request-15 deltas: -12.000, -8.378 ms.
    # Receipt SHA-256 2e86865e0be77f1950e0354ae42b2834d83a658de7d3846637db94c0ec993819.
    ("bisenet-face-parsing@1", "kernels", "float16", "f32-bchw-contiguous", (512, 768)): _profiles(
        _BISENET_FACE_PARSING, _TORCH_214_NATIVE
    ),
}
