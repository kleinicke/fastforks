"""Measured Linux CUDA graph eligibility for selected workloads."""

from typing import TypeAlias

TensorSignature: TypeAlias = tuple[str, tuple[int, ...]]
MeasuredKey: TypeAlias = tuple[str, str, str, str, tuple[int, int] | tuple[TensorSignature, ...]]

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
    ("mobilenet_v2.tv2_in1k@1", "kernels", "float16", "f32-bchw-contiguous", (1080, 1920)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f",),
        ),
    ),
    ("yunet-n@1", "kernels", "float16", "f32-bchw-contiguous", (480, 640)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            4,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
    ),
    ("yunet-n@1", "kernels", "float16", "f32-bchw-contiguous", (720, 1280)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
    ),
    ("yunet-n@1", "kernels", "float16", "u8-chw-rgb-interleaved", (480, 640)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            4,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
    ),
    ("yunet-n@1", "kernels", "float16", "u8-chw-rgb-interleaved", (720, 1280)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06",),
        ),
    ),
    ("yunet-s@1", "kernels", "float16", "f32-bchw-contiguous", (480, 640)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            4,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",),
        ),
    ),
    ("yunet-s@1", "kernels", "float16", "f32-bchw-contiguous", (720, 1280)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",),
        ),
    ),
    ("yunet-s@1", "kernels", "float16", "u8-chw-rgb-interleaved", (480, 640)): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            4,
            (
                "box_nms@daa010d75e1a:sm_120",
                "cudnn_hacc@cudnn92400",
                "fvm_conv@f1d064e539e1:sm_120",
                "fvm_extra@3208ce525a91:sm_120",
                "fvm_mb@720dd3f29bf7:sm_120",
                "fvm_r3@f54eb890e23c:sm_120",
            ),
            ("f2688fecd5aef7d204d04ca12ff377b724978abbe5955740bc67280d99ffe4d6",),
        ),
    ),
    # tapnext@1 48-frame 1080p, 256 queries. Request-15 deltas: -576.038, -544.690 ms.
    # Receipt SHA-256 6b90e52baeb980ec1cae8753b3bb28f050c4b2b28b2a4a37e99b8a534c33c1ae.
    (
        "tapnext@1",
        "kernels",
        "float16",
        "tracking-video-queries",
        (("f32-tchw-cuda-contiguous", (48, 3, 1080, 1920)), ("f32-q3-cuda-contiguous", (256, 3))),
    ): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            "1",
            "4",
            1,
            ("fvm_conv@f1d064e539e1:sm_120", "tapnext_state@be3777b468c9:sm_120"),
            ("19de6b1495555c2afb4b56e823a3f3c2fd7236eea05c53d98feab3cccd1b4a28",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            ("fvm_conv@f1d064e539e1:sm_120", "tapnext_state@be3777b468c9:sm_120"),
            ("19de6b1495555c2afb4b56e823a3f3c2fd7236eea05c53d98feab3cccd1b4a28",),
        ),
    ),
    # tapnext-pp@1 48-frame 1080p, 256 queries. Request-15 deltas: -387.323, -658.914 ms.
    # Receipt SHA-256 fa7461ab8702aab6dd9b48c060540e56460434d15c06132927505f456a3fa93e.
    (
        "tapnext-pp@1",
        "kernels",
        "float16",
        "tracking-video-queries",
        (("f32-tchw-cuda-contiguous", (48, 3, 1080, 1920)), ("f32-q3-cuda-contiguous", (256, 3))),
    ): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            "1",
            "4",
            1,
            ("fvm_conv@f1d064e539e1:sm_120", "tapnext_state@be3777b468c9:sm_120"),
            ("cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da",),
        ),
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            ("fvm_conv@f1d064e539e1:sm_120", "tapnext_state@be3777b468c9:sm_120"),
            ("cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da",),
        ),
    ),
    # rife-4.25-lite@1 native 384x512 midpoint. Request-15 deltas: -73.780, -66.692 ms.
    # Receipt SHA-256 f1017a4099985e0b113885c84d5230da8373177d9e8c6dc414a205003c4272f7.
    (
        "rife-4.25-lite@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            ("cudnn_hacc@cudnn92400",),
            ("81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1",),
        ),
    ),
    # rife-4.25@1 native 384x512 midpoint. Request-15 deltas: -72.846, -65.045 ms.
    # Receipt SHA-256 fdedd72f8e4247764a8056cfe3a9571e2beb61d68a7c1db671371c254b322083.
    (
        "rife-4.25@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            ("cudnn_hacc@cudnn92400",),
            ("6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",),
        ),
        # cudaMallocAsync. Request-15 deltas: -38.921, -62.616 ms.
        # Receipt SHA-256 701da9ddd9164476ec9da9562801d33e2b8001431b645c9e7882bad3b371b82b.
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            ("cudnn_hacc@cudnn92400",),
            ("6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",),
        ),
    ),
    # rife-4.26@1 native 384x512 midpoint. Request-15 deltas: -68.623, -64.815 ms.
    # Receipt SHA-256 40d2536f1757b35c83f3e4a6dc522f60635025b014c5dd8d07975362557d668a.
    (
        "rife-4.26@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, 384, 512)), ("f32-bchw-contiguous", (1, 3, 384, 512))),
    ): (
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "native",
            None,
            None,
            None,
            "4",
            1,
            ("cudnn_hacc@cudnn92400",),
            ("45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",),
        ),
        # cudaMallocAsync. Request-15 deltas: -43.639, -59.427 ms.
        # Receipt SHA-256 cf934ce54c639c2dfa5936ee6d1972b2587fd62fc207ccc15111415f62ca6386.
        (
            "2.14.1+cu130",
            "NVIDIA GeForce RTX 5060 Ti",
            "cudaMallocAsync",
            None,
            None,
            None,
            "4",
            1,
            ("cudnn_hacc@cudnn92400",),
            ("45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",),
        ),
    ),
}
