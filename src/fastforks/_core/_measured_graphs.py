"""Measured Linux CUDA graph eligibility for selected workloads."""

# MobileNetV2 1080p observed margins: -24.133, -7.468 ms.
# Receipt SHA-256 9e8a62ea5eb3f8c831011dedbad5ee94a820768b8fbc9a5f0c61c24fec022f85.
# yunet-n@1 f32-bchw-contiguous 720x1280. Request-15 deltas: -1.088, -2.276 ms.
# Receipt SHA-256 e66f20f98e1f21e843775c4869bad2646621b6de20b253dc6f48533293fe8958. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 720x1280. Request-15 deltas: -1.790, -4.368 ms.
# Receipt SHA-256 d6dad7e8d15d0aa86483a652cf03caad82aa435b41176b2765f80e6188d61842. NVIDIA driver 595.99.02.
# yunet-n@1 f32-bchw-contiguous 480x640. Request-15 deltas: -6.263, -6.719 ms.
# Receipt SHA-256 fe48c53ab136c90edf62a5fb765ca8420ce1e975bcd93c0130d07ca6227841d6. NVIDIA driver 595.99.02.
# yunet-n@1 u8-chw-rgb-interleaved 480x640. Request-15 deltas: -6.401, -7.014 ms.
# Receipt SHA-256 56fec9655513fb4eb3f9b25f5211c220bef2248df41eabd4e0fa3d8a122cba7d. NVIDIA driver 595.99.02.
# yunet-s@1 f32-bchw-contiguous 720x1280. Request-15 deltas: -2.077, -1.159 ms.
# Receipt SHA-256 cf3373eaabb1dbe7232f89cbd0f9f97c66697e077866f3bb2d919fdb9d3c7f16. NVIDIA driver 595.99.02.
# yunet-s@1 f32-bchw-contiguous 480x640. Request-15 deltas: -7.272, -8.492 ms.
# Receipt SHA-256 f9df516c284f0bdafca0a121139a738f66a987c6f89e083e6e88e339fb98cf38. NVIDIA driver 595.99.02.
# yunet-s@1 u8-chw-rgb-interleaved 480x640. Request-15 deltas: -15.523, -7.813 ms.
# Receipt SHA-256 1b6f7c47e38e70eabf19c64d4f42a73f42a2e5f7db2ef8af2b442f555719ec5a. NVIDIA driver 595.99.02.
ENROLLED = frozenset([("classification", "mobilenetv2"), ("pose", "yunet")])
APPROVED = {
    ("mobilenet_v2.tv2_in1k@1", "float16", "f32-bchw-contiguous", (1080, 1920)): (
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
    ("yunet-n@1", "float16", "f32-bchw-contiguous", (480, 640)): (
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
    ("yunet-n@1", "float16", "f32-bchw-contiguous", (720, 1280)): (
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
    ("yunet-n@1", "float16", "u8-chw-rgb-interleaved", (480, 640)): (
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
    ("yunet-n@1", "float16", "u8-chw-rgb-interleaved", (720, 1280)): (
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
    ("yunet-s@1", "float16", "f32-bchw-contiguous", (480, 640)): (
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
    ("yunet-s@1", "float16", "f32-bchw-contiguous", (720, 1280)): (
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
    ("yunet-s@1", "float16", "u8-chw-rgb-interleaved", (480, 640)): (
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
}
