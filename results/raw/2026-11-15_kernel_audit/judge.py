"""Judgement table for the families without kernels (hand-written from the measured numbers) -> writes the md."""
import csv, collections, subprocess
# family: (verdict, users 1-5, expected kernels gain vs portable (x), effort, note)
J = {
 'lightglue': ('NOT real', 5, 1.6, 'M-H', 'portable only 1.015x / 1.0x mem vs original (aliked 40.6 vs 41.2 ms, disk 116.4 vs 116.7); portable runs fp32 -> fp16 convs of DISK U-Net / ALIKED + fp16 attention are untouched; counts are data-dependent only in the post-processing, not in the dense nets'),
 'loftr': ('NOT real', 4, 1.3, 'M', 'portable already 2.2x (110 vs 241 ms), 0.46x mem; "data-dependent match counts" only concerns the final mutual-NN step; the ResNet-FPN + 8 linear-attention blocks are dense and fusable'),
 'grounding_dino': ('PARTLY real', 4, 1.4, 'H', 'measured: fp16-accumulate GEMMs alone gain 0-3 %; the remaining time is multi-scale deformable attention (grid_sample) + small ops, which needs a MSDA kernel + CUDA graphs; portable 1.45x / 0.60x mem already'),
 'codeformer': ('NOT real', 4, 1.4, 'M', 'portable 48.6 vs 85.7 ms (1.76x), 0.53x mem; VQGAN 3x3 convs + GroupNorm dominate -> cudnn_hacc / cutlass_conv + fused GroupNorm epilogue exist in the lib (used by other restoration families)'),
 'waft': ('NOT real', 2, 1.8, 'L', 'portable 790 vs 1533 ms at 1080x1920; the DINOv2-S + DPT feature net uses the same fastbb + dpt_fused kernels the depth_anything_v2 family already has (the card says so); the 5 refinement ViT iterations reuse the timm ViT blocks'),
 'dedode': ('NOT real', 3, 1.3, 'M-H', '"not implemented yet"; portable 1.29x / 0.92x mem (G: 592 vs 666 ms); DeDoDe-G is a DINOv2-L ViT + VGG decoders: fastbb/dpt_fused ViT path + cudnn_hacc convs fit'),
 'loma': ('NOT real', 2, 1.3, 'H', '"not implemented yet"; no speed rows at all yet (gap-fill queue first); DINOv2-G + decoder, large'),
 'flexglue': ('NOT real', 2, 1.4, 'M', 'same dense net / data-dependent post-processing split as LightGlue; no speed rows yet'),
 'xfeat': ('PARTLY real', 3, 1.1, 'M', 'small CNN (8.2 ms extractor); portable 1.14x; the per-point steps are data dependent: low expected gain; lighterglue 22 ms is LightGlue-like (follows the lightglue work)'),
 'matanyone': ('NOT real', 2, 1.3, 'H', 'portable only 1.01x / 0.91x mem (1205 vs 1220 ms at 1080x1920); ResNet encoders + memory readout + object transformer: dense ops, nothing measured; large effort for 2-user family'),
 'flowformer': ('FIX PORTABLE FIRST', 2, 1.4, 'H', 'portable fails the bar (351 vs 303 ms, 3165 vs 2382 MB); the latent-cost-memory attention needs a dedicated kernel, real work'),
 'gmflow': ('NOT real', 3, 1.2, 'M', 'portable ties the original (0.94-0.998x speed on 8 of 9 refs; 1.07x mem at 375x1242); window attention + global softmax matching is plain attention (fastbb / cuBLAS fp16) and the matching softmax is fusable'),
 'flowseek': ('NOT real', 1, 1.3, 'M', 'portable 1.16x / 1.0x mem; the card says the SEA-RAFT engine + DA kernels fit (both exist)'),
 'dedode_dummy': None,
 'parseq': ('PARTLY real', 2, 1.3, 'L', '16 ms; ViT on 128 tokens + 26-step AR decoder: launch-bound -> a CUDA-graph decode (shared cuda_graphs.py) is a cheap, measurable win; not custom kernels'),
 'ppocrv5': ('PARTLY real', 4, 1.1, 'M', 'portable 2.76x already; cuDNN convs + CPU box geometry; only cudnn_hacc fp16 convs left to try (untried)'),
 'coatnet': ('NOT real', 2, 1.3, 'M', '10 refs, 14 ms at 224; MBConv + relative-position attention; classifier kernels runners exist (convnext/vit/repvit/...) and give 1.2-1.5x vs portable'),
 'focalnet': ('NOT real', 2, 1.3, 'M', '6 refs, 17 ms; same classifier-runner argument'),
 'hgnetv2': ('NOT real', 2, 1.3, 'M', '7 refs, 7 ms; plain CNN: cudnn_hacc / cutlass_conv runner like the CNN classifier families'),
 'davit': ('NOT real', 1, 1.3, 'M', '3 refs; portable 1.095x only'),
 'shvit': ('NOT real', 1, 1.2, 'M', '4 refs, 9 ms; portable 1.24x'),
 'efficientformer': ('NOT real', 1, 1.2, 'M', 'portable mem 1.07x above the original (fails the bar: 93 vs 87 MB), 3 refs'),
 'efficientformerv2': ('NOT real', 1, 1.2, 'M', '4 refs, 15 ms'),
 'bisenet': ('REAL', 2, 1.0, '-', 'measured slower (cudnn_hacc 2.77 vs 2.41 ms portable at 512x512; 2.2 ms model)'),
 'inspyrenet': ('REAL', 3, 1.0, '-', 'measured slower (cudnn_hacc 189.9 vs 159.4 ms; fp16-accumulate GEMMs alone 3 %)'),
 'hed': ('UNMEASURED', 3, 1.2, 'L', 'cuDNN VGG CNN; portable already 3.4x (11.3 vs 38.5 ms); the "no kernel measured to pay off" claim has no number: one cudnn_hacc A/B decides'),
 'lineart': ('UNMEASURED', 3, 1.2, 'L', 'portable 1.6x, mem 1.005x of the original (fails the memory bar by 0.5 %); small conv generator; same A/B'),
 'lineart_anime': ('UNMEASURED', 2, 1.2, 'L', 'pix2pix U-Net, 24 ms; portable 2.9x; same A/B'),
 'anyline': ('REAL-ish (tiny CNN + GPU blend)', 2, 1.1, '-', 'portable 12.5x already; no numbers on the kernel side'),
 'pidinet': ('REAL', 2, 1.0, '-', 'few ms plain convs, portable 2.8x / 0.97x mem'),
 'mlsd': ('REAL', 2, 1.0, '-', 'MobileNetV2 small CNN, portable 3.8x'),
 'teed': ('REAL', 2, 1.0, '-', '58 k parameters, few ms, portable 7.2x'),
}
J.pop('dedode_dummy')
fam = {r['family']: r for r in csv.DictReader(open('audit_families.csv'))}
rows = []
for f, (verdict, users, gain, effort, note) in J.items():
    r = fam[f]
    rows.append(dict(family=f, task=r['task'], n_refs=r['n_refs'], verdict=verdict, users_1_5=users, expected_gain_vs_portable=gain,
                     effort=effort, score=round(users * (gain - 1) * 10, 1), med_port_speedup_vs_orig=r['med_port_speedup'],
                     med_port_mem_ratio=r['med_port_mem_ratio'], note=note, card_reason=r['no_kernels_reason']))
rows.sort(key=lambda r: -r['score'])
w = csv.DictWriter(open('no_kernels_ranking.csv', 'w'), list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows)); [print(r['family'], r['verdict'], r['score']) for r in rows[:12]]
