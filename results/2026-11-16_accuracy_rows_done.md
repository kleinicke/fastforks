# Accuracy rows for the 2026-11-15 weights-inventory gaps (2026-11-16)

GPU RTX 5060 Ti, torch 2.14.1+cu130, date 2026-11-16, code commit 33b2432c (src unchanged since; docs commits on top).
Raw: results/raw/2026-11-16_accuracy_rows/ (scripts q1-q6.sh, env.sh, eval outputs, build_rows.py, logs). All GPU steps ran in
capped units (`review/cpuslot --gpu systemd-run -p MemoryMax=20G ... gpulock-low`). Rows: docs/data/*_eval.jsonl, speed_compile.jsonl.

| model | metric (dataset) | original | portable | kernels | note |
|---|---|---|---|---|---|
| nnfoundation-cnn | feature cos (worst) / rel L2, 8 TotalSeg cases | fp16: 0.99999 / 0.0014 | 0.99999 / 0.0017 | 0.99999 / 0.0015 | vs original fp32; raw 2026-11-14 |
| nnfoundation-vit | same | fp16: 0.99947 / 0.0167 | 0.99950 / 0.0165 | 0.99940 / 0.0213 | portable no worse than original fp16 |
| nninteractive | Dice vs GT, 8 cases x 12 prompts | 0.9688 (fp16) | 0.9688 (agree 0.9999) | 0.9688 (agree 0.9999) | raw 2026-11-14/15 |
| pe-core-g14-448 | ImageNet kNN top-1 %, 10k train / 5k test, k=20 | 60.42 fp32 / 60.48 fp16 | 60.40 | 60.48 | PE-Core-L14-336 gets 65.98 in the same protocol; no paper number for this kNN protocol (paper: zero-shot 87 %) |
| prompt-depth-anything-l | metric AbsRel / d1, NYU every 4th (164) | 0.0217 / 0.9876 | 0.0208 / 0.9883 | 0.0208 / 0.9883 | prompt = 192x256 GT cell means; not comparable to paper (ARKit/ScanNet++ L1) |
| prompt-depth-anything-l | iBims (100) | 0.0150 / 0.9930 | 0.0147 / 0.9932 | 0.0147 / 0.9932 | |
| seedvr2-3b | DIV2K crops x4 (512 px, x2): PSNR / SSIM Y; PSNR to original | 22.02 / 0.5865 | 22.00 / 0.5863; 43.13 dB | same | bicubic baseline 29.01 / 0.8075; clips PSNR to original 44.56 dB |
| seedvr2-7b | 2 crops | 23.16 / 0.6228 | 23.13 / 0.6227; 44.85 dB | same | bicubic 31.03; clips 46.14 dB; original run with --offload; portable runs with stream=True (not OOM) |
| wd-eva02-large-tagger-v3 | tag F1 at 0.35 vs orig fp32 / top-1 agree, 500 COCO images | fp16 99.93 / 99.8 | 99.84 / 99.8 | 99.84 / 99.4 | |
| wd-vit-large-tagger-v3 | same | 99.93 / 99.8 | 99.87 / 99.4 | 99.88 / 99.8 | (also wd-vit, wd-convnext rows) |

Notes
- SeedVR2 is generative: PSNR to the GT (22-23 dB) is below plain bicubic (29-31 dB), as for the original; the useful check is PSNR to the original (43-46 dB), consistent with the 44-45 dB of 2026-11-14. Portable and kernels give identical numbers (same output). LPIPS not computed: package not installed (reason is in the card's protocol text).
- Taggers have no public benchmark with their tag set: the original fp32 is the reference (tag F1 / agreement), the original fp16 row shows the original's own half-precision drift.
- Not comparable to a paper: PromptDA (paper evaluates ARKitScenes/ScanNet++), PE-core kNN, nnFoundation/nnInteractive (encoder parity / Dice on our subset; nnInteractive paper reports interactive Dice curves on other sets).
- pointdit-l-512 (994x1482, fastforks/.venv with onnx + TensorRT 10): compile 76.0 ms, first call 57 s (portable 94.6, kernels 61.4); TensorRT fp16 49.1 ms (engine build 79 s + ONNX export 71 s), peak reserved 2626 MiB. Rows in docs/data/speed_compile.jsonl. Also re-ran pointdit-b-512 TensorRT, whose old row failed with "onnx is not installed" (wrong venv): now 15.5 ms. First tries are kept as logs: q5_first_try_wrongdata.log (wt-ports venv: no onnx, synthetic input), q5_second_try_synthetic.log (FASTFORKS_DATA unset: synthetic input).
- Commands: see q1.sh (eval_features.py + acc_collect.py --write --task features), q2.sh (agree_wd.py), q3.sh (tools/eval_prompt_depth.py, new), q4.sh (seedvr2_reference.py + seedvr2_eval.py), q5.sh / q6.sh (speed_check.py run --modes compile tensorrt), build_nn_rows.py, build_rows.py. Docs: tools/gen_docs.py; tests/test_docs.py test_policy.py test_card_sync.py: 1867 passed, 62 skipped (slow).
