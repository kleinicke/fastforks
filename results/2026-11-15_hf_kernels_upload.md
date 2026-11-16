# HF kernels upload (2026-11-15)

Repo fastforks/kernels (existing). Commit message: 'kernels: add gsraster, nf4, raft_flow, ransac, romav2, vol_norm'. Only added files; nothing deleted or overwritten; README untouched.
Method: HEAD of every manifest*.toml [files] URL (anonymous), 42 returned 404; built files found in worktree build/kernels dirs, size + sha256 matched the manifests; uploaded with HfApi.upload_folder (token via systemd EnvironmentFile).

Skipped: none with hash mismatch. dpt_fused (48e0065bb7cf) was already on HF (not 404).

Verification: re-HEAD all manifest URLs: 0 missing; downloaded all 42 files anonymously, size + sha256 match. Load smoke not run (no GPU/review tooling in this worktree).

| lib | file | size | sha256 |
|---|---|---|---|
| gsraster | gsraster-be9d99ba7071-sm_80.fatbin | 65520 | 989414c240631cb04047d446d0e3194370097c542018d65801f0819552562a4d |
| gsraster | gsraster-be9d99ba7071-sm_86.fatbin | 65648 | cad0fec579484bd976ec212b10250a4f4366f4a075654d7a4e5ddeb17a2c51e7 |
| gsraster | gsraster-be9d99ba7071-sm_89.fatbin | 65648 | c7854c8e784ac7ac6a5c834055750993f664e3c984750c6ef2e359f24f022f2c |
| gsraster | gsraster-be9d99ba7071-sm_90.fatbin | 69376 | 0df5c58c29a9687ed8661fa6deec8335536b7e8383f1742cad8f24dfc50b1fc1 |
| gsraster | gsraster-be9d99ba7071-sm_100.fatbin | 79888 | 5fef227612051e99ed545a90b1713fa89ce388c354a98bdbb17c2b4de005c15f |
| gsraster | gsraster-be9d99ba7071-sm_120.fatbin | 79704 | 86b1e7c4df1abccb96477c78ce92119da55f53961606f641a7e3ec8932336c5a |
| gsraster | gsraster-be9d99ba7071-compute_80.fatbin | 17448 | 43284e1dc2e560c22bdb01898a511cddd708272e97d18e65e9ac13ef1433d1c0 |
| nf4 | nf4-e236d80ba78b-sm_80.fatbin | 31736 | fb222ca4fad5ff788c66d42b7b30643517504fdcc1a70078d6cbef2f5496e2d3 |
| nf4 | nf4-e236d80ba78b-sm_86.fatbin | 31736 | dadffb13ef6409b278ecdca52172170591b4b4c4bc9a5e9586c3b1b36b17c6ef |
| nf4 | nf4-e236d80ba78b-sm_89.fatbin | 31608 | 976ee9cf6b5d647a39bf2844b6656ef4c453e6879803d87b01d4a36210cac0ba |
| nf4 | nf4-e236d80ba78b-sm_90.fatbin | 34304 | 1fdf9d59ced2bd0621e3161891916dc6f3d810a12c1716042a682d96555650d4 |
| nf4 | nf4-e236d80ba78b-sm_100.fatbin | 73176 | 974d41dbc20cc1e6732021f1b9670496ad99794129d16e52e1dbceda88e09c4b |
| nf4 | nf4-e236d80ba78b-sm_120.fatbin | 70880 | 0f20bf661ada854b72467d3ee7462d220daf8aff957e2be68b861fbd521c8b4e |
| nf4 | nf4-e236d80ba78b-compute_80.fatbin | 6512 | c03549af08c87b2d15c6c29afc2e589816bf141af29f0ecfa7cc0d364d8794b3 |
| raft_flow | raft_flow-a101a33607b6-sm_80.fatbin | 532656 | 9fcf7712dbf4532147d8f77a530a193d56b9a52a310df5b46ff05fa5dca81328 |
| raft_flow | raft_flow-a101a33607b6-sm_86.fatbin | 533936 | 4ea1b5ef568736c33a93a983e89c6013bd93a4c75faa9c2c2abf5ad7d9a2cc70 |
| raft_flow | raft_flow-a101a33607b6-sm_89.fatbin | 533808 | f8fe4fc6832b5e91a9756e3f230a5cc0946efb40aeaaa51dec408198cc0cce04 |
| raft_flow | raft_flow-a101a33607b6-sm_90.fatbin | 548368 | 6eabfa33ee8d252766afb56ada6cc75823fba528d4607c95ddb4eaac6829d9d5 |
| raft_flow | raft_flow-a101a33607b6-sm_100.fatbin | 610936 | 663d50f436287298262246b34c0e7cef50deb9d94f53b244b928c1c312eade54 |
| raft_flow | raft_flow-a101a33607b6-sm_120.fatbin | 614920 | 80c0f490947d9e979f04d83a318b50a7980a71b90db3c93c71c306f8e08bc9d6 |
| raft_flow | raft_flow-a101a33607b6-compute_80.fatbin | 216736 | ece1436dae500cb75c5f3ce800df2a0c661c3a93225e5051021b905adfc25a26 |
| ransac | ransac-c0033a9074a6-sm_80.fatbin | 617696 | 611b52950881644b7bff652cf449ffc991bf4e9b52dcddde2ec4524ad4023706 |
| ransac | ransac-c0033a9074a6-sm_86.fatbin | 617952 | 0cdee8b34c0609e11340e1b018d37498b5bae7db6016247ca3855513269463a7 |
| ransac | ransac-c0033a9074a6-sm_89.fatbin | 617952 | d54f11d90f10c53695a0fc409d546ee783663ba5ba02a58f877b2bd803bf6dbc |
| ransac | ransac-c0033a9074a6-sm_90.fatbin | 632256 | ae90bea886515100258fb2c36d1770179a0f61aaac6b9be4135bf3d62e38f8d9 |
| ransac | ransac-c0033a9074a6-sm_100.fatbin | 1030872 | a7e7292a8b50b53cf90cac1ca25fa332d8096c04eb01fa442eb0f0204fa6a67d |
| ransac | ransac-c0033a9074a6-sm_120.fatbin | 1592288 | 6a354b7f2e767c19b4e479feaa83ccd2e12f36ce866a69e2267b2d6393dd93b0 |
| ransac | ransac-c0033a9074a6-compute_80.fatbin | 241672 | f004b9163da89243b39e7f716c04f39779a33c97e29f97ffd947db0aac41f021 |
| romav2 | romav2-bb29839cee98-sm_80.fatbin | 101168 | 7441276899fc8c4721063494af5ba86d6a88c0e62eb32643d54f4c45ddc2591a |
| romav2 | romav2-bb29839cee98-sm_86.fatbin | 101552 | a4dc8a3b2504e570a74e0ffa74b251adcbf7a041f4f488f8372a56fe079b08c1 |
| romav2 | romav2-bb29839cee98-sm_89.fatbin | 101424 | f9e7f5eb361424faa0ec5798f29464b23ea6b041fb00aa662dfc0650a722ce48 |
| romav2 | romav2-bb29839cee98-sm_90.fatbin | 110504 | 069d70ca07092dd2541e6e8ff79598c8b07a9c6cdfd9bb7b3cae71a11ef4ec4b |
| romav2 | romav2-bb29839cee98-sm_100.fatbin | 117912 | 007b2cd61ae9815a8430d04ab515908a80e055dbd365619d6734ceb985187451 |
| romav2 | romav2-bb29839cee98-sm_120.fatbin | 124232 | 95052a740526176163433fe2c8fea41a7f57c2f68f76b7aca683daf49d3f2b79 |
| romav2 | romav2-bb29839cee98-compute_80.fatbin | 44304 | f71d32a977465bdb12d5600dad41d704c7b170b963fefd209b57b68d448a2cc9 |
| vol_norm | vol_norm-44036d4da567-sm_80.fatbin | 18800 | 3599e10f6816c450ad22e801ff1c041538c7f3422ef1b825a37fed340548cd47 |
| vol_norm | vol_norm-44036d4da567-sm_86.fatbin | 18928 | d2050ef1c41116141716b642cbcd10fa02c07c0c39a33d1aa782c2b43ab902b2 |
| vol_norm | vol_norm-44036d4da567-sm_89.fatbin | 18928 | 4bb22dd49aaa18959f2512ab3526fa02ad566954af021987312581f8d63912a8 |
| vol_norm | vol_norm-44036d4da567-sm_90.fatbin | 20616 | 5723bf9a9b162cac9dbf96f4f60581d763b592f755011bbc1faccbf89c872be7 |
| vol_norm | vol_norm-44036d4da567-sm_100.fatbin | 37752 | 33c3c266ae8c7b3a6c5dc1e586d78f56654411724fb653c774e6bc3288f6ea2e |
| vol_norm | vol_norm-44036d4da567-sm_120.fatbin | 37904 | 26d52540a5fb037793d8ff5625649afe4892424aeaf699523fd76c2b064f0073 |
| vol_norm | vol_norm-44036d4da567-compute_80.fatbin | 4584 | e25a469444be847022973a63b9c4e28dca73ed296d0ed53bf76e254c932d5f3c |
