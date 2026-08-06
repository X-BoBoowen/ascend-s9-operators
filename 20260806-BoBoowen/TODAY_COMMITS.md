# 2026-08-06 Git 提交清单

时区：Asia/Shanghai

基准：`3447a6beec615ee18ce63104b43a69027d6b1322`

终点：`aa7217a835b49f37b8f210cc785231a80bf81f24`

- `61330ea` `11:37:33` docs(squaresum): record S02BI official result
- `391ddc9` `11:40:50` perf(squaresum): compact small middle rows safely
- `562900b` `11:50:22` perf(squaresum): align packed small-middle phases
- `9ee4464` `12:04:32` docs(squaresum): publish S02BK cloud gate
- `f2b80fe` `12:05:12` chore(artifact): record S02BK cloud evidence
- `27a744c` `12:08:23` test(squaresum): sweep workspace threshold cliff
- `6951f39` `12:10:09` perf(squaresum): lower narrow middle workspace threshold
- `0c66fd7` `12:12:33` test(squaresum): measure narrow workspace crossover
- `de566c8` `12:14:49` perf(squaresum): split packed workspace thresholds
- `5cfc7fe` `12:25:40` docs(squaresum): publish S02BM cloud gate
- `0f00129` `12:26:07` chore(artifact): record S02BM cloud evidence
- `af57972` `12:33:13` perf(squaresum): split noncontiguous long tails
- `8aba0d7` `12:48:09` test(squaresum): cover long-tail split-k structure
- `1d301e5` `12:52:04` docs(squaresum): publish S02BN cloud gate
- `ccd4ae9` `12:52:55` docs(squaresum): record S02BN release snapshot
- `11519a6` `12:53:19` chore(artifact): record S02BN cloud evidence
- `87e25fa` `12:58:30` perf(squaresum): saturate narrow strided cores
- `dfa2d4b` `13:03:09` perf(squaresum): refine narrow strided scheduling
- `0ed9c2d` `13:14:32` perf(squaresum): group narrow strided output rows
- `381c11a` `13:20:35` fix(squaresum): reuse packaged key for grouped rows
- `b2c954a` `13:25:38` fix(squaresum): align grouped scalar row reduction
- `ae56661` `13:29:30` test(squaresum): cover grouped scalar row boundaries
- `ca9903d` `13:37:54` test(squaresum): benchmark grouped scalar scale crossover
- `3848f8e` `13:42:31` docs(squaresum): publish S02BS cloud gate
- `36f0f52` `13:43:08` docs(squaresum): correct S02BS gate count
- `15a824d` `13:43:52` chore(artifact): record S02BS cloud evidence
- `e25c495` `13:51:52` perf(squaresum): add S02BT adaptive padded row grouping
- `1a18e3d` `13:57:24` fix(squaresum): pack S02BT outputs with aligned scalar stores
- `88102e6` `14:01:25` test(squaresum): cover S02BT adaptive grouping boundaries
- `b4979fd` `14:02:35` test(squaresum): keep S02BT boundary cases within rank limit
- `c29a233` `14:13:18` perf(squaresum): widen S02BT command-bound row groups
- `91044b4` `14:16:00` perf(squaresum): balance S02BT grouping by inner width
- `f29006c` `14:23:37` docs(squaresum): record S02BT cloud evidence
- `2d1b2db` `14:24:53` chore(artifact): record S02BT final cloud evidence
- `2f6b35a` `14:30:37` perf(squaresum): add S02BU strided grouped split-K
- `b806810` `14:39:51` fix(squaresum): size S02BU split-K workspace per core
- `1a0ea39` `14:49:33` fix(squaresum): finalize S02BU split-K rows sequentially
- `b34d9fb` `14:54:57` perf(squaresum): gather split-K rows before tree finalize
- `945fda9` `14:57:15` revert(squaresum): keep verified sequential split-K finalize
- `88449b7` `14:58:08` perf(squaresum): match split-K cores to reduction groups
- `fde67e4` `15:00:25` revert(squaresum): retain full-core S02BU split-K
- `2c17660` `15:05:52` perf(squaresum): keep inner2 on grouped row path
- `ad25fbd` `15:10:44` test(squaresum): cover S02CA split-K boundaries
- `44644e5` `15:12:56` docs(squaresum): publish S02CA validation and release
- `6cf1678` `15:16:13` test(squaresum): add cross-path throughput atlas
- `f25d6cb` `15:24:33` experiment(squaresum): compact inner2 split-k reduction
- `529abc8` `15:30:38` test(squaresum): cover compact inner2 split-k boundaries
- `8609bc0` `15:34:03` experiment(squaresum): compact grouped inner2 rows
- `0190443` `15:38:25` fix(squaresum): retain aligned fallback for inner2 rows
- `527a39e` `15:43:50` experiment(squaresum): compact power2 split-k rows
- `1b0a076` `15:52:49` docs(squaresum): hand off S02CD compact release
- `9cc2f6e` `15:53:53` chore(artifact): record S02CD final cloud run
- `25c4a85` `16:13:38` optimize SquareSum fast3 split-K output range
- `68f24b1` `16:14:19` record S02CE cloud repository sync
- `ad4cf44` `16:37:49` optimize large strided SquareSum split-K
- `f7582cd` `16:38:31` record S02CF cloud repository sync
- `712358f` `17:00:32` extend SquareSum noncontiguous split-K to 64 outputs
- `41ba2e0` `17:01:00` record S02CG cloud repository sync
- `e793611` `17:35:08` align SquareSum fastPath4 work to output rows
- `ef72a52` `17:35:56` record S02CL cloud repository sync
- `8844d57` `18:36:59` optimize SquareSumV1 unaligned full rows
- `0b2b6bd` `18:37:52` record S02CM cloud repository sync
- `2892b21` `19:12:31` optimize SquareSumV1 fastPath2 full rows
- `3c058fc` `19:13:16` record S02CN cloud repository sync
- `e5b1758` `20:30:14` perf(squaresum): add official profiler gate and bf16 candidate
- `98fe4c7` `20:40:24` test(squaresum): add domain atlas and official A-B-A gate
- `641ded1` `20:51:57` test(squaresum): isolate S02CQ official cloud gate
- `eda57e0` `21:12:28` perf(squaresum): add general strided split-k candidate
- `e7b5b17` `21:27:33` perf(squaresum): cover short-tail split-k gap
- `cf8087b` `21:37:57` perf(squaresum): extend low-output split-k coverage
- `aa7217a` `21:41:02` test(squaresum): add event fail-fast gates
