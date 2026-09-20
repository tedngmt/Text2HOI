# Text2HOI inference timing on the installed GPU

Measured on an NVIDIA GeForce RTX 3070 Ti Laptop GPU, PyTorch 1.13.0+cu116,
batch size 1, with two CPU threads. CUDA was synchronized at timing boundaries.
The generation and refinement benchmarks ran separately to avoid competing GPU
work. No model was trained or modified for these measurements.

| Work | Mean time | Effective output throughput |
|---|---:|---:|
| Released text-conditioned GRAB mug generation, 1,000 diffusion steps | 7.805 s per 146–147-frame clip | 18.79 frames/s |
| Adapted GraspXL source-motion refinement, 155 frames | 70.55 ms per clip | 2,197 frames/s |
| Adapted bounded refiner alone, 96 frames and cached features | 5.93 ms per window | 16,190 frames/s |

Generation used one warm-up and three measured clips. Their combined 440 valid
output frames took 23.414 seconds; individual clips took 7.575–8.069 seconds.
Each sampling call processes a padded 150-frame sequence and then trims to the
predicted duration. Throughput above counts only valid output frames. Timing
includes duration/contact prediction, diffusion, refiner input construction,
hand refinement, and MANO/object mesh decoding. Models, text encoding, and object
features were already loaded/cached; initial selection/loading/encoding,
rendering, file export, and video compression are excluded.

Refinement used 10 warm-ups and 50 measurements per stage. The full 155-frame
clip uses overlapping 96-frame windows starting at 0, 48, and 59, then blends
them and decodes the right-hand MANO surface. It includes feature construction
but excludes model/data loading, input transfer, source contact-coverage
preparation, signed-distance metrics, rendering, and export. Mean latency was
70.55 ms, median 58.02 ms, and p95 157.11 ms. This is refinement of an existing
motion, not generation of a new motion from text. It reproduced the prior
exported geometry exactly.

These are whole-sequence throughput measurements. A generated clip becomes
available after its computation completes; the FPS numbers do not establish a
causal, interactive controller. The saved videos' 30 FPS is a playback choice,
independent of inference speed.

The authors' supplementary Table S2 reports 150 frames on an RTX 4090 in
0.011 s contact prediction + 4.9 s motion generation + 0.013 s refinement.
Their sum is 4.924 s, equivalent to a calculated 30.46 frames/s. Hardware and
timing boundaries differ from this local benchmark, so these are not a
controlled hardware comparison.

Source: [Text2HOI supplement, section 10](https://arxiv.org/html/2404.00562v2#S10).

Reproducible scripts and raw samples are beside this file:
`benchmark_generation.py`, `generation_benchmark.json`, `refiner_benchmark.py`,
and `refiner_results.json`. The initial generation run completed inference but
failed to write its final report because of a relative script path; the fixed
script was rerun successfully. Reported generation numbers use only that final
successful run (`generation_run_verified.log`).

For reference, the completed mug refiner training used 16 epochs for GraspXL
(816 updates) and 16 total epochs for GRAB (4 + 12 epochs, 3,200 updates).
Training epochs and inference throughput measure different things.
