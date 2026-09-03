# hrtfm_pinn: physics-informed HRTF upsampling with flow matching

This repository contains two deliberately separate research workflows:

1. The original azimuth-conditioned binaural-speech STFT flow in
   `scripts/train.py` and `scripts/sample.py`.
2. An experimental HUTUBS subject-40 HRTF upsampler in
   `scripts/train_hrtf.py` and `scripts/sample_hrtf.py`.

The HRTF workflow does **not** reinterpret speech time frames as spatial
coordinates.  It uses a continuous Cartesian neural field so that the
Helmholtz residual can be differentiated with respect to `(x, y, z)`.

## Subject-40 protocol

By default, the scripts read the sibling file
`../PINN-for-HRTF-upsampling/40.mat` without modifying it.  This is the sample
from Fei Ma et al.'s HRTF PINN interpolation experiment:

- one normalized, complex, single-ear HRTF;
- seven frequencies from 2067.1875 Hz to 14470.3125 Hz;
- 330 measured directions and 1260 total directions on a 0.09 m sphere;
- 930 dense directions held out for post-training evaluation.

The MAT file's so-called left/right parts are positive- and negative-y source
hemispheres, not the listener's two ears.  Mirroring Ma's experiment, the
implementation makes 28 scalar field tasks: 7 frequencies x real/imaginary x
2 hemispheres.  One shared conditional network learns all tasks.

There is a strict leakage boundary:

- `train_hrtf` is the only supervised training signal;
- `total_coor` supplies unlabeled physics-collocation coordinates;
- `total_hrtf` is used only by `sample_hrtf.py` after training, to report
  held-out errors.

Subject 40 is a transductive proof of concept.  It cannot establish
cross-subject generalization; that requires training on dense HRTFs from other
subjects and reserving subject 40 as a test subject.

## Method

Sparse observations are encoded by a permutation-invariant DeepSets encoder.
The conditional neural field receives the current function value, flow time,
Cartesian query coordinate, frequency, real/imaginary component, hemisphere,
and sparse-set context, then predicts the flow velocity.

The source distribution is a finite-rank Gaussian random plane-wave field.
Unlike pointwise white noise, it is a smooth function and can be evaluated
consistently on different coordinate sets.  Every source basis also satisfies
the frequency's homogeneous Helmholtz equation.

On the 165 measured directions in each hemisphere, training uses the linear
flow-matching target.  On randomly sampled coordinates from the 630-point
hemisphere grid, the model is differentiably unrolled from `t=0` to the clean
endpoint and penalized with Ma et al.'s normalized residual:

```text
R(H) = laplacian_x(H) / k^2 + H,       k = 2 pi f / c.
```

The model consumes unit coordinates `u=x/r`, so the implemented equivalent is
`laplacian_u(H)/(k r)^2 + H`.  During sampling, measured directions follow
their exact source-to-observation interpolation path and therefore end at the
observed values to numerical precision.

## Train subject 40

From this repository's root:

```bash
python scripts/train_hrtf.py \
  --input ../PINN-for-HRTF-upsampling/40.mat \
  --output-dir runs/hutubs_subject40/hrtfm_pinn_seed2026/training \
  --device cuda
```

The defaults train for 20,000 optimizer steps.  Physics is warmed up for 500
steps and ramped over the following 2000 steps.  Checkpoints and JSONL loss
logs are written under
`runs/hutubs_subject40/hrtfm_pinn_seed2026/training/`.

A small executable smoke run is:

```bash
python scripts/train_hrtf.py \
  --output-dir runs/hrtf_smoke \
  --steps 2 --batch-size 2 \
  --width 16 --depth 2 --context-dim 16 --observation-width 12 \
  --coordinate-bands 1 --time-dim 8 --prior-modes 4 \
  --physics-warmup-steps 0 --physics-ramp-steps 1 \
  --physics-points 4 --physics-unroll-steps 1 \
  --log-every 1 --save-every 2 --device cpu
```

## Sample and evaluate

```bash
python scripts/sample_hrtf.py \
  --input ../PINN-for-HRTF-upsampling/40.mat \
  --checkpoint runs/hutubs_subject40/hrtfm_pinn_seed2026/training/checkpoint_final.pt \
  --output-dir runs/hutubs_subject40/hrtfm_pinn_seed2026/evaluation/heun40_samples5 \
  --num-samples 5 --integration-steps 40 --device cuda
```

Outputs include:

- `prediction.mat`: mean prediction plus every generated sample, using the
  original `[7, 2, 1260]` layout;
- `evaluation.csv`: the paper's normalized complex error, complex NMSE, and
  magnitude NMSE for known, unknown, and all directions;
- `evaluation.json`: aggregate held-out metrics and sampling metadata.

The measured directions are hard-constrained.  Report unknown-direction
metrics as the primary result; known-direction scores are a consistency check.

Any compatible baseline MAT prediction, including Ma's `40_L3.mat`, can be
evaluated with the exact same implementation:

```bash
python scripts/evaluate_hrtf.py \
  --reference ../PINN-for-HRTF-upsampling/40.mat \
  --prediction ../PINN-for-HRTF-upsampling/40_L3.mat \
  --output-dir runs/baselines/ma_pinn
```

For a data-only flow ablation, train the same architecture with
`--physics-weight 0` and a separate output directory.

## CoSTAR subject-40 run

The checked-in Slurm job requests one full H200 GPU from CoSTAR's `main`
partition, trains the PI-FM model, and then runs five-sample/40-step Heun
evaluation. Clone the private repository into the current home path and create
the dedicated environment once from a login node:

```bash
cd /users/yc01847
git clone https://github.com/v41827/hrtfm_pinn.git
cd /users/yc01847/hrtfm_pinn
source /users/yc01847/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-hrtf.yml
```

Submit from the repository root so that `SLURM_SUBMIT_DIR` resolves correctly:

```bash
cd /users/yc01847/hrtfm_pinn
mkdir -p runs/slurm
sbatch cluster/costar_subject40_hrtfm_pinn.sbatch
```

The default input is the sibling path
`../PINN-for-HRTF-upsampling/40.mat`.  Override it without editing the job file
when the MAT file lives elsewhere:

```bash
HUTUBS_MAT=/users/yc01847/datasets/hutubs/subject_040/40.mat \
  sbatch cluster/costar_subject40_hrtfm_pinn.sbatch
```

Each submitted job writes a collision-free directory:

```text
runs/hutubs_subject40/hrtfm_pinn_seed2026_job_JOBID/
├── run_manifest.txt
├── conda_explicit.txt
├── gpu_info.txt
├── training/
│   ├── config.json
│   ├── training.jsonl
│   └── checkpoint_*.pt
├── evaluation/heun40_samples5/
│   ├── prediction.mat
│   ├── evaluation.csv
│   └── evaluation.json
└── baselines/fei_ma_pinn/       # only when sibling 40_L3.mat exists
    ├── evaluation.csv
    └── evaluation.json
```

Slurm stdout/stderr use the recognizable job name and job ID under
`runs/slurm/`.  The job also records the exact commands, Git commit, Conda
package lock, and GPU information needed to audit a paper run.

When `main` is busy, submit the checkpoint-aware `risk` variant instead:

```bash
cd /users/yc01847/hrtfm_pinn
mkdir -p runs/slurm
sbatch cluster/costar_subject40_hrtfm_pinn_risk.sbatch
```

The `risk` job uses the same model and evaluation configuration, saves every
500 steps, and automatically resumes the newest checkpoint after a Slurm
requeue. Its logs are opened in append mode so restart output is retained.

## Tests

No test-only dependency is required:

```bash
python -m unittest discover -s tests -v
python tools/py_syntax_check.py
```

Tests cover the HUTUBS split, leakage-safe field records, the analytic
plane-wave Helmholtz solution, second-order physics gradients, and hard data
consistency in the sampler.

## Scientific limitations

- The supplied `40.mat` contains one ear.  A binaural study must prepare both
  ears from HUTUBS and add an ear condition; hemispheres are not ears.
- All measurements lie at one radius.  As in Ma et al., the Cartesian PINN
  learns an off-surface extension whose radial derivatives are not directly
  observed.  This assumption must be stated in a paper.
- A single subject provides only 28 related scalar fields.  Flow-matching
  uncertainty and generalization claims require a multi-subject protocol.
- The HRTFs in this MAT file are normalized independently at each frequency.
  Metrics therefore reproduce this experiment but are not yet a full-band
  perceptual HRTF evaluation.

## Method references

- Fei Ma et al., [Spatial Upsampling of Head-Related Transfer Functions Using
  a Physics-Informed Neural Network](https://arxiv.org/abs/2307.14650).
- Kerrigan et al., [Functional Flow
  Matching](https://proceedings.mlr.press/v238/kerrigan24a.html).
- Baldan et al., [Physics vs. Distributions: Pareto Optimal Flow Matching with
  Physics Constraints](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a57483b394a3654f4317051e4ce3b2b8-Abstract-Conference.html).
- HUTUBS [database documentation](https://sofacoustics.org/data/database/hutubs/Documentation.pdf).
