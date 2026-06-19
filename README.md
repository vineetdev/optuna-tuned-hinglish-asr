# Optuna-Tuned Hinglish ASR (Whisper + LoRA)

This repository implements an Optuna-tuned automatic speech recognition (ASR) system for Hinglish (Hindi–English code-mixed speech) by fine-tuning OpenAI Whisper-medium with parameter-efficient LoRA adapters on a technical tutorial dataset.

## Highlights

- Fine-tunes Whisper-medium for Hinglish ASR using parameter‑efficient LoRA on the decoder while freezing the encoder to preserve acoustic representations.
- Uses Optuna (TPE sampler + MedianPruner) to systematically search over LoRA and optimization hyperparameters instead of manual tuning.
- Targets real-world Hinglish technical tutorials (LibreOffice, Linux, software instructions) with mixed Devanagari and Roman scripts.
- Achieves large WER/CER gains over the Whisper-medium baseline on a 100-sample evaluation set, with performance competitive with commercial ASR systems in this domain.[file:2]

## Background

Hinglish (Hindi–English) code-mixed speech is ubiquitous in India but remains under-served by off-the-shelf ASR systems, particularly for technical content where English terminology is embedded in Hindi grammar and often written across scripts.[file:2] This project focuses on building a stronger ASR model for such code-mixed technical tutorials.

The experiments are based on the public **`ujshinglish-compressed`** dataset from Hugging Face, consisting of:
- ~52k training samples and ~3k test samples.
- Technical tutorial domain (LibreOffice, Linux, software usage).
- Hinglish speech with Hindi text (Devanagari) and English technical terms interleaved.[file:2]

## Model and Training Overview

The base model is **`openai/whisper-medium`** (≈763M parameters). The fine-tuning strategy is designed to be parameter-efficient and hardware-friendly:

- **Encoder frozen:** ~307M encoder parameters are kept frozen to retain pretrained acoustic knowledge.[file:2]
- **LoRA on decoder:** Trainable LoRA adapters are applied to decoder attention and feed-forward modules (`q_proj`, `v_proj`, `fc1`, `fc2`).[file:2]
- **Trainable parameters:** Only ~3.19% of the total parameters are updated (~25M parameters out of ~789M with LoRA).[file:2]

Key training configuration for the final run:

- Training samples: 10,000 (subset of training split).[file:2]
- Steps: 1,000 optimization steps.
- Optimizer: AdamW with weight decay.
- Learning rate: ≈7.37e-5 (Optuna-selected).
- Scheduler: Cosine with restarts.
- Effective batch size: 4 (batch size 2 with gradient accumulation).
- Precision: Mixed precision (FP16) for efficiency on GPU.[file:1][file:2]

The training loss decreases by roughly 84% (from ~2.41 to ~0.32), with the cosine-with-restarts schedule helping escape local minima and achieve smoother convergence.[file:2]

## Hyperparameter Optimization with Optuna

Instead of hand-tuning hyperparameters, the notebook uses **Optuna** to run a structured hyperparameter search over LoRA and optimization settings.[file:1][file:2]

Search setup:

- Sampler: TPE (Tree-structured Parzen Estimator).
- Pruner: MedianPruner, pruning trials at step 100 based on interim WER.
- Number of trials: 6.
- Trial training budget: 200 steps per trial on 1,000 samples.
- Objective: Minimize WER on 100 validation samples drawn from the test set.[file:1][file:2]

The search space covers eight key hyperparameters, including:

- LoRA rank `r` (e.g., 4, 8, 16, 32).
- LoRA alpha and dropout.
- Target modules configuration (e.g., `q_proj`, `v_proj` only vs. `q_proj`/`v_proj`/`fc1`/`fc2`).
- Learning rate (log-uniform range).
- Warmup steps.
- Weight decay.
- Scheduler type (linear, cosine, cosine-with-restarts).[file:1]

Each Optuna trial:

1. Samples a hyperparameter configuration.
2. Builds a fresh Whisper-medium + LoRA model with a frozen encoder.
3. Trains for up to 200 steps on 1,000 samples.
4. At step 100, computes a quick WER on a subset of validation samples and reports it to the pruner.
5. Either prunes the trial early or continues to 200 steps.
6. Evaluates final WER/CER on 100 validation samples for the objective value.[file:1]

## Results

All headline metrics below are computed on a 100-sample evaluation set.

### Whisper Baseline vs LoRA Fine-Tuning

| Model                                   | Setting                           | WER (↓) | CER (↓) |
|-----------------------------------------|------------------------------------|--------:|--------:|
| Whisper-medium (baseline)               | Pretrained, no fine-tuning        |  70.62  |  54.92  |
| Whisper-medium + LoRA (3k samples)      | LoRA fine-tune, 3k training samples |  54.22  |  40.98  |
| Whisper-medium + LoRA (Optuna best)     | Optuna-selected config, 10k samples |  36.46  |  30.94  |

[file:2]

Key observations:

- LoRA fine-tuning alone provides a substantial improvement over the baseline Whisper-medium model.
- The best Optuna-tuned configuration achieves a **~34 absolute WER reduction** relative to the baseline (from 70.62 to 36.46), representing a large relative error drop on this evaluation set.[file:2]

### Relation to Commercial Baselines

For context, the project also evaluates commercial ASR systems (such as Gemini-2.5-flash and Sarvam SARA v3) on the same Hinglish technical test data as domain-specific benchmarks.[file:2] The Optuna-tuned Whisper-medium + LoRA model achieves performance that is competitive with or better than these baselines in this setting, despite using a parameter-efficient fine-tuning approach.

## What This Repository Contains

- **`hinglish-asr-optuna-improved.ipynb`**  
  End-to-end notebook containing:
  - Dataset loading and preprocessing for `ujshinglish-compressed`.
  - Baseline evaluation of Whisper-medium and comparison models.
  - LoRA-based parameter-efficient fine-tuning setup.
  - Optuna hyperparameter optimization loop (TPE + MedianPruner).
  - Final fine-tuning run with the best hyperparameters.
  - Detailed WER/CER computation and qualitative error analysis.

- **`Hinglish_ASR_Project_Report.docx`**  
  A detailed project report describing the problem, literature review, data exploration, methodology, evaluation metrics, results, and error analysis for this Hinglish ASR system.[file:2]

## Getting Started

### Installation

The experiments were run with Python, PyTorch, and the Hugging Face ecosystem. A representative environment (see the notebook for exact versions) includes:

```bash
pip install \
  torch \
  torchaudio \
  transformers==4.40.0 \
  datasets==2.19.0 \
  evaluate==0.4.2 \
  jiwer==3.0.4 \
  peft==0.10.0 \
  accelerate==0.29.3 \
  librosa==0.10.1 \
  soundfile \
  optuna==3.6.1 \
  optuna-integration
```

### Quickstart: Run a Lightweight Demo

For recruiters or readers who want to see the pipeline without running the full hyperparameter search and long training:

1. Clone the repository and install the dependencies.
2. Open `hinglish-asr-optuna-improved.ipynb` in Jupyter / VS Code / Kaggle.
3. Run the initial cells:
   - Environment and GPU check.
   - Dataset loading and preprocessing.
   - Baseline Whisper-medium evaluation on a small set of samples.
4. Run a **single Optuna trial** or a reduced number of trials to see how the study is configured and how WER improves.
5. Use the provided helper functions to compute WER/CER and inspect sample-level predictions.

This path is designed to showcase the modeling and optimization logic without requiring the full compute used for the original experiments.[file:1]

## Hardware & Requirements

The original experiments were run on a cloud GPU environment comparable to a **Tesla T4 (16 GB VRAM)**.[file:1] With the parameter-efficient LoRA setup and mixed-precision training, the pipeline is designed to:

- Fit and train on a **single mid-range GPU** (e.g., T4, similar 12–16 GB cards).
- Keep most of the Whisper-medium parameters frozen, updating only a small LoRA adapter subset.

For quick demos and partial runs, a smaller GPU (or even CPU-only for inference only) can be used, but full HPO and fine-tuning are recommended on a GPU with at least 12–16 GB of VRAM.[file:1]

Readers who want a deeper, report-style narrative of the project can open `Hinglish_ASR_Project_Report.docx`, while the notebook serves as the executable artifact showcasing the full training and evaluation flow.[file:2]

## Contact Developer 
Email me at communication.vineet@gmail.com 
