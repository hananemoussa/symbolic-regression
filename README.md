# Evolutionary Search with Reinforcement Learning for Data-driven Discovery

This repository is a fork of the official repository corresponding to the paper: 

> **Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning**  
> Anja Surina, Amin Mansouri, Lars Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, Caglar Gulcehre
> arXiv preprint arXiv:2504.05108 (2025)

In this work, we use their method EvoTune on four symbolic regression from LLM-SRBench benchmark and analyze and discuss results. 


## Overview

**EvoTune** is a framework for discovering new algorithms by combining:

1. Evolutionary search over LLM-generated Python programs, and
2. Reinforcement Learning to fine-tune the search operator - the LLM - based on performance scores of discovered algorithms .



## Repo Structure

```plaintext

symbolic-regression/
├── configs/                  # Hydra-based config system
│   ├── accelerate_config/    # Accelerate configs
│   ├── cluster/              # SLURM / cluster overrides
│   ├── model/                # Model-specific settings
│   ├── sweep/                # Sweep configuration files
│   ├── task/                 # Per-task configs (e.g., bin, tsp, etc.)
│   ├── train/                # Training configuration
│   └── config.yaml           # Default config
├── data/                     # TSP and flatpack datasets
├── installation/             # Dockerfiles for various hardware
├── llm-srbench-dataset/      # Train, ID, and OOD sets of all tasks in LLM-SRBench
├── scripts/                  # Example launch scripts for sweeps
│   ├── run_eval_sweep_example.sh
│   └── run_train_sweep_example.sh
│   ├── run_sr_evotune.sh     # Script to run EvoTune method on a symbolic regression task 
│   └── run_sr_funsearch_baseline.sh # Script to run FunSearch baseline on a symbolic regression task
├── src/
|   ├── packing/              # Core EvoTune framework
|   │   ├── evaluate/         # Task-specific logic (registered via registry)
|   │   │   ├── bin_packing/
|   │   │   ├── flat_pack/
|   │   │   ├── tsp/
|   │   │   ├── symbolic_regression/ # Contains the task specific logic for symbolic regression tasks from LLM-SRBench
|   │   │   ├── registry.py   # Task registry
|   │   │   └── README.md     # How to add new tasks
|   │   ├── funsearch/        # Program database implementation
|   │   ├── logging/          # Logging, statistics, and function tracking
|   │   ├── model/            # Prompting, LLM I/O, inference engine setup
|   │   ├── parallel/         # Multiprocessing producers & consumers
|   │   ├── train/            # DPO pipelines for fine-tuning LLMs
|   │   └── utils/            # Seeding, function helpers, etc.
|   └──  experiments/         # Scripts for specific experiments (train / eval)
├── pyproject.toml
└── LICENSE
```


## How to run code
### Setup & Dependencies

To create the Python environment for running experiments, use one of the provided **Dockerfiles** that matches your machine architecture and desired inference backend:

```plaintext
installation/
├── docker-amd64-cuda-tgi/   # For x86_64 machines using TGI
├── docker-amd64-cuda-vllm/  # For x86_64 machines using vLLM
└── docker-arm64-cuda/       # For ARM64 machines using vLLM
```
### Run EvoTune and FunSearch for Symbolic Regression tasks

1. Navigate to `scripts/run_sr_evotune.sh` and `scripts/run_sr_funsearch_baseline.sh`. These scripts launch SLURM jobs on OSC. They’re customizable: you can change the LLM-SRBench task, backbone model, and other run parameters.
2. Launch the experiments:
   ```bash
   sbatch scripts/run_sr_evotune.sh && sbatch scripts/run_sr_funsearch_baseline.sh
3. During execution, logs (model outputs, per-round metrics, and program database evolution) will be written to `out/logs`.
