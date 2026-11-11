# Correlation-Aware Cardinality Estimation for Semantic Queries over Unstructured Data

This is the implementation of **Coral**, a correlation-aware cardinality estimator for semantic queries over unstructured data. Coral combines reusable offline semantic statistics with influence-guided online refinement to achieve accurate estimates under a limited LLM budget. Datasets and queries are uploaded to [this link](https://figshare.com/s/c3a25fbd5507eaa3c635). 


## Folder Structure

```text
├── src/
│   ├── data/            # Dataset loading and artifact storage.
│   ├── offline/         # Offline learning of join and filter models.
│   ├── online/          # Online sampling, calibration, and cardinality estimation.
│   ├── inference/       # LLM API client and embeddings.
│   ├── config.py        # Configuration.
│   ├── defaults.json    # Default settings.
│   ├── prompts.py       # LLM prompts.
│   ├── evaluation.py    # Q-error evaluation.
│   └── runner.py        # End-to-end training and evaluation pipeline.
├── models.json          # Model endpoints, model paths, and runtime settings.
├── requirements.txt     # Python dependencies.
├── run.py               # Command-line entry point.
└── readme.md
```


## Quick Start

#### Step 1: Set up the environment

Use Python 3.12. From the repository root, run:

```bash
conda create -n coral_env python=3.12 -y
conda activate coral_env
python -m pip install -r requirements.txt
```

#### Step 2: Configure the LLM model and embedding model

Fill in the LLM service and embedding model settings in `models.json`.
By default, the experiments use `Qwen3.8-27B` as the LLM and `Qwen3-Embedding-4B` as the embedding model.

#### Step 3: Download the datasets

Download the project's IMDB, BioDEX, and TREx benchmark datasets and queries.

**[Dataset and Queries Download URL](https://figshare.com/s/c3a25fbd5507eaa3c635).**

#### Step 4: Run from the command line

Run an experiment for the desired dataset:

```bash
python run.py imdb /path/to/imdb
python run.py biodex /path/to/biodex
python run.py trex /path/to/trex
```

Each command performs offline learning followed by online evaluation. The console prints the final Q-error statistics. Results are saved
under `runs/<dataset>/`, including `summary.json`, `results.json`, `run.log`,
trained models, and checkpoints.
