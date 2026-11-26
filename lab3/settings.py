from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
DATA_DIR = ARTIFACTS / "data"
CHECKPOINT_DIR = ARTIFACTS / "checkpoints"

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MODEL_ID = "distilbert/distilgpt2"

SEED = 42
CONTEXT = 128
BATCH_SIZE = 16

ARCHITECTURE = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
}

RUNS = {
    "scratch": {
        "epochs": 6,
        "learning_rate": 3e-4,
    },
    "transfer": {
        "epochs": 3,
        "learning_rate": 5e-5,
    },
}

PROMPTS = [
    "The history of the city",
    "The game was developed",
    "The main character",
]

MAX_NEW_TOKENS = 64
SAMPLING_SEEDS = [42, 43, 44]

STRATEGIES = {
    "greedy": {},
    "temperature": {"temperature": 0.8},
    "top_k": {"temperature": 1.0, "top_k": 40},
    "top_p": {"temperature": 1.0, "top_p": 0.9},
}

HARDWARE = {
    "computer": "MacBook Pro 16-inch, November 2024",
    "chip": "Apple M4 Max",
    "installed_memory_gb": 48,
    "macos_from_user": "26.2",
}