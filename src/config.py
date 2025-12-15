"""
Configuration for recursive depth probe experiment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API CONFIGURATION
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model identifiers
MODELS = {
    "haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "display_name": "Claude 3.5 Haiku"
    },
    "sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4"
    },
    "gpt4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini"
    }
}

# EXPERIMENT PARAMETERS - MAIN (original)
MAIN_DEPTHS = [1, 2, 3, 4, 5, 6]
MAIN_TRIALS_PER_DEPTH = 30

# EXPERIMENT PARAMETERS - EXTENDED
EXTENDED_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
EXTENDED_TRIALS_PER_DEPTH = 30

# PATHS
DATA_DIR = "data"
RESULTS_DIR = "results"
PILOT_RESULTS_DIR = "results/pilot"
MAIN_RESULTS_DIR = "results/main"
EXTENDED_RESULTS_DIR = "results/extended"

# ANALYSIS PARAMETERS
ALPHA = 0.05
COLLAPSE_THRESHOLD = 0.30
CHANCE_LEVEL = 0.17