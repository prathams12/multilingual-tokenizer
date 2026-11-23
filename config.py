
from pathlib import Path

# =============================================
# 1. PROJECT PATHS
# =============================================

BASE_DIR = Path(__file__).resolve().parent

# Directory for multilingual training corpus
OUT_DIR = BASE_DIR / "multilingual_data"


# =============================================
# 2. CORPUS CONFIGURATION
# =============================================

# Total Unicode code points for training
TOTAL_CHARS = 2_000_000_000

# Separate validation data per language
VALID_CHARS_PER_LANG = 1_000_000


# =============================================
# 3. TOKENIZER CONFIGURATION
# =============================================

# Target vocabulary size
VOCAB_SIZE = 64_000


# =============================================
# 4. REPRODUCIBILITY & STREAMING
# =============================================

SEED = 42

# Number of documents in streaming shuffle buffer
SHUFFLE_BUFFER = 256


# =============================================
# 5. LANGUAGE DATASETS
# =============================================

# Language: (dataset repository, subset, percentage)
LANGUAGES = {
    "en": ("HuggingFaceFW/fineweb", "sample-10BT", 40),
    "ja": ("HuggingFaceFW/fineweb-2", "jpn_Jpan", 35),
    "hi": ("HuggingFaceFW/fineweb-2", "hin_Deva", 10),
    "zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", 10),
    "es": ("HuggingFaceFW/fineweb-2", "spa_Latn", 5),
}


# =============================================
# 6. CONFIGURATION VALIDATION
# =============================================

assert len(LANGUAGES) == 5, "Expected exactly 5 languages"

assert sum(
    percentage
    for _, _, percentage in LANGUAGES.values()
) == 100, "Language percentages must total 100"
