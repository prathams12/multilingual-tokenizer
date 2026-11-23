from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent

#Training corpus directory
DATA_DIR= BASE_DIR / 'data'

#trained tokenizer output directory
TOKENIZER_DIR = BASE_DIR / 'artifacts'

#corpus target 2 Billion Unicode Points
TOTAL_CODEPOINTS = 2_000_000_000

#BYTE_LEVEL BPE VOCAB SIZE
VOCAB_SIZE = 64_000

#REPRODUCIBILITY
SEED = 42

# Number of streamed documents in the shuffle buffer
SHUFFLE_BUFFER = 256

# Directory where our corpus will be saved
OUT_DIR = Path("multilingual_data")

# Language: (dataset repository, configuration, percentage)
LANGUAGES = {
    "en": ("HuggingFaceFW/fineweb", "sample-10BT", 40),
    "ja": ("HuggingFaceFW/fineweb-2", "jpn_Jpan", 35),
    "hi": ("HuggingFaceFW/fineweb-2", "hin_Deva", 10),
    "zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", 10),
    "es": ("HuggingFaceFW/fineweb-2", "spa_Latn", 5),
}

assert len(LANGUAGES) == 5
assert sum(item[2] for item in LANGUAGES.values()) == 100

#CREATE DIRECTORIES IF THEY DO NOT EXIST
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
