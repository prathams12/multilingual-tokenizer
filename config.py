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

#CREATE DIRECTORIES IF THEY DO NOT EXIST
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
