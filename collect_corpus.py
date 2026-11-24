import argparse
import json
from datasets import load_dataset

from config import (
    TOTAL_CHARS,
    VALID_CHARS_PER_LANG,
    SEED,
    SHUFFLE_BUFFER,
    OUT_DIR,
    LANGUAGES,
)

def make_quotas(total_chars):
    """
    Calculate the number of characters to collect for each language based on the total character count
    and the specified percentages in LANGUAGES.
    """

    quotas = {}
    for language, information in LANGUAGES.items():
        