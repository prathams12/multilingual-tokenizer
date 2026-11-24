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
        
        quotas[language] = int(total_chars * information[2] / 100)
    
    remaining_chars = total_chars - sum(quotas.values())
    if remaining_chars > 0:
        # Distribute remaining characters to the last language in the qoutas dictionary
        last_language = list(quotas.keys())[-1]
        quotas[last_language] += remaining_chars
    
    return quotas

def get_stream(dataset_name, configuration):
    
    dataset = load_dataset(
        dataset_name,
        name=configuration,
        split="train",
        streaming=True,
    )
    
    dataset = dataset.shuffle(
        buffer_size=SHUFFLE_BUFFER, 
        seed=SEED
    
    )
    
    return dataset
    