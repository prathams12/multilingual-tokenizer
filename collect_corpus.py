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


def fill_split(source,output_path,target_chars):
    
    output_path.parent.mkdir(
        parents=True, 
        exist_ok=True
    )
    
    char_count = 0
    document_count = 0
    
    with output_path.open("w",encoding="utf-8") as file:
        
        while char_count < target_chars:

            try:
                row = next(source)
            
            except StopIteration as error:
                raise RuntimeError(
                    f"Dataset exhausted before reaching the target character count of {target_chars}."
                ) from error
            
            text = row.get("text")
            
            if not isinstance(text, str) or not text.strip():
                continue
            
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                continue
            
            remaining_chars = target_chars - char_count
            
            selected_text = text[:remaining_chars]
            record = {"text": selected_text}
            
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            
            char_count += len(selected_text)
            document_count += 1
            
            if document_count % 1000 == 0:
                print(
                    f"{output_path}: "
                    f"{char_count:,}/{target_chars:,} characters collected "
                )
    return {
        
        "characters" : char_count, 
        "documents" : document_count
        
        }

def collect_all(total_chars, valid_chars_per_lang):
    
    quotas = make_quotas(total_chars)
    
    manifest = {
        
        "target_training_characters" : total_chars,
        "languages" : {}
    }
    
    for lang, information in LANGUAGES.items():
        
        dataset_name, configuration, percentage = information
        
        print(f"Collecting Language: {lang}")
        
        dataset = get_stream(dataset_name, configuration)
        
        source = iter(dataset)
        
        train_path = OUT_DIR / "train" / f"{lang}.jsonl"
        
        train_stats = fill_split(
            source, 
            train_path, 
            quotas[lang]
        )
        
        valid_path = OUT_DIR / "validation" / f"{lang}.jsonl"
        
        valid_stats = fill_split(
            source,
            valid_path,
            valid_chars_per_lang
        )
        
        manifest["languages"][lang] = {
            
            "dataset_name" : dataset_name,
            "configuration" : configuration,
            "percentage" : percentage,
            "train" : train_stats,
            "validation" : valid_stats
        }
        
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        with (OUT_DIR / "MANIFEST.json").open("w",encoding="utf-8") as file:
            json.dump(
                manifest, 
                file, 
                ensure_ascii=False, 
                indent=2
            )
        
        print(f"Corpus collection completed")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--total-chars",
        type=int,
        default=100_000
    )
    
    parser.add_argument(
        "--valid-chars",
        type=int,
        default=1_000
    )
    
    args = parser.parse_args()
    
    collect_all(args.total_chars, args.valid_chars)
    
            

    