
from pathlib import Path
import json

DATA_DIR = Path("multilingual_data")

LANGUAGES = ["en", "ja", "hi", "zh", "es"]


def count_codepoints(path):

    total_chars = 0
    total_docs = 0

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            text = record["text"]

            total_chars += len(text)
            total_docs += 1

    return total_chars, total_docs


for split in ["train", "validation"]:

    print(f"\n{split.upper()} CORPUS")

    for language in LANGUAGES:

        path = DATA_DIR / split / f"{language}.jsonl"

        if not path.exists():
            print(f"{language}: MISSING")
            continue

        try:
            chars, docs = count_codepoints(path)

            print(
                f"{language}: "
                f"{chars:,} code points, "
                f"{docs:,} documents"
            )

        except (json.JSONDecodeError, UnicodeError, KeyError) as error:
            print(f"{language}: INVALID DATA - {error}")