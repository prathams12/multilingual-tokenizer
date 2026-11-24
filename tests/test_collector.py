from pathlib import Path
from collect_corpus import fill_split

documents = iter([
    {"text": "Hi 日😊"},
    {"text": "東京 AI"},
    {"text": "Hola mundo"}
])

train_stats = fill_split(
                documents, 
                Path("test_train.jsonl"), 
                4
    )

valid_stats = fill_split(
                documents,
                Path("test_valid.jsonl"),
                4
    )

print("Train Stats:", train_stats)
print("Validation Stats:", valid_stats)
