
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import argparse
import os

from datasets import load_dataset


def test_spanish(shuffle, max_documents):

    print(f"Shuffle enabled: {shuffle}")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="spa_Latn",
        split="train",
        streaming=True,
        token=os.getenv("HF_TOKEN") or None,
    )

    if shuffle:
        dataset = dataset.shuffle(
            seed=42,
            buffer_size=32,
            max_buffer_input_shards=1, 
            )

    documents_read = 0

    for document in dataset:
        text = document["text"]

        documents_read += 1

        print(
            f"Document {documents_read}: "
            f"{len(text):,} code points"
        )

        if documents_read >= max_documents:
            break

    print(
        f"Completed reading {documents_read} documents."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    parser.add_argument(
        "--docs",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    test_spanish(
        shuffle=args.shuffle,
        max_documents=args.docs,
    )
