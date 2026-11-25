
from datasets import load_dataset


def test_spanish(shuffle=False):

    print(f"\nTesting Spanish | shuffle={shuffle}")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="spa_Latn",
        split="train",
        streaming=True
    )

    if shuffle:
        dataset = dataset.shuffle(
            seed=42,
            buffer_size=32
        )

    for index, document in enumerate(dataset):

        text = document["text"]

        print(
            f"Document {index + 1}: "
            f"{len(text):,} code points"
        )

        if index == 9:
            break

    print("Test completed successfully!")


if __name__ == "__main__":

    test_spanish(shuffle=False)

    #test_spanish(shuffle=True)
