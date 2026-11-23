
import unittest
from pathlib import Path

from config import (
    BASE_DIR,
    OUT_DIR,
    TOTAL_CHARS,
    VALID_CHARS_PER_LANG,
    VOCAB_SIZE,
    SEED,
    SHUFFLE_BUFFER,
    LANGUAGES,
)


class TestTokenizerConfig(unittest.TestCase):

    def test_training_corpus_size(self):
        self.assertEqual(TOTAL_CHARS, 2_000_000_000)

    def test_validation_size(self):
        self.assertEqual(VALID_CHARS_PER_LANG, 1_000_000)

    def test_vocabulary_size(self):
        self.assertEqual(VOCAB_SIZE, 64_000)

    def test_random_seed(self):
        self.assertEqual(SEED, 42)

    def test_shuffle_buffer(self):
        self.assertEqual(SHUFFLE_BUFFER, 256)

    def test_language_count(self):
        self.assertEqual(len(LANGUAGES), 5)

    def test_language_names(self):
        expected = {"en", "ja", "hi", "zh", "es"}
        self.assertEqual(set(LANGUAGES), expected)

    def test_language_distribution(self):
        percentages = [
            details[2]
            for details in LANGUAGES.values()
        ]
        self.assertEqual(sum(percentages), 100)

    def test_training_allocations(self):
        allocations = {
            lang: TOTAL_CHARS * details[2] // 100
            for lang, details in LANGUAGES.items()
        }

        expected = {
            "en": 800_000_000,
            "ja": 700_000_000,
            "hi": 200_000_000,
            "zh": 200_000_000,
            "es": 100_000_000,
        }

        self.assertEqual(allocations, expected)

    def test_total_validation_size(self):
        total_validation = (
            len(LANGUAGES) * VALID_CHARS_PER_LANG
        )
        self.assertEqual(total_validation, 5_000_000)

    def test_output_directory(self):
        self.assertIsInstance(OUT_DIR, Path)
        self.assertEqual(
            OUT_DIR,
            BASE_DIR / "multilingual_data"
        )

    def test_dataset_configuration(self):
        for lang, details in LANGUAGES.items():
            with self.subTest(language=lang):
                repository, subset, percentage = details

                self.assertTrue(repository)
                self.assertTrue(subset)
                self.assertGreater(percentage, 0)


if __name__ == "__main__":
    unittest.main()
