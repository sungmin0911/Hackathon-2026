"""Offline tests for the verification step (no LLM or database needed).
Run:  python -m pytest tests/   (or)   python tests/test_verify.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_reader import Page
from app.verify import verify_row

PAGES = [
    Page(1, "Abstract. Our model achieves 96.1% accuracy on CIFAR-10."),
    Page(6, "Table 2: ResNet-110 has 6.43% er-\nror on CIFAR-10, using 1.7M parameters."),
]
FIELDS = ["Model", "Accuracy"]


def test_verified_across_line_break():
    row = {"Model": "ResNet-110", "Accuracy": "6.43% error", "quote": "ResNet-110 has 6.43% error on CIFAR-10", "page": 6}
    out = verify_row(row, FIELDS, PAGES)
    assert out["status"] == "verified" and out["found_page"] == 6


def test_mismatch_when_value_altered():
    row = {"Model": "ResNet-110", "Accuracy": "93.57%", "quote": "ResNet-110 has 6.43% error on CIFAR-10", "page": 6}
    assert verify_row(row, FIELDS, PAGES)["status"] == "mismatch"


def test_unverified_when_quote_invented():
    row = {"Model": "ViT", "Accuracy": "99.5%", "quote": "ViT reaches 99.5% accuracy on CIFAR-10", "page": 3}
    assert verify_row(row, FIELDS, PAGES)["status"] == "unverified"


def test_numbers_in_names_are_ignored():
    """'DeepSeek-R1' and 'AIME 2024' contain numbers, but they're labels, not values."""
    pages = [Page(13, "AIME 2024 (Pass@1) 16.0 9.3 39.2 63.6 79.2 79.8")]
    row = {"Model": "DeepSeek-R1", "Benchmark": "AIME 2024 (Pass@1)", "Score": "79.8",
           "quote": "AIME 2024 (Pass@1) 16.0 9.3 39.2 63.6 79.2 79.8", "page": 13}
    assert verify_row(row, ["Model", "Benchmark", "Score"], pages)["status"] == "verified"
    row["Score"] = "81.0"
    assert verify_row(row, ["Model", "Benchmark", "Score"], pages)["status"] == "mismatch"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
