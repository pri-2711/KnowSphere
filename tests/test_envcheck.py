"""Environment check: pins are read from requirements.txt and compared with installed versions."""

from src.envcheck import mismatches, pinned_versions


def test_reads_pins_including_extras(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# comment\ntorch==2.14.1\nuvicorn[standard]==0.54.0\nloose>=1.0\n", encoding="utf-8")
    assert pinned_versions(requirements) == {"torch": "2.14.1", "uvicorn": "0.54.0"}


def test_reports_wrong_and_missing_packages() -> None:
    problems = dict((name, (required, installed)) for name, required, installed in
                    mismatches({"pytest": "0.0.1", "surely-not-installed-package": "1.0"}))
    assert problems["pytest"][0] == "0.0.1" and problems["pytest"][1] is not None
    assert problems["surely-not-installed-package"] == ("1.0", None)
