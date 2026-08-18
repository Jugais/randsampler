import tomllib
from pathlib import Path

import mlsampler


def declared_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]["version"]


def test_version_matches_pyproject():
    assert mlsampler.__version__ == declared_version()


def test_version_is_exported():
    assert "__version__" in mlsampler.__all__
