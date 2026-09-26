"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import stemdenoise


def test_version_matches_distribution_metadata() -> None:
    assert stemdenoise.__version__ == version("stemdenoise")
