from pathlib import Path

__version__ = open(Path(__file__).parent.joinpath("VERSION")).read().strip()
