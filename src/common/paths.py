import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("RAUMDEUTER_ROOT", _SRC_DIR.parent)).resolve()

RAW_DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
TRACKEVAL_DIR = PROJECT_ROOT / "TrackEval"   # sn-trackeval 작업 폴더 (외부 도구 전용)