"""
3개 사전학습 모델(COCO / Roboflow / H250)의 다른 값만 모아둠
- weights_path         : 사전학습 가중치 파일 경로
- run_name             : 파인튜닝 결과가 저장될 runs/ 하위 폴더 이름
- pretrained_class_map : baseline(파인튜닝 전) 모델이 예측하는 클래스 인덱스를 
                         GT 클래스(player=0)로 매핑 {예측 클래스: GT 클래스}
"""

from pathlib import Path

from paths import WEIGHTS_DIR, RUNS_DIR

MODELS = {
    "coco": {
        # COCO로 사전학습된 YOLOv8n
        "weights_path": str(WEIGHTS_DIR / "yolov8n.pt"),
        "run_name": "finetune",
        "pretrained_class_map": {0: 0}, # COCO person -> player
    },
    "roboflow": {
        # Roboflow (클래스: ball, goalkeepr, player, referee)
        "weights_path": str(WEIGHTS_DIR / "football-player-detection.pt"),
        "run_name": "roboflow_finetune",
        "pretrained_class_map": {1: 0, 2: 0}, # goalkeeper, player -> player
    },
    "h250": {
        # SoccerNet H250 서브셋으로 사전학습 (클래스: ball, person)
        "weights_path": str(WEIGHTS_DIR / "yolov8n_soccernetv3h250_pretrained.pt"),
        "run_name": "h250_finetune",
        "pretrained_class_map": {1: 0}, # person -> player
    },
}

def get_model_config(name: str) -> dict:
    if name not in MODELS:
        raise ValueError(
            f"알 수 없는 모델 이름: '{name}',  사용 가능: {list(MODELS.keys())}"
        )
    return MODELS[name]

def finetuned_weights_path(name: str) -> Path:
    cfg = get_model_config(name)
    return RUNS_DIR / cfg["run_name"] / "weights" / "best.pt"