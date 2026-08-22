"""
트래킹 결과를 gt.txt와 비교해 HOTA/IDF1을 계산함

사전 준비 (클립/조건마다 필요):
  raumdeuter/TrackEval/data/gt/mot_challenge/SoccerNet-test/<클립>/gt/gt.txt
  raumdeuter/TrackEval/data/gt/mot_challenge/SoccerNet-test/<클립>/seqinfo.ini
  raumdeuter/TrackEval/data/gt/mot_challenge/seqmaps/SoccerNet-test.txt
  raumdeuter/TrackEval/data/trackers/mot_challenge/SoccerNet-test/<조건>/data/<클립>.txt

실행:
  # 현재 (SNMOT-116, 4개 조건)
  python -m src.tracking.run_eval --conditions bytetrack_gt bytetrack_det botsort_gt botsort_det

  # 나중에 다른 클립/조건으로 재사용할 때
  python -m src.tracking.run_eval --seq SNMOT-125 --conditions bytetrack_gt botsort_gt
"""


import argparse

import numpy as np
import trackeval

from src.common.paths import RESULTS_DIR, TRACKEVAL_DIR

TRACKEVAL_DATA = TRACKEVAL_DIR / "data"
CLASS_NAME = "pedestrian"  # sn-trackeval 기본 클래스명

def parse_args():
    parser = argparse.ArgumentParser(description="sn-trackeval로 트래킹 조건들 HOTA/IDF1 비교")
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="비교할 조건 폴더 이름들 (예: bytetrack_gt bytetrack_det botsort_gt botsort_det)",
    )
    parser.add_argument(
        "--seq",
        default="SNMOT-116",
        help="평가할 클립 이름 (TrackEval GT/seqmaps에 이미 등록돼 있어야 함)",
    )
    parser.add_argument(
        "--out_name",
        default="unified_comparison.txt",
        help="results/tracking/ 밑에 저장할 파일명",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"평가할 클립: {args.seq}")
    print(f"평가할 조건: {args.conditions}\n")

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(TRACKEVAL_DATA / "gt" / "mot_challenge") + "/"
    dataset_config["TRACKERS_FOLDER"] = str(TRACKEVAL_DATA / "trackers" / "mot_challenge") + "/"
    dataset_config["BENCHMARK"] = "SoccerNet"
    dataset_config["SPLIT_TO_EVAL"] = "test"
    dataset_config["TRACKERS_TO_EVAL"] = args.conditions

    # SoccerNet gt.txt엔 별도 class 컬럼이 없는 단순 10컬럼 포맷이라
    # MOT17 전용 전처리(distractor 클래스 제거 등)를 끄지 않으면 에러가 날 수 있음
    dataset_config["DO_PREPROC"] = False
    dataset_config["PRINT_CONFIG"] = False

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = True

    evaluator = trackeval.Evaluator(eval_config)
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.Identity()]

    results, messages = evaluator.evaluate([dataset], metrics_list)
    save_unified_comparison(results, args.conditions, args.seq, args.out_name)
    return results


def save_unified_comparison(results: dict, conditions: list[str], seq_name: str, out_name: str) -> None:
    """
    지정한 조건들의 HOTA/DetA/AssA/IDF1/트랙수를 표로 정리
    """
    rows = []
    for cond in conditions:
        d = results["MotChallenge2DBox"][cond][seq_name][CLASS_NAME]
        hota_arr = d["HOTA"]
        row = {
            "condition": cond,
            "HOTA": np.mean(hota_arr["HOTA"]) * 100,
            "DetA": np.mean(hota_arr["DetA"]) * 100,
            "AssA": np.mean(hota_arr["AssA"]) * 100,
            "IDF1": d["Identity"]["IDF1"] * 100,
            "Dets": d["Count"]["Dets"],
            "GT_Dets": d["Count"]["GT_Dets"],
            "IDs": d["Count"]["IDs"],
            "GT_IDs": d["Count"]["GT_IDs"],
        }
        rows.append(row)

    out_dir = RESULTS_DIR / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    header = f"{'condition':<14}{'HOTA':>8}{'DetA':>8}{'AssA':>8}{'IDF1':>8}{'Dets':>8}{'GT_Dets':>9}{'IDs':>6}{'GT_IDs':>8}"
    lines = [f"{seq_name} 기준 조건 비교 (sn-trackeval)", "", header]
    for r in rows:
        lines.append(
            f"{r['condition']:<14}{r['HOTA']:>8.3f}{r['DetA']:>8.3f}{r['AssA']:>8.3f}"
            f"{r['IDF1']:>8.3f}{r['Dets']:>8d}{r['GT_Dets']:>9d}{r['IDs']:>6d}{r['GT_IDs']:>8d}"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[unified_comparison] 저장 완료 -> {out_path}")

if __name__ == "__main__":
    main()