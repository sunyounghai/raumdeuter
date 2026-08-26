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
        "--benchmark",
        default="SoccerNet",
        help="TrackEval GT 폴더의 벤치마크 이름, 최종 폴더명은 <benchmark>-<split> "
             "예: 기본 GT는 'SoccerNet' (-> SoccerNet-test), "
             "player+goalkeeper만 필터링한 GT는 'SoccerNet-players' (-> SoccerNet-players-test)",
    )
    parser.add_argument(
        "--out_name",
        default="unified_comparison.txt",
        help="results/tracking/ 밑에 저장할 파일명 (기본: unified_comparison_<클립>.txt — 클립마다 자동 분리)",
    )
    args = parser.parse_args()
    if args.out_name is None:
        args.out_name = f"unified_comparison_{args.seq}.txt"
    return args


def main():
    args = parse_args()
    print(f"평가할 클립: {args.seq}")
    print(f"평가할 조건: {args.conditions}\n")

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(TRACKEVAL_DATA / "gt" / "mot_challenge") + "/"
    dataset_config["TRACKERS_FOLDER"] = str(TRACKEVAL_DATA / "trackers" / "mot_challenge") + "/"
    dataset_config["BENCHMARK"] = args.benchmark
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


def _load_existing_rows(seq_name: str, out_name: str) -> dict:
    """
    기존 unified_comparison.txt가 있으면 파싱해서 {condition: row_dict} 형태로 반환,
    없으면 빈 딕셔너리 반환, 다른 클립(seq_name)의 파일이면 무시(새로 시작)함.
    """
    out_path = RESULTS_DIR / "tracking" / out_name
    if not out_path.exists():
        return {}

    lines = out_path.read_text().splitlines()
    if not lines or not lines[0].startswith(seq_name):
        return {}

    rows = {}
    data_lines = [l for l in lines[3:] if l.strip()] # 제목/빈줄/헤더 3줄 건너뜀
    for line in data_lines:
        parts = line.split()
        if len(parts) != 9: # condition + HOTA/DetA/AssA/IDF1/Dets/GT_Dets/IDs/GT_IDs = 9개
            continue
        cond, hota, deta, assa, idf1, dets, gt_dets, ids, gt_ids = parts
        rows[cond] = {
            "condition": cond,
            "HOTA": float(hota),
            "DetA": float(deta),
            "AssA": float(assa),
            "IDF1": float(idf1),
            "Dets": int(dets),
            "GT_Dets": int(gt_dets),
            "IDs": int(ids),
            "GT_IDs": int(gt_ids),
        }
    return rows


def save_unified_comparison(results: dict, conditions: list[str], seq_name: str, out_name: str) -> None:
    """
    지정한 조건들의 HOTA/DetA/AssA/IDF1/트랙수를 표로 정리

    기존 파일이 있으면 덮어쓰지 않고 조건 이름 기준으로 병합함
    이번 실행에 없는 이전 조건들의 행은 그대로 유지되고, 겹치는 조건은 이번 결과로 갱신
    """
    rows_by_condition = _load_existing_rows(seq_name, out_name)

    for cond in conditions:
        d = results["MotChallenge2DBox"][cond][seq_name][CLASS_NAME]
        hota_arr = d["HOTA"]
        rows_by_condition[cond] = {
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

    out_dir = RESULTS_DIR / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # 가장 긴 조건 이름에 맞춰 첫 칸 너비를 동적으로 정함 (정렬 깨짐 방지)
    name_width = max(len(c) for c in rows_by_condition.keys()) + 2

    header = f"{'condition':<{name_width}}{'HOTA':>8}{'DetA':>8}{'AssA':>8}{'IDF1':>8}{'Dets':>8}{'GT_Dets':>9}{'IDs':>6}{'GT_IDs':>8}"
    lines = [f"{seq_name} 기준 조건 비교 (sn-trackeval, 누적)", "", header]
    for cond in sorted(rows_by_condition.keys()):
        r = rows_by_condition[cond]
        lines.append(
            f"{r['condition']:<{name_width}}{r['HOTA']:>8.3f}{r['DetA']:>8.3f}{r['AssA']:>8.3f}"
            f"{r['IDF1']:>8.3f}{r['Dets']:>8d}{r['GT_Dets']:>9d}{r['IDs']:>6d}{r['GT_IDs']:>8d}"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[unified_comparison] 저장 완료 -> {out_path}")

if __name__ == "__main__":
    main()