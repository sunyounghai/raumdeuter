"""
gameinfo.ini의 역할 정보(player/goalkeeper/referee/ball)를 이용해
gt.txt에서 특정 역할만 남기고 필터링

- 검출기마다 지원하는 클래스가 다르므로(예: h250은 심판을 검출 못함)
공정하게 비교하려면 "선수만 있는 GT"로 따로 채점해야 함

사용 예:
  python -m src.tracking.filter_gt_by_role \
      --gt data/SoccerNet/tracking-2023/test/SNMOT-116/gt/gt.txt \
      --gameinfo data/SoccerNet/tracking-2023/test/SNMOT-116/gameinfo.ini \
      --keep-roles player goalkeepers \
      --out TrackEval/data/gt/mot_challenge/SoccerNet-test/SNMOT-116/gt/gt_players_only.txt
"""


import argparse
import configparser
import csv
from pathlib import Path


def parse_gameinfo(path: Path) -> dict:
    """
    gameinfo.ini -> {track_id: {"role":..., "team":..., "jersey":...}}
    """

    config = configparser.ConfigParser()
    config.read(path)

    result = {}
    for key, value in config["Sequence"].items():
        if not key.startswith("trackletid_"):
            continue
        track_id = int(key.split("_")[1])
        cls, jersey = [s.strip() for s in value.split(";")]

        if "team left" in cls:
            role, team = cls.replace(" team left", ""), "left"
        elif "team right" in cls:
            role, team = cls.replace(" team right", ""), "right"
        else:
            role, team = cls, None

        result[track_id] = {"role": role, "team": team, "jersey": jersey}

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="gt.txt를 역할 기준으로 필터링")
    parser.add_argument("--gt", type=Path, required=True, help="원본 gt.txt 경로")
    parser.add_argument("--gameinfo", type=Path, required=True, help="gameinfo.ini 경로")
    parser.add_argument(
        "--keep-roles",
        nargs="+",
        default=["player", "goalkeepers"],
        help="남길 역할 목록 (gameinfo.ini의 role 값 기준) "
             "기본값: player, goalkeepers (referee, ball 제외)",
    )
    parser.add_argument("--out", type=Path, required=True, help="필터링된 결과 저장 경로")
    return parser.parse_args()



def main():
    args = parse_args()

    role_map = parse_gameinfo(args.gameinfo)
    keep_ids = {tid for tid, info in role_map.items() if info["role"] in args.keep_roles}

    print(f"gameinfo.ini 전체 트랙: {len(role_map)}개")
    print(f"필터 후 남는 트랙(role in {args.keep_roles}): {len(keep_ids)}개")
    dropped = {info["role"] for tid, info in role_map.items() if tid not in keep_ids}
    print(f"제외되는 role: {dropped}")

    with open(args.gt) as f_in:
        rows = list(csv.reader(f_in))

    filtered = [row for row in rows if int(row[1]) in keep_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerows(filtered)

    print(f"\n전체 행: {len(rows)} -> 필터 후: {len(filtered)} ({len(filtered)/len(rows)*100:.1f}%)")
    print(f"저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()