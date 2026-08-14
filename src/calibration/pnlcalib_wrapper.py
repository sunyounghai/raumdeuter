"""
PnLCalib(external/PnLCalib)을 감싸는 wrapper
목적: PnLCalib의 내부 구현(HRNet, heatmap 디코딩 등)을 몰라도
     get_homography_matrix() 하나로 H 행렬을 받게 함

주의: PnLCalib은 H(homography)를 직접 주지 않음
     3x4 투영행렬 P를 주고, 여기서 직접 H를 유도해야 함
     (아래 _projection_matrix_to_homography 참고)

좌표계: PnLCalib은 피치 "중심"이 원점 (x: -52.5~52.5, y: -34~34)
       왼쪽 아래 코너가 원점인 좌표계(0~105, 0~68)가 필요하면
       PITCH_LENGTH/2, PITCH_WIDTH/2 만큼 평행이동해서 써야 함
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image

# external/PnLCalib을 import 경로에 추가
PNLCALIB_ROOT = Path(__file__).resolve().parents[2] / "external" / "PnLCalib"
sys.path.insert(0, str(PNLCALIB_ROOT))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@dataclass
class PnLCalibModels:
    """load_models() 리턴값, 매 프레임 다시 로드하지 않도록 한 번만 생성"""
    model_kp: torch.nn.Module
    model_line: torch.nn.Module
    device: str
    resize: T.Resize

def load_models(
        weights_kp_path: str,
        weights_line_path: str,
        cfg_kp_path: str = None,
        cfg_line_path: str = None,
        device: str = "cpu",
) -> PnLCalibModels:
    """
    keypoint 모델과 line 모델 로드, 세션당 한 번만 호출

    weights_kp_path / weights_line_path: PnLCalib README의 weights 테이블에서 
        다운로드한 .pt 파일 경로 (예: weights/SV_kp, weights/SV_lines)
    """
    cfg_kp_path = cfg_kp_path or str(PNLCALIB_ROOT / "config" / "hrnetv2_w48.yaml")
    cfg_line_path = cfg_line_path or str(PNLCALIB_ROOT / "config" / "hrnetv2_w48_l.yaml")

    cfg = yaml.safe_load(open(cfg_kp_path, "r"))
    cfg_l = yaml.safe_load(open(cfg_line_path, "r"))

    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(torch.load(weights_kp_path, map_location=device))
    model_kp.to(device).eval()

    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(torch.load(weights_line_path, map_location=device))
    model_line.to(device).eval()

    resize = T.Resize((540, 960))

    return PnLCalibModels(model_kp=model_kp, model_line=model_line, device=device, resize=resize)


def _run_keypoint_inference(frame_bgr: np.ndarray, models: PnLCalibModels,
                            kp_threshold: float, line_threshold: float):
    """
    inference.py의 inference() 함수와 동일한 전처리/후처리
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    tensor = f.to_tensor(pil_frame).float().unsqueeze(0)

    if tensor.size()[-2:] != (540, 960):
        tensor = models.resize(tensor)
    tensor = tensor.to(models.device)

    b, c, h, w = tensor.size()
    with torch.no_grad():
        heatmaps = models.model_kp(tensor)
        heatmaps_l = models.model_line(tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize = True)

    return kp_dict, lines_dict


def _projection_from_cam_params(final_params_dict: dict) -> np.ndarray:
    """
    inference.py의 projection_from_cam_params()와 동일
    3x4 투영행렬 P를 리턴 (world 좌표는 피치 중심이 원점)
    """
    cam_params = final_params_dict["cam_params"]
    fx = cam_params["x_focal_length"]
    fy = cam_params["y_focal_length"]
    pp = np.array(cam_params["principal_point"])
    pos = np.array(cam_params["position_meters"])
    R = np.array(cam_params["rotation_matrix"])

    It = np.eye(4)[:-1]
    It[:, -1] = -pos

    K = np.array([[fx, 0, pp[0]],
                  [0, fy, pp[1]],
                  [0, 0, 1]])
    P = K @ (R @ It)
    return P


def _projection_matrix_to_homography(P: np.ndarray) -> np.ndarray:
    """
    3x4 투영행렬 P -> 피치(Z=0 평면) -> 이미지 3x3 homography

    피치 위의 점은 항상 Z=0이므로,
    image = P @ [X, Y, 0, 1]^T = [P[:,0], P[:,1], P[:,3]] @ [X, Y, 1]^T
    즉 P의 0,1,3번째 열만 뽑으면 그게 pitch(X, Y) -> image 3x3 homography

    주의: 여기서 나오는 H는 "피치 중심이 원점"인 좌표계(X: -52.5~52.5) 기준
    """
    H = P[:, [0, 1, 3]]
    return H

def get_homography_matrix(
        frame_bgr: np.ndarray,
        models: PnLCalibModels,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
        iwidth: int = None,
        iheight: int = None,
        diagnostics: dict | None = None,
) -> np.ndarray | None:
    """
    wrapper의 메인 진입점

    frame_bgr: cv2.imread() 등으로 읽은 BGR 프레임 한 장
    models: load_models()가 리턴한 번들 (세션당 한 번만 만들어서 재사용)
    kp_threshold: keypoint 검출 신뢰도 임계값 (PnLCalib 원본 기본값)
    line_threshold: line 검출 신뢰도 임계값 (PnLCalib 원본 기본값)
    pnl_refine: True면 검출된 line 정보로 카메라 파라미터를 한 번 더 최적화 (더 정확, 더 느림)
    iwidth, iheight: 원본 프레임 크기, None이면 frame_bgr.shape에서 자동으로 읽음
    diagnostics: 비어있는 dict를 넘기면 n_keypoints, n_lines를 채워서 돌려줌
                 (스팟체크 용도, 기본 사용에는 필요 없음)
    """
    h, w = frame_bgr.shape[:2]
    iwidth = iwidth or w
    iheight = iheight or h

    cam = FramebyFrameCalib(iwidth=iwidth, iheight=iheight, denormalize=True)

    kp_dict, lines_dict = _run_keypoint_inference(frame_bgr, models, kp_threshold, line_threshold)

    if diagnostics is not None:
        diagnostics["n_keypoints"] = len(kp_dict)
        diagnostics["n_lines"] = len(lines_dict)

    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

    if final_params_dict is None:
        return None

    P = _projection_from_cam_params(final_params_dict)
    H = _projection_matrix_to_homography(P)
    return H