"""
PyTorch Dataset wrapping nuscenes-devkit, producing everything the LSS
pipeline needs for one training sample: 6 camera images + their intrinsics +
camera->ego calibration + ego->global pose + 3D box annotations transformed
into ego frame.

Usage:
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes', verbose=True)
    dataset = NuScenesBEVDataset(nusc, grid=BEVGridConfig(...))
    sample = dataset[0]

Note: this file is written against the real nuscenes-devkit API. It requires
the actual dataset on disk to run end-to-end (see tests/test_nuscenes_dataset.py
for how we validate the logic without needing the full ~/data download, using
a lightweight fake devkit object that mimics the real one's interface).
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from geometry.transforms import (
    RigidTransform,
    camera_to_ego_transform,
    ego_to_global_transform,
    BEVGridConfig,
)

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes detection classes we care about for a first pass (safety-relevant subset)
DETECTION_CLASSES = [
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.rigid",
    "human.pedestrian.adult",
    "vehicle.bicycle",
    "vehicle.motorcycle",
]


@dataclass
class CameraSample:
    image_path: str
    intrinsics: np.ndarray          # (3, 3)
    cam_to_ego: RigidTransform      # fixed per-camera calibration


@dataclass
class BEVSample:
    token: str
    cameras: Dict[str, CameraSample]
    ego_to_global: RigidTransform
    boxes_ego: List[Dict[str, Any]]  # each box already expressed in ego frame


class NuScenesBEVDataset:
    """Thin adapter over nuscenes-devkit that resolves calibration + boxes into
    the frame conventions used by geometry/transforms.py, for a given `sample` token.

    Deliberately NOT a torch.utils.data.Dataset subclass yet — kept dependency-free
    (numpy only) so it's easy to unit test without installing torch. The Phase B
    model code will wrap this in a thin torch.utils.data.Dataset that just calls
    __getitem__ here and converts arrays to tensors.
    """

    def __init__(self, nusc, grid: BEVGridConfig, sample_tokens: List[str] = None):
        self.nusc = nusc
        self.grid = grid
        self.sample_tokens = sample_tokens or [s["token"] for s in nusc.sample]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _load_camera(self, sample: dict, cam_name: str) -> CameraSample:
        cam_token = sample["data"][cam_name]
        sd_record = self.nusc.get("sample_data", cam_token)
        calib = self.nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        intrinsics = np.array(calib["camera_intrinsic"], dtype=np.float64)
        cam_to_ego = camera_to_ego_transform(calib)
        image_path = self.nusc.get_sample_data_path(cam_token)

        return CameraSample(image_path=image_path, intrinsics=intrinsics, cam_to_ego=cam_to_ego)

    def _load_boxes_in_ego(self, sample: dict, ego_pose: dict) -> List[Dict[str, Any]]:
        """Fetch annotated 3D boxes for this sample (given in global frame by
        the devkit's helper) and re-express them in the ego frame, since that's
        the frame our BEV grid and loss functions operate in.
        """
        from geometry.transforms import global_to_ego_transform

        global_to_ego = global_to_ego_transform(ego_pose)
        boxes_ego = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            if ann["category_name"] not in DETECTION_CLASSES:
                continue

            center_global = np.array(ann["translation"], dtype=np.float64)
            center_ego = global_to_ego.apply(center_global)

            # Box heading also needs rotating into ego frame: compose the box's
            # global rotation with global->ego rotation (rotation-only, no translation).
            from pyquaternion import Quaternion
            box_rot_global = Quaternion(ann["rotation"])
            ego_rot_global = Quaternion(matrix=global_to_ego.rotation)
            box_rot_ego = ego_rot_global * box_rot_global

            boxes_ego.append({
                "category": ann["category_name"],
                "center": center_ego,
                "size": np.array(ann["size"], dtype=np.float64),  # (w, l, h), frame-invariant
                "yaw": box_rot_ego.yaw_pitch_roll[0],
                "instance_token": ann["instance_token"],
            })

        return boxes_ego

    def __getitem__(self, idx: int) -> BEVSample:
        token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", token)

        # Ego pose is looked up via any one sensor's sample_data (they share
        # a timestamp closely enough for CAM_FRONT to be the reference here).
        ref_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", ref_sd["ego_pose_token"])
        ego_to_global = ego_to_global_transform(ego_pose)

        cameras = {name: self._load_camera(sample, name) for name in CAMERA_NAMES}
        boxes_ego = self._load_boxes_in_ego(sample, ego_pose)

        return BEVSample(token=token, cameras=cameras, ego_to_global=ego_to_global, boxes_ego=boxes_ego)