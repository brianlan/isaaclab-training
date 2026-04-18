from __future__ import annotations

import copy
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import virtual_camera as vc
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_callable, unpad_trajectories

from ..camera_calib import PARKING_CAMERA_MOUNTS, PARKING_LOOKUP_FISHEYE_CFG


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)


def _get_voxel_points_in_ego(
    voxel_shape: tuple[int, int, int], voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> torch.Tensor:
    z_count, x_count, y_count = voxel_shape

    fz = z_count / (voxel_range[0][1] - voxel_range[0][0])
    fx = x_count / (voxel_range[1][1] - voxel_range[1][0])
    fy = y_count / (voxel_range[2][1] - voxel_range[2][0])

    cz = -voxel_range[0][0] * fz - 0.5
    cx = -voxel_range[1][0] * fx - 0.5
    cy = -voxel_range[2][0] * fy - 0.5

    vzz, vxx, vyy = torch.meshgrid(
        torch.arange(z_count, dtype=torch.float32),
        torch.arange(x_count, dtype=torch.float32),
        torch.arange(y_count, dtype=torch.float32),
        indexing="ij",
    )

    zz = (vzz - cz) / fz
    xx = (vxx - cx) / fx
    yy = (vyy - cy) / fy
    return torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=0)


def _get_voxel_points_in_ego_numpy(
    voxel_shape: tuple[int, int, int], voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> np.ndarray:
    z_count, x_count, y_count = voxel_shape

    fz = z_count / (voxel_range[0][1] - voxel_range[0][0])
    fx = x_count / (voxel_range[1][1] - voxel_range[1][0])
    fy = y_count / (voxel_range[2][1] - voxel_range[2][0])

    cz = -voxel_range[0][0] * fz - 0.5
    cx = -voxel_range[1][0] * fx - 0.5
    cy = -voxel_range[2][0] * fy - 0.5

    vzz, vxx, vyy = np.meshgrid(np.arange(z_count), np.arange(x_count), np.arange(y_count), indexing="ij")
    zz = (vzz - cz) / fz
    xx = (vxx - cx) / fx
    yy = (vyy - cy) / fy
    return np.array([xx, yy, zz], dtype=np.float32).reshape(3, -1)


class IsaacCameraImage:
    def __init__(
        self,
        cam_id: str,
        cam_type: str,
        img: np.ndarray,
        ego_mask: np.ndarray,
        extrinsic: tuple[np.ndarray, np.ndarray],
        intrinsic: np.ndarray | list[float] | tuple[float, ...],
    ):
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic


class IsaacCameraImageSet:
    def __init__(self, transformables: dict[str, IsaacCameraImage]):
        self.transformables = transformables


class VoxelLookUpTableGenerator:
    def __init__(
        self,
        voxel_feature_config: dict,
        camera_feature_configs: dict[str, dict],
        bilinear_interpolation: bool = False,
    ):
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_shape = self.voxel_feature_config["voxel_shape"]
        self.voxel_range = np.float32(self.voxel_feature_config["voxel_range"])
        self.bilinear_interpolation = bilinear_interpolation
        self.voxel_points = _get_voxel_points_in_ego_numpy(self.voxel_shape, self.voxel_range)

    def generate(self, camera_images: IsaacCameraImageSet | dict[str, IsaacCameraImage], seed: int | None = None):
        ego_distance_max = self.voxel_feature_config["ego_distance_max"]
        ego_distance_step = self.voxel_feature_config["ego_distance_step"]
        distances_ego = np.linalg.norm(self.voxel_points, axis=0)
        distance_bins = np.arange(0, ego_distance_max + ego_distance_step, ego_distance_step)

        lut = {}
        keys = []
        density_maps = []
        if isinstance(camera_images, IsaacCameraImageSet):
            camera_images = camera_images.transformables

        for key in camera_images:
            keys.append(key)
            camera_image = camera_images[key]
            r_ec, t_ec = camera_image.extrinsic
            camera_points = r_ec.T @ (self.voxel_points - t_ec[..., None])
            downscale = self.camera_feature_configs[key]["feature_downscale"]
            resolution = np.array(camera_image.img.shape[:2][::-1]) // downscale
            intrinsic = np.array(camera_image.intrinsic, dtype=np.float32)
            intrinsic[:4] = (intrinsic[:4] + [0.5, 0.5, 0, 0]) / downscale - [0.5, 0.5, 0, 0]
            camera_class = getattr(vc, camera_image.cam_type)
            camera = camera_class(tuple(resolution.tolist()), camera_image.extrinsic, intrinsic)
            uu_float, vv_float = camera.project_points_from_camera_to_image(camera_points)

            ray_distance_start = self.camera_feature_configs[key]["ray_distance_start"]
            ray_distance_step = self.camera_feature_configs[key]["ray_distance_step"]
            ray_distance_num_channel = self.camera_feature_configs[key]["ray_distance_num_channel"]
            distances = np.linalg.norm(camera_points, axis=0)
            dd_float = (distances - ray_distance_start) / ray_distance_step

            uu = np.round(uu_float).astype(int)
            vv = np.round(vv_float).astype(int)
            dd = np.round(dd_float).astype(int)
            dd[dd >= ray_distance_num_channel] = ray_distance_num_channel - 1

            valid_map = (uu >= 0) * (uu < resolution[0]) * (vv >= 0) * (vv < resolution[1]) * (camera_points[2] > 0)
            uv_mask = cv2.resize(camera_image.ego_mask, tuple(resolution.tolist()))
            valid_map *= uv_mask[vv * valid_map, uu * valid_map].astype(bool)

            uu_float[~valid_map] = -1
            vv_float[~valid_map] = -1
            dd_float[~valid_map] = -1
            uu[~valid_map] = -1
            vv[~valid_map] = -1
            dd[~valid_map] = -1
            lut[key] = dict(uu=uu, vv=vv, dd=dd, valid_map=valid_map)

            if self.bilinear_interpolation:
                uu_floor = np.floor(uu_float).astype(int)
                vv_floor = np.floor(vv_float).astype(int)
                uu_ceil = np.ceil(uu_float).astype(int)
                vv_ceil = np.ceil(vv_float).astype(int)
                uu_bilinear_weight = np.ceil(uu_float) - uu_float
                vv_bilinear_weight = np.ceil(vv_float) - vv_float
                valid_map_bilinear = (
                    (uu_floor >= 0)
                    * (uu_ceil < resolution[0])
                    * (vv_floor >= 0)
                    * (vv_ceil < resolution[1])
                    * (camera_points[2] > 0)
                )
                valid_map_bilinear *= uv_mask[vv * valid_map_bilinear, uu * valid_map_bilinear].astype(bool)
                uu_floor[~valid_map_bilinear] = -1
                vv_floor[~valid_map_bilinear] = -1
                uu_ceil[~valid_map_bilinear] = -1
                vv_ceil[~valid_map_bilinear] = -1
                lut[key].update(
                    dict(
                        uu_floor=uu_floor,
                        vv_floor=vv_floor,
                        uu_ceil=uu_ceil,
                        vv_ceil=vv_ceil,
                        uu_bilinear_weight=uu_bilinear_weight,
                        vv_bilinear_weight=vv_bilinear_weight,
                        valid_map_bilinear=valid_map_bilinear,
                    )
                )

            density_map = np.zeros_like(distances_ego)
            for dist_ind in range(len(distance_bins) - 1):
                valid_dist = (distances_ego > distance_bins[dist_ind]) & (distances_ego <= distance_bins[dist_ind + 1])
                shell_mask = valid_dist * valid_map
                valid_u = uu_float[shell_mask]
                valid_v = vv_float[shell_mask]
                if len(valid_u) > 0 and len(valid_v) > 0:
                    density_map[shell_mask] = (valid_u.max() - valid_u.min()) * (valid_v.max() - valid_v.min())
            density_maps.append(density_map)

        density_maps = np.stack(density_maps)
        density_maps_norm = density_maps / (density_maps.sum(axis=0, keepdims=True) + 1.0e-5)
        for key_index, key in enumerate(keys):
            lut[key]["norm_density_map"] = density_maps_norm[key_index]

        rng = np.random.default_rng(seed)
        random_map = rng.random(distances_ego.shape)
        for key_index, key in enumerate(keys):
            acc_density_min = density_maps_norm[:key_index].sum(axis=0)
            acc_density_max = density_maps_norm[: key_index + 1].sum(axis=0)
            lut[key]["valid_map_sampled"] = (random_map >= acc_density_min) * (random_map < acc_density_max)
        return lut


def _build_camera_to_ego_rotation(yaw_deg: float, pitch_deg: float) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    forward = torch.tensor(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=torch.float32,
    )
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    right = _normalize(torch.cross(forward, world_up, dim=0))
    down = _normalize(torch.cross(forward, right, dim=0))
    return torch.stack([right, down, forward], dim=1)


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        relu6: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.act = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class OSABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        repeat: int = 3,
        final_dilation: int = 1,
        relu6: bool = True,
        with_reduce: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvBN(in_channels, mid_channels, stride=stride, padding=1, relu6=relu6))
        for _ in range(repeat - 2):
            self.layers.append(ConvBN(mid_channels, mid_channels, padding=1, relu6=relu6))
        self.layers.append(ConvBN(mid_channels, mid_channels, padding=final_dilation, dilation=final_dilation, relu6=relu6))
        self.reduce = ConvBN(mid_channels * repeat, out_channels, kernel_size=1, padding=0, relu6=relu6) if with_reduce else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        if self.reduce is not None:
            x = self.reduce(x)
        return x


class VoVNetSlimFPN(nn.Module):
    def __init__(self, out_channels: int = 80, relu6: bool = True):
        super().__init__()
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
        self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.out = OSABlock(256, 96, stride=1, repeat=3, with_reduce=False, relu6=relu6)
        self.up_linear = ConvBN(288, out_channels, kernel_size=1, padding=0, relu6=relu6)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.out_relu = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)
        p4 = torch.cat([self.p4_up(osa5), osa4], dim=1)
        p3 = torch.cat([self.p3_up(self.p3_linear(p4)), osa3], dim=1)
        out = self.up(self.up_linear(self.out(p3)))
        return self.out_relu(self.out_bn(out))


class FastRaySpatialTransform(nn.Module):
    def __init__(
        self,
        camera_keys: list[str],
        image_shape: tuple[int, int],
        feature_downscale: int,
        voxel_shape: tuple[int, int, int],
        voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        ray_distance_num_channel: int = 32,
        ray_distance_start: float = 0.5,
        ray_distance_step: float = 0.5,
        ego_distance_max: float = 16.0,
        ego_distance_step: float = 2.0,
        fusion_mode: str = "bilinear_weighted",
        camera_mounts: dict[str, dict] | None = None,
        fisheye_cfg: dict[str, float] | None = None,
    ):
        super().__init__()
        if fusion_mode not in {"weighted", "bilinear_weighted"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.camera_keys = camera_keys
        self.voxel_shape = voxel_shape
        self.fusion_mode = fusion_mode
        self.feature_height = image_shape[0] // feature_downscale
        self.feature_width = image_shape[1] // feature_downscale

        if camera_mounts is None:
            camera_mounts = PARKING_CAMERA_MOUNTS
        if fisheye_cfg is None:
            fisheye_cfg = PARKING_LOOKUP_FISHEYE_CFG
        if "fx" not in fisheye_cfg:
            max_fov = float(fisheye_cfg.get("max_fov", 200.0))
            derived_fx = float(fisheye_cfg.get("nominal_width", image_shape[1])) / math.radians(max_fov)
            derived_fy = derived_fx
            fisheye_cfg = {
                "cx": float(fisheye_cfg.get("optical_centre_x", image_shape[1] * 0.5)),
                "cy": float(fisheye_cfg.get("optical_centre_y", image_shape[0] * 0.5)),
                "fx": derived_fx,
                "fy": derived_fy,
                "p0": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "p3": 0.0,
                "fov": max_fov,
            }

        blank_rgb = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        blank_mask = np.ones((self.feature_height, self.feature_width), dtype=np.uint8)
        camera_images = {}
        for cam_key in camera_keys:
            mount = camera_mounts[cam_key]
            rotation = _build_camera_to_ego_rotation(mount["yaw_deg"], mount["pitch_deg"]).cpu().numpy().astype(np.float32)
            translation = np.asarray(mount["pos"], dtype=np.float32)
            intrinsic = np.array(
                [
                    fisheye_cfg["cx"],
                    fisheye_cfg["cy"],
                    fisheye_cfg["fx"],
                    fisheye_cfg["fy"],
                    fisheye_cfg["p0"],
                    fisheye_cfg["p1"],
                    fisheye_cfg["p2"],
                    fisheye_cfg["p3"],
                ],
                dtype=np.float32,
            )
            camera_images[cam_key] = IsaacCameraImage(
                cam_id=cam_key,
                cam_type="FisheyeCamera",
                img=blank_rgb,
                ego_mask=blank_mask,
                extrinsic=(rotation, translation),
                intrinsic=intrinsic,
            )

        voxel_feature_config = {
            "voxel_shape": voxel_shape,
            "voxel_range": voxel_range,
            "ego_distance_max": ego_distance_max,
            "ego_distance_step": ego_distance_step,
        }
        camera_feature_configs = {
            cam_key: {
                "ray_distance_num_channel": ray_distance_num_channel,
                "ray_distance_start": ray_distance_start,
                "ray_distance_step": ray_distance_step,
                "feature_downscale": feature_downscale,
            }
            for cam_key in camera_keys
        }
        lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=voxel_feature_config,
            camera_feature_configs=camera_feature_configs,
            bilinear_interpolation=fusion_mode == "bilinear_weighted",
        )
        generated = lut_gen.generate(IsaacCameraImageSet(camera_images))

        self.register_buffer("voxel_points", torch.from_numpy(lut_gen.voxel_points), persistent=False)
        for cam_key in camera_keys:
            cam_lookup = generated[cam_key]
            valid_idx = torch.from_numpy(np.nonzero(cam_lookup["valid_map"])[0]).long()
            self.register_buffer(f"{cam_key}_valid_idx", valid_idx, persistent=False)
            self.register_buffer(f"{cam_key}_uu", torch.from_numpy(cam_lookup["uu"][cam_lookup["valid_map"]]).long(), persistent=False)
            self.register_buffer(f"{cam_key}_vv", torch.from_numpy(cam_lookup["vv"][cam_lookup["valid_map"]]).long(), persistent=False)
            self.register_buffer(
                f"{cam_key}_norm_density",
                torch.from_numpy(cam_lookup["norm_density_map"][cam_lookup["valid_map"]]).float(),
                persistent=False,
            )
            self.register_buffer(f"{cam_key}_dd", torch.from_numpy(cam_lookup["dd"][cam_lookup["valid_map"]]).long(), persistent=False)
            if fusion_mode == "bilinear_weighted":
                valid_bilinear = cam_lookup["valid_map_bilinear"]
                bilinear_idx = torch.from_numpy(np.nonzero(valid_bilinear)[0]).long()
                self.register_buffer(f"{cam_key}_valid_bilinear_idx", bilinear_idx, persistent=False)
                self.register_buffer(f"{cam_key}_uu_floor", torch.from_numpy(cam_lookup["uu_floor"][valid_bilinear]).long(), persistent=False)
                self.register_buffer(f"{cam_key}_vv_floor", torch.from_numpy(cam_lookup["vv_floor"][valid_bilinear]).long(), persistent=False)
                self.register_buffer(f"{cam_key}_uu_ceil", torch.from_numpy(cam_lookup["uu_ceil"][valid_bilinear]).long(), persistent=False)
                self.register_buffer(f"{cam_key}_vv_ceil", torch.from_numpy(cam_lookup["vv_ceil"][valid_bilinear]).long(), persistent=False)
                self.register_buffer(
                    f"{cam_key}_uu_weight",
                    torch.from_numpy(cam_lookup["uu_bilinear_weight"][valid_bilinear]).float(),
                    persistent=False,
                )
                self.register_buffer(
                    f"{cam_key}_vv_weight",
                    torch.from_numpy(cam_lookup["vv_bilinear_weight"][valid_bilinear]).float(),
                    persistent=False,
                )
                self.register_buffer(
                    f"{cam_key}_norm_density_bilinear",
                    torch.from_numpy(cam_lookup["norm_density_map"][valid_bilinear]).float(),
                    persistent=False,
                )

    def _lookup(self, cam_key: str, name: str) -> torch.Tensor:
        return getattr(self, f"{cam_key}_{name}")

    def forward(self, camera_feats: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, channels = next(iter(camera_feats.values())).shape[:2]
        z_count, x_count, y_count = self.voxel_shape
        voxel_feats = next(iter(camera_feats.values())).new_zeros((batch_size, channels, z_count * x_count * y_count))

        for cam_key in self.camera_keys:
            feat = camera_feats[cam_key]
            flat = feat.flatten(start_dim=2)
            if self.fusion_mode == "weighted":
                valid_idx = self._lookup(cam_key, "valid_idx")
                pixel_idx = self._lookup(cam_key, "vv") * self.feature_width + self._lookup(cam_key, "uu")
                samples = flat[:, :, pixel_idx]
                voxel_feats[:, :, valid_idx] += self._lookup(cam_key, "norm_density").view(1, 1, -1) * samples
                continue

            valid_idx = self._lookup(cam_key, "valid_bilinear_idx")
            idx_ff = self._lookup(cam_key, "vv_floor") * self.feature_width + self._lookup(cam_key, "uu_floor")
            idx_fc = self._lookup(cam_key, "vv_floor") * self.feature_width + self._lookup(cam_key, "uu_ceil")
            idx_cf = self._lookup(cam_key, "vv_ceil") * self.feature_width + self._lookup(cam_key, "uu_floor")
            idx_cc = self._lookup(cam_key, "vv_ceil") * self.feature_width + self._lookup(cam_key, "uu_ceil")

            uu_weight = self._lookup(cam_key, "uu_weight").view(1, 1, -1).to(feat.dtype)
            vv_weight = self._lookup(cam_key, "vv_weight").view(1, 1, -1).to(feat.dtype)

            samples_ff = flat[:, :, idx_ff]
            samples_fc = flat[:, :, idx_fc]
            samples_cf = flat[:, :, idx_cf]
            samples_cc = flat[:, :, idx_cc]
            interpolated = uu_weight * (samples_ff * vv_weight + samples_cf * (1.0 - vv_weight)) + (1.0 - uu_weight) * (
                samples_fc * vv_weight + samples_cc * (1.0 - vv_weight)
            )
            voxel_feats[:, :, valid_idx] += self._lookup(cam_key, "norm_density_bilinear").view(1, 1, -1) * interpolated

        return voxel_feats.view(batch_size, channels * z_count, x_count, y_count)


class VoxelEncoderFPN(nn.Module):
    def __init__(self, in_channels: int = 128, relu6: bool = True):
        super().__init__()
        self.osa0 = OSABlock(in_channels, 128, 128, stride=1, repeat=3, final_dilation=2, relu6=relu6)
        self.osa1 = OSABlock(128, 128, 128, stride=2, repeat=3, final_dilation=2, relu6=relu6)
        self.osa2 = OSABlock(128, 128, 128, stride=2, repeat=3, final_dilation=2, relu6=relu6)
        self.p1_linear = ConvBN(128, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p1_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.p0_linear = ConvBN(256, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p0_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.out = ConvBN(256, 128, relu6=relu6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        osa0 = self.osa0(x)
        osa1 = self.osa1(osa0)
        osa2 = self.osa2(osa1)
        p1 = torch.cat([osa1, self.p1_up(self.p1_linear(osa2))], dim=1)
        p0 = torch.cat([osa0, self.p0_up(self.p0_linear(p1))], dim=1)
        return self.out(p0)


class ParkingFastRayModel(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set: str,
        output_dim: int,
        camera_keys: list[str],
        goal_key: str,
        kinematics_key: str | None,
        voxel_shape: tuple[int, int, int] = (8, 160, 120),
        voxel_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-1.0, 3.0), (12.0, -12.0), (9.0, -9.0)),
        feature_downscale: int = 4,
        camera_feature_channels: int = 80,
        ray_distance_num_channel: int = 32,
        ray_distance_start: float = 0.5,
        ray_distance_step: float = 0.5,
        ego_distance_max: float = 16.0,
        ego_distance_step: float = 2.0,
        goal_hidden_dims: tuple[int, int] | list[int] = (64, 64),
        head_hidden_dims: tuple[int, int] | list[int] = (256, 128),
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        obs_normalization: bool = False,
        activation: str = "elu",
        distribution_cfg: dict | None = None,
        relu6: bool = True,
        camera_mounts: dict[str, dict] | None = None,
        fisheye_cfg: dict[str, float] | None = None,
        **_: dict,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.camera_keys = camera_keys
        self.goal_key = goal_key
        self.kinematics_key = kinematics_key
        self.obs_normalization = obs_normalization

        image_shape = tuple(obs[camera_keys[0]].shape[-2:])
        self.backbone = VoVNetSlimFPN(out_channels=camera_feature_channels, relu6=relu6)
        self.projector = FastRaySpatialTransform(
            camera_keys=camera_keys,
            image_shape=image_shape,
            feature_downscale=feature_downscale,
            voxel_shape=voxel_shape,
            voxel_range=voxel_range,
            ray_distance_num_channel=ray_distance_num_channel,
            ray_distance_start=ray_distance_start,
            ray_distance_step=ray_distance_step,
            ego_distance_max=ego_distance_max,
            ego_distance_step=ego_distance_step,
            fusion_mode="bilinear_weighted",
            camera_mounts=camera_mounts,
            fisheye_cfg=fisheye_cfg,
        )
        self.voxel_encoder = VoxelEncoderFPN(camera_feature_channels * voxel_shape[0], relu6=relu6)
        self.goal_encoder = MLP(obs[goal_key].shape[-1], goal_hidden_dims[-1], goal_hidden_dims, activation)

        kin_dim = obs[kinematics_key].shape[-1] if kinematics_key is not None and kinematics_key in obs.keys() else 0
        if kin_dim > 0:
            self.kinematics_encoder = MLP(kin_dim, 32, [32], activation)
            kin_out_dim = 32
        else:
            self.kinematics_encoder = None
            kin_out_dim = 0

        if obs_normalization:
            self.goal_normalizer = EmpiricalNormalization(obs[goal_key].shape[-1])
            self.kinematics_normalizer = EmpiricalNormalization(kin_dim) if kin_dim > 0 else nn.Identity()
        else:
            self.goal_normalizer = nn.Identity()
            self.kinematics_normalizer = nn.Identity()

        if head_hidden_dims is None:
            head_hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]

        self.head_input_dim = 128 + goal_hidden_dims[-1] + kin_out_dim
        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.policy_head = MLP(self.head_input_dim, mlp_output_dim, head_hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.policy_head)

    def _extract_image(self, obs, key: str) -> torch.Tensor:
        x = obs[key]
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor for {key}, got shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def get_latent(self, obs) -> torch.Tensor:
        camera_features = {key: self.backbone(self._extract_image(obs, key)) for key in self.camera_keys}
        bev_feats = self.projector(camera_features)
        bev_feats = self.voxel_encoder(bev_feats)
        pooled = F.adaptive_avg_pool2d(bev_feats, output_size=1).flatten(start_dim=1)

        goal = self.goal_normalizer(obs[self.goal_key])
        goal_feat = self.goal_encoder(goal)

        latent_parts = [pooled, goal_feat]
        if self.kinematics_encoder is not None and self.kinematics_key is not None:
            kin = self.kinematics_normalizer(obs[self.kinematics_key])
            latent_parts.append(self.kinematics_encoder(kin))
        return torch.cat(latent_parts, dim=-1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs)
        mlp_output = self.policy_head(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs) -> None:
        if not self.obs_normalization:
            return
        self.goal_normalizer.update(obs[self.goal_key])
        if self.kinematics_encoder is not None and self.kinematics_key is not None:
            self.kinematics_normalizer.update(obs[self.kinematics_key])
