from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from pxr import Usd, UsdGeom


_IGNORED_OBSTACLE_TYPES = {
    "class.misc.dont_care_region",
    "class.parking.text_icon",
    "class.road_marker.geo_shape",
    "class.road_marker.arrow",
}


@dataclass(slots=True)
class ParkingSlot:
    center_xy: tuple[float, float]
    yaw: float
    vertices_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    parkable: bool


@dataclass(slots=True)
class ObstacleBox:
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    yaw: float
    height: float
    z: float
    obj_type: str


@dataclass(slots=True)
class SceneInfo:
    scene_id: str
    scene_dir: Path
    slots: list[ParkingSlot]
    obstacles: list[ObstacleBox]
    bounds_xy: tuple[tuple[float, float], tuple[float, float]]
    visual_usd_path: Path | None
    visual_reference: str | None
    composite_usda_path: Path


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _vertices_to_xy(vertices: list[float]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts = [tuple(vertices[i : i + 2]) for i in range(0, len(vertices), 3)]
    return pts[0], pts[1], pts[2], pts[3]


def _slot_yaw(vertices_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> float:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = vertices_xy
    fx = 0.5 * ((x1 - x0) + (x2 - x3))
    fy = 0.5 * ((y1 - y0) + (y2 - y3))
    return math.atan2(fy, fx)


def _slot_center(vertices_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in vertices_xy]
    ys = [p[1] for p in vertices_xy]
    return sum(xs) / 4.0, sum(ys) / 4.0


def _resolve_visual_usd(scene_dir: Path, scene_asset_dir: Path) -> Path | None:
    candidates = [
        scene_asset_dir / "visual" / "scene.usdz",
        scene_asset_dir / "visual" / "scene.usd",
        scene_asset_dir / "visual" / "scene.usda",
        scene_dir / "visual" / "scene.usdz",
        scene_dir / "visual" / "scene.usd",
        scene_dir / "visual" / "scene.usda",
        scene_dir / "splat" / "scene.usdz",
        scene_dir / "splat" / "scene.usd",
        scene_dir / "splat" / "scene.usda",
        scene_dir / "splat" / "point_cloud.usdz",
        scene_dir / "splat" / "point_cloud.usd",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _make_visual_reference(visual_usd_path: Path | None) -> str | None:
    if visual_usd_path is None:
        return None
    if visual_usd_path.suffix.lower() == ".usdz":
        return f"{visual_usd_path}[gauss.usda]"
    return str(visual_usd_path)


def load_scene_info(scene_dir: str | Path, asset_root: str | Path) -> SceneInfo:
    scene_dir = Path(scene_dir)
    asset_root = Path(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    label_data = json.loads((scene_dir / "label" / "point_cloud.json").read_text())
    polyline_data = json.loads((scene_dir / "polylines" / "point_cloud.json").read_text())

    slots: list[ParkingSlot] = []
    for item in polyline_data:
        if item.get("obj_type") != "class.parking.parking_slot":
            continue
        vertices_xy = _vertices_to_xy(item["vertices"])
        slots.append(
            ParkingSlot(
                center_xy=_slot_center(vertices_xy),
                yaw=_slot_yaw(vertices_xy),
                vertices_xy=vertices_xy,
                parkable=item.get("obj_attr", {}).get("attr.parking.parking_slot.is_parkable", "").endswith(".true"),
            )
        )

    obstacles: list[ObstacleBox] = []
    for item in label_data:
        obj_type = item.get("obj_type", "")
        if obj_type in _IGNORED_OBSTACLE_TYPES:
            continue
        psr = item.get("psr", {})
        pos = psr.get("position", {})
        rot = psr.get("rotation", {})
        scale = psr.get("scale", {})
        obstacles.append(
            ObstacleBox(
                center_xy=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0))),
                size_xy=(abs(float(scale.get("x", 0.0))), abs(float(scale.get("y", 0.0)))),
                yaw=wrap_to_pi(float(rot.get("z", 0.0))),
                height=abs(float(scale.get("z", 0.0))),
                z=float(pos.get("z", 0.0)),
                obj_type=obj_type,
            )
        )

    xs: list[float] = []
    ys: list[float] = []
    for slot in slots:
        for x, y in slot.vertices_xy:
            xs.append(x)
            ys.append(y)
    for obstacle in obstacles:
        half_x = obstacle.size_xy[0] * 0.5
        half_y = obstacle.size_xy[1] * 0.5
        xs.extend([obstacle.center_xy[0] - half_x, obstacle.center_xy[0] + half_x])
        ys.extend([obstacle.center_xy[1] - half_y, obstacle.center_xy[1] + half_y])
    if not xs:
        xs = [-10.0, 10.0]
        ys = [-10.0, 10.0]
    margin = 3.0
    bounds_xy = ((min(xs) - margin, max(xs) + margin), (min(ys) - margin, max(ys) + margin))

    scene_id = scene_dir.name
    scene_asset_dir = asset_root / scene_id
    scene_asset_dir.mkdir(parents=True, exist_ok=True)
    composite_usda_path = scene_asset_dir / "scene_composite.usda"
    visual_usd_path = _resolve_visual_usd(scene_dir, scene_asset_dir)

    return SceneInfo(
        scene_id=scene_id,
        scene_dir=scene_dir,
        slots=slots,
        obstacles=obstacles,
        bounds_xy=bounds_xy,
        visual_usd_path=visual_usd_path,
        visual_reference=_make_visual_reference(visual_usd_path),
        composite_usda_path=composite_usda_path,
    )


def build_scene_composite_usda(scene_info: SceneInfo) -> Path:
    stage = Usd.Stage.CreateNew(str(scene_info.composite_usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/Scene", "Xform")
    stage.SetDefaultPrim(root)

    if scene_info.visual_reference is not None:
        visual_prim = stage.DefinePrim("/Scene/Visual", "Xform")
        visual_prim.GetReferences().AddReference(scene_info.visual_reference)

    stage.GetRootLayer().Save()
    return scene_info.composite_usda_path


def discover_scenes(dataset_root: str | Path, asset_root: str | Path) -> list[SceneInfo]:
    dataset_root = Path(dataset_root)
    scene_infos: list[SceneInfo] = []
    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if not (scene_dir / "label" / "point_cloud.json").exists():
            continue
        if not (scene_dir / "polylines" / "point_cloud.json").exists():
            continue
        scene_info = load_scene_info(scene_dir, asset_root)
        build_scene_composite_usda(scene_info)
        scene_infos.append(scene_info)
    if not scene_infos:
        raise FileNotFoundError(f"No valid parking scenes found under: {dataset_root}")
    return scene_infos
