from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_discover_scenes():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "direct"
        / "parking_end2end"
        / "scene_utils.py"
    )
    spec = importlib.util.spec_from_file_location("parking_scene_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load scene_utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.discover_scenes


def main():
    parser = argparse.ArgumentParser(description="Build parking scene composite USDA files from annotations.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing scene folders.")
    parser.add_argument("--output-root", required=True, help="Directory to write generated USDA files into.")
    args = parser.parse_args()

    discover_scenes = _load_discover_scenes()
    scene_infos = discover_scenes(args.dataset_root, args.output_root)
    for scene_info in scene_infos:
        visual = str(scene_info.visual_usd_path) if scene_info.visual_usd_path is not None else "none"
        print(f"{scene_info.scene_id}\t{scene_info.composite_usda_path}\tvisual={visual}")


if __name__ == "__main__":
    main()
