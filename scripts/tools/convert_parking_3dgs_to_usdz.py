from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


DEFAULT_CONVERTER_SCRIPT = "/home/rlan/projects/SAGE-3D_Official/Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py"


def _load_converter(converter_script: Path):
    spec = importlib.util.spec_from_file_location("sage_ply_to_usdz", converter_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load converter script: {converter_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "convert_ply_to_usdz"):
        raise AttributeError(f"Converter script does not expose convert_ply_to_usdz(): {converter_script}")
    return module.convert_ply_to_usdz


def iter_scene_dirs(dataset_root: Path, scene_ids: list[str] | None):
    allowed = set(scene_ids) if scene_ids else None
    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if allowed is not None and scene_dir.name not in allowed:
            continue
        ply_path = scene_dir / "splat" / "point_cloud.ply"
        if ply_path.exists():
            yield scene_dir, ply_path


def main():
    parser = argparse.ArgumentParser(description="Convert parking-scene 3DGS PLY files into SAGE-compatible USDZ assets.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing parking scene folders.")
    parser.add_argument("--output-root", required=True, help="Directory to write generated scene assets into.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Optional subset of scene ids to convert.")
    parser.add_argument("--converter-script", default=DEFAULT_CONVERTER_SCRIPT, help="Path to SAGE PLY-to-USDZ converter script.")
    parser.add_argument("--max-sh-degree", type=int, default=3, help="Maximum SH degree to export.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted USDZ files.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    converter_script = Path(args.converter_script)
    output_root.mkdir(parents=True, exist_ok=True)

    convert_ply_to_usdz = _load_converter(converter_script)

    converted = 0
    skipped = 0
    for scene_dir, ply_path in iter_scene_dirs(dataset_root, args.scene_ids):
        out_dir = output_root / scene_dir.name / "visual"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "scene.usdz"
        if output_file.exists() and not args.overwrite:
            skipped += 1
            print(f"skip\t{scene_dir.name}\t{output_file}")
            continue
        print(f"convert\t{scene_dir.name}\t{ply_path}\t->\t{output_file}")
        convert_ply_to_usdz(str(ply_path), str(output_file), max_sh_degree=args.max_sh_degree)
        converted += 1

    print(f"done\tconverted={converted}\tskipped={skipped}")


if __name__ == "__main__":
    main()
