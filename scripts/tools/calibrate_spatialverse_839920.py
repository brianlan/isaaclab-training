#!/usr/bin/env python3
"""Calibrate spawn and target positions for SpatialVerse scene 839920."""

import json
import os
import sys

import numpy as np
import PIL.Image


def load_occupancy_metadata(scene_dir):
    occupancy_json_path = os.path.join(scene_dir, "occupancy.json")

    if os.path.exists(occupancy_json_path):
        with open(occupancy_json_path, "r") as f:
            return json.load(f)

    structure_json_path = os.path.join(scene_dir, "structure.json")
    if os.path.exists(structure_json_path):
        with open(structure_json_path, "r") as f:
            json.load(f)
        return {"scale": 0.05, "center": [0.0, 0.0, 0.0], "source": "structure.json"}

    raise FileNotFoundError(f"Neither occupancy.json nor structure.json found in {scene_dir}")


def load_occupancy_image(image_path):
    img = PIL.Image.open(image_path).convert("L")
    return np.array(img)


FREESPACE_THRESHOLD = 200


def get_freespace_mask(occupancy_data):
    return occupancy_data > FREESPACE_THRESHOLD


def find_largest_connected_component(mask):
    from scipy import ndimage

    labeled, num_features = ndimage.label(mask, structure=np.ones((3, 3)))
    if num_features == 0:
        raise ValueError("No connected components found in mask")

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()

    return labeled == largest_label


def compute_principal_axis(points):
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvectors[:, -1], center


def compute_percentile_position(points, direction, reference_point, percentile):
    projections = np.dot(points - reference_point, direction)
    percentile_value = np.percentile(projections, percentile)
    return reference_point + direction * percentile_value


def main():
    scene_id = "839920"
    scene_dir = "/ssd5/datasets/InteriorGS/0001_839920"
    occupancy_image_path = os.path.join(scene_dir, "occupancy.png")
    output_path = ".sisyphus/evidence/task-2-scene-calibration.json"

    if not os.path.exists(occupancy_image_path):
        raise FileNotFoundError(f"Occupancy image not found: {occupancy_image_path}")
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading metadata from {scene_dir}...")
    metadata = load_occupancy_metadata(scene_dir)
    scale = metadata["scale"]

    print(f"Loading occupancy image from {occupancy_image_path}...")
    occupancy_data = load_occupancy_image(occupancy_image_path)
    print(f"Occupancy image shape: {occupancy_data.shape}")

    freespace_mask = get_freespace_mask(occupancy_data)
    print(f"Freespace pixels: {freespace_mask.sum()}")

    print("Finding largest connected walkable component...")
    largest_component = find_largest_connected_component(freespace_mask)
    print(f"Largest component pixels: {largest_component.sum()}")

    y_indices, x_indices = np.where(largest_component)
    points_pixels = np.stack([x_indices, y_indices], axis=1)

    height_pixels, width_pixels = occupancy_data.shape

    center_x, center_y, center_z = metadata["center"]
    center_pixel_x = width_pixels / 2
    center_pixel_y = height_pixels / 2

    points_world = np.zeros_like(points_pixels, dtype=np.float64)
    points_world[:, 0] = center_x + (points_pixels[:, 0] - center_pixel_x) * scale
    points_world[:, 1] = center_y - (points_pixels[:, 1] - center_pixel_y) * scale

    print("Computing principal axis...")
    principal_axis, component_center = compute_principal_axis(points_world)
    print(f"Principal axis direction: {principal_axis}")
    print(f"Component center: {component_center}")

    spawn_world = compute_percentile_position(points_world, principal_axis, component_center, 20)
    print(f"Spawn position (20th percentile): {spawn_world}")

    target_world = compute_percentile_position(points_world, principal_axis, component_center, 80)
    print(f"Target position (80th percentile): {target_world}")

    distance_xy = np.linalg.norm(target_world[:2] - spawn_world[:2])
    print(f"Distance (xy): {distance_xy:.2f}m")

    spawn_jitter_xy_m = 0.25
    spawn_yaw_deg = 10.0

    spawn_z = 0.31
    target_z = 0.31

    output_data = {
        "scene_id": scene_id,
        "interiorgs_scene_dir": scene_dir,
        "spawn_center_xyz": [spawn_world[0], spawn_world[1], spawn_z],
        "spawn_jitter_xy_m": spawn_jitter_xy_m,
        "spawn_yaw_deg": spawn_yaw_deg,
        "target_xyz": [target_world[0], target_world[1], target_z],
        "target_distance_xy_m": round(distance_xy, 3),
    }

    print(f"Writing calibration data to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Calibration complete!")
    print(f"Target distance: {distance_xy:.2f}m (expected range: 3m - 12m)")

    if not (3.0 < distance_xy < 12.0):
        print(f"WARNING: Target distance {distance_xy:.2f}m is outside expected range (3m - 12m)")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
