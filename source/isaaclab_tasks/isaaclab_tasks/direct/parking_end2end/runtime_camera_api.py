from __future__ import annotations

import importlib
import site
import sys
from pathlib import Path


def get_runtime_camera_class():
    try:
        from isaacsim.sensors.camera import Camera

        return Camera
    except ModuleNotFoundError:
        pass

    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.usd.schema.omni_lens_distortion")
        enable_extension("isaacsim.sensors.camera")
        from isaacsim.sensors.camera import Camera

        return Camera
    except Exception:
        pass

    for site_root in site.getsitepackages():
        ext_root = Path(site_root) / "isaacsim" / "exts" / "isaacsim.sensors.camera"
        if ext_root.is_dir():
            ext_root_str = str(ext_root)
            if ext_root_str not in sys.path:
                sys.path.append(ext_root_str)
            try:
                from isaacsim.core.utils.extensions import enable_extension

                enable_extension("omni.usd.schema.omni_lens_distortion")
                enable_extension("isaacsim.sensors.camera")
            except Exception:
                pass
            module = importlib.import_module("isaacsim.sensors.camera")
            return module.Camera

    raise ModuleNotFoundError("Unable to locate isaacsim.sensors.camera Camera extension path.")
