"""Headless MuJoCo rendering on MIG slices, where the NVIDIA EGL driver has no graphics support."""

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def working_egl_devices():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Loaded by path: importing mujoco with MUJOCO_GL=egl fails when no default device initializes.
    root = Path(importlib.util.find_spec("mujoco").submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location("_o3_egl_ext", root / "egl" / "egl_ext.py")
    egl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(egl)
    working = []
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        try:
            display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, device, None)
            if display != egl.EGL_NO_DISPLAY and egl.eglInitialize(display, None, None) == egl.EGL_TRUE:
                working.append(index)
        except Exception:  # noqa: BLE001 - PyOpenGL raises on a failed eglInitialize
            continue
    return working


@contextmanager
def robosuite_import():
    """robosuite asserts at import that MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES.

    Slurm puts a MIG UUID there, unrelated to the software EGL device. CUDA is already initialized,
    so hiding the variable during the import does not change the allocated device.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    hide = visible is not None and not visible.replace(",", "").isdigit()
    if hide:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        yield
    finally:
        if hide:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def main():
    working = working_egl_devices()
    if not working:
        print("no EGL device initializes on this node", file=sys.stderr)
        return 1
    print(working[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
