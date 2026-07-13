"""
Post-process a Flet APK to:
  1. Add the missing ncnn/ncnn.soref marker (ncnn .so is already in lib/)
  2. Replace stub flet_camera-0.0.1 (empty __init__.py) with the real one

Usage:
    python tools/patch_apk_ncnn.py build/apk/laptop-detect.apk
    python tools/patch_apk_ncnn.py build/apk/laptop-detect.apk inspect
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

SOREF_CONTENT = "ncnn.so"
FLET_CAMERA_SRC = os.path.expanduser(
    "~/Coding/python_projects/venv/lib/python3.14/site-packages/flet_camera"
)


def _list_module_dir(path: str) -> list[tuple[str, bytes]]:
    """Recursively list all files in path (excluding __pycache__)."""
    files: list[tuple[str, bytes]] = []
    for root, dirs, names in os.walk(path):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if name.endswith(".pyc"):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, os.path.dirname(path))
            with open(full, "rb") as f:
                files.append((rel, f.read()))
    return files


def add_soref(sp_data: bytes, apk_names: list[str]) -> tuple[bytes, list[str]]:
    """Add ncnn/ncnn.soref to sitepackages."""
    if not apk_names:
        with zipfile.ZipFile(io.BytesIO(sp_data)) as z:
            apk_names = z.namelist()
    else:
        with zipfile.ZipFile(io.BytesIO(sp_data)) as z:
            pass  # just validate

    with zipfile.ZipFile(io.BytesIO(sp_data)) as z:
        if "ncnn/ncnn.soref" in z.namelist():
            print("ncnn/ncnn.soref already exists \u2014 skipping")
            return sp_data, []
        existing = [n for n in apk_names if re.match(r"lib/[^/]+/ncnn\.so$", n)]
        if existing:
            print(f"Found ncnn.so in: {', '.join(existing)}")
        else:
            print("Warning: no lib/<abi>/ncnn.so found — .soref won't help")

    sp_buf = io.BytesIO()
    with zipfile.ZipFile(sp_buf, "w", zipfile.ZIP_DEFLATED) as out:
        with zipfile.ZipFile(io.BytesIO(sp_data)) as inp:
            for item in inp.infolist():
                zi = zipfile.ZipInfo(item.filename)
                zi.compress_type = item.compress_type
                out.writestr(zi, inp.read(item.filename))
        out.writestr("ncnn/ncnn.soref", SOREF_CONTENT)
    print(f"Added ncnn/ncnn.soref \u2192 {SOREF_CONTENT!r}")
    return sp_buf.getvalue(), ["ncnn/ncnn.soref"]


def fix_flet_camera(sp_data: bytes) -> bytes:
    """Replace stub flet_camera with real module from site-packages."""
    # Check if it's already fixed
    with zipfile.ZipFile(io.BytesIO(sp_data)) as z:
        if "flet_camera/camera.py" in z.namelist():
            print("flet_camera/camera.py already exists \u2014 skipping")
            return sp_data

    if not os.path.isdir(FLET_CAMERA_SRC):
        print(f"Warning: {FLET_CAMERA_SRC} not found, can't fix flet_camera")
        return sp_data

    # Read all real flet_camera files
    real_files = _list_module_dir(FLET_CAMERA_SRC)
    real_names = {f[0] for f in real_files}
    print(f"Real flet_camera has {len(real_files)} files: {sorted(real_names)}")

    # Remove stub entries and add real ones
    sp_buf = io.BytesIO()
    with zipfile.ZipFile(sp_buf, "w", zipfile.ZIP_DEFLATED) as out:
        with zipfile.ZipFile(io.BytesIO(sp_data)) as inp:
            for item in inp.infolist():
                # Drop all stub flet_camera entries (except dist-info)
                if item.filename.startswith("flet_camera/") and \
                   not item.filename.startswith("flet_camera-"):
                    continue
                # Drop the dist-info .DS_Store
                if item.filename.endswith(".DS_Store"):
                    continue
                zi = zipfile.ZipInfo(item.filename)
                zi.compress_type = item.compress_type
                out.writestr(zi, inp.read(item.filename))
        # Add real flet_camera files
        for path, content in real_files:
            out.writestr(path, content)
            print(f"  restored {path} ({len(content)} B)")
    print(f"Replaced stub flet_camera with real module ({len(real_files)} files)")
    return sp_buf.getvalue()


def patch_apk(apk_path: str) -> bool:
    apk_path = os.path.abspath(apk_path)
    if not os.path.isfile(apk_path):
        print(f"Error: APK not found at {apk_path}")
        return False

    with open(apk_path, "rb") as f:
        apk_data = f.read()

    apk_zip = zipfile.ZipFile(io.BytesIO(apk_data))
    names = apk_zip.namelist()
    if "assets/sitepackages.zip" not in names:
        print("Error: assets/sitepackages.zip not found")
        return False
    sp_data = apk_zip.read("assets/sitepackages.zip")
    apk_zip.close()

    # Step 1: fix flet_camera
    sp_data = fix_flet_camera(sp_data)

    # Step 2: add ncnn .soref
    sp_data, added = add_soref(sp_data, names)

    # Rebuild APK
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        with zipfile.ZipFile(io.BytesIO(apk_data)) as zin:
            for item in zin.infolist():
                if item.filename == "assets/sitepackages.zip":
                    zi = zipfile.ZipInfo("assets/sitepackages.zip")
                    zi.compress_type = item.compress_type
                    zout.writestr(zi, sp_data)
                else:
                    if not item.filename.endswith(".DS_Store"):
                        zi = zipfile.ZipInfo(item.filename)
                        zi.compress_type = item.compress_type
                        zout.writestr(zi, zin.read(item.filename))

    apk_data = out.getvalue()

    patched = apk_path.replace(".apk", "-patched.apk")
    with open(patched, "wb") as f:
        f.write(apk_data)
    print(f"Patched: {patched} ({len(apk_data)/1024:.0f} KB)")

    zipalign = shutil.which("zipalign")
    if zipalign:
        aligned = patched.replace("-patched", "-aligned")
        subprocess.run([zipalign, "-f", "-p", "4", patched, aligned], check=True)
        os.replace(aligned, patched)
        print("zipalign done")

    print("\nRe-sign & install:")
    ks = os.path.expanduser("~/.android/debug.keystore")
    if os.path.isfile(ks):
        cmd = f"apksigner sign --ks {ks} --ks-pass pass:android --ks-key-alias androiddebugkey {patched}"
    else:
        cmd = f"apksigner sign --ks <keystore> --ks-pass pass:<password> {patched}"
    print(f"  {cmd}")
    print(f"  adb install -r {patched}")
    return True


def inspect_apk(apk_path: str) -> None:
    if not os.path.isfile(apk_path):
        print(f"File not found: {apk_path}")
        return
    with open(apk_path, "rb") as f:
        data = f.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    try:
        print(f"=== {apk_path} ===")
        ncnn_libs = sorted(n for n in names if re.match(r"lib/[^/]+/ncnn\.so$", n))
        if ncnn_libs:
            print("ncnn .so in lib/:")
            for n in ncnn_libs:
                info = z.getinfo(n)
                print(f"  {n}  (compress={info.compress_type}, size={info.file_size})")
        else:
            print("No ncnn .so in lib/")
        if "assets/sitepackages.zip" in names:
            sp_data = z.read("assets/sitepackages.zip")
            with zipfile.ZipFile(io.BytesIO(sp_data)) as sp:
                ncnn = [n for n in sp.namelist() if "ncnn" in n.lower()]
                if ncnn:
                    print(f"ncnn in sitepackages ({len(ncnn)}):")
                    for n in ncnn:
                        print(f"  {n}")
                else:
                    print("No ncnn items in sitepackages")
                fletcam = [n for n in sp.namelist() if "flet_camera" in n.lower()]
                print(f"\nflet_camera in sitepackages ({len(fletcam)}):")
                for n in fletcam:
                    print(f"  {n}")
        else:
            print("No sitepackages.zip in APK")
    finally:
        z.close()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: {sys.argv[0]} <apk> [inspect]")
        sys.exit(1)
    if len(sys.argv) >= 3 and sys.argv[2] == "inspect":
        inspect_apk(sys.argv[1])
    else:
        sys.exit(0 if patch_apk(sys.argv[1]) else 1)
