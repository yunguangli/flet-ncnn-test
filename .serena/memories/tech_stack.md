# Tech Stack

- **Language**: Python 3.10+ (project 3.14 currently)
- **UI framework**: Flet >=0.85.3 (0.85.3 installed)
- **Camera**: flet-camera 0.85.3 (supports web, Android, iOS only)
- **Inference**: Tencent NCNN (python package `ncnn`) with YOLOv11n model
- **Image processing**: opencv-python, numpy
- **Package manager**: uv (with `pyproject.toml`; no requirements.txt)
  - `uv run flet run --web` — dev web server
  - `uv run flet run` — desktop (if unix SPA with local server)
- **Test**: pytest (asyncio_mode = auto), `flet test`
- **Build**: flet CLI (`flet build apk`, `flet build web`, etc.)
- **Version pins**: flet, flet-camera, numpy pinned via uv.lock (lockfile committed)
- **Dev deps** (in `dependency-groups.dev`): flet-cli, flet-desktop, flet-web, flet[test]
