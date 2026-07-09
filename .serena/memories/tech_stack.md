# Tech Stack

- **Language**: Python >=3.10 (dev machine has 3.14)
- **UI**: Flet >=0.85.3
- **Camera**: flet-camera (web / Android / iOS only; desktop uses OpenCV)
- **Inference**: Tencent NCNN Python package; YOLO NCNN export via Ultralytics (yolo11n → `src/assets/model.ncnn.*`)
- **CV/array**: opencv-python, numpy
- **Package manager**: uv + `pyproject.toml` + committed `uv.lock` (no requirements.txt)
- **Tests**: pytest (`asyncio_mode = auto`, `testpaths = ["tests"]`); also `flet test` / `flet[test]` extra
- **Build**: flet CLI (`flet build apk|web|linux|...`)
- **Dev group**: flet-cli, flet-desktop, flet-web, flet[test]
- Product metadata: `[tool.flet]` org=com.mycompany, product=laptop-detect
