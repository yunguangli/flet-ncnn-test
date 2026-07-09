import os
import base64
import cv2
import flet as ft
from ncnn_detector import YOLOv8NCNNDetector

def main(page: ft.Page):
    page.title = "Flet NCNN Object Detection"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    
    # Check if executed inside a mobile container bundle package layer vs raw desktop script executions
    asset_dir = os.path.dirname(__file__) if "ANDROID_ARGUMENT" in os.environ else "assets"
    
    param_path = os.path.join(asset_dir, "model.ncnn.param")
    bin_path = os.path.join(asset_dir, "model.ncnn.bin")
    img_path = os.path.join(asset_dir, "man.png")

    # Initialize Reusable detector class object tracking system
    try:
        # Note: Toggle use_vulkan=True when compiling with 'flet build apk' for fast Android acceleration
        detector = YOLOv8NCNNDetector(param_path, bin_path, use_vulkan=False)
        init_error = None
    except Exception as e:
        detector = None
        init_error = str(e)

    # UI Widgets Definitions
    result_text = ft.Text("Ready", size=16, weight=ft.FontWeight.BOLD)
    
    display_image = ft.Image(
        src=img_path,
        width=400,
        height=400,
        fit=ft.BoxFit.CONTAIN,
        border_radius=ft.BorderRadius.all(10)
    )

    def on_inference_click(e):
        if detector is None:
            result_text.value = f"Initialization Error: {init_error}"
            page.update()
            return

        try:
            # Drop default confidence limit to 0.10 to safely capture low-contrast laptops on black frames
            processed_mat, analysis_summary = detector.detect_and_render(
                img_path, conf_thresh=0.3, nms_thresh=0.45
            )
            
            # Compress image payload straight to memory matrix strings to avoid caching slowdowns
            _, buffer = cv2.imencode('.png', processed_mat)
            b64_string = base64.b64encode(buffer).decode('utf-8')
            
            # Bind assets straight to state visual engines
            display_image.src_base64 = b64_string
            result_text.value = analysis_summary
            
        except Exception as ex_err:
            result_text.value = f"Runtime Error: {str(ex_err)}"
        
        page.update()

    page.add(
        ft.AppBar(title=ft.Text("NCNN Spatial Holding Analysis")),
        ft.Container(height=10),
        display_image,
        ft.Container(height=10),
        ft.Button("Run NCNN Inference", on_click=on_inference_click),
        ft.Container(height=10),
        result_text
    )

if __name__ == "__main__":
    ft.run(main)
