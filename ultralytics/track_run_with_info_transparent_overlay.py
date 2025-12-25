import cv2
from ultralytics import YOLO
import time
import numpy as np # Numpy is required for creating the black block

# 1. Load the model
model = YOLO("/mnt/Data2/check/yolov8n.pt")

# 2. Setup video source and properties
video_path = "/mnt/Data2/check/imp_version/15_minutes_of_heavy_traffic_noise_in_India_14-08-2022_720p.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. Setup Video Writer
output_path = "tracking_results/traffic_run/output_botsort_with_reid_transparent_overlay.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

# Variables for stats
unique_ids = set()
frame_count = 0
prev_time = 0

# --- Configuration for Transparency ---
# Adjust this value between 0.0 (completely invisible background) 
# and 1.0 (completely solid black background).
# 0.5 is a good starting point for a semi-transparent shadow.
OVERLAY_OPACITY = 0.5 
BOX_X1, BOX_Y1 = 10, 10
BOX_X2, BOX_Y2 = 450, 165
# --------------------------------------

print("Starting processing... Press 'q' in the display window to stop early.")

# 4. Run tracking
results = model.track(source=video_path, stream=True, tracker="botsort.yaml", persist=True, verbose=False)
#results = model.track(source=video_path, stream=True, tracker="bytetrack.yaml", persist=True, verbose=False)

for result in results:
    frame_count += 1
    
    # Get the annotated frame (boxes, labels from YOLO)
    annotated_frame = result.plot()
    
    # Calculate Active Tracks and Unique IDs
    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        active_tracks = len(ids)
        unique_ids.update(ids)
    else:
        active_tracks = 0
        
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    # --- ADD TRANSPARENT BACKGROUND ---
    # 1. Extract the Region of Interest (ROI) where the box will go
    # Ensure coordinates don't go out of bounds
    roi = annotated_frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]
    
    # 2. Create a solid black block the same size as the ROI
    # np.zeros creates a black image. roi.shape ensures it matches dimensions.
    black_block = np.zeros(roi.shape, dtype=np.uint8)
    
    # 3. Blend the ROI and the black block together
    # alpha (1.0 - opacity) is the weight for the original image
    # beta (opacity) is the weight for the black block
    blended_roi = cv2.addWeighted(roi, 1.0 - OVERLAY_OPACITY, black_block, OVERLAY_OPACITY, 0)
    
    # 4. Put the blended (darkened) ROI back into the main frame
    annotated_frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2] = blended_roi
    
    # --- ADD TEXT ON TOP ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255) # White text
    thickness = 2
    font_scale = 0.7
    
    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (BOX_X1 + 10, BOX_Y1 + 30), font, font_scale, color, thickness)
    cv2.putText(annotated_frame, f"FPS (Processing): {fps:.1f}", (BOX_X1 + 10, BOX_Y1 + 65), font, font_scale, color, thickness)
    cv2.putText(annotated_frame, f"Active Tracks: {active_tracks}", (BOX_X1 + 10, BOX_Y1 + 100), font, font_scale, color, thickness)
    cv2.putText(annotated_frame, f"Total Unique IDs: {len(unique_ids)}", (BOX_X1 + 10, BOX_Y1 + 135), font, font_scale, color, thickness)
    
    # 5. Write and Show
    out.write(annotated_frame)
    
    # Optional: resize for display if running on a remote server with slow X11 forwarding
    # display_frame = cv2.resize(annotated_frame, (1280, 720)) 
    cv2.imshow("YOLOv8 Transparent Overlay", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopping early...")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete. Video saved to: {output_path}")
print(f"Final Total Unique IDs counted: {len(unique_ids)}")
