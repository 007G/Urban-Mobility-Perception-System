import os
import cv2
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches



def read_kitti_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def read_kitti_velodyne_bin(path):
    """
    Returns Nx3 float32 (x,y,z) in LiDAR frame.
    KITTI .bin is float32 [x,y,z,intensity] per point.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find point cloud: {path}")
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts

def parse_calib_file(path):
    """
    Parses KITTI object calib .txt into dict of name->ndarray.
    Keys of interest: P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find calib file: {path}")
    data = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            nums = np.fromstring(value, sep=' ')
            if key.startswith("P"):
                data[key] = nums.reshape(3, 4)
            elif key in ("R0_rect", "R_rect"):  # some variants use R_rect
                data["R0_rect"] = nums.reshape(3, 3)
            elif key == "Tr_velo_to_cam":
                data[key] = nums.reshape(3, 4)
            else:
                data[key] = nums
    required = ["P2", "R0_rect", "Tr_velo_to_cam"]
    for k in required:
        if k not in data:
            raise ValueError(f"Calibration missing required key: {k}")
    return data


def read_kitti_labels(path):
    """
    Reads KITTI Object Detection label file.
    Returns a list of dicts with keys: 'type', 'bbox' = (l,t,r,b), plus other fields.
    Skips 'DontCare' by default.
    Format per line:
      type, truncation, occlusion, alpha, left, top, right, bottom,
      h, w, l, x, y, z, ry [,score]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find label file: {path}")

    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                # Not a standard line; skip
                continue
            obj_type = parts[0]
            if obj_type.lower() == "dontcare":
                continue  # skip DontCare regions for drawing

            trunc = float(parts[1]); occ = int(float(parts[2])); alpha = float(parts[3])
            l, t, r, b = map(float, parts[4:8])  # 2D bbox
            h, w, l3d = map(float, parts[8:11]) # 3D box dims (h,w,l)
            x, y, z = map(float, parts[11:14])  # 3D location in camera coords
            ry = float(parts[14])               # rotation around Y

            lbl = {
                "type": obj_type,
                "truncation": trunc,
                "occlusion": occ,
                "alpha": alpha,
                "bbox": (l, t, r, b),
                "dimensions": (h, w, l3d),
                "location": (x, y, z),
                "rotation_y": ry
            }
            # optional score (for val/test predictions)
            if len(parts) > 15:
                try:
                    lbl["score"] = float(parts[15])
                except:
                    pass
            labels.append(lbl)
    return labels

# --------------- Geometry helpers ---------------
def make_4x4_from_3x4(M34):
    M44 = np.eye(4, dtype=np.float32)
    M44[:3, :4] = M34
    return M44

def make_4x4_from_3x3(R33):
    M44 = np.eye(4, dtype=np.float32)
    M44[:3, :3] = R33
    return M44

def lidar_to_cam2_transform(calib):
    """
    Build 4x4 transform: T_cam2_velo = [R0_rect 0; 0 1] @ [Tr_velo_to_cam; 0 0 0 1]
    """
    R0 = calib["R0_rect"]            # 3x3
    Tr = calib["Tr_velo_to_cam"]     # 3x4
    R0_44 = make_4x4_from_3x3(R0)    # 4x4
    Tr_44 = make_4x4_from_3x4(Tr)    # 4x4
    return R0_44 @ Tr_44             # 4x4

def project_points_cam(pts_cam, K):
    """
    pts_cam: (N,3) in camera-2 coords. K: 3x3 intrinsics.
    Returns (u,v,z) and a valid mask (z>0).
    """
    X, Y, Z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    valid = Z > 0
    Z_safe = np.where(valid, Z, 1.0)  # avoid div-by-zero (masked anyway)
    u = (K[0,0] * (X / Z_safe)) + K[0,2]
    v = (K[1,1] * (Y / Z_safe)) + K[1,2]
    uvz = np.stack([u, v, Z], axis=1)
    return uvz, valid


def get_points_in_2d_bbox(pts_cam, uvz, valid, bbox, img_shape):
    """
    Extract LiDAR points that fall within a 2D bounding box.
    Returns the 3D camera coordinates of points inside the bbox.
    """
    H, W = img_shape
    l, t, r, b = bbox
    
    # Get projected points
    u, v, z = uvz[:,0], uvz[:,1], uvz[:,2]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep = valid & in_img
    
    # Points inside the bounding box
    in_bbox = (u >= l) & (u <= r) & (v >= t) & (v <= b) & keep
    
    if np.sum(in_bbox) == 0:
        return np.array([]).reshape(0, 3)
    
    return pts_cam[in_bbox]

def calculate_object_center_from_lidar(pts_in_bbox):
    """
    Calculate the center position of an object using LiDAR points.
    Uses median to be robust against outliers.
    """
    if pts_in_bbox.shape[0] == 0:
        return None
    
    # Use median for robustness against outliers
    center = np.median(pts_in_bbox, axis=0)
    return center

def calculate_object_center_from_label(label):
    """
    Get object center directly from KITTI label (camera coordinates).
    """
    x, y, z = label["location"]
    return np.array([x, y, z])

def calculate_distance_3d(center1, center2):
    """
    Calculate Euclidean distance between two 3D points.
    """
    if center1 is None or center2 is None:
        return None
    return np.linalg.norm(center1 - center2)

def calculate_all_distances(labels, pts_cam, uvz, valid, img_shape, method="lidar"):
    """
    Calculate distances between all pairs of objects.
    
    method: "lidar" - use LiDAR points to estimate centers
            "label" - use ground truth centers from labels
            "both"  - calculate both and compare
    """
    distances = {}
    object_centers = {}
    
    # Calculate centers for each object
    for i, label in enumerate(labels):
        obj_id = f"{label['type']}_{i}"
        
        if method in ["lidar", "both"]:
            pts_in_bbox = get_points_in_2d_bbox(pts_cam, uvz, valid, label["bbox"], img_shape)
            lidar_center = calculate_object_center_from_lidar(pts_in_bbox)
            object_centers[f"{obj_id}_lidar"] = lidar_center
            
        if method in ["label", "both"]:
            label_center = calculate_object_center_from_label(label)
            object_centers[f"{obj_id}_label"] = label_center
    
    # Calculate distances between all pairs
    if method == "both":
        # Calculate distances using both methods
        lidar_keys = [k for k in object_centers.keys() if k.endswith("_lidar")]
        label_keys = [k for k in object_centers.keys() if k.endswith("_label")]
        
        for i, j in combinations(range(len(labels)), 2):
            obj1 = f"{labels[i]['type']}_{i}"
            obj2 = f"{labels[j]['type']}_{j}"
            
            # LiDAR-based distance
            lidar_dist = calculate_distance_3d(
                object_centers.get(f"{obj1}_lidar"),
                object_centers.get(f"{obj2}_lidar")
            )
            
            # Label-based distance
            label_dist = calculate_distance_3d(
                object_centers.get(f"{obj1}_label"),
                object_centers.get(f"{obj2}_label")
            )
            
            distances[f"{obj1}↔{obj2}"] = {
                "lidar": lidar_dist,
                "label": label_dist,
                "difference": abs(lidar_dist - label_dist) if lidar_dist and label_dist else None
            }
    else:
        # Calculate distances using single method
        keys = list(object_centers.keys())
        for i, j in combinations(range(len(keys)), 2):
            key1, key2 = keys[i], keys[j]
            dist = calculate_distance_3d(object_centers[key1], object_centers[key2])
            distances[f"{key1}↔{key2}"] = dist
    
    return distances, object_centers

def display_image_with_matplotlib(img, title="KITTI Visualization"):
    """
    Display image using matplotlib (works in Kaggle/Jupyter)
    """
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_image(img, output_path):
    """
    Save image to file
    """
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Saved image: {output_path}")
    else:
        print(f"Failed to save image: {output_path}")
    return success
CLASS_COLOR = {
    "Car":        (0, 255, 0),
    "Truck":      (0, 150, 0),
    "Pedestrian": (0, 0, 255)
}

def color_for_class(name):
    return CLASS_COLOR.get(name, (200, 200, 200))

def draw_labels_2d(img, labels):
    """
    Draw 2D bounding boxes and class labels on image.
    """
    H, W = img.shape[:2]
    for i, det in enumerate(labels):
        l, t, r, b = det["bbox"]
        # clip to image bounds
        l = int(max(0, min(W-1, l)))
        r = int(max(0, min(W-1, r)))
        t = int(max(0, min(H-1, t)))
        b = int(max(0, min(H-1, b)))

        cls = det["type"]
        color = color_for_class(cls)

        # rectangle
        cv2.rectangle(img, (l, t), (r, b), color, 2)

        # label text with background
        label = f"{cls}_{i}"  # Add index for identification
        if "score" in det:
            label += f" {det['score']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (l, t - th - baseline - 2), (l + tw + 2, t), color, -1)
        cv2.putText(img, label, (l + 1, t - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def draw_distance_lines(img, labels, object_centers, distances, pts_cam, uvz, valid, K):
    """
    Draw lines between object centers with distance annotations.
    """
    H, W = img.shape[:2]
    
    # Project 3D centers to image coordinates for drawing
    centers_2d = {}
    for i, label in enumerate(labels):
        obj_id = f"{label['type']}_{i}"
        
        # Try to get LiDAR center first, fallback to label center
        center_3d = object_centers.get(f"{obj_id}_lidar")
        if center_3d is None:
            center_3d = object_centers.get(f"{obj_id}_label")
        
        if center_3d is not None:
            # Project to image
            center_h = np.array([[center_3d[0], center_3d[1], center_3d[2], 1.0]])
            uvz_center, valid_center = project_points_cam(center_3d.reshape(1, -1), K)
            
            if valid_center[0]:
                u, v = int(uvz_center[0, 0]), int(uvz_center[0, 1])
                if 0 <= u < W and 0 <= v < H:
                    centers_2d[obj_id] = (u, v)
                    # Draw center point
                    cv2.circle(img, (u, v), 5, (255, 255, 255), -1)
                    cv2.circle(img, (u, v), 5, (0, 0, 0), 2)
    
    # Draw distance lines
    drawn_pairs = set()
    for distance_key, distance_info in distances.items():
        if isinstance(distance_info, dict):
            # "both" method - use LiDAR distance if available
            dist = distance_info.get("lidar") or distance_info.get("label")
        else:
            dist = distance_info
            
        if dist is None:
            continue
            
        # Parse object names from key
        obj1_name, obj2_name = distance_key.split("↔")
        obj1_name = obj1_name.replace("_lidar", "").replace("_label", "")
        obj2_name = obj2_name.replace("_lidar", "").replace("_label", "")
        
        # Avoid drawing duplicate lines
        pair = tuple(sorted([obj1_name, obj2_name]))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        
        if obj1_name in centers_2d and obj2_name in centers_2d:
            pt1 = centers_2d[obj1_name]
            pt2 = centers_2d[obj2_name]
            
            # Draw line
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)
            
            # Draw distance text at midpoint
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            
            distance_text = f"{dist:.2f}m"
            (tw, th), baseline = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(img, (mid_x - tw//2 - 2, mid_y - th - 2), 
                         (mid_x + tw//2 + 2, mid_y + 2), (0, 0, 0), -1)
            cv2.putText(img, distance_text, (mid_x - tw//2, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

# --------------- Main ---------------
def main(img_path, bin_path, calib_path, label_path, index):
    # Load data
    img = read_kitti_image(img_path)
    H, W = img.shape[:2]
    pts_velo = read_kitti_velodyne_bin(bin_path)
    calib = parse_calib_file(calib_path)

    # Intrinsics from P2
    P2 = calib["P2"]                  # 3x4
    K  = P2[:, :3]                    # 3x3
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Build LiDAR->cam2 transform
    T_cam2_velo = lidar_to_cam2_transform(calib)  # 4x4

    # Transform LiDAR points to cam2 frame
    N = pts_velo.shape[0]
    pts_h = np.hstack([pts_velo, np.ones((N,1), dtype=np.float32)])  # (N,4)
    pts_cam_h = (T_cam2_velo @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]  # (N,3)

    # Project to image
    uvz, valid = project_points_cam(pts_cam, K)

    # Keep points in front & inside image bounds
    u, v, z = uvz[:,0], uvz[:,1], uvz[:,2]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep = valid & in_img

    u_keep = u[keep].astype(np.int32)
    v_keep = v[keep].astype(np.int32)
    z_keep = z[keep]

    # Depth-based coloring (near→far)
    if z_keep.size > 0:
        z_norm = (z_keep - z_keep.min()) / max(1e-6, (z_keep.max() - z_keep.min()))
        step = max(1, len(u_keep) // 60000)  # throttle if too dense
        for uu, vv, zn in zip(u_keep[::step], v_keep[::step], z_norm[::step]):
            c = int(255 * (1 - zn))
            cv2.circle(img, (uu, vv), 1, (c, 255 - c, 0), -1)

    # Read labels
    try:
        labels = read_kitti_labels(label_path)
        if labels:
            #print(f"Found {len(labels)} objects")
            
            # Calculate distances between objects
            #print("\n=== Calculating Distances ===")
            distances, object_centers = calculate_all_distances(
                labels, pts_cam, uvz, valid, (H, W), method="both"
            )
            
            # Print distance results
            for distance_key, distance_info in distances.items():
                if isinstance(distance_info, dict):
                    #print(f"{distance_key}:")
                    if distance_info["lidar"]:
                        #print(f"  LiDAR-based: {distance_info['lidar']:.2f}m")
                        pass
                    if distance_info["label"]:
                        #print(f"  Label-based: {distance_info['label']:.2f}m")
                        pass
                    if distance_info["difference"]:
                        #print(f"  Difference: {distance_info['difference']:.2f}m")
                        pass
                    print()
                else:
                    print(f"{distance_key}: {distance_info:.2f}m")
            
            # Draw labels and distances
            draw_labels_2d(img, labels)
            draw_distance_lines(img, labels, object_centers, distances, pts_cam, uvz, valid, K)
            
            #print(f"Drew {len(labels)} 2D boxes and distances from: {label_path}")
        else:
            print("No drawable labels found (maybe only DontCare).")
    except FileNotFoundError as e:
        print(str(e))

    # Display and save results
    display_image_with_matplotlib(img, f"KITTI {index:06d} with Distances")
    
    out_path = f"overlay_with_distances_{index:06d}.png"
    save_image(img, out_path)

# import random

# if __name__ == "__main__":
#     for i in range(30):
#         index = random.randint(0, 7499)   # random number between 0 and 2499
#         idx_str = f"{index:06d}"          # zero-pad to 6 digits → e.g. "000025"
    
#         img_path   = f"/kaggle/input/kitti-3d-object-detection-dataset/training/image_2/{idx_str}.png"
#         bin_path   = f"/kaggle/input/kitti-3d-object-detection-dataset/training/velodyne/{idx_str}.bin"
#         calib_path = f"/kaggle/input/kitti-3d-object-detection-dataset/training/calib/{idx_str}.txt"
#         label_path = f"/kaggle/input/kitti-3d-object-detection-dataset/training/label_2/{idx_str}.txt"
    
#         main(img_path, bin_path, calib_path, label_path, index)

# for overlaying in seq:
if __name__ == "__main__":

    BASE_PATH = "/kaggle/input/kitti-3d-object-detection-dataset/training"

    img_dir   = os.path.join(BASE_PATH, "image_2")
    lidar_dir = os.path.join(BASE_PATH, "velodyne")
    calib_dir = os.path.join(BASE_PATH, "calib")
    label_dir = os.path.join(BASE_PATH, "label_2")

    # Get all image files in sorted order
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(".png")
    ])

    # Take first 300 sequential frames
    img_files = img_files[:30]

    for frame_id, img_file in enumerate(img_files):
        idx_str = img_file.replace(".png", "")  # e.g. "000000"

        img_path   = os.path.join(img_dir, img_file)
        bin_path   = os.path.join(lidar_dir, f"{idx_str}.bin")
        calib_path = os.path.join(calib_dir, f"{idx_str}.txt")
        label_path = os.path.join(label_dir, f"{idx_str}.txt")

        print(f"Processing frame {frame_id} (KITTI index {idx_str})")

        main(
            img_path=img_path,
            bin_path=bin_path,
            calib_path=calib_path,
            label_path=label_path,
            index=frame_id
        )


# if __name__ == "__main__":

#     BASE_PATH = "/kaggle/input/kitti-3d-object-detection-dataset/training"

#     img_dir   = os.path.join(BASE_PATH, "image_2")
#     lidar_dir = os.path.join(BASE_PATH, "velodyne")
#     calib_dir = os.path.join(BASE_PATH, "calib")
#     label_dir = os.path.join(BASE_PATH, "label_2")

#     # Sequential KITTI frames
#     img_files = sorted([
#         f for f in os.listdir(img_dir)
#         if f.endswith(".png")
#     ])

#     NUM_FRAMES = 300        # 👈change to len(img_files) for ALL frames
#     FPS = 10

#     img_files = img_files[:NUM_FRAMES]

#     video_writer = None

#     for frame_id, img_file in enumerate(img_files):
#         idx_str = img_file.replace(".png", "")

#         img_path   = os.path.join(img_dir, img_file)
#         bin_path   = os.path.join(lidar_dir, f"{idx_str}.bin")
#         calib_path = os.path.join(calib_dir, f"{idx_str}.txt")
#         label_path = os.path.join(label_dir, f"{idx_str}.txt")

#         print(f"[INFO] Processing frame {frame_id}/{NUM_FRAMES} (KITTI {idx_str})")

#         # main() MUST return the annotated frame
#         frame = main(
#             img_path=img_path,
#             bin_path=bin_path,
#             calib_path=calib_path,
#             label_path=label_path,
#             index=frame_id
#         )

#         # Safety check
#         if frame is None:
#             print(f"[WARN] Skipping frame {frame_id}")
#             continue

#         # Initialize VideoWriter ONCE
#         if video_writer is None:
#             h, w = frame.shape[:2]
#             video_writer = cv2.VideoWriter(
#                 "kitti_fusion_video.mp4",
#                 cv2.VideoWriter_fourcc(*"mp4v"),
#                 FPS,
#                 (w, h)
#             )

#         video_writer.write(frame)

#     if video_writer is not None:
#         video_writer.release()

#     print("Video saved as: kitti_fusion_video.mp4")

