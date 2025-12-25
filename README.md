# Urban Mobility Perception System

A perception pipeline for robust multi-object tracking in complex urban intersections. Designed to handle real-world challenges like vehicle occlusions, ID persistence, and edge deployment constraints.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Algorithm Details](#algorithm-details)
- [Comparison with Existing Methods](#comparison-with-existing-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Documentation](#technical-documentation)

---

## Problem Statement

### The Challenge

Urban traffic monitoring systems face several critical challenges:

**Occlusion Handling**: Vehicles frequently disappear behind trucks, buses, or infrastructure for extended periods (5-10 seconds). The system must maintain consistent object IDs throughout these occlusion events.

**Real-Time Performance**: The system must process video streams at real-time speeds while maintaining tracking accuracy.

**Edge Deployment**: The solution must run efficiently on CPU without requiring GPU hardware, making it deployable on edge devices at intersection locations.

**Multi-Class Tracking**: Simultaneous tracking of multiple object classes (cars, trucks, buses, motorcycles, pedestrians) with class-specific handling.

![Problem Visualization]
*Complex intersection scenario showing multiple vehicles and occlusion cases*

### Requirements

- Maintain object IDs through 5-10 second occlusions
- Minimize ID switches and track fragmentation
- Achieve real-time processing speeds on CPU
- Support multi-class object tracking
- Enable future sensor fusion (LiDAR, multi-camera)

---

## Approach

### Analysis of Existing Methods

We analyzed the trade-offs of current tracking approaches:

| Method | Occlusion Handling | Computation | Re-ID Capability | Edge Deployment |
|--------|-------------------|-------------|------------------|-----------------|
| SORT | Poor (1 frame) | Low | None | Yes |
| DeepSORT | Good | High (GPU) | Excellent (CNN) | No |
| ByteTrack | Moderate | Low | None | Yes |
| BoT-SORT | Good | Medium | Optional | Partial |

**Key Insight**: DeepSORT provides robust re-identification but requires GPU. ByteTrack is efficient but lacks appearance-based tracking. We needed to combine these strengths.

### Solution Architecture

Our approach combines efficient motion-based tracking with lightweight appearance features:

```
Input Frame
    |
    v
YOLOv8 Detection
    |
    v
Detection Splitting (High/Low Confidence)
    |
    v
Three-Stage Matching:
  1. High-Confidence IOU Matching
  2. Low-Confidence Recovery  
  3. Appearance-Based Re-ID
    |
    v
Kalman Prediction & Track Management
    |
    v
Output: Tracked Objects [ID, BBox, Class]
```

![Architecture Diagram]
*System architecture showing detection and tracking pipeline*

### Three Core Innovations

**Extended Track Buffer**: Dynamic buffer calculation based on video FPS to handle occlusion duration requirements. For 30 FPS video with 5-second occlusion handling: `track_buffer = 30 fps × 5 seconds = 150 frames`

**Three-Stage Hierarchical Matching**: Extends ByteTrack's two-stage approach with an additional appearance-based recovery stage for long-term occlusions.

**Lightweight Appearance Features**: HSV color histogram representation (62-dimensional) providing CPU-friendly re-identification without deep neural networks.

---

## Algorithm Details

### Tracking Pipeline

The system implements an enhanced ByteTrack algorithm with the following components:

#### 1. Motion Model: Kalman Filter

State vector representation:
```
[x, y, aspect_ratio, height, vx, vy, va, vh]
```

Configuration:
- Position uncertainty: 1/20
- Velocity uncertainty: 1/160

The reduced process noise provides smoother predictions during occlusion periods, preventing track drift when objects are temporarily invisible.

#### 2. Appearance Feature Extraction

**HSV Color Histogram**:
- Hue channel: 30 bins
- Saturation channel: 32 bins
- Total dimension: 62 floats

**Feature Matching**:
```python
similarity = cosine_similarity(feature_track, feature_detection)
appearance_distance = 1.0 - similarity
```

**Temporal Smoothing**: Exponential moving average prevents feature jitter:
```python
new_feature = 0.8 × old_feature + 0.2 × current_observation
```

![Appearance Features]
*HSV histogram extraction from vehicle crops*

#### 3. Three-Stage Association

**Stage 1: High-Confidence Matching**
- Input: Tracked objects + detections (confidence ≥ 0.5)
- Method: IOU-based matching
- Threshold: 0.8
- Purpose: Handle clear, unoccluded objects

**Stage 2: Low-Confidence Recovery**
- Input: Unmatched tracks + low-confidence detections (< 0.5)
- Method: IOU-based matching
- Threshold: 0.5 (relaxed)
- Purpose: Recover partially occluded objects

**Stage 3: Appearance-Based Re-ID**
- Input: Still-unmatched tracks + remaining detections
- Method: Combined metric (50% IOU + 50% appearance)
- Threshold: 0.6
- Purpose: Rescue tracks after extended occlusion

Combined distance calculation:
```python
combined_distance = 0.5 × iou_distance + 0.5 × appearance_distance
```

Hungarian algorithm performs optimal assignment with threshold gating to reject poor matches.

![Three-Stage Process]
*Visual comparison of each matching stage handling different scenarios*

#### 4. Track State Management

Track lifecycle:
```
NEW → TRACKED → LOST → REMOVED
      ↑_________↓
     (re-activate)
```

- NEW: Freshly detected, awaiting confirmation
- TRACKED: Actively matched with detections
- LOST: Missing but kept in memory (within buffer)
- REMOVED: Exceeded buffer limit, permanently deleted

Buffer management ensures tracks persist during occlusion:
```python
if (current_frame - track.last_seen) > track_buffer:
    track.state = REMOVED
```

---

## Comparison with Existing Methods

### Test Configuration

We benchmarked against Ultralytics' latest tracking implementations:

**Trackers Evaluated**:
- ByteTrack (Ultralytics)
- BoT-SORT (Ultralytics)
- BoT-SORT with ReID (Ultralytics)
- Our Enhanced ByteTrack

**Test Environment**:
- Same video input across all methods
- CPU-only mode for fair comparison
- 30 FPS video with multiple occlusion events

### Comparative Analysis

**Computational Efficiency**:

Our tracker achieves similar or better FPS compared to standard ByteTrack while maintaining BoT-SORT level tracking quality:

- HSV histogram extraction: ~0.1ms per detection
- CNN embeddings (BoT-SORT ReID): ~10-50ms per detection
- Speed advantage: 100-500x faster appearance extraction

**Memory Footprint**:

- Our features: 62 floats = 248 bytes per track
- Deep ReID features: 512-2048 floats = 2-8 KB per track
- Memory advantage: 8-32x lower footprint

**Tracking Quality**:

Through testing on urban traffic footage, we observed:

- ByteTrack: Fast but frequent ID switches during occlusions
- BoT-SORT: Good occlusion handling but moderate speed
- BoT-SORT + ReID: Excellent tracking but lowest FPS
- Our Enhanced ByteTrack: BoT-SORT level quality at ByteTrack speed

![Comparison Visualization]
*Side-by-side tracking results showing ID persistence across methods*

### Why This Approach Works

**For Standard ByteTrack**:
- Similar computational cost (tracking overhead ~1-2ms)
- Significantly better occlusion handling
- Appearance features add minimal processing time

**For BoT-SORT with ReID**:
- Comparable tracking accuracy on urban traffic
- Much higher processing speed (no CNN inference)
- Better suited for edge deployment (CPU-only)

The key advantage is achieving deep learning level tracking quality using classical computer vision techniques (HSV histograms), making the system deployable on resource-constrained hardware.

---

## Installation

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration of detection)
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/urban-mobility-tracking.git
cd urban-mobility-tracking

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**requirements.txt**:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
```

---

## Usage

### Basic Usage

```bash
python main.py --video input.mp4 --model yolov8n.pt --output result.mp4
```

### Configuration Examples

**Standard tracking** (handles ~1.5 second occlusions):
```bash
python main.py --video traffic.mp4 --output tracked.mp4
```

**Extended occlusion handling** (5 seconds at 30 FPS):
```bash
python main.py --video traffic.mp4 --track-buffer 150 --output tracked.mp4
```

**Tuned parameters**:
```bash
python main.py \
    --video input.mp4 \
    --track-thresh 0.5 \
    --track-buffer 150 \
    --match-thresh 0.8 \
    --output result.mp4
```

**Disable appearance features** (faster, lower quality):
```bash
python main.py --video input.mp4 --no-appearance --output result.mp4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | Required | Path to input video |
| `--model` | str | yolov8n.pt | YOLOv8 model path |
| `--output` | str | None | Output video path |
| `--track-thresh` | float | 0.5 | Detection confidence threshold |
| `--track-buffer` | int | 50 | Frames to keep lost tracks |
| `--match-thresh` | float | 0.8 | IOU threshold for matching |
| `--no-display` | flag | False | Disable video display |
| `--no-appearance` | flag | False | Disable appearance features |

### Buffer Calculation

For video at FPS `F` requiring `T` seconds occlusion handling:
```
track_buffer = F × T
```

Examples:
- 30 FPS video, 5 second occlusion: `track_buffer = 150`
- 30 FPS video, 10 second occlusion: `track_buffer = 300`
- 25 FPS video, 5 second occlusion: `track_buffer = 125`

---

## Technical Documentation

### Algorithm Selection Rationale

**Why Enhanced ByteTrack?**

ByteTrack provides an efficient foundation with its two-stage detection confidence matching. However, it lacks appearance-based re-identification, limiting its ability to handle extended occlusions.

DeepSORT offers excellent re-identification through CNN-based appearance features but requires GPU acceleration, making edge deployment challenging.

Our approach bridges this gap by adding lightweight appearance features to ByteTrack's efficient architecture. HSV color histograms provide sufficient discriminative power for vehicle tracking (vehicles maintain consistent colors) while remaining computationally efficient.

**Three-Stage Matching Rationale**:

Stage 1 handles the majority of straightforward cases with high confidence and strict IOU matching. Stage 2 recovers partially visible objects using relaxed thresholds. Stage 3 provides a safety net for objects that reappear after extended occlusion, where motion prediction alone is insufficient.

### Implementation Details

**Hardware Utilization**:
- YOLOv8 detection: GPU (if available) or CPU
- Tracking algorithm: CPU only (NumPy/SciPy)
- Appearance extraction: CPU (OpenCV)

**Processing Pipeline**:
1. Detection: ~20-30ms (CPU) or ~3-5ms (GPU)
2. Appearance extraction: ~0.1ms per detection
3. IOU computation: ~1ms
4. Hungarian assignment: ~1ms
5. State update: ~0.5ms

Total tracking overhead: ~3-5ms per frame regardless of image resolution (depends on detection count, not image size).

### Known Limitations

**Extreme Occlusion Duration**: Objects occluded beyond the track buffer limit will be assigned new IDs upon reappearance. Mitigation: Increase buffer or implement post-processing track linking.

**Similar Appearance Confusion**: Objects with very similar colors at the same location may experience ID confusion. Mitigation: Add spatial gating (only compare appearance for nearby objects).

**Lighting Variation**: Significant illumination changes affect HSV histogram matching. Future work: Consider illumination-invariant color spaces or normalization techniques.

**Motion Model Assumption**: Constant velocity assumption breaks down during rapid acceleration or camera motion. Future work: Adaptive motion models or IMU integration.


---

## Output

The system produces an annotated video with:
- Bounding boxes color-coded by object class
- Persistent track IDs throughout occlusions
- Real-time processing statistics overlay

![Output Example]
*Annotated video frame showing tracked vehicles with persistent IDs*

---

## References

- Ultralytics YOLOv8: Object detection framework
- ByteTrack: Multi-object tracking by associating every detection box
- BoT-SORT: Robust associations multi-pedestrian tracking
- KITTI Dataset

---

