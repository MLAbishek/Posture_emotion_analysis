# Mathematics of Posture and Fatigue Analysis

This document explains the mathematical concepts and formulas used in this project to detect posture deviation and fatigue.

## 1. Coordinate System
The computer vision model (MediaPipe) returns **landmarks**. Each landmark represents a specific body part (e.g., nose, shoulder) and has coordinates $(x, y)$:
- $x$: Horizontal position (0 to Image Width)
- $y$: Vertical position (0 to Image Height)
- $(0,0)$ is usually the top-left corner of the image.

## 2. Poses and Vectors
We represent body segments as vectors connecting two landmarks.
For example, the **Neck** can be visualized as a line connecting the **Shoulder** to the **Ear**.

## 3. Posture Analysis: Angle Calculation
To determine if someone is slouching (forward head posture), we calculate the **Neck Inclination Angle**.

We use three points:
1.  **A (Ear)**
2.  **B (Shoulder)** - *The Vertex (pivot point)*
3.  **C (Hip)** - *Reference point for vertical/torso*

The angle $\theta$ at point B is calculated using the **arctangent** function (`atan2`).

### Formula
The angle of a single line segment (e.g., Shoulder to Ear) relative to the horizontal axis is:
$$ \alpha = \text{atan2}(y_{ear} - y_{shoulder}, x_{ear} - x_{shoulder}) $$

To find the angle between two segments (Hip-Shoulder and Shoulder-Ear), we take the difference of their absolute angles:
$$ \theta = |\alpha_{Shoulder \rightarrow Ear} - \alpha_{Shoulder \rightarrow Hip}| $$

*In the code, we simplify this by calculating the angle relative to the vertical or horizontal axis depending on the [calculate_angle](file:///d:/Posture_emotion_analysis/utils/geometry.py#4-22) implementation.*

**Why this works**:
- **Good Posture**: The ear, shoulder, and hip are roughly aligned vertically. The angle is close to **90°** or **180°** (depending on measurement).
- **Slouching**: The head moves forward ($x$ increases), changing the angle.

## 4. Fatigue Analysis: Eye Aspect Ratio (EAR)
We use the **Eye Aspect Ratio (EAR)** to detect drowsiness. The eye is represented by 6 landmarks.

### Formula
$$ \text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||} $$

Where:
- $||p_1 - p_4||$ is the **Horizontal Width** of the eye.
- $||p_2 - p_6||$ and $||p_3 - p_5||$ are the **Vertical Heights** of the eye at two points.

**Logic**:
- **Open Eye**: EAR is constant (approx 0.30 - 0.35).
- **Closed Eye (Blink/Sleep)**: Vertical height approaches 0, so EAR approaches **0.0**.

## 5. Emotion Analysis (Upcoming)
For emotion, we will use **Blendshapes**.
Instead of measuring distances manually, the neural network outputs coefficients (0.0 to 1.0) for specific muscle movements, such as:
- `browDownLeft` (Anger/Focus)
- `mouthSmileLeft` (Happiness)
- `jawOpen` (Surprise)

We interpret these probability scores directly to classify emotions.
