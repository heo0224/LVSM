import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_camera_wireframe(scale: float = 0.5):
    """
    Returns the vertices of a camera wireframe (pyramid shape) in the camera coordinate frame.
    Default convention: Look at +Z, Up is -Y or +Y depending on system (here assumed +Y down for standard CV).

    Returns:
        points: (5, 3) array of vertices [Center, TopLeft, TopRight, BottomRight, BottomLeft]
    """
    # Camera center
    center = np.array([0, 0, 0])

    # Image plane corners (normalized) at z=1
    # Assuming a simple frustum for visualization
    w, h = 1.0, 0.75  # Aspect ratio 4:3
    z = 1.0

    tl = np.array([-w, -h, z])
    tr = np.array([w, -h, z])
    br = np.array([w, h, z])
    bl = np.array([-w, h, z])

    points = np.array([center, tl, tr, br, bl]) * scale
    return points


def plot_camera(ax, c2w_matrix, color="blue", scale=0.5, label=None):
    """
    Visualizes a camera based on the Camera-to-World matrix.

    Args:
        ax: Matplotlib 3D axis
        c2w_matrix: 4x4 Transformation matrix
        color: Color of the camera mesh
        scale: Size of the camera icon
    """
    # 1. Get wireframe points in camera local frame
    local_points = get_camera_wireframe(scale)

    # 2. Transform points to world coordinates
    # Add homogeneous coordinate (1) to multiply with 4x4 matrix
    ones = np.ones((local_points.shape[0], 1))
    local_points_h = np.hstack([local_points, ones])

    # Apply transformation: World_Points = C2W * Local_Points
    world_points = (c2w_matrix @ local_points_h.T).T
    world_points = world_points[:, :3]  # Drop homogeneous coordinate

    # 3. Draw the camera center
    center = world_points[0]
    ax.scatter(center[0], center[1], center[2], color=color, s=50, label=label)

    # 4. Draw the frustum lines
    # Connect center to 4 corners
    for i in range(1, 5):
        ax.plot(
            [center[0], world_points[i, 0]],
            [center[1], world_points[i, 1]],
            [center[2], world_points[i, 2]],
            color=color,
            linewidth=1,
        )

    # Connect corners to form the image plane rectangle
    # Order: TL -> TR -> BR -> BL -> TL
    plane_indices = [1, 2, 3, 4, 1]
    plane_points = world_points[plane_indices]
    ax.plot(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], color=color, linewidth=1)

    # 5. Draw Axes (RGB = XYZ) for orientation check
    # Extract rotation and translation
    R = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]

    axis_length = scale * 0.8
    # X axis (Red)
    ax.quiver(t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], length=axis_length, color="r")
    # Y axis (Green)
    ax.quiver(t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], length=axis_length, color="g")
    # Z axis (Blue) - Viewing direction
    ax.quiver(t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], length=axis_length, color="b")


# --- Main Execution ---

# Input Data (From the user prompt)
# Shape: (2, 4, 4)
extrinsics = np.array(
    [
        [
            [9.9619e-01, 3.7253e-09, -8.7156e-02, 1.0000e00],
            [0.0000e00, 1.0000e00, 2.9802e-08, 0.0000e00],
            [8.7156e-02, -8.9407e-08, 9.9619e-01, -8.7488e-02],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
        [
            [9.9619e-01, 3.7253e-09, 8.7156e-02, -1.0000e00],
            [5.0084e-10, 1.0000e00, 2.9802e-08, 0.0000e00],
            [-8.7156e-02, -8.9407e-08, 9.9619e-01, 8.7489e-02],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
    ]
)
extrinsics = np.array(
    [
        [
            [1.3510e-01, -6.2827e-01, -7.6618e-01, 6.5861e-01],
            [-9.8481e-01, 7.2682e-08, -1.7365e-01, 2.0000e-01],
            [1.0910e-01, 7.7800e-01, -6.1872e-01, 1.6104e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
        [
            [7.2682e-08, -6.2827e-01, -7.7800e-01, 6.5861e-01],
            [-1.0000e00, 7.2682e-08, -1.5212e-07, 0.0000e00],
            [1.5212e-07, 7.7800e-01, -6.2827e-01, 1.6104e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
        [
            [9.9619e-01, 3.7253e-09, 8.7156e-02, -9.9619e-02],
            [0.0000e00, 1.0000e00, -2.9802e-08, 0.0000e00],
            [-8.7156e-02, 8.9407e-08, 9.9619e-01, -8.7155e-03],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
        [
            [9.9619e-01, 3.7253e-09, -8.7156e-02, 9.9619e-02],
            [5.0084e-10, 1.0000e00, -2.9802e-08, 0.0000e00],
            [8.7156e-02, 8.9407e-08, 9.9619e-01, 8.7156e-03],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ],
    ]
)
# Setup Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Settings
is_w2c = False  # Set to True if the input matrices are World-to-Camera

colors = ["cyan", "magenta", "yellow", "red"]
labels = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

for i in range(len(extrinsics)):
    mat = extrinsics[i]

    # Handle W2C vs C2W
    if is_w2c:
        # If input is World-to-Camera, invert it to get Camera pose in World
        pose = np.linalg.inv(mat)
    else:
        # If input is already Camera-to-World (Pose), use as is
        pose = mat

    plot_camera(ax, pose, color=colors[i], scale=0.5, label=labels[i])

# Set Plot Limits and Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Camera Extrinsics Visualization ({'W2C' if is_w2c else 'C2W'})")

# Adjust aspect ratio to be equal (Crucial for 3D visualization)
# Matplotlib 3D doesn't strictly support "equal" aspect ratio easily,
# so we manually set the limits to a cubic bounding box.
all_t = extrinsics[:, :3, 3]
mid_x, mid_y, mid_z = np.mean(all_t, axis=0)
max_range = 2.0  # Arbitrary range for better view

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend()
plt.show()
