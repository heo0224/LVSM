import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_camera_frustum(ax, extrinsic_matrix, color='b', scale=1.0, aspect_ratio=1.33):
    """
    Visualizes a camera frustum in 3D space based on the extrinsic matrix.
    
    Args:
        ax: Matplotlib 3D axis.
        extrinsic_matrix: 4x4 or 3x4 matrix (World -> Camera transformation).
        color: Color of the frustum lines.
        scale: The depth (Z-axis) length of the frustum.
        aspect_ratio: Width/Height ratio of the image plane (e.g., 640/480 = 1.33).
    """
    
    # Ensure matrix is 4x4
    extrinsic = np.eye(4)
    if extrinsic_matrix.shape == (3, 4):
        extrinsic[:3, :] = extrinsic_matrix
    elif extrinsic_matrix.shape == (4, 4):
        extrinsic = extrinsic_matrix
    else:
        raise ValueError("Extrinsic matrix must be 3x4 or 4x4")

    # Calculate Pose (Camera -> World) by inverting Extrinsic (World -> Camera)
    # Pose = [R^T | -R^T * t]
    pose = np.linalg.inv(extrinsic)

    # Define Frustum Geometry in Camera Coordinates (OpenCV Standard)
    # X: Right, Y: Down, Z: Forward
    h = scale / aspect_ratio
    w = scale * aspect_ratio
    z = scale

    # 5 points: Origin, Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # Note: In OpenCV, Y is down. So negative Y is 'up' in 3D space relative to camera.
    local_points = np.array([
        [0, 0, 0, 1],          # 0: Optical Center
        [-w, -h, z, 1],        # 1: Top-Left (in image coords: 0,0)
        [w, -h, z, 1],         # 2: Top-Right
        [w, h, z, 1],          # 3: Bottom-Right
        [-w, h, z, 1],         # 4: Bottom-Left
    ]).T  # Transpose to shape (4, 5)

    # Transform points to World Coordinates
    world_points = pose @ local_points

    # Extract X, Y, Z
    X = world_points[0, :]
    Y = world_points[1, :]
    Z = world_points[2, :]

    # Draw Lines connecting Optical Center to Image Plane Corners
    for i in range(1, 5):
        ax.plot([X[0], X[i]], [Y[0], Y[i]], [Z[0], Z[i]], color=color, linewidth=1)

    # Draw Image Plane Rectangle (1-2-3-4-1)
    ax.plot([X[1], X[2], X[3], X[4], X[1]], 
            [Y[1], Y[2], Y[3], Y[4], Y[1]], 
            [Z[1], Z[2], Z[3], Z[4], Z[1]], color=color, linewidth=1)

    # Draw Local Coordinate Axes at the Camera Origin
    # Red: +X (Right), Green: +Y (Down), Blue: +Z (Forward)
    axis_len = scale * 0.5
    local_axes = np.array([
        [axis_len, 0, 0, 1],  # X-axis tip
        [0, axis_len, 0, 1],  # Y-axis tip
        [0, 0, axis_len, 1]   # Z-axis tip
    ]).T
    
    world_axes = pose @ local_axes
    origin = world_points[:, 0]

    ax.plot([origin[0], world_axes[0, 0]], [origin[1], world_axes[1, 0]], [origin[2], world_axes[2, 0]], 'r-', linewidth=2, label='X (Right)')
    ax.plot([origin[0], world_axes[0, 1]], [origin[1], world_axes[1, 1]], [origin[2], world_axes[2, 1]], 'g-', linewidth=2, label='Y (Down)')
    ax.plot([origin[0], world_axes[0, 2]], [origin[1], world_axes[1, 2]], [origin[2], world_axes[2, 2]], 'b-', linewidth=2, label='Z (Fwd)')

# --- Usage Example ---
if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get Extrinsic from Pose
    cam_1_extrinsics_path = "./raw_data/camera_prev/realsense_extrinsics.npy"
    cam_2_extrinsics_path = "./raw_data/camera_prev/cam_2_extrinsics.npy"
    cam_3_extrinsics_path = "./raw_data/camera_prev/cam_3_extrinsics.npy"

    cam_1_extrinsics = np.load(cam_1_extrinsics_path)
    cam_2_extrinsics = np.load(cam_2_extrinsics_path)
    cam_3_extrinsics = np.load(cam_3_extrinsics_path)

    cam_1_extrinsics = np.linalg.inv(cam_1_extrinsics)
    cam_2_extrinsics = np.linalg.inv(cam_2_extrinsics)
    cam_3_extrinsics = np.linalg.inv(cam_3_extrinsics)

    # opengl to opencv
    opengl_to_opencv = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
    cam_1_extrinsics = opengl_to_opencv @ cam_1_extrinsics
    cam_2_extrinsics = opengl_to_opencv @ cam_2_extrinsics
    cam_3_extrinsics = opengl_to_opencv @ cam_3_extrinsics

    # Plotting
    plot_camera_frustum(ax, cam_1_extrinsics, color='m', scale=0.2)
    plot_camera_frustum(ax, cam_2_extrinsics, color='c', scale=0.2)
    plot_camera_frustum(ax, cam_3_extrinsics, color='y', scale=0.2)

    # Setup plot appearance
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')
    ax.set_title('Camera Frustum Visualization (OpenCV Coords)')
    
    # Add a reference point at world origin
    ax.scatter([0], [0], [0], c='k', marker='o', s=50, label='World Origin')
    
    set_axes_equal(ax) # Important for correct visualization
    plt.legend()
    plt.show()