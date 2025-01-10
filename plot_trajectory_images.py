import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import random
import plotly.graph_objects as go

# Thêm đường dẫn tới thư mục 'nae_core'
sys.path.append(os.path.abspath("/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae"))
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from python_utils import filer

global_printer = Printer()

def rotate_vector(vector, angles):
    """
    Quay một vector xung quanh các trục x, y, z một góc nhất định.

    Args:
        vector (np.ndarray): Vector cần quay (dạng numpy array, 1D).
        angles (tuple): Bộ 3 góc quay (theta_x, theta_y, theta_z) tính bằng độ.

    Returns:
        np.ndarray: Vector sau khi quay.
    """
    # Chuyển góc từ độ sang radian
    theta_x, theta_y, theta_z = np.radians(angles)

    # Ma trận quay quanh trục X
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    # Ma trận quay quanh trục Y
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Ma trận quay quanh trục Z
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Tích hợp ma trận quay
    rotation_matrix = R_z @ R_y @ R_x

    # Quay vector
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector

def setup_camera(trajectory, dis_cam, height_cam=None, angles=(0, 0, 0), up_direction={"x": 0, "y": 1, "z": 0}):
    """
    Tính toán vị trí camera và hướng camera, hỗ trợ góc lệch với pháp tuyến theo trục chỉ định.
    
    Args:
        trajectory (list or array): Danh sách các điểm trên quỹ đạo [(x1, y1, z1), (x2, y2, z2), ...].
        dis_cam (float): Khoảng cách từ camera đến trung điểm của quỹ đạo.
        height_cam (float, optional): Chiều cao cố định của camera (nếu có).
        angle (float, optional): Góc lệch với vector pháp tuyến (tính bằng độ).
        axis (str, optional): Trục quay để tạo góc lệch (x, y, z).
        up_direction (dict, optional): Hướng lên của camera (mặc định là {"x": 0, "y": 1, "z": 0}).

    Returns:
        dict: Cấu hình camera gồm vị trí camera (eye), điểm nhìn (center), và trục nhìn lên (up).
    """
    # Chọn các điểm đầu, giữa, cuối
    A = np.array(trajectory[0])                   # start point
    B = np.array(trajectory[len(trajectory) // 2])  # middle point
    C = np.array(trajectory[-1])                   # end point
    
    # Tính vector pháp tuyến
    AB = B - A
    AC = C - A
    normal_vector = np.cross(AB, AC)  # Tích chéo
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Chuẩn hóa vector pháp tuyến

    rotated_vector = rotate_vector(normal_vector, angles)

    # Trung điểm I của điểm đầu và cuối
    I = (A + C) / 2

    # Tính vị trí camera (C) trên vector pháp tuyến đã quay
    camera_position = I + dis_cam * rotated_vector

    # Gán giá trị chiều cao cố định nếu được chỉ định
    if height_cam is not None:
        if up_direction['x'] == 1:
            camera_position[0] = height_cam
        if up_direction['y'] == 1:
            camera_position[1] = height_cam
        if up_direction['z'] == 1:
            camera_position[2] = height_cam

    # Cấu hình camera
    camera_config = {
        "cam_position": {"x": float(camera_position[0]), "y": float(camera_position[1]), "z": float(camera_position[2])},
        "cam_target": {"x": float(I[0]), "y": float(I[1]), "z": float(I[2])},
        "up_direction": up_direction
    }

    return camera_config

def create_triangle(A, B, C, color='rgba(0, 0, 255, 0.5)'):
    """
    Tạo tam giác từ 3 điểm A, B, C.
    """
    return go.Mesh3d(
        x=[A[0], B[0], C[0]],
        y=[A[1], B[1], C[1]],
        z=[A[2], B[2], C[2]],
        i=[0],  # Chỉ số các điểm tạo mặt tam giác
        j=[1],
        k=[2],
        facecolor=[color],  # Sử dụng facecolor thay vì color
        opacity=0.5  # Độ trong suốt
    )

def create_camera_to_I_line(camera_position, target_point):
    """
    Tạo đường thẳng nối từ camera đến trung điểm I.

    Args:
        camera_position (dict): Vị trí camera {"x": float, "y": float, "z": float}.
        target_point (dict): Điểm mà camera nhìn vào {"x": float, "y": float, "z": float}.

    Returns:
        go.Scatter3d: Đối tượng đường thẳng của Plotly.
    """
    return go.Scatter3d(
        x=[camera_position['x'], target_point['x']],
        y=[camera_position['y'], target_point['y']],
        z=[camera_position['z'], target_point['z']],
        mode='lines',
        line=dict(color='orange', width=5),  # Đường màu cam, độ dày 5
        name="Camera to I"
    )

def configure_scene(camera_position=None, target_point=None, up_direction=None):
    """
    Cấu hình Scene của đồ thị Plotly.

    Args:
        camera_position (dict, optional): Vị trí camera.
        target_point (dict, optional): Điểm camera nhìn vào.
        up_direction (dict, optional): Trục nhìn lên của camera.

    Returns:
        dict: Cấu hình Scene.
    """
    scene_config = dict(
        aspectmode='data',  # Giữ nguyên tỉ lệ
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis')
    )

    if camera_position or target_point or up_direction:
        scene_config["camera"] = {
            "eye": camera_position if camera_position else {"x": 2, "y": 2, "z": 2},
            "center": {"x": target_point.get("x", 0), "y": target_point.get("y", 0), "z": target_point.get("z", 0)} if target_point else {"x": 0, "y": 0, "z": 0},
            "up": up_direction if up_direction else {"x": 0, "y": 0, "z": 1}
        }
    return scene_config

def plot_trajectory_3d(trajectory, camera_position=None, target_point=None, up_direction=None, save_image=False, note="", remove_background=False):
    """
    Vẽ quỹ đạo 3D, tam giác và đường thẳng từ camera đến điểm I.

    Args:
        trajectory (array): Dữ liệu quỹ đạo (N x 3).
        camera_position (dict, optional): Vị trí camera.
        target_point (dict, optional): Điểm camera nhìn vào.
        up_direction (dict, optional): Trục nhìn lên của camera.
        save_image (bool, optional): Lưu ảnh nếu True.
        note (str, optional): Ghi chú thêm trên đồ thị.
        remove_background (bool, optional): Loại bỏ phông nền nếu True.
    """
    # Đảm bảo trajectory là numpy array
    trajectory = np.array(trajectory)

    # Tạo đồ thị
    fig = go.Figure()

    # Thêm đường quỹ đạo
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
        mode='markers',
        line=dict(color='blue', width=2),
        marker=dict(size=5, color='red')
    ))

    # Thêm tam giác từ 3 điểm đầu, giữa, cuối
    A = np.array(trajectory[0])                   # Start point
    B = np.array(trajectory[len(trajectory) // 2])  # Middle point
    C = np.array(trajectory[-1])                   # End point

    fig.add_trace(create_triangle(A, B, C))

    # Thêm đường từ camera đến điểm I
    if camera_position and target_point:
        fig.add_trace(create_camera_to_I_line(camera_position, target_point))

    # Cấu hình Scene
    scene_config = configure_scene(camera_position, target_point, up_direction)

    # Áp dụng cấu hình scene
    fig.update_layout(scene=scene_config, title=note)

    # Loại bỏ phông nền nếu cần
    if remove_background:
        fig.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False
        )

    # Lưu ảnh nếu cần
    if save_image:
        fig.write_image(f"trajectory_{note}.png", scale=2)

    # Hiển thị đồ thị
    fig.show()


def main():
    parent_data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_static/data/nae_paper_dataset/origin'
    thrown_object = 'Bottle'
    data_dir = filer.find_directories_with_keyword(parent_data_dir, thrown_object)[0]
    global_printer.print_green(f'Loading data_dir: {data_dir}')
    data_reader = RoCatDataRawReader(data_dir)
    data_pos = data_reader.read_position()

    # camera_position = {"x": 5, "y": 0, "z": 0}
    # target_point = {"x": 0, "y": 0, "z": 0}

    traj_idx = 0
    up_direction = {"x": 0, "y": 1, "z": 0}  # Định nghĩa trục Y là hướng lên
    dis_cam = 4
    height_cam = None
    camera_rotate_angles = (0, 0, 90)
    note = f'{camera_rotate_angles}'
    camera_config = setup_camera(data_pos[traj_idx], dis_cam=dis_cam, height_cam=height_cam, up_direction=up_direction, angles=camera_rotate_angles) 
    plot_trajectory_3d(data_pos[traj_idx], camera_position=camera_config['cam_position'], target_point=camera_config['cam_target'], up_direction=up_direction, note=note, save_image=False, remove_background=False)

if __name__ == "__main__":
    main()