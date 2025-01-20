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

import numpy as np

def project_to_plane(points, plane_origin, plane_normal):
    """
    Chiếu các điểm lên một mặt phẳng.

    Args:
        points (np.ndarray): Tọa độ các điểm (N x 3).
        plane_origin (np.ndarray): Gốc của mặt phẳng (1 x 3).
        plane_normal (np.ndarray): Pháp tuyến của mặt phẳng (1 x 3).

    Returns:
        np.ndarray: Tọa độ các điểm chiếu trên mặt phẳng (N x 3).
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Chuẩn hóa pháp tuyến
    projected_points = []

    for point in points:
        # Vector từ điểm tới gốc mặt phẳng
        vector_to_origin = point - plane_origin

        # Phép chiếu vuông góc
        projection = vector_to_origin - np.dot(vector_to_origin, plane_normal) * plane_normal

        # Điểm chiếu
        projected_point = plane_origin + projection
        projected_points.append(projected_point)

    return np.array(projected_points)

# Ví dụ sử dụng
points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
plane_origin = np.array([0, 0, 0])  # Gốc mặt phẳng
plane_normal = np.array([0, 0, 1])  # Pháp tuyến mặt phẳng (vuông góc với trục Z)

projected_points = project_to_plane(points, plane_origin, plane_normal)
print("Projected Points:", projected_points)



# def main():
#     parent_data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_static/data/nae_paper_dataset/origin'
#     thrown_object = 'Bottle'
#     data_dir = filer.find_directories_with_keyword(parent_data_dir, thrown_object)[0]
#     global_printer.print_green(f'Loading data_dir: {data_dir}')
#     data_reader = RoCatDataRawReader(data_dir)
#     data_pos = data_reader.read_position()

#     # camera_position = {"x": 5, "y": 0, "z": 0}
#     # target_point = {"x": 0, "y": 0, "z": 0}

#     traj_idx = 0
#     up_direction = {"x": 0, "y": 1, "z": 0}  # Định nghĩa trục Y là hướng lên
#     dis_cam = 4
#     height_cam = None
#     camera_rotate_angles = (0, 0, 90)
#     note = f'{camera_rotate_angles}'
#     camera_config = setup_camera(data_pos[traj_idx], dis_cam=dis_cam, height_cam=height_cam, up_direction=up_direction, angles=camera_rotate_angles) 
#     plot_trajectory_3d(data_pos[traj_idx], camera_position=camera_config['cam_position'], target_point=camera_config['cam_target'], up_direction=up_direction, note=note, save_image=False, remove_background=False)

# if __name__ == "__main__":
#     main()