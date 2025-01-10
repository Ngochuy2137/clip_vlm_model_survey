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

class TrajectoryPlotter:
    def __init__(self, ):
        # self.x = trajectory[0]
        # self.y = trajectory[1]
        # self.z = trajectory[2]
        # self.color = color
        # self.figure = plt.figure(figsize=(10, 8))
        # self.ax = self.figure.add_subplot(111, projection='3d')

        self.util_plotter = Plotter()
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'gray']

    def plot_trajectory(self, trajectory, thown_object, rotate_data_whose_y_up=False):
        """
        Vẽ quỹ đạo 3D.
        """
        ax = self.util_plotter.plot_samples(trajectory, title=thown_object, rotate_data_whose_y_up=rotate_data_whose_y_up)
        plt.show()
        return ax

    def save_specific_views(self, views, ax, prefix="trajectory_view"):
        """
        Lưu các góc nhìn cụ thể vào file.

        Parameters:
            views: Danh sách các góc nhìn dưới dạng [(elev, azim, name), ...].
            prefix: Tiền tố cho tên file (mặc định là 'trajectory_view').
        """
        # get current path of this script
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get the path to the output directory
        output_dir = os.path.join(current_path, "images_trajectory")
        # create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for elev, azim, name in views:
            ax.view_init(elev=elev, azim=azim)  # Thiết lập góc nhìn
            file_name = f"{output_dir}/{prefix}_{name}.png"
            plt.savefig(file_name)  # Lưu ảnh ra file
            print(f"Đã lưu ảnh: {file_name}")


    # def plot_samples(self, samples, title='', rotate_data_whose_y_up=False, plot_all=False, shuffle=False, save_views=None):
    #     """
    #     Plot 3D samples with optional rotation, shuffling, and saving views.

    #     Parameters:
    #         samples: List of 3D sample data (numpy arrays).
    #         title: Title for the plot.
    #         rotate_data_whose_y_up: Rotate data so Y becomes Z (default False).
    #         plot_all: Plot all samples if True, otherwise limit by color list (default False).
    #         shuffle: Shuffle samples before plotting (default False).
    #         save_views: List of views to save in the form [(elev, azim, name), ...].
    #                     If None, no images are saved (default None).
    #     """
    #     print('Plotting samples...')
    #     figure = plt.figure(num=1, figsize=(12, 12))
    #     ax = figure.add_subplot(111, projection='3d')

    #     if shuffle:
    #         random.shuffle(samples)

    #     x_min, x_max = float('inf'), float('-inf')
    #     y_min, y_max = float('inf'), float('-inf')
    #     z_min, z_max = float('inf'), float('-inf')

    #     for i in range(len(samples)):
    #         if i >= len(self.colors) - 1:
    #             if not plot_all:
    #                 break
    #             color_current = self.colors[-1]
    #         else:
    #             color_current = self.colors[i]

    #         sample = np.array(samples[i])

    #         if rotate_data_whose_y_up:
    #             # Change the order of the axis so that z is up and follow the right-hand rule
    #             x_data = sample[:, 0]
    #             y_data = -sample[:, 2]
    #             z_data = sample[:, 1]
    #         else:
    #             x_data = sample[:, 0]
    #             y_data = sample[:, 1]
    #             z_data = sample[:, 2]

    #         ax.plot(x_data, y_data, z_data, 
    #                 'o', color=color_current, alpha=0.5, label='Test ' + str(i + 1) + ' Sample Trajectory')

    #         # Add 'end' text at the last point
    #         ax.text(x_data[-1], y_data[-1], z_data[-1], 'end', color=color_current, fontsize=10)

    #         x_min = min(x_min, x_data.min())
    #         x_max = max(x_max, x_data.max())
    #         y_min = min(y_min, y_data.min())
    #         y_max = max(y_max, y_data.max())
    #         z_min = min(z_min, z_data.min())
    #         z_max = max(z_max, z_data.max())
        
    #     # Calculate ranges
    #     x_range = x_max - x_min
    #     y_range = y_max - y_min
    #     z_range = z_max - z_min
    #     max_range = max(x_range, y_range, z_range)

    #     # Calculate midpoints
    #     x_mid = (x_max + x_min) / 2
    #     y_mid = (y_max + y_min) / 2
    #     z_mid = (z_max + z_min) / 2

    #     # Set equal aspect ratio
    #     ax.set_xlim([x_mid - max_range / 2, x_mid + max_range / 2])
    #     ax.set_ylim([y_mid - max_range / 2, y_mid + max_range / 2])
    #     ax.set_zlim([z_mid - max_range / 2, z_mid + max_range / 2])

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     plt.legend()
    #     plt.title('3D samples ' + title, fontsize=25)


    #     # Save views if requested
    #     if save_views:
    #         for elev, azim, name in save_views:
    #             ax.view_init(elev=elev, azim=azim)  # Set view
    #             file_name = f"{name}.png"
    #             plt.savefig(file_name, bbox_inches='tight')  # Save image
    #             print(f"Saved view: {file_name}")

    #     plt.show()  # Display the plot before saving views

    

def setup_camera(trajectory, dis_cam, height_cam=None, up_direction={"x": 0, "y": 1, "z": 0}):
    """
    Tính toán vị trí camera và hướng camera.
    Returns:
        dict: Cấu hình camera gồm vị trí camera (eye), điểm nhìn (center), và trục nhìn lên (up).
    """
    # Chọn các điểm đầu, giữa, cuối
    A = trajectory[0]                             # start point
    B = trajectory[len(trajectory) // 2]          # middle point
    C = trajectory[-1]                              # end point
    
    # Tính vector pháp tuyến
    AB = B - A
    AC = C - A
    normal_vector = np.cross(AB, AC)  # Tích chéo
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Chuẩn hóa vector pháp tuyến

    # Trung điểm I của điểm đầu và cuối
    I = (A + C) / 2

    # Tính vị trí camera (C) trên đường thẳng song song với vector pháp tuyến
    camera_position = I + dis_cam * normal_vector
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

def calculate_aspect_ratio(x, y, z):
    """
    Tính toán tỉ lệ khung hình của dữ liệu 3D.

    Args:
        x, y, z (list or array): Tọa độ các điểm trên quỹ đạo.

    Returns:
        tuple: Tỉ lệ khung hình (aspect_ratio_x, aspect_ratio_y, aspect_ratio_z).
    """
    delta_x = max(x) - min(x)
    delta_y = max(y) - min(y)
    delta_z = max(z) - min(z)

    # Tính tỉ lệ và chuẩn hóa để chiều cao (y) là chuẩn
    max_delta = max(delta_x, delta_y, delta_z)
    aspect_ratio_x = delta_x / max_delta
    aspect_ratio_y = delta_y / max_delta
    aspect_ratio_z = delta_z / max_delta

    return aspect_ratio_x, aspect_ratio_y, aspect_ratio_z


def save_plot_with_aspect_ratio(fig, x, y, z, note, base_width=1000, scale=2):
    """
    Lưu ảnh Plotly với tỉ lệ khung hình cố định.

    Args:
        fig (go.Figure): Đối tượng Plotly Figure.
        x, y, z (list or array): Tọa độ các điểm trên quỹ đạo.
        save_path (str): Đường dẫn để lưu ảnh.
        base_width (int): Chiều rộng cơ bản của ảnh (mặc định là 1000 pixel).
        scale (float): Hệ số nhân để tăng độ phân giải (mặc định là 2).
    """
    # get current path of this script
    current_path = os.path.dirname(os.path.realpath(__file__))
    # cd ..
    current_path = os.path.dirname(current_path)
    # mkdir images_trajectory
    output_dir = os.path.join(current_path, "images_trajectory")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save_path
    save_path = os.path.join(output_dir, f"{note}.png")


    # Tính tỉ lệ khung hình
    aspect_ratio_x, aspect_ratio_y, aspect_ratio_z = calculate_aspect_ratio(x, y, z)

    # Tính chiều rộng và chiều cao ảnh
    width = int(base_width)
    height = int(base_width * aspect_ratio_y / aspect_ratio_x)  # Giữ đúng tỉ lệ


    # Lưu ảnh
    fig.write_image(save_path, width=width, height=height, scale=scale)
    print(f"Ảnh đã được lưu tại: {save_path} với kích thước {width}x{height} và scale={scale}")

def plot_trajectory_3d(trajectory, camera_position=None, target_point=None, up_direction=None, save_image=False, note="", remove_background=False):
    fig = go.Figure()

    # Thêm đường quỹ đạo
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
        mode='markers',
        line=dict(color='blue', width=2),
        marker=dict(size=5, color='red')
    ))

    # Thiết lập tỷ lệ trục giữ nguyên
    scene_config = dict(
        aspectmode='data',  # Giữ nguyên tỉ lệ
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis')
    )

    # Thêm thông tin camera nếu có
    if camera_position or target_point or up_direction:
        scene_config["camera"] = {
            "eye": camera_position if camera_position else {"x": 2, "y": 2, "z": 2},
            "center": {"x": target_point.get("x", 0), "y": target_point.get("y", 0), "z": target_point.get("z", 0)} if target_point else {"x": 0, "y": 0, "z": 0},
            "up": up_direction if up_direction else {"x": 0, "y": 0, "z": 1},  # Mặc định trục Z hướng lên
        }
    
    # Áp dụng cấu hình scene
    fig.update_layout(scene=scene_config, title='')
    if remove_background:
        fig.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False
        )
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
    camera_config = setup_camera(data_pos[traj_idx], dis_cam=dis_cam, height_cam=height_cam, up_direction=up_direction) 

    plot_trajectory_3d(data_pos[traj_idx], camera_position=camera_config['cam_position'], target_point=camera_config['cam_target'], up_direction=up_direction, note=thrown_object, save_image=True, remove_background=True)

if __name__ == "__main__":
    main()