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
    
    A = np.array(trajectory[0])                   # Start point
    B = np.array(trajectory[len(trajectory) // 2])  # Middle point
    C = np.array(trajectory[-1])                   # End point
    fig.add_trace(create_triangle(A, B, C))
    # Áp dụng cấu hình scene
    fig.update_layout(scene=scene_config, title=note)
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
    camera_rotate_angles = (0, 0, 90)
    note = f'{camera_rotate_angles}'
    camera_config = setup_camera(data_pos[traj_idx], dis_cam=dis_cam, height_cam=height_cam, up_direction=up_direction, angles=camera_rotate_angles) 
    plot_trajectory_3d(data_pos[traj_idx], camera_position=camera_config['cam_position'], target_point=camera_config['cam_target'], up_direction=up_direction, note=note, save_image=True, remove_background=False)

if __name__ == "__main__":
    main()