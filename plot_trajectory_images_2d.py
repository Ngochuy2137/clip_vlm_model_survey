import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import random
import plotly.graph_objects as go

# Thêm đường dẫn tới thư mục 'nae_core'
# sys.path.append(os.path.abspath("/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae"))
# from nae_static.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader
from nae_static.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from python_utils import filer

global_printer = Printer()

def plot_trajectory_2d(trajectory, projection_plane='xy', x_range=None, y_range=None, z_range=None, 
                       save_image=False, note="", debug=False):
    """
    Vẽ quỹ đạo 2D bằng cách chiếu lên mặt phẳng được chọn với khả năng kiểm soát tọa độ, ẩn trục, và lưu ảnh.

    Args:
        trajectory (np.ndarray): Dữ liệu quỹ đạo (Nx3).
        projection_planes (list): Danh sách các mặt phẳng chiếu ('xy', 'xz', 'yz').
        x_range (tuple): Phạm vi trục X (xmin, xmax).
        y_range (tuple): Phạm vi trục Y (ymin, ymax).
        z_range (tuple): Phạm vi trục Z (zmin, zmax).
        save_image (bool): Lưu ảnh nếu True.
        note (str): Ghi chú thêm trên đồ thị.
        debug (bool): Hiển thị hoặc ẩn trục tọa độ.
    """
    trajectory = np.array(trajectory)
    fig = go.Figure()

    # Xử lý chiếu lên các mặt phẳng được chọn
    if projection_plane.lower() == 'xy':
        x, y = trajectory[:, 0], trajectory[:, 1]
        xlabel, ylabel = 'X', 'Y'
    elif projection_plane.lower() == 'xz':
        x, y = trajectory[:, 0], trajectory[:, 2]
        xlabel, ylabel = 'X', 'Z'
    elif projection_plane.lower() == 'yz':
        x, y = trajectory[:, 1], trajectory[:, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("Invalid projection projection_plane. Choose from ['xy', 'xz', 'yz']")

    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='markers', 
        marker=dict(size=5, color='red'),
        name=f"Projection on {projection_plane.upper()} projection_plane"
    ))

    # Cấu hình trục tọa độ
    axis_config = {
        'xaxis': dict(title="X-axis" if 'xy' in projection_planes or 'xz' in projection_planes else "Y-axis", 
                      range=x_range, visible=debug),
        'yaxis': dict(title="Y-axis" if 'xy' in projection_planes else ("Z-axis" if 'xz' in projection_planes else "Z-axis"), 
                      range=y_range if 'xy' in projection_planes else z_range, visible=debug)
    }

    fig.update_layout(
        title=dict(
            text=note,  
            font=dict(size=20, family="Arial"),
            x=0.5,  
            xanchor="center"
        ),
        **axis_config
    )

    # Ẩn trục nếu không ở chế độ debug
    if not debug:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    # Lưu ảnh nếu cần
    if save_image:
        image_filename = f"trajectory_{note.replace(' ', '_')}.png"
        fig.write_image(image_filename, width=2544, height=1303)
        print(f"Image was saved to {image_filename}")

    # Hiển thị đồ thị
    fig.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/ring_frisbee/3-data-preprocessed/ring_frisbee-BW-cof-15'
    thrown_object = 'ring_frisbee'
    SAVE_IMAGE = False
    LOCAL_FRAME_SET = False
    DIS_CAM = 1
    global_printer.print_green(f'Loading data_dir: {data_dir}')
    nae_data_loader = NAEDataLoader()
    data = nae_data_loader.read_n_merge_data(data_dir)
    data_pos = [d['preprocess']['model_data'][:, :3] for d in data]
    traj_idx = [5, 8, 19, 30, 100]

    # Cấu hình vẽ
    projection_planes = ['xy', 'xz', 'yz']  # Chọn mặt phẳng cần chiếu
    x_range = (-1.5, 4.5)
    y_range = (-1.5, 4.5)
    z_range = (-1.5, 4.5)

    for plane in projection_planes:
        plot_trajectory_2d(
            data_pos[0], 
            projection_plane=plane, 
            x_range=x_range, 
            y_range=y_range, 
            z_range=z_range, 
            save_image=SAVE_IMAGE, 
            note="Trajectory Projection", 
            debug=True  # Bật trục tọa độ
        )
        input()
