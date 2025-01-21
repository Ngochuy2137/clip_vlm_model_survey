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

def plot_trajectory_2d(trajectory, object_name, data_type, traj_idx, projection_plane='xy', hor_range=None, ver_range=None, hor_tick_spacing=1, ver_tick_spacing=1,
                       save_image=False, debug=False):
    """
    Vẽ quỹ đạo 2D bằng cách chiếu lên mặt phẳng được chọn với khả năng kiểm soát tọa độ, ẩn trục, và lưu ảnh.

    Args:
        trajectory (np.ndarray): Dữ liệu quỹ đạo (Nx3).
        projection_plane (str): Mặt phẳng chiếu ('xy', 'xz', 'yz', 'zy').
        hor_range (tuple): Phạm vi trục hoành (min, max).
        ver_range (tuple): Phạm vi trục tung (min, max).
        save_image (bool): Lưu ảnh nếu True.
        note (str): Ghi chú thêm trên đồ thị.
        debug (bool): Hiển thị hoặc ẩn trục tọa độ.
    """
    image_name = f"{object_name}-{traj_idx}-{projection_plane.upper()}"
    if debug:
        plot_title = image_name
        image_name = f"{image_name}-debug"
    else:
        plot_title = ''
    trajectory = np.array(trajectory)
    fig = go.Figure()
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a numpy array")
    if trajectory.shape[1] != 3:
        raise ValueError("trajectory must have a shape of (N, 3)")

    # Xử lý chiếu lên các mặt phẳng được chọn và đặt tên trục phù hợp
    plane = projection_plane.lower()
    if len(plane) != 2 or plane[0] not in ['x', 'y', 'z'] or plane[1] not in ['x', 'y', 'z']:
        raise ValueError("Invalid projection_plane. Choose from ['xy', 'xz', 'yz', 'zy']")
    if plane[0] == plane[1]:
        raise ValueError("Projection plane must contain two different axes (e.g., 'xy', 'xz')")
    
    # Chọn trục hoành dựa trên ký tự đầu tiên
    if plane[0] == 'x':
        hor = trajectory[:, 0]
        hor_label = 'X-axis'
    elif plane[0] == 'y':
        hor = trajectory[:, 1]
        hor_label = 'Y-axis'
    else:
        hor = trajectory[:, 2]
        hor_label = 'Z-axis'

    # Chọn trục tung dựa trên ký tự thứ hai
    if plane[1] == 'x':
        ver = trajectory[:, 0]
        ver_label = 'X-axis'
    elif plane[1] == 'y':
        ver = trajectory[:, 1]
        ver_label = 'Y-axis'
    else:
        ver = trajectory[:, 2]
        ver_label = 'Z-axis'

    fig.add_trace(go.Scatter(
        x=hor, 
        y=ver, 
        mode='markers', 
        marker=dict(size=5, color='red'),
        name=f"Projection on {projection_plane.upper()} plane"
    ))

    # Cấu hình trục tọa độ chính xác với mặt phẳng chiếu
    fig.update_layout(
        title=dict(
            text=plot_title,  
            font=dict(size=20, family="Arial"),
            x=0.5,  
            xanchor="center"
        ),
        xaxis=dict(
            title=hor_label, 
            range=hor_range, 
            visible=debug, 
            scaleanchor="y",
            dtick=hor_tick_spacing  # Khoảng cách giữa các tick trục hoành
        ),
        yaxis=dict(
            title=ver_label, 
            range=ver_range, 
            visible=debug,
            dtick=ver_tick_spacing  # Khoảng cách giữa các tick trục tung
        ),
        plot_bgcolor='white'  if not debug else None,  # Đặt nền đồ thị thành màu trắng
        paper_bgcolor='white' if not debug else None,
        width=1303,  # Đặt kích thước hiển thị
        height=1303, # Hình vuông
    )

    # Ẩn trục nếu không ở chế độ debug
    if not debug:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    # Lưu ảnh nếu cần
    if save_image:
        # mkdir folder image
        cur_path = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(cur_path, 'trajectory_images', object_name)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        image_filename = os.path.join(images_dir, f"{image_name}.png")
        # fig.write_image(image_filename, width=2544, height=1303)
        fig.write_image(image_filename, width=1303, height=1303, scale=1)
        print(f"Image was saved to {image_filename}")

    # Hiển thị đồ thị
    if debug:
        fig.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/ring_frisbee/3-data-preprocessed/ring_frisbee-BW-cof-15'
    thrown_object = 'ring_frisbee'
    SAVE_IMAGE = True
    DEBUG = False
    projection_planes = ['xy', 'zy', 'xz']  # Chọn mặt phẳng cần chiếu
    # x_range = (-1.5, 4.5)
    # y_range = (-1.5, 4.5)
    # z_range = (-1.5, 4.5)

    global_printer.print_green(f'Loading data_dir: {data_dir}')
    nae_data_loader = NAEDataLoader()
    # data = nae_data_loader.read_n_merge_data(data_dir)
    data_train, data_val, data_test = nae_data_loader.load_train_val_test_dataset(data_dir)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test
    }
    for key, value in data.items():
        data_pos = [d['preprocess']['model_data'][:, :3] for d in value]
        data_type = key
        # Cấu hình vẽ
        for idx, traj in enumerate(data_pos):
            traj_len = traj.shape[0]
            for plane in projection_planes:
                if plane == 'xy':
                    hor_range = (-1.5, 4.5)
                    ver_range = (-1.5, 4.5)
                elif plane == 'zy':
                    hor_range = (-1.5, 4.5)
                    ver_range = (-1.5, 4.5)
                elif plane == 'xz':
                    hor_range = (-1.5, 4.5)
                    ver_range = (-1.5, 4.5)

                plot_trajectory_2d(
                    data_pos[idx], 
                    object_name=thrown_object,
                    data_type=data_type,
                    traj_idx=idx,
                    projection_plane=plane, 
                    hor_range=hor_range, 
                    ver_range=ver_range, 
                    hor_tick_spacing=0.5,
                    ver_tick_spacing=0.5,
                    save_image=SAVE_IMAGE, 
                    debug=DEBUG  # Bật trục tọa độ
                )
                # input()
