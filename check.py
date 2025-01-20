import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=[0, 1, 2],
    y=[0, 1, 2],
    z=[0, 1, 2],
    mode='lines'
)])

# In thông tin camera
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=2, y=2, z=2),  # Điều chỉnh vị trí camera
            projection=dict(type="perspective")  # Kiểm tra chế độ hiện tại
        )
    )
)
fig.show()
