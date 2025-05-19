import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
from pprint import pprint
import cv2
import os


repo_id = "aubo_delta"
# repo_id = "so100_strawberry_grape"

episodes=[0, 1, 10, 11, 23]
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
dataset = lerobot_dataset.LeRobotDataset(repo_id, episodes=episodes)
# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

print(f'dataset.meta {dataset.meta}')

episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# Then we grab all the image frames from the first camera:
camera_key = dataset.meta.camera_keys[0]
print(f'camera keys {dataset.meta.camera_keys}')
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# The objects returned by the dataset are all torch.Tensors
print(f'frame type {type(frames[0])}')
print(f'frame shape {frames[0].shape}')

# Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
# We can compare this shape with the information available for that feature
pprint(dataset.features[camera_key])
# In particular:
print(dataset.features[camera_key]["shape"])

cur_dir = os.path.dirname(os.path.abspath(__file__))
# save frames to video
# 设置视频参数
video_filename = f'{cur_dir}/output_video.mp4'  # 输出视频文件名
fps = 30  # 帧率
frame_size = (frames[0].shape[2], frames[0].shape[1])  # (宽, 高)

# 创建视频写入对象，使用 mp4v 编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# 将每一帧写入视频
for frame in frames:
    # 将torch.Tensor转换为numpy数组并调整通道顺序
    # Convert tensor to numpy and scale to 0-255 uint8
    frame_np = frame.permute(1, 2, 0).cpu().numpy()
    # print(frame_np)
    if frame_np.dtype == np.float32:
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    print(frame_np.shape)
    video_writer.write(frame_np)

# 释放视频写入对象
video_writer.release()
print(f'视频已保存为 {video_filename}')
