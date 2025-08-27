import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.models.optical_flow import raft_large
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame, target_size=(512, 512)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, target_size)
    tensor = torch.tensor(frame_resized).permute(2, 0, 1).float().unsqueeze(0)  # [1, 3, H, W]
    tensor = (tensor / 255.0) * 2 - 1  # 归一化到 [-1, 1]
    return tensor.to(device)

def compute_bidirectional_flow(frame1, frame2):
    with torch.no_grad():
        # 前向光流：frame1 → frame2
        flow_forward = model(frame1, frame2)[-1]
        # 后向光流：frame2 → frame1
        flow_backward = model(frame2, frame1)[-1]
        return flow_forward.squeeze().cpu().numpy(), flow_backward.squeeze().cpu().numpy()

def warp_image(image, flow):
    """根据光流变形图像"""
    h, w = image.shape[:2]
    x_coords = np.arange(w) + flow[0]
    y_coords = np.arange(h) + flow[1]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # 使用双线性插值
    map_x = x_grid.astype(np.float32)
    map_y = y_grid.astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return warped

backwarp_tenGrid = {}

def warp(tenInput, tenFlow): #B,C,H,W, B,2,H,W
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


def interpolate_flow(flow_before, flow_after, alpha=0.5):
    """插值两帧光流得到中间帧的近似光流"""
    return (1 - alpha) * flow_before + alpha * flow_after

def plot_flow(flow, title="Optical Flow"):
    u, v = flow[0], flow[1]
    plt.figure(figsize=(20, 20))
    plt.quiver(u[::-1, :], v[::-1, :], scale=10, color='r')  # 箭头显示
    plt.title(title)
    plt.axis('off')
    plt.show()


model = raft_large(pretrained=True).eval()

dim = 0
f = np.load('../fy01.npy')[:,dim]
f = np.stack([f,f,f],1)
t = 0.5

f1 = torch.tensor(cv2.resize(f[0].transpose((1,2,0)),[512,512]).transpose(2,0,1)).float().unsqueeze(0)*2-1
f2 = torch.tensor(cv2.resize(f[1].transpose((1,2,0)),[512,512]).transpose(2,0,1)).float().unsqueeze(0)*2-1


with torch.no_grad():
    flow_forward, flow_backward = compute_bidirectional_flow(f1, f2)

flow_t_forward = t * flow_forward
flow_t_backward = (1 - t) * -flow_backward

# warped_frame1 = warp_image(f1[0].cpu().numpy().transpose(1,2,0), flow_t_forward)
# warped_frame2 = warp_image(f2[0].cpu().numpy().transpose(1,2,0), flow_t_backward)

warped_frame1 = warp(f1, torch.tensor(flow_t_forward).unsqueeze(0))
warped_frame2 = warp(f2, torch.tensor(flow_t_backward).unsqueeze(0))
interpolated_frame = (1 - t) * warped_frame1 + t * warped_frame2

plt.figure(figsize=(20, 20))
plt.imshow(f1[0,2], cmap='gray')
plt.show()
plt.figure(figsize=(20, 20))
plt.imshow(f2[0,2], cmap='gray')
plt.show()
plt.figure(figsize=(20, 20))
plt.imshow(interpolated_frame[0,2], cmap='gray')
plt.show()

ff = np.load('../fygt.npy')

plt.figure(figsize=(20, 20))
plt.imshow(ff[dim], cmap='gray')
plt.show()
# 加载预训练模型
# model = RAFT()
# model.load_state_dict(torch.load("raft-things.pth"))
# model = model.cuda().eval()
#
# # 读取两帧图像
# frame0 = cv2.imread("frame0.jpg")  # 初始帧
# frame1 = cv2.imread("frame1.jpg")  # 结束帧
# frame0_tensor = ToTensor()(frame0).unsqueeze(0).cuda()
# frame1_tensor = ToTensor()(frame1).unsqueeze(0).cuda()
#
# # 计算双向光流：前向（0→1）和后向（1→0）
# with torch.no_grad():
#     flow_0_to_1 = model(frame0_tensor, frame1_tensor)[-1]  # 前向光流
#     flow_1_to_0 = model(frame1_tensor, frame0_tensor)[-1]  # 后向光流
#
# # 将后向光流转换到前向坐标系（需反向并取负）
# flow_1_to_0 = -flow_1_to_0
#
# # 线性插值中间时刻 t=0.5 的光流
# t = 0.5
# flow_t = t * flow_0_to_1 + (1 - t) * flow_1_to_0
#
# # 转换为 numpy 并可视化
# flow_t_np = flow_t[0].permute(1, 2, 0).cpu().numpy()
# flow_img = flow_viz.flow_to_image(flow_t_np)
# cv2.imshow(f"RAFT Flow at t={t}", flow_img[..., ::-1])
# cv2.waitKey(0)