import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# ==========================================
# 0. NANO CONFIGURATION (NANO专属配置)
# ==========================================
# 在Nano上强烈建议关闭Matplotlib 3D实时渲染，否则会非常卡甚至死机
ENABLE_3D_PLOT = False  

if ENABLE_3D_PLOT:
    import matplotlib
    matplotlib.use('TkAgg') # 在Ubuntu上通常用TkAgg而不是macosx
    import matplotlib.pyplot as plt

# ==========================================
# 1. KALMAN FILTER FOR TRAJECTORY TRACKING
# ==========================================
class KalmanFilter3D:
    # ... (保持你原有的 KalmanFilter3D 代码完全不变) ...
    def __init__(self):
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 500.0
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05
        self.is_initialized = False

    def predict(self, dt):
        if not self.is_initialized: return None
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        B = np.zeros((6, 1))
        B[2, 0] = 0.5 * (dt ** 2); B[5, 0] = dt
        u = np.array([[-9.81]])
        self.x = np.dot(F, self.x) + np.dot(B, u)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        return self.x[:3].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape((3, 1))
        if not self.is_initialized:
            self.x[:3] = z
            self.is_initialized = True
            return self.x[:3].flatten()
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x[:3].flatten()

    def get_future_trajectory(self, future_time=1.5, step=0.1):
        if not self.is_initialized: return [], [], []
        curr_x, curr_y, curr_z = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]
        pred_x, pred_y, pred_z = [], [], []
        for t in np.arange(0, future_time, step):
            next_x = curr_x + (vx * t)
            next_y = curr_y + (vy * t)
            next_z = curr_z + (vz * t) + (0.5 * -9.81 * (t ** 2))
            if next_z < 0: break
            pred_x.append(next_x); pred_y.append(next_y); pred_z.append(next_z)
        return pred_x, pred_y, pred_z


# ==========================================
# 2. COORDINATE TRANSFORMATIONS & PERCEPTION
# ==========================================
def camera_to_world(X_cam, Y_cam, Z_cam, drone_pos, drone_yaw):
    # ... (保持原有代码不变) ...
    x_body = Z_cam; y_body = -X_cam; z_body = -Y_cam
    body_point = np.array([[x_body], [y_body], [z_body]])
    yaw_rad = np.radians(drone_yaw)
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    drone_translation = np.array([[drone_pos[0]], [drone_pos[1]], [drone_pos[2]]])
    world_point = np.dot(R_z, body_point) + drone_translation
    return world_point.flatten()

def estimate_3d_position(bbox, camera_matrix, real_diameter=0.067):
    # ... (保持原有代码不变) ...
    x1, y1, x2, y2 = bbox
    pixel_diameter = max(x2 - x1, y2 - y1)
    if pixel_diameter <= 0: return None, None, None
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    Z = (real_diameter * fx) / pixel_diameter
    X = ((x1 + (x2 - x1) / 2.0) - cx) * Z / fx
    Y = ((y1 + (y2 - y1) / 2.0) - cy) * Z / fy
    return X, Y, Z


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Initialize YOLO (替换为 .engine 路径，并指定 task)
    ENGINE_PATH = "tennisball_fp16.engine"  # 请确保这个 engine 文件和代码在同级目录，或者写绝对路径
    print(f"Loading TensorRT engine from: {ENGINE_PATH}")
    model = YOLO(ENGINE_PATH, task="detect")

    import jetson.utils

    print("Initializing CSI Camera via jetson.utils...")
    # 初始化 CSI 摄像头：0号摄像头，分辨率1280x720，帧率30
    camera = jetson.utils.videoSource("csi://0", argv=['--input-width=1280', '--input-height=720', '--framerate=30'])
    
    # 抓取第一帧，用于测试和获取分辨率
    img = camera.Capture()
    if img is None: 
        print("Failed to open camera via jetson.utils!")
        exit()
        
    # jetson.utils 返回的是 GPU 内存中的 RGB 图像格式
    # 我们需要将其转换为 NumPy 数组，并将颜色通道转为 OpenCV 和 YOLO 默认的 BGR 格式
    frame = jetson.utils.cudaToNumpy(img).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

    frame_height, frame_width = frame.shape[:2]
    camera_matrix = np.array([
        [616.10721977, 0.0, 393.06343266],
        [0.0, 791.73808426, 168.77310998],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    kf = KalmanFilter3D()

    drone_world_x, drone_world_y, drone_world_z = 0.0, 0.0, 2.0
    drone_yaw_deg = 90.0

    # 2. Setup Matplotlib 3D Environment (条件启动)
    if ENABLE_3D_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    history_raw = deque(maxlen=30)
    history_flt = deque(maxlen=30)
    PLOT_EVERY_N_FRAMES = 5  # 在Nano上进一步降低画图频率
    frame_count = 0
    last_time = time.time()

    print("Starting... Press 'q' in CV window to quit.")

    while True:
        img = camera.Capture()
        if img is None: 
            continue
            
        # 同样转换为 NumPy 和 BGR 格式
        frame = jetson.utils.cudaToNumpy(img).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        predicted_world_pos = kf.predict(dt)
        
        # 使用 TensorRT 进行推理
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        # 3. Perception and Tracking
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                X_cam, Y_cam, Z_cam = estimate_3d_position([x1, y1, x2, y2], camera_matrix)

                if Z_cam is not None:
                    drone_pos = [drone_world_x, drone_world_y, drone_world_z]
                    raw_world_x, raw_world_y, raw_world_z = camera_to_world(
                        X_cam, Y_cam, Z_cam, drone_pos, drone_yaw_deg)
                    raw_world_pos = (raw_world_x, raw_world_y, raw_world_z)
                    history_raw.append(raw_world_pos)

                    filtered_world_pos = kf.update([raw_world_x, raw_world_y, raw_world_z])
                    history_flt.append(tuple(filtered_world_pos))

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 把预测坐标打印到终端（代替卡顿的3D画图）
                    if not ENABLE_3D_PLOT:
                        print(f"FPS: {1/dt:.1f} | KF Pos X:{filtered_world_pos[0]:.2f} Y:{filtered_world_pos[1]:.2f} Z:{filtered_world_pos[2]:.2f}")

        # 计算并在画面上显示 FPS
        cv2.putText(frame, f"FPS: {1/dt:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("UAV Dynamic Capture - TRT", frame)

        # 4. 3D Simulation Plotting (仅在启用时执行)
        if ENABLE_3D_PLOT and frame_count % PLOT_EVERY_N_FRAMES == 0:
            ax.cla()
            ax.scatter(drone_world_x, drone_world_y, drone_world_z, c='black', marker='^', s=100)
            if len(history_raw) > 0:
                raw_xs, raw_ys, raw_zs = zip(*history_raw)
                ax.scatter(raw_xs, raw_ys, raw_zs, c='r', marker='x', alpha=0.5)
            if len(history_flt) > 0:
                flt_xs, flt_ys, flt_zs = zip(*history_flt)
                ax.plot(flt_xs, flt_ys, flt_zs, c='g', linewidth=2)
                ax.scatter(flt_xs[-1], flt_ys[-1], flt_zs[-1], c='g', marker='o', s=50)
            if kf.is_initialized:
                pred_x, pred_y, pred_z = kf.get_future_trajectory(future_time=1.5, step=0.05)
                if len(pred_x) > 0:
                    ax.plot(pred_x, pred_y, pred_z, c='b', linestyle='--', linewidth=2)

            ax.set_xlim([drone_world_x - 5, drone_world_x + 5])
            ax.set_ylim([drone_world_y, drone_world_y + 10])
            ax.set_zlim([0, 5])
            plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()
    if ENABLE_3D_PLOT:
        plt.ioff()
        plt.show()
