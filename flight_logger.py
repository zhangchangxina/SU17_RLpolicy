#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用飞行数据记录节点 (统一版)

可配合任意避障方法使用（雷达避障 EGO-Planner / YOPO / Isaac Lab RL / ...）
只需订阅 Prometheus 标准话题，无需修改避障节点代码。

如果配合 Isaac Lab RL 使用，还会自动订阅额外话题获取:
  - 策略原始动作 (action_raw)
  - 最近障碍物距离 (min_obstacle_dist)
  - 处理后的雷达扫描 (lidar_scan) → 保存为 NPZ

用法:
    # 配合 EGO-Planner 雷达避障
    python3 flight_logger.py _method:=ego_planner _target_x:=22 _target_y:=-22 _target_z:=1

    # 配合 YOPO 深度避障
    python3 flight_logger.py _method:=yopo _target_x:=22 _target_y:=-22 _target_z:=1

    # 配合 Isaac Lab RL
    python3 flight_logger.py _method:=isaaclab_rl _target_x:=22 _target_y:=-22 _target_z:=1

数据保存:
    flight_logs/<method>_<YYYYMMDD_HHMMSS>/flight_<YYYYMMDD_HHMMSS>.csv
    flight_logs/<method>_<YYYYMMDD_HHMMSS>/flight_<YYYYMMDD_HHMMSS>_lidar.npz  (如有)

订阅话题 (通用):
    /uav1/prometheus/state          (UAVState)           → 位置/速度/姿态
    /uav1/mavros/local_position/odom (Odometry)          → 备用里程计
    /uav1/prometheus/command        (UAVCommand)         → 控制指令
    /move_base_simple/goal          (PoseStamped)        → 目标点

订阅话题 (Isaac Lab RL 自动检测):
    /uav1/isaaclab/action_raw       (Float32MultiArray)  → 策略原始输出
    /uav1/isaaclab/min_obstacle_dist (Float32)           → 最近障碍距离
    /uav1/isaaclab/lidar_scan       (Float32MultiArray)  → 雷达扫描 (280维)
"""

import os
import sys
import csv
import time
import numpy as np
from threading import Lock

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import PointCloud2, LaserScan

try:
    from prometheus_msgs.msg import UAVCommand, UAVState
    PROMETHEUS_MSGS_AVAILABLE = True
except ImportError:
    PROMETHEUS_MSGS_AVAILABLE = False
    UAVState = None
    UAVCommand = None
    print("Warning: prometheus_msgs not found, only mavros/odom available")


class FlightLogger:
    """通用飞行数据记录器"""

    def __init__(self):
        rospy.init_node('flight_logger', anonymous=True)

        self.lock = Lock()
        self.uav_id = rospy.get_param('~uav_id', 1)
        self.method = rospy.get_param('~method', 'unknown')
        self.control_freq = rospy.get_param('~log_freq', 10.0)  # Hz

        # 目标点 (用于计算到目标距离)
        self.target_x = rospy.get_param('~target_x', 0.0)
        self.target_y = rospy.get_param('~target_y', 0.0)
        self.target_z = rospy.get_param('~target_z', 1.0)

        # 状态变量
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1.0])  # qx, qy, qz, qw
        self.angular_vel = np.zeros(3)
        self.cmd_vel = np.zeros(3)       # 指令速度
        self.cmd_pos = np.zeros(3)       # 指令位置
        self.cmd_acc = np.zeros(3)       # 指令加速度
        self.cmd_yaw = 0.0
        self.data_received = False
        self.use_prometheus_state = False

        # RL 特有数据 (可选，有数据时自动使用)
        self.action_raw = np.zeros(3)
        self.min_obstacle_dist = -1.0    # -1 = 未知 (尚未收到传感器数据)
        self.lidar_scan = None           # None = 没收到雷达数据
        self._has_action_raw = False
        self._has_min_obstacle = False
        self._has_lidar = False
        
        # CBF 数据 (可选，有数据时自动记录)
        # 7维: barrier, violation, delta_vx, delta_vy, delta_vz, active, min_barrier_dist
        self.cbf_data = np.zeros(7)
        self._has_cbf_data = False

        # 通用传感器障碍物距离 (从 PointCloud2 / LaserScan 计算)
        self._sensor_min_dist = -1.0     # -1 = 未收到
        self._has_sensor_obstacle = False

        # 雷达快照 (用于 NPZ) — Isaac Lab RL 专用 (280维处理后数据)
        self._lidar_snapshots = []       # [(time, pos_x, pos_y, pos_z, yaw_deg, scan)]
        self._frame_counter = 0
        self._lidar_snapshot_interval = 10  # 每 N 帧保存一次

        # 3D 点云快照 (用于 NPZ) — 所有方法通用 (PointCloud2 原始数据)
        self._pcl_snapshots = []         # [(time, pos_x, pos_y, pos_z, yaw_deg, points_enu_Nx3)]
        self._pcl_frame_counter = 0
        self._pcl_snapshot_interval = 5   # 每 N 帧保存一次 (10Hz下约0.5s/帧)
        self._pcl_max_points_per_frame = 2000  # 每帧最多保存点数
        self._has_pcl = False

        # 记录起始时间 (run() 中初始化)
        self.t0 = None

        # 话题前缀
        self.topic_prefix = f'/uav{self.uav_id}'

        # ---- 订阅者 (通用) ----
        if PROMETHEUS_MSGS_AVAILABLE:
            rospy.Subscriber(f'{self.topic_prefix}/prometheus/state',
                             UAVState, self.uav_state_callback)
            rospy.Subscriber(f'{self.topic_prefix}/prometheus/command',
                             UAVCommand, self.command_callback)
            rospy.loginfo("订阅 Prometheus state + command")

        rospy.Subscriber(f'{self.topic_prefix}/mavros/local_position/odom',
                         Odometry, self.odom_callback)
        rospy.loginfo("订阅 Mavros odom (备用)")

        # 目标点动态更新
        rospy.Subscriber(f'{self.topic_prefix}/target_position',
                         PoseStamped, self.target_callback)
        rospy.Subscriber('/move_base_simple/goal',
                         PoseStamped, self.target_callback)

        # ---- 订阅者 (Isaac Lab RL 额外话题, 可选) ----
        rospy.Subscriber(f'{self.topic_prefix}/isaaclab/action_raw',
                         Float32MultiArray, self.action_raw_callback)
        rospy.Subscriber(f'{self.topic_prefix}/isaaclab/min_obstacle_dist',
                         Float32, self.min_obstacle_callback)
        rospy.Subscriber(f'{self.topic_prefix}/isaaclab/lidar_scan',
                         Float32MultiArray, self.lidar_scan_callback)
        rospy.Subscriber(f'{self.topic_prefix}/isaaclab/cbf_data',
                         Float32MultiArray, self.cbf_data_callback)
        rospy.loginfo("订阅 Isaac Lab RL 额外话题 (可选，无数据时自动忽略)")

        # ---- 订阅者 (通用传感器：用于计算最近障碍物距离) ----
        # PointCloud2 (来自 octomap / AirSim LiDAR / Livox 等)
        pcl_topics = [
            f'{self.topic_prefix}/cloud_mid360_body',           # 实机 Livox Mid-360 点云 (PointCloud2)
            '/cloud_effected',                                   # 实机全局点云 (PointCloud2)
            f'{self.topic_prefix}/octomap_point_cloud_centers',
            f'{self.topic_prefix}/prometheus/sensors/3Dlidar',
            f'/airsim_node/uav{self.uav_id}/lidar/LidarSensor1',
            f'/airsim_node/uav{self.uav_id}/lidar/Lidar',
            f'/airsim_node/uav{self.uav_id}/lidarPointCloud2/LidarSensor1',
            f'{self.topic_prefix}/livox/lidar',
            '/livox/lidar',                                      # 实机 Livox 雷达 (无前缀)
            f'{self.topic_prefix}/velodyne_points',
            f'{self.topic_prefix}/prometheus/scan_point_cloud',
        ]
        for topic in pcl_topics:
            rospy.Subscriber(topic, PointCloud2, self.pointcloud_obstacle_callback)
        # LaserScan (2D 雷达)
        scan_topics = [
            f'/airsim_node/uav{self.uav_id}/lidarLaserScan/LidarSensor1',
            f'{self.topic_prefix}/scan',
            f'{self.topic_prefix}/scan_filtered',
            f'{self.topic_prefix}/prometheus/sensors/2Dlidar_scan',
        ]
        for topic in scan_topics:
            rospy.Subscriber(topic, LaserScan, self.laserscan_obstacle_callback)
        rospy.loginfo("订阅通用传感器话题 (自动计算障碍物距离)")

        # ---- 日志文件 ----
        self._run_timestamp = time.strftime('%Y%m%d_%H%M%S')
        self._dir_name = f"{self.method}_{self._run_timestamp}"
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'flight_logs', self._dir_name)
        os.makedirs(self.log_dir, exist_ok=True)

        log_filename = f'flight_{self._run_timestamp}.csv'
        self.log_path = os.path.join(self.log_dir, log_filename)
        self.log_file = open(self.log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            'time', 'x', 'y', 'z',
            'vx_world', 'vy_world', 'vz_world',
            'yaw_deg',
            'target_x', 'target_y', 'target_z',
            'dist_to_target', 'min_obstacle_dist',
            # 策略动作 (原始 → 修正 → 安全过滤)
            'action_raw_vx', 'action_raw_vy', 'action_raw_vz',
            'action_vx', 'action_vy', 'action_vz',
            'cmd_vx', 'cmd_vy', 'cmd_vz',
            # 机体系速度 / 角速度
            'body_vx', 'body_vy', 'body_vz',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
            # 姿态
            'gravity_x', 'gravity_y', 'gravity_z',
            # pose_command (世界系相对目标位置)
            'pose_cmd_x', 'pose_cmd_y', 'pose_cmd_z',
            # 航向
            'current_yaw_deg', 'target_yaw_deg', 'yaw_error_deg',
            # 高度
            'height',
            # 状态标志
            'is_hovering', 'height_protection',
            # CBF 数据 (可选)
            'cbf_barrier', 'cbf_violation',
            'cbf_delta_vx', 'cbf_delta_vy', 'cbf_delta_vz',
            'cbf_active', 'cbf_min_dist',
        ])
        self.log_start_time = None

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("=" * 60)
        rospy.loginfo(f"飞行数据记录器已启动")
        rospy.loginfo(f"  方法: {self.method}")
        rospy.loginfo(f"  无人机: UAV{self.uav_id}")
        rospy.loginfo(f"  目标: ({self.target_x}, {self.target_y}, {self.target_z})")
        rospy.loginfo(f"  记录频率: {self.control_freq} Hz")
        rospy.loginfo(f"  保存路径: {self.log_path}")
        rospy.loginfo("=" * 60)

    # ================================================================
    # 通用回调
    # ================================================================

    def uav_state_callback(self, msg):
        with self.lock:
            self.position[0] = msg.position[0]
            self.position[1] = msg.position[1]
            self.position[2] = msg.position[2]
            self.velocity[0] = msg.velocity[0]
            self.velocity[1] = msg.velocity[1]
            self.velocity[2] = msg.velocity[2]
            self.orientation[0] = msg.attitude_q.x
            self.orientation[1] = msg.attitude_q.y
            self.orientation[2] = msg.attitude_q.z
            self.orientation[3] = msg.attitude_q.w
            self.angular_vel[0] = msg.attitude_rate[0]
            self.angular_vel[1] = msg.attitude_rate[1]
            self.angular_vel[2] = msg.attitude_rate[2]
            self.data_received = True
            self.use_prometheus_state = True

    def odom_callback(self, msg):
        with self.lock:
            if self.use_prometheus_state:
                return  # 已有更好的数据源
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            q = msg.pose.pose.orientation
            w = msg.twist.twist.angular
            self.position[:] = [p.x, p.y, p.z]
            self.velocity[:] = [v.x, v.y, v.z]
            self.orientation[:] = [q.x, q.y, q.z, q.w]
            self.angular_vel[:] = [w.x, w.y, w.z]
            self.data_received = True

    def command_callback(self, msg):
        """Prometheus UAVCommand 回调"""
        with self.lock:
            self.cmd_vel[:] = [msg.velocity_ref[0], msg.velocity_ref[1], msg.velocity_ref[2]]
            self.cmd_pos[:] = [msg.position_ref[0], msg.position_ref[1], msg.position_ref[2]]
            self.cmd_acc[:] = [msg.acceleration_ref[0], msg.acceleration_ref[1], msg.acceleration_ref[2]]
            self.cmd_yaw = msg.yaw_ref

    def target_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        if msg.pose.position.z > 0.1:
            self.target_z = msg.pose.position.z
        rospy.loginfo(f"目标更新: ({self.target_x:.1f}, {self.target_y:.1f}, {self.target_z:.1f})")

    # ================================================================
    # Isaac Lab RL 额外回调 (可选)
    # ================================================================

    def action_raw_callback(self, msg):
        """策略原始输出 (安全过滤前)"""
        with self.lock:
            if len(msg.data) >= 3:
                self.action_raw[:] = msg.data[:3]
                if not self._has_action_raw:
                    self._has_action_raw = True
                    rospy.loginfo("✓ 检测到 Isaac Lab RL action_raw 话题")

    def min_obstacle_callback(self, msg):
        """最近障碍物距离"""
        with self.lock:
            self.min_obstacle_dist = msg.data
            if not self._has_min_obstacle:
                self._has_min_obstacle = True
                rospy.loginfo("✓ 检测到 Isaac Lab RL min_obstacle_dist 话题")

    def lidar_scan_callback(self, msg):
        """处理后的雷达扫描 (280维)"""
        with self.lock:
            self.lidar_scan = np.array(msg.data)
            if not self._has_lidar:
                self._has_lidar = True
                rospy.loginfo(f"✓ 检测到 Isaac Lab RL lidar_scan 话题 ({len(msg.data)} 维)")

    def cbf_data_callback(self, msg):
        """CBF 数据 (7维: barrier, violation, delta_vx/vy/vz, active, min_barrier_dist)"""
        with self.lock:
            if len(msg.data) >= 7:
                self.cbf_data[:] = msg.data[:7]
                if not self._has_cbf_data:
                    self._has_cbf_data = True
                    rospy.loginfo("检测到 CBF 数据话题")

    # ================================================================
    # 通用传感器回调 (计算最近障碍物距离)
    # ================================================================

    def pointcloud_obstacle_callback(self, msg):
        """从 PointCloud2 计算最近障碍物距离 + 保存3D点云快照"""
        try:
            # 解析 PointCloud2 → XYZ 数组
            points = self._parse_pointcloud2(msg)
            if points is None or len(points) == 0:
                return

            # 计算每个点到无人机的距离 (点云在 VehicleInertialFrame = NED 世界系)
            distances = np.linalg.norm(points[:, :3], axis=1)
            # 过滤太近的点 (噪声) 和太远的点
            valid = (distances > 0.3) & (distances < 50.0)
            if valid.any():
                min_dist = float(np.min(distances[valid]))
                with self.lock:
                    self._sensor_min_dist = min_dist
                    if not self._has_sensor_obstacle:
                        self._has_sensor_obstacle = True
                        rospy.loginfo(f"✓ 从 PointCloud2 计算障碍物距离 (当前最近: {min_dist:.2f}m)")

            # ---- 保存3D点云快照 (用于画图) ----
            self._pcl_frame_counter += 1
            if self._pcl_frame_counter % self._pcl_snapshot_interval == 0:
                # 点云在 VehicleInertialFrame (NED) → 转 ENU
                # ENU: x=NED_y, y=NED_x, z=-NED_z
                # 但注意：这里的 points 是相对于无人机的偏移
                # 需要加上无人机当前世界位置
                with self.lock:
                    pos = self.position.copy()
                    quat = self.orientation.copy()

                # 无人机 yaw
                qx, qy, qz, qw = quat
                yaw_rad = np.arctan2(2.0 * (qw * qz + qx * qy),
                                     1.0 - 2.0 * (qy * qy + qz * qz))

                pts = points[:, :3].copy()
                # 过滤有效点
                pts_valid = pts[np.isfinite(pts).all(axis=1) & (distances > 0.3) & (distances < 50.0)]

                if len(pts_valid) > 0:
                    # ============================================================
                    # 点云 raw → 世界 ENU 坐标转换
                    # ============================================================
                    # 实测验证: 到达 callback 的点云满足
                    #   raw = R(+yaw) * world_enu_offset
                    # 因此转世界ENU: world_enu = R(-yaw) * raw + pos
                    # (与 isaaclab_policy_node.py 中 body=R(-2*yaw)*raw 一致:
                    #   body = R(-yaw)*world_enu = R(-yaw)*R(-yaw)*raw = R(-2*yaw)*raw)
                    # ============================================================
                    cos_neg_y = np.cos(-yaw_rad)
                    sin_neg_y = np.sin(-yaw_rad)
                    world_x = pts_valid[:, 0] * cos_neg_y - pts_valid[:, 1] * sin_neg_y + pos[0]
                    world_y = pts_valid[:, 0] * sin_neg_y + pts_valid[:, 1] * cos_neg_y + pos[1]
                    world_z = pts_valid[:, 2] + pos[2]

                    world_pts = np.column_stack([world_x, world_y, world_z])

                    # 降采样 (限制每帧点数, 减小文件大小)
                    max_pts = self._pcl_max_points_per_frame
                    if len(world_pts) > max_pts:
                        idx = np.random.choice(len(world_pts), max_pts, replace=False)
                        world_pts = world_pts[idx]

                    if self.t0 is None:
                        return  # 记录尚未开始
                    t = rospy.get_time() - self.t0
                    self._pcl_snapshots.append((
                        t, pos[0], pos[1], pos[2],
                        np.degrees(yaw_rad), world_pts
                    ))

                    if not self._has_pcl:
                        self._has_pcl = True
                        rospy.loginfo(f"✓ 开始录制 PointCloud2 点云快照 ({len(world_pts)} 点/帧)")

        except Exception as e:
            rospy.logwarn_throttle(10, f"PointCloud2 解析异常: {e}")

    def laserscan_obstacle_callback(self, msg):
        """从 LaserScan 计算最近障碍物距离"""
        try:
            ranges = np.array(msg.ranges)
            # 过滤无效值
            valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
            if valid.any():
                min_dist = float(np.min(ranges[valid]))
                with self.lock:
                    self._sensor_min_dist = min_dist
                    if not self._has_sensor_obstacle:
                        self._has_sensor_obstacle = True
                        rospy.loginfo(f"✓ 从 LaserScan 计算障碍物距离 (当前最近: {min_dist:.2f}m)")
        except Exception as e:
            rospy.logwarn_throttle(10, f"LaserScan 解析异常: {e}")

    @staticmethod
    def _parse_pointcloud2(msg):
        """轻量级 PointCloud2 → numpy XYZ 解析 (无需 ros_numpy, 纯 numpy 高性能)"""
        n_points = msg.width * msg.height
        if n_points == 0:
            return None

        # 查找 x, y, z 字段的偏移
        field_map = {f.name: f for f in msg.fields}
        if 'x' not in field_map or 'y' not in field_map or 'z' not in field_map:
            return None

        ox = field_map['x'].offset
        oy = field_map['y'].offset
        oz = field_map['z'].offset
        point_step = msg.point_step

        # 用 numpy 批量解析 (避免 Python for 循环)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if len(raw) < n_points * point_step:
            return None

        # 将每个点的 x/y/z float32 提取出来
        points = np.zeros((n_points, 3), dtype=np.float32)
        for idx, offset in enumerate([ox, oy, oz]):
            # 创建一个视图: 从 data[offset::point_step] 每隔 point_step 取 4 字节
            byte_indices = np.arange(n_points) * point_step + offset
            byte_indices = byte_indices[:, None] + np.arange(4)[None, :]
            if byte_indices.max() >= len(raw):
                return None
            float_bytes = raw[byte_indices.ravel()].view(np.float32)
            points[:, idx] = float_bytes

        # 过滤 NaN 和无穷值
        valid = np.isfinite(points).all(axis=1)
        return points[valid]

    # ================================================================
    # 记录
    # ================================================================

    def _log_one_frame(self):
        if self.log_start_time is None:
            self.log_start_time = rospy.Time.now()

        t = (rospy.Time.now() - self.log_start_time).to_sec()
        self._frame_counter += 1

        with self.lock:
            pos = self.position.copy()
            vel = self.velocity.copy()
            quat = self.orientation.copy()
            ang_vel = self.angular_vel.copy()
            cmd_v = self.cmd_vel.copy()
            ar = self.action_raw.copy()
            # 优先使用 RL 话题的障碍距离，否则使用传感器计算值
            if self._has_min_obstacle:
                min_obs = self.min_obstacle_dist
            elif self._has_sensor_obstacle:
                min_obs = self._sensor_min_dist
            else:
                min_obs = -1.0  # 无数据
            lidar = self.lidar_scan.copy() if self.lidar_scan is not None else None

        # yaw
        qx, qy, qz, qw = quat
        yaw_rad = np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_deg = np.degrees(yaw_rad)

        # 到目标距离
        dx = self.target_x - pos[0]
        dy = self.target_y - pos[1]
        dz = self.target_z - pos[2]
        dist_to_target = np.sqrt(dx**2 + dy**2 + dz**2)

        # 目标航向
        target_yaw_deg = np.degrees(np.arctan2(dy, dx))
        yaw_error = target_yaw_deg - yaw_deg
        yaw_error = (yaw_error + 180) % 360 - 180

        # 重力投影 (从四元数)
        gx = 2.0 * (qx * qz - qw * qy)
        gy = 2.0 * (qy * qz + qw * qx)
        gz = 1.0 - 2.0 * (qx * qx + qy * qy)

        # 机体系速度 (用 yaw 旋转)
        cos_y = np.cos(-yaw_rad)
        sin_y = np.sin(-yaw_rad)
        body_vx = vel[0] * cos_y - vel[1] * sin_y
        body_vy = vel[0] * sin_y + vel[1] * cos_y
        body_vz = vel[2]

        # 高度保护 / 悬停标记
        height_prot = 1 if pos[2] < 1.0 else 0
        is_hovering = 1 if dist_to_target < 1.5 else 0

        # action_raw: 优先用 RL 话题数据，否则用 cmd_vel
        action_raw_data = ar if self._has_action_raw else cmd_v

        self.log_writer.writerow([
            f'{t:.3f}',
            f'{pos[0]:.4f}', f'{pos[1]:.4f}', f'{pos[2]:.4f}',
            f'{vel[0]:.4f}', f'{vel[1]:.4f}', f'{vel[2]:.4f}',
            f'{yaw_deg:.2f}',
            f'{self.target_x:.4f}', f'{self.target_y:.4f}', f'{self.target_z:.4f}',
            f'{dist_to_target:.4f}', f'{min_obs:.4f}',
            # action_raw
            f'{action_raw_data[0]:.4f}', f'{action_raw_data[1]:.4f}', f'{action_raw_data[2]:.4f}',
            # action (修正后) = cmd_vel
            f'{cmd_v[0]:.4f}', f'{cmd_v[1]:.4f}', f'{cmd_v[2]:.4f}',
            # cmd
            f'{cmd_v[0]:.4f}', f'{cmd_v[1]:.4f}', f'{cmd_v[2]:.4f}',
            # body vel
            f'{body_vx:.4f}', f'{body_vy:.4f}', f'{body_vz:.4f}',
            # angular vel
            f'{ang_vel[0]:.4f}', f'{ang_vel[1]:.4f}', f'{ang_vel[2]:.4f}',
            # gravity projection
            f'{gx:.4f}', f'{gy:.4f}', f'{gz:.4f}',
            # pose command (世界系相对目标位置)
            f'{dx:.4f}', f'{dy:.4f}', f'{dz:.4f}',
            # yaw
            f'{yaw_deg:.2f}', f'{target_yaw_deg:.2f}', f'{yaw_error:.2f}',
            # height
            f'{pos[2]:.4f}',
            # flags
            int(is_hovering), int(height_prot),
            # CBF 数据
            f'{self.cbf_data[0]:.4f}', f'{self.cbf_data[1]:.4f}',
            f'{self.cbf_data[2]:.4f}', f'{self.cbf_data[3]:.4f}', f'{self.cbf_data[4]:.4f}',
            f'{self.cbf_data[5]:.0f}', f'{self.cbf_data[6]:.4f}',
        ])

        # 保存雷达快照 (如果有雷达数据)
        if lidar is not None and self._frame_counter % self._lidar_snapshot_interval == 0:
            self._lidar_snapshots.append((
                t,
                pos[0], pos[1], pos[2],
                yaw_deg,
                lidar.copy()
            ))

    def _on_shutdown(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            rospy.loginfo(f"飞行数据已保存到: {self.log_path}")

        # 保存雷达快照 (npz)
        if self._lidar_snapshots:
            npz_path = self.log_path.replace('.csv', '_lidar.npz')
            times = np.array([s[0] for s in self._lidar_snapshots])
            pos_x = np.array([s[1] for s in self._lidar_snapshots])
            pos_y = np.array([s[2] for s in self._lidar_snapshots])
            pos_z = np.array([s[3] for s in self._lidar_snapshots])
            yaw_degs = np.array([s[4] for s in self._lidar_snapshots])
            scans = np.array([s[5] for s in self._lidar_snapshots])

            # 自动推断 channels × horizontal_points
            n_rays = scans.shape[1] if scans.ndim == 2 else 0
            # 默认 8 channels × 35 horizontal = 280
            channels = 8
            horizontal_points = n_rays // channels if channels > 0 else n_rays

            np.savez_compressed(npz_path,
                                times=times, scans=scans,
                                pos_x=pos_x, pos_y=pos_y, pos_z=pos_z,
                                yaw_deg=yaw_degs,
                                channels=channels,
                                horizontal_points=horizontal_points)
            rospy.loginfo(f"雷达快照已保存到: {npz_path} ({len(times)} 帧)")

        # 保存3D点云快照 (pcl.npz) — 所有方法通用
        if self._pcl_snapshots:
            pcl_path = self.log_path.replace('.csv', '_pcl.npz')
            # 合并所有点云到一个大数组, 同时记录每帧的起止索引
            all_pts = np.concatenate([s[5] for s in self._pcl_snapshots], axis=0)
            frame_sizes = np.array([len(s[5]) for s in self._pcl_snapshots])
            frame_times = np.array([s[0] for s in self._pcl_snapshots])
            frame_pos_x = np.array([s[1] for s in self._pcl_snapshots])
            frame_pos_y = np.array([s[2] for s in self._pcl_snapshots])
            frame_pos_z = np.array([s[3] for s in self._pcl_snapshots])

            np.savez_compressed(pcl_path,
                                points_enu=all_pts,      # (N, 3) 世界 ENU 坐标
                                frame_sizes=frame_sizes,  # 每帧点数
                                frame_times=frame_times,
                                frame_pos_x=frame_pos_x,
                                frame_pos_y=frame_pos_y,
                                frame_pos_z=frame_pos_z)
            rospy.loginfo(f"3D点云快照已保存: {pcl_path} ({len(self._pcl_snapshots)} 帧, {len(all_pts)} 点)")

        rospy.loginfo(f"绘图命令: python3 plot_trajectory.py {self.log_path}")
        rospy.loginfo(f"策略对比: python3 compare_policies.py")

        # 打印检测到的额外数据源
        extras = []
        if self._has_action_raw:
            extras.append("action_raw (RL)")
        if self._has_min_obstacle:
            extras.append("min_obstacle_dist (RL)")
        if self._has_sensor_obstacle:
            extras.append("obstacle_dist (传感器)")
        if self._has_lidar:
            extras.append("lidar_scan (RL)")
        if self._has_pcl:
            extras.append(f"PointCloud2 ({len(self._pcl_snapshots)} 帧)")
        if extras:
            rospy.loginfo(f"检测到的数据源: {', '.join(extras)}")
        else:
            rospy.logwarn("⚠ 未检测到障碍物距离数据源! 请检查传感器/雷达是否在运行")

    # ================================================================
    # 主循环
    # ================================================================

    def run(self):
        rate = rospy.Rate(self.control_freq)

        rospy.loginfo("等待传感器数据...")
        while not rospy.is_shutdown() and not self.data_received:
            rate.sleep()

        rospy.loginfo("收到数据，开始记录!")
        self.t0 = rospy.get_time()  # 初始化起始时间

        frame_count = 0
        while not rospy.is_shutdown():
            try:
                self._log_one_frame()
                frame_count += 1
                if frame_count % 100 == 0:
                    rospy.loginfo_throttle(30, f"已记录 {frame_count} 帧")
            except Exception as e:
                rospy.logwarn(f"记录异常: {e}")
            rate.sleep()


if __name__ == '__main__':
    try:
        logger = FlightLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass
