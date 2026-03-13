#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isaac Lab 策略部署节点
将 Isaac Lab 训练的 UAV 探索策略部署到 Prometheus 仿真系统

观测空间 (PolicyCfgUAV) - 与 velocity_env_cfg.py 完全对齐:
- base_lin_vel: 3维 (机体系线速度 vx, vy, vz)
- base_ang_vel: 3维 (机体系角速度 wx, wy, wz)
- projected_gravity: 3维 (姿态向量)
- base_height: 1维 (高度)
- pose_command: 3维 (目标位置相对坐标 dx, dy, dz) - 机体坐标系
- actions: 3维 (上一步动作 [vx, vy, vz])
- lidar_scan: 280维 (8 channels × 35 horizontal, 360°/10.3°≈35)

总观测维度: 3 + 3 + 3 + 1 + 3 + 3 + 280 = 296

动作空间 (3维) - 方案四 UAVBodyActionAutoYawCfg:
- 水平速度 x: [-1, 1] × 1.0 = [-1, 1] m/s
- 水平速度 y: [-1, 1] × 1.0 = [-1, 1] m/s  
- 垂直速度 z: [-1, 1] × 2.0 = [-2, 2] m/s
- 航向: 自动朝向目标点 (P 控制器)
"""

import os
import sys

import time
import numpy as np
from threading import Lock

# 检查 PyTorch 是否安装
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("=" * 60)
    print("ERROR: PyTorch not installed!")
    print("Please install PyTorch first:")
    print("  pip3 install torch")
    print("Or with CUDA support:")
    print("  pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 60)

import rospy

# ROS 消息类型
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

# Prometheus 消息类型
try:
    from prometheus_msgs.msg import UAVCommand, UAVState, UAVControlState
    PROMETHEUS_MSGS_AVAILABLE = True
except ImportError:
    PROMETHEUS_MSGS_AVAILABLE = False
    UAVState = None
    UAVControlState = None
    print("Warning: prometheus_msgs not found, using geometry_msgs instead")


# ============================================================================
# 增量动力学模型 (与训练代码一致)
# 参考: https://github.com/zhangchangxina/Naci_isaaclab/blob/main/scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/incremental_dynamics.py
# ============================================================================
class IncrementalDynamicsModel(torch.nn.Module):
    """增量动力学模型
    
    输入: state s, delta action du
    输出: next_state = s + B(s) @ du, reward, done_prob
    
    结构:
    - backbone_B: MLP(s) -> hidden
    - B_head: hidden -> B_flat (obs_dim * act_dim)
    - backbone_aux: MLP([s, du]) -> hidden
    - reward_head, done_head
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [512, 512], dt: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dt = dt
        
        # Backbone for B(s)
        dims_B = [obs_dim] + list(hidden_dims)
        layers_B = []
        for i in range(len(dims_B) - 1):
            layers_B.append(torch.nn.Linear(dims_B[i], dims_B[i + 1]))
            layers_B.append(torch.nn.ReLU())
        self.backbone_B = torch.nn.Sequential(*layers_B)
        self.B_head = torch.nn.Linear(dims_B[-1], obs_dim * act_dim)
        
        # Backbone for Reward/Done
        dims_aux = [obs_dim + act_dim] + list(hidden_dims)
        layers_aux = []
        for i in range(len(dims_aux) - 1):
            layers_aux.append(torch.nn.Linear(dims_aux[i], dims_aux[i + 1]))
            layers_aux.append(torch.nn.ReLU())
        self.backbone_aux = torch.nn.Sequential(*layers_aux)
        self.reward_head = torch.nn.Linear(dims_aux[-1], 1)
        self.done_head = torch.nn.Linear(dims_aux[-1], 1)
    
    def forward(self, state: torch.Tensor, delta_action: torch.Tensor):
        """前向传播
        
        Args:
            state: (batch, obs_dim)
            delta_action: (batch, act_dim)
        
        Returns:
            next_state: (batch, obs_dim)
            reward: (batch,)
            done_prob: (batch,)
        """
        # B(s) 动力学
        h_B = self.backbone_B(state)
        B_flat = self.B_head(h_B)
        B = B_flat.view(-1, self.obs_dim, self.act_dim)  # (batch, obs_dim, act_dim)
        
        du = delta_action.unsqueeze(-1)  # (batch, act_dim, 1)
        delta = torch.bmm(B, du).squeeze(-1)  # (batch, obs_dim)
        next_state = (1.0 - self.dt) * state + delta
        
        # 奖励/终止预测
        x_aux = torch.cat([state, delta_action], dim=-1)
        h_aux = self.backbone_aux(x_aux)
        reward = self.reward_head(h_aux).squeeze(-1)
        done_prob = torch.sigmoid(self.done_head(h_aux)).squeeze(-1)
        
        return next_state, reward, done_prob


class IsaacLabPolicyNode:
    """Isaac Lab 策略部署 ROS 节点"""
    
    def __init__(self):
        rospy.init_node('isaaclab_policy_node', anonymous=True)
        
        # 参数配置
        self.uav_id = rospy.get_param('~uav_id', 1)
        
        # 模型加载 - 统一接口
        # 支持格式:
        #   1. 本地文件名: "model_9999.pt" (自动在 models/ 文件夹查找)
        #   2. 本地完整路径: "/path/to/model.pt"
        #   3. Wandb URL: "wandb://entity/project/run_id/model.pt"
        #   4. Wandb 短格式: "wandb:run_id/model.pt" (使用默认 entity/project)
        if rospy.has_param('~model_name'):
            self.model_name = rospy.get_param('~model_name')
        else:
            rospy.logerr("Must specify '~model_name' parameter!")
            sys.exit(1)

        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Wandb 默认配置
        self.wandb_entity = rospy.get_param('~wandb_entity', 'zhangchangxin')
        self.wandb_project = rospy.get_param('~wandb_project', 'UAV_Navigation')
        
        # 动力学模型 checkpoint (可选，用于 CBF)
        # 允许策略使用 JIT，而动力学模型从 checkpoint 加载
        self.dyn_checkpoint_name = rospy.get_param('~dyn_checkpoint', '')
        
        # 解析模型路径
        self.policy_path = self._resolve_model_path(self.model_name)
        self.dyn_checkpoint_path = None
        if self.dyn_checkpoint_name:
            self.dyn_checkpoint_path = self._resolve_model_path(self.dyn_checkpoint_name)
        self.control_freq = rospy.get_param('~control_freq', 10.0)  # 10Hz，与训练一致
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 动作缩放参数 (与训练配置一致 - velocity_env_cfg.py)
        # 4维动作: [vx, vy, vz, yaw_offset]
        # 注意: 第四维是 yaw_offset (航向偏移量)，不是 yaw_rate！
        self.scale_hor = rospy.get_param('~scale_hor', 1.0)  # 水平速度缩放 (m/s) - 改为1.0
        self.scale_z = rospy.get_param('~scale_z', 2.0)      # 垂直速度缩放 (m/s)
        # 方案四: 航向自动朝向目标，不需要 scale_yaw
        self.yaw_p_gain = rospy.get_param('~yaw_p_gain', 2.0)      # 航向 P 控制器增益
        
        # 安全参数
        self.min_altitude = rospy.get_param('~min_altitude', 1.0)  # 最低飞行高度 (米)
        
        # 速度限制 (与 Isaac Lab 训练配置一致 - velocity_env_cfg.py)
        self.max_vel_hor = rospy.get_param('~max_vel_hor', 1.0)    # 最大水平速度 (m/s) - 改为1.0
        self.max_vel_up = rospy.get_param('~max_vel_up', 2.0)      # 最大上升速度 (m/s)
        self.max_vel_down = rospy.get_param('~max_vel_down', 1.0)  # 最大下降速度 (m/s)
        
        # 加速度限制 (匹配 PX4 限制 MPC_ACC_HOR=2.0)
        self.max_acc_hor = rospy.get_param('~max_acc_hor', 2.0)    # 最大水平加速度 (m/s²)
        self.max_acc_up = rospy.get_param('~max_acc_up', 3.0)      # 最大上升加速度 (m/s²)
        self.max_acc_down = rospy.get_param('~max_acc_down', 2.0)  # 最大下降加速度 (m/s²)
        
        # CBF (Control Barrier Function) 安全参数 (与训练配置一致)
        self.use_cbf = rospy.get_param('~use_cbf', False)           # 是否启用 CBF 安全层
        self.cbf_gamma = rospy.get_param('~cbf_gamma', 0.5)        # CBF gamma 参数 (安全裕度)
        self.cbf_safe_dist = rospy.get_param('~cbf_safe_dist', 1.0)  # Barrier 函数作用距离 (米)
        self.cbf_repulsion_dist = rospy.get_param('~cbf_repulsion_dist', 0.5)  # 硬排斥触发距离 (米)
        self.cbf_solver = rospy.get_param('~cbf_solver', 'gradient')  # 求解器: gradient | slsqp
        self.cbf_barrier = rospy.get_param('~cbf_barrier', 'log')     # barrier: log | softplus | reciprocal
        self.cbf_repulsion_gain = rospy.get_param('~cbf_repulsion_gain', 0.0)  # 几何排斥力增益 (0=关闭, 2.0=原始强度)
        
        # 调试选项 - 禁用雷达 (全部设为最大值 5m)
        self.disable_lidar = rospy.get_param('~disable_lidar', False)
        if self.disable_lidar:
            rospy.logwarn("="*60)
            rospy.logwarn("[DEBUG] 雷达已禁用！所有雷达数据将设为 5m (无障碍)")
            rospy.logwarn("="*60)
        
        # 调试选项 - 测试模式 (固定动作输出)
        self.test_mode = rospy.get_param('~test_mode', False)
        if self.test_mode:
            rospy.logwarn("="*60)
            rospy.logwarn("[TEST] 测试模式启用！使用固定动作 (0.5, 0, 0)")
            rospy.logwarn("[TEST] 无人机应该往前飞！如果往后飞说明坐标系有问题")
            rospy.logwarn("="*60)
        
        # CBF 测试模式 - 策略输出固定值，专门验证 CBF 安全层
        self.cbf_test_mode = rospy.get_param('~cbf_test_mode', False)
        self.cbf_test_action_vx = rospy.get_param('~cbf_test_action_vx', 0.0)
        self.cbf_test_action_vy = rospy.get_param('~cbf_test_action_vy', 0.0)
        self.cbf_test_action_vz = rospy.get_param('~cbf_test_action_vz', 0.0)
        if self.cbf_test_mode:
            # CBF 测试模式自动启用 CBF
            self.use_cbf = True
            rospy.logwarn("="*60)
            rospy.logwarn("[CBF-TEST] CBF 测试模式启用!")
            rospy.logwarn(f"[CBF-TEST] 固定动作: vx={self.cbf_test_action_vx}, "
                         f"vy={self.cbf_test_action_vy}, vz={self.cbf_test_action_vz}")
            rospy.logwarn(f"[CBF-TEST] CBF 已强制启用 (gamma={self.cbf_gamma}, safe_dist={self.cbf_safe_dist}m)")
            rospy.logwarn("[CBF-TEST] 如果 CBF 生效，输出速度会与固定动作不同")
            rospy.logwarn("="*60)
        
        # 目标点附近减速/悬停模式
        # none  = 不减速, 策略全程控制
        # decel = 指数减速 (1m内指数衰减, 先慢后快, 障碍物<2m取消减速)
        # hover = 悬停 (XY<1m悬停, 滞回1.5m退出, 障碍物<2m取消悬停)
        self.target_stop_mode = rospy.get_param('~target_stop_mode', 'hover')
        rospy.loginfo(f"目标点停止模式: {self.target_stop_mode}")
        
        # 动作符号修正 (如果策略输出方向相反)
        self.invert_vx = rospy.get_param('~invert_vx', False)
        self.invert_vy = rospy.get_param('~invert_vy', False)
        if self.invert_vx or self.invert_vy:
            rospy.logwarn("="*60)
            rospy.logwarn(f"[FIX] 动作符号修正: invert_vx={self.invert_vx}, invert_vy={self.invert_vy}")
            rospy.logwarn("="*60)
        
        # 目标位置 (可通过话题或参数设置)
        # 注意: AirSim ROS wrapper 已将坐标转为 ENU（Z向上为正）
        self.target_x = rospy.get_param('~target_x', 22.0)
        self.target_y = rospy.get_param('~target_y', 22.0)
        self.target_z = rospy.get_param('~target_z', 1.0)  # 目标高度（ENU，正值=高度）
        
        # 世界坐标系校正: Prometheus/PX4 的"世界"和 AirSim 实际世界的 yaw 偏移
        # 如果 rviz 中 world X 轴方向和 AirSim 中不一致，调整此参数
        # 正值 = Prometheus 世界相对于 AirSim 世界顺时针旋转
        self.world_yaw_offset = rospy.get_param('~world_yaw_offset', 0.0)  # 度
        
        # 雷达参数 (与 Isaac Lab LidarPatternCfg 对齐: 8 channels × 35 horizontal = 280 rays)
        # 模型输入: 291 = 3(gravity) + 1(height) + 3(target) + 4(action) + 280(lidar)
        # 雷达配置与 Isaac Lab LidarPatternCfg 完全对齐:
        #   horizontal_res=10.0 → 360°/10° = 36, 但 360° FOV 排除最后点 → 35 条水平射线
        #   channels=8, vertical_fov=(-7°, 52°)
        self.lidar_range = rospy.get_param('~lidar_range', 5.0)  # clip范围
        self.lidar_channels = rospy.get_param('~lidar_channels', 8)  # 垂直通道数
        self.lidar_horizontal_points = rospy.get_param('~lidar_horizontal_points', 35)  # 水平点数 (360°/10.3°≈35)
        self.lidar_num_rays = self.lidar_channels * self.lidar_horizontal_points  # 总射线数 = 280
        self.use_3d_lidar = rospy.get_param('~use_3d_lidar', True)  # 是否使用3D雷达
        
        # 雷达坐标系修正 (用于对齐训练环境传感器朝向)
        # 例如: AirSim 设置 Pitch=-15，而训练是 +15，可设置 lidar_corr_pitch=30
        self.lidar_corr_roll = rospy.get_param('~lidar_corr_roll', 0.0)
        self.lidar_corr_pitch = rospy.get_param('~lidar_corr_pitch', 0.0)
        self.lidar_corr_yaw = rospy.get_param('~lidar_corr_yaw', 0.0)
        # 轴镜像修正 (当左右/前后翻转时使用)
        self.lidar_flip_x = rospy.get_param('~lidar_flip_x', False)
        self.lidar_flip_y = rospy.get_param('~lidar_flip_y', False)
        self.lidar_flip_z = rospy.get_param('~lidar_flip_z', False)
        self.lidar_corr_R = None
        if abs(self.lidar_corr_roll) > 1e-6 or abs(self.lidar_corr_pitch) > 1e-6 or abs(self.lidar_corr_yaw) > 1e-6:
            self.lidar_corr_R = self._rpy_to_rot_matrix(
                self.lidar_corr_roll, self.lidar_corr_pitch, self.lidar_corr_yaw
            )
            rospy.logwarn("="*60)
            rospy.logwarn(
                f"[Lidar] 使用坐标系修正: roll={self.lidar_corr_roll}°, pitch={self.lidar_corr_pitch}°, yaw={self.lidar_corr_yaw}°"
            )
            rospy.logwarn("="*60)
        
        # 状态变量
        self.lock = Lock()
        self.position = np.zeros(3)      # x, y, z
        self.velocity = np.zeros(3)      # vx, vy, vz
        self.orientation = np.zeros(4)   # qx, qy, qz, qw
        self.angular_vel = np.zeros(3)   # wx, wy, wz
        self.lidar_data = np.full(self.lidar_num_rays, self.lidar_range)  # 初始化为最远距离(无障碍物)
        self.last_action = np.zeros(3)   # [vx, vy, vz] - 3维动作
        self.last_cmd_vel = np.zeros(3)  # 上一次发送的速度命令 [vx, vy, vz]
        self.last_cmd_time = None        # 上一次命令时间
        self.command_id = 0              # Prometheus 命令ID (需要递增)
        self.debug_counter = 0           # 调试计数器
        self.is_hovering = False         # 悬停状态 (滞回: <1.0m进入, >1.5m退出, 障碍物<2m取消)
        self.height_protection = False   # 高度保护触发
        self.data_received = {
            'odom': False,
            'lidar': False
        }
        
        # 内部变量 (供话题发布使用)
        self._log_lidar_scan = np.zeros(self.lidar_num_rays)
        self._log_action_raw = np.zeros(3)
        # 以下变量在 build_observation / publish_command 中更新
        self._log_base_lin_vel = np.zeros(3)
        self._log_base_ang_vel = np.zeros(3)
        self._log_projected_gravity = np.zeros(3)
        self._log_base_height = 0.0
        self._log_pose_command = np.zeros(3)
        self._log_current_yaw_deg = 0.0
        self._log_target_yaw_deg = 0.0
        
        # 动力学模型 (用于 Model-Based CBF)
        self.dynamics_model = None
        
        # 加载策略模型 (可能含动力学模型)
        self.load_policy()
        
        # 如果策略是 JIT 或不含动力学模型，则从独立 checkpoint 加载动力学模型
        if self.dyn_checkpoint_path and self.dynamics_model is None:
            self._load_dynamics_from_path(self.dyn_checkpoint_path)
        
        # 设置话题名称前缀
        self.topic_prefix = f'/uav{self.uav_id}'
        
        # 订阅者
        self.setup_subscribers()
        
        # 发布者
        self.setup_publishers()
        
        # 目标位置订阅 (可动态更新)
        rospy.Subscriber(f'{self.topic_prefix}/prometheus/motion_planning/goal', 
                        PoseStamped, self.target_callback)
        
        # ============================================================
        # 飞行数据记录已移至通用记录器 flight_logger.py
        # 使用方法: python3 flight_logger.py _method:=isaaclab_rl
        # ============================================================
        self._policy_label = self._make_policy_label(self.model_name)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(f"策略标签: {self._policy_label}")
        rospy.loginfo(f"数据记录请使用: ./run_logger.sh {self._policy_label}")
        
        rospy.loginfo(f"Isaac Lab Policy Node initialized for UAV{self.uav_id}")
        rospy.loginfo(f"Policy loaded from: {self.policy_path}")
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo(f"Target position: ({self.target_x}, {self.target_y}, {self.target_z})")
        rospy.loginfo(f"Safety limits:")
        rospy.loginfo(f"  - Min altitude: {self.min_altitude}m")
        rospy.loginfo(f"  - Max velocity: hor={self.max_vel_hor}m/s, up={self.max_vel_up}m/s, down={self.max_vel_down}m/s")
        rospy.loginfo(f"  - Max acceleration: hor={self.max_acc_hor}m/s², up={self.max_acc_up}m/s², down={self.max_acc_down}m/s²")
        
        # CBF 状态 (需要动力学模型才能启用)
        if self.dynamics_model is not None:
            if self.use_cbf:
                rospy.loginfo(f"  - Model-Based CBF: ENABLED, gamma={self.cbf_gamma}, safe_dist={self.cbf_safe_dist}m, solver={self.cbf_solver}, barrier={self.cbf_barrier}")
            else:
                rospy.loginfo(f"  - Model-Based CBF: DISABLED (use_cbf=false)")
        else:
            rospy.logwarn(f"  - Model-Based CBF: DISABLED (no dynamics model found)")
            if self.use_cbf:
                rospy.logwarn(f"    Note: use_cbf=true but no dynamics model, CBF will not be used")
    
    def load_policy(self):
        """加载 PyTorch 策略模型 (支持多种格式: 本地文件 / wandb)"""
        try:
            # policy_path 已在 __init__ 中通过 _resolve_model_path 解析
            rospy.loginfo(f"Loading policy from {self.policy_path}...")
            
            # 方式1: 尝试作为 TorchScript JIT 模型加载
            try:
                self.policy = torch.jit.load(self.policy_path, map_location=self.device)
                self.policy.eval()
                self.policy_type = "jit"
                
                # 检查 JIT 模型是否包含归一化器
                # 通过检查模型代码中是否有 "normalizer" 来判断
                try:
                    code_str = str(self.policy.code)
                    has_normalizer = "normalizer" in code_str.lower()
                    if has_normalizer:
                        rospy.loginfo("Policy loaded as TorchScript JIT model with built-in normalization!")
                    else:
                        rospy.logwarn("Policy loaded as TorchScript JIT model, but normalization not detected!")
                        rospy.logwarn("  If model was exported without normalizer, obs normalization may be needed.")
                except Exception as e:
                    rospy.logwarn(f"Could not check for normalization in JIT model: {e}")
                
                return
            except Exception as e:
                rospy.logwarn(f"JIT load failed: {e}, trying checkpoint format...")
            
            # 方式2: 尝试作为 checkpoint 加载
            checkpoint = torch.load(self.policy_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # RSL-RL checkpoint 格式
                if 'model_state_dict' in checkpoint:
                    rospy.loginfo("Detected RSL-RL checkpoint format")
                    # 需要重建模型结构 - 这里假设是 ActorCritic
                    # 用户需要根据实际模型结构修改
                    self.policy = self._build_actor_critic(checkpoint)
                    self.policy_type = "checkpoint"
                    
                    # 加载动力学模型 (用于 Model-Based CBF)
                    self._load_dynamics_model(checkpoint)
                    
                elif 'actor' in checkpoint:
                    self.policy = checkpoint['actor']
                    self.policy_type = "actor"
                else:
                    rospy.logerr(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")
                    raise ValueError("Cannot determine model structure from checkpoint")
            else:
                # 直接是模型对象
                self.policy = checkpoint
                self.policy_type = "direct"
            
            self.policy.eval()
            rospy.loginfo(f"Policy loaded successfully! Type: {self.policy_type}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load policy: {e}")
            rospy.logerr("Please check if the model file exists and is valid.")
            rospy.logerr("You may need to export the model as TorchScript JIT format.")
            raise

    def _load_dynamics_from_path(self, checkpoint_path: str):
        """从 checkpoint 文件加载动力学模型 (用于 CBF)"""
        try:
            if not checkpoint_path:
                return
            if not os.path.exists(checkpoint_path):
                rospy.logwarn(f"Dynamics checkpoint not found: {checkpoint_path}")
                return
            rospy.loginfo(f"Loading dynamics model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                rospy.logwarn("Dynamics checkpoint is not a dict; skip loading dynamics model.")
                return
            self._load_dynamics_model(checkpoint)
        except Exception as e:
            rospy.logwarn(f"Failed to load dynamics model from checkpoint: {e}")
    
    @staticmethod
    def _make_policy_label(model_name):
        """
        从模型名称生成日志文件夹标签
        
        示例:
            "policy.pt"                    → "policy"
            "model_9999.pt"                → "model_9999"
            "drone_rough_..._ppo_model_19999_policy.pt" → "drone_rough_..._ppo_model_19999_policy"
            "wandb:run_id/model_7000.pt"   → "model_7000"
            "/abs/path/to/my_model.pt"     → "my_model"
        """
        import re
        name = model_name
        # 去掉 wandb 前缀
        if name.startswith('wandb://') or name.startswith('wandb:'):
            name = name.split('/')[-1]
        # 取文件名
        name = os.path.basename(name)
        # 去掉扩展名
        name = os.path.splitext(name)[0]
        # 去掉不安全字符 (只保留字母数字下划线和连字符)
        name = re.sub(r'[^\w\-]', '_', name)
        if not name:
            name = 'unknown_policy'
        return name
    
    def _resolve_model_path(self, model_name):
        """
        统一解析模型路径
        
        支持格式:
        1. "model_9999.pt" → models/model_9999.pt (本地)
        2. "/full/path/model.pt" → 直接使用 (本地)
        3. "wandb://entity/project/run_id/model.pt" → 下载到本地
        4. "wandb:run_id/model.pt" → 使用默认 entity/project 下载
        
        Returns:
            本地模型文件的完整路径
        """
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Wandb 完整 URL
        if model_name.startswith('wandb://'):
            return self._download_from_wandb(model_name)
        
        # Wandb 短格式 (使用默认 entity/project)
        if model_name.startswith('wandb:'):
            short_path = model_name.replace('wandb:', '')
            full_url = f"wandb://{self.wandb_entity}/{self.wandb_project}/{short_path}"
            return self._download_from_wandb(full_url)
        
        # 完整本地路径
        if os.path.isabs(model_name) or model_name.startswith('/'):
            return model_name
        
        # 相对路径 - 在 models/ 文件夹查找
        local_path = os.path.join(self.models_dir, model_name)
        if os.path.exists(local_path):
            return local_path
        
        # 尝试当前目录
        if os.path.exists(model_name):
            return os.path.abspath(model_name)
        
        # 默认返回 models/ 下的路径 (可能不存在，load_policy 会报错)
        return local_path
    
    def _download_from_wandb(self, wandb_path):
        """
        从 wandb 下载模型文件
        
        支持格式:
        1. wandb://entity/project/run_id/model.pt  (从 run files 下载)
        2. wandb://entity/project/artifact_name:version  (从 artifact 下载)
        3. 直接指定 run_id (使用配置的 entity/project)
        
        Args:
            wandb_path: wandb 路径或 run_id
            
        Returns:
            本地模型文件路径
        """
        try:
            import wandb
        except ImportError:
            rospy.logerr("wandb not installed! Run: pip3 install wandb")
            raise ImportError("wandb not installed")
        
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 解析 wandb:// URL
        if wandb_path.startswith('wandb://'):
            parts = wandb_path.replace('wandb://', '').split('/')
            if len(parts) >= 3:
                entity = parts[0]
                project = parts[1]
                run_or_artifact = '/'.join(parts[2:])
            else:
                rospy.logerr(f"Invalid wandb path: {wandb_path}")
                rospy.logerr("Expected: wandb://entity/project/run_id/file.pt")
                raise ValueError("Invalid wandb path")
        else:
            # 使用配置的 entity/project
            entity = self.wandb_entity
            project = self.wandb_project
            run_or_artifact = wandb_path
        
        rospy.loginfo(f"Downloading from wandb: {entity}/{project}/{run_or_artifact}")
        
        # 初始化 wandb API
        api = wandb.Api()
        
        try:
            # 尝试作为 artifact 加载
            if ':' in run_or_artifact:
                artifact_name = f"{entity}/{project}/{run_or_artifact}"
                rospy.loginfo(f"Loading as artifact: {artifact_name}")
                artifact = api.artifact(artifact_name)
                artifact_dir = artifact.download(root=models_dir)
                # 找到 .pt 文件
                for f in os.listdir(artifact_dir):
                    if f.endswith('.pt'):
                        return os.path.join(artifact_dir, f)
                raise FileNotFoundError("No .pt file found in artifact")
            else:
                # 作为 run files 加载
                if '/' in run_or_artifact:
                    run_id, filename = run_or_artifact.rsplit('/', 1)
                else:
                    run_id = run_or_artifact
                    filename = 'model.pt'  # 默认文件名
                
                run_path = f"{entity}/{project}/{run_id}"
                rospy.loginfo(f"Loading from run: {run_path}, file: {filename}")
                
                run = api.run(run_path)
                
                # 下载模型文件
                local_path = os.path.join(models_dir, f"{run_id}_{filename}")
                
                # 检查本地是否已有缓存
                if os.path.exists(local_path):
                    rospy.loginfo(f"Using cached model: {local_path}")
                    return local_path
                
                # 从 wandb 下载
                for file in run.files():
                    if file.name == filename or file.name.endswith(filename):
                        rospy.loginfo(f"Downloading: {file.name}")
                        file.download(root=models_dir, replace=True)
                        downloaded_path = os.path.join(models_dir, file.name)
                        # 重命名以包含 run_id
                        if os.path.exists(downloaded_path):
                            os.rename(downloaded_path, local_path)
                            rospy.loginfo(f"Model saved to: {local_path}")
                            return local_path
                
                raise FileNotFoundError(f"File {filename} not found in run {run_id}")
                
        except Exception as e:
            rospy.logerr(f"Failed to download from wandb: {e}")
            raise
    
    def _build_actor_critic(self, checkpoint):
        """
        从 checkpoint 重建 Actor-Critic 模型
        注意: 这是一个简化版本，可能需要根据实际模型结构修改
        """
        rospy.logwarn("Building model from checkpoint - this may need customization!")
        
        # 获取模型状态字典
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 尝试推断网络结构
        # 这里需要根据您的实际模型结构来定义
        # 以下是一个示例 MLP Actor
        
        import torch.nn as nn
        
        # 从 state_dict 推断输入输出维度
        first_layer_key = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k][0]
        last_layer_key = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k][-1]
        
        input_dim = state_dict[first_layer_key].shape[1]
        output_dim = state_dict[last_layer_key].shape[0]
        
        rospy.loginfo(f"Inferred dimensions: input={input_dim}, output={output_dim}")
        
        # 简单 MLP Actor (可能需要修改)
        class SimpleActor(nn.Module):
            def __init__(self, obs_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.ELU(),
                    nn.Linear(256, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, action_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.net(x)
        
        actor = SimpleActor(input_dim, output_dim)
        
        # 尝试加载权重 (可能需要调整 key 映射)
        try:
            actor.load_state_dict(state_dict, strict=False)
        except Exception as e:
            rospy.logwarn(f"Could not load all weights: {e}")
        
        return actor.to(self.device)
    
    def _load_dynamics_model(self, checkpoint):
        """
        从 checkpoint 加载增量动力学模型 (用于 Model-Based CBF)
        
        参考: https://github.com/zhangchangxina/Naci_isaaclab/blob/main/scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/incremental_dynamics.py
        """
        dyn_state_dicts = checkpoint.get('dyn_state_dicts', [])
        
        if not dyn_state_dicts:
            rospy.logwarn("No dynamics model found in checkpoint (dyn_state_dicts is empty)")
            return
        
        rospy.loginfo(f"Found {len(dyn_state_dicts)} dynamics model(s) in checkpoint")
        
        # 使用第一个动力学模型
        dyn_state_dict = dyn_state_dicts[0]
        
        # 从权重推断网络结构
        try:
            # obs_dim 从 backbone_B 第一层推断
            obs_dim = dyn_state_dict['backbone_B.0.weight'].shape[1]
            # hidden_dims 从 backbone 层推断
            hidden_dim = dyn_state_dict['backbone_B.0.weight'].shape[0]
            # act_dim 从 B_head 输出推断
            b_head_out = dyn_state_dict['B_head.weight'].shape[0]
            act_dim = b_head_out // obs_dim
            
            rospy.loginfo(f"Dynamics model structure: obs_dim={obs_dim}, act_dim={act_dim}, hidden=[{hidden_dim}, {hidden_dim}]")
            
            # 创建动力学模型
            self.dynamics_model = IncrementalDynamicsModel(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                dt=0.0
            ).to(self.device)
            
            # 加载权重
            self.dynamics_model.load_state_dict(dyn_state_dict)
            self.dynamics_model.eval()
            
            rospy.loginfo("Dynamics model loaded successfully! Model-Based CBF enabled.")
            
        except Exception as e:
            rospy.logerr(f"Failed to load dynamics model: {e}")
            self.dynamics_model = None
    
    def setup_subscribers(self):
        """设置订阅者"""
        # 数据源优先级: prometheus/state > mavros/odom
        # 只有当没收到高优先级数据时，才使用低优先级数据
        self.use_prometheus_state = False  # 标记是否收到过 prometheus 数据
        
        if PROMETHEUS_MSGS_AVAILABLE:
            rospy.Subscriber(f'{self.topic_prefix}/prometheus/state',
                            UAVState, self.uav_state_callback)
            rospy.loginfo("Subscribed to Prometheus state (primary)")
        
        # Mavros 作为备用 (只有 prometheus 没数据时才用)
        rospy.Subscriber(f'{self.topic_prefix}/mavros/local_position/odom',
                        Odometry, self.odom_callback)
        rospy.loginfo("Subscribed to Mavros odom (fallback)")
        
        # 3D 激光雷达 (PointCloud2)
        if self.use_3d_lidar:
            # Prometheus 标准 3D 雷达
            rospy.Subscriber(f'{self.topic_prefix}/prometheus/sensors/3Dlidar',
                            PointCloud2, self.pointcloud_callback)
            # Livox 雷达 (实机: /livox/lidar 和 /uav1/cloud_mid360_body)
            # rospy.Subscriber(f'/livox/lidar',
            #                 PointCloud2, self.pointcloud_callback)
            rospy.Subscriber(f'/uav1/cloud_mid360_body',
                            PointCloud2, self.pointcloud_callback)
            # Velodyne 雷达
            rospy.Subscriber(f'{self.topic_prefix}/velodyne_points',
                            PointCloud2, self.pointcloud_callback)
            # 通用点云
            rospy.Subscriber(f'{self.topic_prefix}/points_raw',
                            PointCloud2, self.pointcloud_callback)
            # AirSim 3D 雷达 (多种命名格式)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidar/Lidar',
                            PointCloud2, self.pointcloud_callback)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidar/LidarSensor',
                            PointCloud2, self.pointcloud_callback)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidar/LidarSensor1',
                            PointCloud2, self.pointcloud_callback)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidar/lidar3d',
                            PointCloud2, self.pointcloud_callback)
            # AirSim PointCloud2 格式 (新版本)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidarPointCloud2/Lidar',
                            PointCloud2, self.pointcloud_callback)
            rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidarPointCloud2/LidarSensor1',
                            PointCloud2, self.pointcloud_callback)
        
        # 2D 激光雷达 (LaserScan)
        rospy.Subscriber(f'{self.topic_prefix}/prometheus/sensors/2Dlidar_scan',
                        LaserScan, self.lidar_callback_2d)
        rospy.Subscriber(f'{self.topic_prefix}/scan',
                        LaserScan, self.lidar_callback_2d)
        # Gazebo 滤波后的雷达数据
        rospy.Subscriber(f'{self.topic_prefix}/scan_filtered',
                        LaserScan, self.lidar_callback_2d)
        # AirSim 2D 激光雷达
        rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidarLaserScan/LidarSensor1',
                        LaserScan, self.lidar_callback_2d)
        rospy.Subscriber(f'/airsim_node/uav{self.uav_id}/lidar2d',
                        LaserScan, self.lidar_callback_2d)
    
    def setup_publishers(self):
        """设置发布者"""
        if PROMETHEUS_MSGS_AVAILABLE:
            self.cmd_pub = rospy.Publisher(
                f'{self.topic_prefix}/prometheus/command',
                UAVCommand, queue_size=10)
        else:
            # 使用 geometry_msgs 作为备用
            self.cmd_pub = rospy.Publisher(
                f'{self.topic_prefix}/mavros/setpoint_velocity/cmd_vel',
                TwistStamped, queue_size=10)
        
        # 调试用: 发布观测和动作
        self.obs_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/observation',
            Float32MultiArray, queue_size=10)
        self.action_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/action',
            Float32MultiArray, queue_size=10)
        
        # ---- 供通用 flight_logger 订阅的额外话题 ----
        from std_msgs.msg import Float32
        # 策略原始输出 (安全过滤前)
        self.action_raw_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/action_raw',
            Float32MultiArray, queue_size=10)
        # 最近障碍物距离
        self.min_obstacle_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/min_obstacle_dist',
            Float32, queue_size=10)
        # 处理后的雷达扫描 (280维: 8ch × 35水平)
        self.lidar_scan_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/lidar_scan',
            Float32MultiArray, queue_size=10)
        # CBF 数据 (7维: barrier_value, violation, delta_vx, delta_vy, delta_vz, cbf_active, min_barrier_dist)
        self.cbf_data_pub = rospy.Publisher(
            f'{self.topic_prefix}/isaaclab/cbf_data',
            Float32MultiArray, queue_size=10)
        
        # rviz 可视化
        from visualization_msgs.msg import Marker, MarkerArray
        from nav_msgs.msg import Path
        self.target_marker_pub = rospy.Publisher(
            f'{self.topic_prefix}/policy_target_marker',
            Marker, queue_size=10)
        self.velocity_marker_pub = rospy.Publisher(
            f'{self.topic_prefix}/policy_velocity_marker',
            Marker, queue_size=10)
        self.path_pub = rospy.Publisher(
            f'{self.topic_prefix}/policy_path',
            Path, queue_size=10)
        # 雷达射线可视化
        self.lidar_marker_pub = rospy.Publisher(
            f'{self.topic_prefix}/policy_lidar_rays',
            MarkerArray, queue_size=10)
        
        # 重新发布点云到 Prometheus frame (uav1/lidar_link)
        from sensor_msgs.msg import PointCloud2
        self.lidar_republish_pub = rospy.Publisher(
            f'{self.topic_prefix}/policy_lidar_cloud',
            PointCloud2, queue_size=10)
        
        # 存储路径历史
        self.path_history = []
        self.max_path_points = 500
    
    def uav_state_callback(self, msg):
        """Prometheus UAVState 回调 (优先数据源)
        
        prometheus_msgs/UAVState 包含:
        - position[3]: 位置 [m]
        - velocity[3]: 速度 [m/s]
        - attitude[3]: 姿态 (欧拉角) [rad]
        - attitude_rate[3]: 角速度 [rad/s]
        """
        with self.lock:
            self.position[0] = msg.position[0]
            self.position[1] = msg.position[1]
            self.position[2] = msg.position[2]
            
            self.velocity[0] = msg.velocity[0]
            self.velocity[1] = msg.velocity[1]
            self.velocity[2] = msg.velocity[2]
            
            # 姿态四元数
            self.orientation[0] = msg.attitude_q.x
            self.orientation[1] = msg.attitude_q.y
            self.orientation[2] = msg.attitude_q.z
            self.orientation[3] = msg.attitude_q.w
            
            self.angular_vel[0] = msg.attitude_rate[0]
            self.angular_vel[1] = msg.attitude_rate[1]
            self.angular_vel[2] = msg.attitude_rate[2]
            
            self.data_received['odom'] = True
            self.use_prometheus_state = True  # 标记已收到 prometheus 数据
    
    def odom_callback(self, msg):
        """里程计回调 (仅当没有 prometheus 数据时使用)"""
        with self.lock:
            # 如果已收到 prometheus 数据，跳过 odom 更新
            if hasattr(self, 'use_prometheus_state') and self.use_prometheus_state:
                return
            
            self.position[0] = msg.pose.pose.position.x
            self.position[1] = msg.pose.pose.position.y
            self.position[2] = msg.pose.pose.position.z
            
            self.velocity[0] = msg.twist.twist.linear.x
            self.velocity[1] = msg.twist.twist.linear.y
            self.velocity[2] = msg.twist.twist.linear.z
            
            self.orientation[0] = msg.pose.pose.orientation.x
            self.orientation[1] = msg.pose.pose.orientation.y
            self.orientation[2] = msg.pose.pose.orientation.z
            self.orientation[3] = msg.pose.pose.orientation.w
            
            self.angular_vel[0] = msg.twist.twist.angular.x
            self.angular_vel[1] = msg.twist.twist.angular.y
            self.angular_vel[2] = msg.twist.twist.angular.z
            
            self.data_received['odom'] = True
    
    def pose_callback(self, msg):
        """位置回调 (备用)"""
        with self.lock:
            self.position[0] = msg.pose.position.x
            self.position[1] = msg.pose.position.y
            self.position[2] = msg.pose.position.z
            
            self.orientation[0] = msg.pose.orientation.x
            self.orientation[1] = msg.pose.orientation.y
            self.orientation[2] = msg.pose.orientation.z
            self.orientation[3] = msg.pose.orientation.w
    
    def velocity_callback(self, msg):
        """速度回调 (备用)"""
        with self.lock:
            self.velocity[0] = msg.twist.linear.x
            self.velocity[1] = msg.twist.linear.y
            self.velocity[2] = msg.twist.linear.z
            
            self.angular_vel[0] = msg.twist.angular.x
            self.angular_vel[1] = msg.twist.angular.y
            self.angular_vel[2] = msg.twist.angular.z
    
    def pointcloud_callback(self, msg):
        """3D 点云雷达回调 (PointCloud2)
        
        Isaac Lab 雷达配置:
        - channels=8 (垂直通道)
        - horizontal_fov=360°, horizontal_res≈10.3° → 35 水平点
        - 总射线数: 8 × 35 = 280
        
        输出: 每个射线的距离值 (米)，范围 [0, lidar_range]，不归一化
        """
        with self.lock:
            try:
                points = []
                for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    points.append(point)
                
                if len(points) == 0:
                    rospy.logwarn_throttle(5.0, "Empty point cloud received")
                    return
                
                points = np.array(points)
                
                # 调试: 分析原始点云 X 坐标分布
                x_positive = np.sum(points[:, 0] > 0)
                x_negative = np.sum(points[:, 0] < 0)
                rospy.loginfo_throttle(
                    2.0,
                    f"[Lidar Raw] X>0: {x_positive}, X<0: {x_negative}, "
                    f"X_mean={np.mean(points[:, 0]):.2f}"
                )
                
                # 过滤明显异常的点 (AirSim 偶发 1e8~1e9)
                max_abs = 1e3
                finite_mask = np.isfinite(points).all(axis=1)
                abs_mask = (np.abs(points) < max_abs).all(axis=1)
                valid_xyz_mask = finite_mask & abs_mask
                if not np.all(valid_xyz_mask):
                    dropped = np.sum(~valid_xyz_mask)
                    rospy.logwarn_throttle(2.0, f"[Lidar] Drop invalid xyz points: {dropped}/{len(points)}")
                points = points[valid_xyz_mask]
        
                # 应用坐标系旋转修正 (如果设置了 lidar_corr_roll/pitch/yaw)
                if self.lidar_corr_R is not None:
                    points = (self.lidar_corr_R @ points.T).T
                
                # 应用轴镜像修正 (如果启用)
                if self.lidar_flip_x:
                    points[:, 0] = -points[:, 0]
                if self.lidar_flip_y:
                    points[:, 1] = -points[:, 1]
                if self.lidar_flip_z:
                    points[:, 2] = -points[:, 2]
                
                # ============================================================
                # 点云坐标转换: raw → 机体 heading frame (FLU)
                # ============================================================
                # 实测验证结果:
                #   R(-yaw)  → 障碍物墙会旋转 (错误)
                #   R(-2*yaw) → 障碍物墙稳定，轻微抖动 (正确)
                #   NED→ENU swap + R(-yaw) → 反转+旋转 (更错)
                #
                # 分析: 到达 callback 的点云数据满足:
                #   raw = R(+yaw) * world_enu_offset
                #   即原始点云是世界ENU偏移再旋转了一个+yaw
                # 因此:
                #   body = R(-yaw) * world_enu = R(-yaw) * R(-yaw) * raw = R(-2*yaw) * raw
                #   world_enu = R(-yaw) * raw  (flight_logger.py 用这个)
                #
                # 轻微抖动原因: 角度 bin 量化效应 (水平分辨率 ~10.3°),
                #   yaw 变化时点在相邻 bin 间跳动, 属正常现象.
                # ============================================================
                qx, qy, qz, qw = self.orientation
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                # 绕 Z 轴旋转 -2*yaw (将 raw 点云转到机体 heading frame)
                cos_neg_2yaw = np.cos(-2 * yaw)
                sin_neg_2yaw = np.sin(-2 * yaw)
                x_new = points[:, 0] * cos_neg_2yaw - points[:, 1] * sin_neg_2yaw
                y_new = points[:, 0] * sin_neg_2yaw + points[:, 1] * cos_neg_2yaw
                points[:, 0] = x_new
                points[:, 1] = y_new
                
                # 计算每个点的 3D 距离 (与 Isaac Lab lidar_scan 一致)
                # Isaac Lab: depth = torch.norm(ray_hits_w - pos_w.unsqueeze(1), dim=-1)
                distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)  # 3D 距离
                
                # 记录原始数据统计
                raw_count = len(distances)
                raw_min = np.min(distances) if raw_count > 0 else 0
                
                # 过滤无效点 (放宽范围: 0.05m ~ 100m)
                raw_min = float(np.min(distances)) if len(distances) > 0 else 0.0
                raw_max = float(np.max(distances)) if len(distances) > 0 else 0.0
                valid_mask = (distances > 0.30) & (distances < 7.0)
                points = points[valid_mask]
                distances = distances[valid_mask]
                
                if len(distances) > 0:
                    in_range = np.sum(distances <= self.lidar_range)
                    rospy.loginfo_throttle(
                        2.0,
                        f"[Lidar Dist] raw_min={raw_min:.2f}, raw_max={raw_max:.2f}, "
                        f"in_range(<= {self.lidar_range}m)={in_range}/{len(distances)}"
                    )
                
                if len(points) == 0:
                    rospy.logwarn_throttle(2.0, f"No valid points! Raw: {raw_count}, Min: {raw_min:.2f}")
                    return
                
                # 计算每个点的水平角度和垂直角度
                # AirSim 和 Isaac Lab 都使用 FLU 坐标系 (X=前, Y=左)
                # arctan2(y, x): 0° = 前, 90° = 左, -90° = 右
                horizontal_angles = np.arctan2(points[:, 1], points[:, 0])
                vertical_angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))
                
                # 将点云转换为固定格式的距离数组 (与 Isaac Lab LidarPatternCfg 完全对齐)
                # 水平: -180° 到 180°, 分辨率 10° → 36, 360° FOV 排除最后点 → 35 条射线
                #       射线角度: np.linspace(-180, 180, 36, endpoint=False) = [-180°, -170°, ..., 170°]
                # 垂直: -7° 到 52°, 8 个通道
                #       通道角度: np.linspace(-7, 52, 8) = [-7°, 1.4°, 9.9°, 18.3°, 26.7°, 35.1°, 43.6°, 52°]
                
                h_bins = self.lidar_horizontal_points  # 35 (Isaac Lab: 360°/10°=36, 但排除最后点=35)
                v_bins = self.lidar_channels  # 8
                
                # 初始化距离数组 (最大范围)
                lidar_array = np.ones((v_bins, h_bins)) * self.lidar_range
                
                # 将点分配到对应的 bin
                # Isaac Lab: linspace(-180°, 180°, 36)[:-1] = 35 条射线 (360° FOV 排除最后点)
                # 射线角度: [-180, -170, -160, ..., 160, 170] (每 10° 一条)
                # 索引 0 对应 -180°, 索引 17-18 对应 0° 附近, 索引 34 对应 170°
                h_angles_deg = np.degrees(horizontal_angles)  # 转为度
                # 映射: angle_deg → index, 其中 -180° → 0, 170° → 34
                # 分辨率 = 360° / 36 = 10° (linspace 生成 36 个点再去掉最后一个)
                h_indices = np.round((h_angles_deg + 180) / 10.0).astype(int)
                h_indices = np.clip(h_indices, 0, h_bins - 1)
                
                # 垂直角度: 与 Isaac Lab linspace(-7, 52, 8) 对齐
                # 通道角度: [-7, 1.43, 9.86, 18.29, 26.71, 35.14, 43.57, 52] (间隔 8.43°)
                # 索引 0 对应 -7°, 索引 7 对应 52°
                v_angles_deg = np.degrees(vertical_angles)
                v_step = (52.0 - (-7.0)) / (v_bins - 1)  # = 59/7 ≈ 8.43°
                v_indices = np.round((v_angles_deg - (-7.0)) / v_step).astype(int)
                v_indices = np.clip(v_indices, 0, v_bins - 1)
                
                # 填充最近距离
                front_bin_points = 0  # 统计落入前方 bin 的点数
                for i in range(len(distances)):
                    vi, hi = v_indices[i], h_indices[i]
                    # 前方 bin: h_indices 17-18 对应 h≈0° (正前方)
                    if 16 <= hi <= 19:
                        front_bin_points += 1
                    if distances[i] < lidar_array[vi, hi]:
                        lidar_array[vi, hi] = distances[i]
                
                # 诊断: 前方 bin 落入多少点
                front_min = np.min(lidar_array[:, 16:20]) if lidar_array[:, 16:20].size > 0 else 5.0
                rospy.loginfo_throttle(
                    2.0,
                    f"[Lidar Bin] 前方bin(h=16-19)点数={front_bin_points}, 前方最近={front_min:.2f}m, "
                    f"总有效点={len(distances)}"
                )
                
                # 展平为一维数组 (与 Isaac Lab 格式一致)
                lidar_flat = lidar_array.flatten()
                
                # 裁剪到最大范围 (不归一化，保持原始距离值)
                self.lidar_data = np.clip(lidar_flat, 0, self.lidar_range)
                
                self.data_received['lidar'] = True
                
                # 重新发布原始点云 (用于 RViz 可视化)
                # AirSim VehicleInertialFrame 的点云已经是世界坐标系
                # 直接发布到 world frame，避免 RViz 再次旋转
                try:
                    from std_msgs.msg import Header
                    
                    if len(points) > 0:
                        header = Header()
                        header.stamp = rospy.Time.now()
                        # 使用原始 frame，让 TF 处理
                        header.frame_id = f"uav{self.uav_id}/lidar_link"
                        
                        republish_msg = pc2.create_cloud_xyz32(header, points.tolist())
                        self.lidar_republish_pub.publish(republish_msg)
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"Failed to republish lidar: {e}")
                
            except Exception as e:
                rospy.logerr_throttle(5.0, f"Error processing point cloud: {e}")
    
    def lidar_callback_2d(self, msg):
        """2D 激光雷达回调 (LaserScan) - 备用
        
        将 2D 雷达数据扩展到 3D 格式 (复制到所有垂直通道)
        """
        with self.lock:
            # 将雷达数据转换为数组
            ranges = np.array(msg.ranges)
            
            # 处理无效值
            ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
            ranges = np.where(np.isnan(ranges), msg.range_max, ranges)
            
            # 重采样到水平点数
            if len(ranges) != self.lidar_horizontal_points:
                indices = np.linspace(0, len(ranges) - 1, self.lidar_horizontal_points, dtype=int)
                ranges = ranges[indices]
            
            # 扩展到所有垂直通道 (将 2D 雷达视为中间高度)
            lidar_array = np.tile(ranges, (self.lidar_channels, 1))
            lidar_flat = lidar_array.flatten()
            
            # 裁剪到最大范围 (不归一化，保持原始距离值)
            self.lidar_data = np.clip(lidar_flat, 0, self.lidar_range)
            
            self.data_received['lidar'] = True
    
    def target_callback(self, msg):
        """目标位置回调"""
        with self.lock:
            self.target_x = msg.pose.position.x
            self.target_y = msg.pose.position.y
            self.target_z = msg.pose.position.z
            rospy.loginfo(f"Target updated: ({self.target_x}, {self.target_y}, {self.target_z})")
    
    def build_observation(self):
        """
        构建观测向量 - 与 Isaac Lab PolicyCfgUAV 完全对齐!
        
        顺序与 velocity_env_cfg.py PolicyCfgUAV 一致:
        - base_lin_vel: 3维 (机体系线速度)
        - base_ang_vel: 3维 (机体系角速度)
        - projected_gravity: 3维 (姿态向量)
        - base_height: 1维 (高度)
        - pose_command: 3维 (机体坐标系相对目标位置)
        - actions: 3维 (上一步动作 [vx, vy, vz])
        - lidar_scan: 280维 (8 channels × 35 horizontal)
        
        总维度: 3 + 3 + 3 + 1 + 3 + 3 + 280 = 296
        """
        with self.lock:
            # 获取四元数 (qx, qy, qz, qw)
            quat = self.orientation.copy()
            
            # 1. base_lin_vel: heading frame 线速度 (3维)
            # Isaac Lab 使用 heading frame: 只用 yaw 旋转，Z 与重力对齐
            # 参考: mdp.base_lin_vel 使用 yaw_quat 而不是完整四元数
            vel_world = self.velocity.copy()
            vel_heading = self._world_to_body_yaw_only(vel_world, quat)
            base_lin_vel = np.array([
                vel_heading[0],   # X: 前 (heading frame)
                vel_heading[1],   # Y: 左 (heading frame)
                vel_heading[2]    # Z: 上 (与世界Z一致)
            ])
            
            # 2. base_ang_vel: 机体坐标系角速度 (3维)
            # Mavros 和 Isaac Lab 都是 FLU，不需要取反
            ang_vel_airsim = self.angular_vel.copy()
            base_ang_vel = np.array([
                ang_vel_airsim[0],    # roll rate
                ang_vel_airsim[1],    # pitch rate (FLU)
                ang_vel_airsim[2]     # yaw rate
            ])
            
            # 3. projected_gravity: 重力向量在机体坐标系的投影 (3维)
            # 与 Isaac Lab mdp.projected_gravity 一致
            # 世界坐标系重力向量 [0, 0, -1]，转换到机体坐标系
            projected_gravity = self._compute_projected_gravity(quat)
            
            # 4. base_height: 当前高度 (1维)
            # AirSim ROS wrapper 已转为 ENU（Z向上为正），position[2] 即高度
            raw_height = self.position[2]
            
            # 高度观测放大: z > 0.5m 时，放大为 0.5 + (z-0.5)*8，让策略以为自己很高从而降高
            if raw_height > 0.5:
                obs_height = 0.5 + (raw_height - 0.5) * 8.0
            else:
                obs_height = raw_height
            base_height = np.array([obs_height])
            
            # vz 压缩: z > 0.5m 时，vz 也 /8
            if raw_height > 0.5:
                base_lin_vel[2] = base_lin_vel[2] / 8.0
            
            # 5. pose_command: 目标位置相对坐标 (机体坐标系)
            # 直接在 AirSim 体系下计算，最后转换轴定义
            
            # 世界坐标系差异 (ENU: X=东, Y=北, Z=上)
            dx_world_raw = self.target_x - self.position[0]
            dy_world_raw = self.target_y - self.position[1]
            dz_world = self.target_z - self.position[2]
            
            # 应用世界坐标系校正 (补偿 Prometheus 和 AirSim 的 yaw 偏移)
            if hasattr(self, 'world_yaw_offset') and self.world_yaw_offset != 0:
                offset_rad = np.radians(self.world_yaw_offset)
                cos_off = np.cos(offset_rad)
                sin_off = np.sin(offset_rad)
                dx_world = cos_off * dx_world_raw - sin_off * dy_world_raw
                dy_world = sin_off * dx_world_raw + cos_off * dy_world_raw
            else:
                dx_world = dx_world_raw
                dy_world = dy_world_raw
            
            diff_world = np.array([dx_world, dy_world, dz_world])
            
            # 旋转到机体坐标系 (只使用 yaw，与 Isaac Lab pose_command 一致)
            diff_body_airsim = self._world_to_body_yaw_only(diff_world, quat)
            
            # Mavros 和 Isaac Lab 都是 FLU，不需要取反
            pose_command = np.array([
                diff_body_airsim[0],   # X: 前
                diff_body_airsim[1],   # Y: 左 (FLU)
                diff_body_airsim[2]    # Z: 上
            ])
            
            # 6. 上一步动作 (3维: [vx, vy, vz])
            actions = self.last_action.copy()
            
            # 7. 激光雷达扫描 (280维)
            # Isaac Lab 训练时用的是原始距离值 (米)，clip 到 [0, lidar_range]
            
            # 调试选项: 禁用雷达 (全部设为 5m，无障碍)
            if self.disable_lidar:
                lidar_scan = np.ones(self.lidar_num_rays, dtype=np.float32) * self.lidar_range
                if self.debug_counter % 100 == 0:
                    rospy.logwarn_throttle(10.0, "[DEBUG] 雷达禁用模式: 所有距离 = 5m")
            else:
                lidar_raw = self.lidar_data.copy()
                
                # 检查是否收到有效雷达数据
                if not self.data_received.get('lidar', False) or np.all(lidar_raw == 0):
                    # 没有收到雷达数据，使用最大范围 (表示无障碍)
                    lidar_scan = np.ones(self.lidar_num_rays, dtype=np.float32) * self.lidar_range
                    rospy.logwarn_throttle(5.0, "[WARN] No lidar data received! Using max range.")
                else:
                    # 原始距离值，clip 到 [0, lidar_range]
                    lidar_scan = np.clip(lidar_raw, 0.0, self.lidar_range)
            
            # 调试: 打印详细的坐标转换信息
            if self.debug_counter % 20 == 0:
                rospy.loginfo(f"[DEBUG] World Diff: [{dx_world:.2f}, {dy_world:.2f}]")
                rospy.loginfo(f"[DEBUG] Body Diff (AirSim): [{diff_body_airsim[0]:.2f}, {diff_body_airsim[1]:.2f}]")
                rospy.loginfo(f"[DEBUG] Pose Command (Isaac): [{pose_command[0]:.2f}, {pose_command[1]:.2f}]")
                rospy.loginfo(f"[DEBUG] Action: vx={actions[0]:.2f}, vy={actions[1]:.2f}, vz={actions[2]:.2f}")
                
                # 检查雷达数据 (原始距离值，clip 后)
                min_dist = np.min(lidar_scan)
                mean_dist = np.mean(lidar_scan)
                max_dist = np.max(lidar_scan)
                valid_count = np.sum(lidar_scan < self.lidar_range)
                rospy.loginfo(f"[DEBUG] Lidar: min={min_dist:.2f}m, mean={mean_dist:.2f}m, max={max_dist:.2f}m, valid={valid_count}/280 (clip to {self.lidar_range}m)")
                rospy.loginfo(f"[DEBUG] 雷达数据接收状态: {self.data_received.get('lidar', False)}")
                
                # 打印各方向距离 (考虑雷达15°下倾)
                # 雷达坐标系: 通道0=-7°(实际-22°), 通道2=+10°(实际-5°), 通道3=+18°(实际+3°)
                lidar_2d = lidar_scan.reshape(self.lidar_channels, self.lidar_horizontal_points)
                
                # 水平方向索引: 18=正前方(0°), 9=左侧(-90°), 27=右侧(+90°), 0=正后方(-180°)
                h_front = 18  # 正前方 0°
                
                # 各垂直通道在正前方的距离
                front_per_channel = lidar_2d[:, h_front]
                rospy.loginfo(f"[DEBUG] 前方距离(各通道): ch0(-22°)={front_per_channel[0]:.2f}, ch2(-5°)={front_per_channel[2]:.2f}, ch3(+3°)={front_per_channel[3]:.2f}")
                
                # 真正的机体水平前方 (通道2-3, 水平18)
                true_front = min(front_per_channel[2], front_per_channel[3])
                # 前下方 (通道0-1)
                front_down = min(front_per_channel[0], front_per_channel[1])
                rospy.loginfo(f"[DEBUG] 机体水平前方={true_front:.2f}m, 前下方={front_down:.2f}m")
                
                # 打印最近的几条射线，确认障碍物方向是否在“前方”角度
                flat = lidar_scan.flatten()
                topk = min(5, flat.size)
                nearest_idx = np.argsort(flat)[:topk]
                debug_items = []
                for idx in nearest_idx:
                    vi = idx // self.lidar_horizontal_points
                    hi = idx % self.lidar_horizontal_points
                    h_angle = -180.0 + hi * 10.0
                    v_angle = -7.0 + vi * (59.0 / 7.0)
                    debug_items.append(f"{flat[idx]:.2f}m@h{h_angle:.0f}°/v{v_angle:.0f}°")
                rospy.loginfo(f"[DEBUG] 最近点(top{topk}): " + ", ".join(debug_items))
            
            # 调试信息 - 检查坐标转换
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
            
            if self.debug_counter % 10 == 0:  # 每 1 秒打印一次 (10Hz)
                # 提取 yaw 角度
                qx, qy, qz, qw = quat
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw_deg = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
                
                rospy.loginfo("="*50)
                rospy.loginfo(f"[DEBUG] 观测维度: {3 + 3 + 3 + 1 + 3 + len(actions) + len(lidar_scan)} (expected: 296)")
                rospy.loginfo(f"[DEBUG] 无人机位置(AirSim): [{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}]")
                rospy.loginfo(f"[DEBUG] 无人机 yaw: {yaw_deg:.1f}°")
                rospy.loginfo(f"[DEBUG] base_lin_vel(机体系): [{base_lin_vel[0]:.2f}, {base_lin_vel[1]:.2f}, {base_lin_vel[2]:.2f}]")
                rospy.loginfo(f"[DEBUG] base_ang_vel(机体系): [{base_ang_vel[0]:.2f}, {base_ang_vel[1]:.2f}, {base_ang_vel[2]:.2f}]")
                rospy.loginfo(f"[DEBUG] projected_gravity: [{projected_gravity[0]:.3f}, {projected_gravity[1]:.3f}, {projected_gravity[2]:.3f}]")
                rospy.loginfo(f"[DEBUG] base_height: {base_height[0]:.2f}m")
                rospy.loginfo(f"[DEBUG] 目标位置(AirSim): [{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}]")
                rospy.loginfo(f"[DEBUG] pose_command(机体系): [{pose_command[0]:.2f}, {pose_command[1]:.2f}, {pose_command[2]:.2f}]")
                rospy.loginfo(f"[DEBUG] last_action: [{actions[0]:.3f}, {actions[1]:.3f}, {actions[2]:.3f}]")
            
            # 保存中间变量 (供日志使用，无需额外加锁，主循环单线程调用)
            self._log_base_lin_vel = base_lin_vel.copy()
            self._log_base_ang_vel = base_ang_vel.copy()
            self._log_projected_gravity = projected_gravity.copy()
            self._log_base_height = float(base_height[0])
            self._log_pose_command = pose_command.copy()
            self._log_lidar_scan = lidar_scan.copy()
            
            # 拼接观测 - 顺序与 Isaac Lab PolicyCfgUAV 完全一致!
            obs = np.concatenate([
                base_lin_vel,       # 3 (机体系线速度)
                base_ang_vel,       # 3 (机体系角速度)
                projected_gravity,  # 3 (姿态)
                base_height,        # 1 (高度)
                pose_command,       # 3 (目标位置, 机体坐标系)
                actions,            # 3 ([vx, vy, vz])
                lidar_scan          # 280
            ])
            
            return obs.astype(np.float32)
    
    def _compute_projected_gravity(self, quat):
        """
        计算重力向量在机体坐标系的投影
        与 Isaac Lab mdp.projected_gravity 一致
        
        世界坐标系重力向量: [0, 0, -1]
        通过四元数旋转到机体坐标系
        """
        qx, qy, qz, qw = quat
        
        # 四元数旋转公式: v' = q^(-1) * v * q
        # 对于单位四元数，q^(-1) = [qx, qy, qz, -qw] (共轭)
        # 世界重力向量 g_world = [0, 0, -1]
        
        # 使用旋转矩阵更清晰:
        # R = [[1-2(qy²+qz²),   2(qxqy-qzqw),   2(qxqz+qyqw)],
        #      [2(qxqy+qzqw),   1-2(qx²+qz²),   2(qyqz-qxqw)],
        #      [2(qxqz-qyqw),   2(qyqz+qxqw),   1-2(qx²+qy²)]]
        # g_body = R^T * g_world = R^T * [0, 0, -1]^T
        
        # R^T 的第三列 (取负)
        projected_gravity = np.array([
            -2 * (qx * qz - qy * qw),
            -2 * (qy * qz + qx * qw),
            -(1 - 2 * (qx * qx + qy * qy))
        ])
        
        return projected_gravity
    
    def _rpy_to_rot_matrix(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """将 RPY (度) 转换为旋转矩阵 (Rz * Ry * Rx)."""
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        R_x = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr, cr]])
        R_y = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0],
                        [sy, cy, 0],
                        [0, 0, 1]])
        return R_z @ R_y @ R_x
    
    def _world_to_body_yaw_only(self, vec_world, quat):
        """将向量从世界坐标系转换到机体坐标系 (只使用 yaw 旋转)
        
        用于 pose_command: 匹配 Isaac Lab 的 yaw_quat 处理
        """
        # 提取 yaw 角度
        qx, qy, qz, qw = quat
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # 使用 yaw 旋转矩阵的逆 (只旋转 xy 平面)
        cos_yaw = np.cos(-yaw)  # 逆旋转
        sin_yaw = np.sin(-yaw)
        
        vec_body = np.zeros(3)
        vec_body[0] = cos_yaw * vec_world[0] - sin_yaw * vec_world[1]
        vec_body[1] = sin_yaw * vec_world[0] + cos_yaw * vec_world[1]
        vec_body[2] = vec_world[2]  # z 不变
        
        return vec_body
    
    def _world_to_body(self, vec_world, quat):
        """将向量从世界坐标系转换到机体坐标系 (使用完整四元数)
        
        用于 base_lin_vel: 完整的刚体旋转
        """
        qx, qy, qz, qw = quat
        
        # 四元数旋转矩阵 (逆旋转: q^-1 v q = R^T v)
        # R^T 的公式:
        R00 = 1 - 2*(qy*qy + qz*qz)
        R01 = 2*(qx*qy + qz*qw)
        R02 = 2*(qx*qz - qy*qw)
        R10 = 2*(qx*qy - qz*qw)
        R11 = 1 - 2*(qx*qx + qz*qz)
        R12 = 2*(qy*qz + qx*qw)
        R20 = 2*(qx*qz + qy*qw)
        R21 = 2*(qy*qz - qx*qw)
        R22 = 1 - 2*(qx*qx + qy*qy)
        
        vec_body = np.array([
            R00 * vec_world[0] + R01 * vec_world[1] + R02 * vec_world[2],
            R10 * vec_world[0] + R11 * vec_world[1] + R12 * vec_world[2],
            R20 * vec_world[0] + R21 * vec_world[1] + R22 * vec_world[2]
        ])
        
        return vec_body
    
    def _body_to_world(self, vec_body, quat):
        """将向量从机体坐标系转换到世界坐标系"""
        qx, qy, qz, qw = quat
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        vec_world = np.zeros(3)
        vec_world[0] = cos_yaw * vec_body[0] - sin_yaw * vec_body[1]
        vec_world[1] = sin_yaw * vec_body[0] + cos_yaw * vec_body[1]
        vec_world[2] = vec_body[2]
        
        return vec_world
    
    def run_policy(self, obs):
        """运行策略推理"""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy(obs_tensor)
            return action.cpu().numpy().squeeze()
    
    def _compute_barrier_and_h(self, lidar_data):
        """
        计算 barrier 函数和安全裕度 h
        
        支持三种 barrier 类型 (由 cbf_barrier 参数控制):
          - log:        B = -log(d + eps)           陡峭，近处梯度爆炸
          - softplus:   B = log(1+exp(α(d_s-d)))/α  在 safe_dist 附近平滑过渡 (推荐)
          - reciprocal: B = 1/(d + eps)              比 log 更平滑的衰减
        
        Args:
            lidar_data: 归一化雷达距离数据 [0, 1]
        
        Returns:
            total_barrier: barrier 函数总值 (标量)
            total_h: 安全裕度总值 (标量)
            barrier: 每条射线的 barrier 值 (数组)
        """
        epsilon = 0.05
        dist = np.clip(lidar_data, 1e-6, 1.0)
        
        if self.cbf_barrier == 'softplus':
            # Softplus barrier: B = log(1 + exp(alpha * (d_safe - d))) / alpha
            # d_safe 归一化: cbf_safe_dist / lidar_range
            d_safe = self.cbf_safe_dist / self.lidar_range
            alpha = 10.0  # 控制过渡带宽度
            exponent = alpha * (d_safe - dist)
            # 数值稳定: 当 exponent 很大时直接用线性近似
            barrier = np.where(
                exponent > 20,
                exponent / alpha,  # ≈ d_safe - dist (线性)
                np.log(1 + np.exp(np.clip(exponent, -50, 20))) / alpha
            )
        elif self.cbf_barrier == 'reciprocal':
            # Reciprocal barrier: B = 1/(d + eps) - 1/(1 + eps)
            barrier = 1.0 / (dist + epsilon) - 1.0 / (1.0 + epsilon)
            barrier = np.clip(barrier, 0, None)  # dist > 1 时 B=0
        else:
            # Log barrier (默认): B = -log(d + eps) - offset
            barrier_raw = -np.log(dist + epsilon)
            offset = -np.log(1.0 + epsilon)
            barrier = barrier_raw - offset
        
        total_barrier = np.sum(barrier)
        total_h = np.sum(dist)
        
        # Clip 防止数值爆炸
        total_barrier = np.clip(total_barrier, -1e6, 1e4)
        
        return total_barrier, total_h, barrier
    
    def apply_cbf_safety(self, vx, vy, vz):
        """
        应用 Control Barrier Function (CBF) 安全过滤
        
        使用与 Isaac Lab 训练一致的实现:
        参考: https://github.com/zhangchangxina/Naci_isaaclab/blob/main/scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/cbf.py
        
        注意: CBF 需要动力学模型才能工作
        - 如果没有动力学模型: 不使用 CBF
        - 如果有动力学模型且 use_cbf=true: 使用 Model-Based CBF
        
        同时保存 CBF 中间数据到 self._cbf_log 供主循环发布
        """
        vx_in, vy_in, vz_in = vx, vy, vz
        
        # 默认 CBF 日志: 未激活
        self._cbf_log = {
            'barrier': 0.0, 'violation': 0.0,
            'delta_vx': 0.0, 'delta_vy': 0.0, 'delta_vz': 0.0,
            'active': 0.0, 'min_barrier_dist': 5.0,
        }
        
        # CBF 需要动力学模型
        if self.dynamics_model is None:
            if self.cbf_test_mode:
                rospy.logwarn_throttle(2.0, "[CBF-TEST] 无动力学模型，CBF 无法工作! 请设置 DYN_CHECKPOINT")
            return vx, vy, vz
        
        # 根据参数决定是否使用 CBF
        if not self.use_cbf:
            return vx, vy, vz
        
        with self.lock:
            lidar = self.lidar_data.copy()  # meters [0, lidar_range]
        
        # ---- BUG-FIX: lidar_data 是米 [0, lidar_range], barrier 需要归一化 [0, 1] ----
        lidar_norm = lidar / self.lidar_range  # [0, 1]
        
        # 计算当前 barrier (用于日志) — 使用归一化值
        b_curr, h_curr, _ = self._compute_barrier_and_h(lidar_norm)
        min_lidar_dist = float(np.min(lidar))  # 已经是米，不需要再乘 lidar_range
        
        # 使用 Model-Based CBF (根据求解器选择)
        # 注意: 传入原始米值，子函数内部会各自归一化
        if self.cbf_solver == 'slsqp':
            cbf_vx, cbf_vy, cbf_vz = self._apply_mbcbf_slsqp(vx, vy, vz, lidar)
        else:
            cbf_vx, cbf_vy, cbf_vz = self._apply_mbcbf(vx, vy, vz, lidar)
        
        # ---- 硬排斥力: 不论速度方向，距离 < repulsion_dist 就主动推开 ----
        if self.cbf_repulsion_gain > 0 and min_lidar_dist < self.cbf_repulsion_dist:
            with self.lock:
                lidar_for_repulsion = self.lidar_data.copy()
            
            # 找最危险的方向 (取每个水平方向所有垂直通道的最小值)
            dist_2d = lidar_for_repulsion.reshape(self.lidar_channels, self.lidar_horizontal_points)
            min_dist_per_angle = np.min(dist_2d, axis=0)
            danger_idx = np.argmin(min_dist_per_angle)
            danger_dist = min_dist_per_angle[danger_idx]
            
            if danger_dist < self.cbf_repulsion_dist:
                # 障碍物方向角 (与 Isaac Lab lidar 角度一致)
                danger_angle = np.radians(-180 + danger_idx * 10.0)
                obs_dir_x = np.cos(danger_angle)
                obs_dir_y = np.sin(danger_angle)
                
                # 排斥力: 距离越近越大
                repulsion = (self.cbf_repulsion_dist - danger_dist) * self.cbf_repulsion_gain
                cbf_vx -= obs_dir_x * repulsion
                cbf_vy -= obs_dir_y * repulsion
                
                rospy.logwarn_throttle(1.0,
                    f"[CBF-REPULSION] dist={danger_dist:.2f}m < {self.cbf_repulsion_dist}m, "
                    f"repulsion={repulsion:.2f}, angle={np.degrees(danger_angle):.0f}°")
        
        # 计算 CBF 修正量
        dvx = cbf_vx - vx_in
        dvy = cbf_vy - vy_in
        dvz = cbf_vz - vz_in
        delta_norm = np.sqrt(dvx**2 + dvy**2 + dvz**2)
        cbf_active = 1.0 if delta_norm > 0.01 else 0.0
        
        # 保存 CBF 日志数据
        self._cbf_log = {
            'barrier': float(b_curr),
            'violation': float(getattr(self, '_last_cbf_violation', 0.0)),
            'delta_vx': float(dvx),
            'delta_vy': float(dvy),
            'delta_vz': float(dvz),
            'active': cbf_active,
            'min_barrier_dist': float(min_lidar_dist),
        }
        
        # CBF 测试模式: 详细输出对比日志
        if self.cbf_test_mode:
            if cbf_active > 0.5:
                rospy.logwarn_throttle(1.0,
                    f"[CBF-TEST] CBF 已介入! "
                    f"输入=({vx_in:.3f},{vy_in:.3f},{vz_in:.3f}) -> "
                    f"输出=({cbf_vx:.3f},{cbf_vy:.3f},{cbf_vz:.3f}) "
                    f"delta={delta_norm:.3f} barrier={b_curr:.2f} min_obs={min_lidar_dist:.2f}m")
            else:
                rospy.loginfo_throttle(2.0,
                    f"[CBF-TEST] CBF 未修改动作 "
                    f"({vx_in:.3f},{vy_in:.3f},{vz_in:.3f}) "
                    f"barrier={b_curr:.2f} min_obs={min_lidar_dist:.2f}m (安全)")
        
        return cbf_vx, cbf_vy, cbf_vz
    
    def _apply_mbcbf(self, vx, vy, vz, lidar):
        """
        Model-Based CBF: 使用动力学模型预测下一状态
        
        类似训练代码的 solve_cbf_qp 实现
        
        Args:
            lidar: 米制原始值 [0, lidar_range]，函数内部归一化后用于 barrier
        """
        try:
            # 直接复用 build_observation() — 永远与 policy 输入一致 (296维)
            obs = self.build_observation()
            
            # 转换为 tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            # 动作是 3 维: [vx, vy, vz] (与 last_action 和 act_dim=3 一致)
            action = torch.tensor([vx / self.scale_hor, vy / self.scale_hor, vz / self.scale_z], 
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # BUG-FIX: 归一化 lidar 用于 barrier 计算
            lidar_norm = lidar / self.lidar_range  # [0, 1]
            b_curr, h_curr, _ = self._compute_barrier_and_h(lidar_norm)
            
            # SQP 迭代 (简化版，3 次迭代)
            u = action.clone()
            u.requires_grad = True
            
            for sqp_iter in range(3):
                # 预测下一状态
                with torch.enable_grad():
                    pred_next_state, _, _ = self.dynamics_model(obs_tensor, u)
                
                # 提取预测的 lidar (最后 280 维)
                # 模型输出与输入同尺度 (米), clip 到 [0, lidar_range] 再归一化
                pred_lidar = pred_next_state[0, -self.lidar_num_rays:].detach().cpu().numpy()
                pred_lidar = np.clip(pred_lidar, 0, self.lidar_range)
                
                # 计算预测的 barrier (归一化到 [0, 1])
                b_next, _, _ = self._compute_barrier_and_h(pred_lidar / self.lidar_range)
                
                # 约束违反: B(x') - B(x) - γ * h(x) > 0
                violation = b_next - b_curr - self.cbf_gamma * h_curr
                self._last_cbf_violation = float(violation)
                
                if violation <= 1e-4:
                    # 约束满足，退出
                    break
                
                # 计算梯度 (数值方法) - 对3维速度 [vx, vy, vz] 计算
                eps = 0.01
                grad = np.zeros(3)  # 3维动作
                u_np = u.detach().cpu().numpy().squeeze()
                
                for i in range(3):  # vx, vy, vz
                    u_plus = u_np.copy()
                    u_plus[i] += eps
                    u_plus_tensor = torch.tensor(u_plus, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        pred_plus, _, _ = self.dynamics_model(obs_tensor, u_plus_tensor)
                    pred_lidar_plus = pred_plus[0, -self.lidar_num_rays:].cpu().numpy()
                    pred_lidar_plus = np.clip(pred_lidar_plus, 0, self.lidar_range)
                    b_plus, _, _ = self._compute_barrier_and_h(pred_lidar_plus / self.lidar_range)
                    
                    grad[i] = (b_plus - b_next) / eps
                
                # QP 更新: u = u - λ * grad, where λ = violation / ||grad||²
                grad_norm_sq = np.sum(grad ** 2) + 1e-6
                lam = max(0, violation / grad_norm_sq)
                
                correction = -lam * grad
                u_np = u_np + correction
                u = torch.tensor(u_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                u.requires_grad = True
                
                if sqp_iter == 0 and violation > 0:
                    rospy.loginfo_throttle(2.0, 
                        f"MB-CBF: violation={violation:.3f}, correction={np.linalg.norm(correction):.3f}")
            
            # 转换回速度
            u_final = u.detach().cpu().numpy().squeeze()
            cbf_vx = u_final[0] * self.scale_hor
            cbf_vy = u_final[1] * self.scale_hor
            cbf_vz = u_final[2] * self.scale_z
            
            # 速度限制 (非对称: 上升/下降分开)
            cbf_vx = np.clip(cbf_vx, -self.max_vel_hor, self.max_vel_hor)
            cbf_vy = np.clip(cbf_vy, -self.max_vel_hor, self.max_vel_hor)
            cbf_vz = np.clip(cbf_vz, -self.max_vel_down, self.max_vel_up)
            
            return cbf_vx, cbf_vy, cbf_vz
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"MB-CBF failed: {e}, falling back to geometric CBF")
            # geometric CBF 期望归一化 [0, 1] lidar
            return self._apply_geometric_cbf(vx, vy, vz, lidar / self.lidar_range)
    
    def _apply_mbcbf_slsqp(self, vx, vy, vz, lidar):
        """
        Model-Based CBF (SLSQP 严格求解版)
        
        求解 QP 问题:
            min  ||u - u_nom||²
            s.t. B(f(x, u)) - B(x) - gamma * h(x) <= 0   (CBF 约束)
                 -1 <= u_i <= 1                            (动作范围约束)
        
        使用 scipy.optimize.minimize(method='SLSQP') 保证约束一定被满足
        
        Args:
            lidar: 米制原始值 [0, lidar_range]，函数内部归一化后用于 barrier
        """
        from scipy.optimize import minimize
        
        try:
            # 直接复用 build_observation() — 永远与 policy 输入一致 (296维)
            obs = self.build_observation()
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # 名义动作 (归一化)
            u_nom = np.array([vx / self.scale_hor, vy / self.scale_hor, vz / self.scale_z])
            
            # BUG-FIX: 归一化 lidar 用于 barrier 计算
            lidar_norm = lidar / self.lidar_range  # [0, 1]
            b_curr, h_curr, _ = self._compute_barrier_and_h(lidar_norm)
            
            # 如果当前已经安全且没有 violation, 直接返回
            # 先用名义动作检查
            u_tensor = torch.tensor(u_nom.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                pred_state, _, _ = self.dynamics_model(obs_tensor, u_tensor)
            pred_lidar = pred_state[0, -self.lidar_num_rays:].cpu().numpy()
            pred_lidar = np.clip(pred_lidar, 0, self.lidar_range)
            b_next, _, _ = self._compute_barrier_and_h(pred_lidar / self.lidar_range)
            violation = b_next - b_curr - self.cbf_gamma * h_curr
            self._last_cbf_violation = float(violation)
            
            if violation <= 1e-4:
                # 名义动作已满足 CBF 约束，无需优化
                return vx, vy, vz
            
            # --- SLSQP 求解 ---
            # 缓存用于避免重复计算
            _eval_cache = {}
            
            def _predict_barrier(u3):
                """预测给定动作的 barrier 值 (带缓存)"""
                key = tuple(np.round(u3, 6))
                if key in _eval_cache:
                    return _eval_cache[key]
                u_t = torch.tensor(u3.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    pred, _, _ = self.dynamics_model(obs_tensor, u_t)
                p_lidar = pred[0, -self.lidar_num_rays:].cpu().numpy()
                p_lidar = np.clip(p_lidar, 0, self.lidar_range)
                b, _, _ = self._compute_barrier_and_h(p_lidar / self.lidar_range)
                _eval_cache[key] = b
                return b
            
            def objective(u3):
                """目标: min ||u - u_nom||²"""
                return np.sum((u3 - u_nom) ** 2)
            
            def objective_jac(u3):
                """目标函数梯度"""
                return 2.0 * (u3 - u_nom)
            
            def cbf_constraint(u3):
                """CBF 约束: -(B(f(x,u)) - B(x) - gamma*h(x)) >= 0"""
                b = _predict_barrier(u3)
                return -(b - b_curr - self.cbf_gamma * h_curr)
            
            # 动作范围约束 [-1, 1]
            bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
            
            constraints = [{
                'type': 'ineq',
                'fun': cbf_constraint
            }]
            
            result = minimize(
                objective,
                u_nom,
                jac=objective_jac,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 20, 'ftol': 1e-6, 'disp': False}
            )
            
            u_opt = result.x
            
            # 转换回速度
            cbf_vx = u_opt[0] * self.scale_hor
            cbf_vy = u_opt[1] * self.scale_hor
            cbf_vz = u_opt[2] * self.scale_z
            
            # 速度限制
            cbf_vx = np.clip(cbf_vx, -self.max_vel_hor, self.max_vel_hor)
            cbf_vy = np.clip(cbf_vy, -self.max_vel_hor, self.max_vel_hor)
            cbf_vz = np.clip(cbf_vz, -self.max_vel_down, self.max_vel_up)
            
            # 验证约束是否满足
            final_violation = -cbf_constraint(u_opt)
            if final_violation > 0.01:
                rospy.logwarn_throttle(2.0,
                    f"[CBF-SLSQP] 约束未完全满足! violation={final_violation:.4f}, "
                    f"status={result.message}")
            elif self.cbf_test_mode:
                rospy.loginfo_throttle(2.0,
                    f"[CBF-SLSQP] 求解成功, iterations={result.nit}, "
                    f"constraint_margin={-final_violation:.4f}")
            
            return cbf_vx, cbf_vy, cbf_vz
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"MB-CBF-SLSQP failed: {e}, falling back to gradient method")
            return self._apply_mbcbf(vx, vy, vz, lidar)
    
    def _apply_geometric_cbf(self, vx, vy, vz, lidar):
        """
        几何 CBF: 基于当前雷达距离的简化实现 (当没有动力学模型时使用)
        """
        # 将归一化距离转换为实际距离
        distances = lidar * self.lidar_range  # [0, lidar_range]
        
        if len(distances) != self.lidar_num_rays:
            return vx, vy, vz
        
        dist_2d = distances.reshape(self.lidar_channels, self.lidar_horizontal_points)
        
        # 取每个水平方向的最小距离
        min_dist_per_angle = np.min(dist_2d, axis=0)
        
        # 计算 barrier
        _, h, barrier = self._compute_barrier_and_h(lidar)
        
        # 角度数组
        angles = np.linspace(-np.pi, np.pi, self.lidar_horizontal_points, endpoint=False)
        
        cbf_vx, cbf_vy = vx, vy
        
        # 找到最危险的方向
        barrier_per_angle = -np.log(np.clip(min_dist_per_angle / 5.0, 1e-6, 1.0) + 0.05)
        max_barrier_idx = np.argmax(barrier_per_angle)
        max_barrier = barrier_per_angle[max_barrier_idx]
        min_dist = min_dist_per_angle[max_barrier_idx]
        
        if max_barrier > 0:
            danger_angle = angles[max_barrier_idx]
            obs_dir_x = np.cos(danger_angle)
            obs_dir_y = np.sin(danger_angle)
            
            vel_toward_obs = vx * obs_dir_x + vy * obs_dir_y
            
            if vel_toward_obs > 0:
                dt = 1.0 / self.control_freq if hasattr(self, 'control_freq') else 0.1
                max_safe_vel = self.cbf_gamma * h * (min_dist + 0.05) / (dt * self.lidar_num_rays)
                max_safe_vel = min(max_safe_vel, self.max_vel_hor)
                
                if vel_toward_obs > max_safe_vel:
                    reduction = vel_toward_obs - max_safe_vel
                    cbf_vx -= obs_dir_x * reduction
                    cbf_vy -= obs_dir_y * reduction
                    rospy.loginfo_throttle(2.0, f"Geo-CBF: dist={min_dist:.2f}m, vel_toward reduced")
            
            if min_dist < self.cbf_repulsion_dist and self.cbf_repulsion_gain > 0:
                repulsion = (self.cbf_repulsion_dist - min_dist) * self.cbf_repulsion_gain
                cbf_vx -= obs_dir_x * repulsion
                cbf_vy -= obs_dir_y * repulsion
                rospy.logwarn_throttle(1.0, f"Geo-CBF: Emergency! dist={min_dist:.2f}m repulsion={repulsion:.2f}")
        
        cbf_vx = np.clip(cbf_vx, -self.max_vel_hor * 1.5, self.max_vel_hor * 1.5)
        cbf_vy = np.clip(cbf_vy, -self.max_vel_hor * 1.5, self.max_vel_hor * 1.5)
        
        return cbf_vx, cbf_vy, vz
    
    def publish_command(self, action):
        """
        将策略输出转换为 Prometheus 控制命令 (机体坐标系 + 自动航向)
        
        方案四: 3维动作空间 [vx, vy, vz]，航向自动朝向目标点
        
        使用 Prometheus Move_mode=4 (XYZ_VEL_BODY) + 绝对yaw控制
        
        参考 prometheus_msgs/UAVCommand.msg:
        - Agent_CMD: Move=4
        - Move_mode: XYZ_VEL_BODY=4 (机体坐标系速度控制)
        - velocity_ref[3]: 速度参考值 [m/s] - 机体坐标系
        - Yaw_Rate_Mode: False (使用绝对角度控制)
        - yaw_ref: 目标航向角 [rad] = 目标点方向
        """
        # 缩放动作到机体坐标系速度
        # Isaac Lab 机体坐标系: X=前, Y=左, Z=上 (FLU)
        # Prometheus 机体坐标系: X=前, Y=左, Z=上 (FLU) - 与 Isaac Lab 一致!
        # (参考: uav_controller.cpp rotation_yaw 函数使用标准 FLU→ENU 旋转)
        vx = action[0] * self.scale_hor      # 机体 X (前)
        vy = action[1] * self.scale_hor      # 机体 Y (左, FLU)
        vz = action[2] * self.scale_z        # 机体 Z (上, FLU)
        
        # 航向自动朝向目标点 (方案四: 无 yaw_offset)
        # XYZ_VEL_BODY 模式下 yaw_ref 是相对偏移: yaw_des = yaw_ref + current_yaw
        with self.lock:
            dx = self.target_x - self.position[0]
            dy = self.target_y - self.position[1]
            # 从四元数获取当前 yaw
            qx, qy, qz, qw = self.orientation
            current_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        target_direction_yaw = np.arctan2(dy, dx)  # 目标点方向（世界系）
        
        # 目标1m内冻结yaw，避免近距离方向向量过短导致yaw抖动
        dist_xy_yaw = np.sqrt(dx**2 + dy**2)
        if dist_xy_yaw < 1.0:
            yaw_ref = 0.0  # 相对偏移=0，保持当前航向不变
        else:
            # yaw_ref = 目标yaw - 当前yaw（相对偏移）
            yaw_ref = target_direction_yaw - current_yaw
            # 归一化到 [-π, π]
            yaw_ref = np.arctan2(np.sin(yaw_ref), np.cos(yaw_ref))
        
        # 保存航向信息 (供日志使用)
        self._log_current_yaw_deg = np.degrees(current_yaw)
        self._log_target_yaw_deg = np.degrees(target_direction_yaw)
        
        # 调试: 打印策略原始输出
        if hasattr(self, 'debug_counter') and self.debug_counter % 50 == 0:
            rospy.loginfo(f"[DEBUG] 策略输出(raw): [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
            rospy.loginfo(f"[DEBUG] current_yaw={np.degrees(current_yaw):.1f}°, target_yaw={np.degrees(target_direction_yaw):.1f}°, yaw_ref={np.degrees(yaw_ref):.1f}°")
            rospy.loginfo(f"[DEBUG] 速度命令(机体系): vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
            # 打印 Isaac 动作 (取反前)
            rospy.loginfo(f"[DEBUG] Isaac动作: vx_i={action[0]*self.scale_hor:.2f}, vy_i={action[1]*self.scale_hor:.2f} (vy取反后={vy:.2f})")
        
        # 高度保护机制: 确保飞行高度不低于 min_altitude
        # AirSim ROS wrapper 已转为 ENU（Z向上为正），position[2] 即高度
        with self.lock:
            current_altitude = self.position[2]
        
        self.height_protection = False
        if current_altitude < self.min_altitude:
            # 如果当前高度低于最低高度，强制向上飞
            if vz < 0.5:  # 如果下降或上升速度不够
                vz = 0.5  # 强制向上
                self.height_protection = True
                rospy.logwarn_throttle(2.0, f"Height protection: altitude={current_altitude:.2f}m < {self.min_altitude}m, forcing vz={vz:.2f}")
        elif current_altitude < self.min_altitude + 0.5:
            # 如果接近最低高度，不允许下降
            if vz < 0:
                vz = 0.0
                self.height_protection = True
                rospy.loginfo_throttle(5.0, f"Height protection: near min altitude ({self.min_altitude}m), preventing descent")
        
        # CBF 安全过滤 (基于障碍物距离调整速度)
        vx, vy, vz = self.apply_cbf_safety(vx, vy, vz)
        
        # 速度限制 (clamp to max velocity, 非对称)
        vx = np.clip(vx, -self.max_vel_hor, self.max_vel_hor)
        vy = np.clip(vy, -self.max_vel_hor, self.max_vel_hor)
        vz = np.clip(vz, -self.max_vel_down, self.max_vel_up)
        
        # 加速度限制 (smooth velocity changes, 非对称)
        current_time = rospy.Time.now()
        if self.last_cmd_time is not None:
            dt = (current_time - self.last_cmd_time).to_sec()
            if dt > 0 and dt < 1.0:  # 忽略异常时间间隔
                # 计算最大允许的速度变化
                max_dv_hor = self.max_acc_hor * dt
                max_dv_up = self.max_acc_up * dt
                max_dv_down = self.max_acc_down * dt
                
                # 限制水平速度变化
                dvx = vx - self.last_cmd_vel[0]
                dvy = vy - self.last_cmd_vel[1]
                dvz = vz - self.last_cmd_vel[2]
                
                if abs(dvx) > max_dv_hor:
                    vx = self.last_cmd_vel[0] + np.sign(dvx) * max_dv_hor
                if abs(dvy) > max_dv_hor:
                    vy = self.last_cmd_vel[1] + np.sign(dvy) * max_dv_hor
                # 垂直方向非对称加速度限制
                if dvz > max_dv_up:
                    vz = self.last_cmd_vel[2] + max_dv_up
                elif dvz < -max_dv_down:
                    vz = self.last_cmd_vel[2] - max_dv_down
        
        # 更新上一次命令记录
        self.last_cmd_vel[0] = vx
        self.last_cmd_vel[1] = vy
        self.last_cmd_vel[2] = vz
        self.last_cmd_time = current_time
        
        if PROMETHEUS_MSGS_AVAILABLE:
            cmd = UAVCommand()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"
            
            # 命令ID递增 (重要! Prometheus 需要递增的 Command_ID)
            self.command_id += 1
            cmd.Command_ID = self.command_id
            
            # Agent_CMD: Move=4
            cmd.Agent_CMD = 4  # UAVCommand::Move
            
            # Move_mode: XYZ_VEL_BODY=4 (机体坐标系速度控制)
            # 策略输出: [vx, vy, vz] (3维) - 航向自动朝向目标
            cmd.Move_mode = 4  # UAVCommand::XYZ_VEL_BODY
            
            # 速度参考值 (机体坐标系)
            cmd.velocity_ref[0] = vx
            cmd.velocity_ref[1] = vy
            cmd.velocity_ref[2] = vz
            
            # 航向控制: 自动朝向目标点
            # XYZ_VEL_BODY 模式下 yaw_ref 是相对偏移 (Prometheus 内部: yaw_des = yaw_ref + uav_yaw)
            cmd.Yaw_Rate_Mode = False  # 使用绝对角度控制
            cmd.yaw_ref = yaw_ref      # 相对偏移 = target_yaw - current_yaw
            
            self.cmd_pub.publish(cmd)
        else:
            # 使用 TwistStamped 作为备用
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"
            cmd.twist.linear.x = vx
            cmd.twist.linear.y = vy
            cmd.twist.linear.z = vz
            self.cmd_pub.publish(cmd)
        
        # 更新上一步动作 (归一化值，用于下次观测)
        self.last_action = action.copy()
        
        # 发布调试信息
        action_msg = Float32MultiArray()
        action_msg.data = [vx, vy, vz]
        self.action_pub.publish(action_msg)
        
        # 发布 rviz 可视化
        self.publish_visualization(vx, vy, vz)
    
    def publish_visualization(self, vx, vy, vz):
        """发布 rviz 可视化 marker"""
        from visualization_msgs.msg import Marker
        from nav_msgs.msg import Path
        
        now = rospy.Time.now()
        
        # 1. 目标点 marker (绿色球体)
        target_marker = Marker()
        target_marker.header.stamp = now
        target_marker.header.frame_id = "world"
        target_marker.ns = "target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = self.target_x
        target_marker.pose.position.y = self.target_y
        target_marker.pose.position.z = self.target_z
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.5
        target_marker.scale.y = 0.5
        target_marker.scale.z = 0.5
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 0.8
        target_marker.lifetime = rospy.Duration(0)
        self.target_marker_pub.publish(target_marker)
        
        # 2. 速度命令 marker (红色箭头)
        vel_marker = Marker()
        vel_marker.header.stamp = now
        vel_marker.header.frame_id = "world"
        vel_marker.ns = "velocity"
        vel_marker.id = 0
        vel_marker.type = Marker.ARROW
        vel_marker.action = Marker.ADD
        
        with self.lock:
            pos = self.position.copy()
            quat = self.orientation.copy()
        
        # 把机体系速度转换到世界系 (用于正确显示)
        vel_body = np.array([vx, vy, vz])
        vel_world = self._body_to_world(vel_body, quat)
        
        # 箭头起点 (当前位置)
        vel_marker.points = []
        from geometry_msgs.msg import Point
        start = Point()
        start.x = pos[0]
        start.y = pos[1]
        start.z = pos[2]
        vel_marker.points.append(start)
        
        # 箭头终点 (世界系速度方向)
        end = Point()
        end.x = pos[0] + vel_world[0]
        end.y = pos[1] + vel_world[1]
        end.z = pos[2] + vel_world[2]
        vel_marker.points.append(end)
        
        vel_marker.scale.x = 0.1  # 箭头杆直径
        vel_marker.scale.y = 0.2  # 箭头头直径
        vel_marker.color.r = 1.0
        vel_marker.color.g = 0.0
        vel_marker.color.b = 0.0
        vel_marker.color.a = 1.0
        vel_marker.lifetime = rospy.Duration(0.2)
        self.velocity_marker_pub.publish(vel_marker)
        
        # 3. 飞行路径 (绿色线条)
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "world"
        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        pose.pose.orientation.w = 1.0
        
        self.path_history.append(pose)
        if len(self.path_history) > self.max_path_points:
            self.path_history.pop(0)
        
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "world"
        path_msg.poses = self.path_history
        self.path_pub.publish(path_msg)
        
        # 4. 雷达射线可视化 (每 10 帧更新一次，减少性能开销)
        if hasattr(self, 'viz_counter'):
            self.viz_counter += 1
        else:
            self.viz_counter = 0
        
        if self.viz_counter % 10 == 0:
            self.publish_lidar_visualization(pos, quat, now)
    
    def publish_lidar_visualization(self, pos, quat, now):
        """发布雷达射线可视化"""
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        
        marker_array = MarkerArray()
        
        with self.lock:
            lidar_data = self.lidar_data.copy()
        
        # 雷达参数
        h_bins = self.lidar_horizontal_points  # 35
        v_bins = self.lidar_channels  # 8
        
        # 只显示部分射线 (每隔几条显示一条，避免太密集)
        h_step = 2  # 水平方向每隔 2 条显示
        v_step = 2  # 垂直方向每隔 2 条显示
        
        marker_id = 0
        for vi in range(0, v_bins, v_step):
            for hi in range(0, h_bins, h_step):
                idx = vi * h_bins + hi
                dist = lidar_data[idx]
                
                # 计算射线角度 (与 Isaac Lab 一致)
                h_angle = np.radians(-180 + hi * 10.0)  # 水平角度
                v_angle = np.radians(-7 + vi * (59.0 / 7))  # 垂直角度
                
                # 射线方向 (机体系 FLU)
                ray_x = np.cos(v_angle) * np.cos(h_angle)
                ray_y = np.cos(v_angle) * np.sin(h_angle)
                ray_z = np.sin(v_angle)
                
                # 转换到世界系
                ray_body = np.array([ray_x, ray_y, ray_z])
                ray_world = self._body_to_world(ray_body, quat)
                
                # 创建射线 marker
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = "world"
                marker.ns = "lidar_rays"
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                # 起点 (无人机位置)
                start = Point()
                start.x = pos[0]
                start.y = pos[1]
                start.z = pos[2]
                
                # 终点 (射线命中点)
                end = Point()
                end.x = pos[0] + ray_world[0] * dist
                end.y = pos[1] + ray_world[1] * dist
                end.z = pos[2] + ray_world[2] * dist
                
                marker.points = [start, end]
                marker.scale.x = 0.02  # 线宽
                
                # 颜色: 距离越近越红，越远越绿
                ratio = min(dist / self.lidar_range, 1.0)
                marker.color.r = 1.0 - ratio
                marker.color.g = ratio
                marker.color.b = 0.0
                marker.color.a = 0.6
                marker.lifetime = rospy.Duration(0.5)
                
                marker_array.markers.append(marker)
                marker_id += 1
        
        self.lidar_marker_pub.publish(marker_array)
    
    def _on_shutdown(self):
        """ROS 关闭时清理"""
        rospy.loginfo(f"策略节点已关闭: {self._policy_label}")
    
    def _log_flight_data(self, action, cmd_vel, dist_to_target, min_obstacle_dist):
        """飞行数据记录已移至 flight_logger.py (保留方法签名以保持兼容)"""
        pass
    
    def check_data_ready(self):
        """检查是否收到必要的传感器数据"""
        with self.lock:
            return self.data_received['odom']
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(self.control_freq)
        
        rospy.loginfo("Waiting for sensor data...")
        while not rospy.is_shutdown() and not self.check_data_ready():
            rate.sleep()
        
        rospy.loginfo("Sensor data received. Starting policy execution...")
        
        while not rospy.is_shutdown():
            try:
                # 构建观测
                obs = self.build_observation()
                
                # CBF 测试模式: 使用自定义固定动作，验证 CBF 安全层
                if self.cbf_test_mode:
                    action = np.array([self.cbf_test_action_vx,
                                       self.cbf_test_action_vy,
                                       self.cbf_test_action_vz])
                    self._log_action_raw = action.copy()
                    rospy.loginfo_throttle(2.0,
                        f"[CBF-TEST] 固定动作: vx={action[0]:.2f}, vy={action[1]:.2f}, vz={action[2]:.2f}")
                # 测试模式: 使用固定动作 (验证坐标系)
                elif self.test_mode:
                    # 固定动作: vx=0.5 (往前), vy=0, vz=0
                    action = np.array([0.5, 0.0, 0.0])
                    self._log_action_raw = action.copy()
                    rospy.loginfo_throttle(2.0, f"[TEST] 固定动作: vx=0.5, vy=0, vz=0 → 无人机应该往前飞!")
                else:
                    # 运行策略
                    action_raw = self.run_policy(obs)
                    # 裁剪动作到 [-1, 1]
                    action_raw = np.clip(action_raw, -1.0, 1.0)
                    self._log_action_raw = action_raw.copy()
                    
                    # 应用动作符号修正 (如果策略输出方向相反)
                    action = action_raw.copy()
                    if self.invert_vx:
                        action[0] = -action[0]
                    if self.invert_vy:
                        action[1] = -action[1]
                    
                    # 调试: 显示策略输出和修正后的动作
                    if hasattr(self, 'debug_counter') and self.debug_counter % 20 == 0:
                        rospy.loginfo(f"[DEBUG] 策略输出(原始): [{action_raw[0]:.3f}, {action_raw[1]:.3f}, {action_raw[2]:.3f}]")
                        rospy.loginfo(f"[DEBUG] 动作修正后: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] (invert_vx={self.invert_vx}, invert_vy={self.invert_vy})")
                
                # 目标点附近减速/悬停 (由 target_stop_mode 参数控制)
                dist_xy = np.sqrt(
                    (self.position[0] - self.target_x)**2 +
                    (self.position[1] - self.target_y)**2
                )
                dist_to_target = np.sqrt(dist_xy**2 + (self.position[2] - self.target_z)**2)
                min_obstacle_dist = np.min(self.lidar_data)
                obs_cancel_dist = 3.0  # 障碍物小于此距离时, 取消减速/悬停

                if self.target_stop_mode == 'decel':
                    # 指数减速: XY 1m内指数衰减到0, 先慢后快 (边界处平缓, 目标点附近急减)
                    # 公式: scale = (1 - exp(-alpha*t)) / (1 - exp(-alpha)), t = dist/R
                    decel_radius = 1.0
                    if dist_xy < decel_radius and min_obstacle_dist >= obs_cancel_dist:
                        t = dist_xy / decel_radius  # 归一化距离 [0, 1]
                        alpha = 3.0  # 曲线参数: 越大边界越平缓, 目标附近越陡
                        speed_scale = (1.0 - np.exp(-alpha * t)) / (1.0 - np.exp(-alpha))
                        action = action * speed_scale
                        rospy.loginfo_throttle(2.0, f"[DECEL-EXP] xy={dist_xy:.2f}m, obs={min_obstacle_dist:.2f}m, scale={speed_scale:.2f}")
                    elif min_obstacle_dist < obs_cancel_dist:
                        rospy.loginfo_throttle(2.0, f"[OBS-OVERRIDE] xy={dist_xy:.2f}m, obs={min_obstacle_dist:.2f}m, 取消减速, 全力避障")

                elif self.target_stop_mode == 'hover':
                    # 悬停: XY<1m进入悬停, >1.5m退出(滞回), 障碍物<2m取消悬停
                    if min_obstacle_dist < obs_cancel_dist:
                        self.is_hovering = False
                    elif dist_xy < 1.0:
                        self.is_hovering = True
                    elif dist_xy > 1.5:
                        self.is_hovering = False
                    if self.is_hovering:
                        action = np.array([0.0, 0.0, 0.0])
                        rospy.loginfo_throttle(2.0, f"[HOVER] xy={dist_xy:.2f}m, obs={min_obstacle_dist:.2f}m, 悬停中")
                    elif min_obstacle_dist < obs_cancel_dist:
                        rospy.loginfo_throttle(2.0, f"[OBS-OVERRIDE] xy={dist_xy:.2f}m, obs={min_obstacle_dist:.2f}m, 取消悬停, 全力避障")
                # else: target_stop_mode == 'none', 不做任何处理
                
                # 发布控制命令
                self.publish_command(action)
                
                # 记录飞行数据 (用于绘图)
                self._log_flight_data(
                    action, self.last_cmd_vel,
                    dist_to_target, min_obstacle_dist
                )
                
                # 发布内部数据供 flight_logger 订阅
                from std_msgs.msg import Float32
                ar_msg = Float32MultiArray(data=self._log_action_raw.tolist())
                self.action_raw_pub.publish(ar_msg)
                self.min_obstacle_pub.publish(Float32(data=min_obstacle_dist))
                if hasattr(self, '_log_lidar_scan') and len(self._log_lidar_scan) > 0:
                    ls_msg = Float32MultiArray(data=self._log_lidar_scan.tolist())
                    self.lidar_scan_pub.publish(ls_msg)
                # CBF 数据 (7维)
                if hasattr(self, '_cbf_log'):
                    cbf_msg = Float32MultiArray(data=[
                        self._cbf_log['barrier'],
                        self._cbf_log['violation'],
                        self._cbf_log['delta_vx'],
                        self._cbf_log['delta_vy'],
                        self._cbf_log['delta_vz'],
                        self._cbf_log['active'],
                        self._cbf_log['min_barrier_dist'],
                    ])
                    self.cbf_data_pub.publish(cbf_msg)
                
            except Exception as e:
                rospy.logerr(f"Error in policy execution: {e}")
            
            rate.sleep()


def main():
    try:
        node = IsaacLabPolicyNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node crashed: {e}")
        raise


if __name__ == '__main__':
    main()
