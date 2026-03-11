#!/bin/bash
# 脚本描述: 启动 Isaac Lab 策略节点 + Rviz
# 不启动仿真组件，不会断开现有连接

UAV_ID=1

echo "=========================================="
echo "启动 Isaac Lab 策略 + Rviz"
echo "=========================================="

# 清理旧进程，避免累积
# echo "清理旧进程..."
# pkill -f "isaaclab_policy_node" 2>/dev/null
# pkill -f rviz 2>/dev/null
# sleep 1

source /opt/ros/noetic/setup.bash
source ~/su17_experiment/devel/setup.bash

cd /home/amov/su17_experiment/src/Prometheus/Modules/isaaclabpolicy/

# ============================================================
# 是否启动 airsim_node (获取雷达数据)
# 设为 false 可以测试没有 airsim 时的行为
# ============================================================
ENABLE_AIRSIM=false  # 实机不需要 AirSim

if [ "$ENABLE_AIRSIM" = "true" ]; then
    if ! rosnode list 2>/dev/null | grep -q "airsim"; then
        echo "启动 airsim_node (雷达数据)..."
        roslaunch airsim_ros_pkgs airsim_node.launch output:=log host:=${PX4_SIM_HOST_ADDR:-172.31.80.1} &
        AIRSIM_PID=$!
        sleep 3
    else
        echo "airsim_node 已在运行"
        AIRSIM_PID=""
    fi
else
    echo "[INFO] airsim_node 已禁用 (ENABLE_AIRSIM=false)"
    echo "[INFO] 雷达数据将使用默认最大值 (5m)"
    AIRSIM_PID=""
fi

# 后台启动 rviz
rviz -d /home/amov/su17_experiment/src/Prometheus/Modules/isaaclabpolicy/isaaclab_policy.rviz &
RVIZ_PID=$!

sleep 2

# ============================================================
# 模型配置 - 统一接口
# ============================================================
# 支持格式:
#   本地文件名:    "model_9999.pt"  (自动在 models/ 查找)
#   本地完整路径:  "/path/to/model.pt"
#   Wandb完整URL: "wandb://entity/project/run_id/model.pt"
#   Wandb短格式:  "wandb:run_id/model.pt" (使用下面默认配置)
# ============================================================

# 模型选择 (只需改这一行!)
# MODEL_NAME="policy.pt"           
MODEL_NAME="policy.pt"
# MODEL_NAME="wandb:2025-12-23_17-11-37_drone_experiment_ppo/model_7000.pt"  # Wandb

# 动力学模型 checkpoint (可选，用于 CBF)
# 为空则不加载独立动力学模型
DYN_CHECKPOINT="drone_rough_2026-02-28_12-11-47_drone_experiment_mbppo_model_20000.pt"

# Wandb 默认配置 (使用短格式时生效)
WANDB_ENTITY="zhangchangxin"
WANDB_PROJECT="UAV_Navigation"

# 参数配置 - 目标点
TARGET_X=0.0
TARGET_Y=0.0
TARGET_Z=1.2

# 世界坐标系校正 (Prometheus 和 AirSim 的 yaw 偏移)
# 如果无人机飞错方向，调整此值 (度)
# 正值 = Prometheus 世界相对于 AirSim 世界顺时针旋转
WORLD_YAW_OFFSET=0.0  # world 和 world_enu 方向一致，不需要校正

# 安全参数 (与 Isaac Lab 训练配置一致: velocity_env_cfg.py:213-221)
MIN_ALTITUDE=1.0      # 最低飞行高度 (米)
# 速度限制
SCALE_HOR=1.0         # 水平速度缩放 (m/s) - action=1 → 1m/s
SCALE_Z=2.0           # 垂直速度缩放 (m/s) - action=1 → 2m/s
MAX_VEL_HOR=1.0       # 最大水平速度 (m/s) - _UAV_MAX_VEL_HOR
MAX_VEL_UP=2.0        # 最大上升速度 (m/s) - _UAV_MAX_VEL_UP
MAX_VEL_DOWN=1.0      # 最大下降速度 (m/s) - _UAV_MAX_VEL_DOWN
# 加速度限制
MAX_ACC_HOR=2.0       # 最大水平加速度 (m/s²) - _UAV_ACC_HOR
MAX_ACC_UP=3.0        # 最大上升加速度 (m/s²) - _UAV_ACC_UP
MAX_ACC_DOWN=2.0      # 最大下降加速度 (m/s²) - _UAV_ACC_DOWN

# ============================================================
# CBF 安全参数 (与 run_play_mb.sh 一致)
# 参考: https://github.com/zhangchangxina/Naci_isaaclab/blob/main/run_play_mb.sh
# ============================================================
USE_CBF=true         # 是否启用 CBF 安全层 (CBF_TEST_MODE=true 时自动覆盖)
CBF_SAFE_DIST=1.0       # Barrier 函数作用距离 (米), 进入此范围 barrier 值开始升高
CBF_SOLVER=gradient     # CBF 求解器: gradient=梯度投影(快,不严格), slsqp=严格QP求解(保证约束)
CBF_BARRIER=reciprocal         # Barrier函数: log=对数(陡), softplus=平滑过渡(推荐), reciprocal=倒数
CBF_GAMMA=""        # CBF gamma (留空=根据barrier类型自动选择推荐值)
CBF_REPULSION_DIST=0.0    # 硬排斥触发距离 (米), 小于此距离施加紧急排斥力
CBF_REPULSION_GAIN=0.0    # 硬排斥力增益 (0=关闭, 0.5=轻柔, 2.0=原始强度)
                      #   手动设值会覆盖自动选择，越小越保守
# 根据 barrier 类型自动设置推荐 gamma (不同barrier值域差异大)
if [ -z "$CBF_GAMMA" ]; then
    case "$CBF_BARRIER" in
        log)        CBF_GAMMA=0.01 ;;  # log barrier 梯度陡，gamma 需要小
        softplus)   CBF_GAMMA=0.1  ;;  # softplus 更平滑，gamma 可以稍大
        reciprocal) CBF_GAMMA=0.05 ;;  # reciprocal 介于两者之间
        *)          CBF_GAMMA=0.01 ;;
    esac
    echo "[CBF] Barrier=${CBF_BARRIER} -> 自动 Gamma=${CBF_GAMMA}"
fi
# --- CBF 测试模式 ---
# 策略输出使用自定义固定值，专门验证 CBF 是否生效
# 测试场景:
#   场景1: 动作=(0,0,0) → 悬停，CBF 不应修改
#   场景2: 动作=(0.5,0,0) → 往前飞，如有障碍 CBF 应减速/偏转
#   场景3: 动作=(1.0,0,0) → 全速前进，CBF 应更明显干预
CBF_TEST_MODE=false        # true = 启用 CBF 测试模式
CBF_TEST_ACTION_VX=0.0     # 固定动作 vx (归一化, [-1, 1])
CBF_TEST_ACTION_VY=0.0     # 固定动作 vy
CBF_TEST_ACTION_VZ=0.0     # 固定动作 vz
if [ "$CBF_TEST_MODE" = "true" ]; then
    USE_CBF=true
    echo "[CBF-TEST] CBF 测试模式已启用!"
    echo "[CBF-TEST] 固定动作: vx=${CBF_TEST_ACTION_VX}, vy=${CBF_TEST_ACTION_VY}, vz=${CBF_TEST_ACTION_VZ}"
    echo "[CBF-TEST] CBF 已自动启用 (USE_CBF=true)"
fi

# ============================================================
# 调试选项 - 禁用雷达 (全部设为最大值 5m)
# 用于测试策略是否能根据 pose_command 正确飞向目标
# ============================================================
DISABLE_LIDAR=false   # true = 忽略雷达数据，全部设为 5m (无障碍)

# ============================================================
# 雷达坐标系修正 (与训练传感器朝向对齐)
# 例如: AirSim 设置 Pitch=-15，而训练是 +15 → 设置 PITCH=30
# ============================================================
LIDAR_CORR_ROLL=0.0
LIDAR_CORR_PITCH=0.0
LIDAR_CORR_YAW=0.0

# 轴镜像修正 (当坐标系左右/前后翻转时使用)
LIDAR_FLIP_X=false
LIDAR_FLIP_Y=false
LIDAR_FLIP_Z=false

# ============================================================
# 调试选项 - 测试模式 (固定动作输出)
# 用于验证坐标系和命令发送是否正确
# ============================================================
TEST_MODE=false       # true = 使用固定动作 (0.5, 0, 0)，无人机应该往前飞
                       # false = 使用策略输出

# ============================================================
# 动作符号修正 (如果策略输出方向相反)
# ============================================================
INVERT_VX=false        # true = 取反 vx (策略输出负值时往前飞)
INVERT_VY=false        # true = 取反 vy

# 不需要 world <-> world_enu 静态 TF
# 只使用 Prometheus 发布的 TF (world, uav1/base_link, uav1/lidar_link)
# AirSim 的 TF (world_enu, uav1/LidarSensor1) 不使用

# 让 uav1/lidar_link 相对 uav1/base_link 有真实偏移/俯仰 (与训练一致)
# x=0.13, y=0, z=0.23, pitch=15deg
# 注释掉: Prometheus 可能已经发布这个 TF，避免冲突
# rosrun tf static_transform_publisher 0.13 0 0.23 0 0.261799 0 uav1/base_link uav1/lidar_link 100 &

# 运行策略节点
echo "启动策略节点..."
echo "模型: $MODEL_NAME"
echo "目标点: ($TARGET_X, $TARGET_Y, $TARGET_Z)"
echo "世界坐标系校正: ${WORLD_YAW_OFFSET}°"
echo "安全限制:"
echo "  - 最低高度: ${MIN_ALTITUDE}m"
echo "  - 速度缩放: 水平=${SCALE_HOR}m/s, 垂直=${SCALE_Z}m/s"
echo "  - 最大速度: 水平=${MAX_VEL_HOR}m/s, 上升=${MAX_VEL_UP}m/s, 下降=${MAX_VEL_DOWN}m/s"
echo "  - 最大加速度: 水平=${MAX_ACC_HOR}m/s², 上升=${MAX_ACC_UP}m/s², 下降=${MAX_ACC_DOWN}m/s²"
echo "  - CBF: use_cbf=${USE_CBF}, gamma=${CBF_GAMMA}, safe_dist=${CBF_SAFE_DIST}m, solver=${CBF_SOLVER}, barrier=${CBF_BARRIER}, repulsion=${CBF_REPULSION_DIST}m(gain=${CBF_REPULSION_GAIN})"
echo "  - 雷达禁用: ${DISABLE_LIDAR}"
echo "  - 测试模式: ${TEST_MODE}"
echo "  - CBF测试模式: ${CBF_TEST_MODE} (action=[${CBF_TEST_ACTION_VX}, ${CBF_TEST_ACTION_VY}, ${CBF_TEST_ACTION_VZ}])"
echo "  - 动作取反: vx=${INVERT_VX}, vy=${INVERT_VY}"
echo ""
echo "设置新目标: rostopic pub -1 /uav1/target_position geometry_msgs/PoseStamped \"{pose: {position: {x: 10, y: 5, z: 1.5}}}\""
echo ""

python3 isaaclab_policy_node.py \
    _uav_id:=$UAV_ID \
    _model_name:=$MODEL_NAME \
    _dyn_checkpoint:=$DYN_CHECKPOINT \
    _wandb_entity:=$WANDB_ENTITY \
    _wandb_project:=$WANDB_PROJECT \
    _target_x:=$TARGET_X \
    _target_y:=$TARGET_Y \
    _target_z:=$TARGET_Z \
    _world_yaw_offset:=$WORLD_YAW_OFFSET \
    _min_altitude:=$MIN_ALTITUDE \
    _scale_hor:=$SCALE_HOR \
    _scale_z:=$SCALE_Z \
    _max_vel_hor:=$MAX_VEL_HOR \
    _max_vel_up:=$MAX_VEL_UP \
    _max_vel_down:=$MAX_VEL_DOWN \
    _max_acc_hor:=$MAX_ACC_HOR \
    _max_acc_up:=$MAX_ACC_UP \
    _max_acc_down:=$MAX_ACC_DOWN \
    _use_cbf:=$USE_CBF \
    _cbf_gamma:=$CBF_GAMMA \
    _cbf_safe_dist:=$CBF_SAFE_DIST \
    _cbf_repulsion_dist:=$CBF_REPULSION_DIST \
    _cbf_solver:=$CBF_SOLVER \
    _cbf_barrier:=$CBF_BARRIER \
    _cbf_repulsion_gain:=$CBF_REPULSION_GAIN \
    _disable_lidar:=$DISABLE_LIDAR \
    _lidar_corr_roll:=$LIDAR_CORR_ROLL \
    _lidar_corr_pitch:=$LIDAR_CORR_PITCH \
    _lidar_corr_yaw:=$LIDAR_CORR_YAW \
    _lidar_flip_x:=$LIDAR_FLIP_X \
    _lidar_flip_y:=$LIDAR_FLIP_Y \
    _lidar_flip_z:=$LIDAR_FLIP_Z \
    _test_mode:=$TEST_MODE \
    _cbf_test_mode:=$CBF_TEST_MODE \
    _cbf_test_action_vx:=$CBF_TEST_ACTION_VX \
    _cbf_test_action_vy:=$CBF_TEST_ACTION_VY \
    _cbf_test_action_vz:=$CBF_TEST_ACTION_VZ \
    _invert_vx:=$INVERT_VX \
    _invert_vy:=$INVERT_VY

# 关闭 rviz 和 airsim_node
kill $RVIZ_PID 2>/dev/null
if [ -n "$AIRSIM_PID" ]; then
    kill $AIRSIM_PID 2>/dev/null
fi