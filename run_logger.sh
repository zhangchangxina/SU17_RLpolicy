#!/bin/bash
# ============================================================
# 通用飞行数据记录器 (统一版)
#
# 配合任意避障方法使用，只需在避障方法运行时同时启动本脚本。
# 自动订阅 Prometheus 标准话题，记录飞行数据到 CSV。
# 配合 Isaac Lab RL 时还会自动订阅额外话题（action_raw, lidar等）。
#
# 用法:
#   ./run_logger.sh ego_planner     ← 雷达避障 (EGO-Planner)
#   ./run_logger.sh yopo            ← YOPO 深度避障
#   ./run_logger.sh policy          ← Isaac Lab RL 策略
#   ./run_logger.sh my_method       ← 任意自定义方法名
#   ./run_logger.sh policy true     ← 自动发送目标点
#
# 记录结束后:
#   python3 plot_trajectory.py                  ← 绘制最新一次飞行
#   python3 compare_policies.py                 ← 对比所有方法
# ============================================================

METHOD=${1:-"unknown"}

UAV_ID=1

# 目标点 (与避障方法中设置的一致!)
TARGET_X=0.0
TARGET_Y=0.0
TARGET_Z=1.2

# 记录频率 (Hz)
LOG_FREQ=10.0

source /opt/ros/noetic/setup.bash
source ~/su17_experiment/devel/setup.bash

cd /home/amov/su17_experiment/src/Prometheus/Modules/isaaclabpolicy

echo "=========================================="
echo "飞行数据记录器 (统一版)"
echo "=========================================="
echo "  方法: $METHOD"
echo "  目标: ($TARGET_X, $TARGET_Y, $TARGET_Z)"
echo "  频率: ${LOG_FREQ}Hz"
echo ""
echo "  通用话题: prometheus/state, command, odom"
echo "  RL 话题:  isaaclab/action_raw, min_obstacle_dist, lidar_scan"
echo "            (有数据时自动使用，无数据时忽略)"
echo ""
echo "  Ctrl+C 停止记录"
echo "=========================================="

# 如果设置了 SEND_GOAL=true，启动后自动发布目标点
SEND_GOAL=${2:-"false"}

# 不同避障方法使用不同的目标话题
if [ "$METHOD" = "ego_planner" ]; then
    GOAL_TOPIC="/uav${UAV_ID}/prometheus/motion_planning/goal"
elif [ "$METHOD" = "policy" ]; then
    # Isaac Lab RL 策略使用 motion_planning/goal (实机)
    GOAL_TOPIC="/uav${UAV_ID}/prometheus/motion_planning/goal"
else
    # YOPO 和其他方法使用 move_base_simple/goal
    GOAL_TOPIC="/move_base_simple/goal"
fi

if [ "$SEND_GOAL" = "true" ] || [ "$SEND_GOAL" = "1" ]; then
    echo "  目标话题: $GOAL_TOPIC"
    echo "  3秒后自动发送目标点..."
    (sleep 3 && rostopic pub "$GOAL_TOPIC" geometry_msgs/PoseStamped \
        "{header: {frame_id: 'world'}, pose: {position: {x: $TARGET_X, y: $TARGET_Y, z: $TARGET_Z}, orientation: {w: 1.0}}}" --once \
        && echo "  ✓ 目标已发送: ($TARGET_X, $TARGET_Y, $TARGET_Z)") &
fi

python3 flight_logger.py \
    _uav_id:=$UAV_ID \
    _method:=$METHOD \
    _target_x:=$TARGET_X \
    _target_y:=$TARGET_Y \
    _target_z:=$TARGET_Z \
    _log_freq:=$LOG_FREQ

echo ""
echo "=========================================="
echo "  记录结束，正在自动生成图表..."
echo "=========================================="
python3 plot_trajectory.py
echo "  ✓ 图表生成完毕！"
