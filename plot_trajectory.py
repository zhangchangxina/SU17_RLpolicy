#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 飞行数据综合分析 & 绘图脚本

生成图表:
  Figure 1: 俯视轨迹 + 距离时间曲线          (flight_*_1_trajectory.png)
  Figure 2: 速度分析 (世界系/机体系/速度大小)  (flight_*_2_velocity.png)
  Figure 3: 策略动作分析 (raw/cmd/航向)        (flight_*_3_actions.png)
  Figure 4: 安全指标 (障碍/高度/状态饼图)      (flight_*_4_safety.png)
  Figure 5: 观测空间 (角速度/重力/pose_cmd)    (flight_*_5_observations.png)
  Figure 6: 雷达热力图 (如有 npz 快照)         (flight_*_6_lidar.png)

用法:
    python3 plot_trajectory.py                             # 自动加载最新飞行
    python3 plot_trajectory.py flight_logs/flight_xxx.csv  # 指定文件
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec

# ---- 字体配置: 中文宋体 + 英文/数字 Times New Roman ----
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'serif'],
    'font.serif': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,     # 正确显示负号
    'mathtext.fontset': 'stix',      # 公式字体与 Times 一致
})


# ============================================================================
# 工具函数
# ============================================================================

def load_flight_data(csv_path):
    """加载飞行 CSV 数据"""
    df = pd.read_csv(csv_path)
    print(f"加载数据: {csv_path}")
    print(f"  数据点: {len(df)}")
    print(f"  飞行时长: {df['time'].iloc[-1]:.1f}s")
    print(f"  起点: ({df['x'].iloc[0]:.2f}, {df['y'].iloc[0]:.2f}, {df['z'].iloc[0]:.2f})")
    print(f"  终点: ({df['x'].iloc[-1]:.2f}, {df['y'].iloc[-1]:.2f}, {df['z'].iloc[-1]:.2f})")
    print(f"  目标: ({df['target_x'].iloc[-1]:.2f}, {df['target_y'].iloc[-1]:.2f}, {df['target_z'].iloc[-1]:.2f})")
    return df


def find_latest_log():
    """查找最新的飞行日志文件 (支持按策略分文件夹的新目录结构)"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flight_logs')
    if not os.path.exists(log_dir):
        print(f"飞行日志目录不存在: {log_dir}")
        print("请先运行策略节点飞行一次!")
        sys.exit(1)
    # 搜索两层: flight_logs/flight_*.csv 和 flight_logs/*/flight_*.csv
    csv_files = sorted(glob.glob(os.path.join(log_dir, 'flight_*.csv')))
    csv_files += sorted(glob.glob(os.path.join(log_dir, '*', 'flight_*.csv')))
    csv_files = [f for f in csv_files if not f.endswith('_lidar.npz')]
    if not csv_files:
        print("没有找到飞行日志文件!")
        sys.exit(1)
    # 按修改时间排序，取最新
    csv_files.sort(key=os.path.getmtime)
    print(f"找到 {len(csv_files)} 个日志文件，使用最新的:")
    return csv_files[-1]


def save_fig(fig, csv_path, suffix):
    """保存图表"""
    save_path = csv_path.replace('.csv', f'_{suffix}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存: {os.path.basename(save_path)}")
    plt.close(fig)
    return save_path


# ============================================================================
# 场景障碍物 (从 AirSim API 精确获取)
# ============================================================================

def load_scene_obstacles(csv_path=None):
    """加载场景障碍物 JSON
    
    搜索顺序:
      1. csv 所在目录/scene_obstacles.json
      2. csv 上级目录/scene_obstacles.json
      3. 根据上级目录名匹配: flight_logs/scene_obstacles_<场景名>.json
      4. flight_logs/scene_obstacles.json (全局默认)
    
    Returns:
        tuple: (obstacles_list, scene_name)
    """
    search_paths = []
    if csv_path:
        csv_dir = os.path.dirname(csv_path)
        parent_dir = os.path.dirname(csv_dir)
        search_paths.append(os.path.join(csv_dir, 'scene_obstacles.json'))
        search_paths.append(os.path.join(parent_dir, 'scene_obstacles.json'))
        # 根据目录名自动匹配场景 JSON
        # 例如 flight_logs/factory/factory_policy_.../flight.csv
        #   → parent = flight_logs/factory → scene_dir_name = factory
        #   → 查 flight_logs/scene_obstacles_factory.json
        grandparent_dir = os.path.dirname(parent_dir)
        scene_dir_name = os.path.basename(os.path.normpath(parent_dir))
        scene_json = os.path.join(grandparent_dir, f'scene_obstacles_{scene_dir_name}.json')
        search_paths.append(scene_json)
        # 也尝试从 CSV 目录的文件夹名提取场景名 (如 forest_policy_xxx → forest)
        folder_name = os.path.basename(csv_dir)
        for known in ('pillar', 'factory', 'forest', 'industrial', 'sports', 'indoor', 'outdoor'):
            if folder_name.startswith(known + '_'):
                search_paths.append(os.path.join(parent_dir, f'scene_obstacles_{known}.json'))
                search_paths.append(os.path.join(grandparent_dir, f'scene_obstacles_{known}.json'))
                break
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, 'flight_logs', 'scene_obstacles.json'))

    for p in search_paths:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            return data.get('obstacles', []), data.get('scene_name', None)
    return None, None


# 场景名中英文映射
SCENE_NAMES_CN = {'pillar': '柱林', 'factory': '工厂', 'forest': '树林',
                  'industrial': '厂区', 'sports': '运动场', 'outdoor': '室外',
                  'indoor': '室内'}


def load_pcl_points(csv_path, z_min=None):
    """加载 PointCloud2 点云快照 (_pcl.npz), 返回世界 ENU 坐标 (N,3) 或 None
    
    Args:
        csv_path: CSV 文件路径
        z_min: 最低高度阈值 (m), 低于此高度的点视为地面点并过滤掉.
               如果为 None, 自动根据飞行数据确定 (飞行最低高度 - 1.5m, 至少 0.3m).
    """
    pcl_path = csv_path.replace('.csv', '_pcl.npz')
    if not os.path.exists(pcl_path):
        return None
    try:
        data = np.load(pcl_path)
        pts = data['points_enu']  # (N, 3) 世界 ENU
        if len(pts) == 0:
            return None
        # 自动计算地面过滤阈值: 完全去除地面回波
        # 激光雷达向下扫到地面 (z≈0) 会产生大量地面点,
        # 这些对障碍物可视化无用, 必须完全移除
        if z_min is None:
            try:
                df = pd.read_csv(csv_path)
                flight_z_min = df['z'].min()
                # 飞行最低高度 × 0.5, 至少 0.5m
                # 例: 飞行最低1.68m → z_min=0.84m, 彻底去除地面
                z_min = max(0.5, flight_z_min * 0.5)
            except Exception:
                z_min = 0.5
        # 过滤地面点
        pts = pts[pts[:, 2] >= z_min]
        if len(pts) == 0:
            return None
        return pts
    except Exception:
        return None


def draw_obstacles_2d(ax, obstacles, alpha=0.3, color='#888888', edgecolor='#555555'):
    """在 2D Axes 上画场景障碍物 (俯视图, 细分类型, 不同形状+颜色)"""
    if not obstacles:
        return

    # 类型→样式映射 (颜色区分明显)
    _STYLES = {
        # ---- 植被 ----
        'tree':       {'fc': '#5cb85c', 'ec': '#3a7a3a'},  # 深绿
        'bush':       {'fc': '#a8d5a0', 'ec': '#6aaa60'},  # 浅绿
        # ---- 圆柱/管道 ----
        'cylinder':   {'fc': '#b0b0b0', 'ec': '#666666'},  # 银灰
        'lamp':       {'fc': '#f0e68c', 'ec': '#bbb030'},  # 浅黄 (灯柱)
        # ---- 桶/罐 ----
        'barrel':     {'fc': '#d87d3e', 'ec': '#a05020'},  # 铁锈橙
        # ---- 路障/护栏 ----
        'barrier':    {'fc': '#ff6b6b', 'ec': '#cc3333'},  # 红色警示
        # ---- 方块/箱子 ----
        'cube':       {'fc': '#e8c170', 'ec': '#a08030'},  # 金黄
        'shelf':      {'fc': '#f0a858', 'ec': '#c07020'},  # 橙色
        'furniture':  {'fc': '#c9a86c', 'ec': '#8b6914'},  # 木色
        'equipment':  {'fc': '#7ec8e3', 'ec': '#3a8aaa'},  # 天蓝 (设备)
        # ---- 墙/建筑 ----
        'wall':       {'fc': '#c8a882', 'ec': '#7a5a3a'},  # 棕色
        'building':   {'fc': '#d4a574', 'ec': '#8b6914'},  # 土黄
        # ---- 建筑结构件 ----
        'roof':       {'fc': '#a0522d', 'ec': '#6b3410'},  # 深棕 (屋顶)
        'foundation': {'fc': '#808080', 'ec': '#4a4a4a'},  # 深灰 (地基)
        'stairs':     {'fc': '#b8860b', 'ec': '#8b6508'},  # 暗金 (楼梯)
        'platform':   {'fc': '#9e9e9e', 'ec': '#5e5e5e'},  # 中灰 (平台)
        # ---- 车辆 ----
        'vehicle':    {'fc': '#6eaadc', 'ec': '#2a5a8a'},  # 蓝色
        # ---- 其他 ----
        'sphere':     {'fc': '#d0c0e8', 'ec': '#8070a0'},  # 淡紫
        'rock':       {'fc': '#8b8680', 'ec': '#5a5550'},  # 灰褐
        'unknown':    {'fc': '#cccccc', 'ec': '#999999'},  # 浅灰
    }

    # 这些类型逻辑上应画为矩形 (即使 JSON 没提供 half_x, 也用 radius 推算)
    _RECT_TYPES = {'cube', 'wall', 'shelf', 'building', 'furniture', 'equipment',
                   'roof', 'foundation', 'stairs', 'platform'}

    for ob in obstacles:
        t = ob.get('type', 'unknown')
        if t == 'skip':
            continue

        cx, cy, r = ob['x'], ob['y'], ob['radius']
        st = _STYLES.get(t, _STYLES['unknown'])

        # ---------- 树木: 半透明绿色树冠 + 棕色树干 ----------
        if t == 'tree':
            canopy_r = ob.get('canopy_radius', r)
            trunk_r  = ob.get('trunk_radius', max(0.15, r * 0.15))
            ax.add_patch(Circle((cx, cy), canopy_r,
                                fc='#5cb85c', ec='#3a7a3a', alpha=alpha * 0.5,
                                lw=0.5, zorder=1))
            ax.add_patch(Circle((cx, cy), trunk_r,
                                fc='#8B5A2B', ec='#5a3a10', alpha=min(alpha + 0.2, 0.8),
                                lw=0.5, zorder=2))

        # ---------- 灌木: 浅绿小圆 ----------
        elif t == 'bush':
            ax.add_patch(Circle((cx, cy), r,
                                fc='#a8d5a0', ec='#6aaa60', alpha=alpha * 0.4,
                                lw=0.3, zorder=1))

        # ---------- 路障: 红色菱形标记 ----------
        elif t == 'barrier':
            from matplotlib.patches import RegularPolygon
            ax.add_patch(RegularPolygon((cx, cy), numVertices=4, radius=r,
                                        fc='#ff6b6b', ec='#cc3333', alpha=alpha,
                                        lw=0.5, zorder=1))

        # ---------- 桶/罐: 橙色小圆 + 加粗边框 ----------
        elif t == 'barrel':
            ax.add_patch(Circle((cx, cy), r,
                                fc='#d87d3e', ec='#a05020', alpha=alpha,
                                lw=1.0, zorder=1))

        # ---------- 矩形类: 有 half_x 用精确值, 没有则用 radius 推算 ----------
        elif 'half_x' in ob or t in _RECT_TYPES:
            hx = ob.get('half_x', r * 0.7)
            hy = ob.get('half_y', r * 0.7)
            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                   fc=st['fc'], ec=st['ec'], alpha=alpha,
                                   lw=0.5, zorder=1))

        # ---------- 圆形类 (cylinder/sphere/lamp/rock/vehicle) ----------
        else:
            ax.add_patch(Circle((cx, cy), r,
                                fc=st['fc'], ec=st['ec'], alpha=alpha,
                                lw=0.5, zorder=1))


def draw_obstacles_3d(ax, obstacles, alpha=0.2, color='gray'):
    """在 3D Axes 上画场景障碍物 (细分类型, 不同颜色线框)"""
    if not obstacles:
        return
    _TYPE_COLORS = {
        'tree':       '#3a7a3a',   # 深绿
        'bush':       '#6aaa60',   # 浅绿
        'cylinder':   '#888888',   # 灰
        'barrel':     '#a05020',   # 铁锈橙
        'barrier':    '#cc3333',   # 红
        'lamp':       '#bbb030',   # 浅黄
        'cube':       '#a08030',   # 金黄
        'shelf':      '#c07020',   # 橙
        'furniture':  '#8b6914',   # 木色
        'equipment':  '#3a8aaa',   # 天蓝
        'wall':       '#7a5a3a',   # 棕
        'building':   '#a0522d',   # 深棕
        'roof':       '#6b3410',   # 深棕 (屋顶)
        'foundation': '#4a4a4a',   # 深灰 (地基)
        'stairs':     '#8b6508',   # 暗金 (楼梯)
        'platform':   '#5e5e5e',   # 中灰 (平台)
        'vehicle':    '#2a5a8a',   # 蓝
        'sphere':     '#8070a0',   # 淡紫
        'rock':       '#8b8680',   # 灰褐
    }
    # 这些类型逻辑上应画为长方体 (即使 JSON 没有 half_x, 也用 radius 推算)
    _RECT_TYPES = {'cube', 'wall', 'shelf', 'building', 'furniture', 'equipment',
                   'roof', 'foundation', 'stairs', 'platform'}
    theta = np.linspace(0, 2 * np.pi, 24)
    for ob in obstacles:
        t = ob.get('type', 'unknown')
        if t == 'skip':
            continue
        r = ob['radius']
        if r < 0.1:
            continue
        cx, cy = ob['x'], ob['y']
        z_center = ob.get('z', 5.0)
        h = ob.get('height', 10.0)
        z_lo = max(0, z_center - h / 2)
        z_hi = z_center + h / 2
        c = _TYPE_COLORS.get(t, color)

        if t == 'tree':
            # 树木: 细棕色树干 + 绿色球冠轮廓
            trunk_r = max(0.15, r * 0.15)
            xs_t = cx + trunk_r * np.cos(theta)
            ys_t = cy + trunk_r * np.sin(theta)
            ax.plot(xs_t, ys_t, z_lo * np.ones_like(theta),
                    color='#8B5A2B', alpha=alpha, lw=0.3)
            ax.plot(xs_t, ys_t, z_hi * 0.6 * np.ones_like(theta),
                    color='#8B5A2B', alpha=alpha, lw=0.3)
            for k in range(0, len(theta), 8):
                ax.plot([xs_t[k]] * 2, [ys_t[k]] * 2, [z_lo, z_hi * 0.6],
                        color='#8B5A2B', alpha=alpha * 0.5, lw=0.3)
            canopy_r = ob.get('canopy_radius', r)
            xs_c = cx + canopy_r * np.cos(theta)
            ys_c = cy + canopy_r * np.sin(theta)
            ax.plot(xs_c, ys_c, z_hi * 0.7 * np.ones_like(theta),
                    color='#3a7a3a', alpha=alpha * 0.8, lw=0.4)

        elif t == 'bush':
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            ax.plot(xs, ys, z_lo * np.ones_like(theta),
                    color=c, alpha=alpha * 0.5, lw=0.2)
            z_top = min(z_lo + 1.5, z_hi)
            ax.plot(xs, ys, z_top * np.ones_like(theta),
                    color=c, alpha=alpha * 0.5, lw=0.2)

        elif 'half_x' in ob or t in _RECT_TYPES:
            # 矩形类: 线框长方体 (有精确值用精确值, 没有则用 radius 推算)
            hx = ob.get('half_x', r * 0.7)
            hy = ob.get('half_y', r * 0.7)
            corners = [(cx - hx, cy - hy), (cx + hx, cy - hy),
                       (cx + hx, cy + hy), (cx - hx, cy + hy),
                       (cx - hx, cy - hy)]
            cxs = [p[0] for p in corners]
            cys = [p[1] for p in corners]
            ax.plot(cxs, cys, [z_lo] * 5, color=c, alpha=alpha, lw=0.3)
            ax.plot(cxs, cys, [z_hi] * 5, color=c, alpha=alpha, lw=0.3)
            for p in corners[:4]:
                ax.plot([p[0]] * 2, [p[1]] * 2, [z_lo, z_hi],
                        color=c, alpha=alpha * 0.5, lw=0.2)

        else:
            # 圆形类: 圆柱线框
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            ax.plot(xs, ys, z_lo * np.ones_like(theta), color=c, alpha=alpha, lw=0.3)
            ax.plot(xs, ys, z_hi * np.ones_like(theta), color=c, alpha=alpha, lw=0.3)
            for k in range(0, len(theta), 6):
                ax.plot([xs[k]] * 2, [ys[k]] * 2, [z_lo, z_hi],
                        color=c, alpha=alpha * 0.6, lw=0.3)


# ============================================================================
# 雷达点云 → 世界坐标系
# ============================================================================

def lidar_to_world_xy(npz_path, lidar_range=5.0):
    """
    将雷达快照转换为世界坐标系 XY 点 (用于俯视图障碍物绘制)
    
    优化策略:
      1. 只取水平面附近的通道 (垂直角 < 25°)，过滤掉高仰角噪点
      2. 每帧取每个水平方向最近的一个点 (去重)
      3. 对最终结果做栅格去重 (0.3m 分辨率)，消除多帧重叠
    """
    if not os.path.exists(npz_path):
        return None, None
    
    data = np.load(npz_path)
    times = data['times']
    scans = data['scans']
    pos_x = data['pos_x']
    pos_y = data['pos_y']
    yaw_deg_arr = data['yaw_deg']
    channels = int(data['channels'])        # 8
    h_points = int(data['horizontal_points'])  # 35
    
    # 预计算: 只用水平面附近的通道 (垂直角 < 25°，避免高仰角的天花板/地面噪点)
    valid_channels = []
    for vi in range(channels):
        v_angle_deg = -7.0 + vi * (59.0 / (channels - 1))
        if abs(v_angle_deg) < 25:
            valid_channels.append(vi)
    if not valid_channels:
        valid_channels = [0]  # 至少用第一个通道
    
    all_ox = []
    all_oy = []
    all_dist = []
    
    for i in range(len(times)):
        scan = scans[i].reshape(channels, h_points)
        px, py = pos_x[i], pos_y[i]
        yaw = np.radians(yaw_deg_arr[i])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 对每个水平方向，取有效通道中最近距离
        for hi in range(h_points):
            min_dist = lidar_range
            for vi in valid_channels:
                d = scan[vi, hi]
                if d < min_dist:
                    min_dist = d
            
            if min_dist >= lidar_range - 0.01:
                continue
            
            h_angle = np.radians(-180.0 + hi * 10.0)
            ray_bx = np.cos(h_angle)
            ray_by = np.sin(h_angle)
            ray_wx = cos_yaw * ray_bx - sin_yaw * ray_by
            ray_wy = sin_yaw * ray_bx + cos_yaw * ray_by
            all_ox.append(px + min_dist * ray_wx)
            all_oy.append(py + min_dist * ray_wy)
            all_dist.append(min_dist)
    
    if not all_ox:
        return None, None
    
    ox = np.array(all_ox)
    oy = np.array(all_oy)
    od = np.array(all_dist)
    
    # 栅格去重: 将点snap到 0.3m 网格，每个格子只保留最近的点
    grid_res = 0.3
    grid_keys = (np.round(ox / grid_res).astype(int),
                 np.round(oy / grid_res).astype(int))
    seen = {}
    for j in range(len(ox)):
        key = (grid_keys[0][j], grid_keys[1][j])
        if key not in seen or od[j] < seen[key][2]:
            seen[key] = (ox[j], oy[j], od[j])
    
    if not seen:
        return None, None
    
    vals = list(seen.values())
    return (np.array([v[0] for v in vals]),
            np.array([v[1] for v in vals]),
            np.array([v[2] for v in vals]))


def lidar_to_world_xyz(npz_path, lidar_range=5.0):
    """
    将雷达快照转换为世界坐标系 XYZ 点 (用于3D轨迹图障碍物绘制)
    
    优化: 栅格去重 (0.3m 分辨率) + 仅保留垂直角 < 35° 的通道
    返回: (obs_x, obs_y, obs_z, obs_dist) 或 None
    """
    if not os.path.exists(npz_path):
        return None
    
    data = np.load(npz_path)
    times = data['times']
    scans = data['scans']
    pos_x = data['pos_x']
    pos_y = data['pos_y']
    pos_z = data['pos_z']
    yaw_deg_arr = data['yaw_deg']
    channels = int(data['channels'])
    h_points = int(data['horizontal_points'])
    
    # 只取垂直角 < 35° 的通道 (3D 图中保留更多通道以显示高度信息)
    valid_channels = []
    for vi in range(channels):
        v_angle_deg = -7.0 + vi * (59.0 / max(channels - 1, 1))
        if v_angle_deg < 35:
            valid_channels.append(vi)
    if not valid_channels:
        valid_channels = list(range(channels))
    
    all_ox, all_oy, all_oz, all_dist = [], [], [], []
    
    for i in range(len(times)):
        scan = scans[i].reshape(channels, h_points)
        px, py, pz = pos_x[i], pos_y[i], pos_z[i]
        yaw = np.radians(yaw_deg_arr[i])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        for vi in valid_channels:
            v_angle = np.radians(-7.0 + vi * (59.0 / max(channels - 1, 1)))
            cos_v = np.cos(v_angle)
            sin_v = np.sin(v_angle)
            for hi in range(h_points):
                dist = scan[vi, hi]
                if dist >= lidar_range - 0.01:
                    continue
                
                h_angle = np.radians(-180.0 + hi * 10.0)
                ray_bx = cos_v * np.cos(h_angle)
                ray_by = cos_v * np.sin(h_angle)
                ray_bz = sin_v
                ray_wx = cos_yaw * ray_bx - sin_yaw * ray_by
                ray_wy = sin_yaw * ray_bx + cos_yaw * ray_by
                ray_wz = ray_bz
                all_ox.append(px + dist * ray_wx)
                all_oy.append(py + dist * ray_wy)
                all_oz.append(pz + dist * ray_wz)
                all_dist.append(dist)
    
    if not all_ox:
        return None
    
    ox = np.array(all_ox)
    oy = np.array(all_oy)
    oz = np.array(all_oz)
    od = np.array(all_dist)
    
    # 3D 栅格去重 (0.3m 分辨率)
    grid_res = 0.3
    gx = np.round(ox / grid_res).astype(int)
    gy = np.round(oy / grid_res).astype(int)
    gz = np.round(oz / grid_res).astype(int)
    seen = {}
    for j in range(len(ox)):
        key = (gx[j], gy[j], gz[j])
        if key not in seen or od[j] < seen[key][3]:
            seen[key] = (ox[j], oy[j], oz[j], od[j])
    
    if not seen:
        return None
    
    vals = list(seen.values())
    return (np.array([v[0] for v in vals]),
            np.array([v[1] for v in vals]),
            np.array([v[2] for v in vals]),
            np.array([v[3] for v in vals]))


# ============================================================================
# Figure 1: 俯视轨迹 + 距离时间曲线
# ============================================================================

def load_background_image(csv_path, bg_path=None):
    """加载俯视背景图和坐标映射信息
    
    Returns:
        (image, extent) 或 (None, None)
        extent = [x_min, x_max, y_min, y_max] (ENU 坐标系)
    """
    import json
    
    log_dir = os.path.dirname(csv_path)
    
    # 查找背景图
    if bg_path is None:
        bg_path = os.path.join(log_dir, 'topdown_scene.png')
    if not os.path.exists(bg_path):
        return None, None
    
    # 查找坐标映射
    meta_path = os.path.join(log_dir, 'topdown_meta.json')
    if not os.path.exists(meta_path):
        # 没有 meta 文件，无法对齐
        print("  ⚠ 找到背景图但缺少 topdown_meta.json，无法对齐坐标")
        return None, None
    
    try:
        img = plt.imread(bg_path)
        with open(meta_path) as f:
            meta = json.load(f)
        extent = meta['extent_enu']  # [x_min, x_max, y_min, y_max]
        print(f"  ✓ 加载背景图: {os.path.basename(bg_path)} ({img.shape[1]}×{img.shape[0]})")
        print(f"    坐标范围 ENU: X=[{extent[0]:.0f}, {extent[1]:.0f}], Y=[{extent[2]:.0f}, {extent[3]:.0f}]")
        return img, extent
    except Exception as e:
        print(f"  ⚠ 加载背景图失败: {e}")
        return None, None


def plot_fig1_trajectory(df, csv_path, bg_path=None):
    """俯视轨迹图 + 到目标距离"""
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('飞行总览', fontsize=16, fontweight='bold', y=1.02)

    # --- 左图: 俯视轨迹 (X-Y) ---
    ax1 = fig.add_subplot(121)

    x, y = df['x'].values, df['y'].values
    t = df['time'].values
    speed = np.sqrt(df['vx_world']**2 + df['vy_world']**2).values
    target_x, target_y = df['target_x'].iloc[-1], df['target_y'].iloc[-1]

    # --- 场景背景图 (如果有) ---
    bg_img, bg_extent = load_background_image(csv_path, bg_path)
    if bg_img is not None:
        ax1.imshow(bg_img, extent=bg_extent, aspect='equal', alpha=0.6, zorder=0)

    # 轨迹按速度着色
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, max(speed.max(), 0.1))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(speed[:-1])
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('速度 (m/s)', fontsize=10)

    # 速度箭头
    n_arrows = 15
    step = max(1, len(df) // n_arrows)
    for i in range(0, len(df), step):
        vxw = df['vx_world'].iloc[i]
        vyw = df['vy_world'].iloc[i]
        spd = np.sqrt(vxw**2 + vyw**2)
        if spd > 0.05:
            s = 0.8 / max(speed.max(), 0.1)
            ax1.annotate('', xy=(x[i]+vxw*s, y[i]+vyw*s), xytext=(x[i], y[i]),
                         arrowprops=dict(arrowstyle='->', color='gray', lw=1.0, alpha=0.5))

    # 起/终/目标标记
    ax1.plot(x[0], y[0], 'o', color='limegreen', ms=12,
             markeredgecolor='darkgreen', markeredgewidth=2, zorder=5, label='起点')
    ax1.plot(x[-1], y[-1], 's', color='tomato', ms=12,
             markeredgecolor='darkred', markeredgewidth=2, zorder=5, label='终点')
    ax1.plot(target_x, target_y, '*', color='gold', ms=20,
             markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5, label='目标')
    ax1.plot([x[0], target_x], [y[0], target_y], '--', color='gray', alpha=0.4, lw=1)

    # --- 障碍物 ---
    # 优先: 场景障碍物 JSON (AirSim API 精确网格)
    scene_obs, scene_name = load_scene_obstacles(csv_path)
    if scene_name:
        cn = SCENE_NAMES_CN.get(scene_name, scene_name)
        fig.suptitle(f'飞行总览 — {cn}场景', fontsize=16, fontweight='bold', y=1.02)
    # 轴范围只根据飞行轨迹 + 目标确定 (不被远处障碍物撑大)
    all_x = np.concatenate([x, [target_x]])
    all_y = np.concatenate([y, [target_y]])

    if scene_obs:
        # 只画轨迹附近的障碍物 (避免远处障碍物干扰)
        margin_obs = max(10, (all_x.max() - all_x.min()) * 0.3)
        nearby_obs = [ob for ob in scene_obs
                      if (all_x.min() - margin_obs <= ob['x'] <= all_x.max() + margin_obs and
                          all_y.min() - margin_obs <= ob['y'] <= all_y.max() + margin_obs)]
        draw_obstacles_2d(ax1, nearby_obs, alpha=0.35, color='#aaaaaa', edgecolor='#666666')
    else:
        # 回退: 优先用 _pcl.npz (无 bin 量化跳变), 否则用 _lidar.npz
        pcl_pts_fig1 = load_pcl_points(csv_path)
        if pcl_pts_fig1 is not None:
            ox, oy = pcl_pts_fig1[:, 0], pcl_pts_fig1[:, 1]
            grid_res = 0.15
            gx = np.round(ox / grid_res).astype(int)
            gy = np.round(oy / grid_res).astype(int)
            seen = {}
            for j in range(len(ox)):
                key = (gx[j], gy[j])
                if key not in seen:
                    seen[key] = (ox[j], oy[j])
            if seen:
                vals = list(seen.values())
                fx = np.array([v[0] for v in vals])
                fy = np.array([v[1] for v in vals])
                ax1.scatter(fx, fy, c='gray', s=3, alpha=0.35, zorder=1,
                            edgecolors='none', label=f'障碍物 ({len(fx)} 点)')
        else:
            npz_path = csv_path.replace('.csv', '_lidar.npz')
            result = lidar_to_world_xy(npz_path)
            if result is not None and result[0] is not None:
                obs_x, obs_y, obs_dist = result
                ax1.scatter(obs_x, obs_y, c=obs_dist, cmap='Greys_r',
                            s=3, alpha=0.35, vmin=0, vmax=5.0, zorder=1,
                            label=f'障碍物 ({len(obs_x)} 点)')

    # 统计信息
    direct_dist = np.sqrt((target_x-x[0])**2 + (target_y-y[0])**2)
    path_len = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    final_dist = df['dist_to_target'].iloc[-1]
    eff = direct_dist / max(path_len, 0.01) * 100
    # 到达时间: 首次 dist < 1.5m
    arrive_mask = df['dist_to_target'] < 1.5
    arrive_time = df.loc[arrive_mask, 'time'].iloc[0] if arrive_mask.any() else None

    info = (f'直线距离: {direct_dist:.1f}m\n'
            f'实际路程: {path_len:.1f}m\n'
            f'路径效率: {eff:.0f}%\n'
            f'最终距离: {final_dist:.2f}m\n'
            f'飞行时长: {t[-1]:.1f}s')
    if arrive_time is not None:
        info += f'\n到达时间: {arrive_time:.1f}s'
    ax1.text(0.02, 0.98, info, transform=ax1.transAxes, fontsize=9,
             va='top', bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.8))

    margin = 2.0
    ax1.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax1.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('俯视轨迹（按速度着色）')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # --- 右图: 距离时间曲线 ---
    ax2 = fig.add_subplot(122)
    ax2.plot(t, df['dist_to_target'], color='royalblue', lw=2, label='到目标距离')
    ax2.axhline(y=1.0, color='green', ls='--', alpha=0.5, label='悬停阈值 (1m)')
    if arrive_time is not None:
        ax2.axvline(x=arrive_time, color='gold', ls=':', alpha=0.7, label=f'到达 @{arrive_time:.1f}s')

    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('到目标距离 (m)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')

    ax3 = ax2.twinx()
    obs_dist = df['min_obstacle_dist']
    has_valid_obs = (obs_dist > 0).any()  # -1 表示无数据
    if has_valid_obs:
        valid_mask = obs_dist > 0
        ax3.plot(t[valid_mask], obs_dist[valid_mask], color='orangered', lw=1.5, alpha=0.7, label='最近障碍')
        ax3.axhline(y=1.0, color='red', ls=':', alpha=0.5, label='危险距离 (1m)')
        ax3.set_ylabel('最近障碍距离 (m)', color='orangered')
        ax3.tick_params(axis='y', labelcolor='orangered')
    else:
        ax3.text(0.5, 0.5, '无障碍物距离数据', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12, color='gray', alpha=0.6)
        ax3.set_yticks([])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=8)
    ax2.set_title('距离随时间变化')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, csv_path, '1_trajectory')


# ============================================================================
# Figure 2: 速度分析
# ============================================================================

def plot_fig2_velocity(df, csv_path):
    """速度分析: 世界系 / 机体系 / 速度大小"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('速度分析', fontsize=16, fontweight='bold')
    t = df['time'].values

    # 世界系速度
    ax = axes[0]
    ax.plot(t, df['vx_world'], label='vx', color='tab:red', lw=1.5)
    ax.plot(t, df['vy_world'], label='vy', color='tab:green', lw=1.5)
    ax.plot(t, df['vz_world'], label='vz', color='tab:blue', lw=1.5)
    ax.set_ylabel('世界系速度 (m/s)')
    ax.set_title('世界坐标系速度')
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 机体系速度
    ax = axes[1]
    if 'body_vx' in df.columns:
        ax.plot(t, df['body_vx'], label='body_vx (前)', color='tab:red', lw=1.5)
        ax.plot(t, df['body_vy'], label='body_vy (左)', color='tab:green', lw=1.5)
        ax.plot(t, df['body_vz'], label='body_vz (上)', color='tab:blue', lw=1.5)
    ax.set_ylabel('机体系速度 (m/s)')
    ax.set_title('机体（航向）坐标系速度')
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 速度大小 + 加速度
    ax = axes[2]
    speed_hor = np.sqrt(df['vx_world']**2 + df['vy_world']**2)
    speed_3d = np.sqrt(df['vx_world']**2 + df['vy_world']**2 + df['vz_world']**2)
    ax.plot(t, speed_hor, label='水平速度', color='tab:purple', lw=1.5)
    ax.plot(t, speed_3d, label='三维速度', color='tab:orange', lw=1.5, alpha=0.7)

    # 加速度 (有限差分)
    if len(t) > 2:
        dt = np.diff(t)
        dt[dt == 0] = 0.1
        acc_x = np.diff(df['vx_world'].values) / dt
        acc_y = np.diff(df['vy_world'].values) / dt
        acc_hor = np.sqrt(acc_x**2 + acc_y**2)
        ax_acc = ax.twinx()
        ax_acc.plot(t[1:], acc_hor, label='水平加速度', color='gray', lw=1, alpha=0.4)
        ax_acc.set_ylabel('加速度 (m/s²)', color='gray')
        ax_acc.tick_params(axis='y', labelcolor='gray')

    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('速度与加速度')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, csv_path, '2_velocity')


# ============================================================================
# Figure 3: 策略动作分析
# ============================================================================

def plot_fig3_actions(df, csv_path):
    """策略动作: raw vs cmd + 航向"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('策略动作分析', fontsize=16, fontweight='bold')
    t = df['time'].values

    # 策略原始动作 vs 修正后
    ax = axes[0]
    has_raw = 'action_raw_vx' in df.columns
    if has_raw:
        ax.plot(t, df['action_raw_vx'], '--', color='tab:red', alpha=0.5, lw=1, label='原始_vx')
        ax.plot(t, df['action_raw_vy'], '--', color='tab:green', alpha=0.5, lw=1, label='原始_vy')
        ax.plot(t, df['action_raw_vz'], '--', color='tab:blue', alpha=0.5, lw=1, label='原始_vz')
    ax.plot(t, df['action_vx'], color='tab:red', lw=1.5, label='动作_vx')
    ax.plot(t, df['action_vy'], color='tab:green', lw=1.5, label='动作_vy')
    ax.plot(t, df['action_vz'], color='tab:blue', lw=1.5, label='动作_vz')
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_ylabel('动作值 [-1, 1]')
    ax.set_title('策略输出（虚线=原始，实线=修正后）')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # 实际命令速度
    ax = axes[1]
    ax.plot(t, df['cmd_vx'], color='tab:red', lw=1.5, label='指令_vx')
    ax.plot(t, df['cmd_vy'], color='tab:green', lw=1.5, label='指令_vy')
    ax.plot(t, df['cmd_vz'], color='tab:blue', lw=1.5, label='指令_vz')
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_ylabel('指令速度 (m/s)')
    ax.set_title('最终指令速度（安全过滤后）')
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 航向角
    ax = axes[2]
    if 'current_yaw_deg' in df.columns:
        ax.plot(t, df['current_yaw_deg'], color='tab:blue', lw=1.5, label='当前航向')
        ax.plot(t, df['target_yaw_deg'], color='gold', lw=1.5, ls='--', label='目标航向')
        ax_err = ax.twinx()
        ax_err.fill_between(t, 0, df['yaw_error_deg'], alpha=0.15, color='red')
        ax_err.plot(t, df['yaw_error_deg'], color='red', lw=1, alpha=0.6, label='航向误差')
        ax_err.set_ylabel('航向误差 (°)', color='red')
        ax_err.tick_params(axis='y', labelcolor='red')
    else:
        ax.plot(t, df['yaw_deg'], color='tab:blue', lw=1.5, label='航向角')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('航向角 (°)')
    ax.set_title('航向控制')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, csv_path, '3_actions')


# ============================================================================
# Figure 4: 安全指标
# ============================================================================

def plot_fig4_safety(df, csv_path):
    """安全指标: 障碍距离 + 高度 + 状态饼图"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    fig.suptitle('安全指标', fontsize=16, fontweight='bold')
    t = df['time'].values

    # 障碍物距离
    ax = fig.add_subplot(gs[0, 0])
    obs_dist = df['min_obstacle_dist']
    has_valid_obs = (obs_dist > 0).any()  # -1 表示无数据
    if has_valid_obs:
        valid_mask = obs_dist > 0
        ax.plot(t[valid_mask], obs_dist[valid_mask], color='orangered', lw=1.5)
        ax.axhline(y=1.0, color='red', ls='--', lw=1, alpha=0.7, label='危险距离 (1m)')
        ax.fill_between(t[valid_mask], 0, obs_dist[valid_mask],
                        where=obs_dist[valid_mask] < 1.0,
                        color='red', alpha=0.2, label='危险区域')
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, '无障碍物距离数据\n(传感器未运行)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray', alpha=0.7)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('最近障碍距离 (m)')
    ax.set_title('最近障碍物距离')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 高度
    ax = fig.add_subplot(gs[0, 1])
    if 'height' in df.columns:
        h_col = 'height'
    else:
        h_col = 'z'
    ax.plot(t, df[h_col], color='tab:cyan', lw=1.5, label='高度')
    ax.axhline(y=1.0, color='red', ls='--', lw=1, alpha=0.7, label='最低高度 (1m)')
    if 'height_protection' in df.columns:
        hp_mask = df['height_protection'] > 0
        if hp_mask.any():
            ax.scatter(t[hp_mask], df[h_col][hp_mask], color='red', s=15, zorder=5,
                       label=f'高度保护 ({hp_mask.sum()}次)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('高度 (m)')
    ax.set_title('高度变化')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 状态饼图 (悬停 vs 导航 vs 避障)
    ax = fig.add_subplot(gs[1, 0])
    hover_count = (df['is_hovering'] > 0).sum()
    if has_valid_obs:
        close_obs = ((df['is_hovering'] == 0) & (df['min_obstacle_dist'] > 0) & (df['min_obstacle_dist'] < 1.0)).sum()
    else:
        close_obs = 0
    navigate_count = len(df) - hover_count - close_obs

    labels = ['导航', '悬停', '避障']
    sizes = [navigate_count, hover_count, close_obs]
    colors_pie = ['#4CAF50', '#2196F3', '#FF5722']
    # 过滤掉 0 值
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels_nz, sizes_nz, colors_nz = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('飞行状态分布')

    # 路径效率柱状图
    ax = fig.add_subplot(gs[1, 1])
    direct_dist = np.sqrt(
        (df['target_x'].iloc[-1] - df['x'].iloc[0])**2 +
        (df['target_y'].iloc[-1] - df['y'].iloc[0])**2 +
        (df['target_z'].iloc[-1] - df['z'].iloc[0])**2
    )
    path_len = np.sum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2))
    final_dist = df['dist_to_target'].iloc[-1]

    bars = ax.bar(['直线\n距离', '实际\n路程', '最终\n距目标'],
                  [direct_dist, path_len, final_dist],
                  color=['#4CAF50', '#FF9800', '#F44336'], width=0.5)
    for bar, val in zip(bars, [direct_dist, path_len, final_dist]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}m', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('距离 (m)')
    ax.set_title('路径效率')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return save_fig(fig, csv_path, '4_safety')


# ============================================================================
# Figure 4b: CBF 安全层分析
# ============================================================================

def plot_fig4b_cbf(df, csv_path):
    """CBF Safety Layer: barrier / violation / velocity correction / activation"""
    # 检查是否有 CBF 数据
    if 'cbf_barrier' not in df.columns:
        print("  [跳过] 无 CBF 数据列 (cbf_barrier)")
        return None
    
    # 检查是否全为 0 (CBF 未启用)
    if (df['cbf_active'] == 0).all() and (df['cbf_barrier'] == 0).all():
        print("  [跳过] CBF 未激活 (全部为 0)")
        return None
    
    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.8],
                           hspace=0.35, wspace=0.3)
    fig.suptitle('CBF Safety Layer Analysis', fontsize=16, fontweight='bold')
    t = df['time'].values
    
    # ---- (1) Barrier Value + Min Obstacle Distance ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, df['cbf_barrier'], color='tab:red', lw=1.5, label='Barrier $B(x)$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Barrier Value', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True, alpha=0.3)
    ax.set_title('Barrier Function')
    
    ax2 = ax.twinx()
    if 'cbf_min_dist' in df.columns:
        ax2.plot(t, df['cbf_min_dist'], color='tab:blue', lw=1.2, alpha=0.7, label='Min Obs (m)')
        ax2.set_ylabel('Min Obstacle Dist (m)', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # ---- (2) Constraint Violation ----
    ax = fig.add_subplot(gs[0, 1])
    violation = df['cbf_violation']
    ax.plot(t, violation, color='orangered', lw=1.2)
    ax.fill_between(t, 0, violation, where=violation > 0,
                     color='red', alpha=0.2, label='Violated (>0)')
    ax.fill_between(t, violation, 0, where=violation <= 0,
                     color='green', alpha=0.1, label='Satisfied ($\\leq$0)')
    ax.axhline(y=0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Violation')
    ax.set_title('CBF Constraint Violation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- (3) CBF Velocity Correction ----
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, df['cbf_delta_vx'], color='tab:red', lw=1.5, label='$\\Delta v_x$')
    ax.plot(t, df['cbf_delta_vy'], color='tab:green', lw=1.5, label='$\\Delta v_y$')
    ax.plot(t, df['cbf_delta_vz'], color='tab:blue', lw=1.5, label='$\\Delta v_z$')
    delta_norm = np.sqrt(df['cbf_delta_vx']**2 + df['cbf_delta_vy']**2 + df['cbf_delta_vz']**2)
    ax.fill_between(t, -delta_norm, delta_norm, color='gray', alpha=0.1)
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Correction (m/s)')
    ax.set_title('CBF Velocity Correction ($\\Delta v = v_{cbf} - v_{input}$)')
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ---- (4) Raw Action vs CBF-Corrected Command ----
    ax = fig.add_subplot(gs[2, :])
    has_raw = 'action_raw_vx' in df.columns
    if has_raw:
        ax.plot(t, df['action_raw_vx'], '--', color='tab:red', alpha=0.4, lw=1, label='Raw $v_x$')
        ax.plot(t, df['action_raw_vy'], '--', color='tab:green', alpha=0.4, lw=1, label='Raw $v_y$')
        ax.plot(t, df['action_raw_vz'], '--', color='tab:blue', alpha=0.4, lw=1, label='Raw $v_z$')
    ax.plot(t, df['cmd_vx'], color='tab:red', lw=1.5, label='Final $v_x$')
    ax.plot(t, df['cmd_vy'], color='tab:green', lw=1.5, label='Final $v_y$')
    ax.plot(t, df['cmd_vz'], color='tab:blue', lw=1.5, label='Final $v_z$')
    # CBF active region background
    active = df['cbf_active'].values
    for i in range(len(t) - 1):
        if active[i] > 0.5:
            ax.axvspan(t[i], t[i+1], color='red', alpha=0.08)
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Raw Action vs Final Command (red bg = CBF active)')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- (5) CBF Statistics ----
    ax = fig.add_subplot(gs[3, 0])
    total_frames = len(df)
    active_frames = (df['cbf_active'] > 0.5).sum()
    active_pct = active_frames / total_frames * 100 if total_frames > 0 else 0
    max_violation = df['cbf_violation'].max()
    max_delta = delta_norm.max()
    avg_delta = delta_norm[df['cbf_active'] > 0.5].mean() if active_frames > 0 else 0
    min_obs = df['cbf_min_dist'].min() if 'cbf_min_dist' in df.columns else -1
    
    stats_text = (
        f"CBF Active:  {active_frames}/{total_frames} frames ({active_pct:.1f}%)\n"
        f"Max Violation:  {max_violation:.4f}\n"
        f"Max Correction: {max_delta:.3f} m/s\n"
        f"Avg Correction: {avg_delta:.3f} m/s (when active)\n"
        f"Min Obs Dist:   {min_obs:.2f} m"
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('CBF Statistics')
    
    # ---- (6) CBF Correction Direction (polar) ----
    ax = fig.add_subplot(gs[3, 1], projection='polar')
    if active_frames > 0:
        active_mask = df['cbf_active'] > 0.5
        dvx = df['cbf_delta_vx'][active_mask].values
        dvy = df['cbf_delta_vy'][active_mask].values
        angles = np.arctan2(dvy, dvx)
        magnitudes = np.sqrt(dvx**2 + dvy**2)
        scatter = ax.scatter(angles, magnitudes, c=df['cbf_min_dist'][active_mask],
                           cmap='RdYlGn', s=15, alpha=0.6)
        fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1, label='Obs Dist (m)')
    ax.set_title('CBF Correction Direction\n(color = obs dist)', fontsize=10, pad=15)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, csv_path, '4b_cbf')


# ============================================================================
# Figure 5: 观测空间分析
# ============================================================================

def plot_fig5_observations(df, csv_path):
    """观测空间: 角速度 / 重力投影 / pose_command"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('观测空间分析', fontsize=16, fontweight='bold')
    t = df['time'].values

    # 角速度
    ax = axes[0]
    if 'ang_vel_x' in df.columns:
        ax.plot(t, df['ang_vel_x'], label='ωx (横滚)', color='tab:red', lw=1.2)
        ax.plot(t, df['ang_vel_y'], label='ωy (俯仰)', color='tab:green', lw=1.2)
        ax.plot(t, df['ang_vel_z'], label='ωz (偏航)', color='tab:blue', lw=1.2)
    ax.set_ylabel('角速度 (rad/s)')
    ax.set_title('机体角速度')
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 重力投影
    ax = axes[1]
    if 'gravity_x' in df.columns:
        ax.plot(t, df['gravity_x'], label='gx', color='tab:red', lw=1.2)
        ax.plot(t, df['gravity_y'], label='gy', color='tab:green', lw=1.2)
        ax.plot(t, df['gravity_z'], label='gz', color='tab:blue', lw=1.2)
        ax.axhline(y=-1.0, color='blue', ls=':', alpha=0.3, label='gz=\u22121 (水平)')
    ax.set_ylabel('重力投影')
    ax.set_title('重力投影向量（姿态稳定性）')
    ax.legend(loc='upper right', ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    # pose_command (机体系目标相对位置)
    ax = axes[2]
    if 'pose_cmd_x' in df.columns:
        ax.plot(t, df['pose_cmd_x'], label='cmd_x (前)', color='tab:red', lw=1.5)
        ax.plot(t, df['pose_cmd_y'], label='cmd_y (左)', color='tab:green', lw=1.5)
        ax.plot(t, df['pose_cmd_z'], label='cmd_z (上)', color='tab:blue', lw=1.5)
        # pose_command 模长 = 到目标距离 (机体系)
        cmd_dist = np.sqrt(df['pose_cmd_x']**2 + df['pose_cmd_y']**2 + df['pose_cmd_z']**2)
        ax.plot(t, cmd_dist, label='|cmd| (距离)', color='gray', lw=1, ls='--', alpha=0.6)
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位姿指令 (m)')
    ax.set_title('位姿指令（机体系目标相对位置）')
    ax.legend(loc='upper right', ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, csv_path, '5_observations')


# ============================================================================
# Figure 6: 雷达热力图
# ============================================================================

def plot_fig6_lidar(csv_path):
    """雷达热力图 (从 npz 快照加载)"""
    npz_path = csv_path.replace('.csv', '_lidar.npz')
    if not os.path.exists(npz_path):
        print("  ⊘ 无雷达快照文件，跳过雷达热力图")
        return None

    data = np.load(npz_path)
    times = data['times']
    scans = data['scans']
    channels = int(data['channels'])
    h_points = int(data['horizontal_points'])

    n_snapshots = len(times)
    if n_snapshots == 0:
        print("  ⊘ 雷达快照为空，跳过")
        return None

    # 选取 6 个时间点 (均匀分布)
    n_show = min(6, n_snapshots)
    indices = np.linspace(0, n_snapshots-1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('激光雷达热力图（8垂直 × 35水平）', fontsize=16, fontweight='bold')

    h_angles = np.linspace(-180, 170, h_points)
    v_angles = np.linspace(-7, 52, channels)

    for ax_idx, snap_idx in enumerate(indices):
        ax = axes.flat[ax_idx]
        scan = scans[snap_idx].reshape(channels, h_points)

        im = ax.imshow(scan, aspect='auto', origin='lower',
                       extent=[h_angles[0], h_angles[-1], v_angles[0], v_angles[-1]],
                       cmap='RdYlGn', vmin=0, vmax=5.0)
        ax.set_title(f't = {times[snap_idx]:.1f}s', fontsize=11)
        ax.set_xlabel('水平角 (°)')
        ax.set_ylabel('垂直角 (°)')
        ax.axvline(x=0, color='white', ls='--', lw=0.5, alpha=0.5)

    # 隐藏多余 axes
    for ax_idx in range(n_show, 6):
        axes.flat[ax_idx].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='距离 (m)')
    plt.tight_layout()
    return save_fig(fig, csv_path, '6_lidar')


# ============================================================================
# 动作分布直方图 (bonus)
# ============================================================================

def plot_fig7_distributions(df, csv_path):
    """动作值分布直方图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('动作值分布直方图', fontsize=14, fontweight='bold')

    for i, (col, label, color) in enumerate([
        ('action_vx', '动作_vx', 'tab:red'),
        ('action_vy', '动作_vy', 'tab:green'),
        ('action_vz', '动作_vz', 'tab:blue'),
    ]):
        ax = axes[i]
        ax.hist(df[col], bins=50, color=color, alpha=0.7, edgecolor='black', lw=0.5)
        ax.axvline(x=df[col].mean(), color='black', ls='--', lw=1.5, label=f'均值={df[col].mean():.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('频次')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, csv_path, '7_distributions')


# ============================================================================
# Figure 8: 3D 轨迹图 (带雷达障碍物点云)
# ============================================================================

def _draw_3d_scene(ax, df, csv_path, show_colorbar_ax=None, fig=None):
    """在 3D Axes 上绘制飞行轨迹 + 雷达障碍物 (内部复用函数)
    
    Returns:
        (obs_data, speed_max)  obs_data = None 或 (obs_x, obs_y, obs_z, obs_dist)
    """
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    speed = np.sqrt(df['vx_world']**2 + df['vy_world']**2 + df['vz_world']**2).values
    speed_max = max(speed.max(), 0.1)
    target_x = df['target_x'].iloc[-1]
    target_y = df['target_y'].iloc[-1]
    target_z = df['target_z'].iloc[-1]

    # 飞行轨迹 (按速度着色)
    for i in range(len(x) - 1):
        c = plt.cm.coolwarm(speed[i] / speed_max)
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c, lw=2.0)

    # 起/终/目标
    ax.scatter(x[0], y[0], z[0], color='limegreen', s=120,
               edgecolors='darkgreen', linewidth=2, zorder=10, label='起点', depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], color='tomato', s=120, marker='s',
               edgecolors='darkred', linewidth=2, zorder=10, label='终点', depthshade=False)
    ax.scatter(target_x, target_y, target_z, color='gold', s=250, marker='*',
               edgecolors='darkorange', linewidth=1.5, zorder=10, label='目标', depthshade=False)
    ax.plot([x[0], target_x], [y[0], target_y], [z[0], target_z],
            '--', color='gray', alpha=0.4, lw=1)

    # 障碍物: 优先用场景 JSON, 回退雷达点云 (只画轨迹附近的)
    obs_data = None
    scene_obs, _ = load_scene_obstacles(csv_path)
    if scene_obs:
        margin_obs = max(10, max(x.max()-x.min(), y.max()-y.min()) * 0.3)
        nearby_obs = [ob for ob in scene_obs
                      if (x.min()-margin_obs <= ob['x'] <= x.max()+margin_obs and
                          y.min()-margin_obs <= ob['y'] <= y.max()+margin_obs)]
        # 障碍物太多时 (如森林 900+ 棵树), 只保留距轨迹最近的 MAX_OBS_3D 个
        MAX_OBS_3D = 120
        if len(nearby_obs) > MAX_OBS_3D:
            traj_pts = np.column_stack([x, y])
            for ob in nearby_obs:
                dists = np.sqrt((traj_pts[:, 0] - ob['x'])**2 +
                                (traj_pts[:, 1] - ob['y'])**2)
                ob['_dist_to_traj'] = dists.min()
            nearby_obs.sort(key=lambda o: o['_dist_to_traj'])
            nearby_obs = nearby_obs[:MAX_OBS_3D]
        obs_alpha = 0.15 if len(nearby_obs) > 60 else 0.2
        draw_obstacles_3d(ax, nearby_obs, alpha=obs_alpha, color='gray')
    else:
        # 回退: 优先用 _pcl.npz (无 bin 量化), 否则用 _lidar.npz
        pcl_pts_3d = load_pcl_points(csv_path)
        if pcl_pts_3d is not None:
            ox, oy, oz = pcl_pts_3d[:, 0], pcl_pts_3d[:, 1], pcl_pts_3d[:, 2]
            grid_res_3d = 0.08
            gx = np.round(ox / grid_res_3d).astype(int)
            gy = np.round(oy / grid_res_3d).astype(int)
            gz = np.round(oz / grid_res_3d).astype(int)
            seen3 = {}
            for j in range(len(ox)):
                key = (gx[j], gy[j], gz[j])
                if key not in seen3:
                    seen3[key] = (ox[j], oy[j], oz[j])
            if seen3:
                v3 = list(seen3.values())
                ax.scatter([p[0] for p in v3], [p[1] for p in v3], [p[2] for p in v3],
                           s=1.5, c='dimgray', alpha=0.2, edgecolors='none',
                           depthshade=False)
        else:
            npz_path = csv_path.replace('.csv', '_lidar.npz')
            if os.path.exists(npz_path):
                result = lidar_to_world_xyz(npz_path)
                if result is not None:
                    obs_x, obs_y, obs_z, obs_dist = result
                    ax.scatter(obs_x, obs_y, obs_z, c=obs_dist, cmap='YlOrRd_r',
                               s=1.5, alpha=0.2, vmin=0, vmax=5.0, depthshade=False)
                    obs_data = result

    # 轨迹在地面的投影 (阴影)
    z_floor = min(z.min(), target_z) - 0.5
    ax.plot(x, y, z_floor * np.ones_like(x), color='gray', lw=0.8, alpha=0.25)

    ax.set_xlabel('X (m)', fontsize=9, labelpad=5)
    ax.set_ylabel('Y (m)', fontsize=9, labelpad=5)
    ax.set_zlabel('Z (m)', fontsize=9, labelpad=3)
    ax.tick_params(labelsize=7)

    return obs_data, speed_max


def plot_fig8_trajectory_3d(df, csv_path):
    """3D 轨迹图: 四视角 飞行轨迹 + 场景障碍物"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _obs, _sname = load_scene_obstacles(csv_path)
    has_obstacles = (_obs is not None
                     or os.path.exists(csv_path.replace('.csv', '_lidar.npz')))

    # 根据轨迹方向自动计算最佳方位角
    x = df['x'].values
    y = df['y'].values
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    traj_azim = np.degrees(np.arctan2(dy, dx))  # 轨迹方向角

    # 四个视角: 侧前方、侧后方、俯视、正侧面
    views = [
        {'elev': 35, 'azim': traj_azim - 135, 'title': '侧前方'},
        {'elev': 35, 'azim': traj_azim + 45,  'title': '侧后方'},
        {'elev': 75, 'azim': traj_azim - 90,  'title': '俯视'},
        {'elev': 10, 'azim': traj_azim - 90,  'title': '侧面'},
    ]

    fig = plt.figure(figsize=(18, 14))
    title = '三维飞行轨迹（多视角）'
    if has_obstacles:
        title += ' + 场景障碍物'
    if _sname:
        cn = SCENE_NAMES_CN.get(_sname, _sname)
        title += f' — {cn}场景'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    target_x = df['target_x'].iloc[-1]
    target_y = df['target_y'].iloc[-1]
    target_z = df['target_z'].iloc[-1]
    z = df['z'].values

    for vi, view in enumerate(views):
        ax = fig.add_subplot(2, 2, vi + 1, projection='3d')
        obs_data, speed_max = _draw_3d_scene(ax, df, csv_path)
        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_title(view['title'], fontsize=12, fontweight='bold', pad=10)

        if vi == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)

    # 速度色标 (整张图共用)
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(0, speed_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('速度 (m/s)', fontsize=10)

    # 障碍距离色标 (仅雷达点云模式时显示)
    if obs_data is not None:
        sm2 = plt.cm.ScalarMappable(cmap='YlOrRd_r',
                                     norm=plt.Normalize(0, 5.0))
        sm2.set_array([])
        cbar_ax2 = fig.add_axes([0.92, 0.1, 0.015, 0.35])
        cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
        cbar2.set_label('障碍距离 (m)', fontsize=10)

    # 统计信息
    direct_dist = np.sqrt((target_x-x[0])**2 + (target_y-y[0])**2 + (target_z-z[0])**2)
    path_len = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    final_dist = df['dist_to_target'].iloc[-1]
    eff = direct_dist / max(path_len, 0.01) * 100
    info = (f'直线: {direct_dist:.1f}m | 路程: {path_len:.1f}m | '
            f'效率: {eff:.0f}% | 终距: {final_dist:.2f}m | '
            f'高度: {z.min():.1f}~{z.max():.1f}m')
    fig.text(0.5, 0.01, info, fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.85))

    plt.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.04,
                        wspace=0.1, hspace=0.12)
    return save_fig(fig, csv_path, '8_trajectory_3d')


# ============================================================================
# Fig 9: 雷达点云轨迹图 (2D + 3D)
# ============================================================================

def _lidar_frame_to_fan(scan_1d, channels, h_points, px, py, yaw_rad,
                        lidar_range=5.0, override_yaw_rad=None):
    """将单帧雷达扫描转换为完整扇形数据 (包含未命中射线)。

    Args:
        override_yaw_rad: 若提供，用此角度替代 yaw_rad 做旋转 (用于对齐飞行方向)

    Returns:
        fan_pts: list of (wx, wy) — 按角度排序的扇形边缘点
        hit_pts: list of (wx, wy, dist) — 仅命中障碍物的点
    """
    scan = scan_1d.reshape(channels, h_points)

    # 先用真实 yaw 算出机体系距离，再用目标角度旋转到世界系
    # 如果 override_yaw_rad 被指定，扇形将对齐到该方向而非真实航向
    use_yaw = override_yaw_rad if override_yaw_rad is not None else yaw_rad
    cos_yaw = np.cos(use_yaw)
    sin_yaw = np.sin(use_yaw)

    fan_pts = []
    hit_pts = []

    for hi in range(h_points):
        min_d = lidar_range
        for vi in range(channels):
            v_angle_deg = -7.0 + vi * (59.0 / max(channels - 1, 1))
            if abs(v_angle_deg) < 25:
                if scan[vi, hi] < min_d:
                    min_d = scan[vi, hi]

        h_angle = np.radians(-180.0 + hi * 10.0)
        ray_bx = np.cos(h_angle)
        ray_by = np.sin(h_angle)
        wx = px + min_d * (cos_yaw * ray_bx - sin_yaw * ray_by)
        wy = py + min_d * (sin_yaw * ray_bx + cos_yaw * ray_by)
        fan_pts.append((wx, wy))

        if min_d < lidar_range - 0.01:
            hit_pts.append((wx, wy, min_d))

    return fan_pts, hit_pts


def _lidar_frame_to_3d(scan_1d, channels, h_points, px, py, pz, yaw_rad, lidar_range=5.0):
    """将单帧雷达扫描转换为 3D 世界坐标点列表"""
    scan = scan_1d.reshape(channels, h_points)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    pts = []
    for vi in range(channels):
        v_angle_deg = -7.0 + vi * (59.0 / max(channels - 1, 1))
        if abs(v_angle_deg) > 35:
            continue
        v_angle = np.radians(v_angle_deg)
        cos_v, sin_v = np.cos(v_angle), np.sin(v_angle)
        for hi in range(h_points):
            dist = scan[vi, hi]
            if dist >= lidar_range - 0.01:
                continue
            h_angle = np.radians(-180.0 + hi * 10.0)
            rbx = cos_v * np.cos(h_angle)
            rby = cos_v * np.sin(h_angle)
            rbz = sin_v
            wx = px + dist * (cos_yaw * rbx - sin_yaw * rby)
            wy = py + dist * (sin_yaw * rbx + cos_yaw * rby)
            wz = pz + dist * rbz
            pts.append((wx, wy, wz, dist))
    return pts


def _prepare_pcl_data(csv_path, df):
    """准备雷达点云数据 (2D/3D共用), 返回去重后的点云坐标"""
    npz_path = csv_path.replace('.csv', '_lidar.npz')
    pcl_pts = load_pcl_points(csv_path)
    has_pcl = pcl_pts is not None
    has_lidar = os.path.exists(npz_path)

    if not has_pcl and not has_lidar:
        return None

    # 2D 点云
    fx2d, fy2d = np.array([]), np.array([])
    # 3D 点云
    fx3d, fy3d, fz3d = np.array([]), np.array([]), np.array([])

    if has_pcl:
        ox, oy = pcl_pts[:, 0], pcl_pts[:, 1]
        # 点数多时适度加大栅格 (如森林 35k+ 点)
        n_raw = len(ox)
        grid_res = 0.18 if n_raw > 20000 else (0.16 if n_raw > 10000 else 0.15)
        gx = np.round(ox / grid_res).astype(int)
        gy = np.round(oy / grid_res).astype(int)
        seen = {}
        for j in range(len(ox)):
            key = (gx[j], gy[j])
            if key not in seen:
                seen[key] = (ox[j], oy[j])
        vals = list(seen.values())
        fx2d = np.array([v[0] for v in vals])
        fy2d = np.array([v[1] for v in vals])

        ox3, oy3, oz3 = pcl_pts[:, 0], pcl_pts[:, 1], pcl_pts[:, 2]
        grid_res_3d = 0.10 if n_raw > 20000 else (0.09 if n_raw > 10000 else 0.08)
        gx3 = np.round(ox3 / grid_res_3d).astype(int)
        gy3 = np.round(oy3 / grid_res_3d).astype(int)
        gz3 = np.round(oz3 / grid_res_3d).astype(int)
        seen3 = {}
        for j in range(len(ox3)):
            key = (gx3[j], gy3[j], gz3[j])
            if key not in seen3:
                seen3[key] = (ox3[j], oy3[j], oz3[j])
        v3 = list(seen3.values())
        if v3:
            fx3d = np.array([p[0] for p in v3])
            fy3d = np.array([p[1] for p in v3])
            fz3d = np.array([p[2] for p in v3])

    elif has_lidar:
        lidar_data = np.load(npz_path)
        times = lidar_data['times']
        scans = lidar_data['scans']
        l_pos_x = lidar_data['pos_x']
        l_pos_y = lidar_data['pos_y']
        l_pos_z = lidar_data['pos_z']
        l_yaw = lidar_data['yaw_deg']
        channels = int(lidar_data['channels'])
        h_points = int(lidar_data['horizontal_points'])

        all_ox, all_oy = [], []
        all_3x, all_3y, all_3z = [], [], []
        for i in range(len(times)):
            scan = scans[i].reshape(channels, h_points)
            px, py, pz = l_pos_x[i], l_pos_y[i], l_pos_z[i]
            yaw_r = np.radians(l_yaw[i])
            cos_yaw, sin_yaw = np.cos(yaw_r), np.sin(yaw_r)
            for vi in range(channels):
                v_angle_deg = -7.0 + vi * (59.0 / max(channels - 1, 1))
                v_angle = np.radians(v_angle_deg)
                cos_v, sin_v = np.cos(v_angle), np.sin(v_angle)
                for hi in range(h_points):
                    dist = scan[vi, hi]
                    if dist >= 5.0 - 0.01:
                        continue
                    h_angle = np.radians(-180.0 + hi * 10.0)
                    rbx = cos_v * np.cos(h_angle)
                    rby = cos_v * np.sin(h_angle)
                    rbz = sin_v
                    wx = px + dist * (cos_yaw * rbx - sin_yaw * rby)
                    wy = py + dist * (sin_yaw * rbx + cos_yaw * rby)
                    wz = pz + dist * rbz
                    all_ox.append(wx); all_oy.append(wy)
                    all_3x.append(wx); all_3y.append(wy); all_3z.append(wz)

        # 2D 栅格去重
        if all_ox:
            ox, oy = np.array(all_ox), np.array(all_oy)
            gx = np.round(ox / 0.15).astype(int)
            gy = np.round(oy / 0.15).astype(int)
            seen = {}
            for j in range(len(ox)):
                key = (gx[j], gy[j])
                if key not in seen:
                    seen[key] = (ox[j], oy[j])
            vals = list(seen.values())
            fx2d = np.array([v[0] for v in vals])
            fy2d = np.array([v[1] for v in vals])
        # 3D 栅格去重
        if all_3x:
            ax3 = np.array(all_3x); ay3 = np.array(all_3y); az3 = np.array(all_3z)
            gx3 = np.round(ax3 / 0.08).astype(int)
            gy3 = np.round(ay3 / 0.08).astype(int)
            gz3 = np.round(az3 / 0.08).astype(int)
            seen3 = {}
            for j in range(len(ax3)):
                key = (gx3[j], gy3[j], gz3[j])
                if key not in seen3:
                    seen3[key] = (ax3[j], ay3[j], az3[j])
            v3 = list(seen3.values())
            if v3:
                fx3d = np.array([p[0] for p in v3])
                fy3d = np.array([p[1] for p in v3])
                fz3d = np.array([p[2] for p in v3])

    return {
        'fx2d': fx2d, 'fy2d': fy2d,
        'fx3d': fx3d, 'fy3d': fy3d, 'fz3d': fz3d,
    }


def plot_fig9_trajectory_pcl(df, csv_path, extra_trajectories=None):
    """雷达点云轨迹图 (2D): 俯视轨迹 + 雷达散点"""
    pcl_data = _prepare_pcl_data(csv_path, df)
    if pcl_data is None:
        print("  ⊘ 无雷达/点云数据，跳过雷达点云轨迹图")
        return

    x = df['x'].values
    y = df['y'].values
    speed = np.sqrt(df['vx_world']**2 + df['vy_world']**2 + df['vz_world']**2)
    target_x = df['target_x'].iloc[-1]
    target_y = df['target_y'].iloc[-1]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    all_x = np.concatenate([x, [target_x]])
    all_y = np.concatenate([y, [target_y]])
    margin = max(5, (all_x.max() - all_x.min()) * 0.2)

    # 雷达散点 — 按高度着色 (如有3D数据)
    fx, fy = pcl_data['fx2d'], pcl_data['fy2d']
    n_scan_pts = len(fx)
    if n_scan_pts > 0:
        # 尝试用 3D 点云数据 (有高度信息, 可着色)
        pcl_pts_raw = load_pcl_points(csv_path)
        if pcl_pts_raw is not None and len(pcl_pts_raw) > 0:
            # 2D 栅格去重 (保留高度信息)
            pox, poy, poz = pcl_pts_raw[:, 0], pcl_pts_raw[:, 1], pcl_pts_raw[:, 2]
            grid_res_2d = 0.18 if len(pox) > 20000 else 0.15
            pgx = np.round(pox / grid_res_2d).astype(int)
            pgy = np.round(poy / grid_res_2d).astype(int)
            seen_h = {}
            for j in range(len(pox)):
                key = (pgx[j], pgy[j])
                # 每个 2D 格子保留最高点 (代表树冠高度)
                if key not in seen_h or poz[j] > seen_h[key][2]:
                    seen_h[key] = (pox[j], poy[j], poz[j])
            if seen_h:
                vv = np.array(list(seen_h.values()))
                fx, fy, fz_2d = vv[:, 0], vv[:, 1], vv[:, 2]
                n_scan_pts = len(fx)
                from matplotlib.colors import LinearSegmentedColormap
                tree_cmap_2d = LinearSegmentedColormap.from_list('tree2d', [
                    '#5a3a10', '#3a6a2a', '#7acc50'])
                z_lo_2d = max(np.percentile(fz_2d, 5), 0)
                z_hi_2d = np.percentile(fz_2d, 95)
                z_range_2d = max(z_hi_2d - z_lo_2d, 1.0)
                z_norm_2d = np.clip((fz_2d - z_lo_2d) / z_range_2d, 0, 1)
                ax1.scatter(fx, fy, s=6, c=z_norm_2d, cmap=tree_cmap_2d,
                            alpha=0.65, zorder=2, edgecolors='none',
                            label=f'障碍物 ({n_scan_pts}点, 按高度着色)')
            else:
                ax1.scatter(fx, fy, s=4, c='black', alpha=0.6, zorder=2,
                            edgecolors='none', label=f'雷达点云 ({n_scan_pts}点)')
        else:
            ax1.scatter(fx, fy, s=4, c='black', alpha=0.6, zorder=2,
                        edgecolors='none', label=f'雷达点云 ({n_scan_pts}点)')

    # 轨迹
    ax1.plot(x, y, '-', color='#333333', lw=2.5, alpha=0.6, zorder=3)
    sc1 = ax1.scatter(x, y, c=speed, cmap='coolwarm', s=12, zorder=5, alpha=0.9)

    # 叠加其他方法轨迹
    _extra_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if extra_trajectories:
        for ei, et in enumerate(extra_trajectories):
            edf = load_flight_data(et['csv_path'])
            ec = et.get('color', _extra_colors[ei % len(_extra_colors)])
            ename = et.get('name', f'method_{ei}')
            ax1.plot(edf['x'].values, edf['y'].values, '-', color=ec, lw=2.0,
                     alpha=0.75, zorder=4, label=ename)
            ax1.plot(edf['x'].iloc[0], edf['y'].iloc[0], 'o', color=ec, ms=6,
                     markeredgecolor='k', markeredgewidth=0.5, zorder=6)
            ax1.plot(edf['x'].iloc[-1], edf['y'].iloc[-1], 's', color=ec, ms=6,
                     markeredgecolor='k', markeredgewidth=0.5, zorder=6)
            all_x = np.concatenate([all_x, edf['x'].values])
            all_y = np.concatenate([all_y, edf['y'].values])
        margin = max(5, (all_x.max() - all_x.min()) * 0.15)

    # 起终目标
    ax1.plot(x[0], y[0], 'o', color='limegreen', ms=12,
             markeredgecolor='darkgreen', markeredgewidth=2, zorder=7, label='起点')
    ax1.plot(x[-1], y[-1], 's', color='tomato', ms=12,
             markeredgecolor='darkred', markeredgewidth=2, zorder=7, label='终点')
    ax1.plot(target_x, target_y, '*', color='gold', ms=20,
             markeredgecolor='darkorange', markeredgewidth=1.5, zorder=7, label='目标')

    ax1.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax1.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'俯视轨迹 + 雷达点云 ({n_scan_pts} 障碍点)', fontsize=13)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='速度 (m/s)', shrink=0.7)

    plt.tight_layout()
    save_fig(fig, csv_path, '9_trajectory_pcl_2d')


def _prepare_pcl_3d_forest(csv_path, df, grid_res=0.35, max_pts=6000, z_min=None):
    """为 3D 点云图准备稀疏、按高度着色的点云数据
    
    与 _prepare_pcl_data 不同:
      - 更大的栅格去重 (0.35m vs 0.08m)
      - 限制最大点数 (保留离轨迹近的)
      - 过滤地面点
      - 返回高度信息用于着色
    
    Returns:
        (ox, oy, oz) 或 None  —— 已去重、过滤、限数
    """
    pcl_pts = load_pcl_points(csv_path, z_min=z_min)
    if pcl_pts is None or len(pcl_pts) == 0:
        # fallback: 尝试 _lidar.npz
        npz_path = csv_path.replace('.csv', '_lidar.npz')
        if os.path.exists(npz_path):
            result = lidar_to_world_xyz(npz_path)
            if result is not None:
                ox, oy, oz, _ = result
                pcl_pts = np.column_stack([ox, oy, oz])
            else:
                return None
        else:
            return None

    ox, oy, oz = pcl_pts[:, 0], pcl_pts[:, 1], pcl_pts[:, 2]

    # 3D 栅格去重 (大栅格减密度)
    gx = np.round(ox / grid_res).astype(int)
    gy = np.round(oy / grid_res).astype(int)
    gz = np.round(oz / grid_res).astype(int)
    seen = {}
    for j in range(len(ox)):
        key = (gx[j], gy[j], gz[j])
        if key not in seen:
            seen[key] = (ox[j], oy[j], oz[j])
    if not seen:
        return None
    vals = np.array(list(seen.values()))
    ox, oy, oz = vals[:, 0], vals[:, 1], vals[:, 2]

    # 如果点数仍然太多，保留离轨迹最近的
    if len(ox) > max_pts:
        traj_xy = np.column_stack([df['x'].values, df['y'].values])
        pt_xy = np.column_stack([ox, oy])
        # 对每个点云点，计算到轨迹最近点的距离
        from scipy.spatial import cKDTree
        tree = cKDTree(traj_xy)
        dists, _ = tree.query(pt_xy)
        # 保留最近的 max_pts 个
        keep_idx = np.argsort(dists)[:max_pts]
        ox, oy, oz = ox[keep_idx], oy[keep_idx], oz[keep_idx]

    return ox, oy, oz


def plot_fig10_trajectory_pcl_3d(df, csv_path, extra_trajectories=None):
    """雷达点云轨迹图 (3D): 按高度着色，清晰显示树木结构"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 准备稀疏点云 (大栅格 + 限点数)
    pcl_result = _prepare_pcl_3d_forest(csv_path, df, grid_res=0.35, max_pts=6000)
    if pcl_result is None:
        # 回退到原始方法
        pcl_data = _prepare_pcl_data(csv_path, df)
        if pcl_data is None:
            print("  ⊘ 无雷达/点云数据，跳过 3D 雷达点云图")
            return
        fx3, fy3, fz3 = pcl_data['fx3d'], pcl_data['fy3d'], pcl_data['fz3d']
    else:
        fx3, fy3, fz3 = pcl_result

    x = df['x'].values; y = df['y'].values; z = df['z'].values
    speed = np.sqrt(df['vx_world']**2 + df['vy_world']**2 + df['vz_world']**2)
    target_x = df['target_x'].iloc[-1]
    target_y = df['target_y'].iloc[-1]
    target_z = df['target_z'].iloc[-1]

    fig = plt.figure(figsize=(12, 9))
    ax2 = fig.add_subplot(111, projection='3d')

    # 3D 雷达散点 — 按高度着色 (低→深棕/高→翠绿, 模拟树干→树冠)
    if len(fx3) > 0:
        z_lo = max(np.percentile(fz3, 5), 0)
        z_hi = np.percentile(fz3, 95)
        z_range = max(z_hi - z_lo, 1.0)

        # 自定义 colormap: 棕色(树干) → 深绿 → 亮绿(树冠)
        from matplotlib.colors import LinearSegmentedColormap
        tree_cmap = LinearSegmentedColormap.from_list('tree', [
            '#5a3a10',   # 深棕 (树干底部)
            '#3a6a2a',   # 深绿 (中部)
            '#7acc50',   # 亮绿 (树冠)
        ])

        z_norm = np.clip((fz3 - z_lo) / z_range, 0, 1)
        # 点越高 alpha 越低 (树冠半透明, 不遮挡轨迹)
        alpha_arr = 0.7 - 0.3 * z_norm  # 0.7(低) → 0.4(高)
        # matplotlib 3D scatter 不支持逐点 alpha, 用固定 alpha + 颜色深浅模拟
        ax2.scatter(fx3, fy3, fz3, s=4, c=z_norm, cmap=tree_cmap,
                    alpha=0.5, edgecolors='none', depthshade=True,
                    label=f'障碍物 ({len(fx3)}点, 按高度着色)')

        # 高度色标
        sm = plt.cm.ScalarMappable(cmap=tree_cmap,
                                    norm=plt.Normalize(z_lo, z_hi))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, shrink=0.5, pad=0.08, aspect=20)
        cbar.set_label('高度 (m)', fontsize=9)

    # 3D 轨迹
    speed_max = max(speed.max(), 0.1)
    for i in range(len(x) - 1):
        ax2.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                 color=plt.cm.coolwarm(speed.iloc[i] / speed_max),
                 lw=2.5, alpha=0.9)
    ax2.plot(x, y, np.zeros_like(z), color='gray', alpha=0.15, lw=0.5)

    # 叠加其他方法轨迹
    _extra_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if extra_trajectories:
        for ei, et in enumerate(extra_trajectories):
            edf = load_flight_data(et['csv_path'])
            ec = et.get('color', _extra_colors[ei % len(_extra_colors)])
            ename = et.get('name', f'method_{ei}')
            ax2.plot(edf['x'].values, edf['y'].values, edf['z'].values,
                     color=ec, lw=1.8, alpha=0.7, label=ename)
            ax2.plot(edf['x'].values, edf['y'].values, np.zeros(len(edf)),
                     color=ec, alpha=0.1, lw=0.5)

    # 起终目标
    ax2.scatter(x[0], y[0], z[0], c='limegreen', s=100,
                edgecolors='darkgreen', linewidths=2, zorder=10, label='起点',
                depthshade=False)
    ax2.scatter(x[-1], y[-1], z[-1], c='tomato', s=100, marker='s',
                edgecolors='darkred', linewidths=2, zorder=10, label='终点',
                depthshade=False)
    ax2.scatter(target_x, target_y, target_z, c='gold', s=180, marker='*',
                edgecolors='darkorange', linewidths=1.5, zorder=10, label='目标',
                depthshade=False)

    # XY 范围限制在轨迹附近
    all_tx = np.concatenate([x, [target_x]])
    all_ty = np.concatenate([y, [target_y]])
    if extra_trajectories:
        for et in extra_trajectories:
            edf = load_flight_data(et['csv_path'])
            all_tx = np.concatenate([all_tx, edf['x'].values])
            all_ty = np.concatenate([all_ty, edf['y'].values])
    margin_3d = max(5, max(all_tx.max()-all_tx.min(), all_ty.max()-all_ty.min()) * 0.15)
    ax2.set_xlim(all_tx.min() - margin_3d, all_tx.max() + margin_3d)
    ax2.set_ylim(all_ty.min() - margin_3d, all_ty.max() + margin_3d)

    traj_dx = x[-1] - x[0]; traj_dy = y[-1] - y[0]
    azim = np.degrees(np.arctan2(traj_dy, traj_dx)) - 135
    ax2.view_init(elev=30, azim=azim)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
    ax2.set_title('3D轨迹 + 障碍物点云（按高度着色）', fontsize=13)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.8)

    plt.tight_layout()
    save_fig(fig, csv_path, '10_trajectory_pcl_3d')


def plot_fig11_interactive_3d(df, csv_path, extra_trajectories=None):
    """雷达点云轨迹图 (交互式 3D HTML), 可旋转/缩放/悬停"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  ⊘ 未安装 plotly，跳过交互式 3D 图 (pip install plotly)")
        return

    pcl_data = _prepare_pcl_data(csv_path, df)
    x = df['x'].values; y = df['y'].values; z = df['z'].values
    speed = np.sqrt(df['vx_world']**2 + df['vy_world']**2 + df['vz_world']**2).values
    target_x = df['target_x'].iloc[-1]
    target_y = df['target_y'].iloc[-1]
    target_z = df['target_z'].iloc[-1]

    traces = []

    # 雷达点云
    if pcl_data is not None and len(pcl_data['fx3d']) > 0:
        fx3, fy3, fz3 = pcl_data['fx3d'], pcl_data['fy3d'], pcl_data['fz3d']
        traces.append(go.Scatter3d(
            x=fx3, y=fy3, z=fz3, mode='markers',
            marker=dict(size=1.5, color='gray', opacity=0.35),
            name=f'雷达点云 ({len(fx3)}点)', hoverinfo='skip'
        ))

    # 主轨迹 (按速度着色)
    traces.append(go.Scatter3d(
        x=x, y=y, z=z, mode='lines+markers',
        line=dict(color=speed, colorscale='RdBu_r', width=4,
                  colorbar=dict(title='速度(m/s)', x=1.02)),
        marker=dict(size=2, color=speed, colorscale='RdBu_r'),
        text=[f'时间: {df["time"].iloc[i]:.1f}s<br>速度: {speed[i]:.2f}m/s<br>'
              f'位置: ({x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f})'
              for i in range(len(x))],
        hoverinfo='text', name='轨迹 (policy)'
    ))

    # 地面投影
    traces.append(go.Scatter3d(
        x=x, y=y, z=np.zeros_like(z), mode='lines',
        line=dict(color='lightgray', width=1), opacity=0.3,
        showlegend=False, hoverinfo='skip'
    ))

    # 叠加其他方法
    _extra_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if extra_trajectories:
        for ei, et in enumerate(extra_trajectories):
            edf = load_flight_data(et['csv_path'])
            ec = et.get('color', _extra_colors[ei % len(_extra_colors)])
            ename = et.get('name', f'method_{ei}')
            traces.append(go.Scatter3d(
                x=edf['x'].values, y=edf['y'].values, z=edf['z'].values,
                mode='lines', line=dict(color=ec, width=3),
                name=ename, opacity=0.8
            ))

    # 起终目标标记
    traces.append(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]], mode='markers',
        marker=dict(size=8, color='limegreen', symbol='circle',
                    line=dict(color='darkgreen', width=2)),
        name='起点'
    ))
    traces.append(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
        marker=dict(size=8, color='tomato', symbol='square',
                    line=dict(color='darkred', width=2)),
        name='终点'
    ))
    traces.append(go.Scatter3d(
        x=[target_x], y=[target_y], z=[target_z], mode='markers',
        marker=dict(size=10, color='gold', symbol='diamond',
                    line=dict(color='darkorange', width=2)),
        name='目标'
    ))

    # XY 范围限制在轨迹附近
    all_tx = np.concatenate([x, [target_x]])
    all_ty = np.concatenate([y, [target_y]])
    if extra_trajectories:
        for et in extra_trajectories:
            edf = load_flight_data(et['csv_path'])
            all_tx = np.concatenate([all_tx, edf['x'].values])
            all_ty = np.concatenate([all_ty, edf['y'].values])
    m3d = max(5, max(all_tx.max()-all_tx.min(), all_ty.max()-all_ty.min()) * 0.15)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text='3D 交互式轨迹 + 雷达点云 (可旋转/缩放/悬停)',
                   font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='X (m)', range=[float(all_tx.min()-m3d), float(all_tx.max()+m3d)]),
            yaxis=dict(title='Y (m)', range=[float(all_ty.min()-m3d), float(all_ty.max()+m3d)]),
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1200, height=800,
    )

    out_dir = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    html_path = os.path.join(out_dir, f'{base}_11_3d_interactive.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"  ✓ 已保存交互式3D: {os.path.basename(html_path)} (浏览器打开可旋转)")
    return html_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='UAV 飞行数据综合分析 & 绘图')
    parser.add_argument('csv', nargs='?', default=None, help='飞行日志 CSV 文件路径')
    parser.add_argument('--bg', type=str, default=None,
                        help='场景俯视背景图路径 (叠加到轨迹图上)')
    parser.add_argument('--extra', nargs='*', default=None,
                        help='叠加其他方法轨迹到雷达点云图 (格式: name:csv_path ...)')
    parser.add_argument('--scene', type=str, default=None,
                        help='手动指定场景名 (pillar/factory/forest), 用于加载对应的 scene_obstacles JSON')
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_latest_log()

    # 如果指定了 --scene, 将对应 JSON 复制/链接到 CSV 目录
    if args.scene:
        csv_dir = os.path.dirname(csv_path)
        target_json = os.path.join(csv_dir, 'scene_obstacles.json')
        if not os.path.exists(target_json):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            src_json = os.path.join(script_dir, 'flight_logs', f'scene_obstacles_{args.scene}.json')
            if os.path.exists(src_json):
                import shutil
                shutil.copy2(src_json, target_json)
                print(f"已加载场景: {args.scene} → {target_json}")

    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        sys.exit(1)

    df = load_flight_data(csv_path)
    if len(df) < 2:
        print("数据点太少 (< 2)，无法绘图!")
        sys.exit(1)

    print(f"\n生成图表...")

    # 生成所有图表
    plot_fig1_trajectory(df, csv_path, bg_path=args.bg)  # 轨迹 + 距离 + 背景
    plot_fig2_velocity(df, csv_path)         # 速度分析
    plot_fig3_actions(df, csv_path)          # 策略动作
    plot_fig4_safety(df, csv_path)           # 安全指标
    plot_fig4b_cbf(df, csv_path)            # CBF 安全层分析
    plot_fig5_observations(df, csv_path)     # 观测空间
    plot_fig6_lidar(csv_path)               # 雷达热力图
    plot_fig7_distributions(df, csv_path)    # 动作分布
    plot_fig8_trajectory_3d(df, csv_path)   # 3D轨迹 + 场景障碍物

    # 解析 --extra 参数
    extra_trajs = None
    if args.extra:
        extra_trajs = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, spec in enumerate(args.extra):
            if ':' in spec:
                name, path = spec.split(':', 1)
            else:
                name = os.path.basename(os.path.dirname(spec))
                path = spec
            extra_trajs.append({'name': name, 'csv_path': path,
                                'color': colors[i % len(colors)]})
    plot_fig9_trajectory_pcl(df, csv_path, extra_trajectories=extra_trajs)     # 2D 雷达点云
    plot_fig10_trajectory_pcl_3d(df, csv_path, extra_trajectories=extra_trajs) # 3D 雷达点云 (静态 PNG)
    plot_fig11_interactive_3d(df, csv_path, extra_trajectories=extra_trajs)    # 3D 交互式 HTML

    print(f"\n全部图表已生成! 查看目录: {os.path.dirname(csv_path)}")


if __name__ == '__main__':
    main()
