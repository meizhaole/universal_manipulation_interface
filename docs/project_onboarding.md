# 项目架构与对接指南

本文面向第一次接手项目的开发者，说明两个代码仓库的分工、训练到仿真的数据流，以及当前完成范围。提交记录快照截至 2026-09-30。

![UMI 项目示意图](../assets/umi_teaser.png)

## 项目目标与当前进度

项目希望打通示教数据处理、策略训练、推理和机械臂部署。当前已有 UMI 训练与离线评估流程，也已把官方 UMI 权重接入 ReBot RS 的 Gazebo 仿真回放。

本机训练模型目前仍欠训练：同一批留出 episode 上，报告记录的位置误差为 7.83 cm；官方权重为 0.52 cm。Gazebo 一键回放默认使用官方权重和官方数据集。真实 ReBot Arm 驱动与实时 UMI 策略尚未接成一个实机策略部署链路。详细评测见[训练与离线推理报告](offline_eval_report.md)。

## 工作区与仓库分工

工作区由两个独立 Git 仓库组成，顶层目录本身不是 Git 仓库：

~~~text
umi/
├── universal_manipulation_interface/  # UMI：数据处理、训练、离线推理
└── reBotArmController_ROS2/           # ReBot：ROS 2 驱动、MoveIt、仿真与回放桥接
~~~

Gazebo 一键脚本按兄弟目录查找 UMI 仓库，保留上述目录名和并列关系。若路径不同，直接启动 gazebo_rs_sim.launch.py 时可传入 umi_workspace。

| 仓库 | 主要职责 | 推荐入口 |
| --- | --- | --- |
| universal_manipulation_interface | UMI 数据处理、Diffusion Policy 训练、离线评估、仿真推理 worker | [train.py](../train.py)、[scripts/eval_offline_policy.py](../scripts/eval_offline_policy.py) |
| reBotArmController_ROS2 | 自定义 ROS 2 接口、机械臂驱动、MoveIt 配置、Gazebo 与策略动作桥接 | [start_gazebo_sim.sh](../../reBotArmController_ROS2/scripts/start_gazebo_sim.sh)、[gazebo_rs_sim.launch.py](../../reBotArmController_ROS2/src/rebotarm_moveit_config/launch/gazebo_rs_sim.launch.py) |

ROS 2 工作区主要包含 rebotarm_msgs、rebotarmcontroller、rebotarm_bringup、rebotarm_moveit_config、rebotarm_moveit_demos、rebotarm_agent 和 rebotarm_mujoco_rs。前五个提供接口、硬件控制、启动资源、MoveIt 配置和示例；rebotarm_agent 放置 UMI 回放桥接；rebotarm_mujoco_rs 是另一条 MuJoCo 仿真路径。

## 训练、推理与仿真数据流

~~~mermaid
flowchart LR
    A[示教视频与传感器数据] --> B[UMI SLAM 与数据处理]
    B --> C[Zarr replay buffer]
    C --> D[train.py / Diffusion Policy]
    D --> E[训练 checkpoint]
    E --> F[离线评估]

    G[官方 checkpoint 与 Zarr 验证集] --> H[umi_sim_replay_worker.py]
    H -->|JSON 动作块| I[ROS 2 umi_sim_replay]
    I --> J[MoveIt /compute_ik]
    J --> K[arm 与 gripper FollowJointTrajectory]
    K --> L[Gazebo RS]
    C -. 离线观测窗口 .-> H
~~~

UMI 数据准备由 run_slam_pipeline.py 调度，最终可通过 scripts_slam_pipeline/07_generate_replay_buffer.py 生成训练用 Zarr。训练入口是 train.py，主要配置位于 diffusion_policy/config/；本机正式训练参数记录在 diffusion_policy/config/formal.yaml。

离线评估脚本 scripts/eval_offline_policy.py 复用训练数据集的预处理，可检查 checkpoint 加载、预测结果和误差。scripts/umi_sim_replay_worker.py 从验证集选观测窗口，输出策略预测的动作块。模型动作由相对平移、6D 旋转表示和夹爪组成；worker 将其转成相对末端位姿与夹爪开度的动作块。ROS 2 节点读取动作块，根据当前 TF 计算末端目标、调用 MoveIt 逆解，再发送机械臂和夹爪轨迹到 Gazebo 控制器。

目前 Gazebo 相机图像没有送回 UMI 策略；worker 读取的是 Zarr 中已有的离线观测。因此仿真验证的是权重推理、动作坐标转换、MoveIt 逆解和 ROS 轨迹执行链路，不代表视觉闭环策略已验证。UMI 自带的 eval_real.py 面向其 UR/Franka 实机链路；ReBot 的真实硬件启动入口是 rebotarm_bringup/launch/bringup.launch.py，两条链路还没有合并。

## 2026-09-29 的功能推进

本次整理前检查时，9 月 30 日尚无功能提交；最新一批功能工作发生在 2026-09-29。下面按推进阶段概括：

| 仓库与提交 | 推进内容 |
| --- | --- |
| UMI 2dd64ee、3e656b9 | 增加离线推理脚本、冒烟评估、留出集评估和训练集对照 |
| UMI d4a16c1、7347a17 | 增加训练与离线评估报告、图表，并用官方权重建立同口径对照 |
| UMI e9a993e | 支持官方 checkpoint、验证集口径统一和数据路径覆盖 |
| UMI 05a3732 | 增加官方权重仿真回放 worker，输出 JSON 动作块 |
| ROS 2 ad1a413 | 增加 Gazebo 一键启动脚本 |
| ROS 2 7735d7d | 接通 Gazebo ros2_control、MoveIt 逆解和 UMI 回放节点 |
| ROS 2 92464af | 支持配置仿真回放的动作块数量 |

因此目前可复现的路径是“官方 UMI checkpoint → Gazebo RS 动作回放”。本机训练 checkpoint 已能离线评估，但替换当前仿真默认权重仍需把 checkpoint 和数据集参数从启动入口传到回放节点。ROS 2 仓库检查时 main 比 origin/main 多 3 个提交；这些仿真提交尚未推送。

## 快速启动 Gazebo 策略回放

运行前需要 ROS 2 Jazzy、Gazebo Sim、ros-jazzy-gz-ros2-control、已构建的 ROS 工作区，以及名为 umi 的 Conda 环境。UMI 权重和数据集位于 data/，已由 .gitignore 排除，需要在本机准备：

~~~text
data/pretrained/cup_wild_vit_l_1img.ckpt
data/cup_in_the_wild.zarr.zip
~~~

先在非 Conda 的 ROS 2 Jazzy 环境构建：

~~~bash
cd ~/桌面/umi/reBotArmController_ROS2
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-gz-ros2-control
colcon build --packages-select rebotarmcontroller rebotarm_bringup rebotarm_moveit_config rebotarm_agent --symlink-install
~~~

然后运行仿真：

~~~bash
~/桌面/umi/reBotArmController_ROS2/scripts/start_gazebo_sim.sh
~~~

脚本会启动或复用 Gazebo empty 世界，加载 RS 模型、控制器和 MoveIt，再在 Conda umi 环境中加载官方权重。默认从官方验证集选择 episode，跳过前 15 帧并回放 1 个动作块；可用 MAX_CHUNKS=5 延长回放。完整说明见 [scripts/README.md](../../reBotArmController_ROS2/scripts/README.md)。

## 后续对接重点

1. 将本机训练 checkpoint 在统一的验证集口径下复评，确认训练提升后再进入仿真。
2. 为 Gazebo launch 暴露 checkpoint 与 dataset 参数，明确本机策略替换官方默认权重的操作方式。
3. 设计从 Gazebo 相机到 UMI 观测预处理的输入适配，建立真正的视觉闭环仿真。
4. 将实时 UMI 策略输出适配到 ReBot RS 的 ROS 2 硬件控制链路，再验证真实机械臂部署。

## 相关文档

- [UMI 上游说明](../README.md)
- [ROS 2 中文说明](../../reBotArmController_ROS2/README_zh.md)
- [ROS 2 仿真启动说明](../../reBotArmController_ROS2/scripts/README.md)
- [离线评估报告](offline_eval_report.md)
