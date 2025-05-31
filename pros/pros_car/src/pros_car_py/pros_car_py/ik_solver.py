# ############################################
# A Robot controller for kinematics, dynamics
# and control based on pyBullet framework
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

# Input:
# 1. robot_type: specify urdf file initials eg. if urdf file name is 'ur5.urdf', specify 'ur5'
# 2. controllable_joints: joint indices of controllable joints. If not specified, by default all joints indices except first joint (first joint is fixed joint between robot stand and base)
# 3. end-eff_index: specify the joint indices for end-effector link. If not specified, by default the last controllable_joints is considered as end-effector joint
# 4. time_Step: time step for simulation

import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import os
from ament_index_python.packages import get_package_share_directory
import xml.etree.ElementTree as ET
import math
from scipy.spatial.transform import Rotation as R


class PybulletRobotController:
    def __init__(
        self,
        robot_type="ur5",
        initial_height=0.195,
        controllable_joints=None,
        end_eff_index=None,
        time_step=1e-3,
    ):
        self.robot_type = robot_type
        robot_description_path = get_package_share_directory("robot_description")
        self.urdf_path = os.path.join(robot_description_path, "urdf", "target.urdf")
        self.robot_id = None
        self.num_joints = None
        self.controllable_joints = controllable_joints
        self.end_eff_index = end_eff_index
        self.time_step = time_step
        self.previous_ee_position = None
        self.initial_height = initial_height  # 新增的高度參數

        # 讀取並初始化關節限制
        self.joint_limits = self.get_joint_limits_from_urdf()
        self.num_joints = len(self.joint_limits)  # 使用關節數量設定

    # function to initiate pybullet and engine and create world
    def createWorld(self, GUI=True, view_world=False):
        # load pybullet physics engine
        if GUI:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10
        )
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")
        rotation = R.from_euler("z", 90, degrees=True).as_quat()

        # loading robot into the environment
        urdf_file = "urdf/" + self.robot_type + ".urdf"
        self.robot_id = p.loadURDF(
            self.urdf_path,
            useFixedBase=True,
            basePosition=[0, 0, self.initial_height],
            baseOrientation=rotation,
        )

        self.num_joints = p.getNumJoints(self.robot_id)  # Joints
        print("#Joints:", self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, self.num_joints - 1))
        print("#Controllable Joints:", self.controllable_joints)
        if self.end_eff_index is None:
            self.end_eff_index = self.controllable_joints[-1]
        print("#End-effector:", self.end_eff_index)
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"總關節數量: {self.num_joints}")
        self.controllable_joints = list(range(1, self.num_joints - 1))
        print(f"可控制的關節索引: {self.controllable_joints}")
        print(f"需要提供的初始位置數量: {len(self.controllable_joints)}")

        if view_world:
            while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    # function to joint position, velocity and torque feedback
    def getJointStates(self):
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    # function for setting joint positions of robot
    def setJointPosition(self, position, kp=1.0, kv=1.0):
        # print('Joint position controller')
        zero_vec = [0.0] * len(self.controllable_joints)
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=position,
            targetVelocities=zero_vec,
            positionGains=[kp] * len(self.controllable_joints),
            velocityGains=[kv] * len(self.controllable_joints),
        )
        for _ in range(100):  # to settle the robot to its position
            p.stepSimulation()

    def get_base_position(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        return base_position

    def get_joint_limits_from_urdf(self):
        """
        從 URDF 文件中讀取每個關節的範圍限制。

        Returns:
            joint_limits (dict): 包含每個關節的最小和最大角度限制。
        """
        joint_limits = {}
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            joint_name = joint.get("name")

            # 忽略特定的夹具关节，比如 "gripper_joint"
            if joint_name == "Revolute 6":
                continue

            joint_type = joint.get("type")
            if joint_type == "revolute" or joint_type == "continuous":
                limit = joint.find("limit")
                if limit is not None:
                    lower = float(
                        limit.get("lower", -3.14159)
                    )  # 預設為 -180 度（以 radians 為單位）
                    upper = float(limit.get("upper", 3.14159))  # 預設為 180 度
                    joint_limits[joint_name] = (lower, upper)
        return joint_limits

    def get_current_pose(self, link_index=None):
        """
        根據指定的手臂關節角度來獲取特定連結的世界座標和旋轉矩陣。

        Args:
            link_index (int, optional): 連結索引。默認為倒數第一個連結（即末端執行器）。

        Returns:
            position (np.array): 指定連結的世界坐標 [x, y, z]。
            rotation_matrix (np.array): 指定連結的旋轉矩陣 (3x3)。
        """
        # 獲取機器人的總連結數
        num_links = p.getNumJoints(self.robot_id)

        # 處理負索引，使其轉換為對應的正索引
        if link_index is None:
            link_index = self.end_eff_index  # 默認為末端執行器
        elif link_index < 0:
            link_index = num_links + link_index  # 將負索引轉換為正索引

        # 驗證link_index是否合法
        if link_index < 0 or link_index >= num_links:
            raise ValueError(
                f"Invalid link_index {link_index}. Valid range is 0 to {num_links - 1}."
            )

        # 獲取指定連結的鏈接狀態
        link_state = p.getLinkState(
            self.robot_id, link_index, computeForwardKinematics=True
        )

        # 取得指定連結在世界座標系中的位置和方向（四元數）
        link_position = np.array(link_state[4])  # worldLinkFramePosition
        link_orientation = link_state[5]  # worldLinkFrameOrientation (四元數)

        # 將四元數轉換為旋轉矩陣
        rotation_matrix = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(
            3, 3
        )

        return link_position, rotation_matrix

    def get_base_position(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        return base_position

    def get_camera_pose(self):
        # 加設深度相機裝在倒數第二關節
        camera_link_index = self.end_eff_index - 1

        link_state = p.getLinkState(
            self.robot_id, camera_link_index, computeForwardKinematics=True
        )

        # 抓世界座標
        link_world_position = link_state[4]  # worldLinkFramePosition
        link_world_orientation = link_state[5]

        # 定義相機在倒數第二關節的偏移
        camera_offset_local = [0, 0, 0.01]  # 公尺

        # 計算世界座標
        camera_world_position, camera_world_orientation = p.multiplyTransforms(
            link_world_position,
            link_world_orientation,
            camera_offset_local,
            [0, 0, 0, 1],
        )

        return camera_world_position, camera_world_orientation

    def get_base_pose(self):
        # 抓基座世界座標
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)

        base_position = np.array(base_position)

        # 返回基座位置和方向
        return base_position, base_orientation

    def human_like_wave(self, num_moves=5, steps=30):
        if len(self.joint_limits) != len(self.controllable_joints):
            raise ValueError("關節數量與 joint_limits 數量不匹配")

        angle_sequence = []
        current_positions = np.array(self.getJointStates()[0])  # 初始角度

        for _ in range(num_moves):
            # 生成一組新目標角度，使動作更大
            target_positions = np.array(
                [
                    random.uniform(lower, upper)
                    for lower, upper in self.joint_limits.values()
                ]
            )

            # 平滑過渡
            for step in range(steps):
                t = step / steps
                intermediate_positions = (
                    1 - t
                ) * current_positions + t * target_positions
                angle_sequence.append(intermediate_positions.tolist())

            # 更新當前位置
            current_positions = target_positions

        return angle_sequence

    def random_wave(self, num_moves=5, steps=30):  # 減少過渡步數
        # 檢查 joint_limits 長度是否和可控關節數量一致
        if len(self.joint_limits) != len(self.controllable_joints):
            raise ValueError("關節數量與 joint_limits 數量不匹配")

        angle_sequence = []  # 用於存儲所有的角度組合
        current_positions = np.array(self.getJointStates()[0])  # 初始關節角度

        for _ in range(num_moves):
            # 生成一組新的目標角度，增加目標範圍的隨機幅度
            target_positions = [
                random.uniform(lower - 0.5 * abs(lower), upper + 0.5 * abs(upper))
                for lower, upper in self.joint_limits.values()
            ]
            target_positions = np.clip(
                target_positions,
                [l for l, _ in self.joint_limits.values()],
                [u for _, u in self.joint_limits.values()],
            )
            target_positions = np.array(target_positions)

            # 確認 target_positions 與 current_positions 的形狀匹配
            if current_positions.shape != target_positions.shape:
                raise ValueError("生成的目標角度數量與當前角度數量不一致")

            # 平滑過渡：生成插值角度，過渡到下一組目標角度
            for step in range(steps):
                intermediate_positions = (1 - step / steps) * current_positions + (
                    step / steps
                ) * target_positions
                angle_sequence.append(intermediate_positions.tolist())  # 添加到序列中

            # 更新當前位置為目標位置，以便下一次的平滑過渡
            current_positions = target_positions

        return angle_sequence  # 返回所有角度組合

    def markTarget(self, target_position):
        # 使用紅色標記顯示目標位置
        line_length = 0.1  # 調整標記大小
        p.addUserDebugLine(
            [target_position[0] - line_length, target_position[1], target_position[2]],
            [target_position[0] + line_length, target_position[1], target_position[2]],
            [1, 0, 0],  # 紅色
            lineWidth=3,
        )
        p.addUserDebugLine(
            [target_position[0], target_position[1] - line_length, target_position[2]],
            [target_position[0], target_position[1] + line_length, target_position[2]],
            [1, 0, 0],
            lineWidth=3,
        )
        p.addUserDebugLine(
            [target_position[0], target_position[1], target_position[2] - line_length],
            [target_position[0], target_position[1], target_position[2] + line_length],
            [1, 0, 0],
            lineWidth=3,
        )

    # function to solve forward kinematics
    def solveForwardPositonKinematics(self, joint_pos):
        # get end-effector link state
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        eePose = list(link_trn) + list(p.getEulerFromQuaternion(link_rot))
        return eePose

    def format_joint_angles(joint_angles, precision=3):
        """
        將列表中的所有角度轉換為 float，並保留小數點後指定位數。

        Args:
            joint_angles (list): 原始的關節角度列表。
            precision (int): 保留的小數位數 (預設為 3)。

        Returns:
            list: 格式化後的關節角度列表。
        """
        return [round(float(angle), precision) for angle in joint_angles]

    def moveTowardsTarget(self, target_position, steps=50):
        # 獲取當前末端執行器的位置
        current_position = self.solveForwardPositonKinematics(self.getJointStates()[0])[
            0:3
        ]

        # 計算每一步的位移向量
        step_vector = (np.array(target_position) - np.array(current_position)) / steps
        self.markTarget(target_position)

        # 用於存儲每一步的關節角度（以弧度表示）
        joint_angles_in_radians = []

        # 逐步靠近目標
        for i in range(steps):
            # 計算當前目標位置
            intermediate_position = np.array(current_position) + (i + 1) * step_vector
            # 計算 IK 解
            joint_angles = self.solveInversePositionKinematics(intermediate_position)

            # 將當前關節角度應用到機器人
            if joint_angles and len(joint_angles) >= len(self.controllable_joints):
                # 直接存儲弧度值
                joint_angles_in_radians.append(
                    joint_angles[: len(self.controllable_joints)]
                )

                # self.setJointPosition(joint_angles[:len(self.controllable_joints)])
                time.sleep(0.1)  # 加入延遲觀察
            else:
                print("無法找到合適的解。")
                break

        return joint_angles_in_radians

    # function to solve inverse kinematics
    # 單獨使用要少取一個 因為會輸出 6
    def solveInversePositionKinematics(self, end_eff_pose):
        """
        計算逆向運動學以獲取關節角度，基於給定的末端執行器姿勢。

        Args:
            end_eff_pose (list): 末端執行器的目標位置和姿勢，
                                格式為 [x, y, z, roll, pitch, yaw] (6 個元素) 或 [x, y, z] (3 個元素)。

        Returns:
            list: 對應的關節角度。
        """
        if len(end_eff_pose) == 6:
            joint_angles = p.calculateInverseKinematics(
                self.robot_id,
                self.end_eff_index,
                targetPosition=end_eff_pose[0:3],
                targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]),
            )
        else:
            joint_angles = p.calculateInverseKinematics(
                self.robot_id, self.end_eff_index, targetPosition=end_eff_pose[0:3]
            )

        # 標記末端執行器的位置路徑
        self.markEndEffectorPath()
        return joint_angles

    def markEndEffector(self):
        # 獲取末端執行器的位置
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        ee_position = eeState[0]  # 末端執行器的位置

        # 使用藍色點標記末端執行器位置
        p.addUserDebugPoints(
            position=ee_position, color=[0, 0, 1], size=0.05  # 藍色  # 點的大小
        )

    def markEndEffectorPath(self):
        # 獲取當前末端執行器的位置
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        ee_position = eeState[0]

        # 如果是第一個點，設置 previous_ee_position
        if self.previous_ee_position is None:
            self.previous_ee_position = ee_position

        # 繪製從上次位置到當前位置的線
        p.addUserDebugLine(
            self.previous_ee_position,
            ee_position,
            lineColorRGB=[0, 0, 1],  # 藍色
            lineWidth=2,
        )

        # 更新 previous_ee_position 為當前位置
        self.previous_ee_position = ee_position

    # function to get jacobian
    def getJacobian(self, joint_pos):
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        zero_vec = [0.0] * len(joint_pos)
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            self.end_eff_index,
            com_trn,
            list(joint_pos),
            zero_vec,
            zero_vec,
        )
        J_t = np.asarray(jac_t)
        J_r = np.asarray(jac_r)
        J = np.concatenate((J_t, J_r), axis=0)
        print("Jacobian:", J)
        return J

    # function to solve forward velocity kinematics
    def solveForwardVelocityKinematics(self, joint_pos, joint_vel):
        print("Forward velocity kinematics")
        J = self.getJacobian(joint_pos)
        eeVelocity = J @ joint_vel
        print("End-effector velocity:", eeVelocity)
        return eeVelocity

    # function to solve inverse velocity kinematics
    def solveInverseVelocityKinematics(self, end_eff_velocity):
        print("Inverse velocity kinematics")
        joint_pos, _, _ = self.getJointStates()
        J = self.getJacobian(joint_pos)
        if len(self.controllable_joints) > 1:
            joint_vel = np.linalg.pinv(J) @ end_eff_velocity
        else:
            joint_vel = J.T @ end_eff_velocity
        print("Joint velcoity:", joint_vel)
        return joint_vel

    # function to do joint velcoity control
    def JointVelocityControl(self, joint_velocities, sim_time=2, max_force=200):
        print("Joint velocity controller")
        t = 0
        while t < sim_time:
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                p.VELOCITY_CONTROL,
                targetVelocities=joint_velocities,
                forces=[max_force] * (len(self.controllable_joints)),
            )
            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

    # function to do joint velcoity control
    def endEffectorVelocityControl(self, end_eff_vel, sim_time=2, max_forc=200):
        print("End-effector velocity controller")
        t = 0
        while t < sim_time:
            joint_velocities = self.solveInverseVelocityKinematics(end_eff_vel)
            self.JointVelocityControl(joint_velocities)
            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

    # Function to define GUI sliders (name of the parameter,range,initial value)
    def TaskSpaceGUIcontrol(self, goal, max_limit=3.14, min_limit=-3.14):
        xId = p.addUserDebugParameter("x", min_limit, max_limit, goal[0])  # x
        yId = p.addUserDebugParameter("y", min_limit, max_limit, goal[1])  # y
        zId = p.addUserDebugParameter("z", min_limit, max_limit, goal[2])  # z
        rollId = p.addUserDebugParameter("roll", min_limit, max_limit, goal[3])  # roll
        pitchId = p.addUserDebugParameter(
            "pitch", min_limit, max_limit, goal[4]
        )  # pitch
        yawId = p.addUserDebugParameter("yaw", min_limit, max_limit, goal[5])  # yaw
        return [xId, yId, zId, rollId, pitchId, yawId]

    def ForceGUIcontrol(self, forces, max_limit=1.0, min_limit=-1.0):
        fxId = p.addUserDebugParameter(
            "fx", min_limit, max_limit, forces[0]
        )  # force along x
        fyId = p.addUserDebugParameter(
            "fy", min_limit, max_limit, forces[1]
        )  # force along y
        fzId = p.addUserDebugParameter(
            "fz", min_limit, max_limit, forces[2]
        )  # force along z
        mxId = p.addUserDebugParameter(
            "mx", min_limit, max_limit, forces[3]
        )  # moment along x
        myId = p.addUserDebugParameter(
            "my", min_limit, max_limit, forces[4]
        )  # moment along y
        mzId = p.addUserDebugParameter(
            "mz", min_limit, max_limit, forces[5]
        )  # moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]

    # function to read the value of task parameter
    def readGUIparams(self, ids):
        val1 = p.readUserDebugParameter(ids[0])
        val2 = p.readUserDebugParameter(ids[1])
        val3 = p.readUserDebugParameter(ids[2])
        val4 = p.readUserDebugParameter(ids[3])
        val5 = p.readUserDebugParameter(ids[4])
        val6 = p.readUserDebugParameter(ids[5])
        return np.array([val1, val2, val3, val4, val5, val6])

    # function to get desired joint trajectory
    def getTrajectory(self, thi, thf, tf, dt):
        desired_position, desired_velocity, desired_acceleration = [], [], []
        t = 0
        while t <= tf:
            th = thi + ((thf - thi) / tf) * (
                t - (tf / (2 * np.pi)) * np.sin((2 * np.pi / tf) * t)
            )
            dth = ((thf - thi) / tf) * (1 - np.cos((2 * np.pi / tf) * t))
            ddth = (2 * np.pi * (thf - thi) / (tf * tf)) * np.sin((2 * np.pi / tf) * t)
            desired_position.append(th)
            desired_velocity.append(dth)
            desired_acceleration.append(ddth)
            t += dt
        desired_position = np.array(desired_position)
        desired_velocity = np.array(desired_velocity)
        desired_acceleration = np.array(desired_acceleration)
        return desired_position, desired_velocity, desired_acceleration

    # function to calculate dynamic matrics: inertia, coriolis, gravity
    def calculateDynamicMatrices(self):
        joint_pos, joint_vel, _ = self.getJointStates()
        n_dof = len(self.controllable_joints)
        InertiaMatrix = np.asarray(p.calculateMassMatrix(self.robot_id, joint_pos))
        GravityMatrix = np.asarray(
            p.calculateInverseDynamics(
                self.robot_id, joint_pos, [0.0] * n_dof, [0.0] * n_dof
            )
        )
        CoriolisMatrix = (
            np.asarray(
                p.calculateInverseDynamics(
                    self.robot_id, joint_pos, joint_vel, [0.0] * n_dof
                )
            )
            - GravityMatrix
        )
        return InertiaMatrix, GravityMatrix, CoriolisMatrix

    # Function to simulate free fall under gravity
    def doFreeFall(self):
        p.setRealTimeSimulation(False)
        # Enable torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.controllable_joints)),
        )

        tau = [0.0] * len(self.controllable_joints)  # for free fall under gravity
        while True:
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau,
            )
            p.stepSimulation()
            time.sleep(self.time_step)
        p.disconnect()

    # Function to do inverse dynamics simulation
    def doInverseDynamics(self, th_initial, th_final, final_time=2):
        p.setRealTimeSimulation(False)
        # get the desired trajectory
        q_d, dq_d, ddq_d = self.getTrajectory(
            th_initial, th_final, tf=final_time, dt=self.time_step
        )
        traj_points = q_d.shape[0]
        print("#Trajectory points:", traj_points)

        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(
                self.robot_id,
                link_idx,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=0.0,
            )
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # Enable torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.controllable_joints)),
        )

        kd = 0.7  # from URDF file
        n = 0
        while n < traj_points:
            tau = p.calculateInverseDynamics(
                self.robot_id, list(q_d[n]), list(dq_d[n]), list(ddq_d[n])
            )
            # tau += kd * dq_d[n] #if joint damping is turned off, this torque will not be required
            # print(tau)

            # torque control
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau,
            )
            theta, _, _ = self.getJointStates()
            print("n:{}::th:{}".format(n, theta))

            p.stepSimulation()
            time.sleep(self.time_step)
            n += 1
        print("Desired joint angles:", th_final)
        p.disconnect()

    # Function to do computed torque control
    def computedTorqueControl(
        self, th_initial, th_final, final_time=2, controller_gain=400
    ):
        p.setRealTimeSimulation(False)
        # get the desired trajectory
        q_d, dq_d, ddq_d = self.getTrajectory(
            th_initial, th_final, tf=final_time, dt=self.time_step
        )
        traj_points = q_d.shape[0]
        print("#Trajectory points:", traj_points)

        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(
                self.robot_id,
                link_idx,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=0.0,
            )
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # Enable torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.controllable_joints)),
        )

        Kp = controller_gain
        Kd = 2 * np.sqrt(Kp)
        n = 0
        while n < q_d.shape[0]:

            # get current joint states
            q, dq, _ = self.getJointStates()
            # PD control
            q_e = q_d[n] - np.asarray(q)
            dq_e = dq_d[n] - np.asarray(dq)
            aq = ddq_d[n] + Kp * q_e + Kd * dq_e

            tau = p.calculateInverseDynamics(self.robot_id, list(q), list(dq), list(aq))
            # tau += kd * dq_d[n] # if joint damping is turned off, this torque will not be required
            # print(tau)

            # torque control
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau,
            )

            print("n:{}::th:{}".format(n, q))

            p.stepSimulation()
            time.sleep(self.time_step)
            n += 1
        print("Desired joint angles:", th_final)
        p.disconnect()

    def stop_simulation(self):
        # p.setRealTimeSimulation(False)
        p.disconnect()

    # Function to do impedence control in task space
    def impedenceController(self, th_initial, desired_pose, controller_gain=100):
        p.setRealTimeSimulation(False)
        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(
                self.robot_id,
                link_idx,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=0.0,
            )
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # Enable torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.controllable_joints)),
        )

        kd = 0.7  # from URDF file
        Kp = controller_gain
        Kd = 2 * np.sqrt(Kp)
        Md = 0.01 * np.eye(6)

        # Target position and velcoity
        # xd = np.array([0.4499998573193256, 0.1, 1.95035834701983, 0.0, 1.5707963267948966, 0.0]) #1-link
        # xd = np.array([1.3499995719791142, 0.2, 2.9510750145148816, 0.0, 1.5707963267948966, 0.0]) #2-link
        # xd = np.array([2.199999302511422, 0.3, 2.9517518416742643, 0.0, 1.5707963267948966, 0.0]) #3-link
        # xd = np.array([1.8512362079506117, 0.30000000000000004, 4.138665008474901, -0.0, 1.0000000496605894, -0.0]) #3-link
        # xd = np.array([0.10972055742719365, -0.716441307051838, 1.44670878280948, -1.5700006464761673, 0.0007970376813496536, -1.570796326772595]) #ur5-link
        # xd = np.array([0.6811421738723965, -0.24773390188802563, 1.44670878280948, -1.5700006464761678, 0.0007970376813495148, -0.5007963267725951]) #ur5-link
        # xd = np.array([-0.10857937593446423, 0.7166151451748437, 1.4467087828094798, -1.5700006464761673, 0.0007970376813502642, 1.5692036732274044]) #ur5-link
        xd = desired_pose
        dxd = np.zeros(len(self.controllable_joints))

        # define GUI sliders
        xdGUIids = self.TaskSpaceGUIcontrol(goal=xd)
        ForceInitial = np.zeros(len(self.controllable_joints))
        ForceGUIids = self.ForceGUIcontrol(
            forces=ForceInitial, max_limit=10, min_limit=-10
        )

        while True:
            # read GUI values
            xd = self.readGUIparams(xdGUIids)  # task space goal
            F_ext = self.readGUIparams(ForceGUIids)  # applied external forces

            # get current joint states
            q, dq, _ = self.getJointStates()

            # Error in task space
            x = self.solveForwardPositonKinematics(q)
            x_e = xd - x
            dx = self.solveForwardVelocityKinematics(q, dq)
            dx_e = dxd - dx

            # Task space dynamics
            # Jacobian
            J = self.getJacobian(q)
            J_inv = np.linalg.pinv(J)
            # Inertia matrix in the joint space
            Mq, G, _ = self.calculateDynamicMatrices()
            # Inertia matrix in the task space
            Mx = np.dot(np.dot(np.transpose(J_inv), Mq), J_inv)
            # Force in task space
            Fx = np.dot(
                np.dot(np.linalg.inv(Md), Mx), (np.dot(Kp, x_e) + np.dot(Kd, dx_e))
            )
            # External Force applied
            F_w_ext = np.dot((np.dot(np.linalg.inv(Md), Mx) - np.eye(6)), F_ext)
            Fx += F_w_ext
            # Force in joint space
            Fq = np.dot(np.transpose(J), Fx)

            # Controlled Torque
            tau = G + Fq
            # tau += kd * np.asarray(dq) # if joint damping is turned off, this torque will not be required
            # print('tau:', tau)

            # Activate torque control
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau,
            )

            p.stepSimulation()
            time.sleep(self.time_step)
        p.disconnect()
