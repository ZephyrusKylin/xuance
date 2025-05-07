import numpy as np
import gymnasium as gym # 导入Gymnasium库，用于构建强化学习环境
from gymnasium.spaces import Box, Discrete, Dict as GymDict # 从Gymnasium导入空间类型
from xuance.environment import RawMultiAgentEnv # 从xuance框架导入原始多智能体环境基类
from typing import List, Dict as TypingDict, Optional, Tuple, Any, Union # Python类型提示
import abc # 导入abc模块，用于定义抽象基类

# 假设 satellite_function.py 包含轨道动力学等辅助函数
# 例如：from . import satellite_function as sf (如果satellite_function在同一目录下)
# 根据实际项目结构调整导入路径
from xuance.common import satellite_function as sf # 从xuance.common导入卫星功能模块

# -------------------------------------------------------------------------------------------
# 1. Satellite 类 (卫星类定义)
# -------------------------------------------------------------------------------------------
class Satellite:
    """
    代表单个卫星的通用类，包含其所有属性和基本行为。
    属性通过配置字典在初始化时设置。
    新增质量和最大推力属性。
    """
    def __init__(self, sat_id: str, team_id: str,
                 type_config: TypingDict[str, Any],
                 initial_pos: np.ndarray, initial_vel: np.ndarray,
                ):
        # 初始化卫星基本属性
        self.id = sat_id # 卫星ID
        self.team_id = team_id # 卫星所属队伍ID
        self.type = type_config.get("type_name", "default") # 卫星类型，默认为 "default"

        self.pos = np.array(initial_pos, dtype=np.float32) # 初始位置 [x, y, z]
        self.vel = np.array(initial_vel, dtype=np.float32) # 初始速度 [vx, vy, vz]
        
        self.mass = float(type_config.get("mass", 100.0))  # 卫星质量，单位 kg
        if self.mass <= 0: self.mass = 1.0 # 保证质量为正

        self.max_total_thrust = float(type_config.get("max_total_thrust", 10.0)) # 最大总推力，单位 N

        self.max_fuel = float(type_config.get("max_fuel", 1000.0)) # 最大燃料量
        self.current_fuel = self.max_fuel # 当前燃料量
        # 每牛顿秒的燃料消耗率 (推力 * 时间 = 冲量)，即单位冲量消耗的燃料
        self.fuel_consumption_per_newton_second = float(type_config.get("fuel_consumption_per_newton_second", 0.01))

        self.max_health = float(type_config.get("max_health", 100.0)) # 最大生命值
        self.current_health = self.max_health # 当前生命值
        self.is_destroyed = False # 是否被摧毁

        # 攻击能力相关属性
        self.can_attack = bool(type_config.get("can_attack", False)) # 是否能攻击
        if self.can_attack:
            self.weapon_range = float(type_config.get("weapon_range", 50000.0)) # 武器射程
            self.weapon_damage = float(type_config.get("weapon_damage", 10.0)) # 武器伤害
            self.max_ammo = int(type_config.get("max_ammo", 20)) # 最大弹药量
            self.current_ammo = self.max_ammo # 当前弹药量
            self.fire_cooldown_max = int(type_config.get("fire_cooldown_steps", 5)) # 开火冷却时间（步数）
            self.current_fire_cooldown = 0 # 当前开火冷却
        else:
            # 如果不能攻击，则相关属性设为0或无效值
            self.weapon_range, self.weapon_damage, self.max_ammo, self.current_ammo, self.fire_cooldown_max, self.current_fire_cooldown = 0.0, 0.0, 0, 0, 0, 0

        # 观测能力相关属性
        self.can_observe = bool(type_config.get("can_observe", True)) # 是否能观测
        if self.can_observe:
            self.sensor_range = float(type_config.get("sensor_range", 150000.0)) # 传感器范围
        else:
            self.sensor_range = 0.0
            
        # 期望的编队偏移量（相对于某个参考点，如编队中心或领队）
        self.preferred_formation_offset = np.array(type_config.get("formation_offset", [0,0,0]), dtype=np.float32)

    def consume_fuel(self, actual_thrust_force_magnitude: float, time_delta: float):
        """根据实际施加的推力大小和时间间隔消耗燃料"""
        if not self.is_destroyed and self.current_fuel > 0:
            # 消耗的燃料 = 推力大小 * 单位冲量燃料消耗率 * 时间
            fuel_consumed = actual_thrust_force_magnitude * self.fuel_consumption_per_newton_second * time_delta
            self.current_fuel = max(0, self.current_fuel - fuel_consumed) # 燃料不能为负
    
    def update_kinematics_with_thrust(self, desired_force_vector: np.ndarray, time_delta: float):
        """根据期望的推力矢量更新卫星的速度，并返回实际施加的力矢量"""
        if self.is_destroyed:
            return np.zeros(3, dtype=np.float32) # 被摧毁则不施加力
        
        actual_force_vector = np.zeros(3, dtype=np.float32) # 初始化实际施加的力
        if self.current_fuel > 0: # 只有在有燃料时才能施加推力
            desired_force_magnitude = np.linalg.norm(desired_force_vector) # 计算期望力的大小
            if desired_force_magnitude > 1e-6: # 避免除以零或处理极小的期望力
                # 如果期望力超过最大推力，则按最大推力施加
                if desired_force_magnitude > self.max_total_thrust:
                    actual_force_vector = (desired_force_vector / desired_force_magnitude) * self.max_total_thrust
                else: # 否则按期望力施加
                    actual_force_vector = desired_force_vector
        
        # 根据牛顿第二定律 F=ma => a = F/m 更新速度
        acceleration = actual_force_vector / self.mass
        self.vel += acceleration * time_delta # v_new = v_old + a * dt
        
        actual_thrust_force_magnitude = np.linalg.norm(actual_force_vector) # 计算实际施加推力的大小
        self.consume_fuel(actual_thrust_force_magnitude, time_delta) # 消耗燃料
        
        return actual_force_vector # 返回实际施加的力

    def update_cooldowns(self):
        """更新武器的冷却时间"""
        if self.can_attack and self.current_fire_cooldown > 0:
            self.current_fire_cooldown -= 1

    def take_damage(self, damage: float):
        """卫星受到伤害，减少当前生命值"""
        # 如果卫星已被摧毁或设计上不能被攻击(例如某些类型的卫星没有生命值或无敌)
        # 当前逻辑：如果 can_attack 为 False，则不受伤。这暗示不能攻击的卫星是无敌的或非战斗单位。
        # 如果希望所有卫星都能受伤，应移除 or not self.can_attack
        if self.is_destroyed: return 
        
        self.current_health -= damage
        if self.current_health <= 0:
            self.current_health = 0
            self.is_destroyed = True

    def can_fire_weapon(self) -> bool:
        """判断卫星当前是否可以开火"""
        return self.can_attack and \
               not self.is_destroyed and \
               self.current_ammo > 0 and \
               self.current_fire_cooldown == 0

    def fire_weapon_at_target(self) -> bool:
        """
        执行开火动作。如果成功开火，则减少弹药并重置武器冷却。
        返回: True 如果成功开火, False 否则。
        """
        if self.can_fire_weapon():
            self.current_ammo -= 1
            self.current_fire_cooldown = self.fire_cooldown_max
            return True
        return False

    def get_self_observation_component(self, normalize: bool = True) -> np.ndarray:
        """
        获取卫星自身的观测信息组件。
        包括：位置(3), 速度(3), 燃料(1), 生命值(1), 弹药(1), 存活状态(1)。共10个元素。
        normalize: 是否对燃料、生命值、弹药进行归一化。
        """
        fuel_norm = self.current_fuel / self.max_fuel if self.max_fuel > 0 else 0.0
        health_norm = self.current_health / self.max_health if self.max_health > 0 else 0.0
        ammo_norm = self.current_ammo / self.max_ammo if self.max_ammo > 0 else 0.0
        
        return np.array([
            self.pos[0], self.pos[1], self.pos[2], 
            self.vel[0], self.vel[1], self.vel[2],
            fuel_norm if normalize else self.current_fuel,
            health_norm if normalize else self.current_health,
            ammo_norm if normalize else self.current_ammo,
            1.0 if not self.is_destroyed else 0.0  # 存活状态 (1.0 代表存活, 0.0 代表被摧毁)
        ], dtype=np.float32) # 总共10个元素

    def get_relative_observation_to(self, other_satellite: 'Satellite', normalize: bool = True) -> np.ndarray:
        """
        获取相对于另一个卫星的观测信息。
        包括：相对位置(3), 相对速度(3), 对方生命值(1), 对方存活状态(1), 
               对方是否在己方传感器范围内(1), 对方是否在己方武器范围内(1)。共10个元素。
        normalize: 是否对对方生命值进行归一化。
        """
        if other_satellite.is_destroyed: # 如果对方已被摧毁，其状态信息部分为0
            rel_pos, rel_vel = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            other_health_norm, other_alive = 0.0, 0.0
        else:
            rel_pos = other_satellite.pos - self.pos # 相对位置 = 对方位置 - 自己位置
            rel_vel = other_satellite.vel - self.vel # 相对速度 = 对方速度 - 自己速度
            other_health_norm = other_satellite.current_health / other_satellite.max_health if other_satellite.max_health > 0 else 0.0
            other_alive = 1.0 # 如果未被摧毁，则存活
        
        dist = np.linalg.norm(rel_pos) # 与对方的距离
        
        # 从自身视角判断对方是否在传感器/武器范围内
        in_my_sensor_range = 0.0
        if self.can_observe and not other_satellite.is_destroyed and dist <= self.sensor_range:
            in_my_sensor_range = 1.0
            
        in_my_weapon_range = 0.0
        if self.can_attack and not other_satellite.is_destroyed and dist <= self.weapon_range:
            in_my_weapon_range = 1.0
        
        return np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],         # 相对位置 (3)
            rel_vel[0], rel_vel[1], rel_vel[2],         # 相对速度 (3)
            other_health_norm if normalize else other_satellite.current_health, # 对方生命值 (归一化或原始值) (1)
            other_alive,                                # 对方存活状态 (1)
            in_my_sensor_range,                         # 对方是否在己方传感器程内 (1)
            in_my_weapon_range                          # 对方是否在己方武器射程内 (1)
        ], dtype=np.float32) # 总共10个元素

# -------------------------------------------------------------------------------------------
# 2. MultiSatelliteEnvBase (多卫星环境基类)
# -------------------------------------------------------------------------------------------
class MultiSatelliteEnvBase(abc.ABC): # 继承自抽象基类
    def __init__(self, env_config: Any, scenario_config: TypingDict[str, Any] = None):
        self.config = env_config  # 通用环境配置 (通常是 Namespace 对象)
        self.scenario_config = scenario_config if scenario_config is not None else {}  # 特定场景配置 (字典)
        self.viewer = None # 用于渲染的查看器对象，暂未使用
        
        self.env_id = self.scenario_config.get("scenario_id", "MultiSatelliteBaseScenario-v0") # 场景ID
        self.step_time_interval = getattr(self.config, "step_time_interval", 50.0) # 每个环境步骤代表的物理时间，单位秒
        self.max_episode_steps = int(self.scenario_config.get("max_episode_steps", 1000)) # 每回合最大步数
        self._current_step = 0 # 当前回合的步数计数器
        
        # 是否使用CW (Clohessy-Wiltshire) 动力学模型
        self.use_cw_dynamics = getattr(self.config, "use_cw_dynamics", True)
        if self.use_cw_dynamics:
            # CW动力学参考轨道的初始惯性位置和速度 (通常设为0，表示相对运动)
            self.R_cw_ref_inertial = np.array(getattr(self.config, "cw_ref_inertial_pos", [0.0,0.0,0.0]), dtype=np.float32)
            self.V_cw_ref_inertial = np.array(getattr(self.config, "cw_ref_inertial_vel", [0.0,0.0,0.0]), dtype=np.float32)
        
        # 卫星类型配置字典，键为类型名，值为该类型的属性字典
        self.satellite_type_configs = getattr(self.config, "satellite_types", {"default": {}})
        # 动作缩放因子：将RL策略输出的归一化动作（如[-1,1]）映射到物理单位（如推力N）
        self.action_scale = getattr(self.config, "action_scale", 1.0) 
        
        self.agents: List[str] = [] # 智能体ID列表
        self.agent_ids: List[str] = [] # 通常与 self.agents 相同，为兼容性保留
        self.n_agents: int = 0 # 智能体数量
        self.satellites: TypingDict[str, Satellite] = {} # 存储Satellite对象的字典，键为agent_id
        self.team_assignments: TypingDict[str, List[str]] = {}  # 队伍分配，键为team_id，值为该队伍的agent_id列表
        
        # Gym标准空间定义
        self.observation_space: TypingDict[str, gym.Space] = {} # 每个智能体的观测空间
        self.action_space: TypingDict[str, gym.Space] = {} # 每个智能体的动作空间
        self.state_space: Optional[gym.Space] = None # 全局状态空间 (可选，可由具体场景定义)
        
        self.infos: TypingDict[str, Any] = {} # 存储每个智能体和通用环境信息的字典

        # 初始化流程
        self._initialize_teams_and_agents() # 初始化队伍和智能体 (由子类实现)
        self._initialize_satellites()      # 根据队伍和配置创建卫星实例
        self._define_spaces()              # 定义观测和动作空间 (由子类实现)

    @abc.abstractmethod # 标记为抽象方法，子类必须实现
    def _initialize_teams_and_agents(self):
        """初始化队伍信息 (self.team_assignments) 和智能体列表 (self.agents, self.agent_ids, self.n_agents)"""
        pass
    
    def _initialize_satellites(self):
        """根据队伍和智能体信息，以及场景和通用配置中的卫星组成，创建 Satellite 对象。"""
        self.satellites.clear() # 清除上一回合的卫星实例
        for team_id, agent_id_list in self.team_assignments.items():
            # 确定队伍的卫星类型组成
            composition_key = f"{team_id}_composition" # 例如 "team_A_composition"
            default_composition = ["default"] * len(agent_id_list) # 默认所有智能体都是 "default" 类型
            
            # 优先级: 1. scenario_config (特定场景配置), 2. env_config (通用配置), 3. default_composition
            team_composition_specific = self.scenario_config.get(composition_key)
            team_composition_general = getattr(self.config, composition_key, default_composition) 
            
            team_composition = team_composition_specific if team_composition_specific is not None else team_composition_general
            
            # 如果配置的组成列表长度与智能体数量不匹配，则发出警告并调整
            if len(team_composition) != len(agent_id_list):
                print(f"警告: 队伍 '{team_id}' 的 '{composition_key}' 长度 ({len(team_composition)}) "
                      f"与智能体数量 ({len(agent_id_list)}) 不匹配。将使用 'default' 类型进行填充或截断。")
                if len(team_composition) > len(agent_id_list):
                    team_composition = team_composition[:len(agent_id_list)] # 截断
                else:
                    team_composition.extend(["default"] * (len(agent_id_list) - len(team_composition))) # 填充

            for i, agent_id in enumerate(agent_id_list):
                sat_type_name = team_composition[i] # 获取当前智能体的卫星类型名
                # 从 self.satellite_type_configs 获取该类型的配置，如果找不到则用 "default" 类型的配置
                type_config = self.satellite_type_configs.get(sat_type_name, 
                                                              self.satellite_type_configs.get("default", {}))
                
                if not type_config: # 如果连 "default" 类型配置都没有，则使用一个最小化的后备配置
                    print(f"警告: 卫星类型 '{sat_type_name}' 和 'default' 均未在 satellite_types 中定义。"
                          f"为智能体 {agent_id} 使用最小化后备配置。")
                    type_config = {"type_name": "minimal_default", "mass": 1.0, "max_total_thrust": 0.1}
                else:
                    # 复制配置字典以避免修改原始配置，并确保 type_name 正确
                    type_config = type_config.copy() 
                    type_config["type_name"] = sat_type_name 
                    
                # 获取初始位置和速度 (由子类实现)
                initial_pos, initial_vel = self._get_initial_pos_vel_for_satellite(agent_id, team_id, i, type_config)
                # 创建 Satellite 实例
                self.satellites[agent_id] = Satellite(agent_id, team_id, type_config, initial_pos, initial_vel)
    
    @abc.abstractmethod
    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, 
                                           agent_idx_in_team: int, type_config: TypingDict
                                          ) -> Tuple[np.ndarray, np.ndarray]:
        """为指定智能体获取初始位置和速度 (由子类实现)"""
        pass
    
    @abc.abstractmethod
    def _define_spaces(self):
        """定义每个智能体的观测空间 (self.observation_space) 和动作空间 (self.action_space)，
           以及可选的全局状态空间 (self.state_space) (由子类实现)"""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None
             ) -> Tuple[TypingDict[str, np.ndarray], TypingDict[str, Any]]:
        """重置环境到初始状态"""
        if seed is not None:
            np.random.seed(seed) # 设置随机种子
        self._current_step = 0 # 重置当前步数
        
        self._initialize_satellites() # 重新创建所有卫星实例，恢复其初始状态
        
        observations = self._get_all_observations() # 获取所有智能体的初始观测
        
        # 初始化 self.infos 字典
        self.infos = {agent_id: self._get_agent_info(agent_id) for agent_id in self.agents}
        self.infos["__common__"] = self._get_common_infos() # 包括当前步数、各队存活数量等
        
        return observations, self.infos 

    def _apply_thrust_and_kinematics(self, actions: TypingDict[str, np.ndarray]):
        """应用智能体的动作（推力），更新卫星的运动学状态（速度和位置）"""
        actual_applied_forces = {} # 记录实际施加的力，可用于奖励计算或信息记录
        
        # 1. 应用推力并更新速度 (基于 Satellite.update_kinematics_with_thrust)
        for agent_id, agent_action_vector in actions.items():
            sat = self.satellites.get(agent_id)
            if not sat or sat.is_destroyed: 
                actual_applied_forces[agent_id] = np.zeros(3, dtype=np.float32)
                continue
            
            # agent_action_vector[:3] 应该是归一化的动作 (例如 [-1,1] 范围)
            # self.action_scale 将其转换为物理单位的期望力
            desired_force_cmd = agent_action_vector[:3] * self.action_scale 
            
            # Satellite内部会处理最大推力限制和燃料消耗
            actual_force_vector = sat.update_kinematics_with_thrust(desired_force_cmd, self.step_time_interval)
            actual_applied_forces[agent_id] = actual_force_vector

        # 2. 在新的速度基础上，进行轨道传播以更新位置
        for agent_id in self.agents: 
            sat = self.satellites.get(agent_id)
            if not sat or sat.is_destroyed: continue

            if self.use_cw_dynamics: # 使用CW动力学模型
                # 仅当卫星在CW坐标系中有初始偏移或速度时才进行传播
                if np.any(sat.pos) or np.any(sat.vel): 
                    # R0_t, V0_t 为零，意味着CW方程描述的是相对于旋转坐标系中一个固定点的运动
                    propagator = sf.Clohessy_Wiltshire(R0_c=sat.pos, V0_c=sat.vel, 
                                                       R0_t=np.zeros(3), V0_t=np.zeros(3)) # 参考目标在CW原点静止
                    # State_transition_matrix 返回新的状态 [pos_new, vel_new]
                    s_new_flat, _ = propagator.State_transition_matrix(self.step_time_interval)
                    sat.pos, sat.vel = s_new_flat[0:3], s_new_flat[3:6] 
            else: # 使用简单的欧拉积分更新位置 (适用于非轨道或简化场景)
                sat.pos += sat.vel * self.step_time_interval
        return actual_applied_forces # 返回本步骤实际施加的力

    @abc.abstractmethod
    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], 
                                          actual_applied_forces: TypingDict[str, np.ndarray]):
        """处理特定于场景的动作，例如开火、扫描等 (由子类实现)"""
        pass
    
    @abc.abstractmethod
    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], 
                           actual_applied_forces: TypingDict[str, np.ndarray]
                          ) -> TypingDict[str, float]:
        """计算每个智能体的即时奖励 (由子类实现)"""
        pass
    
    @abc.abstractmethod
    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]:
        """检查回合是否因某些条件而终止 (terminated) 或截断 (truncated) (由子类实现)
           返回两个字典：terminated_dict, truncated_dict
        """
        pass
    
    @abc.abstractmethod
    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray:
        """获取指定智能体的观测 (由子类实现)"""
        pass
    
    def _get_all_observations(self) -> TypingDict[str, np.ndarray]:
        """获取所有智能体的观测字典"""
        obs_dict = {}
        for agent_id in self.agents:
            obs_dict[agent_id] = self._get_observation_for_agent(agent_id)
        return obs_dict

    def _get_agent_info(self, agent_id: str) -> TypingDict[str, Any]:
        """获取指定智能体的辅助信息（燃料、生命值、弹药、状态、位置、速度等）"""
        sat = self.satellites[agent_id]
        return {
            "fuel": sat.current_fuel, "health": sat.current_health,
            "ammo": sat.current_ammo if sat.can_attack else 0, 
            "is_destroyed": sat.is_destroyed,
            "pos": sat.pos.copy(), "vel": sat.vel.copy(), # 对numpy数组使用 .copy() 防止外部修改
            "mass": sat.mass, "max_thrust": sat.max_total_thrust,
            "type": sat.type # 添加卫星类型信息
        }

    def _get_common_infos(self) -> TypingDict[str, Any]:
        """获取环境的通用信息（当前步数、各队伍存活数量等）"""
        common_info = {"step": self._current_step}
        for team_id, agent_id_list in self.team_assignments.items():
            alive_count = sum(1 for aid in agent_id_list if not self.satellites[aid].is_destroyed)
            common_info[f"num_alive_{team_id}"] = alive_count
        return common_info

    def _get_teammate_ids(self, agent_id: str) -> List[str]:
        """获取指定智能体的所有队友ID (不包括自己)"""
        agent_sat = self.satellites.get(agent_id)
        if not agent_sat: return []
        current_team_id = agent_sat.team_id
        if current_team_id not in self.team_assignments: return []
        return [tid for tid in self.team_assignments[current_team_id] if tid != agent_id]

    def _get_opponent_ids(self, agent_id: str) -> List[str]:
        """获取指定智能体的所有对手ID"""
        agent_sat = self.satellites.get(agent_id)
        if not agent_sat: return []
        current_team_id = agent_sat.team_id
        opponent_ids = []
        for team_id, member_ids in self.team_assignments.items():
            if team_id != current_team_id:
                opponent_ids.extend(member_ids)
        return opponent_ids

    def step(self, actions: TypingDict[str, np.ndarray]) -> \
             Tuple[TypingDict[str, np.ndarray], TypingDict[str, float], 
                   TypingDict[str, bool], TypingDict[str, bool], TypingDict[str, Any]]:
        """环境执行一个时间步"""
        self._current_step += 1
        
        # self.infos 字典在每步开始时理论上包含上一步结束时的信息，或reset后的初始信息。
        # 它会在这一步的末尾被完全更新。

        # 1. 更新卫星内部状态（如冷却时间）
        for sat in self.satellites.values(): 
            sat.update_cooldowns()

        # 2. 应用推力，更新运动学状态 (位置，速度)
        actual_applied_forces = self._apply_thrust_and_kinematics(actions)
        
        # 3. 处理特定场景的动作 (如攻击)
        self._handle_scenario_specific_actions(actions, actual_applied_forces)
        
        # 4. 计算即时奖励 (基于当前状态和动作的结果)
        # 注意：子类中实现的 _calculate_rewards 返回的是即时奖励。
        # 终端奖励（如胜利奖励）由子类的 _check_episode_end -> _add_terminal_rewards (或类似方法)
        # 处理，并通常存储在子类的 self.current_rewards_cache 中。
        # 然后子类重写的 step 方法会将 cache 中的终端奖励加到这里的 immediate_rewards 上。
        rewards = self._calculate_rewards(actions, actual_applied_forces) 
        
        # 5. 检查回合是否结束 (终止或截断)
        # 此方法可能会更新 self.infos["__common__"] (例如添加胜利者信息)
        # 并且，在子类中，它可能会填充用于终端奖励的 self.current_rewards_cache
        terminated, truncated = self._check_episode_end() 

        # 6. 获取新的观测
        observations = self._get_all_observations()

        # 7. 更新 self.infos 以反映当前步骤执行完毕后的状态
        for agent_id in self.agents: 
            self.infos[agent_id] = self._get_agent_info(agent_id) 
        
        # _check_episode_end 可能已经更新了 common_info 的一部分 (如winner, termination_reason)
        # _get_common_infos() 会获取最新的步数和存活数量等。
        # 合并更新，确保所有信息都是最新的。
        current_common_info = self._get_common_infos()
        if "__common__" in self.infos: # 如果 _check_episode_end 已写入
            self.infos["__common__"].update(current_common_info)
        else:
            self.infos["__common__"] = current_common_info
        
        # 注意：子类重写的 step 方法负责将 self.current_rewards_cache 中的终端奖励合并到返回的 rewards 中。
        return observations, rewards, terminated, truncated, self.infos

    def get_env_info(self) -> TypingDict[str, Any]:
        """返回环境的静态信息（空间、智能体数量等）"""
        return {
            'state_space': self.state_space, 
            'observation_space': self.observation_space,
            'action_space': self.action_space, 
            'agents': self.agents, 
            'num_agents': self.n_agents,
            'max_episode_steps': self.max_episode_steps, 
            'team_assignments': self.team_assignments
        }

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染环境状态"""
        if mode == 'human': # 文本模式渲染到控制台
            common_info_to_render = self.infos.get("__common__", self._get_common_infos()) # 获取最新的通用信息
            print(f"--- 步骤: {common_info_to_render.get('step', self._current_step)} ---")
            for team_id_key in self.team_assignments.keys():
                print(f"队伍 {team_id_key} 存活: {common_info_to_render.get(f'num_alive_{team_id_key}', 'N/A')}")
            if "winner" in common_info_to_render: 
                print(f"胜利者: {common_info_to_render['winner']}")
            if "termination_reason" in common_info_to_render: 
                print(f"结束原因: {common_info_to_render['termination_reason']}")

            for agent_id in self.agents:
                sat = self.satellites[agent_id]
                # 优先使用 infos 中的信息，如果 infos 不可用或不包含该智能体信息，则直接从卫星对象获取
                agent_info_render = self.infos.get(agent_id, self._get_agent_info(agent_id)) 
                
                hp = agent_info_render.get('health', sat.current_health)
                fuel = agent_info_render.get('fuel', sat.current_fuel)
                ammo = agent_info_render.get('ammo', sat.current_ammo if sat.can_attack else 0)
                status = "已摧毁" if agent_info_render.get('is_destroyed', sat.is_destroyed) else f"HP:{hp:.0f} Fuel:{fuel:.0f} Ammo:{ammo}"
                print(f"  {sat.id} (队伍: {sat.team_id}, 类型: {sat.type}): "
                      f"位置={sat.pos.round(0)}, 速度={sat.vel.round(1)}, {status}")
        
        elif mode == 'rgb_array': # 返回一个代表环境状态的RGB图像数组 (简化2D渲染)
            canvas_size=200 # 画布大小 (像素)
            world_scale=0.00002 # 世界坐标到画布坐标的缩放因子 (根据卫星典型位置调整)
            canvas=np.ones((canvas_size,canvas_size,3),dtype=np.uint8)*240 # 浅灰色背景 (RGB)
            
            # 定义队伍颜色 (BGR格式，因为OpenCV常用BGR，但这里直接用RGB也可以)
            team_colors={"team_A":[0,0,200],"team_B":[200,0,0],"default_team":[100,100,100]} 
            
            def world_to_canvas(pos_xy: np.ndarray) -> Tuple[int, int]:
                """将世界坐标系下的X,Y位置转换为画布像素坐标"""
                # 画布中心为原点，Y轴向上为正 (标准笛卡尔)，但图像Y轴向下为正，故对Y坐标取反
                cx = int(canvas_size/2 + pos_xy[0] * world_scale)
                cy = int(canvas_size/2 - pos_xy[1] * world_scale) # Y轴反转
                return np.clip(cx, 0, canvas_size-1), np.clip(cy, 0, canvas_size-1)

            for sat_id in self.agents: 
                sat = self.satellites[sat_id]
                if not sat.is_destroyed:
                    # 基于X,Y位置渲染；此2D视图忽略Z轴
                    cx,cy = world_to_canvas(sat.pos) 
                    color = team_colors.get(sat.team_id, team_colors["default_team"]) # 获取队伍颜色
                    # 根据生命值百分比确定半径大小 (例如，满血时半径为3像素，低血量时为1像素)
                    radius = max(1, int(3 * (sat.current_health / sat.max_health if sat.max_health > 0 else 0.1))) 
                    
                    # 绘制一个近似的填充圆形
                    for r_offset in range(-radius, radius + 1):
                        for c_offset in range(-radius, radius + 1):
                            if r_offset**2 + c_offset**2 <= radius**2: # 点在圆内
                                # 确保绘制在画布边界内
                                plot_y = np.clip(cy + r_offset, 0, canvas_size - 1)
                                plot_x = np.clip(cx + c_offset, 0, canvas_size - 1)
                                canvas[plot_y, plot_x] = color
            return canvas
        return None # 其他模式暂不支持

    def close(self): 
        """关闭环境，释放资源 (例如渲染窗口)"""
        if self.viewer is not None:
            # self.viewer.close() # 如果使用Pyglet等图形库
            self.viewer = None
        pass # 目前无特殊资源需清理

    def avail_actions(self) -> TypingDict[str, Optional[np.ndarray]]: 
        """返回每个智能体的可用动作掩码。
           对于连续动作空间，通常返回 None (所有动作可用)。
           对于离散动作空间，返回一个布尔或0/1数组，标记哪些动作可用。
        """
        return {aid: None for aid in self.agents} # 默认所有连续动作可用

    def agent_mask(self) -> TypingDict[str, bool]: 
        """返回一个布尔掩码，指示哪些智能体当前是活跃的 (未被摧毁)"""
        return {aid: not self.satellites[aid].is_destroyed for aid in self.agents}

    def state(self) -> Optional[np.ndarray]:
        """
        返回全局状态表示。
        默认情况下，按排序后的智能体ID顺序拼接所有智能体的观测。
        如果具体场景定义了更特定的全局状态，则应重写此方法。
        如果 self.state_space 被定义，此方法理想情况下应返回与该空间形状和类型匹配的状态。
        """
        if not self.agents: # 如果没有智能体，则没有状态
            if self.state_space is not None: # 如果定义了状态空间，返回匹配该空间的空状态
                return np.zeros(self.state_space.shape, dtype=self.state_space.dtype)
            return np.array([], dtype=np.float32) # 否则返回空数组

        obs_list = []
        # 对智能体ID进行排序，以确保拼接顺序的一致性
        # 这对于期望固定顺序状态向量的MARL算法至关重要
        sorted_agent_ids = sorted(self.agents) 
        
        for agent_id in sorted_agent_ids:
            agent_obs = self._get_observation_for_agent(agent_id)
            obs_list.append(agent_obs.flatten()) # 如果观测是多维的，则展平

        try:
            concatenated_state = np.concatenate(obs_list).astype(np.float32)
            # 可选：如果定义了 self.state_space，则验证拼接后的状态形状是否匹配
            if self.state_space is not None and concatenated_state.shape != self.state_space.shape:
                # 这可能表示拼接观测与定义的状态空间之间存在不匹配。
                # 目前，我们将返回拼接的观测。需要特定状态结构（而非简单拼接）的场景应重写 state()。
                # print(f"警告: 拼接状态形状 {concatenated_state.shape} "
                #       f"与定义的状态空间形状 {self.state_space.shape} 不匹配。"
                #       f"返回拼接的观测。")
                pass # 继续使用拼接的状态
            return concatenated_state
        except ValueError: 
            # 如果拼接失败（例如 obs_list 为空，尽管上面已处理；或单个观测不适合直接拼接）
            # 尝试返回一个期望总观测维度的零向量（如果可能）
            if self.observation_space and sorted_agent_ids and sorted_agent_ids[0] in self.observation_space:
                sample_obs_space = self.observation_space[sorted_agent_ids[0]]
                if sample_obs_space is not None and hasattr(sample_obs_space, 'shape') and sample_obs_space.shape is not None:
                    try:
                        # 从一个智能体的观测空间计算期望的总维度
                        single_obs_dim = np.prod(sample_obs_space.shape) # 处理多维观测
                        return np.zeros(single_obs_dim * self.n_agents, dtype=np.float32)
                    except TypeError: # 处理 shape 为 () 的标量情况
                         return np.zeros(self.n_agents, dtype=np.float32) if sample_obs_space.shape == () else np.array([], dtype=np.float32)
            return np.array([], dtype=np.float32) # 最终的后备方案：空数组


# -------------------------------------------------------------------------------------------
# 3. 具体任务场景类: OneOnOnePursuitEnv (一对一追逐场景)
#    V3.5 更新: 修复终端奖励合并逻辑, 修正动作空间定义
# -------------------------------------------------------------------------------------------
class OneOnOnePursuitEnv(MultiSatelliteEnvBase):
    def __init__(self, env_config: TypingDict[str, Any], scenario_config: TypingDict[str, Any]):
        self.pursuer_id = scenario_config.get("pursuer_id", "pursuer_0") # 追逐者ID
        self.evader_id = scenario_config.get("evader_id", "evader_0")   # 逃跑者ID
        super(OneOnOnePursuitEnv, self).__init__(env_config, scenario_config) # 调用父类构造函数
        
        # 奖励相关参数
        self.capture_reward = float(self.scenario_config.get("capture_reward", 200.0)) # 捕获成功奖励
        self.evasion_reward = float(self.scenario_config.get("evasion_reward", 200.0)) # 逃跑成功奖励 (追逐者失败)
        self.distance_reward_scale = float(self.scenario_config.get("distance_reward_scale", 0.001)) # 距离变化奖励的缩放因子
        self.time_penalty = float(self.scenario_config.get("time_penalty_1v1", -0.1)) # 每步的时间惩罚 (应为负值)
        self.d_capture = float(self.scenario_config.get("d_capture_1v1", 10000.0)) # 定义捕获成功的距离阈值
        
        self.last_distance = np.inf # 上一步的两者间距离，用于计算距离变化奖励
        # self.current_rewards_cache 用于存储回合结束时的终端奖励，这些奖励会在step()方法中加到最后一步的即时奖励上
        self.current_rewards_cache: TypingDict[str, float] = {aid: 0.0 for aid in self.agents}


    def _initialize_teams_and_agents(self): 
        """定义队伍和智能体 (追逐者和逃跑者各属一队)"""
        self.pursuer_team_name = self.scenario_config.get("pursuer_team_name", "team_A") # 追逐者队伍名
        self.evader_team_name = self.scenario_config.get("evader_team_name", "team_B")   # 逃跑者队伍名
        self.team_assignments = {
            self.pursuer_team_name: [self.pursuer_id],
            self.evader_team_name: [self.evader_id]
        }
        self.agents = [self.pursuer_id, self.evader_id]
        self.agent_ids = self.agents # 为兼容MARL框架
        self.n_agents = 2

    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, 
                                           agent_idx_in_team: int, type_config: TypingDict
                                          ) -> Tuple[np.ndarray, np.ndarray]: 
        """根据场景配置设置追逐者和逃跑者的初始位置和速度"""
        if agent_id == self.pursuer_id:
            pos = np.array(self.scenario_config.get("initial_pursuer_pos", [-100000.0, 0.0, 0.0]), dtype=np.float32)
            vel = np.array(self.scenario_config.get("initial_pursuer_vel", [10.0, 0.0, 0.0]), dtype=np.float32)
        else: # 逃跑者
            pos = np.array(self.scenario_config.get("initial_evader_pos", [100000.0, 0.0, 0.0]), dtype=np.float32)
            vel = np.array(self.scenario_config.get("initial_evader_vel", [-10.0, 0.0, 0.0]), dtype=np.float32)
        return pos, vel

    def _define_spaces(self): 
        """定义观测空间和动作空间"""
        # 观测向量组成: 相对位置(3), 相对速度(3), 追逐者绝对位置(3), 追逐者绝对速度(3), 逃跑者绝对位置(3), 逃跑者绝对速度(3)
        obs_dim = 3 + 3 + 3 + 3 + 3 + 3 # 共18维
        common_obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = {aid: common_obs_space for aid in self.agents}
        
        # 在这个完全可观测的1v1场景中，全局状态与单个智能体的观测相同
        self.state_space = common_obs_space 
        
        act_dim = 3 # 动作是三维推力 (x, y, z)
        # 动作空间通常是归一化的，例如 [-1, 1]。
        # MultiSatelliteEnvBase 中的 self.action_scale 会将这个归一化动作缩放到物理单位。
        common_act_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.action_space = {aid: common_act_space for aid in self.agents}


    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray: 
        """获取1v1追逐场景的观测 (对两个智能体都相同)"""
        pursuer = self.satellites[self.pursuer_id]
        evader = self.satellites[self.evader_id]

        # 如果回合因一方被摧毁而实际结束 (虽然1v1追逐主要靠捕获/燃料耗尽结束)
        if pursuer.is_destroyed or evader.is_destroyed: 
             return np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)

        rel_pos = pursuer.pos - evader.pos # 逃跑者指向追逐者的向量
        rel_vel = pursuer.vel - evader.vel

        # 在此场景中，两个智能体都观察到相同的全局状态信息
        return np.concatenate([
            rel_pos, rel_vel, 
            pursuer.pos, pursuer.vel, 
            evader.pos, evader.vel
        ]).astype(np.float32)

    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], 
                                          actual_applied_forces: TypingDict[str, np.ndarray]):
        """1v1追逐场景中，除了推力之外没有其他特定动作 (例如没有武器)"""
        pass

    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], 
                           actual_applied_forces: TypingDict[str, np.ndarray]
                          ) -> TypingDict[str, float]:
        """计算1v1追逐的即时奖励。终端奖励由 _add_terminal_rewards 处理并在 step() 中合并。"""
        step_rewards = {aid: 0.0 for aid in self.agents}
        pursuer = self.satellites[self.pursuer_id]
        evader = self.satellites[self.evader_id]

        if pursuer.is_destroyed or evader.is_destroyed: # 如果一方已被摧毁，则不计算奖励
            return step_rewards

        current_distance = np.linalg.norm(pursuer.pos - evader.pos) # 当前距离

        # 距离变化奖励: 追逐者因距离缩短而获益，逃跑者因距离拉大而获益
        distance_delta = self.last_distance - current_distance # 为正表示距离缩短
        step_rewards[self.pursuer_id] += distance_delta * self.distance_reward_scale
        step_rewards[self.evader_id] -= distance_delta * self.distance_reward_scale # 逃跑者目标是拉大距离

        # 时间惩罚 (双方都受到)
        step_rewards[self.pursuer_id] += self.time_penalty 
        step_rewards[self.evader_id] += self.time_penalty 
        
        # 燃料/推力惩罚 (与施加的推力大小成正比，鼓励节省能量)
        # explicit_force_penalty_factor_1v1 在配置中应为正数，这里通过减法施加惩罚
        force_penalty_factor = self.scenario_config.get("explicit_force_penalty_factor_1v1", 0.0001)
        if self.pursuer_id in actual_applied_forces:
            pursuer_force_mag = np.linalg.norm(actual_applied_forces[self.pursuer_id])
            step_rewards[self.pursuer_id] -= pursuer_force_mag * force_penalty_factor
        if self.evader_id in actual_applied_forces:
            evader_force_mag = np.linalg.norm(actual_applied_forces[self.evader_id])
            step_rewards[self.evader_id] -= evader_force_mag * force_penalty_factor
            
        self.last_distance = current_distance # 更新上一步距离
        return step_rewards

    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]: 
        """检查1v1追逐场景的结束条件"""
        terminated = {aid: False for aid in self.agents} # 是否终止
        truncated = {aid: False for aid in self.agents}  # 是否截断 (例如超时)
        terminated["__all__"] = False; truncated["__all__"] = False # 全局结束标志
        
        pursuer = self.satellites[self.pursuer_id]; evader = self.satellites[self.evader_id]
        
        # 此方法会填充 self.current_rewards_cache (用于终端奖励)
        # 并在 self.infos["__common__"] 中设置回合结束原因和胜利者
        
        termination_reason = None # 初始化回合结束原因

        # 首先检查是否有卫星被摧毁 (虽然在标准1v1追逐中不常见，但 Satellite 类支持此状态)
        if pursuer.is_destroyed: 
            terminated["__all__"] = True
            self._add_terminal_rewards(evader_wins_by_pursuer_fail=True) # 逃跑者因追逐者失败而获胜
            termination_reason = "pursuer_destroyed"
        elif evader.is_destroyed: 
            terminated["__all__"] = True
            self._add_terminal_rewards(capture=True) # 逃跑者被摧毁视为捕获成功
            termination_reason = "evader_destroyed"
        else: # 如果双方都还存活，检查其他追逐特定条件
            current_distance = np.linalg.norm(pursuer.pos - evader.pos) 
            if current_distance <= self.d_capture: # 捕获成功
                terminated["__all__"] = True
                self._add_terminal_rewards(capture=True)
                termination_reason = "capture"
            elif pursuer.current_fuel <= 0 and not terminated["__all__"]: # 追逐者燃料耗尽
                terminated["__all__"] = True
                self._add_terminal_rewards(pursuer_fuel_out=True) # 逃跑者获胜
                termination_reason = "pursuer_fuel_out"
            elif evader.current_fuel <= 0 and not terminated["__all__"]: # 逃跑者燃料耗尽
                terminated["__all__"] = True 
                self._add_terminal_rewards(evader_fuel_out=True) # 追逐者获胜
                termination_reason = "evader_fuel_out"
        
        # 检查是否超时 (如果尚未因其他原因终止)
        if self._current_step >= self.max_episode_steps and not terminated["__all__"]:
            truncated["__all__"] = True
            self._add_terminal_rewards(timeout=True) # 超时通常算逃跑者成功
            termination_reason = "timeout"
        
        # 如果回合结束 (终止或截断)
        if terminated["__all__"] or truncated["__all__"]:
            for aid in self.agents: # 同步所有智能体的结束状态
                terminated[aid] = terminated["__all__"]
                truncated[aid] = truncated["__all__"]
            
            if "__common__" not in self.infos: self.infos["__common__"] = {} # 确保 common info 字典存在
            if termination_reason: self.infos["__common__"]["termination_reason"] = termination_reason
            
            # 根据结束原因判断胜利者
            if termination_reason in ["capture", "evader_fuel_out", "evader_destroyed"]:
                self.infos["__common__"]["winner"] = self.pursuer_id # 追逐者胜利
            elif termination_reason in ["pursuer_fuel_out", "timeout", "pursuer_destroyed"]:
                 self.infos["__common__"]["winner"] = self.evader_id # 逃跑者胜利
            else: # 其他情况或平局 (例如，如果未来添加了双方同时耗尽燃料等)
                 self.infos["__common__"]["winner"] = "draw"

        return terminated, truncated

    def _add_terminal_rewards(self, capture=False, pursuer_fuel_out=False, 
                              evader_fuel_out=False, timeout=False, 
                              evader_wins_by_pursuer_fail=False): 
        """根据结束条件，填充 self.current_rewards_cache 以存储终端奖励。
           这些缓存的奖励会在 step() 方法中被加到即时奖励上。
        """
        # 在添加新的终端奖励前，清空缓存 (确保只包含当前回合结束事件的奖励)
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents}

        if capture: # 追逐者捕获逃跑者
            self.current_rewards_cache[self.pursuer_id] += self.capture_reward
            self.current_rewards_cache[self.evader_id] -= self.capture_reward # 逃跑者因被捕获而受惩罚
        elif pursuer_fuel_out or timeout or evader_wins_by_pursuer_fail: # 逃跑者 "获胜" 的几种情况
            self.current_rewards_cache[self.evader_id] += self.evasion_reward
            self.current_rewards_cache[self.pursuer_id] -= self.evasion_reward # 追逐者因未能捕获而受惩罚
        elif evader_fuel_out: # 逃跑者燃料耗尽，视为追逐者获胜 (奖励可能低于直接捕获)
             self.current_rewards_cache[self.pursuer_id] += self.capture_reward * 0.75 # 按比例缩放的捕获奖励
             self.current_rewards_cache[self.evader_id] -= self.capture_reward * 0.75
            
    def step(self, actions: TypingDict[str, np.ndarray]): 
        """执行一步环境逻辑，并合并终端奖励"""
        # self.current_rewards_cache 会在 _add_terminal_rewards (如果回合结束) 中被填充。
        # 它仅用于存储当前步骤的终端奖励（如果这是回合的最后一步）。
        
        obs, immediate_rewards, terminated, truncated, info = super().step(actions) # 调用父类的step获取即时结果
        
        # 如果本步骤导致回合结束，则将缓存的终端奖励加到即时奖励上
        if terminated.get("__all__", False) or truncated.get("__all__", False): # 使用 .get 避免KeyError
            for agent_id in self.agents:
                immediate_rewards[agent_id] += self.current_rewards_cache.get(agent_id, 0.0)
        
        return obs, immediate_rewards, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None): 
        """重置环境，并初始化场景特定状态"""
        obs_dict, info_dict = super().reset(seed, options) # 调用父类reset，会初始化卫星、观测、信息等
        
        # 在卫星被初始化后，重置场景特定的状态
        pursuer = self.satellites[self.pursuer_id]; evader = self.satellites[self.evader_id]
        self.last_distance = np.linalg.norm(pursuer.pos - evader.pos) # 初始化上一步距离
        
        # 为新回合清空终端奖励缓存
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents} 
        
        # self.infos 已被 super().reset() 更新为最新的智能体和通用信息
        return obs_dict, self.infos

# -------------------------------------------------------------------------------------------
# 3. 具体任务场景类: ManyVsManyCombatEnv (多对多对抗场景)
#    V3.5 更新: 修复终端奖励合并, 修正动作/观测空间维度, 修复 attacker_id 笔误
# -------------------------------------------------------------------------------------------
class ManyVsManyCombatEnv(MultiSatelliteEnvBase):
    def __init__(self, env_config: TypingDict[str, Any], scenario_config: TypingDict[str, Any]):
        # 调用父类构造函数会触发 _initialize_teams_and_agents, _initialize_satellites, _define_spaces
        super(ManyVsManyCombatEnv, self).__init__(env_config, scenario_config) 
        
        # 加载战斗场景的奖励配置
        self.reward_config = self.scenario_config.get("reward_config_combat", {
            "damage_dealt_factor": 10.0,        # 对敌造成伤害的奖励因子
            "team_damage_dealt_factor": 2.0,    # 队友对敌造成伤害的团队奖励因子
            "destroyed_enemy_factor": 100.0,    # 摧毁敌人的奖励因子
            "team_destroyed_enemy_factor": 20.0,# 队友摧毁敌人的团队奖励因子
            "health_lost_penalty_factor": -0.5, # 自身损失生命值的惩罚因子 (应为负)
            "ally_destroyed_penalty_factor": -50.0, # 队友被摧毁的惩罚因子 (应为负)
            "explicit_force_penalty_factor": self.scenario_config.get("explicit_force_penalty_factor_combat", -0.0001), # 使用推力的惩罚 (应为负)
            "ammo_consumption_penalty_factor": -0.1, # 消耗弹药的惩罚因子 (应为负)
            "time_penalty_factor": -0.01,       # 时间流逝惩罚因子 (应为负)
            "win_bonus": 500.0,                 # 胜利奖励
            "lose_penalty": -500.0,             # 失败惩罚 (应为负)
        })
        self.last_known_enemy_targets: TypingDict[str, int] = {} # 记录每个智能体上一步尝试攻击的目标索引 (在对手列表中的索引)
        self.damage_info_this_step: TypingDict[str, List[TypingDict[str, Any]]] = {} # 记录本步骤中各攻击者造成的伤害事件
        
        # self.current_rewards_cache 用于存储回合结束时的终端奖励 (胜利/失败奖励/惩罚)
        # 在 __init__ 中初始化，确保 self.agents 已经由父类初始化完毕
        self.current_rewards_cache: TypingDict[str, float] = {aid: 0.0 for aid in self.agents}


    def _initialize_teams_and_agents(self): 
        """初始化队伍和智能体数量及ID"""
        self.num_team_A = int(self.scenario_config.get("num_team_A", 3)) # A队智能体数量
        self.num_team_B = int(self.scenario_config.get("num_team_B", 3)) # B队智能体数量
        
        self.n_agents = self.num_team_A + self.num_team_B # 总智能体数量
        self.team_A_ids = [f"teamA_{i}" for i in range(self.num_team_A)] # A队智能体ID列表
        self.team_B_ids = [f"teamB_{i}" for i in range(self.num_team_B)] # B队智能体ID列表
        self.agents = self.team_A_ids + self.team_B_ids # 所有智能体ID列表
        self.agent_ids = self.agents
        self.team_assignments = {"team_A": self.team_A_ids, "team_B": self.team_B_ids} # 队伍分配

    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, 
                                           agent_idx_in_team: int, type_config: TypingDict
                                          ) -> Tuple[np.ndarray, np.ndarray]: 
        """为多对多场景设置卫星的初始位置和速度，使两队在战场两侧对峙"""
        spacing = float(self.scenario_config.get("initial_spacing", 20000.0)) # 智能体间距
        team_offset_x = float(self.scenario_config.get("initial_team_offset_x", 250000.0)) # 队伍在X轴上的偏移
        
        num_in_this_team = self.num_team_A if team_id == "team_A" else self.num_team_B # 当前队伍的智能体数量
        
        if team_id == "team_A": 
            pos_x = -team_offset_x - agent_idx_in_team * spacing # A队在X轴负方向
            vel_x = np.random.uniform(5, 15) # A队初始向X轴正方向运动
        else: # team_B
            pos_x = team_offset_x + agent_idx_in_team * spacing # B队在X轴正方向
            vel_x = np.random.uniform(-15, -5) # B队初始向X轴负方向运动
        
        # 在Y,Z平面上散开，并加入随机性
        pos_y = np.random.uniform(-50000,50000) + (agent_idx_in_team - num_in_this_team / 2.0) * spacing * 0.5
        pos_z = np.random.uniform(-10000,10000)
        vel_y = np.random.uniform(-5,5)
        vel_z = np.random.uniform(-2,2)
        
        return np.array([pos_x,pos_y,pos_z]), np.array([vel_x,vel_y,vel_z])

    def _define_spaces(self): 
        """定义多对多战斗的观测空间和动作空间"""
        self.observe_n_closest_enemies = int(self.scenario_config.get("observe_n_closest_enemies",3)) # 观测最近敌人的数量
        self.observe_n_closest_teammates = int(self.scenario_config.get("observe_n_closest_teammates",2)) # 观测最近队友的数量
        
        # 从 Satellite 类获取自身观测和相对观测的维度
        self.self_obs_dim = 10  # 来自 get_self_observation_component()
        self.other_sat_obs_dim = 10 # 来自 get_relative_observation_to() (已从8修正为10)
        
        # 每个智能体的总观测维度
        obs_dim_per_agent = (self.self_obs_dim + 
                             self.observe_n_closest_enemies * self.other_sat_obs_dim + 
                             self.observe_n_closest_teammates * self.other_sat_obs_dim)
        
        common_obs_s = Box(low=-np.inf,high=np.inf,shape=(obs_dim_per_agent,),dtype=np.float32)
        self.observation_space = {aid: common_obs_s for aid in self.agents}
        
        # 全局状态空间:
        # 对于多对多场景，全局状态可以是所有智能体观测的拼接，或更结构化的表示。
        # 默认情况下，基类的 state() 方法会通过拼接观测来生成。
        # 如果需要特定的全局状态结构，应在此处定义 self.state_space，并重写本类的 state() 方法。
        # 例如: self.state_space = Box(low=-np.inf, high=np.inf, shape=(self.n_agents * obs_dim_per_agent,), dtype=np.float32)
        # 为简单起见，当前让基类 state() 处理拼接，或如果需要特定状态则子类重写。

        # 动作空间: 3维用于推力，外加N维用于目标选择的 logits
        # N 是智能体可能需要从中选择的最大对手数量。
        self.max_opponents_for_action = 0
        # _initialize_teams_and_agents 应该已经设置了 num_team_A 和 num_team_B
        if hasattr(self, 'num_team_A') and hasattr(self, 'num_team_B'): 
             if self.num_team_A > 0 and self.num_team_B > 0: # 仅当双方都有智能体时才有意义
                self.max_opponents_for_action = max(self.num_team_A, self.num_team_B)
        
        action_target_dim = max(1, self.max_opponents_for_action) # 至少有1个目标logit (即使没有对手，也可能用于“不攻击”或虚拟目标)
        action_dim_per_agent = 3 + action_target_dim # 3维推力 + 目标选择logits
        
        # 推力动作归一化到 [-1, 1]
        thrust_low = -np.ones(3, dtype=np.float32); thrust_high = np.ones(3, dtype=np.float32)
        # 目标选择 logits 也可以在归一化范围内，例如 [-1, 1] 或任何适合 logits 的范围。
        target_low = -np.ones(action_target_dim, dtype=np.float32) 
        target_high = np.ones(action_target_dim, dtype=np.float32) 
        
        common_act_s = Box(low=np.concatenate([thrust_low, target_low]), 
                           high=np.concatenate([thrust_high, target_high]), dtype=np.float32)
        self.action_space = {aid: common_act_s for aid in self.agents}


    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray: 
        """为战斗场景中的智能体构建观测"""
        agent_sat = self.satellites[agent_id]
        if agent_sat.is_destroyed: 
            # 如果智能体被摧毁，返回正确形状的全零观测
            return np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)

        own_obs = agent_sat.get_self_observation_component() # 自身观测 (形状: self_obs_dim)
        enemy_obs_list, teammate_obs_list = [], []
        
        # 观测最近的N个敌人
        opp_ids_all = self._get_opponent_ids(agent_id) 
        live_opp_with_dist = [] # 存储 (距离, 敌人ID)
        for oid in opp_ids_all:
            opp_sat = self.satellites.get(oid) 
            if opp_sat and not opp_sat.is_destroyed: # 检查敌人是否存活
                dist_to_opp = np.linalg.norm(agent_sat.pos - opp_sat.pos)
                if agent_sat.can_observe and dist_to_opp <= agent_sat.sensor_range: # 检查是否在传感器范围内
                    live_opp_with_dist.append((dist_to_opp, oid))
        
        live_opp_with_dist.sort(key=lambda x: x[0]) # 按距离排序
        for i in range(self.observe_n_closest_enemies): 
            if i < len(live_opp_with_dist): # 如果有足够的敌人
                enemy_sat = self.satellites[live_opp_with_dist[i][1]]
                enemy_obs_list.append(agent_sat.get_relative_observation_to(enemy_sat))
            else: # 如果敌人数量不足，用零向量填充
                enemy_obs_list.append(np.zeros(self.other_sat_obs_dim, dtype=np.float32))
        
        # 观测最近的M个队友 (不包括自己)
        mate_ids_all = self._get_teammate_ids(agent_id) 
        live_mate_with_dist = [] # 存储 (距离, 队友ID)
        for mid in mate_ids_all:
            mate_sat = self.satellites.get(mid)
            if mate_sat and not mate_sat.is_destroyed: # 检查队友是否存活
                 # 假设队友信息总是可观测的，或者可以为其设置不同的传感器逻辑/范围
                 live_mate_with_dist.append((np.linalg.norm(agent_sat.pos - mate_sat.pos), mid))
        
        live_mate_with_dist.sort(key=lambda x: x[0]) # 按距离排序
        for i in range(self.observe_n_closest_teammates): 
            if i < len(live_mate_with_dist): # 如果有足够的队友
                teammate_sat = self.satellites[live_mate_with_dist[i][1]]
                teammate_obs_list.append(agent_sat.get_relative_observation_to(teammate_sat))
            else: # 如果队友数量不足，用零向量填充
                teammate_obs_list.append(np.zeros(self.other_sat_obs_dim, dtype=np.float32))
            
        # 拼接所有观测部分
        # 确保所有子观测在拼接前都是一维数组
        flat_enemy_obs = [obs.flatten() for obs in enemy_obs_list]
        flat_teammate_obs = [obs.flatten() for obs in teammate_obs_list]

        return np.concatenate([own_obs.flatten()] + flat_enemy_obs + flat_teammate_obs).astype(np.float32)


    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], 
                                          actual_applied_forces: TypingDict[str, np.ndarray]):
        """处理战斗场景中的攻击动作"""
        self.damage_info_this_step.clear() # 清除上一步的伤害信息

        for agent_id, agent_action_vector in actions.items():
            attacker = self.satellites[agent_id] # 获取攻击方卫星
            
            # 如果攻击方已被摧毁或不能开火 (例如冷却中、没弹药)
            if attacker.is_destroyed or not attacker.can_fire_weapon():
                self.last_known_enemy_targets[agent_id] = -1 # 标记为没有有效目标或无法开火
                continue

            target_logits = agent_action_vector[3:] # 假设动作向量前3个元素是推力，其余是目标选择logits
            
            # 获取当前攻击者的所有对手ID列表
            # 重要: 如果logits通过索引映射到这些对手，则此列表的顺序必须一致。
            # 更稳健的方法可能是让动作包含目标ID，或将logits映射到所有可能敌人槽位的固定列表。
            # 为简单起见，我们使用 _get_opponent_ids() 返回的顺序，但这可能导致非平稳性（如果对手列表顺序频繁变化）。
            # 一种常见方法：按字母顺序或固定方案对对手ID进行排序。
            opponents_for_attacker = sorted(self._get_opponent_ids(agent_id)) # 排序以获得一定程度的一致性

            chosen_target_idx_in_opp_list = -1 # 在 (排序后的) 对手列表中的选定目标索引
            highest_logit = -np.inf # 用于选择logit值最高的目标

            if not opponents_for_attacker: # 如果没有可攻击的对手
                 self.last_known_enemy_targets[agent_id] = -1
                 continue

            # 遍历logits，最多处理 可用对手数量 和 logits长度 中的较小者
            num_potential_targets = min(len(target_logits), len(opponents_for_attacker))

            for i in range(num_potential_targets):
                actual_opponent_id = opponents_for_attacker[i] # 根据排序列表获取实际对手ID
                if not self.satellites[actual_opponent_id].is_destroyed: # 只能攻击存活的对手
                    if target_logits[i] > highest_logit:
                        highest_logit = target_logits[i]
                        chosen_target_idx_in_opp_list = i 
            
            self.last_known_enemy_targets[agent_id] = chosen_target_idx_in_opp_list # 记录选择的目标索引

            if chosen_target_idx_in_opp_list != -1: # 如果成功选择了目标
                target_id = opponents_for_attacker[chosen_target_idx_in_opp_list] # 获取目标卫星的ID
                target_sat = self.satellites[target_id] # 获取目标卫星对象
                
                # 检查目标是否在武器射程内
                if np.linalg.norm(attacker.pos - target_sat.pos) <= attacker.weapon_range:
                    if attacker.fire_weapon_at_target(): # 尝试开火 (此方法会消耗弹药并设置冷却)
                        damage_amount = attacker.weapon_damage # 获取武器伤害值
                        target_sat.take_damage(damage_amount) # 对目标造成伤害 (目标生命值更新，is_destroyed可能改变)
                        
                        # 记录伤害事件
                        if agent_id not in self.damage_info_this_step: # 已修正: 使用 agent_id
                            self.damage_info_this_step[agent_id] = []
                        self.damage_info_this_step[agent_id].append({
                            "target_id": target_id, 
                            "damage": damage_amount, 
                            "destroyed_target": target_sat.is_destroyed # 记录造成伤害后目标是否被摧毁
                        })

    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], 
                           actual_applied_forces: TypingDict[str, np.ndarray]
                          ) -> TypingDict[str, float]:
        """计算战斗场景的即时奖励。终端奖励由 _add_win_lose_rewards_combat 处理并在 step() 中合并。"""
        rewards = {aid: 0.0 for aid in self.agents} # 初始化所有智能体的奖励为0
        rc = self.reward_config # 奖励配置的缩写

        # --- 基于动作和即时结果的个体和团队奖励/惩罚 ---
        for agent_id in self.agents:
            sat = self.satellites[agent_id]
            if sat.is_destroyed: continue # 被摧毁的智能体不计算奖励

            # 1. 使用推力的惩罚 (燃料消耗由Satellite类隐式处理，这里是显式推力惩罚)
            if agent_id in actual_applied_forces:
                force_magnitude = np.linalg.norm(actual_applied_forces[agent_id])
                # explicit_force_penalty_factor 在配置中应为负值，或在此处乘以-1
                rewards[agent_id] += force_magnitude * rc.get("explicit_force_penalty_factor", 0.0) 

            # 2. 时间惩罚
            # time_penalty_factor 在配置中应为负值
            rewards[agent_id] += rc.get("time_penalty_factor",0.0) 

            # 3. 战斗行为的奖励/惩罚 (造成伤害、摧毁敌人)
            if agent_id in self.damage_info_this_step: # 如果该智能体在本步骤中造成了伤害
                num_successful_shots_this_agent = 0 # 记录本智能体成功射击次数
                for attack_event in self.damage_info_this_step[agent_id]:
                    num_successful_shots_this_agent +=1
                    
                    # 对该智能体造成伤害的奖励
                    rewards[agent_id] += attack_event["damage"] * rc.get("damage_dealt_factor",0.0)
                    
                    # 对该智能体造成伤害的团队奖励 (给予所有存活的队友)
                    for teammate_id in self._get_teammate_ids(agent_id): 
                        if not self.satellites[teammate_id].is_destroyed:
                            rewards[teammate_id] += attack_event["damage"] * rc.get("team_damage_dealt_factor",0.0)
                    
                    # 摧毁敌人的奖励
                    if attack_event["destroyed_target"]:
                        rewards[agent_id] += rc.get("destroyed_enemy_factor",0.0)
                        # 摧毁敌人的团队奖励
                        for teammate_id in self._get_teammate_ids(agent_id): 
                            if not self.satellites[teammate_id].is_destroyed:
                                rewards[teammate_id] += rc.get("team_destroyed_enemy_factor",0.0)
                
                # 消耗弹药的惩罚 (按成功射击次数计算)
                # ammo_consumption_penalty_factor 在配置中应为负值
                rewards[agent_id] += num_successful_shots_this_agent * rc.get("ammo_consumption_penalty_factor",0.0)

        # --- 对负面团队事件的惩罚 (例如队友被摧毁) ---
        # 注意: "just_destroyed_this_step" 标志在 _check_episode_end 中更新。
        # 因此，如果在这里使用该标志，它反映的是 *当前* 步骤逻辑中 _check_episode_end 运行 *之前* 的状态，
        # 实际上可能指的是 *上一步* 或本步骤信息更新极早期的摧毁事件。
        # 要对 *当前步骤战斗阶段* 中被摧毁的队友立即施加惩罚，
        # 可能需要比较当前 is_destroyed 状态与步骤开始时的状态。
        # 目前，我们使用可用的信息，并承认此特定惩罚可能存在一步延迟。
        for agent_id in self.agents:
            sat = self.satellites[agent_id]
            if sat.is_destroyed: continue

            # 损失生命值的惩罚 (自身造成的或未被上述覆盖的敌人造成的)
            # 这需要将当前生命值与步骤开始时的生命值进行比较，实现起来较复杂。
            # 暂时假设 health_lost_penalty_factor 未使用或以其他方式处理。

            # 如果有队友被摧毁，则施加惩罚
            # (基于 _check_episode_end 中更新的 infos，因此可能略有延迟)
            teammates_ids = self._get_teammate_ids(agent_id) 
            for mate_id in teammates_ids:
                mate_info = self.infos.get(mate_id, {}) # 获取队友在步骤开始时的信息
                # self.satellites[mate_id].is_destroyed 是战斗结束后的 *当前* 状态。
                # "just_destroyed_this_step" 标志由 _check_episode_end 设置。
                # 如果使用该标志，意味着对在步骤 t 中被摧毁的队友的惩罚在步骤 t 就给出。
                if self.satellites[mate_id].is_destroyed:
                    # 检查 _check_episode_end 设置的标志，该标志表明导致 *当前* _check_episode_end 调用的摧毁事件。
                    if mate_info.get("just_destroyed_this_step", False): 
                         # ally_destroyed_penalty_factor 在配置中应为负值
                         rewards[agent_id] += rc.get("ally_destroyed_penalty_factor",0.0) 
        return rewards
        
    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]:
        """检查战斗场景的结束条件"""
        terminated = {aid: False for aid in self.agents}
        truncated = {aid: False for aid in self.agents}
        terminated["__all__"] = False; truncated["__all__"] = False
        
        # 更新所有智能体的 "just_destroyed_this_step" 和 "was_destroyed_prev_step" 信息
        # 这些信息用于 _calculate_rewards (计算 ally_destroyed_penalty) 和判断胜负条件
        for agent_id in self.agents:
            if agent_id not in self.infos: self.infos[agent_id] = {} # 确保字典存在
            
            # "was_destroyed_at_step_start" 应反映当前步骤的战斗/动作发生 *之前* 的状态。
            # 它通常在 *上一步* 的 _check_episode_end 结束时或当前步骤信息初始化时设置。
            # 我们假设 self.infos[agent_id]["is_destroyed"] 是在 *本* 步骤开始时通过 _get_agent_info 设置的。
            was_destroyed_at_step_start = self.infos[agent_id].get("is_destroyed", 
                                                                  self.satellites[agent_id].is_destroyed) # 如果信息中没有，则用当前状态作为后备

            is_just_destroyed_this_step = self.satellites[agent_id].is_destroyed and \
                                          not was_destroyed_at_step_start
            
            self.infos[agent_id]["just_destroyed_this_step"] = is_just_destroyed_this_step
            # 为下一步计算 "just_destroyed_this_step" 存储当前摧毁状态
            self.infos[agent_id]["was_destroyed_prev_step"] = self.satellites[agent_id].is_destroyed 

        # 计算双方存活数量
        num_A_alive = sum(1 for aid in self.team_A_ids if not self.satellites[aid].is_destroyed)
        num_B_alive = sum(1 for aid in self.team_B_ids if not self.satellites[aid].is_destroyed)
        
        # 更新通用信息中的最新存活数量
        if "__common__" not in self.infos: self.infos["__common__"] = {}
        self.infos["__common__"]["num_alive_team_A"] = num_A_alive
        self.infos["__common__"]["num_alive_team_B"] = num_B_alive

        winner = None # 初始化胜利者
        termination_reason = None # 初始化结束原因

        # 判断基于歼灭的胜利条件
        if num_A_alive == 0 and num_B_alive > 0: 
            winner = "team_B"; termination_reason = "team_A_eliminated" # A队全灭，B队胜
        elif num_B_alive == 0 and num_A_alive > 0: 
            winner = "team_A"; termination_reason = "team_B_eliminated" # B队全灭，A队胜
        elif num_A_alive == 0 and num_B_alive == 0: 
            winner = "draw_mutual_destruction"; termination_reason = "mutual_elimination" # 双方全灭，平局
        
        if winner: # 如果已决出胜负
            terminated["__all__"] = True # 标记全局终止
            self._add_win_lose_rewards_combat(winner) # 添加胜利/失败奖励到 self.current_rewards_cache
            self.infos["__common__"]["winner"] = winner
            self.infos["__common__"]["termination_reason"] = termination_reason
        
        # 如果未因歼灭而终止，则检查是否超时
        if self._current_step >= self.max_episode_steps and not terminated["__all__"]:
            truncated["__all__"] = True # 标记全局截断
            termination_reason = "timeout"
            # 根据超时时的存活数量判断胜负
            if num_A_alive > num_B_alive: timeout_winner = "team_A_timeout"
            elif num_B_alive > num_A_alive: timeout_winner = "team_B_timeout"
            else: timeout_winner = "draw_timeout" # 存活数量相同则平局
            
            self._add_win_lose_rewards_combat(timeout_winner, is_timeout=True)
            self.infos["__common__"]["winner"] = timeout_winner
            self.infos["__common__"]["termination_reason"] = termination_reason
            
        # 如果全局终止或截断，则同步所有智能体的状态
        if terminated["__all__"] or truncated["__all__"]:
            for aid in self.agents: 
                terminated[aid] = terminated["__all__"]
                truncated[aid] = truncated["__all__"]
        else: 
            # 如果游戏未全局结束，检查是否有智能体因燃料耗尽而单独终止
            for agent_id in self.agents:
                if not self.satellites[agent_id].is_destroyed and self.satellites[agent_id].current_fuel <= 0:
                    self.satellites[agent_id].is_destroyed = True # 标记为被摧毁
                    terminated[agent_id] = True # 该智能体单独终止
                    if agent_id not in self.infos: self.infos[agent_id] = {}
                    self.infos[agent_id]["termination_reason"] = "fuel_exhausted"
                    # 注意: 单个智能体燃料耗尽可能在下一步导致全局胜利条件满足。
                    # 当前逻辑是记录个体终止。更复杂的逻辑可能需要在此处重新评估全局胜负。
                    # 但标准MARL通常让环境继续，直到全局终止条件满足。
        return terminated, truncated

    def _add_win_lose_rewards_combat(self, winner_status: str, is_timeout: bool = False): 
        """根据胜利/失败状态，填充 self.current_rewards_cache 以存储终端奖励。
           这些奖励会在 step() 方法中被加到即时奖励上。
        """
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents} # 添加前清空缓存

        rc = self.reward_config 
        win_b = rc.get("win_bonus",500.0) # 获取胜利奖励值
        lose_p = rc.get("lose_penalty",-500.0) # 获取失败惩罚值 (应为负)

        # 如果是超时获胜 (且不是平局)，奖励可能减半
        if is_timeout and "draw" not in winner_status : 
            win_b *= 0.5 
            lose_p *= 0.5

        if winner_status in ["team_A", "team_A_timeout"]: # A队胜利
            for aid in self.team_A_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += win_b # 给A队存活成员加奖
            for aid in self.team_B_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += lose_p # 给B队存活成员加罚
        elif winner_status in ["team_B", "team_B_timeout"]: # B队胜利
            for aid in self.team_B_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += win_b # 给B队存活成员加奖
            for aid in self.team_A_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += lose_p # 给A队存活成员加罚
        # 对于 "draw" (平局) 状态，除非奖励配置中明确指定，否则不添加额外的胜负奖励。
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None): 
        """重置环境，并初始化场景特定状态"""
        obs_dict, info_dict = super().reset(seed, options) # 父类reset会初始化卫星、观测、infos等
        
        self.last_known_enemy_targets = {agent_id: -1 for agent_id in self.agents} # 重置上一目标记录
        self.damage_info_this_step.clear() # 清空伤害记录
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents} # 重置终端奖励缓存
        
        # 清理上一回合可能残留的、与智能体相关的特定信息
        for agent_id in self.agents:
            if agent_id not in self.infos: self.infos[agent_id] = {} # 理论上父类reset已创建
            self.infos[agent_id].pop("just_destroyed_this_step", None)
            self.infos[agent_id].pop("was_destroyed_prev_step", None)
            self.infos[agent_id].pop("termination_reason", None) # 清除智能体特定的结束原因
        
        # 清理通用信息中与上一回合相关的结束信息
        if "__common__" in self.infos:
            self.infos["__common__"].pop("winner", None)
            self.infos["__common__"].pop("termination_reason", None)

        return obs_dict, self.infos

    def step(self, actions: TypingDict[str, np.ndarray]): 
        """执行一步环境逻辑，并合并终端奖励"""
        # self.current_rewards_cache 会在 _add_win_lose_rewards_combat (如果回合结束) 中被填充。
        obs, immediate_rewards, terminated, truncated, info = super().step(actions) # 调用父类step
        
        # 如果本步骤导致回合结束，则将缓存的终端奖励加到即时奖励上
        if terminated.get("__all__", False) or truncated.get("__all__", False): # 使用 .get 避免KeyError
            for agent_id in self.agents:
                immediate_rewards[agent_id] += self.current_rewards_cache.get(agent_id, 0.0)
        
        return obs, immediate_rewards, terminated, truncated, info

# -------------------------------------------------------------------------------------------
# 4. 顶层环境类 (XuanCe 框架入口)
#    V3.5 此部分无显著逻辑更改，依赖于子环境的修复
# -------------------------------------------------------------------------------------------
AVAILABLE_SATELLITE_SCENARIOS = { # 可用场景的字典，键为场景ID，值为对应的环境类
    "one_on_one_pursuit": OneOnOnePursuitEnv,
    "many_vs_many_combat": ManyVsManyCombatEnv,
}
class SatelliteMultiAgentEnv(RawMultiAgentEnv): # 继承自Xuance的原始多智能体环境基类
    def __init__(self, config: Any): # config 是 XuanCe 传入的 Namespace 配置对象
        super(SatelliteMultiAgentEnv, self).__init__() # 初始化父类 RawMultiAgentEnv
        self.env_name = config.env_name # 环境名称，例如 "Satellite"
        self.scenario_id = config.env_id # 具体场景ID，例如 "one_on_one_pursuit"
        
        if self.scenario_id not in AVAILABLE_SATELLITE_SCENARIOS:
            raise ValueError(f"未知的卫星场景 ID: {self.scenario_id}。可用场景: {list(AVAILABLE_SATELLITE_SCENARIOS.keys())}")
        
        ScenarioClass = AVAILABLE_SATELLITE_SCENARIOS[self.scenario_id] # 获取对应的场景类
        
        # 从主配置对象中获取特定于此场景的配置字典
        # 假设 config.scenario_configs 是一个字典，其键是 scenario_id
        scenario_specific_config_dict = {}
        _scenario_configs_main = getattr(config, 'scenario_configs', {}) # 主配置中的场景配置集合
        if isinstance(_scenario_configs_main, dict):
            scenario_specific_config_dict = _scenario_configs_main.get(self.scenario_id, {})
        else:
            print(f"警告: 主配置中的 'scenario_configs' 不是一个字典。场景 '{self.scenario_id}' 将使用空配置。")

        # 实例化具体的场景环境
        # 将主配置 (作为 env_config) 和特定场景的配置字典传入
        self.scenario_env = ScenarioClass(env_config=config, scenario_config=scenario_specific_config_dict)
        
        # 将场景环境的属性暴露给 XuanCe 框架
        self.agents = self.scenario_env.agents # 智能体列表
        self.agent_ids = self.scenario_env.agent_ids # 通常与 agents 相同
        self.n_agents = self.scenario_env.n_agents # 智能体数量
        
        self.observation_space = self.scenario_env.observation_space # 观测空间
        self.action_space = self.scenario_env.action_space       # 动作空间
        self.state_space = self.scenario_env.state_space         # 全局状态空间 (如果场景定义了)
        
        self.max_episode_steps = self.scenario_env.max_episode_steps # 每回合最大步数
        
        # MARL框架 (如Xuance) 常用的属性
        self.num_agents = self.n_agents 
        self._episode_step = 0 # 此包装器的局部步数计数器，与 scenario_env 同步
        self.max_cycles = self.max_episode_steps # 用于兼容某些日志记录器/运行器的逻辑

    def get_env_info(self) -> TypingDict[str, Any]:
        """获取底层场景的环境信息，并添加包装器特定的信息"""
        info = self.scenario_env.get_env_info()
        info['env_name'] = self.env_name # 来自顶层配置
        info['scenario_id'] = self.scenario_id # 来自顶层配置
        return info

    def reset(self, seed: Optional[int]=None, options:Optional[TypingDict]=None
             ) -> Tuple[TypingDict[str,np.ndarray], TypingDict[str,Any]]:
        """重置环境"""
        obs,info = self.scenario_env.reset(seed=seed,options=options)
        self._episode_step = self.scenario_env._current_step # 同步步数计数器
        
        # 确保 __common__ 字典存在并添加 episode_step
        if "__common__" not in info: info["__common__"] = {}
        info["__common__"]["episode_step"] = self._episode_step
        return obs,info

    def step(self, actions:TypingDict[str,np.ndarray]) -> \
             Tuple[TypingDict[str,np.ndarray],TypingDict[str,float],
                   TypingDict[str,bool],TypingDict[str,bool],TypingDict[str,Any]]:
        """执行一步环境逻辑"""
        obs,rewards,terminated,truncated,info = self.scenario_env.step(actions)
        self._episode_step = self.scenario_env._current_step # 同步步数计数器

        # 确保 __common__ 字典存在并添加 episode_step
        if "__common__" not in info: info["__common__"] = {}
        info["__common__"]["episode_step"] = self._episode_step
        return obs,rewards,terminated,truncated,info

    def render(self,mode:str='human') -> Optional[np.ndarray]: 
        """渲染环境"""
        return self.scenario_env.render(mode=mode)

    def close(self): 
        """关闭环境"""
        self.scenario_env.close()

    def avail_actions(self) -> TypingDict[str,Optional[np.ndarray]]: 
        """获取可用动作"""
        return self.scenario_env.avail_actions()

    def agent_mask(self) -> TypingDict[str,bool]: 
        """获取存活智能体掩码"""
        return self.scenario_env.agent_mask()

    def state(self) -> Optional[np.ndarray]: 
        """获取全局状态"""
        return self.scenario_env.state()

# --- 主测试代码块 (V3.5 - 修正1v1测试的配置键名) ---
if __name__ == '__main__':
    from argparse import Namespace # 用于创建类似字典的配置对象
    # from copy import deepcopy # 当前未使用

    print("--- 测试 One-on-One Pursuit 环境 (含质量和推力) ---")
    # 定义用于场景配置和队伍组成键的队伍名称
    pursuer_team_name_test = "team_A"
    evader_team_name_test = "team_B"

    pursuit_scenario_config_main = { 
        "scenario_id": "one_on_one_pursuit", # 场景ID
        "max_episode_steps": 500, # 最大回合步数
        "initial_pursuer_pos": [-50000.0, 0.0, 0.0], "initial_evader_pos": [50000.0, 0.0, 0.0], # 初始位置
        "initial_pursuer_vel": [10.0, 0.0, 0.0], "initial_evader_vel": [-10.0, 0.0, 0.0],       # 初始速度
        "d_capture_1v1": 5000.0, # 捕获距离阈值
        "time_penalty_1v1": -0.05, # 时间惩罚 (确保为负)
        "explicit_force_penalty_factor_1v1": 0.00001, # 推力惩罚因子 (在奖励计算中会减去，所以正值代表惩罚幅度)
        
        "pursuer_team_name": pursuer_team_name_test, # 在场景中定义队伍名称
        "evader_team_name": evader_team_name_test,
        # 队伍组成配置的键名必须与队伍名称匹配, 例如 "team_A_composition"
        f"{pursuer_team_name_test}_composition": ["pursuer_heavy"], 
        f"{evader_team_name_test}_composition": ["evader_light"],
    }
    general_env_config_dict_1v1_main = { 
        "env_name": "SatelliteUnifiedTest", "env_id": "one_on_one_pursuit", # 环境名和场景ID
        "step_time_interval": 20.0, # 时间步长
        "action_scale": 20.0, # 动作缩放因子 (如果归一化动作为1.0，则最大推力为此值)
        "use_cw_dynamics": True, # 使用CW动力学
        "cw_ref_inertial_pos": [0.,0.,0.], "cw_ref_inertial_vel": [0.,0.,0.], # CW参考轨道参数
        "satellite_types": { # 定义卫星类型
            "default": {"mass": 100.0, "max_total_thrust": 10.0, "max_fuel": 500, "fuel_consumption_per_newton_second": 0.01, "type_name":"default_sat"},
            "pursuer_heavy": {"mass": 150.0, "max_total_thrust": 15.0, "max_fuel": 1200, "fuel_consumption_per_newton_second": 0.015, "type_name": "pursuer_heavy"},
            "evader_light": {"mass": 80.0, "max_total_thrust": 8.0, "max_fuel": 700, "fuel_consumption_per_newton_second": 0.008, "type_name": "evader_light"}
        },
        "scenario_configs": { # 此外部字典的键是 scenario_id
            "one_on_one_pursuit": pursuit_scenario_config_main # 对应的场景配置
        }
    }
    config_namespace_1v1_main = Namespace(**general_env_config_dict_1v1_main) # 转换为Namespace对象

    try:
        env_1v1 = SatelliteMultiAgentEnv(config_namespace_1v1_main) # 创建环境实例
        print(f"1v1 环境 '{env_1v1.scenario_env.env_id}' 创建成功。")
        print(f"  智能体: {env_1v1.agents}")
        # 打印智能体类型以确认配置是否正确加载
        pursuer_actual_id = env_1v1.scenario_env.pursuer_id
        evader_actual_id = env_1v1.scenario_env.evader_id
        print(f"  追逐者ID: {pursuer_actual_id}, 类型: {env_1v1.scenario_env.satellites[pursuer_actual_id].type}")
        print(f"  逃跑者ID: {evader_actual_id}, 类型: {env_1v1.scenario_env.satellites[evader_actual_id].type}")

        obs, info = env_1v1.reset() # 重置环境
        
        for i in range(10): # 运行少量步骤进行快速测试
            actions = {aid: env_1v1.action_space[aid].sample() for aid in env_1v1.agents} # 随机采样动作
            o,r,t,tc,step_info = env_1v1.step(actions) # 执行一步
            
            # 打印一些关键信息
            # 使用 .get() 避免因字典键不存在而引发的KeyError
            p_fuel = step_info.get(pursuer_actual_id,{}).get('fuel',0)
            e_fuel = step_info.get(evader_actual_id,{}).get('fuel',0)
            p_pos = step_info.get(pursuer_actual_id,{}).get('pos',np.zeros(3))
            e_pos = step_info.get(evader_actual_id,{}).get('pos',np.zeros(3))
            dist = np.linalg.norm(p_pos - e_pos)

            if i % 1 == 0 or t.get("__all__", False) or tc.get("__all__", False): 
                print(f"步骤 {i+1}: P_奖励={r.get(pursuer_actual_id,0):.2f}, E_奖励={r.get(evader_actual_id,0):.2f}, "
                      f"P_燃料={p_fuel:.1f}, E_燃料={e_fuel:.1f}, "
                      f"距离={dist:.0f}")
            if t.get("__all__", False) or tc.get("__all__", False): # 如果回合结束
                common_step_info = step_info.get('__common__',{})
                term_reason = common_step_info.get('termination_reason', common_step_info.get('winner', 'N/A'))
                print(f"回合结束原因: {term_reason}")
                break
        env_1v1.close() # 关闭环境
    except Exception as e: 
        print(f"1v1测试出错: {e}")
        import traceback; traceback.print_exc() # 打印详细错误信息

    print("\n--- 测试 Many-vs-Many Combat 环境 (例: 2v2) ---")
    num_a_test = 2 # 使用较小队伍规模进行快速测试
    num_b_test = 2
    combat_scenario_config_test = {
        "scenario_id": "many_vs_many_combat", "max_episode_steps": 200, # 较短的回合
        "num_team_A": num_a_test, 
        "num_team_B": num_b_test,
        "team_A_composition": ["fighter_std"] * num_a_test, # A队组成
        "team_B_composition": ["fighter_std"] * num_b_test, # B队组成 (例如，战斗机 vs 战斗机)
        "initial_spacing": 10000.0, "initial_team_offset_x": 100000.0, # 初始布局参数
        # 观测的最近敌人/队友数量 (不超过实际数量)
        "observe_n_closest_enemies": min(num_b_test, 2), 
        "observe_n_closest_teammates": min(num_a_test -1 if num_a_test > 0 else 0, 1), 
        "reward_config_combat": { # 战斗奖励配置
            "damage_dealt_factor": 12.0, "team_damage_dealt_factor": 2.5,
            "destroyed_enemy_factor": 150.0, "team_destroyed_enemy_factor": 30.0,
            "health_lost_penalty_factor": -0.0, # 通常为负，如果不需要则设为0
            "ally_destroyed_penalty_factor": -60.0,
            "explicit_force_penalty_factor": -0.00001, # 确保惩罚因子为负
            "ammo_consumption_penalty_factor": -0.15,
            "time_penalty_factor": -0.02,
            "win_bonus": 700.0, "lose_penalty": -700.0,
        }
    }
    general_env_config_dict_combat_test = {
        "env_name": "SatelliteUnifiedTest", "env_id": "many_vs_many_combat",
        "step_time_interval": 15.0, 
        "action_scale": 15.0, 
        "use_cw_dynamics": True, 
        "satellite_types": { 
            "default": {"mass": 100.0, "max_total_thrust": 10.0, "max_fuel": 500, "fuel_consumption_per_newton_second": 0.01, "type_name":"default_sat"},
            "fighter_std": {"mass": 120.0, "max_total_thrust": 20.0, "max_fuel": 800, "max_health": 120, "can_attack": True, "weapon_range": 70000, "weapon_damage": 20, "max_ammo": 15, "fire_cooldown_steps": 2, "sensor_range": 180000, "fuel_consumption_per_newton_second": 0.012, "type_name": "fighter_std"},
            "scout_std": {"mass": 70.0, "max_total_thrust": 10.0, "max_fuel": 1200, "max_health": 80, "can_attack": False, "sensor_range": 250000, "fuel_consumption_per_newton_second": 0.007, "type_name": "scout_std"}
        },
        "scenario_configs": { 
             "many_vs_many_combat": combat_scenario_config_test 
        }
    }
    config_namespace_combat_test = Namespace(**general_env_config_dict_combat_test)

    try:
        env_combat_test = SatelliteMultiAgentEnv(config_namespace_combat_test)
        print(f"对抗 {num_a_test}v{num_b_test} 环境 '{env_combat_test.scenario_env.env_id}' 创建成功。智能体总数: {len(env_combat_test.agents)}")
        # 可以取消注释以下行来打印更详细的队伍信息
        # print(f"  A队智能体 ({len(env_combat_test.scenario_env.team_A_ids)}): {env_combat_test.scenario_env.team_A_ids}")
        # print(f"  B队智能体 ({len(env_combat_test.scenario_env.team_B_ids)}): {env_combat_test.scenario_env.team_B_ids}")
        # print(f"  A队类型: {[env_combat_test.scenario_env.satellites[aid].type for aid in env_combat_test.scenario_env.team_A_ids]}")
        # print(f"  B队类型: {[env_combat_test.scenario_env.satellites[aid].type for aid in env_combat_test.scenario_env.team_B_ids]}")

        obs_c, info_c = env_combat_test.reset()
        for i in range(10): # 运行少量步骤
            actions_c = {aid: env_combat_test.action_space[aid].sample() for aid in env_combat_test.agents}
            oc,rc,tc,tcc,step_info_c = env_combat_test.step(actions_c)
            
            # 安全地获取用于打印奖励的智能体ID (如果队伍存在)
            teamA0_id = env_combat_test.scenario_env.team_A_ids[0] if env_combat_test.scenario_env.team_A_ids else None
            teamB0_id = env_combat_test.scenario_env.team_B_ids[0] if env_combat_test.scenario_env.team_B_ids else None
            rew_A0_val = rc.get(teamA0_id, 0) if teamA0_id else 0 
            rew_B0_val = rc.get(teamB0_id, 0) if teamB0_id else 0
            
            common_info_step = step_info_c.get('__common__',{}) # 获取通用信息
            if i % 1 == 0 or tc.get("__all__", False) or tcc.get("__all__", False): 
                print(f"步骤 {i+1} 对抗: A0_奖励={rew_A0_val:.2f}, B0_奖励={rew_B0_val:.2f}, "
                      f"A_存活={common_info_step.get('num_alive_team_A','N/A')}, B_存活={common_info_step.get('num_alive_team_B','N/A')}")
            if tc.get("__all__", False) or tcc.get("__all__", False): 
                winner_info = common_info_step.get('winner', 'N/A')
                reason_info = common_info_step.get('termination_reason', 'N/A')
                print(f"回合结束原因: {winner_info}, 具体: {reason_info}")
                break
        env_combat_test.close()
    except Exception as e: 
        print(f"对抗 {num_a_test}v{num_b_test} 测试出错: {e}")
        import traceback; traceback.print_exc()
