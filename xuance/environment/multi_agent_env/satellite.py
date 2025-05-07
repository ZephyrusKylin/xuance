import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict as GymDict
from xuance.environment import RawMultiAgentEnv
from typing import List, Dict as TypingDict, Optional, Tuple, Any, Union
import abc # For abstract base class

# 假设 satellite_function.py 包含轨道动力学等辅助函数
from xuance.common import satellite_function as sf

# -------------------------------------------------------------------------------------------
# 1. Satellite 类 (与V3.1/V3.2/V3.3版本一致)
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
                 # env_action_scale: float # 不再需要单独传入 env_action_scale 到 Satellite 内部
                ):
        self.id = sat_id
        self.team_id = team_id
        self.type = type_config.get("type_name", "default")

        self.pos = np.array(initial_pos, dtype=np.float32)
        self.vel = np.array(initial_vel, dtype=np.float32)
        
        self.mass = float(type_config.get("mass", 100.0)) 
        if self.mass <= 0: self.mass = 1.0 

        self.max_total_thrust = float(type_config.get("max_total_thrust", 10.0)) 

        self.max_fuel = float(type_config.get("max_fuel", 1000.0))
        self.current_fuel = self.max_fuel
        self.fuel_consumption_per_newton_second = float(type_config.get("fuel_consumption_per_newton_second", 0.01))

        self.max_health = float(type_config.get("max_health", 100.0))
        self.current_health = self.max_health
        self.is_destroyed = False

        self.can_attack = bool(type_config.get("can_attack", False))
        if self.can_attack:
            self.weapon_range = float(type_config.get("weapon_range", 50000.0))
            self.weapon_damage = float(type_config.get("weapon_damage", 10.0))
            self.max_ammo = int(type_config.get("max_ammo", 20))
            self.current_ammo = self.max_ammo
            self.fire_cooldown_max = int(type_config.get("fire_cooldown_steps", 5))
            self.current_fire_cooldown = 0
        else:
            self.weapon_range, self.weapon_damage, self.max_ammo, self.current_ammo, self.fire_cooldown_max, self.current_fire_cooldown = 0.0, 0.0, 0, 0, 0, 0

        self.can_observe = bool(type_config.get("can_observe", True))
        if self.can_observe:
            self.sensor_range = float(type_config.get("sensor_range", 150000.0))
        else:
            self.sensor_range = 0.0
            
        self.preferred_formation_offset = np.array(type_config.get("formation_offset", [0,0,0]), dtype=np.float32)

    def consume_fuel(self, actual_thrust_force_magnitude: float, time_delta: float):
        if not self.is_destroyed and self.current_fuel > 0:
            fuel_consumed = actual_thrust_force_magnitude * self.fuel_consumption_per_newton_second * time_delta
            self.current_fuel = max(0, self.current_fuel - fuel_consumed)
    
    def update_kinematics_with_thrust(self, desired_force_vector: np.ndarray, time_delta: float):
        if self.is_destroyed:
            return np.zeros(3, dtype=np.float32) 
        actual_force_vector = np.zeros(3, dtype=np.float32)
        if self.current_fuel > 0: 
            desired_force_magnitude = np.linalg.norm(desired_force_vector)
            if desired_force_magnitude > 1e-6: 
                if desired_force_magnitude > self.max_total_thrust:
                    actual_force_vector = (desired_force_vector / desired_force_magnitude) * self.max_total_thrust
                else:
                    actual_force_vector = desired_force_vector
        acceleration = actual_force_vector / self.mass
        self.vel += acceleration * time_delta
        actual_thrust_force_magnitude = np.linalg.norm(actual_force_vector)
        self.consume_fuel(actual_thrust_force_magnitude, time_delta)
        return actual_force_vector 

    def update_cooldowns(self):
        if self.can_attack and self.current_fire_cooldown > 0:
            self.current_fire_cooldown -= 1

    def take_damage(self, damage: float):
        if self.is_destroyed or not self.can_attack: return
        self.current_health -= damage
        if self.current_health <= 0:
            self.current_health = 0; self.is_destroyed = True

    def can_fire_weapon(self) -> bool:
        return self.can_attack and not self.is_destroyed and self.current_ammo > 0 and self.current_fire_cooldown == 0

    def fire_weapon_at_target(self) -> bool:
        if self.can_fire_weapon():
            self.current_ammo -= 1
            self.current_fire_cooldown = self.fire_cooldown_max
            return True
        return False

    def get_self_observation_component(self, normalize: bool = True) -> np.ndarray:
        fuel_norm = self.current_fuel / self.max_fuel if self.max_fuel > 0 else 0
        health_norm = self.current_health / self.max_health if self.max_health > 0 else 0
        ammo_norm = self.current_ammo / self.max_ammo if self.max_ammo > 0 else 0
        return np.array([
            self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2],
            fuel_norm if normalize else self.current_fuel,
            health_norm if normalize else self.current_health,
            ammo_norm if normalize else self.current_ammo,
            1.0 if not self.is_destroyed else 0.0
        ], dtype=np.float32)

    def get_relative_observation_to(self, other_satellite: 'Satellite', normalize: bool = True) -> np.ndarray:
        if other_satellite.is_destroyed:
            rel_pos, rel_vel, other_health_norm, other_alive = np.zeros(3), np.zeros(3), 0.0, 0.0
        else:
            rel_pos = other_satellite.pos - self.pos
            rel_vel = other_satellite.vel - self.vel
            other_health_norm = other_satellite.current_health / other_satellite.max_health if other_satellite.max_health > 0 else 0
            other_alive = 1.0
        dist = np.linalg.norm(rel_pos)
        in_my_sensor_range = 1.0 if self.can_observe and dist <= self.sensor_range and not other_satellite.is_destroyed else 0.0
        in_my_weapon_range = 1.0 if self.can_attack and dist <= self.weapon_range and not other_satellite.is_destroyed else 0.0
        return np.array([
            rel_pos[0], rel_pos[1], rel_pos[2], rel_vel[0], rel_vel[1], rel_vel[2],
            other_health_norm if normalize else other_satellite.current_health, other_alive,
            in_my_sensor_range, in_my_weapon_range
        ], dtype=np.float32)

# -------------------------------------------------------------------------------------------
# 2. MultiSatelliteEnvBase (V3.3 - 修复配置获取, 添加辅助方法)
# -------------------------------------------------------------------------------------------
class MultiSatelliteEnvBase(abc.ABC):
    def __init__(self, env_config: Any, scenario_config: TypingDict[str, Any] = None): # env_config is Namespace
        self.config = env_config 
        self.scenario_config = scenario_config if scenario_config is not None else {} 
        self.viewer = None
        self.env_id = self.scenario_config.get("scenario_id", "MultiSatelliteBaseScenario-v0")
        self.step_time_interval = getattr(self.config, "step_time_interval", 50.0)
        self.max_episode_steps = int(self.scenario_config.get("max_episode_steps", 1000))
        self._current_step = 0
        self.use_cw_dynamics = getattr(self.config, "use_cw_dynamics", True)
        if self.use_cw_dynamics:
            self.R_cw_ref_inertial = np.array(getattr(self.config, "cw_ref_inertial_pos", [0.0,0.0,0.0]), dtype=np.float32)
            self.V_cw_ref_inertial = np.array(getattr(self.config, "cw_ref_inertial_vel", [0.0,0.0,0.0]), dtype=np.float32)
        
        self.satellite_type_configs = getattr(self.config, "satellite_types", {"default": {}})
        self.action_scale = getattr(self.config, "action_scale", 1.6)
        
        self.agents: List[str] = []
        self.agent_ids: List[str] = []
        self.n_agents: int = 0
        self.satellites: TypingDict[str, Satellite] = {}
        self.team_assignments: TypingDict[str, List[str]] = {} 
        self.observation_space: TypingDict[str, gym.Space] = {}
        self.action_space: TypingDict[str, gym.Space] = {}
        self.state_space: Optional[gym.Space] = None
        self.infos: TypingDict[str, Any] = {}

        self._initialize_teams_and_agents()
        self._initialize_satellites() 
        self._define_spaces()

    @abc.abstractmethod
    def _initialize_teams_and_agents(self): pass
    
    def _initialize_satellites(self):
        self.satellites.clear()
        for team_id, agent_id_list in self.team_assignments.items():
            composition_key = f"{team_id}_composition"
            default_composition = ["default"] * len(agent_id_list)
            team_composition_specific = self.scenario_config.get(composition_key)
            team_composition_general = getattr(self.config, composition_key, default_composition) 
            team_composition = team_composition_specific if team_composition_specific is not None else team_composition_general
            if len(team_composition) != len(agent_id_list):
                print(f"警告: {composition_key} 长度 ({len(team_composition)}) 与队伍 {team_id} 中智能体数量 ({len(agent_id_list)}) 不匹配。将使用默认类型。")
                team_composition = default_composition
            for i, agent_id in enumerate(agent_id_list):
                sat_type_name = team_composition[i]
                type_config = self.satellite_type_configs.get(sat_type_name, self.satellite_type_configs.get("default", {}))
                if not type_config: 
                    type_config = {"type_name": "minimal_default", "mass": 1.0, "max_total_thrust": 0.1}
                else:
                    type_config["type_name"] = sat_type_name 
                initial_pos, initial_vel = self._get_initial_pos_vel_for_satellite(agent_id, team_id, i, type_config)
                self.satellites[agent_id] = Satellite(agent_id, team_id, type_config, initial_pos, initial_vel)
    
    @abc.abstractmethod
    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, agent_idx_in_team: int, type_config: TypingDict) -> Tuple[np.ndarray, np.ndarray]: pass
    @abc.abstractmethod
    def _define_spaces(self): pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None) -> Tuple[TypingDict[str, np.ndarray], TypingDict[str, Any]]:
        if seed is not None: np.random.seed(seed)
        self._current_step = 0
        self._initialize_satellites()
        observations = self._get_all_observations()
        self.infos = {agent_id: self._get_agent_info(agent_id) for agent_id in self.agents}
        self.infos["__common__"] = self._get_common_infos()
        return observations, self.infos 

    def _apply_thrust_and_kinematics(self, actions: TypingDict[str, np.ndarray]):
        actual_applied_forces = {} 
        for agent_id, agent_action_vector in actions.items():
            sat = self.satellites.get(agent_id)
            if not sat or sat.is_destroyed: 
                actual_applied_forces[agent_id] = np.zeros(3, dtype=np.float32)
                continue
            desired_force_cmd_scaled = agent_action_vector[:3] * self.action_scale 
            actual_force_vector = sat.update_kinematics_with_thrust(desired_force_cmd_scaled, self.step_time_interval)
            actual_applied_forces[agent_id] = actual_force_vector
        for agent_id in self.agents:
            sat = self.satellites.get(agent_id)
            if not sat or sat.is_destroyed: continue
            if self.use_cw_dynamics:
                if np.any(sat.pos) or np.any(sat.vel): 
                    propagator = sf.Clohessy_Wiltshire(R0_c=sat.pos, V0_c=sat.vel, 
                                                       R0_t=np.zeros(3), V0_t=np.zeros(3))
                    s_new_flat, _ = propagator.State_transition_matrix(self.step_time_interval)
                    sat.pos, sat.vel = s_new_flat[0:3], s_new_flat[3:6] 
            else: 
                sat.pos += sat.vel * self.step_time_interval
        return actual_applied_forces 

    @abc.abstractmethod
    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]): pass
    @abc.abstractmethod
    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]) -> TypingDict[str, float]: pass
    @abc.abstractmethod
    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]: pass 
    
    @abc.abstractmethod
    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray: pass
    
    def _get_all_observations(self) -> TypingDict[str, np.ndarray]:
        obs_dict = {}
        for agent_id in self.agents:
            obs_dict[agent_id] = self._get_observation_for_agent(agent_id)
        return obs_dict

    def _get_agent_info(self, agent_id: str) -> TypingDict[str, Any]:
        sat = self.satellites[agent_id]
        return {"fuel": sat.current_fuel, "health": sat.current_health,
                "ammo": sat.current_ammo if sat.can_attack else 0, "is_destroyed": sat.is_destroyed,
                "pos": sat.pos.copy(), "vel": sat.vel.copy(), "mass": sat.mass, "max_thrust": sat.max_total_thrust}

    def _get_common_infos(self) -> TypingDict[str, Any]:
        common_info = {"step": self._current_step}
        for team_id, agent_id_list in self.team_assignments.items():
            alive_count = sum(1 for aid in agent_id_list if not self.satellites[aid].is_destroyed)
            common_info[f"num_alive_{team_id}"] = alive_count
        return common_info

    def _get_teammate_ids(self, agent_id: str) -> List[str]:
        agent_sat = self.satellites.get(agent_id)
        if not agent_sat: return []
        current_team_id = agent_sat.team_id
        if current_team_id not in self.team_assignments: return []
        return [tid for tid in self.team_assignments[current_team_id] if tid != agent_id]

    def _get_opponent_ids(self, agent_id: str) -> List[str]:
        agent_sat = self.satellites.get(agent_id)
        if not agent_sat: return []
        current_team_id = agent_sat.team_id
        opponent_ids = []
        for team_id, member_ids in self.team_assignments.items():
            if team_id != current_team_id:
                opponent_ids.extend(member_ids)
        return opponent_ids

    def step(self, actions: TypingDict[str, np.ndarray]) -> \
             Tuple[TypingDict[str, np.ndarray], TypingDict[str, float], TypingDict[str, bool], TypingDict[str, bool], TypingDict[str, Any]]:
        self._current_step += 1
        self.infos = {agent_id: self._get_agent_info(agent_id) for agent_id in self.agents}
        self.infos["__common__"] = self._get_common_infos() 
        for sat in self.satellites.values(): sat.update_cooldowns()
        actual_applied_forces = self._apply_thrust_and_kinematics(actions)
        self._handle_scenario_specific_actions(actions, actual_applied_forces) 
        rewards = self._calculate_rewards(actions, actual_applied_forces) 
        terminated, truncated = self._check_episode_end() 
        observations = self._get_all_observations()
        for agent_id in self.agents: # Update agent-specific info after all state changes
            self.infos[agent_id].update(self._get_agent_info(agent_id))
        # Common info might have been updated by _check_episode_end (e.g. winner)
        # So, _get_common_infos() might be called again or merged carefully
        self.infos["__common__"].update(self._get_common_infos()) # Re-call to get latest num_alive after potential destructions in step

        return observations, rewards, terminated, truncated, self.infos

    def get_env_info(self) -> TypingDict[str, Any]:
        return {'state_space': self.state_space, 'observation_space': self.observation_space,
                'action_space': self.action_space, 'agents': self.agents, 'num_agents': self.n_agents,
                'max_episode_steps': self.max_episode_steps, 'team_assignments': self.team_assignments}
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode == 'human':
            common_info_to_render = self.infos.get("__common__", self._get_common_infos())
            print(f"--- 步骤: {common_info_to_render.get('step', self._current_step)} ---")
            for team_id_key in self.team_assignments.keys():
                print(f"队伍 {team_id_key} 存活: {common_info_to_render.get(f'num_alive_{team_id_key}', 'N/A')}")
            if "winner" in common_info_to_render: print(f"胜利者: {common_info_to_render['winner']}")
            if "termination_reason" in common_info_to_render: print(f"结束原因: {common_info_to_render['termination_reason']}")

            for agent_id in self.agents:
                sat = self.satellites[agent_id]
                agent_info_render = self.infos.get(agent_id, self._get_agent_info(agent_id)) 
                hp = agent_info_render.get('health', sat.current_health)
                fuel = agent_info_render.get('fuel', sat.current_fuel)
                ammo = agent_info_render.get('ammo', sat.current_ammo if sat.can_attack else 0)
                status = "已摧毁" if agent_info_render.get('is_destroyed', sat.is_destroyed) else f"HP:{hp:.0f} Fuel:{fuel:.0f} Ammo:{ammo}"
                print(f"  {sat.id} ({sat.team_id}, {sat.type}): Pos={sat.pos.round(0)}, Vel={sat.vel.round(1)}, {status}")
        elif mode == 'rgb_array': 
            canvas_size=200; world_scale=0.0002; canvas=np.ones((canvas_size,canvas_size,3),dtype=np.uint8)*240
            team_colors={"team_A":[0,0,200],"team_B":[200,0,0],"default_team":[100,100,100]}
            def wtc(p): return tuple(np.clip(int(canvas_size/2+p[i]*world_scale),0,canvas_size-1)for i in range(2))
            for sat_id in self.agents: 
                sat = self.satellites[sat_id]
                if not sat.is_destroyed:
                    cx,cy=wtc(sat.pos);color=team_colors.get(sat.team_id,team_colors["default_team"])
                    r=max(1,int(3*(sat.current_health/sat.max_health if sat.max_health > 0 else 0.1)))
                    for ro in range(-r,r+1):
                        for co in range(-r,r+1):
                            if ro**2+co**2<=r**2:canvas[np.clip(cy+ro,0,canvas_size-1),np.clip(cx+co,0,canvas_size-1)]=color
            return canvas
        return None
    def close(self): pass
    def avail_actions(self) -> TypingDict[str, Optional[np.ndarray]]: return {aid: None for aid in self.agents}
    def agent_mask(self) -> TypingDict[str, bool]: return {aid: not self.satellites[aid].is_destroyed for aid in self.agents}
    def state(self) -> Optional[np.ndarray]:
        if self.state_space is None: 
            obs_list = []
            sorted_agent_ids = sorted(self.agents) 
            for agent_id in sorted_agent_ids:
                obs_list.append(self._get_observation_for_agent(agent_id))
            try: 
                return np.concatenate(obs_list).astype(np.float32)
            except ValueError: 
                if self.observation_space and sorted_agent_ids and sorted_agent_ids[0] in self.observation_space:
                    sample_obs_space = self.observation_space[sorted_agent_ids[0]]
                    if sample_obs_space is not None and sample_obs_space.shape is not None:
                        shape_len = sample_obs_space.shape[0]
                        return np.zeros(shape_len * self.n_agents, dtype=np.float32)
                return np.array([], dtype=np.float32) 
        return None 


# -------------------------------------------------------------------------------------------
# 3. 具体任务场景类 (OneOnOnePursuitEnv - V3.3)
# -------------------------------------------------------------------------------------------
class OneOnOnePursuitEnv(MultiSatelliteEnvBase):
    def __init__(self, env_config: TypingDict[str, Any], scenario_config: TypingDict[str, Any]):
        self.pursuer_id = scenario_config.get("pursuer_id", "pursuer_0")
        self.evader_id = scenario_config.get("evader_id", "evader_0")
        super(OneOnOnePursuitEnv, self).__init__(env_config, scenario_config)
        
        self.capture_reward = float(self.scenario_config.get("capture_reward", 200.0))
        self.evasion_reward = float(self.scenario_config.get("evasion_reward", 200.0))
        self.distance_reward_scale = float(self.scenario_config.get("distance_reward_scale", 0.001))
        self.time_penalty = float(self.scenario_config.get("time_penalty_1v1", -0.1))
        self.d_capture = float(self.scenario_config.get("d_capture_1v1", 10000.0))
        self.last_distance = np.inf
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents}

    def _initialize_teams_and_agents(self): 
        self.team_assignments = {
            self.scenario_config.get("pursuer_team_name", "team_A"): [self.pursuer_id],
            self.scenario_config.get("evader_team_name", "team_B"): [self.evader_id]
        }
        self.agents = [self.pursuer_id, self.evader_id]
        self.agent_ids = self.agents
        self.n_agents = 2

    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, agent_idx_in_team: int, type_config: TypingDict) -> Tuple[np.ndarray, np.ndarray]: 
        if agent_id == self.pursuer_id:
            pos = np.array(self.scenario_config.get("initial_pursuer_pos", [-100000.0, 0.0, 0.0]), dtype=np.float32)
            vel = np.array(self.scenario_config.get("initial_pursuer_vel", [10.0, 0.0, 0.0]), dtype=np.float32)
        else: 
            pos = np.array(self.scenario_config.get("initial_evader_pos", [100000.0, 0.0, 0.0]), dtype=np.float32)
            vel = np.array(self.scenario_config.get("initial_evader_vel", [-10.0, 0.0, 0.0]), dtype=np.float32)
        return pos, vel

    def _define_spaces(self): 
        obs_dim = 18
        common_obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = {aid: common_obs_space for aid in self.agents}
        self.state_space = common_obs_space 
        act_dim = 3
        common_act_space = Box(low=-self.action_scale, high=self.action_scale, shape=(act_dim,), dtype=np.float32)
        self.action_space = {aid: common_act_space for aid in self.agents}

    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray: 
        pursuer = self.satellites[self.pursuer_id]
        evader = self.satellites[self.evader_id]
        if pursuer.is_destroyed or evader.is_destroyed: 
             return np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
        rel_pos = pursuer.pos - evader.pos
        rel_vel = pursuer.vel - evader.vel
        return np.concatenate([rel_pos, rel_vel, pursuer.pos, pursuer.vel, evader.pos, evader.vel]).astype(np.float32)

    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]): pass

    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]) -> TypingDict[str, float]:
        step_rewards = {aid: 0.0 for aid in self.agents}
        pursuer = self.satellites[self.pursuer_id]
        evader = self.satellites[self.evader_id]
        current_distance = np.linalg.norm(pursuer.pos - evader.pos)

        step_rewards[self.pursuer_id] += (self.last_distance - current_distance) * self.distance_reward_scale
        step_rewards[self.evader_id] += (current_distance - self.last_distance) * self.distance_reward_scale
        
        step_rewards[self.pursuer_id] += self.time_penalty
        step_rewards[self.evader_id] += self.time_penalty 
        
        pursuer_force_mag = np.linalg.norm(actual_applied_forces[self.pursuer_id])
        step_rewards[self.pursuer_id] -= pursuer_force_mag * self.scenario_config.get("explicit_force_penalty_factor_1v1", 0.0001) 
        evader_force_mag = np.linalg.norm(actual_applied_forces[self.evader_id])
        step_rewards[self.evader_id] -= evader_force_mag * self.scenario_config.get("explicit_force_penalty_factor_1v1", 0.0001)

        self.last_distance = current_distance
        return step_rewards

    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]: 
        terminated = {aid: False for aid in self.agents}
        truncated = {aid: False for aid in self.agents}
        terminated["__all__"] = False; truncated["__all__"] = False
        pursuer = self.satellites[self.pursuer_id]; evader = self.satellites[self.evader_id]
        current_distance = np.linalg.norm(pursuer.pos - evader.pos) 
        
        termination_reason = self.infos["__common__"].get("termination_reason") 

        if current_distance <= self.d_capture:
            terminated["__all__"] = True; self._add_terminal_rewards(capture=True); termination_reason = "capture"
        elif pursuer.current_fuel <= 0 and not terminated["__all__"]:
            terminated["__all__"] = True; self._add_terminal_rewards(pursuer_fuel_out=True); termination_reason = "pursuer_fuel_out"
        elif evader.current_fuel <= 0 and not terminated["__all__"]:
            terminated["__all__"] = True; self._add_terminal_rewards(evader_fuel_out=True); termination_reason = "evader_fuel_out"
        
        if self._current_step >= self.max_episode_steps and not terminated["__all__"]:
            truncated["__all__"] = True; self._add_terminal_rewards(timeout=True); termination_reason = "timeout"
        
        if terminated["__all__"] or truncated["__all__"]:
            for aid in self.agents: 
                terminated[aid] = terminated["__all__"]
                truncated[aid] = truncated["__all__"]
            # self.infos["__common__"] is initialized in step method of base class
            if termination_reason: self.infos["__common__"]["termination_reason"] = termination_reason
        return terminated, truncated

    def _add_terminal_rewards(self, capture=False, pursuer_fuel_out=False, evader_fuel_out=False, timeout=False): 
        if capture:
            self.current_rewards_cache[self.pursuer_id] += self.capture_reward
            self.current_rewards_cache[self.evader_id] -= self.capture_reward 
        elif pursuer_fuel_out: 
            self.current_rewards_cache[self.evader_id] += self.evasion_reward
            self.current_rewards_cache[self.pursuer_id] -= self.evasion_reward 
        elif evader_fuel_out: 
             self.current_rewards_cache[self.pursuer_id] += self.capture_reward * 0.75 
             self.current_rewards_cache[self.evader_id] -= self.capture_reward * 0.75
        elif timeout: 
            self.current_rewards_cache[self.evader_id] += self.evasion_reward
            self.current_rewards_cache[self.pursuer_id] -= self.evasion_reward
            
    def step(self, actions: TypingDict[str, np.ndarray]): 
        # current_rewards_cache is initialized in reset and before calling super().step()
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents}
        return super().step(actions) 
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None): 
        obs_dict, info_dict = super().reset(seed, options)
        pursuer = self.satellites[self.pursuer_id]; evader = self.satellites[self.evader_id]
        self.last_distance = np.linalg.norm(pursuer.pos - evader.pos)
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents}
        # self.infos is already populated by super().reset()
        return obs_dict, self.infos

# --- 多对多对抗环境 (V3.3 - 确保self.infos在_check_episode_end前初始化) ---
class ManyVsManyCombatEnv(MultiSatelliteEnvBase):
    def __init__(self, env_config: TypingDict[str, Any], scenario_config: TypingDict[str, Any]):
        super(ManyVsManyCombatEnv, self).__init__(env_config, scenario_config)
        self.reward_config = self.scenario_config.get("reward_config_combat", {
            "damage_dealt_factor": 10.0, "team_damage_dealt_factor": 2.0,
            "destroyed_enemy_factor": 100.0, "team_destroyed_enemy_factor": 20.0,
            "health_lost_penalty_factor": -0.5, 
            "ally_destroyed_penalty_factor": -50.0,
            "explicit_force_penalty_factor": self.scenario_config.get("explicit_force_penalty_factor_combat", -0.0001),
            "ammo_consumption_penalty_factor": -0.1, "time_penalty_factor": -0.01, 
            "win_bonus": 500.0, "lose_penalty": -500.0,
        })
        self.last_known_enemy_targets = {}
        self.damage_info_this_step: TypingDict[str, List[TypingDict[str, Any]]] = {} 
        self.current_rewards_cache = {}
        # self.infos is initialized in MultiSatelliteEnvBase.reset/step

    def _initialize_teams_and_agents(self): 
        self.num_team_A = int(self.scenario_config.get("num_team_A", 3))
        self.num_team_B = int(self.scenario_config.get("num_team_B", 3))
        # DEBUG PRINT:
        print(f"[ManyVsManyCombatEnv._initialize_teams_and_agents] num_team_A from scenario_config: {self.scenario_config.get('num_team_A', 'Not Found, Defaulting to 3')}")
        print(f"[ManyVsManyCombatEnv._initialize_teams_and_agents] num_team_B from scenario_config: {self.scenario_config.get('num_team_B', 'Not Found, Defaulting to 3')}")
        print(f"[ManyVsManyCombatEnv._initialize_teams_and_agents] Actual self.num_team_A: {self.num_team_A}, self.num_team_B: {self.num_team_B}")

        self.n_agents = self.num_team_A + self.num_team_B
        self.team_A_ids = [f"teamA_{i}" for i in range(self.num_team_A)]
        self.team_B_ids = [f"teamB_{i}" for i in range(self.num_team_B)]
        self.agents = self.team_A_ids + self.team_B_ids; self.agent_ids = self.agents
        self.team_assignments = {"team_A": self.team_A_ids, "team_B": self.team_B_ids}

    def _get_initial_pos_vel_for_satellite(self, agent_id: str, team_id: str, agent_idx_in_team: int, type_config: TypingDict) -> Tuple[np.ndarray, np.ndarray]: 
        spacing = float(self.scenario_config.get("initial_spacing", 20000.0))
        team_offset_x = float(self.scenario_config.get("initial_team_offset_x", 250000.0))
        num_in_team = self.num_team_A if team_id == "team_A" else self.num_team_B
        if team_id == "team_A": pos_x = -team_offset_x - agent_idx_in_team * spacing; vel_x = np.random.uniform(5, 15)
        else: pos_x = team_offset_x + agent_idx_in_team * spacing; vel_x = np.random.uniform(-15, -5)
        pos_y = np.random.uniform(-50000,50000)+(agent_idx_in_team-num_in_team/2)*spacing*0.5; pos_z=np.random.uniform(-10000,10000)
        vel_y=np.random.uniform(-5,5); vel_z=np.random.uniform(-2,2)
        return np.array([pos_x,pos_y,pos_z]), np.array([vel_x,vel_y,vel_z])

    def _define_spaces(self): 
        self.observe_n_closest_enemies = int(self.scenario_config.get("observe_n_closest_enemies",3))
        self.observe_n_closest_teammates = int(self.scenario_config.get("observe_n_closest_teammates",2))
        self.self_obs_dim=10; self.other_sat_obs_dim=8
        obs_dim_per_agent=self.self_obs_dim+self.observe_n_closest_enemies*self.other_sat_obs_dim+self.observe_n_closest_teammates*self.other_sat_obs_dim
        obs_s=Box(low=-np.inf,high=np.inf,shape=(obs_dim_per_agent,),dtype=np.float32)
        self.observation_space={aid:obs_s for aid in self.agents}; self.state_space=obs_s 
        
        self.max_opponents_for_action = 0
        if hasattr(self, 'num_team_A') and hasattr(self, 'num_team_B'): 
             if self.num_team_A > 0 and self.num_team_B > 0:
                self.max_opponents_for_action = max(self.num_team_A, self.num_team_B)
        
        action_target_dim = max(1, self.max_opponents_for_action) 
        action_dim_per_agent = 3 + action_target_dim
        
        thrust_low = -np.full(3, self.action_scale); thrust_high = np.full(3, self.action_scale)
        target_low = -np.ones(action_target_dim); target_high = np.ones(action_target_dim) 
        act_s=Box(low=np.concatenate([thrust_low, target_low]), high=np.concatenate([thrust_high, target_high]), dtype=np.float32)
        self.action_space={aid:act_s for aid in self.agents}

    def _get_observation_for_agent(self, agent_id: str) -> np.ndarray: 
        agent_sat = self.satellites[agent_id]
        if agent_sat.is_destroyed: return np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
        own_obs = agent_sat.get_self_observation_component()
        enemy_obs_list, teammate_obs_list = [], []
        
        opp_ids_all = self._get_opponent_ids(agent_id) 
        live_opp_with_dist = []
        for oid in opp_ids_all:
            opp_sat = self.satellites.get(oid) # Safe get
            if opp_sat and not opp_sat.is_destroyed and np.linalg.norm(agent_sat.pos - opp_sat.pos) <= agent_sat.sensor_range:
                live_opp_with_dist.append((np.linalg.norm(agent_sat.pos - opp_sat.pos), oid))
        live_opp_with_dist.sort(key=lambda x: x[0])
        for i in range(self.observe_n_closest_enemies): 
            enemy_obs_list.append(agent_sat.get_relative_observation_to(self.satellites[live_opp_with_dist[i][1]]) if i < len(live_opp_with_dist) else np.zeros(self.other_sat_obs_dim,dtype=np.float32))
        
        mate_ids_all = self._get_teammate_ids(agent_id) 
        live_mate_with_dist = []
        for mid in mate_ids_all:
            mate_sat = self.satellites.get(mid) # Safe get
            if mate_sat and not mate_sat.is_destroyed: 
                 live_mate_with_dist.append((np.linalg.norm(agent_sat.pos - mate_sat.pos), mid))
        live_mate_with_dist.sort(key=lambda x: x[0])
        for i in range(self.observe_n_closest_teammates): 
            teammate_obs_list.append(agent_sat.get_relative_observation_to(self.satellites[live_mate_with_dist[i][1]]) if i < len(live_mate_with_dist) else np.zeros(self.other_sat_obs_dim,dtype=np.float32))
            
        return np.concatenate([own_obs] + enemy_obs_list + teammate_obs_list).astype(np.float32)

    def _handle_scenario_specific_actions(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]):
        self.damage_info_this_step.clear() 
        for agent_id, agent_action_vector in actions.items():
            attacker = self.satellites[agent_id]
            if attacker.is_destroyed or not attacker.can_fire_weapon():
                self.last_known_enemy_targets[agent_id] = -1; continue
            target_logits = agent_action_vector[3:] 
            opponents_for_attacker = self._get_opponent_ids(agent_id) 
            chosen_target_original_idx, highest_logit = -1, -np.inf
            for i in range(len(target_logits)): 
                if i < len(opponents_for_attacker): 
                    actual_opponent_id = opponents_for_attacker[i]
                    if not self.satellites[actual_opponent_id].is_destroyed:
                        if target_logits[i] > highest_logit:
                            highest_logit = target_logits[i]; chosen_target_original_idx = i
            self.last_known_enemy_targets[agent_id] = chosen_target_original_idx
            if chosen_target_original_idx != -1:
                target_id = opponents_for_attacker[chosen_target_original_idx]
                target_sat = self.satellites[target_id] 
                if np.linalg.norm(attacker.pos - target_sat.pos) <= attacker.weapon_range:
                    if attacker.fire_weapon_at_target(): 
                        damage_amount = attacker.weapon_damage
                        target_sat.take_damage(damage_amount)
                        if attacker_id not in self.damage_info_this_step: self.damage_info_this_step[attacker_id] = []
                        self.damage_info_this_step[attacker_id].append({"target_id": target_id, "damage": damage_amount, "destroyed_target": target_sat.is_destroyed})

    def _calculate_rewards(self, actions: TypingDict[str, np.ndarray], actual_applied_forces: TypingDict[str, np.ndarray]) -> TypingDict[str, float]:
        rewards = {aid: 0.0 for aid in self.agents}
        rc = self.reward_config 
        for agent_id in self.agents:
            sat = self.satellites[agent_id]
            if sat.is_destroyed: continue
            force_magnitude = np.linalg.norm(actual_applied_forces[agent_id])
            rewards[agent_id] += force_magnitude * rc.get("explicit_force_penalty_factor", 0.0) 
            rewards[agent_id] += rc.get("time_penalty_factor",0.0) 
            if agent_id in self.damage_info_this_step: 
                num_successful_shots = 0
                for attack_event in self.damage_info_this_step[agent_id]:
                    num_successful_shots +=1
                    rewards[agent_id] += attack_event["damage"] * rc.get("damage_dealt_factor",0.0)
                    for teammate_id in self._get_teammate_ids(agent_id): 
                        if not self.satellites[teammate_id].is_destroyed:
                            rewards[teammate_id] += attack_event["damage"] * rc.get("team_damage_dealt_factor",0.0)
                    if attack_event["destroyed_target"]:
                        rewards[agent_id] += rc.get("destroyed_enemy_factor",0.0)
                        for teammate_id in self._get_teammate_ids(agent_id): 
                            if not self.satellites[teammate_id].is_destroyed:
                                rewards[teammate_id] += rc.get("team_destroyed_enemy_factor",0.0)
                rewards[agent_id] += num_successful_shots * rc.get("ammo_consumption_penalty_factor",0.0)
        for agent_id in self.agents:
            sat = self.satellites[agent_id]
            if sat.is_destroyed: continue
            teammates_ids = self._get_teammate_ids(agent_id) 
            for mate_id in teammates_ids:
                # Check if self.infos[mate_id] exists and then get "just_destroyed_this_step"
                mate_info = self.infos.get(mate_id, {})
                if self.satellites[mate_id].is_destroyed and mate_info.get("just_destroyed_this_step", False): 
                     rewards[agent_id] += rc.get("ally_destroyed_penalty_factor",0.0)
        return rewards
        
    def _check_episode_end(self) -> Tuple[TypingDict[str, bool], TypingDict[str, bool]]:
        terminated = {aid: False for aid in self.agents}
        truncated = {aid: False for aid in self.agents}
        terminated["__all__"] = False; truncated["__all__"] = False
        
        # Ensure self.infos is initialized for all agents before checking destruction status
        for agent_id in self.agents:
            if agent_id not in self.infos: self.infos[agent_id] = {} # Initialize if not present
            is_just_destroyed = self.satellites[agent_id].is_destroyed and \
                                not self.infos.get(agent_id,{}).get("was_destroyed_prev_step", False)
            if is_just_destroyed: self.infos[agent_id]["just_destroyed_this_step"] = True
            self.infos[agent_id]["was_destroyed_prev_step"] = self.satellites[agent_id].is_destroyed

        num_A = sum(1 for aid in self.team_A_ids if not self.satellites[aid].is_destroyed)
        num_B = sum(1 for aid in self.team_B_ids if not self.satellites[aid].is_destroyed)
        winner = None
        if num_A == 0 and num_B > 0: winner = "team_B"
        elif num_B == 0 and num_A > 0: winner = "team_A"
        elif num_A == 0 and num_B == 0: winner = "draw_mutual_destruction"
        
        if winner:
            terminated["__all__"] = True; self._add_win_lose_rewards_combat(winner)
            if self.infos.get("__common__") is None: self.infos["__common__"] = {}
            self.infos["__common__"]["winner"] = winner
        
        if self._current_step >= self.max_episode_steps and not terminated["__all__"]:
            truncated["__all__"] = True; timeout_winner = "draw_timeout"
            if num_A > num_B: timeout_winner = "team_A_timeout"
            elif num_B > num_A: timeout_winner = "team_B_timeout"
            self._add_win_lose_rewards_combat(timeout_winner, is_timeout=True)
            if self.infos.get("__common__") is None: self.infos["__common__"] = {}
            self.infos["__common__"]["winner"] = timeout_winner
            
        if terminated["__all__"] or truncated["__all__"]:
            for aid in self.agents: terminated[aid] = terminated["__all__"]; truncated[aid] = truncated["__all__"]
        else: 
            for agent_id in self.agents:
                if self.satellites[agent_id].current_fuel <= 0 and not self.satellites[agent_id].is_destroyed:
                    self.satellites[agent_id].is_destroyed = True; terminated[agent_id] = True
                    if agent_id not in self.infos: self.infos[agent_id] = {}
                    self.infos[agent_id]["termination_reason"] = "fuel_exhausted"
        return terminated, truncated

    def _add_win_lose_rewards_combat(self, winner_status: str, is_timeout: bool = False): 
        rc = self.reward_config 
        win_b = rc.get("win_bonus",500.0) * 0.5 if is_timeout and "draw" not in winner_status else rc.get("win_bonus",500.0)
        lose_p = rc.get("lose_penalty",-500.0) * 0.5 if is_timeout and "draw" not in winner_status else rc.get("lose_penalty",-500.0)
        if winner_status in ["team_A", "team_A_timeout"]:
            for aid in self.team_A_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += win_b
            for aid in self.team_B_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += lose_p
        elif winner_status in ["team_B", "team_B_timeout"]:
            for aid in self.team_B_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += win_b
            for aid in self.team_A_ids: 
                if not self.satellites[aid].is_destroyed : self.current_rewards_cache[aid] += lose_p
    
    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict] = None): 
        obs_dict, info_dict = super().reset(seed, options) # This initializes self.infos
        self.last_known_enemy_targets = {agent_id: -1 for agent_id in self.agents}
        self.damage_info_this_step = {} 
        self.current_rewards_cache = {aid: 0.0 for aid in self.agents}
        # self.infos is already populated by super().reset()
        # Clean up per-agent info from previous episode if any
        for agent_id in self.agents:
            if agent_id not in self.infos: self.infos[agent_id] = {} 
            self.infos[agent_id].pop("just_destroyed_this_step", None)
            self.infos[agent_id].pop("was_destroyed_prev_step", None)
        return obs_dict, self.infos

    def step(self, actions: TypingDict[str, np.ndarray]): 
        # Call base class step which orchestrates calls to abstract methods
        return super().step(actions) # This will use the overridden methods from ManyVsManyCombatEnv

# -------------------------------------------------------------------------------------------
# 4. 顶层环境类 (XuanCe 入口) - (V3.3)
# -------------------------------------------------------------------------------------------
AVAILABLE_SATELLITE_SCENARIOS = {
    "one_on_one_pursuit": OneOnOnePursuitEnv,
    "many_vs_many_combat": ManyVsManyCombatEnv,
}
class SatelliteMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, config: Any): # config is Namespace from XuanCe
        super(SatelliteMultiAgentEnv, self).__init__()
        self.env_name = config.env_name 
        self.scenario_id = config.env_id
        if self.scenario_id not in AVAILABLE_SATELLITE_SCENARIOS:
            raise ValueError(f"Unknown satellite scenario ID: {self.scenario_id}. Available: {list(AVAILABLE_SATELLITE_SCENARIOS.keys())}")
        ScenarioClass = AVAILABLE_SATELLITE_SCENARIOS[self.scenario_id]
        
        scenario_specific_config = {}
        _scenario_configs_dict = getattr(config, 'scenario_configs', {})
        if isinstance(_scenario_configs_dict, dict):
            scenario_specific_config = _scenario_configs_dict.get(self.scenario_id, {})
        
        self.scenario_env = ScenarioClass(env_config=config, scenario_config=scenario_specific_config)
        
        self.agents = self.scenario_env.agents
        self.agent_ids = self.scenario_env.agent_ids
        self.n_agents = self.scenario_env.n_agents
        self.observation_space = self.scenario_env.observation_space
        self.action_space = self.scenario_env.action_space
        self.state_space = self.scenario_env.state_space
        self.max_episode_steps = self.scenario_env.max_episode_steps
        self.num_agents = self.n_agents 
        self._episode_step = 0 
        self.max_cycles = self.max_episode_steps 

    def get_env_info(self) -> TypingDict[str, Any]:
        info = self.scenario_env.get_env_info(); info['env_name']=self.env_name; info['scenario_id']=self.scenario_id
        return info
    def reset(self, seed: Optional[int]=None, options:Optional[TypingDict]=None) -> Tuple[TypingDict[str,np.ndarray], TypingDict[str,Any]]:
        obs,info=self.scenario_env.reset(seed=seed,options=options); self._episode_step=self.scenario_env._current_step
        if "__common__" not in info: info["__common__"]={}
        info["__common__"]["episode_step"]=self._episode_step; return obs,info
    def step(self, actions:TypingDict[str,np.ndarray]) -> \
             Tuple[TypingDict[str,np.ndarray],TypingDict[str,float],TypingDict[str,bool],TypingDict[str,bool],TypingDict[str,Any]]:
        obs,rewards,terminated,truncated,info=self.scenario_env.step(actions); self._episode_step=self.scenario_env._current_step
        if "__common__" not in info: info["__common__"]={}
        info["__common__"]["episode_step"]=self._episode_step; return obs,rewards,terminated,truncated,info
    def render(self,mode:str='human') -> Optional[np.ndarray]: return self.scenario_env.render(mode=mode)
    def close(self): self.scenario_env.close()
    def avail_actions(self) -> TypingDict[str,Optional[np.ndarray]]: return self.scenario_env.avail_actions()
    def agent_mask(self) -> TypingDict[str,bool]: return self.scenario_env.agent_mask()
    def state(self) -> Optional[np.ndarray]: return self.scenario_env.state()

# --- Main test block (V3.4 - 明确7v7测试配置) ---
if __name__ == '__main__':
    from argparse import Namespace 
    from copy import deepcopy 

    print("--- 测试 One-on-One Pursuit 环境 (含质量和推力) ---")
    pursuit_scenario_config_main = { # Renamed to avoid conflict
        "scenario_id": "one_on_one_pursuit", 
        "max_episode_steps": 500, 
        "initial_pursuer_pos": [-50000.0, 0.0, 0.0], "initial_evader_pos": [50000.0, 0.0, 0.0],
        "initial_pursuer_vel": [10.0, 0.0, 0.0], "initial_evader_vel": [-10.0, 0.0, 0.0],
        "d_capture_1v1": 5000.0, "time_penalty_1v1": -0.05,
        "alpha_team_composition": ["pursuer_heavy"], "beta_team_composition": ["evader_light"],
        "explicit_force_penalty_factor_1v1": -0.00001 
    }
    general_env_config_dict_1v1_main = { # Renamed to avoid conflict
        "env_name": "SatelliteUnifiedTest", "env_id": "one_on_one_pursuit", 
        "step_time_interval": 20.0, "action_scale": 20.0, 
        "use_cw_dynamics": True, 
        "cw_ref_inertial_pos": [0.,0.,0.], "cw_ref_inertial_vel": [0.,0.,0.],
        "satellite_types": {
            "default": {"mass": 100.0, "max_total_thrust": 10.0, "max_fuel": 500, "fuel_consumption_per_newton_second": 0.01, "type_name":"default_sat"},
            "pursuer_heavy": {"mass": 150.0, "max_total_thrust": 15.0, "max_fuel": 1200, "fuel_consumption_per_newton_second": 0.015, "type_name": "pursuer_heavy"},
            "evader_light": {"mass": 80.0, "max_total_thrust": 8.0, "max_fuel": 700, "fuel_consumption_per_newton_second": 0.008, "type_name": "evader_light"}
        },
        "scenario_configs": { 
            "one_on_one_pursuit": pursuit_scenario_config_main 
        }
    }
    config_namespace_1v1_main = Namespace(**general_env_config_dict_1v1_main)

    try:
        env_1v1 = SatelliteMultiAgentEnv(config_namespace_1v1_main)
        print(f"1v1 环境 '{env_1v1.scenario_env.env_id}' 创建成功。智能体: {env_1v1.agents}")
        obs, info = env_1v1.reset()
        pursuer_id_actual = env_1v1.scenario_env.pursuer_id
        evader_id_actual = env_1v1.scenario_env.evader_id
        for i in range(200): 
            actions = {aid: env_1v1.action_space[aid].sample() for aid in env_1v1.agents}
            o,r,t,tc,step_info = env_1v1.step(actions)
            if i % 20 == 0 or t["__all__"] or tc["__all__"]: 
                print(f"Step {i+1}: P_rew={r[pursuer_id_actual]:.2f}, E_rew={r[evader_id_actual]:.2f}, P_fuel={step_info[pursuer_id_actual]['fuel']:.1f}, E_fuel={step_info[evader_id_actual]['fuel']:.1f}")
            if t["__all__"] or tc["__all__"]: 
                print(f"回合结束原因: {step_info['__common__'].get('termination_reason', step_info['__common__'].get('winner', 'N/A'))}")
                break
        env_1v1.close()
    except Exception as e: print(f"1v1测试出错: {e}"); import traceback; traceback.print_exc()

    print("\n--- 测试 Many-vs-Many Combat 环境 (7v7) ---")
    # 显式设置7v7
    num_a_test = 7
    num_b_test = 7
    combat_scenario_config_7v7 = {
        "scenario_id": "many_vs_many_combat", "max_episode_steps": 500, 
        "num_team_A": num_a_test, 
        "num_team_B": num_b_test,
        # 确保composition列表长度与队伍大小匹配，否则将使用default类型
        "team_A_composition": ["fighter_std"] * num_a_test, 
        "team_B_composition": ["scout_std"] * num_b_test,
        "reward_config_combat": { # 使用一个完整的奖励配置
            "damage_dealt_factor": 12.0, "team_damage_dealt_factor": 2.5,
            "destroyed_enemy_factor": 150.0, "team_destroyed_enemy_factor": 30.0,
            "health_lost_penalty_factor": -0.6, "ally_destroyed_penalty_factor": -60.0,
            "explicit_force_penalty_factor": -0.00001,
            "ammo_consumption_penalty_factor": -0.15, "time_penalty_factor": -0.02, 
            "win_bonus": 700.0, "lose_penalty": -700.0,
        }
    }
    general_env_config_dict_combat_7v7 = {
        "env_name": "SatelliteUnifiedTest", "env_id": "many_vs_many_combat",
        "step_time_interval": 15.0, "action_scale": 25.0, 
        "use_cw_dynamics": True, 
        "cw_ref_inertial_pos": [0.,0.,0.], "cw_ref_inertial_vel": [0.,0.,0.],
        "satellite_types": { 
            "default": {"mass": 100.0, "max_total_thrust": 10.0, "max_fuel": 500, "fuel_consumption_per_newton_second": 0.01, "type_name":"default_sat"},
            "fighter_std": {"mass": 120.0, "max_total_thrust": 20.0, "max_fuel": 800, "max_health": 120, "can_attack": True, "weapon_range": 70000, "weapon_damage": 20, "max_ammo": 15, "fire_cooldown_steps": 2, "sensor_range": 180000, "fuel_consumption_per_newton_second": 0.012, "type_name": "fighter_std"},
            "scout_std": {"mass": 70.0, "max_total_thrust": 10.0, "max_fuel": 1200, "max_health": 80, "can_attack": False, "sensor_range": 250000, "fuel_consumption_per_newton_second": 0.007, "type_name": "scout_std"}
        },
        "scenario_configs": {
             "many_vs_many_combat": combat_scenario_config_7v7 # 使用7v7的特定配置
        }
    }
    config_namespace_combat_7v7 = Namespace(**general_env_config_dict_combat_7v7)

    try:
        env_combat_7v7 = SatelliteMultiAgentEnv(config_namespace_combat_7v7)
        print(f"Combat 7v7 环境 '{env_combat_7v7.scenario_env.env_id}' 创建成功。智能体总数: {len(env_combat_7v7.agents)}")
        print(f"  A队智能体 ({len(env_combat_7v7.scenario_env.team_A_ids)}): {env_combat_7v7.scenario_env.team_A_ids}")
        print(f"  B队智能体 ({len(env_combat_7v7.scenario_env.team_B_ids)}): {env_combat_7v7.scenario_env.team_B_ids}")

        obs_c, info_c = env_combat_7v7.reset()
        for i in range(200): 
            actions_c = {aid: env_combat_7v7.action_space[aid].sample() for aid in env_combat_7v7.agents}
            oc,rc,tc,tcc,step_info_c = env_combat_7v7.step(actions_c)
            
            # 安全获取第一个智能体的ID用于打印奖励，如果队伍存在的话
            teamA0_id = env_combat_7v7.scenario_env.team_A_ids[0] if env_combat_7v7.scenario_env.team_A_ids else None
            teamB0_id = env_combat_7v7.scenario_env.team_B_ids[0] if env_combat_7v7.scenario_env.team_B_ids else None
            rew_A0 = rc.get(teamA0_id, 0) if teamA0_id else 0 # 使用 .get 以防ID不存在于奖励字典中
            rew_B0 = rc.get(teamB0_id, 0) if teamB0_id else 0
            
            if i % 10 == 0 or tc["__all__"] or tcc["__all__"]: 
                print(f"Step {i+1} Combat 7v7: A0_rew={rew_A0:.2f}, B0_rew={rew_B0:.2f}, A_alive={step_info_c['__common__']['num_alive_team_A']}, B_alive={step_info_c['__common__']['num_alive_team_B']}")
            if tc["__all__"] or tcc["__all__"]: 
                print(f"回合结束原因: {step_info_c['__common__'].get('winner', 'N/A')}")
                break
        env_combat_7v7.close()
    except Exception as e: print(f"Combat 7v7 测试出错: {e}"); import traceback; traceback.print_exc()

