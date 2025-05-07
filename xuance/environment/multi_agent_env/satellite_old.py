import numpy as np
import gymnasium as gym # 从 gymnasium 导入 spaces，这是一个更现代的 Gym API 分支
from gymnasium.spaces import Box, Discrete, Dict as GymDict
# from gymnasium.spaces import Box
from xuance.environment import RawMultiAgentEnv
# 假设 satellite_function.py 在同一目录或在 PYTHONPATH 中可访问
from xuance.common import satellite_function as sf
from typing import List, Dict as TypingDict, Optional, Tuple, Any
# class One_on_one_purchase(RawMultiAgentEnv):
#     """
#     一个用于卫星追逃博弈的多智能体强化学习环境。

#     智能体:
#     - pursuer_0: 追捕卫星。
#     - evader_0: 逃逸卫星。

#     观测空间 (每个智能体):
#     - 一个18维向量，包含:
#         - 相对位置 (追捕者 - 逃逸者) [x, y, z]
#         - 相对速度 (追捕者 - 逃逸者) [vx, vy, vz]
#         - 追捕者绝对位置 [x, y, z]
#         - 追捕者绝对速度 [vx, vy, vz]
#         - 逃逸者绝对位置 [x, y, z]
#         - 逃逸者绝对速度 [vx, vy, vz]

#     动作空间 (每个智能体):
#     - 一个3维连续向量，代表推力 [ax, ay, az]。
#       每个分量被限制在 -1.6 和 1.6 之间。

#     奖励:
#     - 追捕者: 因接近逃逸者而获得奖励，因距离太远或处于危险区域而受到惩罚，
#                捕获成功则获得巨大奖励。
#     - 逃逸者: 获得与追捕者相反的奖励 (例如，因增加距离而获得奖励，
#                因被捕获而受到惩罚)。
#     """
#     def __init__(self, env_config: dict = None):
#         super(One_on_one_purchase, self).__init__()

#         # 默认配置，可以被 env_config 覆盖
#         config = {
#             "initial_pursuer_pos": np.array([200000.0, 0.0, 0.0]),       # 追捕者初始位置
#             "initial_pursuer_vel": np.array([0.0, 0.0, 0.0]),          # 追捕者初始速度
#             "initial_evader_pos": np.array([180000.0, 0.0, 0.0]),      # 逃逸者初始位置 (更近以便快速交互)
#             "initial_evader_vel": np.array([0.0, 0.0, 0.0]),           # 逃逸者初始速度
#             "initial_fuel_pursuer": 320.0,                            # 追捕者初始燃料
#             "initial_fuel_evader": 320.0,                             # 逃逸者初始燃料
#             "d_capture": 10000.0,                                     # 捕获距离 (米)
#             "d_range_dangerous_zone_check": 100000.0,                 # 开始检查危险区域的距离阈值
#             "max_episode_steps": 500,                                 # 最大回合步数
#             "win_reward": 100.0,                                      # 通用胜利奖励 (例如，逃逸者达到最大步数)
#             "capture_reward_pursuer": 150.0,                          # 追捕者捕获奖励
#             "capture_penalty_evader": -150.0,                         # 逃逸者被捕惩罚
#             "step_time_interval": 100.0,                              # CW传播的时间间隔 (秒)
#             "action_scale": 1.6                                       # 动作缩放因子
#         }
#         if env_config:
#             config.update(env_config)

#         self.env_id = config.get("env_id", "One_on_one_purchase")
#         self.num_agents = 2
#         self.agents = ["pursuer_0", "evader_0"]
#         self.agent_ids = ["pursuer_0", "evader_0"] # 为了与某些 MARL 框架兼容

#         self._initial_pursuer_pos = np.array(config["initial_pursuer_pos"], dtype=np.float32)
#         self._initial_pursuer_vel = np.array(config["initial_pursuer_vel"], dtype=np.float32)
#         self._initial_evader_pos = np.array(config["initial_evader_pos"], dtype=np.float32)
#         self._initial_evader_vel = np.array(config["initial_evader_vel"], dtype=np.float32)
        
#         self.initial_fuel_pursuer = config["initial_fuel_pursuer"]
#         self.initial_fuel_evader = config["initial_fuel_evader"]
#         self.d_capture = config["d_capture"]
#         self.d_range_dangerous_zone_check = config["d_range_dangerous_zone_check"]
#         self.max_episode_steps = config["max_episode_steps"]
#         self.win_reward = config["win_reward"] 
#         self.capture_reward_pursuer = config["capture_reward_pursuer"]
#         self.capture_penalty_evader = config["capture_penalty_evader"]
#         self.step_time_interval = config["step_time_interval"]
#         self.action_scale = config["action_scale"]

#         # 定义观测和动作空间
#         # 观测: [相对位置(3), 相对速度(3), 追捕者位置(3), 追捕者速度(3), 逃逸者位置(3), 逃逸者速度(3)]
#         obs_dim = 18
#         obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
#         obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        
#         self.observation_space = {
#             agent: Box(low=obs_low, high=obs_high, dtype=np.float32) for agent in self.agents
#         }
#         # 全局状态空间 (可以与连接的观测相同，或是一个特定的全局视角)
#         self.state_space = Box(low=obs_low, high=obs_high, dtype=np.float32) # 示例: 与单个智能体的观测相同

#         act_dim = 3
#         act_low = np.full(act_dim, -self.action_scale, dtype=np.float32)
#         act_high = np.full(act_dim, self.action_scale, dtype=np.float32)
#         self.action_space = {
#             agent: Box(low=act_low, high=act_high, dtype=np.float32) for agent in self.agents
#         }

#         self._current_step = 0
#         self.pursuer_pos = np.copy(self._initial_pursuer_pos)
#         self.pursuer_vel = np.copy(self._initial_pursuer_vel)
#         self.evader_pos = np.copy(self._initial_evader_pos)
#         self.evader_vel = np.copy(self._initial_evader_vel)
#         self.fuel_pursuer = self.initial_fuel_pursuer
#         self.fuel_evader = self.initial_fuel_evader
#         self.distance = np.inf
#         self.num_dangerous_zones = 0

#     def get_env_info(self):
#         return {
#             'state_space': self.state_space,
#             'observation_space': self.observation_space,
#             'action_space': self.action_space,
#             'agents': self.agents,
#             'num_agents': self.num_agents,
#             'max_episode_steps': self.max_episode_steps
#         }

#     def _get_observation(self, agent_id):
#         # 在此环境中，两个智能体获得相同的完整状态信息。
#         # 如果需要部分可观测性，可以自定义此部分。
#         obs = np.concatenate([
#             self.pursuer_pos - self.evader_pos,
#             self.pursuer_vel - self.evader_vel,
#             self.pursuer_pos,
#             self.pursuer_vel,
#             self.evader_pos,
#             self.evader_vel
#         ]).astype(np.float32)
#         return obs

#     def _get_all_observations(self):
#         return {agent: self._get_observation(agent) for agent in self.agents}

#     def state(self):
#         """返回环境的全局状态。"""
#         # 在这里，全局状态与任何智能体的观测相同。
#         return self._get_observation(self.agents[0]) 

#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             super().reset(seed=seed) # 用于与 gym 的种子设定功能潜在的兼容

#         self.pursuer_pos = np.copy(self._initial_pursuer_pos)
#         self.pursuer_vel = np.copy(self._initial_pursuer_vel)
#         self.evader_pos = np.copy(self._initial_evader_pos)
#         self.evader_vel = np.copy(self._initial_evader_vel)

#         self.fuel_pursuer = self.initial_fuel_pursuer
#         self.fuel_evader = self.initial_fuel_evader
        
#         self._current_step = 0
#         self.distance = np.linalg.norm(self.pursuer_pos - self.evader_pos)
#         self.num_dangerous_zones = 0 # 重置危险区域数量

#         observations = self._get_all_observations()
#         infos = {agent: {} for agent in self.agents}
        
#         # 在 xuance 中, RawMultiAgentEnv 的 reset 返回 (obs, info)
#         # 为了与可能返回 (obs, reward, terminated, truncated, info) 的环境兼容
#         # 我们确保输出格式符合 xuance 对 reset 的期望。
#         return observations, infos


#     def step(self, actions: dict):
#         self._current_step += 1
        
#         pursuer_action = np.clip(actions["pursuer_0"], -self.action_scale, self.action_scale)
#         evader_action = np.clip(actions["evader_0"], -self.action_scale, self.action_scale)

#         prev_distance = self.distance

#         # 根据动作（脉冲）更新速度
#         # 假设动作是 delta-v 分量
#         self.pursuer_vel += pursuer_action
#         self.evader_vel += evader_action
        
#         # 燃料消耗
#         self.fuel_pursuer -= np.sum(np.abs(pursuer_action))
#         self.fuel_evader -= np.sum(np.abs(evader_action))

#         # 使用 Clohessy-Wiltshire 方程传播状态
#         # 原始代码使用 CW 进行相对运动计算，但将其应用于绝对状态。
#         # 为简单起见并与原始代码保持一致，我们假设 CW 应用于相对坐标系，
#         # 或者位置/速度已经处于适合 CW 的坐标系中。
#         # 如果这些是惯性状态，则完整的轨道传播器会更准确，
#         # 但我们遵循 'environment.py' 的结构。
        
#         # sf.Clohessy_Wiltshire 接收 R0_c, V0_c, R0_t, V0_t 并返回新的 (state_c, state_t)
#         # 其中每个状态是 [pos, vel].ravel()
#         # 我们需要确保输入位于正确的坐标系中（例如，相对于目标轨道）。
#         # 目前，我们直接使用绝对状态，就像原始 'environment.py' 的 step 中那样。
#         # 这可能意味着它们已经处于希尔坐标系或类似的局部坐标系中。
        
#         # 要使用 sf.Clohessy_Wiltshire，它期望 R0_c, V0_c, R0_t, V0_t。
#         # 让我们假设当前的位置和速度是此步骤的 R0, V0。
#         cw_propagator = sf.Clohessy_Wiltshire(
#             R0_c=self.pursuer_pos, V0_c=self.pursuer_vel,
#             R0_t=self.evader_pos, V0_t=self.evader_vel
#         )
#         # 传播一个时间步长（例如，原始代码中的100秒）
#         s_1_new_flat, s_2_new_flat = cw_propagator.State_transition_matrix(self.step_time_interval)

#         self.pursuer_pos = s_1_new_flat[0:3]
#         self.pursuer_vel = s_1_new_flat[3:6]
#         self.evader_pos = s_2_new_flat[0:3]
#         self.evader_vel = s_2_new_flat[3:6]

#         self.distance = np.linalg.norm(self.pursuer_pos - self.evader_pos)

#         # 计算危险区域数量（如果适用于奖励）
#         # 这可能计算量很大。如果不是关键，可以考虑简化或移除。
#         # 目前，我们保持与原始逻辑相似。
#         if self.distance < self.d_range_dangerous_zone_check:
#              self._calculate_number_dangerous_zones() # 为清晰起见重命名

#         # 奖励
#         pursuer_reward = 0.0
#         evader_reward = 0.0

#         # 1. 追捕者的基本基于距离的奖励
#         if self.distance < prev_distance:
#             pursuer_reward += 1.0  # 更近
#         else:
#             pursuer_reward -= 1.0  # 更远或相同

#         # 2. 距离过远的惩罚 (原始逻辑: d_capture <= dis <= 4 * d_capture)
#         if self.d_capture <= self.distance <= 4 * self.d_capture:
#             pursuer_reward -= 1.0 
#         elif self.distance > 4 * self.d_capture:
#             pursuer_reward -= 2.0

#         # 3. 追捕者的危险区域惩罚
#         if self.num_dangerous_zones == 0:
#             pursuer_reward -= 1.0 
#         else:
#             # 原始代码在这里对追捕者有积极影响，似乎与直觉相反
#             # 假设这意味着追捕者正在迫使逃逸者进入危险区域。
#             # 如果是关于追捕者自身危险的惩罚，或者关于逃逸者危险的奖励，则会很复杂。
#             # 原始 'environment.py' 将此添加到 pursuer_reward。
#             # 为简单起见，我们假设这是对追捕者自身危险的惩罚。
#             # 或者，如果是关于 *目标* 危险，则对追捕者是积极的。
#             # 根据原始代码，它是 `self.dangerous_zone * 0.5`。
#             # 我们坚持原始含义：更多的危险区域可能意味着更多的机会。
#             pursuer_reward += self.num_dangerous_zones * 0.5


#         # 4. 塑形奖励 (原始的 pv1-pv4) - 这些很复杂，需要仔细调整。
#         # 为简化此重构，我们将使用上述组件并添加捕获奖励。
#         # 原始 reward_of_action 函数：
#         # pv1: 追捕者速度与逃逸者速度的相似性 (直接追击)
#         # pv2: 追捕者速度与 (追捕者位置 - 逃逸者先前位置) 的相似性 (跟踪)
#         # pv3: 追捕者位置与逃逸者先前位置的相似性 (对齐)
#         # pv4: (追捕者位置 - 逃逸者位置) 与追捕者动作的负相似性 (动作效率)
        
#         # 简化：我们专注于捕获和距离。
#         # 如果需要，可以重新添加复杂的塑形奖励。

#         # 终止与截断
#         terminated = {agent: False for agent in self.agents}
#         terminated["__all__"] = False
#         truncated = {agent: False for agent in self.agents}
#         truncated["__all__"] = False

#         info = {agent: {} for agent in self.agents}
#         info["pursuer_0"]["fuel"] = self.fuel_pursuer
#         info["evader_0"]["fuel"] = self.fuel_evader
#         info["pursuer_0"]["distance"] = self.distance
#         info["evader_0"]["distance"] = self.distance


#         if self.distance <= self.d_capture:
#             pursuer_reward += self.capture_reward_pursuer
#             evader_reward += self.capture_penalty_evader # 逃逸者因被捕而受到惩罚
#             terminated["__all__"] = True
#             info["pursuer_0"]["capture"] = True
#             info["evader_0"]["captured"] = True
        
#         # 燃料检查 (简单版本：如果任何一方耗尽，回合可能结束)
#         # 这在原始奖励中不是主要终止条件，但最好有。
#         if self.fuel_pursuer <= 0 or self.fuel_evader <= 0:
#             # 这里没有特定的奖励变化，但对于耗尽燃料的一方可能是损失。
#             # 目前，让它由 max_episode_steps 或捕获来处理。
#             # 如果追捕者燃料耗尽且未捕获，则逃逸者获胜。
#             if self.fuel_pursuer <= 0 and not terminated["__all__"]:
#                  evader_reward += self.win_reward # 如果追捕者燃料耗尽，逃逸者获胜
#                  # terminated["__all__"] = True # 可以在此终止
#             if self.fuel_evader <= 0 and not terminated["__all__"]:
#                  pursuer_reward += self.win_reward * 0.5 # 如果逃逸者燃料耗尽，追捕者获得一些奖励
#                  # terminated["__all__"] = True # 可以在此终止
#             # 为简单起见，我们主要依靠 max_steps 和捕获进行终止。
#             # 添加燃料耗尽信息
#             if self.fuel_pursuer <= 0: info["pursuer_0"]["fuel_out"] = True
#             if self.fuel_evader <= 0: info["evader_0"]["fuel_out"] = True


#         if self._current_step >= self.max_episode_steps:
#             truncated["__all__"] = True
#             if not terminated["__all__"]: # 如果尚未因捕获而终止
#                 # 如果达到最大步数，且逃逸者未被捕获，则逃逸者“获胜”
#                 evader_reward += self.win_reward 
#                 info["evader_0"]["evaded_timeout"] = True


#         # 逃逸者的奖励可以是追捕者奖励的负数，或者更具体。
#         # 原始 'environment.py' Flag==1 中 escaper_reward = -pursuer_reward。
#         # 我们让它更明确：增加距离获得奖励，减少距离受到惩罚。
#         if not terminated["__all__"]: # 如果未被捕获
#             if self.distance > prev_distance:
#                 evader_reward += 1.0
#             else:
#                 evader_reward -= 1.0
#             # 因处于危险区域而受到惩罚 (如果这是关于逃逸者的安全)
#             # evader_reward -= self.num_dangerous_zones * 0.5 


#         rewards = {"pursuer_0": pursuer_reward, "evader_0": evader_reward}
#         observations = self._get_all_observations()

#         # 如果 __all__ 为 true，确保所有单个智能体的终止/截断标志也设置为 true
#         if terminated["__all__"]:
#             for agent in self.agents:
#                 terminated[agent] = True
#         if truncated["__all__"]:
#             for agent in self.agents:
#                 truncated[agent] = True
        
#         return observations, rewards, terminated, truncated, info

#     def _calculate_number_dangerous_zones(self):
#         """
#         计算危险区域的数量。
#         这是 satellite_function.py 中复杂计算的占位符。
#         原始函数需要转换为惯性系，这增加了复杂性。
#         在此重构中，我们将简化或假设它已处理。
#         如果需要 `relative_state_to_absolute_state`，它应该是此类的一部分。
#         """
#         # 如有必要，将当前相对状态转换为绝对惯性状态
#         # R0_c_abs, V0_c_abs = self._relative_to_absolute_state(self.pursuer_pos, self.pursuer_vel)
#         # R0_t_abs, V0_t_abs = self._relative_to_absolute_state(self.evader_pos, self.evader_vel)

#         # 在此示例中，我们假设当前状态可直接使用，或者计算已简化。原始计算非常具体。
#         # 如果需要完全准确性，请确保坐标系正确。
#         # 原始 `calculate_number_hanger_area` 使用 `fuel_c` 作为 `Delta_V_c`。
        
#         # 简化：如果完整计算对于每个步骤来说太慢或太复杂：
#         # self.num_dangerous_zones = 0 # 如果计算被跳过/简化，则默认为0
#         # return

#         try:
#             # 原始代码使用相对位置/速度作为 sf.Clohessy_Wiltshire 的输入，
#             # 但对于 sf.Time_window_of_danger_zone，它暗示了惯性状态。
#             # 这是一个潜在的不一致点或需要转换的地方。
#             # 目前，我们假设 self.pursuer_pos/vel 和 self.evader_pos/vel
#             # 适用于简化检查或已处于所需坐标系中。
#             # 原始 environment.py 中的 `relative_state_to_absolute_state` 方法
#             # 表明主要状态 (self.Pursuer_position 等) 是相对的 (例如在 CW 坐标系中)。
#             # 如果是这样，它们在调用 `Time_window_of_danger_zone` 之前需要转换。

#             # 我们现在假设位置处于局部坐标系 (如 CW 坐标系)
#             # 并且需要 `relative_state_to_absolute_state`。
#             R0_c_abs, V0_c_abs = self._relative_to_absolute_state(self.pursuer_pos, self.pursuer_vel)
#             R0_t_abs, V0_t_abs = self._relative_to_absolute_state(self.evader_pos, self.evader_vel)

#             danger_zone_calculator = sf.Time_window_of_danger_zone(
#                 R0_c=R0_c_abs, V0_c=V0_c_abs,
#                 R0_t=R0_t_abs, V0_t=V0_t_abs,
#                 Delta_V_c=max(0.1, self.fuel_pursuer), # 确保 Delta_V_c 为正
#                 time_step=1 # 与原始代码一致
#             )
#             self.num_dangerous_zones = danger_zone_calculator.calculate_number_of_hanger_area()
#         except Exception as e:
#             # print(f"计算危险区域时出错: {e}")
#             self.num_dangerous_zones = 0 # 如果计算失败，则默认为0

#     @staticmethod
#     def _relative_to_absolute_state(R0_relative, V0_relative):
#         """
#         将状态从相对坐标系 (例如，围绕参考点的 CW 坐标系) 转换为绝对惯性坐标系。
#         这基于原始的 environment.py。
#         参考点 [N50,E0] 需要定义或传递。
#         """
#         # CW 参考点 [N50,E0] 在惯性坐标系中的状态 (原始示例值)
#         R_cw_ref_inertial = np.array([27098000.0, 32306000.0, 0.0])
#         V_cw_ref_inertial = np.array([-2350.0, 1970.0, 0.0])
        
#         R_abs = R_cw_ref_inertial + R0_relative
#         V_abs = V_cw_ref_inertial + V0_relative
#         return R_abs, V_abs

#     def render(self, mode='human'):
#         # 基本渲染，可以扩展 (例如，使用 matplotlib 进行二维投影)
#         if mode == 'human':
#             print(f"步骤: {self._current_step}")
#             print(f"  追捕者 位置: {self.pursuer_pos}, 速度: {self.pursuer_vel.round(2)}, 燃料: {self.fuel_pursuer:.1f}")
#             print(f"  逃逸者 位置: {self.evader_pos}, 速度: {self.evader_vel.round(2)}, 燃料: {self.fuel_evader:.1f}")
#             print(f"  距离: {self.distance:.2f}, 危险区域: {self.num_dangerous_zones}")
#             print("-" * 20)
#         elif mode == 'rgb_array':
#             # 返回用于视频录制等的 RGB 数组。
#             # 这需要更复杂的 可视化。
#             # 目前，返回一个占位符。
#             # 创建一个简单的 64x64x3 数组
#             canvas = np.zeros((64, 64, 3), dtype=np.uint8)
#             # 简单表示：追捕者蓝色，逃逸者红色
#             def to_canvas_coords(pos, canvas_size=64, scale=0.0001): # 根据需要调整比例
#                 # 居中并缩放位置
#                 x = int(canvas_size / 2 + pos[0] * scale)
#                 y = int(canvas_size / 2 + pos[1] * scale)
#                 return np.clip(x, 0, canvas_size-1), np.clip(y, 0, canvas_size-1)

#             px, py = to_canvas_coords(self.pursuer_pos)
#             ex, ey = to_canvas_coords(self.evader_pos)
            
#             canvas[py, px, 2] = 255 # 追捕者为蓝色
#             canvas[ey, ex, 0] = 255 # 逃逸者为红色
#             return canvas
#         return None # 或为不支持的模式引发错误

#     def close(self):
#         # 如果需要，清理任何资源
#         pass

#     def avail_actions(self):
#         """返回每个智能体的可用动作字典。
#         对于连续动作空间，这通常是 None。"""
#         return {agent: None for agent in self.agents}

#     def agent_mask(self):
#         """返回一个布尔掩码，指示哪些智能体当前处于活动状态。"""
#         # 在此博弈中，两个智能体在回合结束前始终处于活动状态。
#         return {agent: True for agent in self.agents}


# import numpy as np
# import gymnasium as gym
# from gymnasium.spaces import Box
# from xuance.environment import RawMultiAgentEnv
# # 假设 satellite_function.py 在同一目录或在 PYTHONPATH 中可访问
# import satellite_function as sf


class Satellite:
    """代表单个卫星的类，包含其所有属性。"""
    def __init__(self, sat_id: str, team_id: str, config: TypingDict[str, Any], initial_pos: np.ndarray, initial_vel: np.ndarray):
        self.id = sat_id
        self.team_id = team_id
        self.type = config.get("type", "default")
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.vel = np.array(initial_vel, dtype=np.float32)
        
        self.max_fuel = float(config.get("max_fuel", 1000.0))
        self.current_fuel = self.max_fuel
        
        self.max_health = float(config.get("max_health", 100.0))
        self.current_health = self.max_health
        
        self.weapon_range = float(config.get("weapon_range", 50000.0)) # 武器射程
        self.weapon_damage = float(config.get("weapon_damage", 10.0))  # 武器伤害
        self.max_ammo = int(config.get("max_ammo", 20))             # 最大弹药量
        self.current_ammo = self.max_ammo
        self.fire_cooldown_max = int(config.get("fire_cooldown_steps", 5)) # 射击冷却时间（步数）
        self.current_fire_cooldown = 0

        self.sensor_range = float(config.get("sensor_range", 150000.0)) # 传感器范围
        
        self.is_destroyed = False
        self.action_scale = float(config.get("action_scale", 1.6)) # 与主环境配置一致
        self.fuel_consumption_rate = float(config.get("fuel_consumption_rate", 0.1)) # 每单位推力的燃料消耗

    def update_state(self, thrust: np.ndarray, time_delta: float):
        """根据推力更新速度和燃料。位置更新由CW方程处理。"""
        if self.is_destroyed:
            self.vel = np.zeros(3, dtype=np.float32) # 被摧毁后停止移动
            return

        applied_thrust = np.clip(thrust, -self.action_scale, self.action_scale)
        self.vel += applied_thrust # 简单模型：推力直接改变速度增量，更精确模型应考虑质量和推力大小
        
        fuel_consumed = np.sum(np.abs(applied_thrust)) * self.fuel_consumption_rate
        self.current_fuel = max(0, self.current_fuel - fuel_consumed)
        if self.current_fuel == 0:
            # 燃料耗尽的逻辑，例如无法再施加推力，可以在环境主循环中检查
            pass
        if self.current_fire_cooldown > 0:
            self.current_fire_cooldown -=1

    def take_damage(self, damage: float):
        if self.is_destroyed:
            return
        self.current_health -= damage
        if self.current_health <= 0:
            self.current_health = 0
            self.is_destroyed = True
            # print(f"卫星 {self.id} 已被摧毁!")

    def can_fire(self) -> bool:
        return not self.is_destroyed and self.current_ammo > 0 and self.current_fire_cooldown == 0

    def fire_weapon(self):
        if self.can_fire():
            self.current_ammo -= 1
            self.current_fire_cooldown = self.fire_cooldown_max
            return True
        return False

    def get_observation_component(self) -> np.ndarray:
        """返回该卫星自身状态的观测部分。"""
        return np.array([
            self.pos[0], self.pos[1], self.pos[2],
            self.vel[0], self.vel[1], self.vel[2],
            self.current_fuel / self.max_fuel, # 归一化燃料
            self.current_health / self.max_health, # 归一化健康值
            self.current_ammo / self.max_ammo if self.max_ammo > 0 else 0, # 归一化弹药
            1.0 if not self.is_destroyed else 0.0 # 存活状态
        ], dtype=np.float32)
    
    @staticmethod
    def get_inertial_state_from_cw(relative_pos, relative_vel, cw_ref_inertial_pos, cw_ref_inertial_vel):
        """将CW相对状态转换为惯性状态"""
        R_abs = cw_ref_inertial_pos + relative_pos
        V_abs = cw_ref_inertial_vel + relative_vel
        return R_abs, V_abs

    @staticmethod
    def get_cw_state_from_inertial(inertial_pos, inertial_vel, cw_ref_inertial_pos, cw_ref_inertial_vel):
        """将惯性状态转换为CW相对状态"""
        R_rel = inertial_pos - cw_ref_inertial_pos
        V_rel = inertial_vel - cw_ref_inertial_vel
        return R_rel, V_rel








class One_on_one_purchase(RawMultiAgentEnv):
    """
    一个优化后的、去除了内部Flag切换机制的多智能体强化学习环境，用于卫星追逃博弈。

    智能体:
    - pursuer_0: 追捕卫星。
    - evader_0: 逃逸卫星。

    观测空间 (每个智能体):
    - 一个18维向量，包含:
        - 相对位置 (追捕者 - 逃逸者) [x, y, z]
        - 相对速度 (追捕者 - 逃逸者) [vx, vy, vz]
        - 追捕者绝对位置 (在CW参考系下的相对位置) [x, y, z]
        - 追捕者绝对速度 (在CW参考系下的相对速度) [vx, vy, vz]
        - 逃逸者绝对位置 (在CW参考系下的相对位置) [x, y, z]
        - 逃逸者绝对速度 (在CW参考系下的相对速度) [vx, vy, vz]

    动作空间 (每个智能体):
    - 一个3维连续向量，代表推力 [ax, ay, az]。
      每个分量被限制在动作范围内。

    奖励:
    - 追捕者: 基于接近目标、捕获、燃料消耗等因素。
    - 逃逸者: 基于远离追捕者、存活、燃料消耗等因素。
    """
    def __init__(self, env_config: dict = None):
        super(One_on_one_purchase, self).__init__()

        # 默认配置，可以被 env_config 覆盖
        config = {
            "initial_pursuer_pos": np.array([200000.0, 0.0, 0.0]),
            "initial_pursuer_vel": np.array([0.0, 0.0, 0.0]),
            "initial_evader_pos": np.array([180000.0, 0.0, 0.0]),
            "initial_evader_vel": np.array([0.0, 0.0, 0.0]),
            "initial_fuel_pursuer": 320.0,
            "initial_fuel_evader": 320.0,
            "d_capture": 10000.0,  # 捕获距离 (米)
            "d_range_dangerous_zone_check": 100000.0, # 开始检查危险区域的距离阈值
            "max_episode_steps": 500,
            "win_reward": 100.0,
            "capture_reward_pursuer": 150.0,
            "capture_penalty_evader": -150.0,
            "step_time_interval": 100.0, # CW传播的时间间隔 (秒)
            "action_scale": 1.6,
            "fuel_penalty_factor": 0.01, # 燃料消耗惩罚因子
             # CW 参考点 [N50,E0] 在惯性系下的状态 (来自原始 environment.py)
            "cw_ref_inertial_pos": np.array([27098000.0, 32306000.0, 0.0]),
            "cw_ref_inertial_vel": np.array([-2350.0, 1970.0, 0.0]),
        }
        if env_config:
            config.update(env_config)

        self.env_id = config.get("env_id", "SatelliteGame-v2")
        self.num_agents = 2
        self.agents = ["pursuer_0", "evader_0"]
        self.agent_ids = self.agents

        self._initial_pursuer_pos = np.array(config["initial_pursuer_pos"], dtype=np.float32)
        self._initial_pursuer_vel = np.array(config["initial_pursuer_vel"], dtype=np.float32)
        self._initial_evader_pos = np.array(config["initial_evader_pos"], dtype=np.float32)
        self._initial_evader_vel = np.array(config["initial_evader_vel"], dtype=np.float32)
        
        self.initial_fuel_pursuer = config["initial_fuel_pursuer"]
        self.initial_fuel_evader = config["initial_fuel_evader"]
        self.d_capture = config["d_capture"]
        self.d_range_dangerous_zone_check = config["d_range_dangerous_zone_check"]
        self.max_episode_steps = config["max_episode_steps"]
        self.win_reward = config["win_reward"]
        self.capture_reward_pursuer = config["capture_reward_pursuer"]
        self.capture_penalty_evader = config["capture_penalty_evader"]
        self.step_time_interval = config["step_time_interval"]
        self.action_scale = config["action_scale"]
        self.fuel_penalty_factor = config["fuel_penalty_factor"]

        self.R_cw_ref_inertial = np.array(config["cw_ref_inertial_pos"], dtype=np.float32)
        self.V_cw_ref_inertial = np.array(config["cw_ref_inertial_vel"], dtype=np.float32)

        obs_dim = 18
        obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        
        self.observation_space = {
            agent: Box(low=obs_low, high=obs_high, dtype=np.float32) for agent in self.agents
        }
        self.state_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        act_dim = 3
        act_low = np.full(act_dim, -self.action_scale, dtype=np.float32)
        act_high = np.full(act_dim, self.action_scale, dtype=np.float32)
        self.action_space = {
            agent: Box(low=act_low, high=act_high, dtype=np.float32) for agent in self.agents
        }

        self._current_step = 0
        self.pursuer_pos = np.copy(self._initial_pursuer_pos)
        self.pursuer_vel = np.copy(self._initial_pursuer_vel)
        self.evader_pos = np.copy(self._initial_evader_pos)
        self.evader_vel = np.copy(self._initial_evader_vel)
        self.fuel_pursuer = self.initial_fuel_pursuer
        self.fuel_evader = self.initial_fuel_evader
        self.distance = np.inf
        self.num_dangerous_zones = 0

    def get_env_info(self):
        return {
            'state_space': self.state_space,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps
        }

    def _get_observation(self, agent_id):
        obs = np.concatenate([
            self.pursuer_pos - self.evader_pos,
            self.pursuer_vel - self.evader_vel,
            self.pursuer_pos,
            self.pursuer_vel,
            self.evader_pos,
            self.evader_vel
        ]).astype(np.float32)
        return obs

    def _get_all_observations(self):
        return {agent: self._get_observation(agent) for agent in self.agents}

    def state(self):
        return self._get_observation(self.agents[0]) 

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        self.pursuer_pos = np.copy(self._initial_pursuer_pos)
        self.pursuer_vel = np.copy(self._initial_pursuer_vel)
        self.evader_pos = np.copy(self._initial_evader_pos)
        self.evader_vel = np.copy(self._initial_evader_vel)

        self.fuel_pursuer = self.initial_fuel_pursuer
        self.fuel_evader = self.initial_fuel_evader
        
        self._current_step = 0
        self.distance = np.linalg.norm(self.pursuer_pos - self.evader_pos)
        self.num_dangerous_zones = 0

        observations = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict):
        self._current_step += 1
        
        pursuer_action_raw = actions["pursuer_0"]
        evader_action_raw = actions["evader_0"]

        pursuer_action = np.clip(pursuer_action_raw, -self.action_scale, self.action_scale)
        evader_action = np.clip(evader_action_raw, -self.action_scale, self.action_scale)

        prev_distance = self.distance

        self.pursuer_vel += pursuer_action
        self.evader_vel += evader_action
        
        fuel_consumed_pursuer = np.sum(np.abs(pursuer_action))
        fuel_consumed_evader = np.sum(np.abs(evader_action))
        self.fuel_pursuer -= fuel_consumed_pursuer
        self.fuel_evader -= fuel_consumed_evader

        cw_propagator = sf.Clohessy_Wiltshire(
            R0_c=self.pursuer_pos, V0_c=self.pursuer_vel,
            R0_t=self.evader_pos, V0_t=self.evader_vel
        )
        s_1_new_flat, s_2_new_flat = cw_propagator.State_transition_matrix(self.step_time_interval)

        self.pursuer_pos = s_1_new_flat[0:3]
        self.pursuer_vel = s_1_new_flat[3:6]
        self.evader_pos = s_2_new_flat[0:3]
        self.evader_vel = s_2_new_flat[3:6]

        self.distance = np.linalg.norm(self.pursuer_pos - self.evader_pos)

        if self.distance < self.d_range_dangerous_zone_check:
             self._calculate_number_dangerous_zones()

        # 初始化奖励
        pursuer_reward = 0.0
        evader_reward = 0.0

        # 1. 距离变化奖励/惩罚
        pursuer_reward += (prev_distance - self.distance) * 0.001 # 距离减小则为正
        evader_reward += (self.distance - prev_distance) * 0.001 # 距离增大则为正

        # 2. 燃料消耗惩罚
        pursuer_reward -= fuel_consumed_pursuer * self.fuel_penalty_factor
        evader_reward -= fuel_consumed_evader * self.fuel_penalty_factor
        
        # 3. 危险区域影响 (示例：对双方都是惩罚)
        # pursuer_reward -= self.num_dangerous_zones * 0.1
        # evader_reward -= self.num_dangerous_zones * 0.1
        # 或者根据原始逻辑，追捕者可能因目标进入危险区而受益
        pursuer_reward += self.num_dangerous_zones * 0.05 # 假设追捕者将逃逸者逼入险境是有利的

        # 终止与截断
        terminated = {agent: False for agent in self.agents}
        terminated["__all__"] = False
        truncated = {agent: False for agent in self.agents}
        truncated["__all__"] = False

        info = {agent: {} for agent in self.agents}
        info["pursuer_0"]["fuel"] = self.fuel_pursuer
        info["evader_0"]["fuel"] = self.fuel_evader
        info["pursuer_0"]["distance"] = self.distance
        info["evader_0"]["distance"] = self.distance
        info["common"] = {"num_dangerous_zones": self.num_dangerous_zones}


        if self.distance <= self.d_capture:
            pursuer_reward += self.capture_reward_pursuer
            evader_reward += self.capture_penalty_evader
            terminated["__all__"] = True
            info["pursuer_0"]["capture_status"] = "captured_target"
            info["evader_0"]["capture_status"] = "was_captured"
        
        if self.fuel_pursuer <= 0 and not terminated["__all__"]:
            evader_reward += self.win_reward # 逃逸者因追捕者燃料耗尽而获胜
            terminated["__all__"] = True
            info["pursuer_0"]["capture_status"] = "fuel_exhausted"
            info["evader_0"]["capture_status"] = "evaded_pursuer_fuel_out"
        
        if self.fuel_evader <= 0 and not terminated["__all__"]:
            pursuer_reward += self.win_reward * 0.5 # 追捕者因逃逸者燃料耗尽而部分获胜
            terminated["__all__"] = True
            info["pursuer_0"]["capture_status"] = "target_fuel_exhausted"
            info["evader_0"]["capture_status"] = "fuel_exhausted"

        if self._current_step >= self.max_episode_steps and not terminated["__all__"]:
            truncated["__all__"] = True
            evader_reward += self.win_reward # 逃逸者因超时而获胜
            info["pursuer_0"]["capture_status"] = "timeout_target_evaded"
            info["evader_0"]["capture_status"] = "evaded_by_timeout"

        rewards = {"pursuer_0": pursuer_reward, "evader_0": evader_reward}
        observations = self._get_all_observations()

        if terminated["__all__"]:
            for agent in self.agents: terminated[agent] = True
        if truncated["__all__"]:
            for agent in self.agents: truncated[agent] = True
        
        return observations, rewards, terminated, truncated, info

    def _relative_to_absolute_state(self, R0_relative, V0_relative):
        R_abs = self.R_cw_ref_inertial + R0_relative
        V_abs = self.V_cw_ref_inertial + V0_relative
        return R_abs, V_abs

    def _calculate_number_dangerous_zones(self):
        try:
            R0_c_abs, V0_c_abs = self._relative_to_absolute_state(self.pursuer_pos, self.pursuer_vel)
            R0_t_abs, V0_t_abs = self._relative_to_absolute_state(self.evader_pos, self.evader_vel)

            danger_zone_calculator = sf.Time_window_of_danger_zone(
                R0_c=R0_c_abs, V0_c=V0_c_abs,
                R0_t=R0_t_abs, V0_t=V0_t_abs,
                Delta_V_c=max(0.1, self.fuel_pursuer),
                time_step=1 
            )
            self.num_dangerous_zones = danger_zone_calculator.calculate_number_of_hanger_area()
        except Exception as e:
            # print(f"计算危险区域时出错: {e}") # 调试时可以取消注释
            self.num_dangerous_zones = 0

    def render(self, mode='human'):
        if mode == 'human':
            print(f"步骤: {self._current_step}, 距离: {self.distance:.2f}")
            print(f"  追捕者: Pos={self.pursuer_pos.round(1)}, Vel={self.pursuer_vel.round(2)}, Fuel={self.fuel_pursuer:.1f}")
            print(f"  逃逸者: Pos={self.evader_pos.round(1)}, Vel={self.evader_vel.round(2)}, Fuel={self.fuel_evader:.1f}")
            print(f"  危险区域数量: {self.num_dangerous_zones}")
        elif mode == 'rgb_array':
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            def to_canvas_coords(pos, canvas_size=64, scale=0.00005): # 调整比例以适应位置范围
                x = int(canvas_size / 2 + pos[0] * scale)
                y = int(canvas_size / 2 + pos[1] * scale) # 通常可视化X-Y平面
                return np.clip(x, 0, canvas_size-1), np.clip(y, 0, canvas_size-1)
            px, py = to_canvas_coords(self.pursuer_pos)
            ex, ey = to_canvas_coords(self.evader_pos)
            canvas[py, px, 2] = 255 
            canvas[ey, ex, 0] = 255 
            return canvas
        return None

    def close(self):
        pass

    def avail_actions(self):
        return {agent: None for agent in self.agents}

    def agent_mask(self):
        return {agent: True for agent in self.agents}





REGISTRY = {
    "1on1_puchase": One_on_one_purchase,
    # you can add your customized scenarios here.
}









class SatelliteEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(SatelliteEnv, self).__init__()
        self.env = gym.make(config.env_id)
        self.num_agents = self.env.env.n_agents  # the number of agents
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.seed = config.env_seed  # random seed
        self.env.seed(self.seed)
        self.env.reset(seed=self.seed)

        self.observation_space = {k: self.env.observation_space[i] for i, k in enumerate(self.agents)}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents)}
        self.dim_state = sum([self.observation_space[k].shape[-1] for k in self.agents])
        self.state_space = Box(-np.inf, np.inf, shape=[self.dim_state, ], dtype=np.float32)

        self.max_episode_steps = self.env.env.max_steps
        self._episode_step = 0  # initialize the current step
    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.env_info["n_agents"],
                'max_episode_steps': self.max_episode_steps,
                'num_enemies': self.env.n_enemies}
    def close(self):
        """Close your environment here"""
        self.env.close()

    def render(self, render_mode):
        """Render the environment, and return the images"""
        return self.env.render(render_mode)
    def reset(self):
        """Reset your environment, and return initialized observations and other information."""
        obs, info = self.env.reset()
        obs = np.array(obs)
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        info = {}
        self._episode_step = 0
        return obs_dict, info
    def state(self):
        """Get the global state of the environment in current step."""
        return self.state_space.sample()
    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        actions_mask_list = self.env.get_avail_actions()
        return {key: actions_mask_list[index] for index, key in enumerate(self.agents)}
    

# if __name__ == '__main__':
#     # 用法示例:
#     env_config_custom = {
#         "max_episode_steps": 100,
#         "d_capture": 5000.0, # 米
#         "initial_pursuer_pos": np.array([10000.0, 0.0, 0.0]),
#         "initial_evader_pos": np.array([0.0, 0.0, 0.0]),
#         "step_time_interval": 50.0 # 每步的秒数
#     }
#     env = One_on_one_purchase(env_config=env_config_custom)
    
#     print("环境已初始化。")
#     print(f"智能体数量: {env.num_agents}")
#     print(f"智能体: {env.agents}")
#     print(f"追捕者观测空间: {env.observation_space['pursuer_0']}")
#     print(f"追捕者动作空间: {env.action_space['pursuer_0']}")
#     print(f"最大回合步数: {env.max_episode_steps}")

#     obs, info = env.reset()
#     print("\n初始观测:")
#     for agent_id, agent_obs in obs.items():
#         print(f"  {agent_id}: {agent_obs.shape}") # {agent_obs}")

#     total_reward_pursuer = 0
#     total_reward_evader = 0

#     for i in range(env.max_episode_steps + 5):
#         actions = {
#             "pursuer_0": env.action_space["pursuer_0"].sample(), # 随机动作
#             "evader_0": env.action_space["evader_0"].sample()   # 随机动作
#         }
        
#         next_obs, rewards, terminated, truncated, info = env.step(actions)
        
#         total_reward_pursuer += rewards["pursuer_0"]
#         total_reward_evader += rewards["evader_0"]

#         if i % 20 == 0 or terminated["__all__"] or truncated["__all__"]:
#             print(f"\n--- 步骤 {i+1} ---")
#             # env.render() # 在控制台打印状态
#             print(f"动作: P={actions['pursuer_0'].round(2)}, E={actions['evader_0'].round(2)}")
#             print(f"奖励: P={rewards['pursuer_0']:.2f}, E={rewards['evader_0']:.2f}")
#             print(f"终止: {terminated['__all__']}, 截断: {truncated['__all__']}")
#             print(f"追捕者信息: {info['pursuer_0']}")
#             print(f"逃逸者信息: {info['evader_0']}")
#             # print(f"追捕者下一观测: {next_obs['pursuer_0']}")


#         if terminated["__all__"] or truncated["__all__"]:
#             print(f"\n回合在 {i+1} 步后结束。")
#             if info["pursuer_0"].get("capture"):
#                  print("追捕者捕获了逃逸者！")
#             elif info["evader_0"].get("evaded_timeout"):
#                  print("逃逸者成功逃脱直到超时！")
#             elif info["pursuer_0"].get("fuel_out"):
#                  print("追捕者燃料耗尽。")
#             elif info["evader_0"].get("fuel_out"):
#                  print("逃逸者燃料耗尽。")
#             break
        
#         obs = next_obs

#     print(f"\n追捕者总奖励: {total_reward_pursuer:.2f}")
#     print(f"逃逸者总奖励: {total_reward_evader:.2f}")
    
#     env.close()

    # 与 XuanCe 集成示例 (来自 ippo_new_env.py):
    # from xuance.common import get_configs, recursive_dict_update
    # from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
    # from xuance.torch.utils.operations import set_seed
    # from xuance.torch.agents import IPPO_Agents
    # import argparse

    # REGISTRY_MULTI_AGENT_ENV["SatelliteGame"] = SatelliteGameMAEnv
    # # 然后遵循 XuanCe 脚本结构，例如:
    # # configs_dict = get_configs(file_dir="path_to_your_satellite_game_config.yaml")
    # # configs = argparse.Namespace(**configs_dict)
    # # set_seed(configs.seed)
    # # envs = make_envs(configs) # 这将使用 REGISTRY_MULTI_AGENT_ENV
    # # Agents = IPPO_Agents(config=configs, envs=envs)
    # # Agents.train(...)
if __name__ == '__main__':
    env_config_custom = {
        "max_episode_steps": 200,
        "d_capture": 8000.0,
        "initial_pursuer_pos": np.array([10000.0, 1000.0, 0.0]),
        "initial_evader_pos": np.array([0.0, 0.0, 0.0]),
        "step_time_interval": 50.0 
    }
    env = One_on_one_purchase(env_config=env_config_custom)
    
    obs, info = env.reset()
    print("初始观测 (追捕者):", obs["pursuer_0"])
    
    for i in range(env.max_episode_steps + 5):
        actions = {
            "pursuer_0": env.action_space["pursuer_0"].sample(),
            "evader_0": env.action_space["evader_0"].sample()
        }
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 20 == 0 or terminated["__all__"] or truncated["__all__"]:
            print(f"\n--- 步骤 {i+1} ---")
            env.render()
            print(f"动作: P={actions['pursuer_0'].round(2)}, E={actions['evader_0'].round(2)}")
            print(f"奖励: P={rewards['pursuer_0']:.2f}, E={rewards['evader_0']:.2f}")
            print(f"终止: {terminated['__all__']}, 截断: {truncated['__all__']}")
            print(f"信息: {info}")

        if terminated["__all__"] or truncated["__all__"]:
            print(f"\n回合在 {i+1} 步后结束。")
            break
        obs = next_obs
    env.close()