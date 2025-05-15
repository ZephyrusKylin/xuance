import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt_display
import time 
class Satellite:
    """
    代表单个卫星的类，存储其状态信息。
    单位约定:
        位置: km
        速度: km/s
        轨道根数: a (km), 角度 (rad)
        时间: s
    """
    def __init__(self, id_num, team, initial_pos, initial_orbit, initial_vel=None, function='versatile', initial_epoch_time=0.0): # Added initial_epoch_time
        self.id = id_num
        self.team = team
        
        self.pos = np.array(initial_pos, dtype=np.float32)  # [x, y, z] (km)
        self.vel = np.array(initial_vel, dtype=np.float32) if initial_vel is not None else np.zeros(3, dtype=np.float32) # [vx, vy, vz] (km/s)
        self.orbit = np.array(initial_orbit, dtype=np.float32)  # [a(km), e, i(rad), RAAN(rad), arg_pe(rad), true_anomaly(rad)]
        self.status = 'active'
        self.function = function
        self.epoch_time = initial_epoch_time # 卫星轨道根数对应的历元时间 (s)
        
        self.current_action_type = None
        self.action_duration_remaining = 0.0  # (s)
        
        self.maneuver_target_pos = None # (km)
        self.maneuver_target_orbit = None # [a(km), e, i(rad), RAAN(rad), arg_pe(rad), true_anomaly(rad)]
        self.maneuver_target_vel = None # (km/s)
        
        self.observing_target_id = None
        self.attacking_target_id = None
        
        self.is_changing_formation = False
        self.formation_change_target_idx = None

class MultiAgentSatelliteEnv(gym.Env):
    """
    多智能体卫星对抗环境。
    遵循 Gymnasium API。
    单位约定与Satellite类一致。
    """
    metadata = {'render_modes': ['human', 'rgb_array_matplotlib', 'rgb_array_pybullet'], 'render_fps': 4}

    def __init__(self, num_friendly_satellites, num_enemy_satellites,
                 friendly_configs=None,
                 enemy_initial_configs=None,
                 initial_state_options=None,
                 dt_seconds=10.0,
                 max_episode_time_seconds=24*3600):
        super().__init__()

        self.num_friendly = num_friendly_satellites
        self.num_enemy = num_enemy_satellites
        self.num_satellites = num_friendly_satellites + num_enemy_satellites
        
        self.friendly_configs = friendly_configs if friendly_configs else []
        self.enemy_initial_configs = enemy_initial_configs if enemy_initial_configs else []
        self.initial_state_options = initial_state_options if initial_state_options else {}

        self.satellites = []
        self.num_formation_types = 5 
        self.mu = 398600.4418  # 地球引力常数 (km^3/s^2)

        self.possible_functions = ['versatile', 'observer', 'attacker', 'enemy_default', 'unknown', 'enemy_scout', 'enemy_destroyer', 'visual_sat'] # Added example enemy functions
        self.num_functions = len(self.possible_functions)

        obs_dim_per_sat = 3 + 3 + 6 + 1 + self.num_functions # pos, vel, orbit, status, function_one_hot
        total_obs_dim = obs_dim_per_sat * self.num_satellites
        
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)
            for i in range(self.num_friendly)
        })

        MAX_POS_VAL = 20000
        MAX_A_VAL = 42000 
        MIN_A_VAL = 6600 
        MAX_E_VAL = 0.99
        MAX_ORBIT_PARAMS_ABS = np.array([MAX_A_VAL, MAX_E_VAL, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        MIN_ORBIT_PARAMS_ABS = np.array([MIN_A_VAL, 0, 0, 0, 0, 0])
        MIN_DURATION = 60.0
        MAX_DURATION = 3600.0 * 3

        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Dict({
                "action_type": spaces.Discrete(5), 
                "target_idx": spaces.Discrete(self.num_satellites),
                "maneuver_target_pos": spaces.Box(low=-MAX_POS_VAL, high=MAX_POS_VAL, shape=(3,), dtype=np.float32),
                "maneuver_target_orbit": spaces.Box(low=MIN_ORBIT_PARAMS_ABS, high=MAX_ORBIT_PARAMS_ABS, shape=(6,), dtype=np.float32),
                "maneuver_duration": spaces.Box(low=MIN_DURATION, high=MAX_DURATION, shape=(1,), dtype=np.float32),
                "formation_target_idx": spaces.Discrete(self.num_formation_types),
                "formation_duration": spaces.Box(low=MIN_DURATION, high=MAX_DURATION, shape=(1,), dtype=np.float32),
            }) for i in range(self.num_friendly)
        })

        self.current_time = 0.0 # s
        self.dt = dt_seconds    # s
        self.max_episode_time = max_episode_time_seconds # s
        self.earth_radius_km = 6371.0 # Define earth_radius_km for the class
        self._initialize_satellites()

    def _keplerian_to_cartesian(self, kep_elements):
        a, e, i, raan, argp, nu = kep_elements
        e = np.clip(e, 0, 0.999) 
        if a <= 1e-3: return np.array([0,0,0], dtype=np.float32)
        
        # More robust calculation of E using atan2 for true anomaly to eccentric anomaly
        # E = 2 * np.arctan(np.sqrt(max(0,(1-e)/(1+e))) * np.tan(nu/2)) # This can be unstable if 1+e is near zero or tan(nu/2) is large
        
        # Alternative robust E calculation using cos(E) and sin(E)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)
        
        cos_E = (e + cos_nu) / (1 + e * cos_nu)
        # Clip cos_E to avoid domain errors with arccos due to potential floating point inaccuracies
        cos_E = np.clip(cos_E, -1.0, 1.0)
        
        # sin_E has the same sign as sin_nu
        sin_E_val_sq = (1 - e**2) * sin_nu**2 / (1 + e * cos_nu)**2 if (1+e*cos_nu)**2 > 1e-9 else (1-e**2)*sin_nu**2 # Avoid division by zero
        sin_E = np.sign(sin_nu) * np.sqrt(max(0, sin_E_val_sq)) # max(0,..) for robustness
        
        E = np.arctan2(sin_E, cos_E)


        r_val = a * (1 - e * np.cos(E))
        if r_val < 0 : r_val = 0 # Should not happen with correct E
        Px = np.cos(argp) * np.cos(raan) - np.sin(argp) * np.sin(raan) * np.cos(i)
        Py = np.cos(argp) * np.sin(raan) + np.sin(argp) * np.cos(raan) * np.cos(i)
        Pz = np.sin(argp) * np.sin(i)
        Qx = -np.sin(argp) * np.cos(raan) - np.cos(argp) * np.sin(raan) * np.cos(i)
        Qy = -np.sin(argp) * np.sin(raan) + np.cos(argp) * np.cos(raan) * np.cos(i)
        Qz = np.cos(argp) * np.sin(i)
        x_eci = r_val * (Px * np.cos(nu) + Qx * np.sin(nu))
        y_eci = r_val * (Py * np.cos(nu) + Qy * np.sin(nu))
        z_eci = r_val * (Pz * np.cos(nu) + Qz * np.sin(nu))
        return np.array([x_eci, y_eci, z_eci], dtype=np.float32)

    def _keplerian_to_cartesian_velocity(self, kep_elements):
        a, e, i, raan, argp, nu = kep_elements
        e = np.clip(e, 0, 0.999)
        if a <= 1e-3 or self.mu <= 0: return np.array([0,0,0], dtype=np.float32)
        p = a * (1 - e**2)
        if p <= 1e-3: return np.array([0,0,0], dtype=np.float32) 
        sqrt_mu_p = np.sqrt(self.mu / p)
        vr = sqrt_mu_p * e * np.sin(nu)
        v_nu = sqrt_mu_p * (1 + e * np.cos(nu))
        Px = np.cos(argp) * np.cos(raan) - np.sin(argp) * np.sin(raan) * np.cos(i)
        Py = np.cos(argp) * np.sin(raan) + np.sin(argp) * np.cos(raan) * np.cos(i)
        Pz = np.sin(argp) * np.sin(i)
        Qx = -np.sin(argp) * np.cos(raan) - np.cos(argp) * np.sin(raan) * np.cos(i)
        Qy = -np.sin(argp) * np.sin(raan) + np.cos(argp) * np.cos(raan) * np.cos(i)
        Qz = np.cos(argp) * np.sin(i)
        vx = vr * (Px * np.cos(nu) + Qx * np.sin(nu)) + v_nu * (-Px * np.sin(nu) + Qx * np.cos(nu))
        vy = vr * (Py * np.cos(nu) + Qy * np.sin(nu)) + v_nu * (-Py * np.sin(nu) + Qy * np.cos(nu))
        vz = vr * (Pz * np.cos(nu) + Qz * np.sin(nu)) + v_nu * (-Pz * np.sin(nu) + Qz * np.cos(nu))
        return np.array([vx, vy, vz], dtype=np.float32)

    def _initialize_satellites(self):
        self.satellites = []
        
        for i in range(self.num_friendly):
            config = self.friendly_configs[i] if i < len(self.friendly_configs) else {}
            default_orbit = [7000.0 + i * 200, 0.01, np.deg2rad(10), 0, 0, 0]
            default_function = 'versatile'
            initial_orbit = np.array(config.get('initial_orbit', default_orbit), dtype=np.float32)
            initial_pos = np.array(config.get('initial_pos', self._keplerian_to_cartesian(initial_orbit)), dtype=np.float32)
            initial_vel = np.array(config.get('initial_vel', self._keplerian_to_cartesian_velocity(initial_orbit)), dtype=np.float32)
            function = config.get('function', default_function)
            pos_from_orbit = self._keplerian_to_cartesian(initial_orbit)
            vel_from_orbit = self._keplerian_to_cartesian_velocity(initial_orbit)
            if not np.allclose(initial_pos, pos_from_orbit, atol=1e-1): initial_pos = pos_from_orbit
            if not np.allclose(initial_vel, vel_from_orbit, atol=1e-3): initial_vel = vel_from_orbit
            if function not in self.possible_functions: function = 'unknown'
            self.satellites.append(Satellite(id_num=i, team='friendly', 
                                             initial_pos=initial_pos, initial_orbit=initial_orbit, 
                                             initial_vel=initial_vel, function=function,
                                             initial_epoch_time=self.current_time))
        
        for i in range(self.num_enemy):
            enemy_id = self.num_friendly + i
            config = self.enemy_initial_configs[i] if i < len(self.enemy_initial_configs) else {}
            enemy_opts_key_orbit = f'enemy_{i}_orbit'
            default_orbit = np.array(self.initial_state_options.get(enemy_opts_key_orbit, 
                                                                  [10000.0 + i*200, 0.02, np.deg2rad(20), np.pi, 0, np.pi/2]), 
                                     dtype=np.float32)
            default_function_enemy = 'enemy_default'
            if 'initial_orbit' in config: initial_orbit = np.array(config['initial_orbit'], dtype=np.float32)
            else: initial_orbit = default_orbit
            initial_pos = np.array(config.get('initial_pos', self._keplerian_to_cartesian(initial_orbit)), dtype=np.float32)
            initial_vel = np.array(config.get('initial_vel', self._keplerian_to_cartesian_velocity(initial_orbit)), dtype=np.float32)
            function = config.get('function', default_function_enemy)
            pos_from_orbit = self._keplerian_to_cartesian(initial_orbit)
            vel_from_orbit = self._keplerian_to_cartesian_velocity(initial_orbit)
            if not np.allclose(initial_pos, pos_from_orbit, atol=1e-1): initial_pos = pos_from_orbit
            if not np.allclose(initial_vel, vel_from_orbit, atol=1e-3): initial_vel = vel_from_orbit
            if function not in self.possible_functions: function = 'unknown'
            self.satellites.append(Satellite(id_num=enemy_id, team='enemy',
                                             initial_pos=initial_pos, initial_orbit=initial_orbit,
                                             initial_vel=initial_vel, function=function,
                                             initial_epoch_time=self.current_time))

        for sat in self.satellites:
            sat.current_action_type = None
            sat.action_duration_remaining = 0.0
            sat.status = 'active'
            sat.epoch_time = self.current_time 
            pos_from_orbit = self._keplerian_to_cartesian(sat.orbit)
            vel_from_orbit = self._keplerian_to_cartesian_velocity(sat.orbit)
            if not np.allclose(sat.pos, pos_from_orbit, atol=1e-1): sat.pos = pos_from_orbit
            if not np.allclose(sat.vel, vel_from_orbit, atol=1e-3): sat.vel = vel_from_orbit

    def _get_obs(self):
        all_sats_state_parts = []
        for sat in self.satellites:
            status_val = 1.0 if sat.status == 'active' else 0.0
            pos_clean = np.nan_to_num(sat.pos, nan=0.0, posinf=1e7, neginf=-1e7)
            vel_clean = np.nan_to_num(sat.vel, nan=0.0, posinf=1e2, neginf=-1e2) 
            orbit_clean = np.nan_to_num(sat.orbit, nan=0.0, posinf=1e8, neginf=-1e8)
            function_one_hot = np.zeros(self.num_functions, dtype=np.float32)
            if sat.function in self.possible_functions:
                function_idx = self.possible_functions.index(sat.function)
                function_one_hot[function_idx] = 1.0
            else:
                if 'unknown' in self.possible_functions:
                     function_one_hot[self.possible_functions.index('unknown')] = 1.0
            all_sats_state_parts.extend([*pos_clean, *vel_clean, *orbit_clean, status_val, *function_one_hot])
        full_obs_vector = np.array(all_sats_state_parts, dtype=np.float32)
        observations = {f"agent_{i}": full_obs_vector.copy() for i in range(self.num_friendly)}
        return observations
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.current_time = 0.0
        if options: 
            self.friendly_configs = options.get('friendly_configs', self.friendly_configs)
            self.enemy_initial_configs = options.get('enemy_initial_configs', self.enemy_initial_configs)
            self.initial_state_options = options.get('initial_state_options', self.initial_state_options)
        self._initialize_satellites() # This will populate self.satellites
        
        # Reset matplotlib related flags if they exist, so renderer re-initializes plot objects
        if hasattr(self, '_matplotlib_initialized'):
            self._matplotlib_initialized = False 
        if hasattr(self, 'mpl_sat_plots'):
            self.mpl_sat_plots = []
        if hasattr(self, 'mpl_orbit_plots'):
            self.mpl_orbit_plots = []
        if hasattr(self, 'mpl_view_init_done'):
            del self.mpl_view_init_done
        observations = self._get_obs()
        info = self.get_agent_infos()
        return observations, info

    def step(self, actions):
        rewards = {f"agent_{i}": 0.0 for i in range(self.num_friendly)}
        terminations = {f"agent_{i}": False for i in range(self.num_friendly)}
        truncations = {f"agent_{i}": False for i in range(self.num_friendly)}
        
        for i in range(self.num_friendly):
            agent_sat = self.satellites[i]
            if agent_sat.status == 'destroyed' or agent_sat.action_duration_remaining > 1e-5: continue 
            action_dict = actions[f"agent_{i}"]
            action_type = action_dict["action_type"]
            agent_sat.observing_target_id = None
            agent_sat.attacking_target_id = None
            if action_type == 0: agent_sat.current_action_type = "NO_OP"
            elif action_type == 1: 
                agent_sat.current_action_type = "MANEUVER"
                agent_sat.maneuver_target_orbit = np.array(action_dict["maneuver_target_orbit"], dtype=np.float32)
                agent_sat.maneuver_target_pos = self._keplerian_to_cartesian(agent_sat.maneuver_target_orbit)
                agent_sat.maneuver_target_vel = self._keplerian_to_cartesian_velocity(agent_sat.maneuver_target_orbit)
                duration = float(action_dict["maneuver_duration"][0])
                agent_sat.action_duration_remaining = max(self.dt, duration)
                agent_sat._maneuver_initial_pos = np.copy(agent_sat.pos)
                agent_sat._maneuver_initial_vel = np.copy(agent_sat.vel)
                agent_sat._maneuver_initial_orbit = np.copy(agent_sat.orbit)
                agent_sat._maneuver_total_set_duration = agent_sat.action_duration_remaining
            elif action_type == 2: 
                agent_sat.current_action_type = "OBSERVE"
                target_global_idx = action_dict["target_idx"]
                if 0 <= target_global_idx < self.num_satellites and \
                   self.satellites[target_global_idx].team == 'enemy' and \
                   self.satellites[target_global_idx].status == 'active':
                    agent_sat.observing_target_id = self.satellites[target_global_idx].id
                else: rewards[f"agent_{i}"] -= 0.1 
            elif action_type == 3: 
                agent_sat.current_action_type = "ATTACK"
                target_global_idx = action_dict["target_idx"]
                if 0 <= target_global_idx < self.num_satellites and \
                   self.satellites[target_global_idx].team == 'enemy' and \
                   self.satellites[target_global_idx].status == 'active':
                    agent_sat.attacking_target_id = self.satellites[target_global_idx].id
                else: rewards[f"agent_{i}"] -= 0.1
            elif action_type == 4: 
                agent_sat.current_action_type = "CHANGE_FORMATION"
                agent_sat.formation_change_target_idx = action_dict["formation_target_idx"]
                duration = float(action_dict["formation_duration"][0])
                agent_sat.action_duration_remaining = max(self.dt, duration)
                agent_sat.is_changing_formation = True
        
        self.current_time += self.dt
        for sat_idx, sat in enumerate(self.satellites):
            if sat.status == 'destroyed': continue
            if sat.action_duration_remaining > 1e-5: 
                time_consumed_this_step = min(self.dt, sat.action_duration_remaining)
                if sat.current_action_type == "MANEUVER":
                    total_maneuver_time = sat._maneuver_total_set_duration
                    elapsed_maneuver_time = total_maneuver_time - (sat.action_duration_remaining - time_consumed_this_step)
                    progress_fraction = min(elapsed_maneuver_time / total_maneuver_time, 1.0) if total_maneuver_time > 1e-6 else 1.0
                    sat.pos = sat._maneuver_initial_pos + (sat.maneuver_target_pos - sat._maneuver_initial_pos) * progress_fraction
                    sat.vel = sat._maneuver_initial_vel + (sat.maneuver_target_vel - sat._maneuver_initial_vel) * progress_fraction
                    sat.orbit = sat._maneuver_initial_orbit + (sat.maneuver_target_orbit - sat._maneuver_initial_orbit) * progress_fraction
                sat.action_duration_remaining -= time_consumed_this_step
                if sat.action_duration_remaining <= 1e-5: 
                    sat.action_duration_remaining = 0.0
                    if sat.current_action_type == "MANEUVER":
                        sat.orbit = np.copy(sat.maneuver_target_orbit)
                        sat.pos = self._keplerian_to_cartesian(sat.orbit)
                        sat.vel = self._keplerian_to_cartesian_velocity(sat.orbit)
                        sat.epoch_time = self.current_time # Update epoch_time on maneuver completion
                        del sat._maneuver_initial_pos, sat._maneuver_initial_vel, sat._maneuver_initial_orbit, sat._maneuver_total_set_duration
                        if sat.team == 'friendly': rewards[f"agent_{sat.id}"] += 2.0
                    elif sat.current_action_type == "CHANGE_FORMATION":
                        sat.is_changing_formation = False
                        # Here, epoch_time is not directly tied to orbit change in the same way as maneuver,
                        # but if formation change implies a new reference state, epoch_time could be updated.
                        # For now, only maneuver explicitly updates epoch with new orbit.
                        if sat.team == 'friendly': rewards[f"agent_{sat.id}"] += 1.0
                    sat.current_action_type = None
            elif sat.status == 'active': 
                 self._propagate_orbit_simple(sat, self.dt)

        for friendly_agent_id in range(self.num_friendly):
            friendly_sat = self.satellites[friendly_agent_id]
            if friendly_sat.status != 'active': continue
            if friendly_sat.attacking_target_id is not None:
                target_sat = next((s for s in self.satellites if s.id == friendly_sat.attacking_target_id), None)
                if target_sat and target_sat.status == 'active':
                    distance = np.linalg.norm(friendly_sat.pos - target_sat.pos)
                    ATTACK_RANGE_KM = 1000.0
                    if distance < ATTACK_RANGE_KM:
                        hit_prob = max(0, (1.0 - distance / ATTACK_RANGE_KM)**2) 
                        effective_hit_prob_this_step = 1 - (1 - hit_prob)**(self.dt / 1.0)
                        if random.random() < effective_hit_prob_this_step:
                            target_sat.status = 'destroyed'
                            rewards[f"agent_{friendly_agent_id}"] += 20.0
                friendly_sat.attacking_target_id = None
            if friendly_sat.observing_target_id is not None:
                target_sat = next((s for s in self.satellites if s.id == friendly_sat.observing_target_id), None)
                if target_sat and target_sat.status == 'active':
                    distance = np.linalg.norm(friendly_sat.pos - target_sat.pos)
                    OBSERVE_RANGE_KM = 5000.0
                    if distance < OBSERVE_RANGE_KM:
                        rewards[f"agent_{friendly_agent_id}"] += (1.0 - distance / OBSERVE_RANGE_KM) * 0.5 
                friendly_sat.observing_target_id = None
        
        num_friendly_active = sum(1 for i in range(self.num_friendly) if self.satellites[i].status == 'active')
        num_enemy_active = sum(1 for s in self.satellites if s.team == 'enemy' and s.status == 'active')
        episode_terminated = False
        if num_friendly_active == 0:
            episode_terminated = True
            for i in range(self.num_friendly): rewards[f"agent_{i}"] -= 50.0
        if self.num_enemy > 0 and num_enemy_active == 0: # 敌方全灭 (前提是初始有敌人):
            episode_terminated = True
            for i in range(self.num_friendly): rewards[f"agent_{i}"] += 100.0
        episode_truncated = self.current_time >= self.max_episode_time
        if episode_truncated:
            for i in range(self.num_friendly): rewards[f"agent_{i}"] -= 10.0
        for i in range(self.num_friendly):
            terminations[f"agent_{i}"] = episode_terminated
            truncations[f"agent_{i}"] = episode_truncated
            if self.satellites[i].status == 'active': rewards[f"agent_{i}"] -= 0.01

        observations = self._get_obs()
        info = self.get_agent_infos()
        return observations, rewards, terminations, truncations, info

    def _propagate_orbit_simple(self, satellite, dt_seconds):
        if satellite.orbit[0] <= 1e-3: return
        a, e, i, raan, argp, nu = satellite.orbit
        e = np.clip(e, 0, 0.999)
        n = np.sqrt(self.mu / (a**3))
        
        # Robust E0 calculation
        cos_nu0 = np.cos(nu)
        sin_nu0 = np.sin(nu)
        cos_E0 = (e + cos_nu0) / (1 + e * cos_nu0)
        cos_E0 = np.clip(cos_E0, -1.0, 1.0)
        sin_E0_val_sq = (1 - e**2) * sin_nu0**2 / (1 + e * cos_nu0)**2 if (1+e*cos_nu0)**2 > 1e-9 else (1-e**2)*sin_nu0**2
        sin_E0 = np.sign(sin_nu0) * np.sqrt(max(0, sin_E0_val_sq))
        E0 = np.arctan2(sin_E0, cos_E0)
        
        M0 = E0 - e * np.sin(E0) 
        M1 = (M0 + n * dt_seconds)
        
        E1 = M1 # Initial guess for Newton's method
        for _ in range(8): # Iterate to solve Kepler's equation for E1
            f_E = E1 - e * np.sin(E1) - M1
            f_prime_E = 1 - e * np.cos(E1)
            if abs(f_prime_E) < 1e-9: break # Avoid division by zero
            delta_E = f_E / f_prime_E
            E1 = E1 - delta_E
            if abs(delta_E) < 1e-8: break # Convergence
            
        # Robust nu_new calculation from E1
        cos_E1 = np.cos(E1)
        sin_E1 = np.sin(E1)
        cos_nu1 = (cos_E1 - e) / (1 - e * cos_E1)
        cos_nu1 = np.clip(cos_nu1, -1.0, 1.0)
        sin_nu1_val_sq = (1 - e**2) * sin_E1**2 / (1 - e * cos_E1)**2 if (1-e*cos_E1)**2 > 1e-9 else (1-e**2)*sin_E1**2
        sin_nu1 = np.sign(sin_E1) * np.sqrt(max(0, sin_nu1_val_sq))
        nu_new = np.arctan2(sin_nu1, cos_nu1)

        satellite.orbit[5] = nu_new % (2 * np.pi)
        satellite.pos = self._keplerian_to_cartesian(satellite.orbit)
        satellite.vel = self._keplerian_to_cartesian_velocity(satellite.orbit)
        satellite.epoch_time = self.current_time

    def get_agent_infos(self):
        infos = {}
        for i in range(self.num_friendly):
            sat = self.satellites[i]
            agent_name = f"agent_{i}"
            infos[agent_name] = {
                "id": sat.id,
                "team": sat.team,
                "function": sat.function,
                "position_km": sat.pos.tolist(),
                "velocity_km_s": sat.vel.tolist(),
                "orbit_kep_km_rad": sat.orbit.tolist(),
                "status": sat.status,
                "current_action": sat.current_action_type,
                "action_time_left_s": sat.action_duration_remaining,
                "epoch_time_s": sat.epoch_time # Added epoch_time
            }
            if sat.observing_target_id is not None: infos[agent_name]["observing_target_id"] = sat.observing_target_id
            if sat.attacking_target_id is not None: infos[agent_name]["attacking_target_id"] = sat.attacking_target_id
        return infos

    # def render(self, mode='human'):
    #     if mode == 'human':
    #         print(f"--- 当前仿真时间: {self.current_time:.2f} s ({self.current_time/3600:.2f} h) ---")
    #         for i, sat in enumerate(self.satellites):
    #             team_symbol = "[友]" if sat.team == 'friendly' else "[敌]"
    #             status_symbol = "✅" if sat.status == 'active' else "❌"
    #             func_str = f"({sat.function})"
    #             action_info = ""
    #             if sat.action_duration_remaining > 0:
    #                 action_info = f" ({sat.current_action_type}, 剩余 {sat.action_duration_remaining:.1f}s)"
    #             elif sat.current_action_type == "OBSERVE" and sat.observing_target_id is not None:
    #                  action_info = f" (正在观测 目标ID:{sat.observing_target_id})"
    #             elif sat.current_action_type == "ATTACK" and sat.attacking_target_id is not None:
    #                  action_info = f" (正在攻击 目标ID:{sat.attacking_target_id})"
    #             pos_str = f"位置(km):({sat.pos[0]:.1f}, {sat.pos[1]:.1f}, {sat.pos[2]:.1f})"
    #             vel_str = f"速度(km/s):({sat.vel[0]:.2f}, {sat.vel[1]:.2f}, {sat.vel[2]:.2f})"
    #             orbit_a = sat.orbit[0]
    #             orbit_e = sat.orbit[1]
    #             orbit_i_deg = np.rad2deg(sat.orbit[2])
    #             orbit_raan_deg = np.rad2deg(sat.orbit[3])
    #             orbit_argp_deg = np.rad2deg(sat.orbit[4])
    #             orbit_nu_deg = np.rad2deg(sat.orbit[5])
    #             epoch_str = f"历元: {sat.epoch_time:.1f}s" # Epoch string for rendering
    #             orbit_str = (f"轨道: a={orbit_a:.0f}km, e={orbit_e:.4f}, "
    #                          f"i={orbit_i_deg:.1f}°, RAAN={orbit_raan_deg:.1f}°, "
    #                          f"ω={orbit_argp_deg:.1f}°, ν={orbit_nu_deg:.1f}° ({epoch_str})")
    #             print(f"卫星 {sat.id} {team_symbol}{status_symbol}{func_str}: {pos_str} | {vel_str} | {orbit_str}{action_info}")
    #         print("-" * 100)
    def _initialize_pybullet_renderer(self, render_width=320, render_height=240):
        """
        初始化PyBullet渲染器。
        在第一次调用 PyBullet 渲染模式时执行。
        单位约定：PyBullet中的单位将直接对应环境中的公里(km)。
        """
        import pybullet as p # 动态导入
        import pybullet_data

        self.render_width_px = render_width
        self.render_height_px = render_height
        self.pb_client_id = -1 # 初始化 PyBullet 客户端 ID

        try:
            # 尝试连接到共享内存，如果GUI已手动启动
            # self.pb_client_id = p.connect(p.SHARED_MEMORY)
            # if self.pb_client_id < 0:
            # 对于 rgb_array，通常使用 DIRECT 模式
            self.pb_client_id = p.connect(p.DIRECT)
        except p.error:
            # 如果共享内存失败或 p.GUI 不可用，则回退到 DIRECT 模式
            self.pb_client_id = p.connect(p.DIRECT)
        
        if self.pb_client_id < 0:
            raise ConnectionError("无法连接到 PyBullet。")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pb_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.pb_client_id) # 在DIRECT模式下禁用GUI窗口
        p.setGravity(0, 0, 0, physicsClientId=self.pb_client_id)

        # 创建地球 (一个大的蓝色球体)
        # 环境中的单位是km, PyBullet中的单位也视为km
        self.earth_radius_km = 6371.0 # 地球半径 (km)
        earth_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, 
                                                 radius=self.earth_radius_km, 
                                                 rgbaColor=[0.3, 0.3, 0.8, 1], 
                                                 physicsClientId=self.pb_client_id)
        self.earth_pb_id = p.createMultiBody(baseVisualShapeIndex=earth_visual_shape, 
                                             basePosition=[0,0,0], 
                                             physicsClientId=self.pb_client_id)

        # 为每个卫星创建PyBullet表示 (小球体)
        self.satellite_pb_ids = []
        # 卫星的视觉半径 (km) - 为了可见性，可以不成比例放大
        # 例如，如果卫星本身尺寸很小，但在数万公里外，直接用其实际大小可能看不见
        # 这里我们用一个固定的、相对较大的可视化半径
        satellite_visual_radius_km = 100.0 # 视觉半径设为100km，以便观察

        for i, sat in enumerate(self.satellites):
            color = [0.9, 0.2, 0.2, 1] if sat.team == 'friendly' else \
                    ([0.2, 0.9, 0.2, 1] if sat.team == 'enemy' else [0.7, 0.7, 0.7, 1])
            
            visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, 
                                                  radius=satellite_visual_radius_km, 
                                                  rgbaColor=color, 
                                                  physicsClientId=self.pb_client_id)
            # 初始将卫星放置在远离视野的地方，在render循环中更新其位置
            initial_far_pos = [200000 + i*1000, 200000, 200000] 
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, 
                                        basePosition=initial_far_pos, 
                                        physicsClientId=self.pb_client_id)
            self.satellite_pb_ids.append(body_id)
        
        # 存储一个标志，表示PyBullet已初始化
        self._pybullet_initialized = True
    def _initialize_matplotlib_renderer(self, render_width_px=640, render_height_px=480):
        """
        初始化Matplotlib 3D渲染器。
        在第一次调用 Matplotlib 渲染模式时执行。
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # 用于3D绘图
        plt.style.use('dark_background') # 使用深色背景主题

        self.mpl_render_width_px = render_width_px
        self.mpl_render_height_px = render_height_px
        dpi = 100 # 可以根据需要调整DPI
        # Close existing figure if it exists to prevent multiple windows
        if hasattr(self, 'mpl_fig') and self.mpl_fig is not None:
            plt.close(self.mpl_fig)
        self.mpl_fig = plt.figure(figsize=(self.mpl_render_width_px/dpi, self.mpl_render_height_px/dpi), dpi=dpi)
        self.mpl_ax = self.mpl_fig.add_subplot(111, projection='3d')
        
        # 预先绘制地球 (一个球体)
        self.earth_radius_km = 6371.0 # 与PyBullet中一致
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        earth_x = self.earth_radius_km * np.outer(np.cos(u), np.sin(v))
        earth_y = self.earth_radius_km * np.outer(np.sin(u), np.sin(v))
        earth_z = self.earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        # self.mpl_ax.plot_surface(earth_x, earth_y, earth_z, color='deepskyblue', alpha=0.7, rstride=4, cstride=4, linewidth=0)
        self.mpl_earth_surface = self.mpl_ax.plot_surface(earth_x, earth_y, earth_z, color='deepskyblue', alpha=0.6, rstride=4, cstride=4, linewidth=0, zorder=1) # Earth zorder
        # 存储卫星的绘图对象 (用于更新)
        self.mpl_sat_plots = []
        self.mpl_orbit_plots = []
        if hasattr(self, 'satellites'): # Ensure satellites list exists
            for i, sat in enumerate(self.satellites): # 假设 self.satellites 已可用
                color_dot = 'red' if sat.team == 'friendly' else 'lime'
                color_orbit = 'salmon' if sat.team == 'friendly' else 'lightgreen'
                
                # 初始绘制一个点，后续更新其位置
                sat_plot, = self.mpl_ax.plot([], [], [], marker='o', markersize=5, color=color_dot, label=f"Sat {sat.id} ({sat.team})", zorder=5)
                self.mpl_sat_plots.append(sat_plot)
                
                # 初始绘制一条线，后续更新其路径
                orbit_plot, = self.mpl_ax.plot([], [], [], color=color_orbit, linestyle='--', linewidth=1, zorder=3)
            self.mpl_orbit_plots.append(orbit_plot)
        
        self.mpl_ax.set_xlabel("X (km)")
        self.mpl_ax.set_ylabel("Y (km)")
        self.mpl_ax.set_zlabel("Z (km)")
        self.mpl_ax.set_title("卫星轨道可视化")
        # self.mpl_ax.legend() # 图例可能会使图像混乱，根据需要启用

        self._matplotlib_initialized = True
        self.mpl_view_init_done = False # Flag to set view only once


    def _plot_orbit_matplotlib(self, ax, kep_elements, color='gray', num_points=200):
        """
        辅助函数：根据开普勒六根数在Matplotlib 3D轴上绘制完整轨道。
        kep_elements: [a, e, i, RAAN, arg_pe, nu] - nu将被忽略，因为我们绘制整个轨道。
        """
        # 确保 _keplerian_to_cartesian 方法可用 (来自 "代号1代码")
        if not hasattr(self, '_keplerian_to_cartesian'):
            print("错误: _keplerian_to_cartesian 方法未在类中定义。")
            return np.array([]), np.array([]), np.array([])

        a, e, i_rad, raan_rad, argp_rad, _ = kep_elements # 忽略当前的真近点角
        
        # 生成一系列真近点角来绘制轨道
        nu_values = np.linspace(0, 2 * np.pi, num_points)
        orbit_points_x = []
        orbit_points_y = []
        orbit_points_z = []

        for nu_val in nu_values:
            current_kep = np.array([a, e, i_rad, raan_rad, argp_rad, nu_val], dtype=np.float32)
            pos_cartesian = self._keplerian_to_cartesian(current_kep)
            orbit_points_x.append(pos_cartesian[0])
            orbit_points_y.append(pos_cartesian[1])
            orbit_points_z.append(pos_cartesian[2])
        
        return np.array(orbit_points_x), np.array(orbit_points_y), np.array(orbit_points_z)
    def render(self, mode='human'):
        if mode == 'human':
            current_time = getattr(self, 'current_time', 0.0)
            satellites_list = getattr(self, 'satellites', [])
            print(f"--- 当前仿真时间: {current_time:.2f} s ({current_time/3600:.2f} h) ---")
            for i, sat in enumerate(satellites_list):
                team_symbol = "[友]" if sat.team == 'friendly' else "[敌]"
                status_symbol = "✅" if sat.status == 'active' else "❌"
                func_str = f"({getattr(sat, 'function', 'N/A')})"
                action_info = ""
                if getattr(sat, 'action_duration_remaining', 0) > 0:
                    action_info = f" ({getattr(sat, 'current_action_type', 'N/A')}, 剩余 {sat.action_duration_remaining:.1f}s)"
                pos_str = f"位置(km):({getattr(sat, 'pos', [0,0,0])[0]:.1f}, {getattr(sat, 'pos', [0,0,0])[1]:.1f}, {getattr(sat, 'pos', [0,0,0])[2]:.1f})"
                vel_str = f"速度(km/s):({getattr(sat, 'vel', [0,0,0])[0]:.2f}, {getattr(sat, 'vel', [0,0,0])[1]:.2f}, {getattr(sat, 'vel', [0,0,0])[2]:.2f})"
                orbit = getattr(sat, 'orbit', np.zeros(6))
                orbit_a, orbit_e = orbit[0], orbit[1]
                orbit_i_deg, orbit_raan_deg = np.rad2deg(orbit[2]), np.rad2deg(orbit[3])
                orbit_argp_deg, orbit_nu_deg = np.rad2deg(orbit[4]), np.rad2deg(orbit[5])
                epoch_str = f"历元: {getattr(sat, 'epoch_time', 0.0):.1f}s"
                orbit_str = (f"轨道: a={orbit_a:.0f}km, e={orbit_e:.4f}, "
                             f"i={orbit_i_deg:.1f}°, RAAN={orbit_raan_deg:.1f}°, "
                             f"ω={orbit_argp_deg:.1f}°, ν={orbit_nu_deg:.1f}° ({epoch_str})")
                print(f"卫星 {getattr(sat, 'id', 'N/A')} {team_symbol}{status_symbol}{func_str}: {pos_str} | {vel_str} | {orbit_str}{action_info}")
            print("-" * 100)
            return None
        
        elif mode == 'rgb_array_matplotlib':
            import matplotlib.pyplot as plt
            if not hasattr(self, '_matplotlib_initialized') or not self._matplotlib_initialized \
                or not hasattr(self, 'mpl_fig') or self.mpl_fig is None or not plt.fignum_exists(self.mpl_fig.number): # Check if fig still exists
                if not hasattr(self, 'satellites') or not self.satellites:
                    # print("Warning: Cannot initialize Matplotlib renderer, satellites not populated.")
                    return np.zeros((self.mpl_render_height_px if hasattr(self, 'mpl_render_height_px') else 480, 
                                     self.mpl_render_width_px if hasattr(self, 'mpl_render_width_px') else 640, 3), dtype=np.uint8)
                self._initialize_matplotlib_renderer()
            
            # Ensure plot objects match current number of satellites
            if len(self.mpl_sat_plots) != len(self.satellites) or len(self.mpl_orbit_plots) != len(self.satellites):
                # This can happen if reset is called and number of satellites changes. Re-initialize plots.
                # print("Warning: Number of satellites changed or plot objects mismatch. Re-initializing Matplotlib plots.")
                # Clear old plot objects from axes first
                for plot_obj_list in [self.mpl_sat_plots, self.mpl_orbit_plots]:
                    for plot_obj in plot_obj_list:
                        if plot_obj in self.mpl_ax.lines or plot_obj in self.mpl_ax.collections:
                             plot_obj.remove()
                self.mpl_sat_plots = []
                self.mpl_orbit_plots = []
                for i, sat in enumerate(self.satellites):
                    color_dot = 'red' if sat.team == 'friendly' else 'lime'
                    color_orbit = 'salmon' if sat.team == 'friendly' else 'lightgreen'
                    sat_plot, = self.mpl_ax.plot([], [], [], marker='o', markersize=5, color=color_dot, label=f"Sat {sat.id} ({sat.team})")
                    self.mpl_sat_plots.append(sat_plot)
                    orbit_plot, = self.mpl_ax.plot([], [], [], color=color_orbit, linestyle='--', linewidth=1)
                    self.mpl_orbit_plots.append(orbit_plot)


            max_coord = self.earth_radius_km 
            for i, sat in enumerate(self.satellites):
                sat_plot = self.mpl_sat_plots[i]
                orbit_plot = self.mpl_orbit_plots[i]
                if sat.status == 'active':
                    sat_plot.set_data_3d([sat.pos[0]], [sat.pos[1]], [sat.pos[2]])
                    sat_plot.set_visible(True)
                    orbit_x, orbit_y, orbit_z = self._plot_orbit_matplotlib(self.mpl_ax, sat.orbit)
                    orbit_plot.set_data_3d(orbit_x, orbit_y, orbit_z)
                    orbit_plot.set_visible(True)
                    current_max_abs_pos = np.max(np.abs(sat.pos))
                    apogee_dist_approx = sat.orbit[0] * (1 + sat.orbit[1]) if sat.orbit[0] > 0 else current_max_abs_pos
                    current_max_orbit_extent = max(current_max_abs_pos, apogee_dist_approx)
                    if current_max_orbit_extent > max_coord: max_coord = current_max_orbit_extent
                else:
                    sat_plot.set_visible(False)
                    orbit_plot.set_visible(False)
            limit = max_coord * 1.2 if max_coord > 0 else self.earth_radius_km * 2
            self.mpl_ax.set_xlim([-limit, limit])
            self.mpl_ax.set_ylim([-limit, limit])
            self.mpl_ax.set_zlim([-limit, limit])
            if not self.mpl_view_init_done: # Set initial view once per figure
                self.mpl_ax.view_init(elev=25., azim=135)
                self.mpl_view_init_done = True
            self.mpl_fig.canvas.draw()
            buf = self.mpl_fig.canvas.buffer_rgba()
            rgb_array = np.asarray(buf)[:, :, :3]
            return rgb_array
            
        else:
            raise NotImplementedError(f"渲染模式 '{mode}' 尚不支持或未实现。")

    def close(self):
        """
        清理环境资源，例如关闭PyBullet连接。
        这个方法应该在环境结束时被调用。
        """
        # --- 这是 close 方法的扩展，用于关闭PyBullet ---
        if hasattr(self, '_pybullet_initialized') and self._pybullet_initialized:
            if hasattr(self, 'pb_client_id') and self.pb_client_id >= 0:
                try:
                    import pybullet as p
                    p.disconnect(physicsClientId=self.pb_client_id)
                    print("PyBullet 渲染器已关闭。")
                except p.error:
                    print("关闭 PyBullet 渲染器时出错 (可能已断开连接)。")
                except ImportError:
                    pass # PyBullet 未导入，无需操作
                self.pb_client_id = -1
                self._pybullet_initialized = False
        if hasattr(self, '_matplotlib_initialized') and self._matplotlib_initialized:
            if hasattr(self, 'mpl_fig') and self.mpl_fig is not None:
                try:
                    import matplotlib.pyplot as plt
                    if plt.fignum_exists(self.mpl_fig.number):
                        plt.close(self.mpl_fig)
                        print("Matplotlib 渲染器已关闭。")
                except Exception: pass # Ignore errors during close
            self.mpl_fig = None # Ensure fig is cleared
            self._matplotlib_initialized = False
        # 如果父类有close方法，也应该调用
        # super().close() 
        print("MultiAgentSatelliteEnv 已关闭。")


# === 环境使用示例 ===
if __name__ == '__main__':
    FLAG = 1
    if FLAG == 0:
        friendly_agent_configs = [
            {'initial_orbit': [7000, 0.001, np.deg2rad(28.5), np.deg2rad(10), np.deg2rad(5), np.deg2rad(0)], 'function': 'observer'},
            {'initial_orbit': [7200, 0.002, np.deg2rad(28.5), np.deg2rad(10), np.deg2rad(50), np.deg2rad(90)], 'function': 'attacker'}
        ]
        enemy_agent_configs = [
            {'initial_orbit': [12500, 0.01, 0.523, 1.57, 0.1, 0.3], 'function': 'enemy_scout'},
            {'initial_orbit': [15000, 0.05, np.deg2rad(60), np.deg2rad(120), np.deg2rad(30), np.deg2rad(180)], 'function': 'enemy_destroyer'}
        ]
        
        env = MultiAgentSatelliteEnv(num_friendly_satellites=len(friendly_agent_configs), 
                                    num_enemy_satellites=len(enemy_agent_configs),
                                    friendly_configs=friendly_agent_configs,
                                    enemy_initial_configs=enemy_agent_configs,
                                    dt_seconds=60.0)

        reset_options = {'friendly_configs': friendly_agent_configs, 'enemy_initial_configs': enemy_agent_configs}
        obs, info = env.reset(options=reset_options)
        
        print("\n--- 初始状态 ---")
        env.render()
        # print("初始信息 (Agent 0):", info["agent_0"])

        example_action_agent_0 = {
            "action_type": 1, "target_idx": 0, 
            "maneuver_target_orbit": np.array([7500, 0.002, np.deg2rad(28.0), np.deg2rad(12), np.deg2rad(8), np.deg2rad(30)], dtype=np.float32),
            "maneuver_duration": np.array([1800.0], dtype=np.float32),
            "maneuver_target_pos": np.zeros(3), 
            "formation_target_idx": 0, "formation_duration": np.array([60.0], dtype=np.float32)
        }
        example_action_agent_1 = { "action_type": 0, "target_idx": 0,
            "maneuver_target_pos": np.zeros(3), "maneuver_target_orbit": np.zeros(6),
            "maneuver_duration": np.array([env.dt]), "formation_target_idx": 0,
            "formation_duration": np.array([env.dt])
        }
        actions_dict = {"agent_0": example_action_agent_0}
        if env.num_friendly > 1: actions_dict["agent_1"] = example_action_agent_1
        
        for step_num in range(5):
            print(f"\n>>> 第 {step_num + 1} 步 <<<")
            current_actions_for_step = {}
            for i in range(env.num_friendly):
                agent_name = f"agent_{i}"
                if env.satellites[i].action_duration_remaining <= 1e-5:
                    current_actions_for_step[agent_name] = actions_dict.get(agent_name, example_action_agent_1)
                else: current_actions_for_step[agent_name] = example_action_agent_1
            observations, rewards, terminations, truncations, info = env.step(current_actions_for_step)
            env.render()
            for i in range(env.num_friendly):
                agent_name = f"agent_{i}"
                print(f"奖励 ({agent_name}): {rewards[agent_name]:.3f}")
                print(f"终止 ({agent_name}): {terminations[agent_name]}, 截断 ({agent_name}): {truncations[agent_name]}")
                print(f"信息 ({agent_name}): 历元 = {info[agent_name]['epoch_time_s']:.1f}s")

            if all(terminations.values()) or all(truncations.values()):
                print("所有 Agent 终止或场景截断，结束模拟。")
                break
        # try:
        #     rgb_image_pybullet = env.render(mode='rgb_array_pybullet')
        #     if rgb_image_pybullet is not None:
        #         print(f"PyBullet 渲染图像形状: {rgb_image_pybullet.shape}")
        #         pass # 可以使用matplotlib等库显示图像
        # except ImportError:
        #     print("PyBullet 未安装，跳过 PyBullet 渲染示例。")
        # except ConnectionError as e:
        #     print(f"PyBullet 连接错误: {e}")
        # except Exception as e:
        #     print(f"PyBullet 渲染时发生未知错误: {e}")

        try:
            rgb_image_mpl = env.render(mode='rgb_array_matplotlib')
            if rgb_image_mpl is not None:
                print(f"Matplotlib 渲染图像形状: {rgb_image_mpl.shape}")
                import matplotlib.pyplot as plt_display
                plt_display.imshow(rgb_image_mpl)
                plt_display.title("Matplotlib Render Output")
                plt_display.show()
        except ImportError:
            print("Matplotlib 未安装，跳过 Matplotlib 渲染示例。")
        except Exception as e:
            print(f"Matplotlib 渲染时发生错误: {e}")
            import traceback
            traceback.print_exc()

        env.close()

        '''
        print("--- 开始执行新增测试用例 (Matplotlib轨道动画) ---")
        # 定义一个卫星进行测试
        # 近地轨道示例: a=8000km, e=0.1, i=45deg
        single_sat_config = [
            {'initial_orbit': [
                8000.0,                  # a (km)
                0.1,                     # e
                np.deg2rad(45.0),        # i (rad)
                np.deg2rad(20.0),        # RAAN (rad)
                np.deg2rad(10.0),        # arg_pe (rad)
                0.0                      # initial true anomaly (rad)
            ], 'function': 'visual_sat'}
        ]

        # 使用 dt=60s 进行动画
        env_anim = MultiAgentSatelliteEnv(num_friendly_satellites=1, 
                                    num_enemy_satellites=0,
                                    friendly_configs=single_sat_config,
                                    dt_seconds=60.0) 
        
        obs_anim, info_anim = env_anim.reset(options={'friendly_configs': single_sat_config})
        
        # 计算轨道周期
        a_km_anim = env_anim.satellites[0].orbit[0]
        orbital_period_seconds_anim = 2 * np.pi * np.sqrt(a_km_anim**3 / env_anim.mu)
        print(f"动画卫星轨道参数: a={a_km_anim:.0f}km, e={env_anim.satellites[0].orbit[1]:.3f}")
        print(f"计算得到的轨道周期: {orbital_period_seconds_anim:.2f} 秒 (~{(orbital_period_seconds_anim/3600):.2f} 小时)")

        dt_anim = env_anim.dt # 使用环境的dt
        num_simulation_steps_anim = int(orbital_period_seconds_anim / dt_anim) + 10 # 模拟一整圈再加几步
        
        print(f"将以 dt={dt_anim}s 模拟 {num_simulation_steps_anim} 步来观察完整轨道。")

        # 初始化 Matplotlib 渲染器 (这将创建图形窗口)
        # _initialize_matplotlib_renderer 会在第一次调用render时自动执行
        
        plt_display.ion() # 打开交互模式以便逐帧更新
        
        # 渲染第一帧
        initial_frame_anim = env_anim.render(mode='rgb_array_matplotlib')
        if initial_frame_anim is not None:
            # imshow可以用来显示render返回的图像数组，但对于动画，我们主要依赖figure的更新
            # mock_env.mpl_ax.imshow(initial_frame, aspect='auto', extent=(-a_km*1.5,a_km*1.5,-a_km*1.5,a_km*1.5))
            env_anim.mpl_fig.suptitle(f"时间: {env_anim.current_time:.0f}s / {orbital_period_seconds_anim:.0f}s", fontsize=10)
            plt_display.pause(1) 

        for step_anim in range(num_simulation_steps_anim):
            # 1. 更新卫星状态 (使用环境的传播逻辑)
            #    对于这个测试，我们不发送动作，只观察自然轨道运动
            #    所以我们直接调用 _propagate_orbit_simple 来更新单个卫星的状态
            #    或者，如果想测试完整的step，可以构造一个NO_OP动作
            
            # 手动传播单个卫星，模拟自然运动
            env_anim._propagate_orbit_simple(env_anim.satellites[0], dt_anim)
            env_anim.current_time += dt_anim # 手动更新环境时间，因为我们没调用完整的step

            # 2. 渲染当前帧
            current_nu_anim = env_anim.satellites[0].orbit[5]
            print(f"动画模拟步骤 {step_anim+1}/{num_simulation_steps_anim}, 当前时间: {env_anim.current_time:.0f}s, nu: {np.rad2deg(current_nu_anim):.1f}°")
            
            # 调用render来更新Matplotlib图形并获取图像数组（如果需要）
            rgb_image_mpl_anim = env_anim.render(mode='rgb_array_matplotlib') 
            
            if rgb_image_mpl_anim is not None:
                env_anim.mpl_fig.suptitle(f"时间: {env_anim.current_time:.0f}s / {orbital_period_seconds_anim:.0f}s (nu: {np.rad2deg(current_nu_anim):.1f}°)", fontsize=10)
                plt_display.pause(0.01) 
            else:
                print("Matplotlib 动画渲染失败。")
                break
            
            if env_anim.current_time > orbital_period_seconds_anim + dt_anim * 5: 
                print("已完成一圈多的轨道动画模拟。")
                break

        print("轨道动画模拟完成。")
        plt_display.ioff() 
        plt_display.show() # 保持窗口打开直到手动关闭
        
        env_anim.close()
        print("--- 新增测试用例 (Matplotlib轨道动画) 执行完毕 ---")
        '''
        print("--- 开始执行新增测试用例 (Matplotlib轨道动画) ---")
        # single_sat_config = [
        #     {'initial_orbit': [
        #         8000.0, 0.1, np.deg2rad(45.0), np.deg2rad(20.0), np.deg2rad(10.0), 0.0
        #     ], 'function': 'visual_sat'}
        # ]
        # env_anim = MultiAgentSatelliteEnv(num_friendly_satellites=1, 
        #                              num_enemy_satellites=0,
        #                              friendly_configs=single_sat_config,
        #                              dt_seconds=60.0) 
        # obs_anim, info_anim = env_anim.reset(options={'friendly_configs': single_sat_config})
        friendly_agent_configs_anim = [
            {'initial_orbit': [7000, 0.001, np.deg2rad(28.5), np.deg2rad(10), np.deg2rad(5), np.deg2rad(0)], 'function': 'observer'},
            {'initial_orbit': [7200, 0.002, np.deg2rad(28.5), np.deg2rad(10), np.deg2rad(50), np.deg2rad(90)], 'function': 'attacker'}
        ]
        enemy_agent_configs_anim = [
            {'initial_orbit': [12500, 0.01, 0.523, 1.57, 0.1, 0.3], 'function': 'enemy_scout'},
            {'initial_orbit': [15000, 0.05, np.deg2rad(60), np.deg2rad(120), np.deg2rad(30), np.deg2rad(180)], 'function': 'enemy_destroyer'}
        ]
        
        env_anim = MultiAgentSatelliteEnv(num_friendly_satellites=len(friendly_agent_configs), 
                                    num_enemy_satellites=len(enemy_agent_configs),
                                    friendly_configs=friendly_agent_configs,
                                    enemy_initial_configs=enemy_agent_configs,
                                    dt_seconds=60.0)

            
        obs_anim, info_anim = env_anim.reset(options={
            'friendly_configs': friendly_agent_configs_anim,
            'enemy_initial_configs': enemy_agent_configs_anim
        })
            # 计算所有卫星的轨道周期并找到最大值
        max_orbital_period_all_sats = 0
        all_orbital_periods = []
        for sat_idx, sat_obj in enumerate(env_anim.satellites):
            a_km_sat = sat_obj.orbit[0]
            if a_km_sat > 0: # Ensure semi-major axis is positive
                period_s = 2 * np.pi * np.sqrt(a_km_sat**3 / env_anim.mu)
                all_orbital_periods.append(period_s)
                print(f"卫星 {sat_obj.id} ({sat_obj.team}, {sat_obj.function}): a={a_km_sat:.0f}km, 周期={period_s:.0f}s (~{(period_s/3600):.1f}h)")
                if period_s > max_orbital_period_all_sats:
                    max_orbital_period_all_sats = period_s
            else:
                print(f"卫星 {sat_obj.id} ({sat_obj.team}) 轨道参数无效 (a={a_km_sat}), 无法计算周期。")
                all_orbital_periods.append(0)


        if max_orbital_period_all_sats == 0 and env_anim.satellites: # Fallback if no valid periods
            print("警告: 未能计算任何有效轨道周期。默认模拟100步。")
            max_orbital_period_all_sats = env_anim.dt * 50 # Default to some steps
        
        # 模拟大约两周（两倍最长轨道周期）
        simulation_duration_seconds = max_orbital_period_all_sats * 2
        num_simulation_steps_anim = int(simulation_duration_seconds / env_anim.dt) + 1
        
        # 更新环境的最大幕时长以匹配我们的目标，或确保它足够长
        env_anim.max_episode_time = simulation_duration_seconds + env_anim.dt * 5 # Add a small buffer

        print(f"最长轨道周期: {max_orbital_period_all_sats:.0f} 秒.")
        print(f"将以 dt={env_anim.dt}s 模拟 {num_simulation_steps_anim} 步 (总时长约 {simulation_duration_seconds/3600:.1f} 小时) 来观察约两周轨道。")
        
        plt_display.ion()
        initial_frame_anim = env_anim.render(mode='rgb_array_matplotlib')
        if initial_frame_anim is not None and hasattr(env_anim, 'mpl_fig') and env_anim.mpl_fig is not None:
            env_anim.mpl_fig.suptitle(f"时间: {env_anim.current_time:.0f}s / {simulation_duration_seconds:.0f}s", fontsize=10)
            plt_display.pause(1) 

        # 为所有我方智能体构造 NO_OP 动作
        actions_for_step_anim = {}
        for i in range(env_anim.num_friendly):
            agent_name = f"agent_{i}"
            actions_for_step_anim[agent_name] = {
                "action_type": 0, "target_idx": 0,
                "maneuver_target_pos": np.zeros(3, dtype=np.float32),
                "maneuver_target_orbit": np.zeros(6, dtype=np.float32),
                "maneuver_duration": np.array([env_anim.dt], dtype=np.float32),
                "formation_target_idx": 0,
                "formation_duration": np.array([env_anim.dt], dtype=np.float32)
            }

        for step_anim in range(num_simulation_steps_anim):
            obs_anim, rewards_anim, terminations_anim, truncations_anim, info_anim = env_anim.step(actions_for_step_anim)
            
            print(f"多卫星动画模拟步骤 {step_anim+1}/{num_simulation_steps_anim}, 当前时间: {env_anim.current_time:.0f}s")
            # 可以在这里打印每个卫星的nu角，如果需要
            # for sat_idx, sat_obj in enumerate(env_anim.satellites):
            #     if sat_obj.team == 'friendly' or sat_obj.team == 'enemy': # Print for all
            #         print(f"  Sat {sat_obj.id} nu: {np.rad2deg(sat_obj.orbit[5]):.1f}°")

            rgb_image_mpl_anim = env_anim.render(mode='rgb_array_matplotlib') 
            
            if rgb_image_mpl_anim is not None:
                if hasattr(env_anim, 'mpl_fig') and env_anim.mpl_fig is not None:
                    env_anim.mpl_fig.suptitle(f"时间: {env_anim.current_time:.0f}s / {simulation_duration_seconds:.0f}s", fontsize=10)
                plt_display.pause(0.01) 
            else:
                print("Matplotlib 动画渲染失败。")
                break
            
            # 检查是否所有智能体都已终止或截断
            all_terminated = all(terminations_anim.values())
            all_truncated = all(truncations_anim.values()) # Check if all agents are truncated
            
            if all_terminated or all_truncated:
                print(f"动画场景因终止({all_terminated})或截断({all_truncated})而结束。")
                break
            
            # 额外的循环中断条件 (如果step中的截断逻辑不够精确)
            if env_anim.current_time >= simulation_duration_seconds:
                print(f"已达到目标模拟时长 {simulation_duration_seconds:.0f}s。")
                # 确保截断标志被正确设置
                for i in range(env_anim.num_friendly):
                    truncations_anim[f"agent_{i}"] = True
                break


        print("多卫星轨道动画模拟完成。")
        plt_display.ioff() 
        plt_display.show() 
        
        env_anim.close()
        print("--- 新增测试用例 (Matplotlib 多卫星轨道动画) 执行完毕 ---")
    elif FLAG == 1:
        print("--- 开始执行新增测试用例 (Matplotlib GEO观测场景动画) ---")
    
        GEO_ALTITUDE = 35786.0  # km
        EARTH_RADIUS = 6371.0   # km
        GEO_SEMI_MAJOR_AXIS = EARTH_RADIUS + GEO_ALTITUDE # approx 42157 km

        friendly_agent_configs_geo_anim = [
            {'initial_orbit': [GEO_SEMI_MAJOR_AXIS - 2000, 0.001, np.deg2rad(1.0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)], 
            'function': 'observer'},
            {'initial_orbit': [GEO_SEMI_MAJOR_AXIS, 0.002, np.deg2rad(1.5), np.deg2rad(120), np.deg2rad(30), np.deg2rad(60)], 
            'function': 'observer'},
            {'initial_orbit': [GEO_SEMI_MAJOR_AXIS + 2000, 0.001, np.deg2rad(0.5), np.deg2rad(240), np.deg2rad(60), np.deg2rad(120)], 
            'function': 'observer'}
        ]
        enemy_agent_configs_geo_anim = [
            {'initial_orbit': [GEO_SEMI_MAJOR_AXIS, 0.001, np.deg2rad(0.1), 0, 0, 0], 
            'function': 'enemy_geo_target'}
        ]

        # 计算动画参数
        GEO_PERIOD_SECONDS = 2 * np.pi * np.sqrt(GEO_SEMI_MAJOR_AXIS**3 / MultiAgentSatelliteEnv(0,0).mu) # Use mu from a dummy env instance
        SIMULATION_TARGET_REAL_TIME_SECONDS = 60 # Animation to run for 1 minute
        ANIMATION_FPS = 20
        
        total_simulation_game_time_seconds = GEO_PERIOD_SECONDS * 2 # Enemy orbits twice
        num_simulation_steps_anim = int(SIMULATION_TARGET_REAL_TIME_SECONDS * ANIMATION_FPS)
        dt_seconds_for_anim = total_simulation_game_time_seconds / num_simulation_steps_anim
        
        print(f"GEO轨道周期: {GEO_PERIOD_SECONDS:.0f} 秒 (~{(GEO_PERIOD_SECONDS/3600):.1f} 小时)")
        print(f"模拟总博弈时长: {total_simulation_game_time_seconds:.0f} 秒 (~{(total_simulation_game_time_seconds/3600):.1f} 小时)")
        print(f"动画将运行约 {SIMULATION_TARGET_REAL_TIME_SECONDS} 秒 (真实时间)。")
        print(f"仿真步数: {num_simulation_steps_anim}, 每步博弈时长 (dt): {dt_seconds_for_anim:.2f} 秒")


        env_geo_anim = MultiAgentSatelliteEnv(num_friendly_satellites=len(friendly_agent_configs_geo_anim), 
                                    num_enemy_satellites=len(enemy_agent_configs_geo_anim),
                                    friendly_configs=friendly_agent_configs_geo_anim,
                                    enemy_initial_configs=enemy_agent_configs_geo_anim,
                                    dt_seconds=dt_seconds_for_anim,
                                    max_episode_time_seconds=total_simulation_game_time_seconds + dt_seconds_for_anim * 5) # Set max episode time
        
        obs_geo_anim, info_geo_anim = env_geo_anim.reset(options={
            'friendly_configs': friendly_agent_configs_geo_anim,
            'enemy_initial_configs': enemy_agent_configs_geo_anim
        })
        
        plt_display.ion()
        initial_frame_geo_anim = env_geo_anim.render(mode='rgb_array_matplotlib')
        if initial_frame_geo_anim is not None and hasattr(env_geo_anim, 'mpl_fig') and env_geo_anim.mpl_fig is not None:
            env_geo_anim.mpl_fig.suptitle(f"时间: {env_geo_anim.current_time:.0f}s / {total_simulation_game_time_seconds:.0f}s", fontsize=10)
            plt_display.pause(1) 

        actions_for_step_geo_anim = {}
        for i in range(env_geo_anim.num_friendly):
            agent_name = f"agent_{i}"
            actions_for_step_geo_anim[agent_name] = {
                "action_type": 0, "target_idx": 0, # NO_OP
                "maneuver_target_pos": np.zeros(3, dtype=np.float32),
                "maneuver_target_orbit": np.zeros(6, dtype=np.float32),
                "maneuver_duration": np.array([env_geo_anim.dt], dtype=np.float32),
                "formation_target_idx": 0,
                "formation_duration": np.array([env_geo_anim.dt], dtype=np.float32)
            }

        for step_geo_anim in range(num_simulation_steps_anim):
            obs_geo_anim, rewards_geo_anim, terminations_geo_anim, truncations_geo_anim, info_geo_anim = env_geo_anim.step(actions_for_step_geo_anim)
            
            print(f"GEO场景动画步骤 {step_geo_anim+1}/{num_simulation_steps_anim}, 当前博弈时间: {env_geo_anim.current_time:.0f}s")

            rgb_image_mpl_geo_anim = env_geo_anim.render(mode='rgb_array_matplotlib') 
            
            if rgb_image_mpl_geo_anim is not None:
                if hasattr(env_geo_anim, 'mpl_fig') and env_geo_anim.mpl_fig is not None:
                    env_geo_anim.mpl_fig.suptitle(f"时间: {env_geo_anim.current_time:.0f}s / {total_simulation_game_time_seconds:.0f}s", fontsize=10)
                plt_display.pause(1.0 / ANIMATION_FPS) # Pause for ANIMATION_FPS
            else:
                print("Matplotlib GEO场景动画渲染失败。")
                break
            
            all_terminated = all(terminations_geo_anim.values())
            all_truncated = all(truncations_geo_anim.values()) 
            
            if all_terminated or all_truncated:
                print(f"GEO场景动画因终止({all_terminated})或截断({all_truncated})而结束。")
                break
            
            # This secondary check might be redundant if max_episode_time handles truncation correctly
            if env_geo_anim.current_time >= total_simulation_game_time_seconds:
                print(f"已达到GEO场景目标模拟时长 {total_simulation_game_time_seconds:.0f}s。")
                break

        print("GEO场景轨道动画模拟完成。")
        plt_display.ioff() 
        plt_display.show() 
        
        env_geo_anim.close()
        print("--- 新增测试用例 (Matplotlib GEO场景动画) 执行完毕 ---")



    