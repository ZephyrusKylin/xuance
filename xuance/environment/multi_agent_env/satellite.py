import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from xuance.environment import RawMultiAgentEnv
try:
    import rware
except ImportError:
    pass


class One_on_one_purchase():
    __annotations__ = {
        "Pursuer_position": np.ndarray,
        "Pursuer_vector": np.ndarray,
        "Escaper_position": np.ndarray,
        "Escaper_vector": np.ndarray
    }
    def __init__(self, Pursuer_position=np.array([2000, 2000, 1000]), Pursuer_vector=np.array([1.71, 1.14, 1.3]),
                 Escaper_position=np.array([1000, 2000, 0]), Escaper_vector=np.array([1.71, 1.14, 1.3]),
                 M=0.4, dis_safe=1000, d_capture=100000, Flag=0, fuel_c=320, fuel_t=320, d_range=100000, args=None):

        self.Pursuer_position = Pursuer_position
        self.Pursuer_vector = Pursuer_vector
        self.Escaper_position = Escaper_position
        self.Escaper_vector = Escaper_vector
        self.dis_dafe = dis_safe            # 碰撞距离
        self.d_capture = d_capture          # 抓捕距离
        self.Flag = Flag                    # 训练追捕航天器(0)、逃逸航天器(1)或者测试的标志(2)
        self.M = M                          # 航天器质量
        self.d_capture = d_capture
        self.burn_reward = 0
        self.win_reward = 100
        self.dangerous_zone = 0             # 危险区数量
        self.fuel_c = fuel_c                # 抓捕航天器燃料情况
        self.fuel_t = fuel_t                # 抓捕航天器燃料情况
        self.dis = np.inf                   # 博弈距离
        self.d_range = d_range              # 使用可达域博弈的最小距离
        self.max_episode_steps = args.max_episode_steps
        self.ellipse_params = []            # 椭圆拟合参数
        # 椭圆拟合训练网络
        self.trian_elliptical_fitting = real_time_data_process.network_method_train(pretrain=False)
        # 下面声明其基本属性
        position_low = np.array([-500000, -500000, -500000, -10000000, -10000000, -10000000,-10000000, -10000000, -10000000])
        position_high = np.array([500000, 500000, 500000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000])
        velocity_low = np.array([-10000, -10000, -10000, -50000, -50000, -50000, -50000, -50000, -50000])
        velocity_high = np.array([10000, 10000, 10000, 50000, 50000, 50000, 50000, 50000, 50000])
        observation_low = np.concatenate((position_low, velocity_low))
        observation_high = np.concatenate((position_high, velocity_high))
        self.observation_space = gym.spaces.Box(low=observation_low, high=observation_high, shape=(18,), dtype=np.float32)
        self.action_space = np.array([[-1.6, 1.6],
                                      [-1.6, 1.6],
                                      [-1.6, 1.6]])
        n_actions = 5  # 策略的数量
        self.action_space_beta = gym.spaces.Discrete(n_actions)    # {0,1,2,3,4}



    def reset(self, Flag):
        self.Pursuer_position = np.array([200000, 0 ,0])
        self.Pursuer_vector = np.array([0, 0, 0])
        self.pursuer_reward = 0.0
        self.Escaper_position = np.array([18000, 0, 0])
        self.Escaper_vector = np.array([0, 0, 0])
        self.escaper_reward = 0.0

        self.Flag = Flag

        s = np.array([self.Pursuer_position - self.Escaper_position, self.Pursuer_vector - self.Escaper_vector,
                      self.Pursuer_position, self.Pursuer_vector, self.Escaper_position, self.Escaper_vector]).ravel()

        return s

    def render(self):





























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