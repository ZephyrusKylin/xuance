import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from xuance.environment import RawMultiAgentEnv
try:
    import rware
except ImportError:
    pass


class One_on_one_purchase():


































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