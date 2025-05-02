# xuance/torch/utils/kb/kb_module.py

import yaml
from typing import Dict, Any
from .behavior_tree import BehaviorTree
from .blackboard import Blackboard

class KnowledgeModule:
    """
    加载并执行层次化知识库（Behavior Tree + Blackboard）。
    """

    def __init__(self, kb_yaml_path: str):
        # 读取 YAML 配置
        with open(kb_yaml_path, 'r') as f:
            kb_config = yaml.safe_load(f)
        # 初始化黑板
        self.blackboard = Blackboard()
        # 构建行为树
        self.bt = BehaviorTree.from_config(kb_config, self.blackboard)

    def query(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入原始观测字典，返回高层指令建议。
        e.g. {'move_dir': [dx, dy], 'attack': 0 or 1}
        """
        # 把观测写入黑板
        self.blackboard['obs'] = obs
        # 运行行为树一次 tick
        status = self.bt.tick()
        # 从黑板读出最终动作
        action = self.blackboard.get('action', {})
        return action
##TODO Action具体加权方式