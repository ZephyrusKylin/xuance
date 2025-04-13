# xuance/torch/utils/kb/behavior_tree.py

from typing import List, Dict, Any
from .blackboard import Blackboard

class NodeStatus:
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class BaseNode:
    def __init__(self, name: str, blackboard: Blackboard):
        self.name = name
        self.bb = blackboard

    def tick(self) -> int:
        raise NotImplementedError

    @staticmethod
    def from_config(cfg: Dict[str, Any], bb: Blackboard):
        ntype = cfg.get('type')
        name = cfg.get('name', cfg.get('id', ''))
        if ntype == 'sequence':
            children = [BaseNode.from_config(c, bb) for c in cfg['children']]
            return Sequence(name, children, bb)
        if ntype == 'selector':
            children = [BaseNode.from_config(c, bb) for c in cfg['children']]
            return Selector(name, children, bb)
        if ntype == 'condition':
            return ConditionNode(name, cfg['condition'], bb)
        if ntype == 'action':
            return ActionNode(name, cfg['action'], bb)
        raise ValueError(f"Unknown node type: {ntype}")

class Sequence(BaseNode):
    def __init__(self, name: str, children: List[BaseNode], bb: Blackboard):
        super().__init__(name, bb)
        self.children = children

    def tick(self) -> int:
        for child in self.children:
            status = child.tick()
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class Selector(BaseNode):
    def __init__(self, name: str, children: List[BaseNode], bb: Blackboard):
        super().__init__(name, bb)
        self.children = children

    def tick(self) -> int:
        for child in self.children:
            status = child.tick()
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class ConditionNode(BaseNode):
    def __init__(self, name: str, cond_cfg: Dict[str, Any], bb: Blackboard):
        super().__init__(name, bb)
        self.cond_cfg = cond_cfg

    def tick(self) -> int:
        obs = self.bb['obs']
        ctype = self.cond_cfg['type']
        # 示例条件：any_enemy_within
        if ctype == 'any_enemy_within':
            radius = self.cond_cfg['radius_km']
            my_pos = obs['self_pos']
            for e in obs['enemy_positions']:
                if (e - my_pos).norm() <= radius:
                    self.bb['target'] = e
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE
        # 可扩展更多条件
        return NodeStatus.FAILURE

class ActionNode(BaseNode):
    def __init__(self, name: str, action_cfg: Dict[str, Any], bb: Blackboard):
        super().__init__(name, bb)
        self.action_cfg = action_cfg

    def tick(self) -> int:
        # 根据配置生成动作写入黑板
        if self.action_cfg['type'] == 'move_towards_target':
            target = self.bb.get('target')
            if target is None:
                return NodeStatus.FAILURE
            my_pos = self.bb['obs']['self_pos']
            dir_vec = (target - my_pos)
            dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
            self.bb['action'] = {'move_dir': [dir_vec.x, dir_vec.y], 'attack': 0}
            return NodeStatus.SUCCESS
        if self.action_cfg['type'] == 'attack_target':
            if 'target' not in self.bb:
                return NodeStatus.FAILURE
            self.bb['action'] = {'move_dir': [0.0, 0.0], 'attack': 1}
            return NodeStatus.SUCCESS
        # 可扩展更多动作
        return NodeStatus.FAILURE

class BehaviorTree:
    def __init__(self, root: BaseNode, bb: Blackboard):
        self.root = root
        self.bb = bb

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], bb: Blackboard):
        root_cfg = cfg['strategies'][0]  # 例如首个策略为根
        root = BaseNode.from_config(root_cfg, bb)
        return cls(root, bb)

    def tick(self) -> int:
        return self.root.tick()
