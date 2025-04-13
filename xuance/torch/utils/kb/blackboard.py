# xuance/torch/utils/kb/blackboard.py

class Blackboard(dict):
    """
    共享内存，用于节点间读写中间状态和动作。
    继承自 dict，简化使用。
    """
    def __init__(self):
        super().__init__()
        # 预设一些键
        self['obs'] = None
        self['target'] = None
        self['action'] = None
