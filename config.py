# 探索配置
EXPLORATION_NOISE = 0.1
MAX_ACTION = 300.0
MIN_ACTION = -300.0

# 多智能体特定配置
CENTRALIZED_TRAINING = True    # 中心化训练
DECENTRALIZED_EXECUTION = True # 去中心化执行
SHARE_PARAMETERS = False       # 是否共享参数
USE_GLOBAL_STATE = True        # 是否使用全局状态

# 实验配置
SEED = 42
EXPERIMENT_NAME = "sensor_network_masac"
LOG_DIR = "logs"
SAVE_MODEL = True
EVAL_FREQ = 10000             # 评估频率
SAVE_FREQ = 50000             # 保存频率
LOG_FREQ = 1000               # 日志频率

# 环境并行配置
NUM_ENVS = 8                  # 并行环境数量
