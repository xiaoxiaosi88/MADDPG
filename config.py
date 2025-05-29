import numpy as np

ANCHORS_POS = np.array([[-300, -300], [300, -300], [-300, 300],  [300, 300]], dtype=np.float32)  ## 锚节点位置
SENSORS_POS =  np.array([[0, 0], [150, 0], [-150, 0],  [0, 150],[0,-150]], dtype=np.float32) ## 非锚节点位置
ESTIMATED_POSITIONS = np.array([[0, 0], [20, 0], [-10, 0],  [90,80],[50,-150]], dtype=np.float32)
COMMUNICATION_RANGE = 250
INITIAL_POS_BOUNDS_IVA =np.array([[-300.0,300.0], [-300.0, 300.0]], dtype=np.float32)
NOISE_STD = float(2)
MAX_EPISODE_STEPS = 50
DIMENSION = 2
RENDER_MODE = None
HIDDEN_UNITS = 256
LEARNING_RATE = 1e-3
GAMMA = 0.99
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
# MASAC特定参数
TAU = 0.005  # 目标网络软更新系数
ALPHA = 0.2  # 熵正则化系数
AUTOTUNE = True  # 是否自动调整熵系数
POLICY_FREQUENCY = 2  # 策略网络更新频率
TARGET_NETWORK_FREQUENCY = 1  # 目标网络更新频率
LEARNING_STARTS = 1000  # 开始学习前的预热步数
TOTAL_TIMESTEPS = 2000000  # 总训练步数

# 日志和实验记录参数
SEED = 42 # 随机种子
EXPERIMENT_NAME = "sensor_network_masac"  # 实验名称
LOG_DIR = "logs"  # 日志目录
SAVE_MODEL = True  # 是否保存模型