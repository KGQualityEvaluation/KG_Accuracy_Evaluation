import pickle

TRUE_LABEL_DICT_dir = '../SYN-IG/new-SYN10wLabelDict.pickle'
TRUE_NOTE_DICT_dir = '../SYN-IG/new-SYN10wNoteDict.pickle'
SAMPLES_dir = '../SYN-IG/new-SYN10w.pickle'

#INITIAL_STRATIFICATION = [(0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]

INITIAL_STRATIFICATION = [(0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]

#INITIAL_STRATIFICATION = [(0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]

#INITIAL_STRATIFICATION = [(0.90, 0.95), (0.95, 0.96), (0.96, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 0.995), (0.995, 1.01)]

# 20层
#INITIAL_STRATIFICATION = [(0.90, 0.95), (0.95, 0.955), (0.955, 0.96), (0.96, 0.965), (0.965, 0.97), (0.97, 0.975), (0.975, 0.98), (0.98, 0.985), (0.985, 0.99), (0.99, 0.991), (0.991, 0.992), (0.992, 0.993), (0.993, 0.994), (0.994, 0.995), (0.995, 0.996), (0.996, 0.997), (0.997, 0.998), (0.998, 0.999), (0.999, 1), (1, 1.01)]

P1 = 0.5
P2 = 0.5

#TRUE_LABEL_DICT = {}
#with open('../NELL/label_dict.pickle', 'rb') as f:
with open(TRUE_LABEL_DICT_dir, 'rb') as f:
    TRUE_LABEL_DICT = pickle.load(f)

with open(TRUE_NOTE_DICT_dir, 'rb') as f:
    TRUE_NOTE_DICT = pickle.load(f)

with open(SAMPLES_dir, 'rb') as f:
    samples = pickle.load(f)

A1, A2, A3, A4 = 5.7, 11, 10, 1.2
EPSILON1, EPSILON2 = 0.05, 0.05
ALPHA1, ALPHA2 = 0.05, 0.05

# MCTS终止条件类型
THRE_TYPE = 'time'
# MCTS总时间上限
TIME_THRE = 1
# MCTS总模拟次数上限
SIMU_THRE = 2000

# UCB中的C
C = 6