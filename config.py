import warnings
warnings.filterwarnings("ignore")

CONTEXT_LEN = 64
PREDICTION_LEN = 10
NUM_FEATURES = 9
BATCH_SIZE = 64
MAX_EPOCHS = 50
HIDDEN_SIZE = 512
RNN_LAYERS = 4
LEARNING_RATE = 1e-3

MAX_PONDERS = 20
TIME_PENALTY = 0.01
ACT_HIDDEN_SIZE = 128

DATA_PATH = "data/Location1.csv"
DEEPAR_MODEL_PATH = "deepar_model.pt"
ACT_MODEL_PATH = "act_model.pt"

POPULATION_SIZE = 20
NUM_PARENTS_MATING = 10
MUTATION_PERCENT = 35
KEEP_ELITISM = 3
REPS_PER_MEASUREMENT = 10
WARMUP_REPS = 5
ENERGY_SAMPLE_INTERVAL = 0.001

CPU_TDP_WATTS = 65
CPU_IDLE_WATTS = 10

ALL_FEATURES = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
    'winddirection_100m', 'windgusts_10m'
]
