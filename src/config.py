from torch.cuda import is_available
from torch.functional import F


USE_GPU = True
DEVICE = "cuda" if USE_GPU and is_available() else "cpu"

TRAIN_VAL_SPLIT = 0.9
BATCH_SIZE = 64
EPOCHES = 20

MNIST_PATH="data/FashionMNIST/raw"


NOISE_DIM = 100

D_LOSS_FUNCTION = F.cross_entropy
G_LOSS_FUNCTION = F.cross_entropy

D_LEARNING_RATE  =0.00008
G_LEARNING_RATE =0.0004

BETA1 = 0.5
BETA2 = 0.999

EVAL_EVERY=2