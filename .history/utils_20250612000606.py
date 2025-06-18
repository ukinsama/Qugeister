# utils.py


# -------------------------------------------------------------------------------
# セル 1: インポート
# -------------------------------------------------------------------------------
import os
import tqdm
import time
import random
from copy import deepcopy
import abc
import json # JSONモジュールをインポート

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pennylane as qml
from pennylane import numpy as pnp # PennyLaneのnumpy
from pennylane.optimize import NesterovMomentumOptimizer
# from qiskit_aer import AerSimulator # PennyLane-Qiskitを使う場合