import sys
sys.path.append("../")
import pandas as pd
from utils import ca as cu, behave as bu, plot as pu, db as db, tune as tu, stats as su
import numpy as np
from paths.config import M2PConfig
import matplotlib.pyplot as plt

cfg = M2PConfig()
df_exps, df_roi, df_ca, df_behave, df_behave_ca = db.get_ca_behave_data(cfg)