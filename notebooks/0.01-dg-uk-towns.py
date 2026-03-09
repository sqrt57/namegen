# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")

# %%
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('../data/raw/uk-towns.txt', skipinitialspace=True)

# %%
df

# %%
df['Town'].drop_duplicates().sort_values()

# %%
df['County'].drop_duplicates().sort_values()

# %%
df['Town'].str.lower().drop_duplicates().sort_values()

# %%
Counter(''.join(df['Town'].str.lower().drop_duplicates().sort_values()))

# %%
Counter(''.join(df['County'].str.lower().drop_duplicates().sort_values()))
