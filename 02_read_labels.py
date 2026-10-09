# 02_read_labels.py
# Goal: load the label table (data.csv) into Python and look at it.

# Bring in the pandas tool. We rename it to "pd" by convention
import pandas as pd 

data_frame = pd.read_csv("data/data.csv", index_col=0)
print("shape:", data_frame.shape)
print("columns:", list(data_frame.columns))
print(data_frame.head())

