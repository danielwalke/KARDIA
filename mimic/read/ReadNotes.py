import os
import pandas as pd

data = pd.read_csv("../mimic-iv-3.1/note/discharge.csv")
print(data.head())

print(data.shape)
patient_mask = data["subject_id"] == 13999829
print(data[patient_mask]["hadm_id"])
