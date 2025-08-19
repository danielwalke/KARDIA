import os
import json
import numpy as np

notes = os.listdir("batched_notes")
labels = []
for note in notes:
	with open("batched_notes/"+note,"r") as f:
		data = json.load(f)
		labels.append(data["label"])
print(np.unique(np.array(labels), return_counts=True))
