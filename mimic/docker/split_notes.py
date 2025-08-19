import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

from tqdm import tqdm

labeled_notes = os.listdir("labeled_notes")

row_ids = []
labels = []
row_id_to_json = dict()

for note in tqdm(labeled_notes, desc="reading labeled notes"):
	with open(os.path.join("labeled_notes", note), "r") as f:
		data = json.load(f)
		row_ids.append(data["row_id"])
		labels.append(data["label"])
		row_id_to_json[data["row_id"]] = data
row_ids = np.array(row_ids)
labels = np.array(labels)

train_row_ids, test_row_ids = train_test_split(row_ids, test_size=0.05, stratify=labels)

os.makedirs("batched_notes", exist_ok=True)
current_file = 0
for note in tqdm(labeled_notes):
	row_id = int(note.split(".")[0])

	if row_id not in test_row_ids: continue
	current_file+=1
	with open(os.path.join("batched_notes", f"{current_file}.json"), "w") as f:
		f.write(json.dumps(row_id_to_json[row_id], indent=4))
