import json
import os
import shutil

out_dir = "20250812_qwen_1_7b_kgs_out"

batched_notes = os.listdir("batched_notes")
row_id_to_batched_id = dict()
for file in batched_notes:
    with open(os.path.join("batched_notes", file), "r") as f:
        note = json.load(f)
        batched_id = file.split(".")[0]
        row_id_to_batched_id[note["row_id"]] = batched_id

folders = os.listdir("20250807_qwen_1_7b_kgs_out")
os.makedirs(out_dir, exist_ok=True)
for folder in folders:
    is_file = "." in folder
    if is_file: continue
    folder = int(folder)
    os.makedirs(f"{out_dir}/{row_id_to_batched_id[folder]}", exist_ok=True)
    shutil.copyfile(f'20250807_qwen_1_7b_kgs_out/{folder}/graph.json', f'{out_dir}/{row_id_to_batched_id[folder]}/graph.json')
