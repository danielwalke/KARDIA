import json
import os
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from tqdm import tqdm
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

kg_dirs = os.listdir("./revised_kgs")

for dir in tqdm(kg_dirs):
	graph = json.load(open(os.path.join("./revised_kgs", dir, "graph.json"), "r"))
	node_emb = model.encode(graph["entities"])
	edges = model.encode(graph["edges"])
	np.save(os.path.join("./revised_kgs", dir, "node_emb.npy"), node_emb)
	np.save(os.path.join("./revised_kgs", dir, "edge_emb.npy"), edges)