import os

from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from mimic.constants.ICD_Codes import sepsis_all_icd_codes
from mimic.orm_create.mimiciv_v3_orm import QwenEmbedding, Patients, Admissions, DiagnosesIcd, Labevents, Emar, \
	DLabitems, Note, Labels
from mimic.read.pre_processing.DB_History_Update import DB_Note_Connection
from mimic.read.pre_processing.LabelUpdater import LabelUpdater
from mimic.read.pre_processing.QwenEmbeddingsUpdater import QwenEmbeddingsUpdater


class DataRetriever:
	def __init__(self):
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def read_labels(self, limit):
		"""
		Returns all labels in the complete database
		:return: label information (dict)
		"""
		assert isinstance(limit, int), "limit must be an integer!"
		if limit == -1:
			label_information = self.session.query(Labels.row_id, Labels.label, Labels.hadm_id).all()
		else:
			label_information = self.session.query(Labels.row_id, Labels.label, Labels.hadm_id).limit(limit).all()
		return list(map(lambda d: {
			"row_id": d.row_id,
			"label": d.label,
			"hadm_id": d.hadm_id,
		}, label_information))

	def read_embeddings(self):
		embedding_data = self.session.query(QwenEmbedding.row_id, QwenEmbedding.embedding).limit(70_000).all()
		return list(map(lambda d: {
			"row_id": d.row_id,
			"embedding": d.embedding
		}, embedding_data))


if __name__ == '__main__':
	os.makedirs("./data", exist_ok=True)

	data_retriever = DataRetriever()

	LABEL_PATH = "./data/labels.csv"
	EMBEDDING_ROW_ID_PATH = "./data/embedding_row_ids.csv"
	EMBEDDING_PATH = "./data/embedding.npy"
	if os.path.exists(LABEL_PATH):
		label_df = pd.read_csv(LABEL_PATH)
	else:
		label_data = data_retriever.read_labels(-1)
		label_df = pd.DataFrame(label_data)
		label_df["label"] = label_df["label"].astype(int)
		label_df.to_csv(LABEL_PATH, index=False)

	if os.path.exists(EMBEDDING_PATH):
		embedding_df = pd.read_csv(EMBEDDING_ROW_ID_PATH)
	else:
		embedding_data = data_retriever.read_embeddings()
		embedding_df = pd.DataFrame(list(map(lambda x: x["row_id"], embedding_data)))
		embedding_df.to_csv(EMBEDDING_ROW_ID_PATH, index=False)
		embeddings = list(map(lambda x: x["embedding"], embedding_data))
		embeddings = np.array(embeddings)
		np.save(EMBEDDING_PATH, embeddings)

	print(label_df.shape)
	print(embedding_df.shape)
