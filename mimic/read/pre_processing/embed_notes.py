from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from mimic.constants.ICD_Codes import sepsis_all_icd_codes
from mimic.orm_create.mimiciv_v3_orm import QwenEmbedding, Patients, Admissions, DiagnosesIcd, Labevents, Emar, DLabitems, Note
from mimic.read.pre_processing.DB_History_Update import DB_Note_Connection
from mimic.read.pre_processing.QwenEmbeddingsUpdater import QwenEmbeddingsUpdater


class NoteInformationRetriever:
	def __init__(self):
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def read_embedding_row_ids(self):
		embeddings = self.session.query(QwenEmbedding.row_id).all()
		return list(map(lambda x: x[0], embeddings))

	def read_notes(self, limit):
		"""
		Returns all discharge note histories in the complete database
		:return: discharge information (dict)
		"""
		assert isinstance(limit, int), "limit must be an integer!"
		if limit == -1:
			note_information = self.session.query(Note).order_by(Note.charttime).all()
		else:
			note_information = self.session.query(Note).order_by(Note.charttime).limit(limit).all()
		return list(map(lambda d: d.to_dict(), note_information))

	def read_note_count(self):
		"""
		Count all notes
		:return: notes_count (int)
		"""
		notes_count = self.session.query(func.count(Note.row_id)).scalar()
		return notes_count


if __name__ == '__main__':
	note_information_retriever = NoteInformationRetriever()

	note_information = note_information_retriever.read_notes(-1)
	existing_embedding_row_ids = note_information_retriever.read_embedding_row_ids() #list(map(lambda x: x["row_id"], note_information_retriever.read_embeddings()))
	qwenEmbeddingsUpdater = QwenEmbeddingsUpdater()
	qwenEmbeddingsUpdater.create_table_if_not_exists()

	embedder = OllamaEmbeddings(model="qwen3:32b")
	zero_hist_count = 0
	for i in tqdm(range(len(note_information))):
		if note_information[i]["row_id"] in existing_embedding_row_ids: continue
		if len(note_information[i]["text"]) == 0:
			zero_hist_count += 1
			continue
		embedded_text = embedder.embed_query(note_information[i]["text"])
		qwenEmbeddingsUpdater.add_new_embedding({
			'row_id': note_information[i]["row_id"],
			'note_id': note_information[i]["note_id"],
			'subject_id': note_information[i]["subject_id"],
			'hadm_id': note_information[i]["hadm_id"],
			'charttime': note_information[i]["charttime"],
			'embedding': embedded_text
		})
	qwenEmbeddingsUpdater.close_connection()
	print(zero_hist_count)