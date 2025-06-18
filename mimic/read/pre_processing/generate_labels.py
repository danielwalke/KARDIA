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


class NoteInformationRetriever:
	def __init__(self):
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def read_notes(self, limit):
		"""
		Returns all discharge notes in the complete database
		:return: discharge information (dict)
		"""
		assert isinstance(limit, int), "limit must be an integer!"
		existing_row_ids = [] #self.get_labeled_row_ids()
		if limit == -1:

			note_information = self.session.query(Note.row_id, Note.hadm_id).where(Note.row_id.notin_(existing_row_ids)).all()
		else:
			note_information = self.session.query(Note.row_id, Note.hadm_id).where(Note.row_id.notin_(existing_row_ids)).limit(limit).all()
		return list(map(lambda d: {
			"row_id": d[0],
			"hadm_id": d[1],
		}, note_information))

	def read_diagnoses_information(self, hadm_ids):
		"""
		Returns all diagnoses for given gadm_ids in the complete database
		:param hadm_ids: list of hadmission ids
		:return: all diagnoses information (dict)
		"""
		diagnoses = self.session.query(DiagnosesIcd).where(DiagnosesIcd.hadm_id.in_(hadm_ids)).order_by(
			DiagnosesIcd.seq_num).all()
		return list(map(lambda d: d.to_dict(), diagnoses))

	def get_labeled_row_ids(self):
		return list(map(lambda x: x[0], self.session.query(Labels.row_id).all()))




if __name__ == '__main__':
	# label_updater = LabelUpdater()
	# label_updater.create_table_if_not_exists()

	note_information_retriever = NoteInformationRetriever()
	note_information = note_information_retriever.read_notes(-1)
	hadm_ids = list(map(lambda x: x["hadm_id"], note_information))
	diagnoses = note_information_retriever.read_diagnoses_information(list(set(hadm_ids)))
	labels = []

	print("Start iterating")
	for i, hadm_id in tqdm(enumerate(hadm_ids), desc="Processing hadmissions", total=len(hadm_ids)):
		filtered_diagnoses = list(filter(lambda x: x["hadm_id"] == hadm_id, diagnoses))
		icd_codes = set(map(lambda x: x["icd_code"], filtered_diagnoses))
		sepsis_intersection_codes = list(icd_codes.intersection(set(sepsis_all_icd_codes)))
		has_sepsis = len(sepsis_intersection_codes) > 0
		if has_sepsis:
			print(icd_codes)
			labels.append(1)
			print(filtered_diagnoses)
			break
		else:
			labels.append(0)
		# label_updater.add_new_label({
		# 	'row_id': note_information[i]["row_id"],
		# 	'hadm_id': note_information[i]["hadm_id"],
		# 	'label': has_sepsis,
		# 	'icd_codes': list(icd_codes)
		# })
	print(np.unique(np.array(labels), return_counts=True))