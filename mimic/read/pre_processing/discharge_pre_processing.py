import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import pandas as pd

from mimic.constants.ICD_Codes import sepsis_all_icd_codes
from mimic.orm_create.mimiciv_v3_orm import Discharge, Patients, Admissions, DiagnosesIcd, Labevents, Emar, DLabitems
from mimic.read.pre_processing.DB_History_Update import DB_Note_Connection


class DischargeInformationRetriever:
	def __init__(self):
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def read_discharge_information(self, limit):
		"""
		Returns all discharge notes in the complete database
		:return: discharge information (dict)
		"""
		assert isinstance(limit, int), "limit must be an integer!"
		if limit == -1:
			discharge_information = self.session.query(Discharge).order_by(Discharge.charttime).all()
		else:
			discharge_information = self.session.query(Discharge).order_by(Discharge.charttime).limit(limit).all()
		return list(map(lambda d: d.to_dict(), discharge_information))

	def read_discharge_subject_count(self):
		"""
		Counts all patients that have discharge information
		:return: patient_count (int)
		"""
		discharge_subject_counts = self.session.query(Discharge.subject_id, func.count(Discharge.subject_id)).group_by(Discharge.subject_id).order_by(func.count(Discharge.subject_id).desc()).all()
		return discharge_subject_counts

	def read_discharge_count(self):
		"""
		Count all discharge notes
		:return: discharge_count (int)
		"""
		discharge_counts = self.session.query(func.count(Discharge.row_id)).scalar()
		return discharge_counts


if __name__ == '__main__':
	discharge_information_retriever = DischargeInformationRetriever()
	discharge_information = discharge_information_retriever.read_discharge_information(limit=-1)
	discharge_texts = list(map(lambda r: r["text"], discharge_information))


	db_note_connection = DB_Note_Connection()
	db_note_connection.drop_table()
	db_note_connection.create_table_if_not_exists()
	def filter_discharge_text(text):
		filtered_chunks = []
		included_information = ["History of Present Illness:", "Past Medical History:", "Social History:",
								"Physical Exam:"]  ##TODO Discuss whether to integrate Physical Exam
		for chunk in text.split("\n\n"):
			if any(list(map(lambda title: title in chunk, included_information))):
				filtered_chunks.append(chunk)
		return "\n\n".join(filtered_chunks)
	filtered_discharge_texts = list(map(filter_discharge_text, discharge_texts))
	for i in range(len(filtered_discharge_texts)):
		note = {
            'row_id': discharge_information[i]["row_id"],
            'note_id': discharge_information[i]["note_id"],
            'subject_id': discharge_information[i]["subject_id"],
            'hadm_id': discharge_information[i]["hadm_id"],
            'charttime': discharge_information[i]["charttime"],
            'text': filtered_discharge_texts[i]
        }
		db_note_connection.add_new_note(note)
	# # print(discharge_information_retriever.read_discharge_subject_count())

	# print(discharge_information)
