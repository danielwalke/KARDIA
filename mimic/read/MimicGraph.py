import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import pandas as pd
from mimic.orm_create.mimiciv_v3_orm import Discharge, Patients, Admissions, DiagnosesIcd, Labevents, Emar, DLabitems
class MimicGraph:
	def __init__(self, subject_id):
		self.subject_id = subject_id
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def read_patient_information(self):
		patient_information = self.session.query(Patients).where(Patients.subject_id == self.subject_id).first()
		return patient_information.to_dict()

	def read_discharge_information(self):
		discharge_information = self.session.query(Discharge).where(Discharge.subject_id == self.subject_id).order_by(Discharge.charttime).all()
		return list(map(lambda d: d.to_dict(), discharge_information))

	def read_admission_information(self):
		admission_information = self.session.query(Admissions).where(Admissions.subject_id == self.subject_id).order_by(Admissions.admittime).all()
		return list(map(lambda d: d.to_dict(), admission_information))

	def read_diagnoses_information(self):
		diagnoses = self.session.query(DiagnosesIcd).where(DiagnosesIcd.subject_id == self.subject_id).order_by(DiagnosesIcd.seq_num).all()
		return list(map(lambda d: d.to_dict(), diagnoses))

	def read_labs_information(self):
		lab_information = self.session.query(Labevents).where(Labevents.subject_id == self.subject_id).order_by(Labevents.charttime).all()
		return list(map(lambda d: d.to_dict(), lab_information))

	def read_treatment_information(self):
		treatment_information = self.session.query(Emar).where(Emar.subject_id == self.subject_id).order_by(Emar.charttime).all()
		return list(map(lambda d: d.to_dict(), treatment_information))

	def read_d_labs_information(self, itemids):
		dlab_information = self.session.query(DLabitems).where(DLabitems.itemid.in_(itemids)).all()
		return list(map(lambda d: d.to_dict(), dlab_information))

	def read_most_conducted_lab_ids(self):
		lab_id_counts = self.session.query(Labevents.itemid, func.count(Labevents.itemid)).group_by(Labevents.itemid).order_by(func.count(Labevents.itemid).desc()).limit(100).all()
		return lab_id_counts

	def close_session(self):
		self.session.close()

if __name__ == '__main__':
	subject_id = 10000032


	mimic_graph = MimicGraph(subject_id)

	lab_item_counts = mimic_graph.read_most_conducted_lab_ids()
	abundant_lab_ids = list(map(lambda r: r[0], lab_item_counts))
	abundant_d_labs = mimic_graph.read_d_labs_information(abundant_lab_ids)
	print(abundant_d_labs)
	exit()
	patient_information = mimic_graph.read_patient_information()

	discharge_information = mimic_graph.read_discharge_information()


	print(discharge_information[0]["text"])
	admission_information = mimic_graph.read_admission_information()

	diagnoses_information = mimic_graph.read_diagnoses_information()

	labs_information = mimic_graph.read_labs_information()

	lab_df = pd.DataFrame(labs_information)
	print(lab_df["value"].value_counts())
	num_value_nan_mask = lab_df["valuenum"].isna()

	lab_df.loc[~num_value_nan_mask, "value"] = lab_df["valuenum"][~num_value_nan_mask]
	neg_mask = lab_df["value"] == "NEG"
	lab_df.loc[neg_mask, "value"] = -1
	lab_df = lab_df[~lab_df["value"].isin(["___", "HOLD.  DISCARD GREATER THAN 4 HOURS OLD.", "POS", "HOLD.  DISCARD GREATER THAN 24 HRS OLD.", "NONE", "HOLD."])] ## Filer out isnce I have no ideawhat these valus hould mean if they are not in valuenum like the others

	lab_df["value"] = lab_df["value"].astype(float)

	labitem_value_count = lab_df["itemid"].value_counts()
	labitems_with_min_freq = labitem_value_count[labitem_value_count > 10].index.tolist()
	lab_df = lab_df[lab_df["itemid"].isin(labitems_with_min_freq)]
	dlab_information = mimic_graph.read_d_labs_information(lab_df["itemid"].unique().tolist())
	dlab_information = {e["itemid"]: e["label" ]for e in dlab_information}
	print(dlab_information)

	lab_df["itemid"] = lab_df["itemid"].replace(dlab_information, regex=True)
	print(lab_df)
	pivoted_lab_information = pd.pivot_table(lab_df, columns = "itemid", index = "charttime", values="value", aggfunc="mean")
	print(pivoted_lab_information.columns)
	treatment_information = mimic_graph.read_treatment_information()

	print(list(map(lambda d: d["admit_time"].strftime("%d-%m-%Y-%H:%M"), admission_information)))
	pass