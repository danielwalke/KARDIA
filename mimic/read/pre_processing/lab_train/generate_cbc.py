import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mimic.meta.LabItemProp import LabItemProp
from mimic.orm_create.mimiciv_v3_orm import DLabitems, Labels
import time
import numpy as np

def execute_raw_sql(session, sql: str):
	"""
	Execute a raw SQL query using SQLAlchemy ORM session and return all results.

	Parameters:
	- session: SQLAlchemy ORM session
	- sql (str): Raw SQL query string

	Returns:
	- List of result rows as dictionaries
	"""
	result = session.execute(text(sql))
	return [dict(row._mapping) for row in result]


class CBCRetriever():
	def __init__(self):
		DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
		engine = create_engine(DB_URI)
		Session = sessionmaker(bind=engine)
		self.session = Session()

	def get_cbc_item_ids(self):
		lab_items_prop = LabItemProp(self.session, DLabitems.itemid)
		lab_item_ids = list(lab_items_prop.get_all_method_results().values())
		return lab_item_ids

	def get_hadm_ids_with_labels(self):
		hadm_ids = self.session.query(Labels.hadm_id).all()
		return list(map(lambda x: x.hadm_id, hadm_ids))

	def get_cbc_information(self):
		"""
		For now lets only keep first measurements + treat admissions as independent
		:return:
		"""
		hadm_ids = self.get_hadm_ids_with_labels()
		hadm_id_list = ', '.join(f"'{hid}'" for hid in hadm_ids)
		cbc_item_ids = self.get_cbc_item_ids()
		itemid_list = ', '.join(str(iid) for iid in cbc_item_ids)

		# Create raw SQL query
		sql = f"""
		        SELECT * FROM labevents
		        WHERE (hadm_id IN ({hadm_id_list})
		        AND itemid IN ({itemid_list}));
		    """
		lab_events = execute_raw_sql(self.session, sql)
		item_ids = list(map(lambda l: l["itemid"], lab_events))
		lab_hadm_ids = list(map(lambda l: l["hadm_id"], lab_events))
		# print(lab_hadm_ids[:100])
		# if 20122532 in lab_hadm_ids:
		# 	print("In lab hadm ids")
		values = list(map(lambda l: l["value"], lab_events))
		chat_times = list(map(lambda l: l["charttime"], lab_events))
		valuenum = list(map(lambda l: l["valuenum"], lab_events))
		lab_hadm_ids = np.expand_dims(np.array(lab_hadm_ids), axis=1)
		item_ids = np.expand_dims(np.array(item_ids), axis=1)
		values = np.expand_dims(values, axis=1)
		chat_times = np.expand_dims(chat_times, axis=1)
		valuenum = np.expand_dims(valuenum, axis=1)
		sepsis_data = np.concatenate([lab_hadm_ids, item_ids, values, chat_times, valuenum], axis=1)
		lab_data = pd.DataFrame(data=sepsis_data, columns=["HadmId", "ItemId", "Value", "Charttime", "valuenum"]).dropna()
		num_value_nan_mask = lab_data["valuenum"].isna()
		lab_data.loc[~num_value_nan_mask, "Value"] = lab_data["valuenum"][~num_value_nan_mask]
		neg_mask = lab_data["Value"] == "NEG"
		lab_data.loc[neg_mask, "Value"] = -1
		##TODO Need to combine with charttime again since sometimes measurement is performed previously but only for one -> So i wanna pick the first complete CBC measurement
		lab_data = lab_data[~lab_data["Value"].isin(
			["___", "HOLD.  DISCARD GREATER THAN 4 HOURS OLD.", "POS", "HOLD.  DISCARD GREATER THAN 24 HRS OLD.",
			 "NONE", "HOLD."])]
		lab_data["Value"] = lab_data["Value"].astype(np.float32)

		lab_data = lab_data.sort_values(by=["Charttime"])
		lab_data_pivot = lab_data.pivot_table(index="HadmId", columns="ItemId", values="Value",
											  aggfunc='first').reset_index()
		# lab_data_pivot[['hadm_id', 'Charttime']] = lab_data_pivot['idx'].str.split('__', n=1, expand=True)
		lab_data_pivot = lab_data_pivot #.dropna()
		lab_data_pivot["HadmId"] = lab_data_pivot["HadmId"].astype(int)
		lab_data_pivot.to_csv("cbc_data.csv", index=False)
		print(lab_data_pivot.shape)


		return lab_data_pivot


if __name__ == "__main__":
	cbc_retriever = CBCRetriever()
	res = cbc_retriever.get_cbc_information()
