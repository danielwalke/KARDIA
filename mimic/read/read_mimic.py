import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from mimic.constants.ICD_Codes import sepsis_all_icd_codes
from mimic.meta.LabItemProp import LabItemProp
from mimic.orm_create.mimiciv_v3_orm import Base, Admissions, DLabitems, DiagnosesIcd, Pharmacy, \
    Patients, Emar  # This is the class auto-generated from admissions.csv
from sqlalchemy import and_, or_
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)

DB_URI = "postgresql://postgres:password@localhost:5432/MimicIV"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()


def get_sepsis_hadm(session):
    diagnoses_hadm = session.query(DiagnosesIcd).where(
        DiagnosesIcd.icd_code.in_(sepsis_all_icd_codes)).all()
    return diagnoses_hadm


def get_subject_ids(session, diagnoses_hadm_ids):
    sepsis_hadm = session.query(Admissions).where(Admissions.hadm_id.in_(diagnoses_hadm_ids)).distinct().all()
    sepsis_subj_ids = list(map(lambda res: res.subject_id, sepsis_hadm))
    return sepsis_subj_ids


def get_sepsis_patients(session, sepsis_subj_ids):
    patients = session.query(Patients).filter(and_(Patients.subject_id.in_(sepsis_subj_ids), Patients.anchor_age >= 18)).all()
    return patients

def get_emar(session, sepsis_hadm_ids):
    ## TODO I need to use emar and emar detail instead of pharmacy and filtere based o charttime and event_txt == Administered
    sepsis_pharmacy = session.query(Emar).filter(and_(Emar.hadm_id.in_(sepsis_hadm_ids), Emar.event_txt == 'Administered')).all()
    return sepsis_pharmacy


def get_lab_events(session, diagnoses_hadm_ids, lab_item_ids):

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

    hadm_id_list = ', '.join(f"'{hid}'" for hid in diagnoses_hadm_ids)
    itemid_list = ', '.join(str(iid) for iid in lab_item_ids)

    # Create raw SQL query
    sql = f"""
            SELECT * FROM labevents
            WHERE hadm_id IN ({hadm_id_list})
            AND itemid IN ({itemid_list});
        """

    results = execute_raw_sql(session, sql)
    return results



sepsis_hadm = get_sepsis_hadm(session)

adult_patients = get_sepsis_patients(session, list(map(lambda x: x.subject_id, sepsis_hadm)))
adult_patients_subj_ids  = list(map(lambda x: x.subject_id, adult_patients))
sepsis_hadm = list(filter(lambda x: x.subject_id in adult_patients_subj_ids, sepsis_hadm))
sepsis_hadm_ids = list(map(lambda x: x.hadm_id, sepsis_hadm))

sepsis_pharmacy = get_emar(session,list(map(str, sepsis_hadm_ids)))
meds = np.expand_dims(list(map(lambda x: x.medication, sepsis_pharmacy)), axis=1)
starttimes = np.expand_dims(list(map(lambda x: x.charttime, sepsis_pharmacy)), axis = 1)
hadm_ids = np.expand_dims(list(map(lambda x: x.hadm_id, sepsis_pharmacy)), axis=1)

pharmacy_df = pd.DataFrame(data=np.concatenate((meds, starttimes, hadm_ids), axis = 1), columns=['meds', 'starttime', 'hadm_id'])
print(pharmacy_df)

## TODO: Group medications based on name into similar ones -> Antibiotics, anticoagulants etc
# print(",".join(list(filter(lambda x: x is not None, list(set(meds.squeeze().tolist()))))))

## TODO refactor
sepsis_hadm_ids = list(map(str, sepsis_hadm_ids))
lab_items_prop = LabItemProp(session, DLabitems.itemid)
lab_item_ids = list(lab_items_prop.get_all_method_results().values())
lab_events = get_lab_events(session, sepsis_hadm_ids, lab_item_ids)

item_ids = list(map(lambda l: l["itemid"], lab_events))
lab_hadm_ids = list(map(lambda l: l["hadm_id"], lab_events))
values = list(map(lambda l: l["value"], lab_events))
chat_times = list(map(lambda l: l["charttime"], lab_events))
lab_hadm_ids = np.expand_dims(np.array(lab_hadm_ids), axis=1)
item_ids = np.expand_dims(np.array(item_ids), axis=1)
values = np.expand_dims(values, axis=1)
chat_times = np.expand_dims(chat_times, axis=1)
sepsis_data = np.concatenate([lab_hadm_ids, item_ids, values, chat_times], axis = 1)
lab_data = pd.DataFrame(data = sepsis_data, columns = ["HadmId", "ItemId", "Value", "Charttime"]).dropna()
lab_data["idx"] = lab_data["HadmId"].astype(str) + "__" + lab_data["Charttime"].astype(str)
lab_data["Value"] = lab_data["Value"].str.replace("<", "").astype(np.float32)

lab_data_pivot = lab_data.pivot_table(index = "idx", columns = "ItemId", values = "Value", aggfunc='mean').reset_index()
lab_data_pivot[['hadm_id', 'Charttime']] = lab_data_pivot['idx'].str.split('__', n=1, expand=True)
lab_data_pivot = lab_data_pivot.dropna()

print(lab_data_pivot)
lab_data_pivot["hadm_id"] = lab_data_pivot["hadm_id"].astype(np.float32)
pharmacy_df["hadm_id"] = pharmacy_df["hadm_id"].astype(np.float32)
merged_pharmacy_lab_df = pd.merge(pharmacy_df, lab_data_pivot, on = "hadm_id", how = "inner")
merged_pharmacy_lab_df["Charttime"] = pd.to_datetime(merged_pharmacy_lab_df["Charttime"])
merged_pharmacy_lab_df["starttime"] = pd.to_datetime(merged_pharmacy_lab_df["starttime"])

time_window = timedelta(hours=1)
medication_time_mask = (merged_pharmacy_lab_df["starttime"] >= merged_pharmacy_lab_df["Charttime"]) & (merged_pharmacy_lab_df["starttime"] <= merged_pharmacy_lab_df["Charttime"] + time_window)
merged_pharmacy_lab_df = merged_pharmacy_lab_df[medication_time_mask]
print(merged_pharmacy_lab_df["meds"].value_counts())
print(pd.DataFrame(merged_pharmacy_lab_df["meds"].value_counts()))
print(merged_pharmacy_lab_df.shape)
print(merged_pharmacy_lab_df.duplicated(subset = lab_data_pivot.columns).sum())
# le = LabelEncoder()
# labels = le.fit_transform(merged_pharmacy_lab_df["meds"]).astype(np.int8)
# X = merged_pharmacy_lab_df[lab_item_ids].values.astype(np.float32)
#
# X_train, X_test, y_train, y_test = train_test_split(X, labels)
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
# model = RandomForestClassifier(class_weight = 'balanced', n_estimators = 300, n_jobs = 3)
# model.fit(X_train_std, y_train)
# y_pred = model.predict(X_test_std)
# acc = accuracy_score(y_test, y_pred)
# acc_train = accuracy_score(y_train, model.predict(X_train_std))
# print(acc)
# print(acc_train)



ages = list(map(lambda p: p.anchor_age, adult_patients))
genders = list(map(lambda p: p.gender, adult_patients))
dods = list(map(lambda p: p.dod, adult_patients))


import matplotlib.pyplot as plt

plt.violinplot(ages)
plt.show()
session.close()
"""
Ordered lab event count:
SELECT 
    COUNT(labevents.hadm_id) AS labcounts, 
    labevents.itemid, 
    d_labitems.label
FROM labevents
LEFT JOIN d_labitems ON labevents.itemid = d_labitems.itemid
GROUP BY labevents.itemid, d_labitems.label
ORDER BY labcounts DESC;

"""


#
# def query_thrombosis_information(session):
#     # Query all hadm_id values
#     potential_diagnoses = session.query(DIcdDiagnoses).all()
#     potentially_relevant_icd_codes = []
#     for row in potential_diagnoses:
#         # if row.long_title in ["Portal vein thrombosis", "Arterial embolism and thrombosis"]:
#         #     potentially_relevant_icd_codes.append(row.icd_code)
#         if "thrombosis" in row.long_title and not "due to thrombosis" in row.long_title:
#             potentially_relevant_icd_codes.append(row.icd_code)
#             continue
#         # print(row.icd_code)
#         # print(row.icd_version)
#         # print(row.long_title)
#
#     print(potentially_relevant_icd_codes)
#     diagnoses_hadm_ids = session.query(DiagnosesIcd.hadm_id).where(
#         DiagnosesIcd.icd_code.in_(potentially_relevant_icd_codes)).all()
#     diagnoses_hadm_ids = list(map(lambda row: row.hadm_id, diagnoses_hadm_ids))
#
#     admissions = session.query(Admissions).where(Admissions.hadm_id.in_(diagnoses_hadm_ids)).all()
#     patient_ids = list(map(lambda row: row.subject_id, admissions))
#
#     patients = session.query(Patients).where(Patients.subject_id.in_(list(set(patient_ids)))).all()
#     print(len(patients))
#     # Close the session
#     session.close()
