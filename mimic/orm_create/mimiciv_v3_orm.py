from sqlalchemy import Column, INTEGER, TEXT, TIMESTAMP, VARCHAR, ForeignKey, NUMERIC, BIGINT, DATE, ARRAY, REAL, \
    BOOLEAN
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Emar(Base):
    __tablename__ = 'emar'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    emar_id = Column(TEXT(), primary_key=False)
    emar_seq = Column(INTEGER(), primary_key=False)
    poe_id = Column(TEXT(), primary_key=False)
    pharmacy_id = Column(INTEGER(), primary_key=False)
    enter_provider_id = Column(TEXT(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    medication = Column(TEXT(), primary_key=False)
    event_txt = Column(TEXT(), primary_key=False)
    scheduletime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "emar_id": self.emar_id,
            "emar_seq": self.emar_seq,
            "medication": self.medication,
            "charttime": self.charttime
        }

class DLabitems(Base):
    __tablename__ = 'd_labitems'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    itemid = Column(INTEGER(), primary_key=False)
    label = Column(TEXT(), primary_key=False)
    fluid = Column(TEXT(), primary_key=False)
    category = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "itemid": self.itemid,
            "label": self.label,
        }


class DIcdProcedures(Base):
    __tablename__ = 'd_icd_procedures'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    icd_code = Column(TEXT(), primary_key=False)
    icd_version = Column(INTEGER(), primary_key=False)
    long_title = Column(TEXT(), primary_key=False)

class DIcdDiagnoses(Base):
    __tablename__ = 'd_icd_diagnoses'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    icd_code = Column(TEXT(), primary_key=False)
    icd_version = Column(INTEGER(), primary_key=False)
    long_title = Column(TEXT(), primary_key=False)

class DHcpcs(Base):
    __tablename__ = 'd_hcpcs'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    code = Column(TEXT(), primary_key=False)
    category = Column(TEXT(), primary_key=False)
    long_description = Column(TEXT(), primary_key=False)
    short_description = Column(TEXT(), primary_key=False)

class Drgcodes(Base):
    __tablename__ = 'drgcodes'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    drg_type = Column(TEXT(), primary_key=False)
    drg_code = Column(INTEGER(), primary_key=False)
    description = Column(TEXT(), primary_key=False)
    drg_severity = Column(INTEGER(), primary_key=False)
    drg_mortality = Column(INTEGER(), primary_key=False)

class DiagnosesIcd(Base):
    __tablename__ = 'diagnoses_icd'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    seq_num = Column(INTEGER(), primary_key=False)
    icd_code = Column(TEXT(), primary_key=False)
    icd_version = Column(INTEGER(), primary_key=False)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "seq_num": self.seq_num,
            "icd_code": self.icd_code,
        }

class Admissions(Base):
    __tablename__ = 'admissions'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    admittime = Column(TIMESTAMP(), primary_key=False)
    dischtime = Column(TIMESTAMP(), primary_key=False)
    deathtime = Column(TIMESTAMP(), primary_key=False)
    admission_type = Column(TEXT(), primary_key=False)
    admit_provider_id = Column(TEXT(), primary_key=False)
    admission_location = Column(TEXT(), primary_key=False)
    discharge_location = Column(TEXT(), primary_key=False)
    insurance = Column(TEXT(), primary_key=False)
    language = Column(TEXT(), primary_key=False)
    marital_status = Column(TEXT(), primary_key=False)
    race = Column(TEXT(), primary_key=False)
    edregtime = Column(TIMESTAMP(), primary_key=False)
    edouttime = Column(TIMESTAMP(), primary_key=False)
    hospital_expire_flag = Column(INTEGER(), primary_key=False)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "admit_time":  self.admittime,
            "disch_time": self.dischtime,
        }


class DischargeDetail(Base):
    __tablename__ = 'discharge_detail'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    field_name = Column(TEXT(), primary_key=False)
    field_value = Column(TEXT(), primary_key=False)
    field_ordinal = Column(INTEGER(), primary_key=False)

class Discharge(Base):
    __tablename__ = 'discharge'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    note_type = Column(TEXT(), primary_key=False)
    note_seq = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    text = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "row_id": self.row_id,
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "note_type": self.note_type,
            "note_seq": self.note_seq,
            "text": self.text,
            "charttime": self.charttime,
        }

class Note(Base):
    __tablename__ = 'notes'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    text = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "row_id": self.row_id,
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "text": self.text,
            "charttime": self.charttime,
        }

class RevisedNote(Base):
    __tablename__ = 'revised_note'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    text = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "row_id": self.row_id,
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "text": self.text,
            "charttime": self.charttime,
        }

class PreprocessedRevisedNote(Base):
    __tablename__ = 'preprocessed_revised_note'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    text = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "row_id": self.row_id,
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "text": self.text,
            "charttime": self.charttime,
        }

class QwenEmbedding(Base):
    __tablename__ = 'qwen_embeddings'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    embedding = Column(ARRAY(REAL), primary_key=False)

    def to_dict(self):
        return {
            "row_id": self.row_id,
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "embedding": self.embedding,
            "charttime": self.charttime,
        }

class Labels(Base):
    __tablename__ = 'labels'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    label = Column(BOOLEAN(), primary_key=False)
    icd_codes = Column(ARRAY(TEXT), primary_key=False)


    def to_dict(self):
        return {
            "row_id": self.row_id,
            "hadm_id": self.hadm_id,
            "label": self.label,
            "icd_codes": self.icd_codes,
        }

class Radiology(Base):
    __tablename__ = 'radiology'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    note_type = Column(TEXT(), primary_key=False)
    note_seq = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    text = Column(TEXT(), primary_key=False)

class RadiologyDetail(Base):
    __tablename__ = 'radiology_detail'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    note_id = Column(TEXT(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    field_name = Column(TEXT(), primary_key=False)
    field_value = Column(TEXT(), primary_key=False)
    field_ordinal = Column(INTEGER(), primary_key=False)

class Caregiver(Base):
    __tablename__ = 'caregiver'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    caregiver_id = Column(INTEGER(), primary_key=False)

class Chartevents(Base):
    __tablename__ = 'chartevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    value = Column(TEXT(), primary_key=False)
    valuenum = Column(NUMERIC(), primary_key=False)
    valueuom = Column(TEXT(), primary_key=False)
    warning = Column(INTEGER(), primary_key=False)

class Datetimeevents(Base):
    __tablename__ = 'datetimeevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    value = Column(TEXT(), primary_key=False)
    valueuom = Column(TEXT(), primary_key=False)
    warning = Column(INTEGER(), primary_key=False)

class DItems(Base):
    __tablename__ = 'd_items'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    itemid = Column(INTEGER(), primary_key=False)
    label = Column(TEXT(), primary_key=False)
    abbreviation = Column(TEXT(), primary_key=False)
    linksto = Column(TEXT(), primary_key=False)
    category = Column(TEXT(), primary_key=False)
    unitname = Column(TEXT(), primary_key=False)
    param_type = Column(TEXT(), primary_key=False)
    lownormalvalue = Column(TEXT(), primary_key=False)
    highnormalvalue = Column(TEXT(), primary_key=False)

class Icustays(Base):
    __tablename__ = 'icustays'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    first_careunit = Column(TEXT(), primary_key=False)
    last_careunit = Column(TEXT(), primary_key=False)
    intime = Column(TIMESTAMP(), primary_key=False)
    outtime = Column(TIMESTAMP(), primary_key=False)
    los = Column(NUMERIC(), primary_key=False)

class Transfers(Base):
    __tablename__ = 'transfers'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    transfer_id = Column(INTEGER(), primary_key=False)
    eventtype = Column(TEXT(), primary_key=False)
    careunit = Column(TEXT(), primary_key=False)
    intime = Column(TIMESTAMP(), primary_key=False)
    outtime = Column(TIMESTAMP(), primary_key=False)

class Services(Base):
    __tablename__ = 'services'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    transfertime = Column(TIMESTAMP(), primary_key=False)
    prev_service = Column(TEXT(), primary_key=False)
    curr_service = Column(TEXT(), primary_key=False)

class Ingredientevents(Base):
    __tablename__ = 'ingredientevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    starttime = Column(TIMESTAMP(), primary_key=False)
    endtime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    amount = Column(NUMERIC(), primary_key=False)
    amountuom = Column(TEXT(), primary_key=False)
    rate = Column(TEXT(), primary_key=False)
    rateuom = Column(TEXT(), primary_key=False)
    orderid = Column(INTEGER(), primary_key=False)
    linkorderid = Column(INTEGER(), primary_key=False)
    statusdescription = Column(TEXT(), primary_key=False)
    originalamount = Column(NUMERIC(), primary_key=False)
    originalrate = Column(NUMERIC(), primary_key=False)

class Inputevents(Base):
    __tablename__ = 'inputevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    starttime = Column(TIMESTAMP(), primary_key=False)
    endtime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    amount = Column(NUMERIC(), primary_key=False)
    amountuom = Column(TEXT(), primary_key=False)
    rate = Column(TEXT(), primary_key=False)
    rateuom = Column(TEXT(), primary_key=False)
    orderid = Column(INTEGER(), primary_key=False)
    linkorderid = Column(INTEGER(), primary_key=False)
    ordercategoryname = Column(TEXT(), primary_key=False)
    secondaryordercategoryname = Column(TEXT(), primary_key=False)
    ordercomponenttypedescription = Column(TEXT(), primary_key=False)
    ordercategorydescription = Column(TEXT(), primary_key=False)
    patientweight = Column(NUMERIC(), primary_key=False)
    totalamount = Column(NUMERIC(), primary_key=False)
    totalamountuom = Column(TEXT(), primary_key=False)
    isopenbag = Column(INTEGER(), primary_key=False)
    continueinnextdept = Column(INTEGER(), primary_key=False)
    statusdescription = Column(TEXT(), primary_key=False)
    originalamount = Column(NUMERIC(), primary_key=False)
    originalrate = Column(NUMERIC(), primary_key=False)

class Provider(Base):
    __tablename__ = 'provider'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    provider_id = Column(TEXT(), primary_key=False)

class ProceduresIcd(Base):
    __tablename__ = 'procedures_icd'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    seq_num = Column(INTEGER(), primary_key=False)
    chartdate = Column(TIMESTAMP(), primary_key=False)
    icd_code = Column(TEXT(), primary_key=False)
    icd_version = Column(INTEGER(), primary_key=False)

class Prescriptions(Base):
    __tablename__ = 'prescriptions'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    pharmacy_id = Column(INTEGER(), primary_key=False)
    poe_id = Column(TEXT(), primary_key=False)
    poe_seq = Column(INTEGER(), primary_key=False)
    order_provider_id = Column(TEXT(), primary_key=False)
    starttime = Column(TIMESTAMP(), primary_key=False)
    stoptime = Column(TIMESTAMP(), primary_key=False)
    drug_type = Column(TEXT(), primary_key=False)
    drug = Column(TEXT(), primary_key=False)
    formulary_drug_cd = Column(TEXT(), primary_key=False)
    gsn = Column(TEXT(), primary_key=False)
    ndc = Column(TEXT(), primary_key=False)
    prod_strength = Column(TEXT(), primary_key=False)
    form_rx = Column(TEXT(), primary_key=False)
    dose_val_rx = Column(TEXT(), primary_key=False)
    dose_unit_rx = Column(TEXT(), primary_key=False)
    form_val_disp = Column(TEXT(), primary_key=False)
    form_unit_disp = Column(TEXT(), primary_key=False)
    doses_per_24_hrs = Column(INTEGER(), primary_key=False)
    route = Column(TEXT(), primary_key=False)

class PoeDetail(Base):
    __tablename__ = 'poe_detail'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    poe_id = Column(TEXT(), primary_key=False)
    poe_seq = Column(INTEGER(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    field_name = Column(TEXT(), primary_key=False)
    field_value = Column(TEXT(), primary_key=False)

class Poe(Base):
    __tablename__ = 'poe'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    poe_id = Column(TEXT(), primary_key=False)
    poe_seq = Column(INTEGER(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    ordertime = Column(TIMESTAMP(), primary_key=False)
    order_type = Column(TEXT(), primary_key=False)
    order_subtype = Column(TEXT(), primary_key=False)
    transaction_type = Column(TEXT(), primary_key=False)
    discontinue_of_poe_id = Column(TEXT(), primary_key=False)
    discontinued_by_poe_id = Column(TEXT(), primary_key=False)
    order_provider_id = Column(TEXT(), primary_key=False)
    order_status = Column(TEXT(), primary_key=False)

class Pharmacy(Base):
    __tablename__ = 'pharmacy'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    pharmacy_id = Column(INTEGER(), primary_key=False)
    poe_id = Column(TEXT(), primary_key=False)
    starttime = Column(TIMESTAMP(), primary_key=False)
    stoptime = Column(TIMESTAMP(), primary_key=False)
    medication = Column(TEXT(), primary_key=False)
    proc_type = Column(TEXT(), primary_key=False)
    status = Column(TEXT(), primary_key=False)
    entertime = Column(TIMESTAMP(), primary_key=False)
    verifiedtime = Column(TIMESTAMP(), primary_key=False)
    route = Column(TEXT(), primary_key=False)
    frequency = Column(TEXT(), primary_key=False)
    disp_sched = Column(TEXT(), primary_key=False)
    infusion_type = Column(TEXT(), primary_key=False)
    sliding_scale = Column(TEXT(), primary_key=False)
    lockout_interval = Column(TEXT(), primary_key=False)
    basal_rate = Column(TEXT(), primary_key=False)
    one_hr_max = Column(TEXT(), primary_key=False)
    doses_per_24_hrs = Column(INTEGER(), primary_key=False)
    duration = Column(TEXT(), primary_key=False)
    duration_interval = Column(TEXT(), primary_key=False)
    expiration_value = Column(INTEGER(), primary_key=False)
    expiration_unit = Column(TEXT(), primary_key=False)
    expirationdate = Column(TIMESTAMP(), primary_key=False)
    dispensation = Column(TEXT(), primary_key=False)
    fill_quantity = Column(TEXT(), primary_key=False)

class Patients(Base):
    __tablename__ = 'patients'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    gender = Column(TEXT(), primary_key=False)
    anchor_age = Column(INTEGER(), primary_key=False)
    anchor_year = Column(INTEGER(), primary_key=False)
    anchor_year_group = Column(TEXT(), primary_key=False)
    dod = Column(DATE(), primary_key=False)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "age": self.anchor_age,
            "gender": self.gender,
            "dod": self.dod,
        }

class Omr(Base):
    __tablename__ = 'omr'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    chartdate = Column(TIMESTAMP(), primary_key=False)
    seq_num = Column(INTEGER(), primary_key=False)
    result_name = Column(TEXT(), primary_key=False)
    result_value = Column(TEXT(), primary_key=False)

class Microbiologyevents(Base):
    __tablename__ = 'microbiologyevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    microevent_id = Column(INTEGER(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(TEXT(), primary_key=False)
    micro_specimen_id = Column(INTEGER(), primary_key=False)
    order_provider_id = Column(TEXT(), primary_key=False)
    chartdate = Column(TIMESTAMP(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    spec_itemid = Column(INTEGER(), primary_key=False)
    spec_type_desc = Column(TEXT(), primary_key=False)
    test_seq = Column(INTEGER(), primary_key=False)
    storedate = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    test_itemid = Column(INTEGER(), primary_key=False)
    test_name = Column(TEXT(), primary_key=False)
    org_itemid = Column(TEXT(), primary_key=False)
    org_name = Column(TEXT(), primary_key=False)
    isolate_num = Column(TEXT(), primary_key=False)
    quantity = Column(TEXT(), primary_key=False)
    ab_itemid = Column(TEXT(), primary_key=False)
    ab_name = Column(TEXT(), primary_key=False)
    dilution_text = Column(TEXT(), primary_key=False)
    dilution_comparison = Column(TEXT(), primary_key=False)
    dilution_value = Column(TEXT(), primary_key=False)
    interpretation = Column(TEXT(), primary_key=False)
    comments = Column(TEXT(), primary_key=False)

class Labevents(Base):
    __tablename__ = 'labevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    labevent_id = Column(INTEGER(), primary_key=False)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(TEXT(), primary_key=False)
    specimen_id = Column(INTEGER(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    order_provider_id = Column(TEXT(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    value = Column(TEXT(), primary_key=False)
    valuenum = Column(NUMERIC(), primary_key=False)
    valueuom = Column(TEXT(), primary_key=False)
    ref_range_lower = Column(NUMERIC(), primary_key=False)
    ref_range_upper = Column(NUMERIC(), primary_key=False)
    flag = Column(TEXT(), primary_key=False)
    priority = Column(TEXT(), primary_key=False)
    comments = Column(TEXT(), primary_key=False)

    def to_dict(self):
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "itemid": self.itemid,
            "charttime": self.charttime,
            "value": self.value,
            "valuenum": self.valuenum,
        }

class Hcpcsevents(Base):
    __tablename__ = 'hcpcsevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    chartdate = Column(TIMESTAMP(), primary_key=False)
    hcpcs_cd = Column(TEXT(), primary_key=False)
    seq_num = Column(INTEGER(), primary_key=False)
    short_description = Column(TEXT(), primary_key=False)

class EmarDetail(Base):
    __tablename__ = 'emar_detail'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    emar_id = Column(TEXT(), primary_key=False)
    emar_seq = Column(INTEGER(), primary_key=False)
    parent_field_ordinal = Column(NUMERIC(), primary_key=False)
    administration_type = Column(TEXT(), primary_key=False)
    pharmacy_id = Column(INTEGER(), primary_key=False)
    barcode_type = Column(TEXT(), primary_key=False)
    reason_for_no_barcode = Column(TEXT(), primary_key=False)
    complete_dose_not_given = Column(TEXT(), primary_key=False)
    dose_due = Column(TEXT(), primary_key=False)
    dose_due_unit = Column(TEXT(), primary_key=False)
    dose_given = Column(TEXT(), primary_key=False)
    dose_given_unit = Column(TEXT(), primary_key=False)
    will_remainder_of_dose_be_given = Column(TEXT(), primary_key=False)
    product_amount_given = Column(TEXT(), primary_key=False)
    product_unit = Column(TEXT(), primary_key=False)
    product_code = Column(TEXT(), primary_key=False)
    product_description = Column(TEXT(), primary_key=False)
    product_description_other = Column(TEXT(), primary_key=False)
    prior_infusion_rate = Column(TEXT(), primary_key=False)
    infusion_rate = Column(TEXT(), primary_key=False)
    infusion_rate_adjustment = Column(TEXT(), primary_key=False)
    infusion_rate_adjustment_amount = Column(TEXT(), primary_key=False)
    infusion_rate_unit = Column(TEXT(), primary_key=False)
    route = Column(TEXT(), primary_key=False)
    infusion_complete = Column(TEXT(), primary_key=False)
    completion_interval = Column(TEXT(), primary_key=False)
    new_iv_bag_hung = Column(TEXT(), primary_key=False)
    continued_infusion_in_other_location = Column(TEXT(), primary_key=False)
    restart_interval = Column(TEXT(), primary_key=False)
    side = Column(TEXT(), primary_key=False)
    site = Column(TEXT(), primary_key=False)
    non_formulary_visual_verification = Column(TEXT(), primary_key=False)

class Outputevents(Base):
    __tablename__ = 'outputevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    charttime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    value = Column(TEXT(), primary_key=False)
    valueuom = Column(TEXT(), primary_key=False)

class Procedureevents(Base):
    __tablename__ = 'procedureevents'
    __table_args__ = {'schema': 'public'}

    row_id = Column(BIGINT(), primary_key=True)
    subject_id = Column(INTEGER(), primary_key=True)
    hadm_id = Column(INTEGER(), primary_key=False)
    stay_id = Column(INTEGER(), primary_key=False)
    caregiver_id = Column(INTEGER(), primary_key=False)
    starttime = Column(TIMESTAMP(), primary_key=False)
    endtime = Column(TIMESTAMP(), primary_key=False)
    storetime = Column(TIMESTAMP(), primary_key=False)
    itemid = Column(INTEGER(), primary_key=False)
    value = Column(TEXT(), primary_key=False)
    valueuom = Column(TEXT(), primary_key=False)
    location = Column(TEXT(), primary_key=False)
    locationcategory = Column(TEXT(), primary_key=False)
    orderid = Column(INTEGER(), primary_key=False)
    linkorderid = Column(INTEGER(), primary_key=False)
    ordercategoryname = Column(TEXT(), primary_key=False)
    ordercategorydescription = Column(TEXT(), primary_key=False)
    patientweight = Column(NUMERIC(), primary_key=False)
    isopenbag = Column(INTEGER(), primary_key=False)
    continueinnextdept = Column(INTEGER(), primary_key=False)
    statusdescription = Column(TEXT(), primary_key=False)
    originalamount = Column(NUMERIC(), primary_key=False)
    originalrate = Column(NUMERIC(), primary_key=False)

