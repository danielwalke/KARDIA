from typing import List

MEDICAL_RELATIONSHIP_TYPES: List[str] = [
    "HAS"
#     "HAS_SYMPTOM",              # Connects a patient or condition to a symptom (e.g., Patient HAS_SYMPTOM Fever)
#     "DIAGNOSED_WITH",           # Indicates a formal diagnosis (e.g., Patient DIAGNOSED_WITH Type 2 Diabetes)
#     "PRESCRIBED_MEDICATION",    # Links a patient or condition to a prescribed medication (e.g., Patient PRESCRIBED_MEDICATION Insulin)
#     "TAKES_MEDICATION",         # Indicates current use of a medication (e.g., Patient TAKES_MEDICATION Lisinopril)
#     "ALLERGIC_TO",              # Specifies an allergy (e.g., Patient ALLERGIC_TO Penicillin)
#     "UNDERWENT_PROCEDURE",      # Indicates a procedure was performed (e.g., Patient UNDERWENT_PROCEDURE Colonoscopy)
#     "EXHIBITS_BEHAVIOR",        # Links to a lifestyle behavior (e.g., Patient EXHIBITS_BEHAVIOR Smoking)
#     "AFFECTED_BY_CONTEXT",      # Links to a social factor (e.g., Patient AFFECTED_BY_CONTEXT Unemployment)
#     "HAS_FAMILY_HISTORY_OF",    # Indicates a condition present in the patient's family (e.g., Patient HAS_FAMILY_HISTORY_OF Heart Disease)
#     "COMPLICATES",              # Shows one condition exacerbating another (e.g., Diabetes COMPLICATES Wound Healing)
#     "TREATS",                   # Indicates a treatment for a condition/symptom (e.g., Lisinopril TREATS Hypertension)
#     "CAUSES_ADVERSE_EFFECT",    # Links a treatment/substance to an adverse effect (e.g., Medication X CAUSES_ADVERSE_EFFECT Nausea)
#     "SHOWS_VITAL_SIGN",         # Connects to an observed vital sign (e.g., Patient SHOWS_VITAL_SIGN Elevated Blood Pressure)
#     "USES_SUBSTANCE",           # Indicates use of a specific substance (e.g., Patient USES_SUBSTANCE Alcohol)
#     "IMPROVES",                 # Indicates something that alleviates a condition/symptom (e.g., Rest IMPROVES Headache)
#     "WORSENS",                  # Indicates something that aggravates a condition/symptom (e.g., Stress WORSENS Migraine)
#     "ASSOCIATED_WITH",           # A general relationship if a more specific one isn't suitable
#     "IS_FAMILY_OF",
#     "HAD_EVENT",
#     "HAD_DISEASE",
# "HAD_MEDICAL_TEST",
# "HAS_PATIENT_ATTRIBUTE",
]