from typing import List

MEDICAL_NODE_TYPES: List[str] = [
    "Patient",
    "PatientAttribute",
    "FamilyMember",
    "Disease",
    "Event",
    "MedicalTest",
    "Symptom",             # Discrete signs or indications of a health issue (e.g., "Headache", "Fever", "Chronic Cough")
    "MedicalCondition",    # Diagnosed diseases, disorders, or significant health states (e.g., "Hypertension", "Type 2 Diabetes", "Asthma Attack")
    "Medication",          # Pharmaceutical drugs or therapeutic substances (e.g., "Lisinopril 10mg", "Metformin", "Aspirin")
    "MedicalProcedure",    # Interventions, tests, or surgeries (e.g., "Colonoscopy", "Appendectomy", "Blood Test")
    "LifestyleBehavior",   # Habits or ongoing activities affecting health (e.g., "Smoking", "Regular Aerobic Exercise", "High-Sodium Diet")
    "SocialContext",       # Social and environmental factors (e.g., "Unemployed", "Lives Alone", "High-Stress Work Environment")
    "Allergen",            # Substances causing allergic reactions (e.g., "Penicillin", "Peanuts", "Dust Mites")
    "VitalSign",           # Clinical measurements indicating basic body functions (e.g., "Elevated Blood Pressure", "Heart Rate 72 bpm", "Body Temperature 37.0°C")
    "Substance",           # Consumed substances, often referring to those with potential for misuse (e.g., "Tobacco", "Alcohol", "Opioids")
    "AdverseEffect"        # Negative outcomes or side effects (e.g., "Medication-induced Rash", "Post-operative Infection")
]