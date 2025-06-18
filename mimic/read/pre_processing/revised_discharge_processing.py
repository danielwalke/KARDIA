from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from mimic.orm_create.mimiciv_v3_orm import Note, Discharge, RevisedNote, Base


def process_text(text):
	lines = text.split("\n")
	heading = None
	note = ""
	included_information = ["Allergies", "Chief Complaint", "History of Present Illness", "Past Medical History", "Social History",
							"Physical Exam"]
	for line in lines:
		if line.strip().endswith(":"): ## TODO This doesnt hold for all headings...
			heading = line.split(":")[0]
		if heading is None or any(list(map(lambda title: title in heading, included_information))):
			note += line
			note+= "\n"
	return note


DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
discharge_notes = session.query(Discharge).all()
new_notes = []
for note in discharge_notes:
	new_note = process_text(note.text)
	new_notes.append(RevisedNote(row_id = note.row_id, note_id = note.note_id, subject_id = note.subject_id, hadm_id=note.hadm_id, charttime=note.charttime, text=new_note))
with Session() as session:
	session.add_all(new_notes)
	session.commit()