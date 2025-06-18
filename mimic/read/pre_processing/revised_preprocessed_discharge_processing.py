from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from mimic.orm_create.mimiciv_v3_orm import Note, Discharge, PreprocessedRevisedNote, Base, RevisedNote


def process_text(text):
	text = text.lower()
	## TODO Lemmatize + stop word removal 
	return note


DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
discharge_notes = session.query(RevisedNote).all()
new_notes = []
for note in discharge_notes:
	new_note = process_text(note.text)
	new_notes.append(PreprocessedRevisedNote(row_id = note.row_id, note_id = note.note_id, subject_id = note.subject_id, hadm_id=note.hadm_id, charttime=note.charttime, text=new_note))
with Session() as session:
	session.add_all(new_notes)
	session.commit()