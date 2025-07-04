from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from mimic.orm_create.mimiciv_v3_orm import Note, Discharge, PreprocessedRevisedNote, Base, RevisedNote
import spacy

def process_text(text):
	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)
	## Lemmatization, stop word removal and rmeoval of punctuations
	lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
	lemmatized_text = ' '.join(lemmatized_tokens)
	return lemmatized_text


DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
discharge_notes = session.query(RevisedNote).all()
new_notes = []
for note in tqdm(discharge_notes):
	print(note.text)
	new_note = process_text(note.text)
	new_notes.append(PreprocessedRevisedNote(row_id = note.row_id, note_id = note.note_id, subject_id = note.subject_id, hadm_id=note.hadm_id, charttime=note.charttime, text=new_note))
	print(new_note)
	break
exit()
with Session() as session:
	session.add_all(new_notes)
	session.commit()