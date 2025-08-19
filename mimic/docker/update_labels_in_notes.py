import json
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from mimic.orm_create.mimiciv_v3_orm import Labels

DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()
labels = session.query(Labels).all()
label_dict = {label.row_id: int(label.label) for label in labels}
notes = os.listdir("notes")
notes.sort()
os.makedirs("labeled_notes", exist_ok=True)

for note in tqdm(notes):
	with open(f"notes/{note}", 'r') as f:
		note_content = json.loads(f.read())
		note_content["label"] = label_dict[note_content["row_id"]]
	with open(f"labeled_notes/{note}", 'w') as f:
		f.write(json.dumps(note_content, indent=4))
