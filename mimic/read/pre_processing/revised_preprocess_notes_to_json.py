from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mimic.orm_create.mimiciv_v3_orm import PreprocessedRevisedNote

DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()
queried_notes = session.query(PreprocessedRevisedNote).all()
print(len(queried_notes))