from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
import os

def generate_orm_classes(uri, schema, output_path="mimiciv_v3_orm.py"):
    engine = create_engine(uri)
    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema)
    Base = declarative_base(metadata=metadata)

    with open(output_path, 'w') as f:
        f.write("from sqlalchemy import Column, INTEGER, TEXT, TIMESTAMP, VARCHAR, ForeignKey, NUMERIC\n")
        f.write("from sqlalchemy.ext.declarative import declarative_base\n")
        f.write("Base = declarative_base()\n\n")

        for table_name, table in metadata.tables.items():
            if not table_name.startswith(f'{schema}.'):
                continue
            cls_name = ''.join(part.capitalize() for part in table_name.replace(f'{schema}.', '').split('_'))
            f.write(f"class {cls_name}(Base):\n")
            f.write(f"    __tablename__ = '{table_name.replace(f'{schema}.', '')}'\n")
            f.write(f"    __table_args__ = {{'schema': '{schema}'}}\n\n")
            ## TODO some dont have primary key
            for col in table.columns:
                if col.name == "subject_id":
                    f.write(f"    {col.name} = Column({repr(col.type)}, primary_key={True})\n")

                    continue
                f.write(f"    {col.name} = Column({repr(col.type)}, primary_key={col.primary_key})\n")
            f.write("\n")

    print(f"ORM classes written to {output_path}")

if __name__ == '__main__':
    # Example URI: postgresql://username:password@localhost:5432/mimiciv
    db_uri = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
    schema = "mimicIV"

    if not db_uri:
        raise ValueError("Please set DB_URI as an environment variable.")

    generate_orm_classes(db_uri, schema)
