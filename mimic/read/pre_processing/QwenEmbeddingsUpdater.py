import psycopg

class QwenEmbeddingsUpdater:
    def __init__(self):
        DB_NAME = "mimicIV_v3"
        DB_USER = "postgres"
        DB_PASSWORD = "password"
        DB_HOST = "localhost"
        DB_PORT = "5432"
        self.conn = psycopg.connect(
            f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}")


    def create_table_if_not_exists(self):
        """
        Creates the 'embedding' table if it does not already exist in the database.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qwen_embeddings (
                    row_id BIGINT,
                    note_id TEXT,
                    subject_id INT,
                    hadm_id INT,
                    charttime TIMESTAMP,
                    embedding REAL[]
                );
            """)
            self.conn.commit()
        print("Table 'qwen_embeddings' created or already exists.")

    def add_new_embedding(self, embedding_data: dict):
        """
        Adds a new row to the 'qwen_embeddings' table.

        Args:
            embedding_data (dict): A dictionary containing the note information with the following keys:
                              'row_id', 'note_id', 'subject_id', 'hadm_id', 'charttime', 'embedding'.

        Raises:
            ValueError: If any required key is missing from note_data.
        """
        required_keys = ['row_id', 'note_id', 'subject_id', 'hadm_id', 'charttime', 'embedding']
        for key in required_keys:
            if key not in embedding_data:
                raise ValueError(f"Missing required key in note_data: '{key}'")

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO qwen_embeddings (row_id, note_id, subject_id, hadm_id, charttime, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (embedding_data['row_id'],
                 embedding_data['note_id'],
                 embedding_data['subject_id'],
                 embedding_data['hadm_id'],
                 embedding_data['charttime'],
                 embedding_data['embedding'])
            )
            self.conn.commit()

    def drop_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""DROP TABLE IF EXISTS qwen_embeddings""")
            self.conn.commit()
        print("Table 'qwen_embeddings' dropped if existed")

    def close_connection(self):
        self.conn.close()