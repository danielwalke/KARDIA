import psycopg

class LabelUpdater:
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
                CREATE TABLE IF NOT EXISTS labels (
                    row_id BIGINT,
                    hadm_id INT,
                    label BOOL,
                    icd_codes TEXT[]
                );
            """)
            self.conn.commit()
        print("Table 'labels' created or already exists.")

    def add_new_label(self, label_data: dict):
        """
        Adds a new row to the 'qwen_embeddings' table.

        Args:
            embedding_data (dict): A dictionary containing the label information with the following keys:
                              'row_id', 'hadm_id', 'label', 'icd_codes'.

        Raises:
            ValueError: If any required key is missing from note_data.
        """
        required_keys = ['row_id', 'hadm_id', 'label', 'icd_codes']
        for key in required_keys:
            if key not in label_data:
                raise ValueError(f"Missing required key in note_data: '{key}'")

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO labels (row_id, hadm_id, label, icd_codes)
                VALUES (%s, %s, %s, %s);
                """,
                (label_data['row_id'],
                 label_data['hadm_id'],
                 label_data['label'],
                 label_data['icd_codes'])
            )
            self.conn.commit()

    def drop_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""DROP TABLE IF EXISTS labels""")
            self.conn.commit()
        print("Table 'labels' dropped if existed")

    def close_connection(self):
        self.conn.close()