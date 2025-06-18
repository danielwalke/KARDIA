import psycopg
from datetime import datetime


class DB_Note_Connection:
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
        Creates the 'notes' table if it does not already exist in the database.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    row_id BIGINT,
                    note_id TEXT,
                    subject_id INT,
                    hadm_id INT,
                    charttime TIMESTAMP,
                    text TEXT
                );
            """)
            self.conn.commit()
        print("Table 'notes' created or already exists.")

    def add_new_note(self, note_data: dict):
        """
        Adds a new row to the 'notes' table.

        Args:
            note_data (dict): A dictionary containing the note information with the following keys:
                              'row_id', 'note_id', 'subject_id', 'hadm_id', 'charttime', 'text'.

        Raises:
            ValueError: If any required key is missing from note_data.
        """
        required_keys = ['row_id', 'note_id', 'subject_id', 'hadm_id', 'charttime', 'text']
        for key in required_keys:
            if key not in note_data:
                raise ValueError(f"Missing required key in note_data: '{key}'")

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO notes (row_id, note_id, subject_id, hadm_id, charttime, text)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (note_data['row_id'],
                 note_data['note_id'],
                 note_data['subject_id'],
                 note_data['hadm_id'],
                 note_data['charttime'],
                 note_data['text'])
            )
            self.conn.commit()

    def drop_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""DROP TABLE IF EXISTS notes""")
            self.conn.commit()
        print("Table 'notes' dropped if existed")

    def close_connection(self):
        self.conn.close()


if __name__ == "__main__":
    # Database connection parameters
    # Replace with your actual database credentials
    try:
        # Establish a connection to the PostgreSQL database
        db_note_connection = DB_Note_Connection()
        print("Database connection established successfully.")

        # # 1. Create the table if not exists
        # db_note_connection.create_table_if_not_exists()
        #
        # # 2. Example usage of add_new_note function
        # new_note_data_1 = {
        #     'row_id': 1001,
        #     'note_id': 'NOTE_XYZ_001',
        #     'subject_id': 10,
        #     'hadm_id': 200,
        #     'charttime': datetime(2023, 1, 15, 10, 30, 0),
        #     'text': 'Patient presented with fever and cough.'
        # }
        # db_note_connection.add_new_note(new_note_data_1)
        # db_note_connection.close_connection()

    except psycopg.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
