import os
import argparse
import csv
import re
import psycopg2
from psycopg2 import sql


# Basic type inference functions (assuming infer_sql_type remains the same)
def infer_sql_type(value):
	"""
	Infer SQL data type from a sample CSV value.
	"""
	# Null or empty
	if value == '' or value.lower() in ('null', 'none'):
		return None
	# Integer
	if re.fullmatch(r"[+-]?\d+", value):
		return 'INTEGER'
	# Float
	if re.fullmatch(r"[+-]?(?:\d*\.)?\d+", value):
		return 'NUMERIC'
	# ISO date or datetime
	if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?", value):
		# contains time?
		return 'TIMESTAMP' if 'T' in value or ' ' in value else 'DATE'
	# Default to text
	return 'TEXT'


def upload_and_create_schema(conn, schema, directory):
	"""
	Automatically create schema, infer tables and types from CSVs, then load data,
	adding a synthetic auto-incrementing primary key to each table.
	"""
	cur = conn.cursor()
	# Create schema if not exists (using the fix from the previous issue)
	create_schema_sql = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
	cur.execute(create_schema_sql)
	conn.commit()

	for fname in os.listdir(directory):
		if not fname.lower().endswith('.csv'):
			continue
		table = os.path.splitext(fname)[0]
		csv_path = os.path.join(directory, fname)

		print(f"Processing {csv_path}...")
		# Read header and first data row
		with open(csv_path, newline='', encoding='utf-8') as f:
			reader = csv.reader(f)
			header = next(reader)
			try:
				sample = next(reader)
			except StopIteration:
				sample = [''] * len(header)

		# Infer column types from CSV
		columns_from_csv = []  # Stores (cleaned_col_name, sql_type)
		for col, val in zip(header, sample):
			col_clean = re.sub(r"[^a-zA-Z0-9_]", "_", col).lower()

			# Your existing type inference logic based on column names and values
			if re.search(r"date|time", col_clean):
				sql_type = 'TIMESTAMP'
			elif re.search(r"icd_code", col_clean):
				sql_type = 'TEXT'
			elif "dose_given" == col_clean or "product_amount_given" == col_clean:
				sql_type = 'TEXT'
			elif "hcpcs_cd" == col_clean:
				sql_type = 'TEXT'
			elif "value" == col_clean or "dose_val_rx" == col_clean or "form_val_disp" == col_clean or "ndc" == col_clean or "gsn" == col_clean or "disp_sched" == col_clean or "code" == col_clean:
				sql_type = 'TEXT'
			elif "valuenum" == col_clean or "patientweight" == col_clean or "originalamount" == col_clean or "amount" == col_clean or "totalamount" == col_clean or "originalrate" == col_clean or "ref_range_lower" == col_clean or "ref_range_upper" == col_clean:
				sql_type = 'NUMERIC'
			elif re.search(r"given", col_clean) and not re.search(r"unit", col_clean) and not re.search(
					r"complete_dose_not_given", col_clean) and not re.search(r"will_remainder_of_dose_be_given",
																			 col_clean):
				sql_type = 'NUMERIC'
			else:
				sql_type = infer_sql_type(val) or 'TEXT'
			columns_from_csv.append((col_clean, sql_type))

		# Drop table if exists
		cur.execute(
			sql.SQL("DROP TABLE IF EXISTS {}.{};")
			.format(sql.Identifier(schema), sql.Identifier(table))
		)
		conn.commit()

		# Build CREATE TABLE statement
		table_column_definitions_sql = []
		synthetic_pk_name = "row_id"  # Name for your auto-incrementing PK column

		# 1. Add the synthetic auto-incrementing primary key column
		table_column_definitions_sql.append(
			sql.SQL("{} BIGSERIAL PRIMARY KEY").format(sql.Identifier(synthetic_pk_name))
		)

		# 2. Add columns derived from the CSV file
		csv_column_identifiers_for_copy = []  # To list columns in the COPY statement
		for name, typ in columns_from_csv:
			table_column_definitions_sql.append(sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(typ)))
			csv_column_identifiers_for_copy.append(sql.Identifier(name))

		# The previous hardcoded PK line (`sql.SQL("PRIMARY KEY ({})").format(sql.Identifier("p_id"))`) is removed.

		create_table_statement = sql.SQL("CREATE TABLE {}.{} ({})").format(
			sql.Identifier(schema),
			sql.Identifier(table),
			sql.SQL(', ').join(table_column_definitions_sql)
		)
		cur.execute(create_table_statement)
		conn.commit()

		csv_cols_formatted = [f"{name}({typ})" for name, typ in columns_from_csv]
		print(
			f"Created table {schema}.{table} with synthetic PK '{synthetic_pk_name}' "
			f"and data columns: {', '.join(csv_cols_formatted)}"
		)

		# Load data using COPY, specifying the target columns from the CSV
		with open(csv_path, 'r', encoding='utf-8') as f:
			copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH CSV HEADER").format(
				sql.Identifier(schema),
				sql.Identifier(table),
				sql.SQL(', ').join(csv_column_identifiers_for_copy)  # Specify CSV columns
			)
			cur.copy_expert(copy_sql, f)
		conn.commit()
		print(f"Loaded data into {schema}.{table}.")

	cur.close()


if __name__ == '__main__':
	r"""
	Example usage:
		python .\upload_mimic_data.py --host localhost  --port 5432 --dbname mimicIV_v3 --user postgres --password password --schema mimicIV_v3 --dir .\data\mimic-iv-3.1\hosp\
	"""
	parser = argparse.ArgumentParser(
		description='Auto-create schema and bulk load MIMIC-IV CSVs into Postgres with synthetic PKs')
	# Arguments remain the same as your original script
	parser.add_argument('--host', required=True, help='Postgres host')
	parser.add_argument('--port', default=5432, type=int, help='Postgres port')
	parser.add_argument('--dbname', required=True, help='Database name')
	parser.add_argument('--user', required=True, help='Postgres user')
	parser.add_argument('--password', required=True, help='Postgres password')
	parser.add_argument('--schema', default='public', help='Target schema')
	parser.add_argument('--dir', required=True, help='Directory with CSV files')
	args = parser.parse_args()

	conn = psycopg2.connect(
		host=args.host,
		port=args.port,
		dbname=args.dbname,
		user=args.user,
		password=args.password
	)
	try:
		upload_and_create_schema(conn, args.schema, args.dir)
	finally:
		conn.close()
	print("All files processed.")

