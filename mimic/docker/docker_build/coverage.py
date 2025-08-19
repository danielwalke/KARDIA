import json
import logging
import os
import sys
from datetime import date
import os

if r"danie\git" in os.getcwd():
    os.chdir("../")

KG_OUTPUT_DIR ="kgs_out"
lower_limit = sys.argv[2]

logging.basicConfig(
        filename=f"{KG_OUTPUT_DIR}/coverage_{lower_limit}_{date.today()}.log",
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO
)

kg_dirs = os.listdir(f"./{KG_OUTPUT_DIR}")
logging.info(f"Analyzing {len(kg_dirs)} KGs")
coverage_list = []
for kg_dir in kg_dirs:
    if not os.path.isdir(f"./{KG_OUTPUT_DIR}/{kg_dir}"):
        logging.info(f"Skipping {kg_dir}")
        continue
    logging.info(f"Loading {kg_dir}")
    with open(f"./{KG_OUTPUT_DIR}/{kg_dir}/graph.json", "r") as f:
        graph = json.load(f)
        logging.info(f"Loaded {kg_dir}")
        extracted_information = graph["entities"] + [rel[0] for rel in graph["relations"]] + [rel[1] for rel in graph["relations"]] + [rel[2] for rel in graph["relations"]]
        extracted_information = [i.split(" ") for i in extracted_information]
        flat_extracted_information = set([itm.replace("ANONYMIZED", "___") for row in extracted_information for itm in row])
    logging.info(f"Loading note for {kg_dir}")
    with open(f"./notes/{kg_dir}.json", "r") as f:
        logging.info(f"Loaded note for {kg_dir}")
        note = json.load(f)
        text = note["text"]
        words = set(text.split(" "))
    logging.info(f"Calculate coverage")
    intersected_words = words.intersection(flat_extracted_information)

    word_len = len(words)
    coverage = None
    if word_len == 0:
        coverage = 0
    else:
        coverage = len(intersected_words) / (word_len)
    logging.info(f"{kg_dir} - coverage: {coverage:.4f}")
    coverage_list.append(coverage)
avg_coverage = sum(coverage_list) / len(coverage_list)
logging.info(f"Average coverage for this batch: {avg_coverage:.4f}")