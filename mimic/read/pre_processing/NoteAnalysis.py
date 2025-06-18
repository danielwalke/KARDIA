import re
import json
import os
from kg_gen import KGGen
from typing import Optional, Dict, List
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument
from mimic.read.pre_processing.embed_notes import NoteInformationRetriever
import time
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tqdm import tqdm
from mimic.read.pre_processing.schema.nodes import MEDICAL_NODE_TYPES
from mimic.read.pre_processing.schema.relationships import MEDICAL_RELATIONSHIP_TYPES
from mimic.read.prompts.concept_extraction.SystemPrompt import LLM_SYSTEM_PROMPT_TEMPLATE
from mimic.read.prompts.concept_extraction.UserPrompt import LLM_USER_PROMPT_TEMPLATE


class OntologyCodes(BaseModel):
    SNOMED_CT: Optional[str] = Field(None, description="SNOMED CT code, if applicable")
    ICD_10: Optional[str] = Field(None, description="ICD-10 code, if applicable")
    LOINC: Optional[str] = Field(None, description="LOINC code, if applicable")
    RxNorm: Optional[str] = Field(None, description="RxNorm code, if applicable for medications")
    OTHER: Optional[Dict[str, str]] = Field(None, description="Other ontology codes as key-value pairs")

class Concept(BaseModel):
    id: str = Field(description="Unique identifier for the concept (e.g., 'concept_1')")
    text_span: str = Field(description="The exact text span from the original narrative")
    extracted_term: str = Field(description="The specific term identified in the text")
    normalized_term: str = Field(description="A standardized or canonical form of the term")
    semantic_type: str = Field(description="Category of the concept (e.g., 'Medical Condition', 'Lifestyle Factor')")
    ontology_codes: Optional[OntologyCodes] = Field(None, description="Mappings to standard medical terminologies")
    negated: bool = Field(default=False, description="True if the concept is explicitly negated in the text")
    uncertainty: str = Field(default="none", description="Level of uncertainty (e.g., 'none', 'possible', 'probable')")

class Relationship(BaseModel):
    source_concept_id: str = Field(description="ID of the source concept in the relationship")
    target_concept_id: str = Field(description="ID of the target concept in the relationship")
    relationship_type: str = Field(description="Type of relationship (e.g., 'CAUSES', 'ASSOCIATED_WITH')")
    context: Optional[str] = Field(None, description="Brief justification or textual evidence for the relationship")

class MedicalOntologyOutput(BaseModel):
    concepts: List[Concept] = Field(description="A list of extracted medical and social concepts")
    relationships: List[Relationship] = Field(description="A list of identified relationships between concepts")


def invoke_llm(chain, note_text):
	input_data = {
		"narrative_text": note_text
	}
	response = chain.invoke(input_data)
	return response

def count_words_in_line(word_dict, line):
	for word in line.strip().split(" "):
		word = word.lower().strip().replace("___", "DEIDENTIFIED_PERSON")
		for sub_word in word.split("/"):
			if sub_word not in word_dict:
				word_dict[sub_word] = 0
			word_dict[sub_word] += 1


def count_words_in_text(word_dict, note_text):
	pattern = r"[\'\"\`.?,;:\-\%()\[\]{}=\d]"
	note_text = re.sub(pattern, "", note_text)
	for line in note_text.strip().split("\n"):
		count_words_in_line(word_dict, line)


def create_medical_knowledge_graph(llm: BaseLanguageModel, text_history: str) -> GraphDocument:
    """
    Generates a knowledge graph from a given medical or social history text using an LLM.

    This function initializes an LLMGraphTransformer with predefined medical-specific
    node and relationship types to guide the graph extraction process.

    Args:
        llm: An initialized Langchain-compatible Large Language Model instance.
        text_history: A string containing the social or medical history.

    Returns:
        A GraphDocument object representing the extracted knowledge graph.
        This object contains lists of nodes and relationships.
    """
    # Initialize the LLMGraphTransformer with the provided LLM and the
    # predefined allowed node and relationship types for the medical domain.
    # Note: Parameter names for allowed types might vary slightly across Langchain versions
    # (e.g., 'allowed_labels' instead of 'allowed_nodes').
    # For recent versions of `langchain_experimental`, `allowed_nodes` and `allowed_relationships` are standard.
    print(MEDICAL_NODE_TYPES)
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        # allowed_nodes=MEDICAL_NODE_TYPES,
        # allowed_relationships=MEDICAL_RELATIONSHIP_TYPES,
		# node_properties=["value", "name"]
    )

    # The transformer expects a list of Langchain Document objects.
    print(text_history)
    document = Document(page_content=text_history)
    documents_to_process = [document]

    # Convert the document(s) into graph document(s).
    # This method returns a list of GraphDocument objects.
    graph_documents: List[GraphDocument] = llm_transformer.convert_to_graph_documents(documents_to_process)

    # For a single input document, we expect a single GraphDocument in the list.
    # If no graph elements are found, the transformer might return a GraphDocument with empty nodes/relationships,
    # or potentially an empty list if it couldn't process the document at all.
    if graph_documents:
        return graph_documents[0]
    else:
        # If the transformer returns an empty list (e.g., if the text is empty or unprocessable),
        # return an empty GraphDocument.
        return GraphDocument(nodes=[], relationships=[], source=document)


def create_knowledge_graph(note: dict, port: int):
	text_history = f"""
			**Anamnesis for Patient {note["row_id"]}**
			{note["text"]}
	"""
	os.makedirs("kgs/", exist_ok=True)
	# Initialize KGGen with optional configuration
	kg = KGGen(
		model="ollama_chat/qwen3:32b",  # Default model
		temperature=0.0,  # Default temperature
		api_key=None,  # Optional if set in environment or using a local model
		api_base=f"http://127.0.0.1:{port}"
	)
	kg.generate(
		input_data=text_history,
		context=f"Extract medical Patient history for patient {note['row_id']}",
		output_folder=f"kgs/{note['row_id']}",
	)



def extract_word_count_json(note):
	word_dict = dict()
	count_words_in_text(word_dict, note["text"])
	frequent_word_tuples = list(filter(lambda t: t[1] > 1, word_dict.items()))
	infrequent_word_tuples = list(filter(lambda t: t[1] < 1, word_dict.items()))
	print(infrequent_word_tuples)
	sorted_word_dict = dict(sorted(frequent_word_tuples, key=lambda item: -item[1]))



if __name__ == '__main__':
	start_time = time.time()
	note_retriever = NoteInformationRetriever()
	notes = note_retriever.read_notes(2)
	llm = ChatOllama(model="qwen3:32b",temperature=0)
	# prompt = ChatPromptTemplate.from_messages([
	# 	("system", LLM_SYSTEM_PROMPT_TEMPLATE),
	# 	("user", LLM_USER_PROMPT_TEMPLATE)
	# ])
	# parser = JsonOutputParser(pydantic_object=MedicalOntologyOutput)
	# chain = prompt | llm | parser
	for note in tqdm(notes):
		create_knowledge_graph(note = note, port = 11436)
	print("--- %s seconds ---" % (time.time() - start_time))
