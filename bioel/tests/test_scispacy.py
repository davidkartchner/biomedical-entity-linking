##### Test file that you must put into the "package" folder #####
import unittest
from tqdm import tqdm

import spacy
from bioel.ontology import BiomedicalOntology
from bioel.models.scispacy.candidate_generation import CandidateGenerator
from bioel.models.scispacy.scispacy_embeddings import KnowledgeBaseEmbeddings
from bioel.models.scispacy.entity_linking import EntityLinker

class TestScispacy(unittest.TestCase):
    
	def test_kb_generation(self):
		test_cases = [
			{
				"file_path": "./kb_test2.jsonl",
				"name": "test_onto",
			},]

		for case in tqdm(test_cases):
			ontology = BiomedicalOntology.load_json(**case)
			myembed = KnowledgeBaseEmbeddings(ontology)                            				 # initializes the class for the embeddings given an Ontology in the memory
			myembed.create_tfidf_ann_index("./bioel/models/scispacy/kb_paths_scispacy")          # creates the tfidf vectors and the ann indexes
			kb = myembed.serialized_kb()                                         				 # returns a NamedTuple containing the necessariy paths to the embeddings and the needed dectionnaries in the memory
			print(kb)
   

	def test_candidate_generation(self):
		test_cases = [
		{
			"file_path": "./kb_test2.jsonl",
			"name": "test_onto",
		},]

		for case in tqdm(test_cases):
			ontology = BiomedicalOntology.load_json(**case)
			embeddings = KnowledgeBaseEmbeddings(ontology)                            				 # initializes the class for the embeddings given an Ontology in the memory
			embeddings.create_tfidf_ann_index("./bioel/models/scispacy/kb_paths_scispacy")           # creates the tfidf vectors and the ann indexes
			kb = embeddings.serialized_kb()                                         				 # returns a NamedTuple containing the necessariy paths to the embeddings and the needed dectionnaries in the memory
			
			generator = CandidateGenerator(kb)         # initializes the candidate generator for the serialized KB                                 
			mentions = ["enzyme"]                      # simple example with 1 mention 
			k = 10                                     # number of K nearest neighbors
			candidates = generator(mentions,k)         # calls the candidate generator for this mention
			print(candidates)                          # We display all the top candidates

	def test_entity_linker(self):

		test_cases = [
			{
				"file_path": "./kb_test2.jsonl",
				"name": "test_onto",
			},]

		for case in tqdm(test_cases):
			ontology = BiomedicalOntology.load_json(**case)
			embeddings = KnowledgeBaseEmbeddings(ontology)                            				 # initializes the class for the embeddings given an Ontology in the memory
			embeddings.create_tfidf_ann_index("./bioel/models/scispacy/kb_paths_scispacy")           # creates the tfidf vectors and the ann indexes
			kb = embeddings.serialized_kb()                                         				 # returns a NamedTuple containing the necessariy paths to the embeddings and the needed dectionnaries in the memory

			nlp = spacy.load("en_core_sci_sm")               # Load a Pretrained model "en_core_sci_sm" available at https://github.com/allenai/scispacy/tree/main
															 # this is the small model but one can feel free to use other available models or train its own model 
															 # by using Spacy's specific config files 
			linker = EntityLinker(kb, threshold = 0.6)       # adjusts the similarity score threshold if you don't find any entity for a mention

			doc = nlp("Branching Enzyme")                    # simple example with a 2 words text
			print(doc.ents)

			linker(doc)
			entity = doc.ents[1]           # we pick the second mention token "Enzyme"
			print(entity)                  # returns a tuple containing the identified mentions by the NER
			print(entity._.kb_ents)        # returns a list of tuples containing the cuis and the matching score linked to each mention in the text

			for umls_ent in entity._.kb_ents:
				print(linker.kb.cui_to_entity[umls_ent[0]])           # retrieves all the information about the entity as defined in BiomedicalEntity class given the cui in umls_ent[0]

if __name__ == "__main__":
    unittest.main()