##### Test file that you must put into the "package" folder #####
import spacy
import sys
sys.path.insert(0, "./scispacy")

from ontology import BiomedicalOntology
from candidate_generation import CandidateGenerator
from scispacy_embeddings import KnowledgeBaseEmbeddings
from entity_linking import EntityLinker

## GENERATE THE KNOWLEDGE BASE FROM THE ONTOLOGIES ##

myonto = BiomedicalOntology("test")             # we first initializes our BiomedicalOntology object with a name "test"
myonto.load_json("./umls_mesh_2022.jsonl")      # we load the KB using the path to the ontology json file, here I used Mesh 2022
print("done")

myembed = KnowledgeBaseEmbeddings(myonto)                            # initializes the class for the embeddings given an Ontology in the memory
myembed.create_tfidf_ann_index("scispacy/kb_paths_scispacy")         # creates the tfidf vectors and the ann indexes
kb = myembed.serialized_kb()                                         # returns a NamedTuple containing the necessariy paths to the embeddings and the needed dectionnaries in the memory
#print(kb[5])

### CANDIDATES GENERATION TEST ###

'''
cand_gen = CandidateGenerator(kb)  # initializes the candidate generator for the serialized KB                                 
my_mentions = ["enzyme"]           # simple example with 1 mention 
k = 10                             # number of K nearest neighbors
candidates = cand_gen(my_mentions,k)     # calls the candidate generator for this mention
print(candidates)                         # We display all the top candidates
'''

### ENTITY LINKER TEST ###

nlp = spacy.load("en_core_sci_sm")               # Load a Pretrained model "en_core_sci_sm" available at https://github.com/allenai/scispacy/tree/main
												 # this is the small model but one can feel free to use other available models or train its own model by using Spacy's specific config files 

mylinker = EntityLinker(kb, threshold = 0.6)     # adjusts the similarity score threshold if you don't find any entity for a mention

doc = nlp("Branching Enzyme")      # simple example with a 2 words text
print(doc.ents)

mylinker(doc)
myentity = doc.ents[1]           # we pick the second mention token "Enzyme"
print(myentity)                  # returns a tuple containing the identified mentions by the NER
print(myentity._.kb_ents)        # returns a list of tuples containing the cuis and the matching score linked to each mention in the text

for umls_ent in myentity._.kb_ents:
	print(mylinker.kb.cui_to_entity[umls_ent[0]])           # retrieves all the information about the entity as defined in BiomedicalEntity class given the cui in umls_ent[0]

