import unittest
import numpy as np

from tqdm import tqdm

from bioel.ontology import BiomedicalOntology


class TestOntology(unittest.TestCase):

    def test_medic_loader(self):
        """
        TestCase - 1: Read from the local UMLS Directory and check for data formatting issues in the entities.
        """
        test_cases = [
            {
                "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
                "name": "MEDIC",
                "abbrev": None,
            }
        ]

        for case in tqdm(test_cases):
            ontology = BiomedicalOntology.load_medic(**case)
            print(list(ontology.entities.values())[:5])
            self.check_multiprefix(ontology)
            self.check_unique_cui(ontology)

    def test_mesh_loader(self):
        """
        TestCase - 1: Read from the local UMLS Directory and check for data formatting issues in the entities.
        """
        test_cases = [
            {
                "filepath": "/mitchell/entity-linking/2017AA/META/",
                "name": "MESH",
                "abbrev": None,
            }
        ]

        for case in tqdm(test_cases):
            ontology = BiomedicalOntology.load_mesh(**case)
            print(list(ontology.entities.values())[:5])
            self.check_multiprefix(ontology)
            self.check_unique_cui(ontology)

    def test_umls_loader(self):
        """
        TestCase - 1: Read from the local UMLS Directory and check for data formatting issues in the entities.
        """
        test_cases = [
            {
                "filepath": "/mitchell/entity-linking/2017AA/META/",
                "name": "UMLS",
                "abbrev": None,
            }
        ]

        for case in tqdm(test_cases):
            ontology = BiomedicalOntology.load_umls(**case)
            print(list(ontology.entities.values())[:5])
            self.check_multiprefix(ontology)
            self.check_unique_cui(ontology)

    def test_obo_loader(self):

        test_cases = [
            {
                "filepath": "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo",
                "name": "disease ontology",
                "prefix_to_keep": None,
                "entity_type": "disease",
                "abbrev": "doid",
            },
            {
                "filepath": "http://purl.obolibrary.org/obo/cl.obo",
                "name": "cell ontology",
                "prefix_to_keep": "CL",
                "entity_type": "cell line & type",
                "abbrev": "cl",
            },
            # {'filepath': 'http://purl.obolibrary.org/obo/uberon.obo',
            # 'name': 'uberon',
            # 'prefix_to_keep': 'UBERON',
            # 'entity_type': 'tissue',
            # 'abbrev': 'uberon'},
            {
                "filepath": "http://purl.obolibrary.org/obo/obi.obo",
                "name": "ontology of biological investigations",
                "prefix_to_keep": "OBI",
                "entity_type": "assay",
                "abbrev": "obi",
            },
            {
                "filepath": "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo",
                "name": "cellosaurus",
                "prefix_to_keep": None,
                "entity_type": "cell line & type",
                "abbrev": "cvcl",
            },
            # {'filepath': 'http://purl.obolibrary.org/obo/go.obo',
            # 'name': 'gene ontology',
            # 'prefix_to_keep': None,
            # 'entity_type': 'subcellular',
            # 'abbrev': 'go'},
        ]

        for case in tqdm(test_cases):
            ontology = BiomedicalOntology.load_obo(**case)
            print(list(ontology.entities.values())[:5])
            self.check_multiprefix(ontology)
            self.check_unique_cui(ontology)

    def check_multiprefix(self, ontology: BiomedicalOntology):
        """
        Make each entity has a single prefix
        """
        error_raised = False
        errors = []
        for cui, entity in ontology.entities.items():
            if len(entity.cui.split(":")) > 2:
                raise AssertionError(
                    f"Entities should only have one prefix, but the entity {entity.cui} has more than 1"
                )

    def check_unique_cui(self, ontology: BiomedicalOntology):
        """
        Make sure all CUIs in ontology are unique
        """
        cuis = [entity.cui for cui, entity in ontology.entities.items()]
        if len(cuis) != len(set(cuis)):
            raise AssertionError("Ontology contains duplicate CUIs")


if __name__ == "__main__":
    unittest.main()
