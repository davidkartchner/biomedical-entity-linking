import ujson
import sys
import logging
import pandas as pd

from tqdm.auto import tqdm
from collections import defaultdict
from tqdm.auto import tqdm

sys.path.append("..")
from umls_utils import UmlsMappings


logger = logging.getLogger()
logger.setLevel(logging.INFO)

tqdm.pandas()
pd.set_option("display.max_rows", 200)


lowercase = False
for year in [2017, 2022]:


    # 2017 UMLS
    umls = UmlsMappings(
        umls_dir=f"/mitchell/entity-linking/{year}AA/META/",
        debug=False,
        force_reprocess=True,
    )

    # # Mapping of HGNC to Entrez
    # hgnc_to_entrez = {
    #     k: v
    #     for k, v in pd.read_csv(
    #         "../data/proteins.tsv",
    #         delimiter="\t",
    #         names=["source", "target"],
    #     )
    #     .set_index("source")
    #     .to_dict()["target"]
    #     .items()
    #     if k.startswith("hgnc")
    # }

    # entrez_to_hgnc = {
    #     val: key for key, val in hgnc_to_entrez.items() if key.startswith("hgnc")
    # }


    ###############################
    # Mappings Between Ontologies #
    ###############################

    # umls_to_mesh = umls.get_mapping("MSH", other_prefix="MESH")
    # mesh_to_umls = umls.get_mapping("MSH", reverse=True, other_prefix="MESH")
    # # print(len(umls_to_mesh))
    # # print(len(mesh_to_umls))

    # # Write to file
    # with open(f"../data/{year}/umls2mesh.json", "w") as f:
    #     f.write(ujson.dumps(umls_to_mesh))

    # with open(f"../data/{year}/mesh2umls.json", "w") as f:
    #     f.write(ujson.dumps(mesh_to_umls))


    # umls_to_hgnc = {
    #     k: v.replace("HGNC", "hgnc")
    #     for k, v in umls.get_mapping(
    #         "HGNC",
    #         mapping_col="scui",
    #     ).items()
    # }
    # hgnc_to_umls = {
    #     k.replace("HGNC", "hgnc"): v
    #     for k, v in umls.get_mapping("HGNC", mapping_col="scui", reverse=True).items()
    # }
    # umls_to_entrez = {
    #     key: hgnc_to_entrez[val]
    #     for key, val in umls_to_hgnc.items()
    #     if val in hgnc_to_entrez
    # }
    # entrez_to_umls = {v: k for k, v in umls_to_entrez.items()}
    # # print(len(umls_to_hgnc))
    # # print(len(hgnc_to_umls))

    # with open(f"../data/{year}/umls2entrez.json", "w") as f:
    #     f.write(ujson.dumps(umls_to_entrez))

    # with open(f"../data/{year}/entrez2umls.json", "w") as f:
    #     f.write(ujson.dumps(entrez_to_umls))


    # umls_to_ncbi = umls.get_mapping("NCBI", mapping_col="scui", other_prefix="NCBI")
    # ncbi_to_umls = umls.get_mapping(
    #     "NCBI", reverse=True, mapping_col="scui", other_prefix="NCBI"
    # )
    # # print(len(umls_to_ncbi))
    # # print(len(ncbi_to_umls))


    # umls_to_omim = umls.get_mapping("OMIM", other_prefix="OMIM")
    # omim_to_umls = umls.get_mapping("OMIM", reverse=True, other_prefix="OMIM")
    # # print(len(umls_to_omim))
    # # print(len(omim_to_umls))

    # mapping_dicts = [
    #     umls_to_omim,
    #     umls_to_mesh,
    #     umls_to_ncbi,
    #     umls_to_entrez,
    # ]
    # omnimap_with_duplicates = defaultdict(set)
    # for d in mapping_dicts:
    #     for key, val in d.items():
    #         omnimap_with_duplicates[key].add(val)


    # umls_to_mesh_omim = defaultdict(set)
    # for d in [umls_to_mesh, umls_to_omim]:
    #     for key, val in d.items():
    #         umls_to_mesh_omim[key].add(val)

    # omnimap_with_duplicates = {
    #     key: list(val) for key, val in omnimap_with_duplicates.items()
    # }
    # umls_to_mesh_omim = {key: list(val) for key, val in umls_to_mesh_omim.items()}

    # with open(f"../data/{year}/umls2entrez_mesh_omim_ncbi.json", "w") as f:
    #     f.write(ujson.dumps(omnimap_with_duplicates))

    # with open(f"../data/{year}/umls2mesh_omim.json", "w") as f:
    #     f.write(ujson.dumps(umls_to_mesh_omim))


    # ##################
    # # Alias Mappings #
    # ##################

    # Make MeSH Mappings
    print("Getting MeSH Aliases")
    mesh_to_alias = umls.get_aliases(
        ontologies_to_include=["MSH"],
        use_umls_curies=False,
        mapping_cols={"MSH": "sdui"},
        prefixes={"MSH": "MESH"},
        lowercase=lowercase,
    )

    print("Getting MeSH Names")
    mesh_to_name = umls.get_canonical_name(
        ontologies_to_include=["MSH"],
        use_umls_curies=False,
        mapping_cols={"MSH": "sdui"},
        prefixes={"MSH": "MESH"},
        lowercase=lowercase,
    )

    # alias_to_mesh = umls.get_aliases(
    #     ontologies_to_include=["MSH"],
    #     use_umls_curies=False,
    #     mapping_cols={"MSH": "sdui"},
    #     prefixes={"MSH": "MESH"},
    #     reverse=True,
    #     lowercase=lowercase,
    # )


    # mesh_to_alias_chem_only = umls.get_aliases(
    #     ontologies_to_include=["MSH"],
    #     groups_to_include=["CHEM"],
    #     use_umls_curies=False,
    #     mapping_cols={"MSH": "sdui"},
    #     prefixes={"MSH": "MESH"},
    #     lowercase=lowercase,
    # )

    with open(f"../data/{year}/mesh_to_alias.txt", "w") as f:
        f.write(
            "\n".join(
                [
                    curie + "||" + alias
                    for curie, alias_list in mesh_to_alias.items()
                    for alias in alias_list
                ]
            )
        )


    with open(f"../data/{year}/mesh_to_name.json", "w") as f:
        f.write(ujson.dumps(mesh_to_name))

    # with open(f"../data/{year}/mesh_to_alias_chem_only.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [
    #                 curie + "||" + alias
    #                 for curie, alias_list in mesh_to_alias_chem_only.items()
    #                 for alias in alias_list
    #             ]
    #         )
    #     )

    # with open(f"../data/{year}/alias_to_mesh.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [alias + "||" + "|".join(curie) for alias, curie in alias_to_mesh.items()]
    #         )
    #     )


    # # OMIM Mappings
    # omim_to_alias = umls.get_aliases(
    #     ontologies_to_include=["OMIM"],
    #     use_umls_curies=False,
    #     mapping_cols={"OMIM": "sdui"},
    #     prefixes={"OMIM": "OMIM"},
    #     lowercase=lowercase,
    # )

    # alias_to_omim = umls.get_aliases(
    #     ontologies_to_include=["OMIM"],
    #     use_umls_curies=False,
    #     mapping_cols={"OMIM": "sdui"},
    #     prefixes={"OMIM": "OMIM"},
    #     reverse=True,
    #     lowercase=lowercase,
    # )


    # with open(f"../data/{year}/omim_to_alias.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [
    #                 curie + "||" + alias
    #                 for curie, alias_list in omim_to_alias.items()
    #                 for alias in alias_list
    #             ]
    #         )
    #     )

    # with open(f"../data/{year}/alias_to_omim.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [alias + "||" + "|".join(curie) for alias, curie in alias_to_omim.items()]
    #         )
    #     )


    # # Combine MeSH and OMIM for NCBI-Disease corpus
    # alias_to_mesh_and_omim = defaultdict(list)
    # for alias, curie_list in alias_to_mesh.items():
    #     alias_to_mesh_and_omim[alias] = curie_list

    # for alias, curie_list in alias_to_omim.items():
    #     alias_to_mesh_and_omim[alias].extend(curie_list)

    # mesh_and_omim_to_alias_disease_only = umls.get_aliases(
    #     ontologies_to_include=["MSH"],
    #     groups_to_include=["DISO"],
    #     use_umls_curies=False,
    #     mapping_cols={"MSH": "sdui", "OMIM": "sdui"},
    #     prefixes={"MSH": "MESH", "OMIM": "OMIM"},
    #     lowercase=lowercase,
    # )

    # with open(f"../data/{year}/mesh_and_omim_to_alias_disease_only.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [
    #                 curie + "||" + alias
    #                 for curie, alias_list in mesh_and_omim_to_alias_disease_only.items()
    #                 for alias in alias_list
    #             ]
    #         )
    #     )


    # Full UMLS mappings/aliases
    print("UMLS Aliases")
    umls_to_alias = umls.get_aliases(
        ontologies_to_include="all",
        use_umls_curies=True,
        lowercase=lowercase,
    )

    print("UMLS Canonical Names")
    umls_to_name = umls.get_canonical_name(ontologies_to_include="all",
        use_umls_curies=True,
        lowercase=lowercase,)

    with open(f"../data/{year}/umls_to_name.json", "w") as f:
        f.write(ujson.dumps(umls_to_name))

    # alias_to_umls = umls.get_aliases(
    #     ontologies_to_include="all",
    #     use_umls_curies=True,
    #     reverse=True,
    #     lowercase=lowercase,
    # )

    # with open("../data/no_english_filter/umls_to_alias.txt", "w") as f:
    with open(f"../data/{year}/umls_to_alias.txt", "w") as f:
        f.write(
            "\n".join(
                [
                    "UMLS:" + curie + "||" + alias
                    for curie, alias_list in umls_to_alias.items()
                    for alias in alias_list
                ]
            )
        )

    # with open(f"../data/{year}/alias_to_umls.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [alias + "||" + "|".join(curie) for alias, curie in alias_to_umls.items()]
    #         )
    #     )


    # ST21PV mappings/aliases
    st21pv_vocabs = [
        "MSH",
        "CPT",
        "FMA",
        "GO",
        "HGNC",
        "HPO",
        "ICD10",
        "ICD10CM",
        "ICD9CM",
        "MDR",
        "MTH",
        "NCBI",
        "NCI",
        "NDDF",
        # "MED-RT",
        "NDFRT",
        "OMIM",
        "RXNORM",
        "SNOMEDCT_US",
    ]
    unique_vocabs = umls.umls.sab.unique()
    for x in st21pv_vocabs:
        if x not in unique_vocabs:
            print(x)
            print(unique_vocabs)
        assert x in unique_vocabs

    st21pv_types = ujson.load(open("../data/st21pv_subtypes.json", "r"))


    print("Getting ST21PV Aliases")
    st21pv_to_alias = umls.get_aliases(
        ontologies_to_include=st21pv_vocabs,
        types_to_include=st21pv_types,
        use_umls_curies=True,
        lowercase=lowercase,
    )

    # print("UMLS Canonical Names")
    # umls_to_name = umls.get_canonical_name(ontologies_to_include=st21pv_vocabs,
    #     types_to_include=st21pv_types,
    #     use_umls_curies=True,
    #     lowercase=lowercase,)


    # alias_to_st21pv = umls.get_aliases(
    #     ontologies_to_include=st21pv_vocabs,
    #     types_to_include=st21pv_types,
    #     use_umls_curies=True,
    #     reverse=True,
    #     lowercase=lowercase,
    # )


    # with open("../data/no_english_filter/umls_to_alias.txt", "w") as f:
    with open(f"../data/{year}/st21pv_to_alias.txt", "w") as f:
        f.write(
            "\n".join(
                [
                    "UMLS:" + curie + "||" + alias
                    for curie, alias_list in st21pv_to_alias.items()
                    for alias in alias_list
                ]
            )
        )

    # with open(f"../data/{year}/alias_to_st21pv.txt", "w") as f:
    #     f.write(
    #         "\n".join(
    #             [alias + "||" + "|".join(curie) for alias, curie in alias_to_st21pv.items()]
    #         )
    #     )
