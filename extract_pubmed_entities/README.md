## Extract Self-Supervised Biomedical Entity Mentions from Pubmed XML Files

### Usage

1. Install modified `pubmed_xml`: 
    ```bash
    cd pubmed_xml_modified
    pip install .
    cd ..
    ```

1. Build Trie for efficiently searchign mentions in corpus. Use `build-trie-demo.ipynb` for reference.

1. Extract mentions from Pubmed XML files:
    ```bash
    python extract_pubmed_entities.py
    ```
    Set appropriate paths in the script before running it.