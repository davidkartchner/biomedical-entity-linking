#!/bin/bash

# This script processes the abbreviations from a text file of articles

# Get text of all articles in benchmark
python scripts/extract_document_text.py

# Run Ab3p 
cd Ab3P
./identify_abbr ../data/all_article_text.txt > ../data/raw_abbreviations.txt

# Extract abbreviation dictionary from processed file
cd ..
python scripts/process_abbreviations.py