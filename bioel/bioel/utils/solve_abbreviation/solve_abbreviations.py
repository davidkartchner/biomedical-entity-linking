import subprocess
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from solve_abbreviation.abbreviations_utils import (
    extract_document_text,
    process_abbreviations,
)


# Function to run shell commands
def run_command(command, cwd=None):
    """Executes a shell command in a subprocess, handling exceptions."""
    try:
        subprocess.run(command, shell=True, text=True, cwd=cwd, check=True)
        logging.info(f"Successfully executed command: {command}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}. Error: {e}")
        raise


def create_abbrev(output_dir, all_articles_path, raw_abbreviations_path):

    # Install the packages
    run_command("git clone https://github.com/davidkartchner/Ab3P.git Ab3P")
    run_command("git clone https://github.com/ncbi-nlp/NCBITextLib.git NCBITextLib")
    # # Initialize and update submodules
    # run_command("git submodule init")
    # run_command("git submodule update")

    # Install NCBITextLib
    run_command("make", cwd="NCBITextLib/lib")

    # Build Ab3P and run tests
    run_command("make", cwd="Ab3P")
    run_command("make test", cwd="Ab3P")

    # Get text of all articles in benchmark
    extract_document_text(output_dir)

    all_articles_path = os.path.join(output_dir, "all_article_text.txt")
    raw_abbreviations_path = os.path.join(output_dir, "abbreviations.json")
    # Run Ab3P to detect abbreviations
    run_command(
        f"./identify_abbr {all_articles_path} > {raw_abbreviations_path}",
        cwd="Ab3P",
    )

    # Extract abbreviation dictionary from processed file
    process_abbreviations(output_dir)


if __name__ == "__main__":
    output_dir = "/home2/cye73/data_test2"
    create_abbrev(output_dir)
