import subprocess
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from bioel.utils.solve_abbreviation.abbreviations_utils import (
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


def clone_or_update_repo(repo_url, directory_name):
    """Clone a repo if it doesn't exist, or pull updates if it does."""
    if not os.path.exists(directory_name):
        run_command(f"git clone {repo_url} {directory_name}")
        print(f"Cloned {directory_name} from {repo_url}")
    else:
        run_command("git pull", cwd=directory_name)
        print(
            f"Updated {directory_name} with the latest changes from the remote repository."
        )


def create_abbrev(output_dir, all_dataset):
    """
    Create abbreviations.json that will contains abbreviations for each document in the specified datas
    ---------
    Parameter
    - output_dir : Path to directory where to save "abbreviations.json" file
    - all_dataset : Datasets for which you want to find abbreviations
    Ex : all_dataset = ["medmentions_full", "bc5cdr", "gnormplus", "ncbi_disease", "nlmchem", "nlm_gene"]
    """

    # Install the packages
    clone_or_update_repo("https://github.com/davidkartchner/Ab3P.git", "Ab3P")
    clone_or_update_repo("https://github.com/ncbi-nlp/NCBITextLib.git", "NCBITextLib")
    # # Initialize and update submodules
    # run_command("git submodule init")
    # run_command("git submodule update")

    # Install NCBITextLib
    run_command("make", cwd="NCBITextLib/lib")

    # Build Ab3P and run tests
    run_command("make", cwd="Ab3P")
    run_command("make test", cwd="Ab3P")

    # Get text of all articles in benchmark
    extract_document_text(output_dir, all_dataset)

    all_articles_path = os.path.join(output_dir, "all_article_text.txt")
    raw_abbreviations_path = os.path.join(output_dir, "raw_abbreviations.txt")
    # Run Ab3P to detect abbreviations
    run_command(
        f"./identify_abbr {all_articles_path} > {raw_abbreviations_path}",
        cwd="Ab3P",
    )

    # Extract abbreviation dictionary from processed file
    process_abbreviations(output_dir, all_dataset)


if __name__ == "__main__":
    all_dataset = [
        "medmentions_full",
        "bc5cdr",
        "gnormplus",
        "ncbi_disease",
        "nlmchem",
        "nlm_gene",
    ]
    output_dir = ""
    create_abbrev(output_dir, all_dataset)
