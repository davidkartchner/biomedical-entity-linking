import argparse

from bioel.models.krissbert.model.model import Krissbert
from bioel.models.krissbert.data.utils import BigBioDataset

def main():
    parser = argparse.ArgumentParser(description="Generate Krissbert prototypes")
    parser.add_argument("--model_name_or_path", type=str,
                        default="microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
                        help="Path to the pre-trained model")
    parser.add_argument("--dataset_name", type=str, default="nlmchem", help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generating prototypes")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default='./prototypes/', help="Directory to save the model")

    args = parser.parse_args()

    model = Krissbert(args.model_name_or_path)
    model.cuda()

    ds = BigBioDataset(args.dataset_name, splits=["train"])
    model.generate_prototypes(ds, args.output_dir, args.batch_size, args.max_length)

    print("Done generating prototypes!")

if __name__ == "__main__":
    main()