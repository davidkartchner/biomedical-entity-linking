import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_overlap(df, dataset_name=None, plot=False):
    """
    Overlap between train mentions and test mentions.

    Return both the % of entities that overlap and the % of mentions with entities that appear in the train set (since some entities appear more than once)
    """
    # Convert db_ids and type to sets
    df["db_ids_set"] = df["db_ids"].apply(set)
    df["type_set"] = df["type"].apply(set)

    # Get unique entities in data splits
    train_df = df.query('split == "train"')
    test_df = df.query('split == "test"')
    train_valid_df = df.query('split != "test"')

    # Flatten db_ids lists and convert to sets for uniqueness
    unique_train = set().union(*train_df.db_ids_set)
    unique_test = set().union(*test_df.db_ids_set)
    unique_train_valid = set().union(*train_valid_df.db_ids_set)
    unique_ents = set().union(*df.db_ids_set)

    # Flatten type lists for calculations
    all_types = df.explode("type")["type"]
    unique_types = set(all_types.unique())

    # Get total number of unique entities by type
    df_exploded = df.explode("db_ids").explode("type")
    ents_by_type = (
        df_exploded.groupby("type").agg({"db_ids": "nunique"}).to_dict()["db_ids"]
    )
    mention_counts_by_type = (
        df_exploded.groupby("type").agg({"db_ids": "count"}).to_dict()["db_ids"]
    )

    # Get distribution of entities
    if plot:
        test_df["in_train"] = test_df.db_ids_set.apply(lambda x: bool(x & unique_train))
        curie_counts = (
            test_df.explode("db_ids")
            .groupby("db_ids")
            .agg({"mention_id": "count", "in_train": "first"})
            .rename({"mention_id": "cui_num_test_mentions"}, axis=1)
        )
        sns.displot(
            data=curie_counts, x="cui_num_test_mentions", hue="in_train", log_scale=True
        )
        if dataset_name is not None:
            plt.title(dataset_name)
        plt.show()

    # Get overlap of mentions
    train_test_ent_overlap = len(unique_test & unique_train) / len(unique_test)
    test_ent_overlap = len(unique_test & unique_train_valid) / len(unique_test)
    mention_overlap = (
        test_df["db_ids_set"].apply(lambda x: bool(x & unique_train)).mean()
    )

    return {
        "unique_ents": len(unique_ents),
        "ent_overlap": train_test_ent_overlap,
        "mention_overlap": mention_overlap,
        "unique_types": len(unique_types),
        "total_documents": len(df.document_id.unique()),
        "train_documents": len(train_df.document_id.unique()),
        "test_documents": len(test_df.document_id.unique()),
        "total_mentions": df.shape[0],
        "train_mentions": train_df.shape[0],
        "test_mentions": test_df.shape[0],
        "has_validation_set": ("validation" in df.split.unique()),
    }
