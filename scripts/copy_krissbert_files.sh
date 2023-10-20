for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full" 
do
    cp krissbert/usage/model_output/no_abbr_res/$DATASET.json results/krissbert/no_abbr_res/$DATASET.json
    cp krissbert/usage/model_output/with_abbr_res/$DATASET.json results/krissbert/with_abbr_res/$DATASET.json
done