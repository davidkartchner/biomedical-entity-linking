conCUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=nlm_gene train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=gnormplus train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=nlmchem train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=bc5cdr train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=ncbi_disease train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=medmentions_full train_data.splits=[train]

CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=medmentions_st21pv train_data.splits=[train]