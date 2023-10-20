import os
import datetime
import glob


def make_unique_model_savepath(
    dir,
    model,
    dataset,
    file_ext,
    params=None,
):
    """
    Make unique path to model logs
    """
    # save_dir = os.path.join(dir, model, dataset, str(datetime.date.today()))
    save_dir = os.path.join(dir, dataset)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if params is not None:
        param_str = "-".path.join("_".join(list(x)) for x in params.items())
    else:
        param_str = ""

    save_prefix = os.path.join(save_dir, param_str)
    file_num = len(glob.glob(save_prefix + "*"))

    # Create unique save path
    if param_str == "":
        save_path = save_prefix + f"{file_num}.{file_ext}"
    else:
        save_path = save_prefix + f"_{file_num}.{file_ext}"

    return save_path
