from bioel.models.arboel.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.crossencoder.my_train_cross import (
    CrossEncoderDataModule,
    CrossEncoder,
)


def main(args):
    data_module = CrossEncoderDataModule(params=args)

    MyModel = CrossEncoder.load_from_checkpoint(
        params=args, checkpoint_path=args["model_checkpoint"]
    )

    trainer = L.Trainer(
        # limit_test_batches=4,
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.test(model=MyModel, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
