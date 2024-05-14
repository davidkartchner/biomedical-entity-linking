from bioel.models.arboel.biencoder.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.crossencoder.data.CrossEncoderLightningDataModule import (
    CrossEncoderDataModule,
)
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)


def main(args):
    data_module = CrossEncoderDataModule(params=args)

    MyModel = LitCrossEncoder.load_from_checkpoint(
        params=args, checkpoint_path=args["crossencoder_checkpoint"]
    )

    trainer = L.Trainer(
        limit_test_batches=1,
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
