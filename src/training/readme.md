# Training

- dataset_and_features_loader is also using the Weighted sampling to balance the ratio 1:10 and prevent overfitting

# Usage

#### To create a config

python3 src/training/trainer.py --create-config configs/training_config.yaml

#### To start training

python3 src/training/trainer.py --config configs/training_config.yaml --experiment-name callina-wakeword-detection-v1.0

* The --experiment-name allows me to create multiple version if needed each with different parameters (if I change the config)

#### To resume if training interupted

python3 src/training/trainer.py --config configs/training_config.yaml --resume models/checkpoints/best_model_epoch_050.pt

#### To monitor progress right after the training has begin

tensorboard --logdir logs/training/

# Note

- The trainer.py will actually use values specified in the training_config.yaml instead of some hardcoded values in the loader.py and models.py! This makes it easier to change values and experiment with different settings!