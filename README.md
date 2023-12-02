# Capstone-Quarter1

# Training Models

Example Usage
```
python3 xray_main.py
```

Example Custom Usage
```
python3 xray_main.py --config my_new_config.json
```

Output
```
results.csv - Predictions of ln(BNPP + 1) values and actual ln(BNPP + 1) values
Loss_results_plot.png - Training Loss plot and regression plot
log.txt - Outputs of epoch loss for train and validation sets
best_model_params.pt - Saved weights from model with lowest validation loss
```

## Build targets

`all`, `train_model`, and `test_model`

- `all`: trains the model and tests it
- `train_model`: only trains a model without predicting on test dataset
- `test_model`: only predicts on test dataset with provided model weights

ex.

```
python3 xray_main.py test_model --config my_new_config.json
```

Will load the saved_weights from `my_new_config.json` and then predict on the test dataset

## Configuration file
- model: Model to be used. Options are 'resnet' and 'vgg'
- dataloaders:
    - batch_size
    - shuffle:
    - num_workers
    - use_custom_transforms
- training:
    - epochs: Number of Epochs. 
    - criterion: Loss Function. Options are 'MAE' and 'MSE'
    - lr: Learning Rate.
    - use_scheduler: Use Learning Rate Scheduler. 
    - scheduler_step_size: LR scheduler step size. 
    - lr_decay_rate: LR Scheduler decay rate. 
    - use_estop: Use early stopping
    - estop_num_epochs: Number of epochs to wait before early stopping
- filepaths:
    - data_dir_path: path to were the data is
    - hdf5_stem: file name stem
    - train_dataset: filepath to train dataset
    - val_dataset: filepath to val dataset
    - test_dataset: filepath to test dataset
    - results_csv_path: filepath where test dataset predictions are stored
    - saved_weights_path: filepath for where to save/load weights
    - loss_plot_path: filepath to save a training loss plot
    - results_plot_path: filepath to save a regression plot
    - combined_plot_path: filepath to save a training loss and regression plot
