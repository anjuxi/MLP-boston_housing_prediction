# Fully Connected Feedforward Network — Boston Housing Price Prediction

This project uses a multi-layer fully connected neural network to predict Boston housing prices. It's a regression task on the classic Boston Housing dataset, which has 13 features and a target of median home value.

Dataset: `https://www.kaggle.com/datasets/altavish/boston-housing-dataset`

## Project Highlights

- **Framework**: PyTorch
- **Multi-layer network** with BatchNorm and Dropout to improve generalization
- **Visualization**: feature histograms, correlation heatmap, prediction scatter plot, residual plots
- **Early stopping** to prevent overfitting, with automatic best-model saving
- **Learning rate decay** for adaptive optimization
- **Resume training** from an existing checkpoint

## Dataset

**Boston Housing** has 506 records and 13 features:

| Feature | Description |
| -------- | -------------------------------------------------- |
| CRIM | Per-capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq. ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river, else 0) |
| NOX | Nitric oxide concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Weighted distance to five Boston employment centers |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents |
| LSTAT | Percentage of lower-status population |
| **MEDV** | **Median value of owner-occupied homes (thousands of dollars) — target** |

## Files

| File | Description |
| ------------------------------ | ------------------------------------- |
| `house_price_prediction.ipynb` | Notebook source, good for learning and debugging |
| `HousingData.csv` | Boston Housing dataset |
| `house_price_model.pth` | Trained model weights |

## Usage

```bash
jupyter notebook house_price_prediction.ipynb
```

Run the cells in order.

## Model Architecture

```
Input (13) → Hidden 1 (128) → Hidden 2 (64) → Hidden 3 (32) → Output (1)
                ↓               ↓               ↓
            BatchNorm       BatchNorm       BatchNorm
            ReLU            ReLU            ReLU
            Dropout(0.2)    Dropout(0.2)    Dropout(0.2)
```

**Total parameters**: about 12,609

## Training Parameters

| Parameter | Value | Description |
| ----------------------- | ----- | ------------ |
| `BATCH_SIZE` | 256 | Batch size |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `EPOCHS` | 300 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | 30 | Early-stopping patience |
| `TEST_SIZE` | 0.2 | Test set fraction |

After training, the notebook generates these plots:

1. **Feature histograms** — distribution of each feature
2. **Correlation heatmap** — correlation between features
3. **Prediction scatter plot** — actual vs. predicted
4. **Residual plots** — residual distribution and residuals vs. predictions

## Evaluation Metrics

- **MSE** (mean squared error)
- **RMSE** (root mean squared error)
- **MAE** (mean absolute error)
- **R² score**

## Resume Training

To continue from an existing checkpoint, change the flag in the code:

```python
CONTINUE_TRAIN = True  # set to True
```

The script will load `house_price_model.pth` and continue training.

## Single Prediction Example

```python
# Pass 13 feature values
price = predict_single(model, 0.00632, 18.0, 2.31, 0, 0.538, 6.575,
                       65.2, 4.09, 1, 296, 15.3, 396.9, 4.98)
print(f"Predicted price: {price:.2f} thousand dollars")
```

## Contact

**Dr. Ailan Anjuxi (谙弆悕)**

- Email: anjuxi.ME@outlook.com
- SIP: sip:anjuxi@sip.linphone.org
