# MSCS 634 - Lab 2: KNN and Radius Neighbors on the Wine Dataset

**Author:** Umama Azhar
**Notebook:** `Lab2.ipynb`

## Purpose

This lab compares two distance-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the Wine dataset from scikit-learn (178 wines, 13 chemical features, 3 classes). The goal is to see how the main parameter of each model (k for KNN, radius for RNN) affects test accuracy, and to think about when each model is the better choice.

**What the notebook does**
1. Loads and explores the dataset (shape, features, class counts, feature ranges) and makes an 80/20 train/test split (142 training wines, 36 test wines, `random_state=42`).
2. Trains KNN for k = 1, 5, 11, 15, 21 and records test accuracy.
3. Trains RNN for radius = 350, 400, 450, 500, 550, 600 and records test accuracy.
4. Plots accuracy against k and against radius, then compares the two models.

## Results

| KNN (k) | Accuracy | Correct / 36 |
|---|---|---|
| 1 | 0.7778 | 28 |
| 5 | 0.7222 | 26 |
| 11 | 0.7500 | 27 |
| 15 | 0.7500 | 27 |
| 21 | 0.7778 | 28 |

| RNN (radius) | Accuracy | Correct / 36 |
|---|---|---|
| 350 | 0.7500 | 27 |
| 400 to 600 | 0.7222 | 26 |

## Key insights

- **k had only a small effect.** KNN accuracy dipped at k = 5 and recovered by k = 21, with a total spread of 2 wines out of 36. There was no steady upward or downward trend.
- **Radius had almost no effect once it was large enough.** RNN accuracy was 0.7500 at radius 350 and stayed at 0.7222 for every radius from 400 to 600.
- **The two models performed about the same.** The best KNN result (0.7778) and the best RNN result (0.7500) differ by one wine, which is too small a gap to call either model better with only 36 test wines.
- **Overall accuracy (about 72-78%) is moderate.** The features were left unscaled, so `proline` (values up to 1680) dominates the distance calculation and the other features contribute little. I believe this limits accuracy, but I did not test a scaled version in this lab.
- **Why RNN flattened out is a hypothesis.** I suspect larger circles keep adding more of the same neighbors, but I did not verify this by counting the voters in each circle.

## When to prefer each model

- **KNN** is a good default: every test point gets a fixed number of voters, even in sparse regions, and k does not depend on the scale of the features.
- **RNN** fits when a fixed distance has real meaning and neighborhoods should vary in size. Its weakness is that the radius has to match the scale of the data. My radii of 350 to 600 only made sense because raw distances here are in the hundreds.

## Challenges and decisions

- **Left the data unscaled on purpose.** The lab's radius values assume raw distances. Scaling would have shrunk distances so that every radius captured the whole training set.
- **Used `outlier_label="most_frequent"` in RNN.** With a small radius, a test wine can have no training wines inside its circle, which would otherwise raise an error. This option makes the model fall back to the most common training class.
- **Kept the y-axis the same (0.6 to 0.9) on all plots.** A fixed range keeps a 2-wine difference from looking like a huge swing and makes the KNN and RNN plots directly comparable.
- **Small test set.** With 36 test wines, one wine equals about 2.8 percentage points, so I treated small differences cautiously.
- **Notebook run order.** I hit a `NameError` when a cell ran before the train/test split cell had been run. I fixed it by running cells in order and using Restart & Run All to confirm the notebook runs from top to bottom.

## Files

- `Lab2.ipynb`: the Jupyter Notebook with all code, plots, and written observations
- `README.md`: this file

## How to run

1. Install the requirements: `pip install numpy pandas scikit-learn matplotlib jupyter`
2. Open `Lab2.ipynb` in Jupyter and choose **Kernel → Restart & Run All**.
