# Generalization Under Distribution Shift: Augmentation, Domain Adaptation, and Transfer on MNIST

These experiments study how a LeNet classifier behaves when the test distribution differs from the training distribution, and which tools recover the lost accuracy. MNIST digits are rotated by 15° (mild shift) and 90° (severe shift).

Assignment 1 for **Trustworthy Machine Learning** (Dr. Maryam Amirmazlaghani), Amirkabir University of Technology, Spring 2026.

## Notebooks

| Notebook | Topic |
|---|---|
| `01_feature_selection.ipynb` | Feature importance from a decision-tree regressor; all features vs. the single most important one, with and without normalization (house-price data) |
| `02_data_augmentation.ipynb` | LeNet baseline vs. rotation/flip augmentation, autoencoder-based augmentation, and MixUp |
| `03_domain_adaptation.ipynb` | Unsupervised adaptation to rotated MNIST: importance reweighting, DANN (gradient-reversal feature alignment), and CycleGAN domain translation, compared with a supervised upper bound |
| `04_transfer_learning.ipynb` | Transfer to rotated MNIST from only 200 labeled target samples: which layers to freeze vs. training from scratch |

## Results

**Domain adaptation** (`03`): test accuracy on the target domain. The target labels are unseen, except in the upper-bound row.

| Method | Rot15 | Rot90 |
|---|---|---|
| No adaptation | 94.81% | 16.92% |
| Importance reweighting | 95.50% | 16.91% |
| DANN (feature alignment) | **95.91%** | 16.76% |
| CycleGAN translation | 95.77% | 17.29% |
| Supervised on target (upper bound) | 98.63% | 98.53% |

All three methods recover part of the 15° gap. None helps at 90°: when source and target barely overlap, unsupervised alignment fails, and only labeled target data closes the gap. Per-class gains and t-SNE plots are in `figures/`.

![Method comparison on Rot15](figures/3_Method%20Comparison%20on%20Rot15.png)

**Transfer learning** (`04`): 200 labeled Rot15 samples, LeNet pretrained on clean MNIST

| Setting | Trainable layers | Rot15 test accuracy |
|---|---|---|
| Head only | fc1–fc3 | 95.64% |
| Conv2 + head | conv2, fc1–fc3 | 95.27% |
| Full fine-tune | all | **96.10%** |
| From scratch | all, random init | 80.23% |

**Data augmentation** (`02`): clean MNIST test accuracy. The baseline is 98.87%; rotation/flip gives 98.59%, autoencoder augmentation 99.11%, and MixUp 99.07%. Rotation and flip slightly hurt, probably because a horizontally flipped digit is often not a valid digit.

**Feature selection** (`01`): using only `Area` lowers R² from 0.9996 to 0.9986 and nearly doubles RMSE. Normalization has no effect, since tree splits do not depend on feature scale.

The full analysis, including the bias–variance discussion and the written answers, is in `report/report_fa.pdf` (Persian).

## Running

```bash
pip install -r requirements.txt
python -c "from torchvision.datasets import MNIST; MNIST('data', download=True); MNIST('data', train=False, download=True)"
jupyter lab
```

The notebooks read MNIST from `./data` and save figures to `./figures`. `data/house_prices.csv` (provided by the course) is included because it is small.

## License

[MIT](LICENSE)
