#MLDNet: Multi-level Lesion-aware and Detail-injection Network for Polyp Segmentation

## Overview

Automatic polyp segmentation in colonoscopy images is critical for the early diagnosis and treatment of colorectal cancer. However, achieving accurate and robust segmentation is a challenging task due to the significant variations in polyp size, shape, and quantity, particularly in the presence of large, small, or numerous polyps. While recent segmentation methods have achieved notable success, most fail to produce stable results because they do not effectively capture feature interactions across different levels within the image, nor do they integrate both positional and detailed polyp information.

To address these issues, we propose MLDNet (Multi-Level Lesion-aware and Detail-injection Network), a novel approach that introduces a set of specialized modules to enhance feature fusion and segmentation performance.

- Multi-level Position and Detail Fusion (MPDF): Ensures that each feature layer incorporates detailed positional information from all levels, enabling the network to better handle polyp variations.
- Selective Step Feature Aggregation (SSFA): Aggregates features from adjacent layers selectively to refine the feature representation at each level.
- Multi-level Detail Injection (MDI): Performs feature fusion and injects detailed information into the aggregated features, improving the model’s ability to segment complex polyp shapes.

## Usage
### Setup
```
python 3.8
pytorch 1.11.0
cuda 11.3
```

### Downloading necessary data:
- downloading training dataset and move it into ```./data/TrainDataset/```, which can be found in this download link [Google Drive](https://drive.google.com/drive/folders/1hzUdDh157JIb4x8JbkTlzpmrM8EF4Lgk?usp=drive_link). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).
- downloading testing dataset and move it into ```./data/TestDataset/```, which can be found in this download link [Google Drive](https://drive.google.com/drive/folders/1I626JwNulDnHxIBCho2VSU6KvwcxVWHo?usp=drive_link). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
- downloading PVTv2 weights and and move it into ./lib/, which can be found in this download link [Google Drive](https://drive.google.com/file/d/1_VPrmDgz02iBdTAISM2JA7ZWcMf3lIdz/view?usp=drive_link).

### Train or Test
```
cd MLDNet
python train.py
or
python test.py
```

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

## Acknowledgement
Thanks [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of MLDNet.
