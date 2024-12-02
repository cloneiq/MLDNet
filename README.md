#MLDNet: Multi-level Lesion-aware and Detail-injection Network for Polyp Segmentation

## Overview

Automatic polyp segmentation in colonoscopy images is critical for the early diagnosis and treatment of colorectal cancer. However, achieving accurate and robust segmentation is a challenging task due to the significant variations in polyp size, shape, and quantity, particularly in the presence of large, small, or numerous polyps. While recent segmentation methods have achieved notable success, most fail to produce stable results because they do not effectively capture feature interactions across different levels within the image, nor do they integrate both positional and detailed polyp information.

To address these issues, we propose MLDNet (Multi-Level Lesion-aware and Detail-injection Network), a novel approach that introduces a set of specialized modules to enhance feature fusion and segmentation performance.

- Multi-level Position and Detail Fusion (MPDF): Ensures that each feature layer incorporates detailed positional information from all levels, enabling the network to better handle polyp variations.
- Selective Step Feature Aggregation (SSFA): Aggregates features from adjacent layers selectively to refine the feature representation at each level.
- Multi-level Detail Injection (MDI): Performs feature fusion and injects detailed information into the aggregated features, improving the model’s ability to segment complex polyp shapes.
