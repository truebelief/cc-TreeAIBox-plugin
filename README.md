# <img src="./img/logo.png" width="30" alt="GitHub Logo"> cc-TreeAIBox-plugin

AI-Enhanced Toolset for 3D Tree Processing: A CloudCompare Plugin (alpha 0.1)

This is a Python Plugin for CloudCompare. You can register the python file `TreeAIBox.py` as the main program. This is just a fresh prototype with bugs and imperfections. I will release a stable version after enough tests. The code format also needs standardization.

**Note:** Tested only in Windows OS, RTX3090 (VRAM: 24GB) With CUDA; I am currently making the TreeAIBox installable akin to the 3DFin plugin, thanks to Romain Janvier's suggestion!

<div align="center">
  <img src="https://github.com/user-attachments/assets/dcf4e7de-adea-493f-9fff-04e082efa2d1" width="30%" alt="20240822_TreeAIBoxQSM">
</div>

## Requirements

There are several Python library dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-image==0.22.0 timm numpy_groupies numpy_indexed scikit-learn
```

Make sure the scikit-image version is correct. You can either use the Python package manager of the CloudCompare, or navigate to the default python interpreter of the CloudCompare and run the pip install commands via CLI.

> **Important:** If you have installed CloudCompare to Program Files, there might be issues with pip being prohibited from installing to the default site-package folder. Please run pip command or launch the CloudCompare as the administrator.

I avoid using excessive dependencies unless speed optimization is essential. Simplicity is an art.

## Basic Modules (Updated Aug 22, 2024)

### WoodCls
![image](https://github.com/user-attachments/assets/d5df6a25-bd46-4e4f-a735-73d729957a76)
This module classifies the stem and branch components from a tree scan based on the 3D SegFormer deep learning model.

- Please download the model before applying the classification.
- The classification results will be presented as scalar fields of `stemcls` and `branchcls`.
- More trained AI models will be added in the future.

### QSM
![image](https://github.com/user-attachments/assets/5513699e-56e7-4f2a-a25d-3b33f8a74fea)
This module reconstructs wood architecture from a tree scan. There's much space for improvement.

- The initial segmentation isolates point clusters.
- Stem and branch have customizable scale parameters (MeanShift bandwidth).
- Clusters should be large enough to cover the stem cross-section.
- After skeletonization:
  - Wood architecture will be saved as an XML file
  - Meshes will be saved as an OBJ file
  - Skeleton curves are drawn on CloudCompare

Future upgrades may include an AI-based version of this tool.

## Known Issues

1. **Occasionally flashing:** Sometimes the program will flash due to the computation overhead.

2. **Curve issues:**
   - The Bezier curve is overly bent;
   - The node connection sometimes is poor. The current version of my QSM is sensitive to the point gaps;
   - The joints between two branches sometimes do not follow the point cloud forms
   
3. **Short stems and branches:** The final curves and meshes do not fully capture the length of stems and branches. This is because centroids from initial segments were used as nodes, whereas branch tips need to be added as tree nodes.

4. **Mesh visualization:** Need to visualize the meshes directly, instead of saving and loading the meshes.

5. **Performance on certain tree types:** Current algorithm works poorly on branchy and fat trees (tropical trees).


<video src='https://github.com/user-attachments/assets/bf5f7b6a-5a50-43ba-876b-29e5c9cbff03' width=180/>

