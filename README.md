# DSCH-Net - IEEE Transactions on Geoscience and Remote Sensing (TGRS)

Implementation of the paper [DSCH-Net: Diffusion-State-Contextual Hybrid Network for Physics-Inspired and Direction-Aware Dehazing of Remote Sensing Imagery](https://ieeexplore.ieee.org/document/11184784/), published in IEEE Transactions on Geoscience and Remote Sensing (TGRS).

Atmospheric haze degrades remote sensing images by reducing contrast, distorting colors, and obscuring fine details, which negatively impacts downstream tasks. DSCH-Net is proposed to address these challenges by employing a physics-inspired, direction-aware deep learning framework for single-image dehazing.

The model integrates a PDE-based diffusion block to suppress haze while preserving structural edges. A multi-directional state-space module efficiently captures long-range spatial dependencies with linear complexity.
Additionally, a multi-dilation residual block enhances fine textures and small structures. A selective fusion gating mechanism stabilizes feature fusion and reduces halo artifacts. The architecture follows an encoder–decoder design for effective feature learning and reconstruction.

Extensive experiments demonstrate strong performance across multiple remote sensing datasets. DSCH-Net achieves up to 32.20 dB PSNR and 0.980 SSIM, outperforming existing methods. Overall, it provides an efficient, robust, and scalable solution for real-world remote sensing dehazing applications.

# Coding Hierarchy
```bash
DSCH-Net/
│── README.md  
│── requirements.txt
│── Dataset/
│   ├── RSID
│   ├── RICE1/2
│   ├── Haze1K
│   ├── DHID
│── Pre_trained/
│   ├── Model
│── Output
│   ├── dehazed_images
│── Train.py
│── Test.py
│── Arguments.py
│── Data_utils.py
│── Metrices.py
│── Loss.py
│── DSCH_Net.py
│── LICENSE
│── CITATION.cff  # Citation info
```


# Environment and Dependencies
```bash
# create a new environment
conda create -n dsch-net python=3.10

# activate environment
conda activate dsch-net

# install dependencies
pip install -r requirements.txt
```

#Test
```bash
python test.py 

#Train model
bash
python train.py 
```
