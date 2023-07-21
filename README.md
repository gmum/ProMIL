# ProMIL: Probabilistic Multiple Instance Learning for Medical Imaging

This repository contains the code for the paper [ProMIL: Probabilistic Multiple Instance Learning for Medical Imaging](https://arxiv.org/abs/2306.10535). The paper was accepted at the conference ECAI 2023.


## How to use this code

Before using the code prepare the data following the instructions from https://github.com/apardyl/ProtoMIL

### Requirements

We use the following packages:

```
- python=3.10
- pytorch 2.0
```

### Running the code

Run specific main file responsible for a given dataset. The model assumes that the representations for patches from big histology datasets (e.g. TCGA NSCLC) are already extracted using pretrained network.


## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{struski2023promil,
  title={ProMIL: Probabilistic Multiple Instance Learning for Medical Imaging},
  author={Struski, {\L}ukasz and Rymarczyk, Dawid and Lewicki, Arkadiusz and Sabiniewicz, Robert and Tabor, Jacek and Zieli{\'n}ski, Bartosz},
  journal={arXiv preprint arXiv:2306.10535},
  year={2023}
}

```
