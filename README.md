# Hidden Interaction Tensor Factorization (IJCAI-18)

This repo contains the PyTorch implementation of the paper `Joint Learning of Phenotypes and Diagnosis-Medication Correspondence via Hidden Interaction Tensor Factorization` in IJCAI-18. [[paper]](http://www.ijcai.org/proceedings/2018/504) [[dataset]](https://mimic.physionet.org/)

## Requirements
The codes have been tested with the following packages:
- Python 3.6  
- PyTorch 0.4.1  

## Quick Demo
To run the model with a quick demo data, simply clone the repo and decompress the data archive by executing the following commands:
```bash
git clone git@github.com:jakeykj/hitf.git
cd hitf
tar -xzvf demo_data.tar.gz
python train.py ./demo_data/
```
A folder `./results/` will be automatically created and the results will be saved there.

## Data Format and Organization
The data are stored in three seperate files contained in a folder (we refer to its path by `<DATA_PATH>`): `<DATA_PATH>/D.csv`, `<DATA_PATH>/M.csv` and `<DATA_PATH>/labels.csv`.
- **`D.csv`** and **`M.csv`**: contain the patient-by-diagnosis binary matrix and the patient-by-medication counting matrix, respectively. These two files should be comma seperated.
- **`labels.csv`**: contains the binary label information for each patient in each line. The order of patients must be aligned with the two above matrices.

If you use other datasets, you can organize the input data in the same format described above, and pass the `<DATA_PATH>` as a parameter to the training script:
```bash
python train.py <DATA_PATH>
```


## Citation
If you find the paper or the implementation helpful, please cite the following paper:
```bib
@inproceedings{yin2018joint,
  title={Joint learning of phenotypes and diagnosis-medication correspondence via hidden interaction tensor factorization},
  author={Yin, Kejing and Cheung, William K and Liu, Yang and Fung, Benjamin C. M. and Poon, Jonathan},
  booktitle={Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence ({IJCAI-18})},
  pages={3627--3633},
  year={2018},
  organization={AAAI Press}
}
```

## Contact
For relevant enquires, please contact Mr. Kejing Yin by email: cskjyin [AT] comp [DOT] hkbu.edu.hk  

---
:point_right: Check out [my home page](https://kejing.me) for more research work by us.
