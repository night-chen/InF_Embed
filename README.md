## IF_Embed
This is the implementation of IF-Embed repository.


### Setup
Create conda environment and install relevant packages:
```bash
# Create a conda environment named 'if_embed' with Python 3.10
conda create -n if_embed python=3.10 -y

# Activate the environment
conda activate if_embed

# Install required Python packages
pip install -r requirements.txt

# Install flash-attn
python -m pip install flash_attn
```


### Train with different configurations:

Modify key configurations in `update_args.py`, you can create a list of sequential training jobs:

```python
experiments = [
        {"model_type": "basic", "model": "Qwen/Qwen2.5-1.5B", "pooling": "last", "share_encoder": True, "num_train_epochs": 2, "contrast_mode": "qk", "data_reverse": False, "padding_side": "left", "train_file": "aarontrinh02/ms_marco_synthetic_data"},
    ]
```
Please refer to `run.py` for detailed hyperparameters.
Use one-line command for running a list of sequential training jobs:
```bash
python update_args.py
```


### Evaluation

For evaluation, we also provide one-line commands for both Bright and MAIR:

```bash
### For Bright
python bright_update_args.py

### For MAIR
python mair_update_args.py
```


### The Loss Map

| `model_type` | `contrast_mode`       | Corresponding Loss                  |
|--------------|-----------------------|-------------------------------------|
| basic        | qk                    | $\ell^{\text{uni}}_{P}$             |
| basic        | kq                    | $\ell^{\text{uni}}_{IQ}$            |
| basic        | only_neg              | $\ell^{\text{uni}}_{I}$             |
| map          | no_trick              | $\ell^{\text{uni}}_{P, IQ}$         |
| map          | qk_with_neg           | $\ell^{\text{uni}}_{P, I}$          |
| map          | kq_with_neg           | $\ell^{\text{uni}}_{I, IQ}$         |
| map          | no_trick_with_neg     | $\ell^{\text{uni}}_{P, I, IQ}$      |
| map_add      | no_trick              | $\ell^{\text{multi}}_{P, IQ}$       |
| map_add      | qk_with_neg           | $\ell^{\text{multi}}_{P, I}$        |
| map_add      | kq_with_neg           | $\ell^{\text{multi}}_{I, IQ}$       |
| map_add      | no_trick_with_neg     | $\ell^{\text{multi}}_{P, I, IQ}$    |

