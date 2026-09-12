# LoRA Finetuning Implementation for DiffDock-EM

This document summarizes the LoRA (Low-Rank Adaptation) finetuning system implemented for DiffDock-EM with cryo-EM ring features.

## New Files Created

### 1. `models/lora.py` - Complete LoRA Implementation

- **`LoRALinear` class** - Wraps Linear layers with trainable low-rank matrices (B initialized to zero for identity-preserving initialization)
- **`apply_lora_to_model()`** - Applies LoRA to target modules (conv_layers, rec_emb_layers, final_conv, etc.)
- **`freeze_non_lora_params()`** - Freezes all params except LoRA and ring embedding
- **`get_lora_state_dict()`** - Extracts LoRA-only weights for small checkpoint saves
- **`load_lora_state_dict()`** - Loads LoRA weights into a model
- **`merge_lora_weights()`** - Merges LoRA into base weights for faster inference

## Modified Files

### 2. `datasets/constants.py`

Added:
- `AMINO_ACIDS` set - Standard amino acid 3-letter codes
- `NUCLEOTIDES` set - DNA/RNA residue names for ring feature alignment
- `CA_ATOM_INDEX = 1` - Cα index in the 14-atom representation

### 3. `data/ring_features.py`

Added:
- `align_ring_features_to_residues()` - Uses Cα coordinates directly from `all_coords[:, 1, :]` for proteins
- `align_ring_features_to_atoms()` - Direct atom-level alignment for AAModel
- Enhanced `ring_dim` handling for configurable feature channels

### 4. `train_em.py`

Added LoRA support:
- New arguments: `--use_lora`, `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`
- Automatic LoRA application after loading pretrained weights
- Saves both full model and LoRA-only weights
- Enhanced optimizer with separate parameter groups for pretrained, ring embedding, and LoRA params

### 5. `inference_em.py`

Added LoRA loading:
- Auto-detects LoRA from config.yml
- Applies LoRA structure before loading weights
- Supports separate LoRA weights file via `--lora_weights`

## Usage Examples

### Training with LoRA

```bash
python train_em.py \
  --train_split /path/to/train.txt \
  --val_split /path/to/val.txt \
  --data_dir /path/to/processed \
  --ring_features_dir /path/to/ring_features \
  --pretrained_model /path/to/diffdock_weights.pt \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --ring_dim 4 \
  --n_epochs 100 \
  --lr 1e-4 \
  --run_name my_lora_finetune
```

### Inference with LoRA-finetuned Model

```bash
python inference_em.py \
  --model_dir workdir_em/my_lora_finetune \
  --ckpt best_ema_model.pt \
  --protein_path protein.pdb \
  --ligand_description "SMILES" \
  --ring_features_path ring_features.npz \
  --out_dir results
```

### Training without LoRA (Full Finetuning)

```bash
python train_em.py \
  --train_split /path/to/train.txt \
  --val_split /path/to/val.txt \
  --data_dir /path/to/processed \
  --ring_features_dir /path/to/ring_features \
  --pretrained_model /path/to/diffdock_weights.pt \
  --ring_dim 4 \
  --ring_lr_multiplier 10.0 \
  --n_epochs 100 \
  --lr 1e-4 \
  --run_name my_full_finetune
```

## Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| LoRA rank (r) | 8 | Balance between capacity and efficiency |
| LoRA alpha | 16 | Standard scaling factor (alpha/r = 2) |
| Trainable params | ~1-3% | Efficient finetuning with minimal overfitting risk |
| Ring/LoRA LR multiplier | 10x | New parameters need faster learning |
| B matrix init | zeros | Preserves pretrained behavior at initialization |
| Target modules | conv_layers, rec_emb_layers, final_conv, lig_emb_layers | Core model components |

## Configuration

### Default LoRA Configuration

```yaml
# LoRA settings
use_lora: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules: conv_layers,rec_emb_layers,final_conv,lig_emb_layers

# Ring features
ring_dim: 4  # Channels for radii [4, 5, 6, 7] Å

# Training
lr: 1e-4
ring_lr_multiplier: 10.0
n_epochs: 100
batch_size: 8
```

## Output Files

Training saves the following files in the run directory:

| File | Description |
|------|-------------|
| `config.yml` | Full training configuration |
| `best_model.pt` | Best validation loss model (full state dict) |
| `best_ema_model.pt` | Best EMA model weights |
| `best_lora_weights.pt` | LoRA-only weights (if using LoRA) |
| `last_model.pt` | Last epoch checkpoint with optimizer state |
| `epoch{N}_model.pt` | Periodic checkpoints |

## Model Architecture Support

- **CGModel** (Coarse-grained, Cα-based): Ring features injected at residue level
- **AAModel** (All-atom): Ring features injected at atom level via `--all_atoms` flag

## Baseline Comparison Targets

| Metric | Baseline DiffDock | Target with LoRA+Ring |
|--------|-------------------|----------------------|
| Top-1 RMSD < 2Å | 26.7% | >35% |
| Best-of-10 RMSD < 2Å | 40.0% | >50% |
