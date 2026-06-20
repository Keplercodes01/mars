# MARS
Multimodal Action Reasoning System

## Hypothesis
Decoder-only transformer with joint angles as primary token.
Next-frame prediction. Same scaling behavior as LLMs.

## Results
| layers | steps | val loss |
|--------|-------|----------|
| 1      | 20000 | 0.000335 |

## Architecture
- ActionEncoder: linear projection 156 → 256
- TransformerBlock: causal self-attention + FFN
- ActionDecoder: linear projection 256 → 156

## Hyperparameters
| parameter    | value     |
|--------------|-----------|
| seed         | 42        |
| action_dim   | 156       |
| d_model      | 256       |
| num_heads    | 8         |
| max_seq      | 128       |
| batch_size   | 128       |
| lr           | 1e-4      |
| optimizer    | AdamW     |
| weight_decay | 0.01      |
| grad_clip    | 1.0       |
| lr_schedule  | cosine    |


## Dataset
CMU subset of AMASS — 3.5M frames, 156-dim SMPL joint angles

## Training
MSE loss on next-frame prediction. Causal self-attention.
