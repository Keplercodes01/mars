# MARS
Multimodal Action Reasoning System

## Hypothesis
Decoder-only transformer with joint angles as primary token.
Next-frame prediction. Same scaling behavior as LLMs.

## Val Loss vs Layers (CMU Subset, 20000 steps)

| layers | step_500 | step_1000 | step_5000 | step_10000 | step_20000 |
|--------|----------|-----------|-----------|------------|------------|
| 1      | 0.011590 | 0.004736  | 0.000998  | 0.000592   | 0.000499   |
| 2      | 0.008233 | 0.003801  | 0.001015  | 0.000638   | 0.000517   |
| 3      | 0.007526 | 0.003955  | 0.000976  | 0.000639   | 0.000491   |
| 4      | 0.006247 | 0.003567  | 0.001060  | 0.000620   | 0.000470   |
| 5      | 0.006416 | 0.004097  | 0.001067  | 0.000618   | 0.000464   |
| 6      | 0.006667 | 0.003912  | 0.001107  | 0.000643   | 0.000474   |
| 7      | 0.006369 | 0.004393  | 0.001022  | 0.000637   | 0.000464   |
| 8      | 0.006458 | 0.003842  | 0.001137  | 0.000671   | 0.000478   |
| 9      | 0.006503 | 0.004289  | 0.001113  | 0.000682   | 0.000487   |
| 10     | 0.006719 | 0.004854  | 0.001213  | 0.000664   | 0.000478   |

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
