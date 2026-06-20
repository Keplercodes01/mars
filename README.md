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
- Training: MSE, AdamW, cosine LR decay
- Dataset: CMU subset of AMASS
