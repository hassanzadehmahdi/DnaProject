Final Recommendation
🔥 Use ProtT5 for per-residue embeddings — especially for your case: small data, imbalanced classes, and token-level prediction.

Later, you can experiment with ESM2 if you:

Have more compute (ESM2 is heavier)

Want to combine sequence + structure features


🧪 (Optional) Stage 6: Enhancements
Use CRF on classifier output

Use data augmentation (sliding window, masking)

Ensemble multiple models

Clean noisy labels using self-training or filtering