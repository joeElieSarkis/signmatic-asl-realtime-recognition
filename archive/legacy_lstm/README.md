# Legacy LSTM Baselines

These scripts are preserved as earlier baseline experiments from the SignMatic thesis development process.

They are not used by the final deployed system. The final model is the Transformer encoder implemented in:

`src/hybrid/train_transformer_model.py`

The final runtime path uses:

`src/hybrid/realtime_hybrid_inference.py` for laptop validation  
`jetson/signmatic_kiosk.py` for Jetson deployment