python experiments/inference.py -cfg "baselines/DCRNN/INRIX_SPEED_AZ.py" -ckpt "checkpoints/DCRNN/INRIX_SPEED_AZ_100_12_12/6e543a86b226f7b599c3e16ac373b03a/DCRNN_best_val_MAE.pt" -i "./datasets/raw_data/INRIX_SPEED_AZ/processed/input.csv" -o "out_dcrnn.csv"

python experiments/inference.py -cfg "baselines/GWNet/INRIX_SPEED_AZ.py" -ckpt "checkpoints/GraphWaveNet/INRIX_SPEED_AZ_100_12_12/e852a4f8fd3eeacb331b2c860c420021/GraphWaveNet_best_val_MAE.pt" -i "./datasets/raw_data/INRIX_SPEED_AZ/processed/input.csv" -o "out_gwnet.csv"

python experiments/inference.py -cfg "baselines/STID/INRIX_SPEED_AZ.py" -ckpt "checkpoints/STID/INRIX_SPEED_AZ_100_12_12/089e9aa6fd198ebf7d2accb687f9ec15/STID_best_val_MAE.pt" -i "./datasets/raw_data/INRIX_SPEED_AZ/processed/input.csv" -o "out_stid.csv"