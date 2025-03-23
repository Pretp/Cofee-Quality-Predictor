import numpy as np

class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_value, mse_reduction) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val

        # Mean of target values for regression
        self.prediction_value = prediction_value

        # Measure feature importance using MSE reduction
        self.mse_reduction = mse_reduction
        self.feature_importance = self.data.shape[0] * self.mse_reduction

        self.left = None
        self.right = None

    def node_def(self) -> str:
        if self.left or self.right:
            return f"NODE | MSE Reduction = {self.mse_reduction:.4f} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            return f"LEAF | Prediction = {self.prediction_value:.4f}"
