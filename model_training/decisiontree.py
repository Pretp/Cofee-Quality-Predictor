import numpy as np
from treenode import TreeNode

class DecisionTreeRegressor():
    def __init__(self, max_depth=4, min_samples_leaf=1, min_mse_reduction=0.0, numb_of_features_splitting=None, feature_importances=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_mse_reduction = min_mse_reduction
        self.numb_of_features_splitting = numb_of_features_splitting
        self.feature_importances = feature_importances if feature_importances is not None else {}

    def _mse(self, values):
        mean = np.mean(values)
        return np.mean((values - mean) ** 2)

    def _split(self, data, feature_idx, feature_val):
        mask = data[:, feature_idx] < feature_val
        return data[mask], data[~mask]

    def _select_features_to_use(self, data):
        feature_idx = list(range(data.shape[1] - 1))

        if self.numb_of_features_splitting == "sqrt":
            return np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))), replace=False)
        elif self.numb_of_features_splitting == "log":
            return np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))), replace=False)
        else:
            return feature_idx
        
    def _calculate_mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _update_feature_importance(self, feature_idx, mse_reduction):
        if feature_idx not in self.feature_importances:
            self.feature_importances[feature_idx] = 0
        self.feature_importances[feature_idx] += mse_reduction

    def _find_best_split(self, data):
        best_split = None
        best_mse_reduction = 0

        X, y = data[:, :-1], data[:, -1]
        current_mse = np.mean((y - np.mean(y)) ** 2)
        n_samples, n_features = X.shape

        # Ensure feature importance exists
        if not hasattr(self, "feature_importances"):
            self.feature_importances = {}

        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])

            # Handle binary features explicitly (True/False or 0/1)
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                thresholds = [0.5]  # Binary split point
            else:
                # For continuous features, sort and check all unique midpoints
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # Skip invalid splits

                left_y, right_y = y[left_mask], y[right_mask]
                left_mse = np.mean((left_y - np.mean(left_y)) ** 2) if left_y.size else 0
                right_mse = np.mean((right_y - np.mean(right_y)) ** 2) if right_y.size else 0

                # Weighted average of MSE
                mse_reduction = current_mse - (len(left_y) * left_mse + len(right_y) * right_mse) / n_samples

                # Track the best split and update feature importance
                if mse_reduction > best_mse_reduction:
                    best_split = (data[left_mask], data[right_mask], feature_idx, threshold)
                    best_mse_reduction = mse_reduction

                    # Update feature importance
                    if feature_idx not in self.feature_importances:
                        self.feature_importances[feature_idx] = 0
                    self.feature_importances[feature_idx] += mse_reduction

        return best_split, best_mse_reduction if best_split else (None, 0)

    def _create_tree(self, data, depth):
        # Base case: Stop if max depth is reached or not enough samples for a valid split
        if depth >= self.max_depth or len(data) < 2 * self.min_samples_leaf:
            return TreeNode(data, None, None, np.mean(data[:, -1]), 0.0)

        # Find the best split
        best_split = self._find_best_split(data)

        if not best_split or best_split[0] is None:
            
            # Return a leaf node if no valid split is found
            return TreeNode(data, None, None, np.mean(data[:, -1]), 0.0)

        (left, right, feature_idx, feature_val), mse_reduction = best_split

        # Stop splitting if the MSE reduction is too small
        if mse_reduction < self.min_mse_reduction:
            return TreeNode(data, None, None, np.mean(data[:, -1]), 0.0)

        # Create a decision node
        node = TreeNode(data, feature_idx, feature_val, np.mean(data[:, -1]), mse_reduction)
        node.left = self._create_tree(left, depth + 1)
        node.right = self._create_tree(right, depth + 1)

        return node

    def _predict_sample(self, sample):
        node = self.tree

        while node.left and node.right:
            if sample[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return node.prediction_value

    def train(self, X_train, y_train):
        # Ensure feature_importances is initialized
        self.feature_importances = {}

        # Combine X and y for easier manipulation
        data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)

        # Build the tree
        self.tree = self._create_tree(data, depth=0)

        # Normalize feature importances after training
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            for feature in self.feature_importances:
                self.feature_importances[feature] /= total_importance

        # Ensure all features are represented (even if not split on)
        for feature_idx in range(X_train.shape[1]):
            if feature_idx not in self.feature_importances:
                self.feature_importances[feature_idx] = 0.0

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def print_tree(self):
        def recurse(node, level=0):
            if node:
                print("    " * level + "-> " + node.node_def())
                recurse(node.left, level + 1)
                recurse(node.right, level + 1)

        recurse(self.tree)
