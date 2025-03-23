import pandas as pd
import numpy as np
from decisiontree import DecisionTreeRegressor
from train_test_split import custom_train_test_split
from graphviz import Digraph
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data.xlsx")
df.replace({'TRUE': 1, 'FALSE': 0}, inplace=True)
X_train, X_test, y_train, y_test = custom_train_test_split(df, 'Total.Cup.Points', test_size=0.2, seed=1898)
X_train = X_train.replace({True: 1, False: 0})
X_test = X_test.replace({True: 1, False: 0})
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
# Initialize Decision Tree for Regression

tree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, min_mse_reduction=0.01)

# Train the model
tree.train(X_train, y_train)

print("Feature Importances:", tree.feature_importances)


# Predict on the test set
y_pred = tree.predict(X_test)

# Calculate Mean Squared Error (MSE) as a performance metric
mse = np.mean((y_pred - y_test) ** 2)
rsqure=r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(rsqure)

print(tree.tree)

def visualize_tree(tree, feature_names):
    dot = Digraph()

    def add_nodes_edges(node, parent=None, edge_label=""):
        if node is None:
            return

        # Create a unique ID for each node
        node_id = str(id(node))

        # If it's a leaf node, show the prediction value
        if node.left is None and node.right is None:
            label = f"Predict: {node.prediction_value:.4f}"
        else:
            feature_name = feature_names[node.feature_idx]
            label = f"{feature_name} <= {node.feature_val:.4f}\nMSE Red.: {node.mse_reduction:.4f}"

        dot.node(node_id, label)

        # Connect to the parent
        if parent:
            dot.edge(parent, node_id, label=edge_label)

        # Recurse on left and right children
        add_nodes_edges(node.left, node_id, "True")
        add_nodes_edges(node.right, node_id, "False")

    # Start adding nodes from the root
    add_nodes_edges(tree.tree)

    return dot


feature_names = df.drop('Total.Cup.Points', axis=1).columns.tolist()
dot = visualize_tree(tree, feature_names)
dot.render("decision_tree", format='png', view=True)  # Outputs to decision_tree.png