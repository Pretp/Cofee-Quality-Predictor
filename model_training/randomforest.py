import numpy as np
from decisiontree import DecisionTreeRegressor

class RandomForestRegressor():
    """
    Random Forest Regressor
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, n_base_learner=10, max_depth=5,min_mse_reduction=0.01, min_samples_leaf=1, \
                 numb_of_features_splitting=None, bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_mse_reduction = min_mse_reduction
        self.min_samples_leaf = min_samples_leaf
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        """
        Creates bootstrap samples for each base learner
        """
        bootstrap_samples_X = []
        bootstrap_samples_Y = []

        for i in range(self.n_base_learner):

            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0]

            sampled_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            bootstrap_samples_X.append(X[sampled_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])

        return bootstrap_samples_X, bootstrap_samples_Y

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """Trains the model with given X and Y datasets"""
        bootstrap_samples_X, bootstrap_samples_Y = self._create_bootstrap_samples(X_train, Y_train)

        self.base_learner_list = []
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTreeRegressor(
                                                    max_depth=self.max_depth,
                                                    min_samples_leaf=self.min_samples_leaf,
                                                    min_mse_reduction=self.min_mse_reduction,
                                                    numb_of_features_splitting=self.numb_of_features_splitting
                                                )

            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)

        # Calculate feature importance
        self.feature_importances = self._calculate_rf_feature_importance(self.base_learner_list)

    def _predict_w_base_learners(self,  X_set: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        pred_list = []
        for base_learner in self.base_learner_list:
            pred_list.append(base_learner.predict(X_set))

        return pred_list

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted values for a given data set"""

        base_learners_preds = self._predict_w_base_learners(X_set)

        # Average the predictions of base learners
        preds = np.mean(base_learners_preds, axis=0)

        return preds

    def _calculate_rf_feature_importance(self, base_learners):
        """Calculates the average feature importance of the base learners"""
        feature_importance_dict_list = []
        for base_learner in base_learners:
            feature_importance_dict_list.append(base_learner.feature_importances)

        feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
        average_feature_importance = np.mean(feature_importance_list, axis=0)

        return average_feature_importance
