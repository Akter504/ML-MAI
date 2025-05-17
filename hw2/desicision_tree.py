import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    # print("DEBUG: feature_vector =", feature_vector)
    # print("DEBUG: type(feature_vector) =", type(feature_vector))

    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    diffs = np.diff(feature_sorted)
    thresholds = (feature_sorted[:-1] + feature_sorted[1:]) / 2
    valid = diffs != 0

    thresholds = thresholds[valid]
    if not isinstance(thresholds, np.ndarray) or thresholds.size == 0:
        return np.array([]), np.array([]), None, None
    total_count = len(target_sorted)
    total_pos = np.sum(target_sorted)
    total_neg = total_count - total_pos

    left_counts = np.cumsum(target_sorted[:-1])[valid]
    right_counts = total_pos - left_counts

    left_sizes = np.cumsum(np.ones_like(target_sorted[:-1]))[valid]
    right_sizes = total_count - left_sizes

    p1_left = left_counts / left_sizes
    p0_left = 1 - p1_left
    h_left = 1 - p1_left**2 - p0_left**2

    p1_right = right_counts / right_sizes
    p0_right = 1 - p1_right
    h_right = 1 - p1_right**2 - p0_right**2

    ginis = - (left_sizes / total_count) * h_left - (right_sizes / total_count) * h_right
    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth = 0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_ratio = sorted(ratio.items(), key=lambda x: x[1])
                sorted_categories = list(map(lambda x: x[0], sorted_ratio))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if gini is None or threshold is None:
              continue

            temp_split = feature_vector < threshold
            left_size = np.sum(temp_split)
            right_size = len(temp_split) - left_size

            if self._min_samples_leaf is not None and (
                left_size < self._min_samples_leaf or right_size < self._min_samples_leaf
            ):
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
      if node["type"] == "terminal":
          return node["class"]

      feature = node["feature_split"]
      feature_type = self._feature_types[feature]

      if feature_type == "real":
          if x[feature] < node["threshold"]:
              return self._predict_node(x, node["left_child"])
          else:
              return self._predict_node(x, node["right_child"])

      elif feature_type == "categorical":
          if x[feature] in node["categories_split"]:
              return self._predict_node(x, node["left_child"])
          else:
              return self._predict_node(x, node["right_child"])
      else:
          raise ValueError("Unknown feature type")
    
    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def fit(self, X, y):
        self._tree = {}
        X = np.array(X)
        y = np.array(y)
        self._fit_node(X, y, self._tree)
    

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)