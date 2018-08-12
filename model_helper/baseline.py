from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from io import StringIO
import numpy as np
from IPython.display import Image
import pydotplus
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold


def describe_tree_txt(estimator, df_test):
    # node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    # leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     df_test.columns[feature[i]],
                     threshold[i],
                     children_right[i],
                     ))


def get_default_estimator(df_train, target_train):
    estimator = DecisionTreeRegressor(max_depth=5, min_samples_split=30, min_samples_leaf=20)
    cv = cross_val_score(estimator, df_train, target_train, cv=5)

    print(cv)

    estimator.fit(df_train, target_train)
    return estimator


def show_tree(estimator, df_columns):
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=df_columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())


def print_graph(clf, feature_names):
    """Print decision tree.
       Source: https://www.dataquest.io/blog/introduction-to-ensembles/?utm_campaign=Data%2BElixir&utm_medium=web&utm_source=Data_Elixir_165
    """
    graph = export_graphviz(
        clf,
        label="root",
        proportion=True,
        impurity=False,
        out_file=None,
        feature_names=feature_names,
        class_names={0: "D", 1: "R"},
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(graph)
    return Image(graph.create_png())


def get_d_tree_baseline(df_train, df_test, y_train, y_test, max_depth, fill_na):
    """
    # Usage:
    :return:
    """
    t1 = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    x_train = df_train.copy()
    x_train.fillna(fill_na, inplace=True)
    x_test = df_test.copy()
    x_test.fillna(fill_na, inplace=True)

    t1.fit(x_train, y_train)
    p = t1.predict_proba(x_test)[:, 1]

    print("Decision tree ROC-AUC score: %.3f" % roc_auc_score(y_test, p))
    print_graph(t1, x_test.columns)


def get_d_tree_baseline_multiple(df_train, df_test, y_train, y_test,
                                 max_depths=[1, 2, 3, 4, 5, 6, 7],
                                 fill_nas=[-999999, -1, 0, 'mean', 'median', 10000000]):
    for max_depth in max_depths:
        print('Depth: ' + str(max_depth))
        for fill_na in fill_nas:
            t1 = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            x_train = df_train.copy()

            if fill_na == 'mean':
                x_train.fillna(x_train.mean(), inplace=True)
            elif fill_na == 'median':
                x_train.fillna(x_train.median(), inplace=True)
            else:
                x_train.fillna(fill_na, inplace=True)

            x_test = df_test.copy()

            if fill_na == 'mean':
                x_test.fillna(x_test.mean(), inplace=True)
            elif fill_na == 'median':
                x_test.fillna(x_test.median(), inplace=True)
            else:
                x_test.fillna(fill_na, inplace=True)

            t1.fit(x_train, y_train)
            p = t1.predict_proba(x_test)[:, 1]

            print("%.3f ROC-AUC %s" % (roc_auc_score(y_test, p),
                                       'NA replaced with: ' + str(fill_na)))
            print_graph(t1, x_test.columns)


def get_logistic_baseline_multiple(df_train, df_test, y_train, y_test,
                                   _cs=[1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3],
                                   fill_nas=[-999999, -1, 0, 'mean', 'median', 10000000]):
    print('\t'.join(['C', 'AUC', 'loss', 'mean repl.']))
    for _C in _cs:
        print(str(_C), end='\t')
        for i, fill_na in enumerate(fill_nas):
            if i > 0:
                print('\t', end='')

            clf = LogisticRegression(C=_C, max_iter=100, random_state=42)

            x_train = df_train.copy()

            if fill_na == 'mean':
                x_train.fillna(x_train.mean(), inplace=True)
            elif fill_na == 'median':
                x_train.fillna(x_train.median(), inplace=True)
            else:
                x_train.fillna(fill_na, inplace=True)

            x_test = df_test.copy()

            if fill_na == 'mean':
                x_test.fillna(x_test.mean(), inplace=True)
            elif fill_na == 'median':
                x_test.fillna(x_test.median(), inplace=True)
            else:
                x_test.fillna(fill_na, inplace=True)

            clf.fit(x_train, y_train)
            p = clf.predict_proba(x_test)[:, 1]

            print("%.2f\t%.4f\t%s" % (roc_auc_score(y_test, p) * 100,
                                      round(log_loss(y_test, p), 4),
                                      str(fill_na)))
        print('\n', end='')


def get_cross_val_probability(clf, X, y, n_splits=5, shuffle=False, is_stratified=False, random_state=42):
    if is_stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    return cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]


# export_graphviz(dtreg, out_file='data/tree.dot')

# dot_data = StringIO()
# export_graphviz(dtreg, out_file=dot_data)

# print(dot_data.getvalue())
# pydot.graph_from_dot_data(dot_data.getvalue()).write_pdf("data/pydot_try.pdf")

# plt.hist(target_train)

# X_test = df_test_copy.iloc[:100, :]

# tree.export_graphviz(estimator, out_file='data/tree.dot')

# sample_id = 0
# node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                     node_indicator.indptr[sample_id + 1]]

# print('Rules used to predict sample %s: ' % sample_id)
# for node_id in node_index:
#     if leave_id[sample_id] != node_id:
#         continue

#     if (X_test.values[sample_id, feature[i]] <= threshold[node_id]):
#         threshold_sign = "<="
#     else:
#         threshold_sign = ">"

#     print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
#           % (node_id, sample_id, feature[node_id], X_test[sample_id, feature[node_id]],
#              threshold_sign, threshold[node_id]))

# For a group of samples, we have the following common node.
# sample_ids = [0, 1]
# common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids))

# common_node_id = np.arange(n_nodes)[common_nodes]

# print("\nThe following samples %s share the node %s in the tree"
#       % (sample_ids, common_node_id))
# print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
