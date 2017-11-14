from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from io import StringIO
from sklearn import tree
import numpy as np
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


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