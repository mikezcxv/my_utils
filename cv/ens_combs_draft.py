from sklearn.linear_model import LinearRegression

# np.set_printoptions(linewidth=100)
DEBUG_FN = False
# for model_id in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
n_train = 750
summary = np.zeros((2 ** (len(checkpoints)), len(checkpoints) + 1))
for comb_id in tqdm(range(1, 2 ** (len(checkpoints)))):
    if DEBUG_FN: __print__(comb_id, bin(comb_id), end=':\t\t')
    moldels_sum = 0
    model_avg = 0
    for model_id in range(0, len(checkpoints)):
        if (comb_id & (2 ** model_id) == (2 ** model_id)):
            if DEBUG_FN: __print__(f'{model_id}', end=' ')
            moldels_sum += ar.df[ar.df.ratio > 1.25][f'score_model_{model_id}']
#                 moldels_sum += fold_predictions_valid[idx, 0, model_id]
            model_avg += 1
            summary[comb_id, model_id] = 1
#         summary[comb_id, -1] = np_quadratic_kappa(valid_labels.squeeze()[idx], res_to_classes(moldels_sum / model_avg, bound=bound).astype(int), N=5)
#         summary[comb_id, -1] = np_quadratic_kappa(ar.df[ar.df.ratio > 1.25][target], res_to_classes(moldels_sum / model_avg, bound=bound).astype(int), N=5)
        if model_avg:
            summary[comb_id, -1] = mean_squared_error(ar.df[ar.df.ratio > 1.25][target], moldels_sum / model_avg)
    if DEBUG_FN: print()

reg = LinearRegression().fit(-summary[1:, :-1], summary[1:, -1])
print(np.mean(summary[np.argsort(summary[1:, -1], axis=0)][:15, :].astype(np.float16), axis=0))
print(np.round(reg.coef_ * 1000, 3), list(np.argsort(-reg.coef_)))

#     mini_train = np.average(fold_predictions_valid[:, :n_stop_fold, :], weights=(0, 0, 2, 2, 8, 10, 2, 2, 2, 2), axis=2).squeeze()
#     np_quadratic_kappa(valid_labels.squeeze(), res_to_classes(mini_train, bound=bound).astype(int), N=5)

# 0.9023, 0.9069, 0.9106, 0.9151, 0.9250, 0.9279, 0.9165, 0.9242, 0.9150, 0.9197
# [0.      0.4666  0.06665 0.8667  1.      0.06665 0.2666  0.2666  0.4666 0.06665 0.06665 0.1333  0.8613 ]
# [-0.852  1.015 -0.298  5.697  5.166 -1.267  1.789 -0.444  2.945  1.282  0.05  -0.057] [3, 4, 8, 6, 9, 1, 10, 11, 2, 7, 0, 5]
