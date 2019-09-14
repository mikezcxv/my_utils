import torch
import numpy as np
from sklearn.metrics import mean_squared_error


class SimpleLenDataset(Dataset):
    def __len__(self):
        return len(self.data)


class DatasetCropped(SimpleLenDataset):
    def __init__(self, prepared_df, mode, transform=None, cut=False, zoom=False, rot=False, cut_oversampled=False,
                 zoom_oversampled=False,
                 cut_out=False, resize_size=False):
        #         self.data = pd.read_csv(csv_file)
        self.data = prepared_df
        self.transform = transform
        self.cut = cut
        self.zoom = zoom
        self.rot = rot
        self.mode = mode
        self.cut_oversampled = cut_oversampled
        self.zoom_oversampled = zoom_oversampled
        self.cut_out = cut_out
        self.resize_size = resize_size

    def __getitem__(self, idx):
        img_name = os.path.join('../input', self.data.loc[idx, 'id_code'])
        this_img = self.data.loc[idx]

        if self.mode == 'train':
            #             image = circle_crop2(img_name)
            #             image = Image.fromarray(image)
            image = PIL.Image.open(img_name)
        else:
            image = PIL.Image.open(img_name)

        if self.cut_out:
            j, k, w, h = self.cut_out
            #             print(image.width, image.height)
            if self.resize_size:
                image = cv2.resize(np.array(image), self.resize_size)
            # image = image.transform(self.resize_size, method=PIL.Image.MESH, resample=0, fill=1)

            #             print(image.width, image.height)
            image = np.array(image)
            image[j * h:(j + 1) * h, k * w:(k + 1) * w, :] = [0, 0, 0]
            image = Image.fromarray(image)

        if self.mode == 'train':
            cut_vars, cut_p = [], 0
            z_lower, z_lower_p, z_upper, z_upper_p = False, 0, False, 0
            rot_max_deg, rot_p = 0, 0

            if self.cut: cut_vars, cut_p = self.cut
            if self.zoom: z_lower, z_lower_p, z_upper, z_upper_p = self.zoom
            if self.rot: rot_max_deg, rot_p = self.rot

            if self.rot:
                image = gausian_img_rotation(image, max_deg=rot_max_deg, p=rot_p)
            if this_img['oversampled']:
                if self.zoom_oversampled:
                    z_lower, z_lower_p, z_upper, z_upper_p = self.zoom_oversampled
                if self.cut_oversampled:
                    cut_vars, cut_p = self.cut_oversampled

        this_img = self.data.loc[idx]
        label = torch.tensor(this_img['diagnosis'])
        #         classes_stat = {0: 1.0, 2: 1.4, 1: 3.0, 4: 7.0, 3: 7.3}
        #         classes_weights = {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.5, 4: 2.0}
        classes_weights = {0: 1.0, 1: 1.2, 2: 1.1, 3: 1.2, 4: 2}
        #         0: '0.05 [448]', 1: '0.25 [92]', 2: '0.35 [253]', 3: '0.58 [42]', 4: '2.83 [81]'}
        sample_weights = torch.tensor(this_img[['size_ratio']].apply(
            lambda x: classes_weights[this_img[target]] * (1.05 if x > 1.25 else 1.0)
        ))
        ratio_info = torch.tensor(this_img['size_ratio'])

        image = self.transform(image)
        return (image, label, sample_weights, ratio_info)


# SubsetSampler


class AnalyzeNNPerception():
    """
    Example:
        ap = AnalyzeNNPerception(ids = np.array([907, 948]))
        ap.set_crop_predictions(_bs=32, w=16, h=16, n=1, img_size=224) # valid_idx[fold][:n]
        ap.analyze()
    """
    def __init__(self, ids, transfors):
        self.ids = ids
        self.transfors = transfors

    def set_crop_predictions(self, _bs=32, w=16, h=16, n=1, img_size=224):
        tsfm_train = sellf.transforms.Compose(tfms[0])
        # Img height * width * n images
        results_map = np.zeros((img_size // h, img_size // w, n))  # np.zeros((100, min(len(valid_idx[fold]), n)))
        for j in range(img_size // h):
            for k in range(img_size // w):
                valid_sampler = SubsetSampler(self.ids)
                experiment_dataset = DatasetCropped(df, transform=tsfm_train, cut=CUT, zoom=ZOOM, rot=ROT,
                                                    mode='train',
                                                    zoom_oversampled=ZOOM_OVERSAMPLED,
                                                    cut_oversampled=CUT_OVERSAMPLED,
                                                    cut_out=(j, k, w, h), resize_size=(img_size, img_size))

                data_loader_valid = torch.utils.data.DataLoader(experiment_dataset, batch_size=min(_bs, n),
                                                                num_workers=4, sampler=valid_sampler)
                valid_predictions, valid_labels = predict_valid(model, data_loader_valid, min(_bs, n))
                results_map[j, k, :] = valid_predictions.squeeze()[:n]

        self.results_map = results_map

    def analyze(self):
        valid_tsfm = transforms.Compose(tfms[0][:-1])
        experiment_dataset = RetinopathyDatasetE(df, transform=tsfm_train, cut=CUT, zoom=ZOOM, rot=ROT, mode='train',
                                                 zoom_oversampled=ZOOM_OVERSAMPLED, cut_oversampled=CUT_OVERSAMPLED)
        data_loader_valid = torch.utils.data.DataLoader(experiment_dataset, batch_size=min(_bs, n), num_workers=4,
                                                        sampler=SubsetSampler(self.ids))
        for (w, e, r, t) in data_loader_valid:
            for i in range(n):
                f, ax = plt.subplots(1, 3, figsize=(18, 5))
                ax[0].set_title(valid_labels[i].squeeze())
                ax[0].imshow(transforms.ToPILImage()(w[i]))
                ax[2].hist(self.results_map[:, :, i])
                #         rm[:, :, i] = (rm[:, :, i] - np.mean(rm[:, :, i])) / np.std(rm[:, :, i])
                #         ax[1].imshow(np.stack([np.zeros(rm[:, :, i].shape), rm[:, :, i], np.zeros(rm[:, :, i].shape)], axis=2));
                sns.heatmap(self.results_map[:, :, i], ax=ax[1])
                plt.show()


class AnalyzeResults():
    """
    ar = AnalyzeResults(analyze_df, analyze_df['score'], bound)
    print(ar.confusion_matrix())
    ar.show_top(n=20, mode='most_std')
    """
    read_img_shape = False

    def __init__(self, df, predictions, bound):
        df['predicted'] = res_to_classes(predictions, bound=bound, shift=0).astype(int)
        df['score'] = predictions
        df['diff'] = df['diagnosis'] - df['predicted']
        df['diff_abs'] = np.abs(df['diff'])
        df['dev'] = np.abs(df['diagnosis'] - df['score'])
        #         df[df['diff'] > 0].sort_values('diff', ascending=False)
        df['pairs'] = df.apply(lambda x: str(x['diagnosis']) + '_' + str(x['predicted']), axis=1)
        self.df = df
        #         self.idx2id = dict(zip(valid.index, range(len(valid))))

        if_fix_optimistic = local_quadratic_kappa(df[df['diff'] >= 0]['diagnosis'],
                                                  res_to_classes(df[df['diff'] >= 0]['score'], bound=bound,
                                                                 shift=0).astype(int), N=5)

        print(f"Correctly: {len(df[df['diff'] == 0])}; Optimistic: {len(df[df['diff'] < 0])}; " + \
              f"Pessimistic: {len(df[df['diff'] > 0])} Accuracy: {self.accuracy()} " + \
              f"If fix optimistic: {if_fix_optimistic:.3f}"
              )

    #         self.read_img_shape()
    #         self.collect_models_stat()
    # # Correctly: 776; Optimistic: 55; Pessimistic: 85
    # # Fix pessimistic: 0.967; Fix optimistic: .961

    def accuracy(self):
        return np.round(accuracy_score(self.df['diagnosis'], self.df['predicted']), 3)

    def confusion_matrix(self):
        return confusion_matrix(self.df['diagnosis'], self.df['predicted'])

    def get_single_model_prediction(self, idx, model_idx):
        return float(fold_predictions_valid[:, :n_stop_fold, model_idx].squeeze()[idx])

    def get_single_model_predictions(self, model_idx):
        return float(fold_predictions_valid[:, :n_stop_fold, model_idx].squeeze())

    def get_single_model_prediction_tta(self, idx, model_idx, ver=0):
        return float(fold_predictions_valid_tta[:, :n_stop_fold, model_idx, ver].squeeze()[idx])

    def read_img_shape(self):
        self.df['height'] = self.df.apply(lambda x: np.array(PIL.Image.open('../input/' + x['id_code'])).shape[0],
                                          axis=1)
        self.df['width'] = self.df.apply(lambda x: np.array(PIL.Image.open('../input/' + x['id_code'])).shape[1],
                                         axis=1)
        self.df['ratio'] = self.df['width'] / self.df['height']
        self.df['is_correct'] = self.df['predicted'] == self.df['diagnosis']
        self.read_img_shape = True

    def collect_models_stat(self):
        for point_idx in range(len(checkpoints)):
            tmp = fold_predictions_valid[:, fold, point_idx].squeeze()
            self.df[f'score_model_{point_idx}'] = tmp
            self.df[f'pred_model_{point_idx}'] = res_to_classes(tmp, bound=bound).astype(int)

        self.df['models_score_std'] = self.df[self.df.columns[self.df.columns.str.match('score')]].apply(
            lambda x: np.std(x), axis=1)
        self.df['models_score_range'] = self.df[self.df.columns[self.df.columns.str.match('score')]].apply(
            lambda x: np.max(x) - np.min(x), axis=1)

    def show_top(self, n=3, mode='errors'):
        if mode == 'errors':
            l = self.df[self.df['diff'] != 0].sort_values('diff_abs', ascending=False)
        elif mode == 'correct':
            l = self.df[self.df['diff'] == 0].sort_values('diff_abs', ascending=False)
        elif mode == 'most_std':
            l = self.df.sort_values('models_score_std', ascending=False)

        for i, info in enumerate(l.iterrows()):
            #             item, idx = info[1], self.idx2id[info[0]]
            item = info[1]
            img = PIL.Image.open('../input/' + item['id_code'])

            f, ax = plt.subplots(1, 2, figsize=(12, 5))
            plt.title(f'Real: {item["diagnosis"]} Predicted: {item["predicted"]} Scorre: {item["score"]:.2f}')
            ax[0].imshow(img)
            ax[1].plot(item['score'], 'o', label='Prediction')
            if item["diagnosis"] < len(bound):
                ax[1].axhline(y=bound[item["diagnosis"]], color='r', linestyle='-')
            if 0 <= item["diagnosis"] <= len(bound):
                ax[1].axhline(y=bound[item["diagnosis"] - 1], color='r', linestyle='-')
            # ax[1].hist(score_by_models)

            plt.show() and plt.clf()

            if i + 1 >= n:
                break

    def show_top_errors(self, n=3):
        self.show_top(n=n, mode='errors')

    def show_top_correct(self, n=3):
        self.show_top(n=n, mode='correct')


def get_mse_report(a, b, n_repeat=100):
    r = []
    errors = []
    for i in range(5):
        idx = np.where(b == i)
        err = mean_squared_error(a[idx], b[idx])
        errors.append(err)
        r.append(str(np.round(err, 2)) + f' [{a[idx].shape[0]}]')

    q = []
    for i in range(n_repeat):
        idx = np.random.choice(range(len(a)), int(.9 * len(a)))
        c = mean_squared_error(a[idx], b[idx])
        q.append(c)
    print(dict(zip(range(5), r)))
    print('All', np.round(mean_squared_error(a, b), 3), 'm:',
          np.round(np.mean(q), 3), 'STD: ', np.round(np.std(q), 3),
          )

    return q


class ResultsExplainer:
    def plot_losses_by_epoch(self, epochs=(0, 3), figsize=(12, 6), fold=0):
        plt.figure(figsize=figsize)
        for i in range(epochs):
            plt.plot(train_losses_hist[fold][i], label=f'Epoch {i}');
        plt.legend();

    def get_mse_report(self, a, b, n_repeat=100):
        r = []
        errors = []
        for i in range(5):
            idx = np.where(b == i)
            err = mean_squared_error(a[idx], b[idx])
            errors.append(err)
            r.append(str(np.round(err, 2)) + f' [{a[idx].shape[0]}]')

        q = []
        for i in range(n_repeat):
            idx = np.random.choice(range(len(a)), int(.9 * len(a)))
            c = mean_squared_error(a[idx], b[idx])
            q.append(c)
        print(dict(zip(range(5), r)))
        print('All', np.round(mean_squared_error(a, b), 3), 'm:', np.round(np.mean(q), 3), 'STD: ',
              np.round(np.std(q), 3))

        return q

    def glimpse_predictions_scores(self, n=100):
        plt.figure(figsize=(17, 6))
        plt.plot(list(range(len(fold_predictions_avg[:n]))), fold_predictions_avg[:n]);


def anlyze_APTOS():
    a, b = predict_valid(model, data_loader_valid, batch_size)
    q = ResultsExplainer().get_mse_report(a.squeeze(), b.squeeze(), n_repeat=100)
    plt.hist(q, bins=20);

    from sklearn.linear_model import LinearRegression

    counts = np.array([448, 92, 253, 42, 81])
    # counts = [0.11, 0.13, 0.46, 0.27, 0.04]

    results = np.array([
        [0.04, 0.24, 0.21, 0.25, 1.7, .731],
        [0.04, 0.33, 0.31, 0.25, 1.48, .762],
        [0.05, 0.27, 0.26, 0.29, 2.07, .751],
        [0.06, 0.26, 0.22, 0.32, 1.76, .742],
        [0.07, 0.30, 0.26, 0.32, 1.37, .746],
        [0.05, 0.37, 0.18, 0.4, 1.72, .72],  # 0.516  Kappa nan 0.807 0.941 MSE 0.277 0.674
        [0.11, 0.46, 0.17, 0.2, 1.41, .711],
        # Correctly: 736; Optimistic: 116; Pessimistic: 64 Accuracy: 0.803 If fix optimistic: 0.954 ~ .71
        [0.04, 0.23, 0.25, 0.5, 1.67, .749],
        [0.06, 0.32, 0.2, 0.49, 1.64, .746],
        [0.03, 0.31, 0.22, 0.37, 1.64, .752],
    ])
    r = results.copy()
    results[:, :-1] = np.round(results[:, :-1] * counts / np.sum(counts), 3)

    l = LinearRegression().fit(results[:, :-1], results[:, -1])
    print(l.coef_)

    l.predict(np.array([[0.05, 0.4, 0.19, 0.2, 1.57]]) * counts / np.sum(counts))


def analyze_APTOS2():
    def get_inference_model(point='checkpoint.pt'):
        model = get_model(model_type)
        model.to(device)
        model.load_state_dict(torch.load(point))
        freeze_model(model)
        model.eval()
        return model

    def get_inference_model_cached(model, point='checkpoint.pt'):
        model.load_state_dict(torch.load(point))
        freeze_model(model)
        model.eval()
        return model

    _bs = 16
    #     model = get_inference_model()
    valid_sampler = SubsetSampler(valid_idx[fold])
    data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=_bs, num_workers=4, sampler=valid_sampler)
    valid_predictions, valid_labels = predict_valid(model, data_loader_valid, _bs)
    analyze_df = df.loc[valid_idx[0]]
    analyze_df['labels'] = valid_labels
    analyze_df['score'] = valid_predictions.squeeze()
    analyze_df['predicted'] = res_to_classes(analyze_df['score'], bound=bound, shift=0).astype(int)

    FOR_ALL_CHECKPOINTS = False
    if FOR_ALL_CHECKPOINTS:
        n_checkpoins = 7
        all_preictions = np.zeros((len(valid_labels), n_checkpoins))
        for i in range(1, n_checkpoins + 1):
            point = f'/kaggle/working/checkpoint_epoch_{i}.pt'
            model = get_inference_model_cached(model, point)
            valid_predictions, valid_labels = predict_valid(model, data_loader_valid, _bs)
            all_preictions[:, i - 1] = valid_predictions[:, 0]

        # all_preictions[:, i] = valid_predictions[:, 0]
        # all_preictions
        # valid_predictions
        # np.zeros((len(valid_labels), n_checpoins))
        # all_preictions.shape
        # all_p = all_preictions.copy()
        # all_preictions = np.concatenate((all_p, valid_predictions), axis=1)[:, 1:]

        # for i in range(n_checkpoins):
        #     plt.hist(all_preictions[:, i], bins=10);
        #     plt.show()

        w = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        w = (0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 1)
        w = (0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1)
        w = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)
        w = (0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0)
        # 7:1, 10: 2, 11: 1, 12:1, 13:1
        k = np.average(all_preictions[:, :], weights=w[:n_checkpoins], axis=1)
        k_all = local_quadratic_kappa(valid_labels, res_to_classes(k, bound=bound).astype(int), 5)
        print('MSE:', np.round(mean_squared_error(valid_labels, k), 3), 'Kappa:', np.round(k_all, 3))
