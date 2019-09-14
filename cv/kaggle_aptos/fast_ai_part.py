# Fast ai section
def _get_preds(learn, ds_type):
    p, _ = learn.get_preds(ds_type=ds_type, pbar=None)
    return to_np(to_cpu(p).squeeze())


def lr_find(learn, start_lr=None):
    if start_lr is None:
        learn.lr_find()
    else:
        learn.lr_find(start_lr=start_lr)
    learn.recorder.plot(suggestion=True)


def lr_plot(learn):
    learn.recorder.plot_lr()
    learn.recorder.plot_losses()


IS_ONLY_INFERENCE = False

#### Hyperparams
IMG_SIZE = 224 # 280 # IMG_SIZE1 = 128 # IMG_SIZE2 = 224 # IMG_SIZE3 = 299 # 320 with lower bs
BS = 16 # 8 ?32
N_FOLDS = 5 # 5 VALID_PERCENT=0.2
# IS_NORMALIZE = True

df = df_source.sample(frac=1, random_state=42).reset_index()[['id_code', 'diagnosis']]

IS_DEBUG = False
if IS_DEBUG:
#     df = df.sample(frac=.2, random_state=42)
#     IMG_SIZE = int(2 * (224 / 3))
#     N_FOLDS = 3
#     !cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth
#     IMG_SIZE = 74
#     df, df_test = df[:20 * BS], df_test[:2 * BS]
    df, df_test = df[:len(df) // 4], df_test[:2 * BS]
    print(f'[IS_DEBUG] Using less data: train {df.shape} test {df_test.shape}')

folds = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED)

# folds = KFold(n_splits=N_FOLDS, random_state=SEED)
for fold, (trn_idx, val_idx) in enumerate(folds.split(df, df[target])):
    train, valid = df.loc[trn_idx], df.loc[val_idx]
    res[fold]['val'], res[fold]['train'], res[fold]['test'] = valid, df.copy(), df_test.copy()

#     tfms = None
#     my_tfms = ([RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'zeros'}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                 RandTransform(tfm=TfmAffine (dihedral_affine), kwargs={}, p=.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                 #  RandTransform(tfm=TfmPixel (flip_lr), kwargs={}, p=0.5, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                  RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
# #                  RandTransform(tfm=TfmAffine (rotate), kwargs={'degrees': (-180, 180)}, p=0.5, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                 #  RandTransform(tfm=TfmAffine(_rot90_affine), kwargs={}, p=0.5, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                 RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.4), 'row_pct': (0, 1), 'col_pct': (0, 1)}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                  RandTransform(tfm=TfmLighting(brightness), kwargs={'change': (0.4, 0.6)}, p=0.1, resolved={}, do_run=True, is_random=True, use_on_y=True),
#                 #  RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.8, 1.25)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True)
#             ],
#             [RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)]
#     )

    # Try to disable symmetric warp
    # my_tfms = get_transforms(flip_vert=True, max_rotate=45, max_warp=None, p_affine=0, max_zoom=1.1)
    my_tfms = ([RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'zeros'}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
                RandTransform(tfm=TfmAffine (dihedral_affine), kwargs={}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
                RandTransform(tfm=TfmLighting(brightness), kwargs={'change': (0.4, 0.6)}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
                #                 RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
                RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.4), 'row_pct': (0, 1), 'col_pct': (0, 1)}, p=0.3, resolved={}, do_run=True, is_random=True, use_on_y=True),
                RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.2, 0.5)}, p=0.1, resolved={}, do_run=True, is_random=True, use_on_y=True)
              ], [RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)]
    )

    src = (ImageList.from_df(df, path=TRAIN_IMG_FOLDER, cols=0)
            .split_by_idxs(trn_idx, val_idx) # .split_by_rand_pct(valid_pct=VALID_PERCENT, seed=SEED)
            .label_from_df(cols=1)
            .add_test(['../input/aptos2019-blindness-detection/test_images/' + f for f in df_test.id_code]))

    data = (src.transform(my_tfms, size=IMG_SIZE, padding_mode='zeros').databunch(bs=BS).normalize(imagenet_stats))
#     data = (src.transform(my_tfms, size=IMG_SIZE, padding_mode='reflection').databunch(bs=BS).normalize(imagenet_stats))
    print(f"[{fold}] fold : Train {len(data.train_ds)} / Val {len(data.valid_ds)} / Test: {len(data.test_ds)}")
    data.show_batch(rows=3, figsize=(5, 5))

#     MyKappaScore(weights='quadratic', bound=[.5, 1.5, 2.5, 3.5]), MyKappaScore(weights='quadratic', bound=[1, 1.5, 2.5, 3.5]),
    # https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
#                               metrics=[r2_score]
    learn = custom_cnn_learner(data, model_dir='/kaggle/working/models', metrics=[accuracy, KappaScore(weights='quadratic')])

    # learn = ConvLearner.pretrained(senext_50, data, xtra_cut=2)
#     learn.loss_func = MSELossFlat() # criterion = torch.nn.MSELoss()
    break

    if not IS_ONLY_INFERENCE:
        train_params = [(1e-04, 2, 'pred_0', False, False), # lr = 5e-05 lr = 1e-06 lr = 6.3e-5 # prev 1e-3 9e-3, 5e-3, 1e-2 #  lr = 1e-4
#                         (1e-04, 1, 'pred_1', False, False),
                        (1e-05, 1, 'pred_1', True, True),
                        (1e-08, 1, 'pred_2', True, False)]

        # Stored after unfreezing model: fold_0point_pred_2.pth
        for lr, cycles, point, is_save, to_unfreeze in train_params:
            print(f'Fold {fold}; Epoch 1; Point {point}')
            learn.fit_one_cycle(cycles, lr)
            res[fold]['test'][point], res[fold]['val'][point] = _get_preds(learn, DatasetType.Test), _get_preds(learn, DatasetType.Valid)
            if is_save:
#                 TODO uncomment if need prev model
#                 learn.save(f'fold_{fold}point_{point}')
                learn.save(f'fold_{fold}point_{point}.v2')
            if to_unfreeze:
                for name, param in learn.model.named_parameters():
                    param.requires_grad = True
                    # print(name, param.requires_grad)

        break

#         lr = lr = 1e-05, 8e-4, lr = 1e-06
        learn.fit_one_cycle(1, lr)
    else:
        learn.load('fold_0point_pred_2')
        for param in model.parameters():
            param.requires_grad = False
        # ??
        learn.model.eval()

        point = 'pred_0'
        res[fold]['test'][point], res[fold]['val'][point] = _get_preds(learn, DatasetType.Test), _get_preds(learn, DatasetType.Valid)

    #     lr = 1e-3 # prev 9e-3, 5e-3, 1e-2
    #     learn.fit_one_cycle(2, lr)
    #     learn.save(f'r101_fold_{fold}_point2')

    #     learn.data = (src.transform(tfms, size=IMG_SIZE2, padding_mode='zeros').databunch(bs=BS).normalize(imagenet_stats))
    #     learn.fit_one_cycle(2, slice(2e-5, 3e-4))
    #     learn.data = (src.transform(tfms, size=IMG_SIZE3, padding_mode='zeros').databunch(bs=BS).normalize(imagenet_stats))
    #     learn.fit_one_cycle(3, slice(7e-5, 4e-3))

    #     learners.append(learn)
    if fold > 1:
        break

# data_loader_train.show_batch(rows=3, figsize=(5, 5))
