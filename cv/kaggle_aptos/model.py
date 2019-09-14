

def train_model(model, patience, n_epochs, is_amp=True, is_custom_loss=True):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    n_batches = int(len(data_loader_train))
    train_losses_hist = np.zeros((n_epochs, n_batches))
    all_metrics = []
    for epoch in range(n_epochs):
        if epoch == n_freeze:
            unfreeze_model(model)

        model.train()  # Set model to train mode
        epoch_stat = {'train': {'steps': 0, 'loss': 0}, 'valid': {'steps': 0, 'loss': 0}}
        #         scheduler.step()

        tk0 = tqdm(data_loader_train, total=n_batches)
        for step, (inputs, labels, sample_weights, ratio_info) in enumerate(tk0):
            labels = labels.view(-1, 1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            sample_weights = sample_weights.to(device, dtype=torch.float)

            outputs = model(inputs)  # Forward step
            loss = criterion(outputs, labels)  # Get Losses
            if is_custom_loss:
                loss *= sample_weights
                loss = loss.sum() / sample_weights.sum()

            if is_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            epoch_stat['train']['loss'] += loss.item()  # Accumulate losses;
            train_losses_hist[epoch, step] = loss.item()
            #             epoch_stat['train']['total'] += inputs.size(0)

            if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()

            tk0.set_postfix(loss=(epoch_stat['train']['loss'] / (step + 1)))

        # if step % 10 == 0:
        #                 train_losses.append(loss.item())
        #                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, step * len(data), len(data_loader_train.dataset), 100. * step / len(data_loader_train), loss_data))

        epoch_loss = epoch_stat['train']['loss'] / len(data_loader_train)
        # len(epoch_stat['train']['steps'])
        model.eval()

        val_loss_mse, split_loss_less, split_loss_more = 0, 0, 0
        y_hat_all, y_all, ratio_info_all = np.array([]), np.array([]), np.array([])

        for step, (inputs, labels, sample_weights, ratio_info) in enumerate(data_loader_valid):
            labels = labels.view(-1, 1)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(inputs)

            y_hat, y = torch.Tensor.cpu(outputs.view(-1)), torch.Tensor.cpu(labels.view(-1))
            y_hat_all = np.append(y_hat_all, np.array(y_hat))
            y_all = np.append(y_all, np.array(y))
            ratio_info_all = np.append(ratio_info_all, np.array(ratio_info))

            val_loss_mse += torch.nn.functional.mse_loss(y_hat, y).mean().item()
            y_hat = torch.from_numpy(res_to_classes(np.array(y_hat), bound=bound, shift=0)).type(torch.float32)

            epoch_stat['valid']['loss'] += quadratic_kappa(y_hat, y).mean().item()

        tr = 1.25
        idx_lower = ratio_info_all <= tr
        idx_upper = ratio_info_all > tr
        k_above = local_quadratic_kappa(y_all[idx_upper], res_to_classes(y_hat_all[idx_upper], bound=bound).astype(int),
                                        5)
        k_below = local_quadratic_kappa(y_all[idx_lower], res_to_classes(y_hat_all[idx_lower], bound=bound).astype(int),
                                        5)
        val_loss_mse_above = math.sqrt(np.sum((y_hat_all[idx_upper] - y_all[idx_upper]) ** 2) / len(y_all[idx_upper]))
        val_loss_mse_below = math.sqrt(np.sum((y_hat_all[idx_lower] - y_all[idx_lower]) ** 2) / len(y_all[idx_lower]))

        val_loss = epoch_stat['valid']['loss'] / len(data_loader_valid)
        val_loss_mse /= len(data_loader_valid)

        all_metrics.append(OrderedDict({
            'train_loss': epoch_loss,
            'val_kappa': val_loss,
            'val_kappa_above': k_above,
            'val_kappa_below': k_below,
            'val_mse': val_loss_mse,
            'val_mse_above': val_loss_mse_above,
            'val_mse_below': val_loss_mse_below
        }))
        print(f'Epoch {epoch}/{n_epochs - 1} Train Loss: {epoch_loss:.3f} ',
              f'Kappa {val_loss:.3f} {k_above:.3f} {k_below:.3f}',  # f'Kappa {val_loss:.3f} {k_below:.3f}',
              f'MSE {val_loss_mse:.3f} {val_loss_mse_above:.3f}')

        get_mse_report(y_hat_all, y_all)

        if epoch == 0:
            perc_below = np.round(np.sum(idx_lower) * 100 / len(y_all), 1)
            print(f'Perc: {perc_below}% ')

        scheduler.step(val_loss_mse_above)  # scheduler.step(val_loss_mse)
        #         for param_group in optimizer.param_groups: print('lr:', np.round(param_group['lr'], 9))
        early_stopping((val_loss_mse + val_loss_mse_above + (1 - k_below / 5) / 2) / 3,
                       model)  # val_loss_mse + (1 - k_below / 5 ) / 2 ) / 3

        if epoch >= 1:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, train_losses_hist, all_metrics


def get_model(model_type, num_classes):
    if model_type in ['efficientnet-b5', 'efficientnet-b4', 'efficientnet-b3']:
        model = EfficientNet.from_name(model_type)
        model.load_state_dict(torch.load(eff_net_paths[model_type]))
        freeze_model(model)
        model._fc = torch.nn.Linear(model._fc.in_features, NUM_CLASSES)
    elif model_type == 'resnext101':
        model = resnext101_32x16d_wsl(path=MODEL_PATH_RESNEXT)
        freeze_model(model)
        model.fc = torch.nn.Linear(2048, num_classes)
    elif model_type == 'densenet201':
        model = densenet201(path=DENSENET101_PETRAINED_PATH)
        freeze_model(model)
        model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_classes,
                                           bias=True)
    elif model_type == 'inceptionv4':
        model = inceptionv4()
        freeze_model(model)
        model.classif = torch.nn.Linear(in_features=model.classif.in_features, out_features=num_classes, bias=True)
    else:
        raise NotImplementedError(f'{MODEL} is not valid model alias')

    model.to(device)
    return model


def predict(model, dl):
    dataset_len = len(dl.dataset)
    preds = np.zeros((dataset_len, 1))
    bs = dl.batch_size

    for i, (x_batch, _) in enumerate(dl):
        pred = model(x_batch.to(device))
        preds[i * bs:(i + 1) * bs] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
    return preds


def fit():
    # %%time
    seed_everything(SEED)

    # from torch.optim.lr_scheduler import CyclicLR
    # from warmup_scheduler import GradualWarmupScheduler
    # from torch.optim.lr_scheduler.CosineAnnealingLR

    # CUT = False # ([1.01, 1.02, 1.03, 1.035, 1.04, 1.045, 1.05, 1.055, 1.1, 1.15, 1.2], .8)
    # CUT_VALIDATE = False # ([1.1, 1.2, 1.33], .5) # ([1.1, 1.2, 1.2, 1.25, 1.33], .3)

    # print('Try zoom and crop for sqaure images. Revert validation strategy')
    # print('Try weighted by classes loss. Added agressive zooming and cropping. Removed rotation')
    # print('Add zoom, crop and brightness augmentations. Save all epochs models. Acc step: 2')
    # print('Dense net: Add zoom, crop; rotate only round images')
    #  Add contrast aug.
    # print('Dense net: Carefull circle crop for all images. Custom loss. Fixed es')
    # print('Eff net 4. Disabled Carefull circle crop. HEavilly oversampled 3 and 4 classes')
    # print('Eff net 4. Add more zoom for cropped. Also to validation')
    print('Eff net 5. Shif all images. .. Next - try to remove jitter')

    # *** < Hyperparams ***
    NUM_CLASSES = 1
    # IS_FINAL = False
    IS_AMP, IS_CUSTOM_LOSS = False, False

    lr = 8e-5  # 1e-4 8e-5 8e-5 3e-5
    # lr_unfrozen = slice(3e-8, 5e-6)
    img_size = 224  # 224 # 224 if not IS_FINAL else 456  # 224 - prev experiment
    batch_size = 16
    n_epochs, n_freeze, patience = 15, 1, 3  # !!! revert !
    n_folds, n_stop_fold = 4, 1
    bound = [0.7, 1.5, 2.5, 3.25]  # [0.7, 1.5, 2.5, 3.3]
    accumulation_steps = 1
    IS_ADD_PREV_DATA = True
    # OVERSAMPLE_NEW_DATASET = [1, 3, 1, 3, 5] # 'more' # 'less'
    # OVERSAMPLE_NEW_DATASET = [1, 2, 1, 4, 3] # 'more' # 'less'
    # OVERSAMPLE_NEW_DATASET = [1, 2, 1, 3, 3] # 'more' # 'less'
    OVERSAMPLE_NEW_DATASET = [1, 1, 1, 3, 3]  # 'more' # 'less'
    # 0    1805
    # 1     370
    # 2     999
    # 3     193
    # 4     295

    OVERSAMPLE_RECTANGULAR = False
    REMOVE_DUPLICATES_FROM_TRAIN = True
    PREV_CHUNK_INDEX = 0
    DEBUG_PERCENTAGE = None  # None use .05 if quick test needed

    # CUT = ([1.1, 1.2, 1.2, 1.25, 1.33], .3)
    # ZOOM = ((1.1, 1.33), .3, (1.1, 1.2), .1)
    # ROT = (180, .5) # False

    ROT = (90, .5)  # False # (90, .9)

    # ZOOM = ((1.12, 1.23), .5, (1.12, 1.2), .5)
    ZOOM = ((1.12, 1.23), 0, (1.12, 1.2), 0)
    ZOOM_OVERSAMPLED = ((1.12, 1.23), 0, (1.12, 1.2), 0)  # ((1.12, 1.23), .3, (1.12, 1.2), .3)
    CUT_OVERSAMPLED = False  # ([1.1, 1.2, 1.33], .7)

    ZOOM_VALIDATE = False  # ((1.12, 1.23), .5, (1.12, 1.2), .5) # ((1.1, 1.33), .3, (1.1, 1.2), .1) # ((1.1, 1.2), .5, (1, 1), 0)
    ROT_VALIDATE = False

    # *** Hyperparams /> ***
    model_type = 'efficientnet-b4'  # efficientnet-b3, 'inceptionv4' # 'efficientnet-b5' # 'resnext101' # 'efficientnet-b5' # 'efficientnet-b4', 'densenet201'

    df, train_idx, valid_idx = glue_df_gen2(df_source, df_prev_chunks, replace_pairs, OVERSAMPLE_NEW_DATASET,
                                            IS_ADD_PREV_DATA, REMOVE_DUPLICATES_FROM_TRAIN, is_stratified=False,
                                            debug_percentage=DEBUG_PERCENTAGE,
                                            oversample_rectangular=OVERSAMPLE_RECTANGULAR, fr_prev=1, df_sub=None)

    # , shrink_validation_bound=1.2

    channel_stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ResizeMethod.SQUISH
    list_default_transformations = [transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                    transforms.Normalize(**channel_stats)]
    list_train_tfms = [
                          transforms.RandomHorizontalFlip(),
                          #     transforms.RandomRotation(degrees=90),
                          transforms.RandomVerticalFlip(),
                          # 135 ?worse , transforms.ColorJitter(saturation=0.05)  # transforms.RandomCrop(img_size, padding_mode='reflect'),
                          #      RandomChoiceAndApply([
                          #         transforms.ColorJitter(brightness=0.2),
                          #         transforms.ColorJitter(contrast=(0.9, 1.3)),
                          # #         transforms.ColorJitter(saturation=0.1)
                          #      ], p=.05)
                      ] + list_default_transformations

    tsfm_train = transforms.Compose(list_train_tfms)
    tsfm_valid = transforms.Compose(list_default_transformations)

    train_dataset = RetinopathyDatasetTrain(df, transform=tsfm_train, cut=False, zoom=ZOOM, rot=ROT, mode='train',
                                            zoom_oversampled=ZOOM_OVERSAMPLED, cut_oversampled=CUT_OVERSAMPLED,
                                            jitter_p=.8, jitter_p2=.7, jitter_large_p=.8, jitter_large_p2=.7,
                                            shift_x_p=.5, shift_y_p=.5)
    valid_dataset = RetinopathyDatasetTrain(df, transform=tsfm_valid, cut=False, zoom=False, rot=False, mode='valid')
    test_dataset = RetinopathyDatasetTest(csv_file=SUBMISSION_FILE, transform=tsfm_valid)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training
    fold_predictions = np.zeros((len(test_dataset), n_folds))
    train_losses_hist, all_metrics_hist = [], []
    for fold in np.arange(n_folds):
        show_batch_info(train_dataset, train_idx[fold], valid_dataset, valid_idx[fold], test_dataset)
        device = torch.device("cuda:0")
        model = get_model(model_type, NUM_CLASSES)
        if IS_CUSTOM_LOSS:
            criterion = torch.nn.MSELoss(reduction='none')
        else:
            criterion = torch.nn.MSELoss()  # criterion = torch.nn.L1Loss() - +- the same as MSE

            #     criterion = AdjMSELoss()

            # https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
            # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no?rq=1

            #     plist = [
            #         {'params': model.conv1.parameters(), 'lr': lr/100},
            #         {'params': model.bn1.parameters(), 'lr': lr/100},
            #         {'params': model.layer1.parameters(), 'lr': lr},
            #         {'params': model.layer2.parameters(), 'lr': lr/2},
            #         {'params': model.layer3.parameters(), 'lr': lr/2},
            #         {'params': model.layer4.parameters(), 'lr': lr/2},
            #         {'params': model.fc.parameters(), 'lr': lr}
            #     ]

        plist = [{'params': model.parameters(), 'lr': lr}]
        optimizer = optim.Adam(plist, lr=lr)
        #     optimizer = RAdam(plist, lr=lr)
        #     optimizer = optim.SGD(plist, lr=lr, momentum=0.9)
        #     optimizer = optim.RMSprop(plist, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.01, momentum=0, centered=False)
        #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, )
        #     scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        # !!!  TODO revert if needed factor=0.1
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.15, patience=1,
                                                   verbose=True, threshold=0.0001, threshold_mode='rel',
                                                   cooldown=0, min_lr=0, eps=1e-08)
        #     scheduler = CyclicLR(optimizer, base_lr=lr / 20, max_lr=lr * 5, cycle_momentum=False, step_size_up=1000)
        #     eta_min = 1e-6 T_max = 10 T_mult = 1 restart_decay = 0.97
        #     scheduler = lr_scheduler.CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)

        # TODO revert
        if IS_AMP:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        train_sampler = SubsetRandomSampler(train_idx[fold])
        valid_sampler = SubsetRandomSampler(valid_idx[fold])

        data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                                        sampler=train_sampler)
        data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4,
                                                        sampler=valid_sampler)

        model, _train_losses_hist, _all_metrics = train_model(model, patience, n_epochs, IS_AMP, IS_CUSTOM_LOSS)
        train_losses_hist.append(_train_losses_hist)
        all_metrics_hist.append(_all_metrics)
        model.eval()
        freeze_model(model)

        if DEBUG_PERCENTAGE is None:
            fold_predictions[:, fold] = predict(model, data_loader_test).reshape(fold_predictions.shape[0])

            #     del(data_loader_train, data_loader_valid) # model
        gc.collect()
        torch.cuda.empty_cache()
        if fold >= n_stop_fold - 1:
            break

    if DEBUG_PERCENTAGE is None:
        test_preds_avg_fods = np.mean(fold_predictions[:, :n_stop_fold], axis=1)
        fold_predictions_avg = res_to_classes(test_preds_avg_fods, bound=bound, shift=0).astype(int)
        plt.hist(fold_predictions_avg);
    else:
        print('submission data is not predicted! Set DEBUG_PERCENTAGE to None')


def fit_fine_tune():
    print('Eff net 4. fine tune on unseen data')

    # *** < Hyperparams ***
    NUM_CLASSES = 1
    IS_AMP, IS_CUSTOM_LOSS = False, False

    lr = 1e-5  # 1e-4 8e-5 8e-5 3e-5
    # lr_unfrozen = slice(3e-8, 5e-6)
    img_size = 380  # 456 # 224 if not IS_FINAL else 456  # 224 - prev experiment
    batch_size = 16
    n_epochs, n_freeze, patience = 15, 0, 3
    n_folds, n_stop_fold = 4, 1
    accumulation_steps = 1
    IS_ADD_PREV_DATA = True
    OVERSAMPLE_NEW_DATASET = [1, 1, 1, 3, 3]  # 'more' # 'less'
    # 0    1805
    # 1     370
    # 2     999
    # 3     193
    # 4     295

    OVERSAMPLE_RECTANGULAR = False
    REMOVE_DUPLICATES_FROM_TRAIN = True
    PREV_CHUNK_INDEX = 1
    DEBUG_PERCENTAGE = None  # None use .05 if quick test needed

    # CUT = ([1.1, 1.2, 1.2, 1.25, 1.33], .3)
    # ZOOM = ((1.1, 1.33), .3, (1.1, 1.2), .1)

    ROT = False  # (90, .9)
    ZOOM = ((1.12, 1.23), 0, (1.12, 1.2), 0)
    ZOOM_OVERSAMPLED = ((1.12, 1.23), 0, (1.12, 1.2), 0)  # ((1.12, 1.23), .3, (1.12, 1.2), .3)
    CUT_OVERSAMPLED = False  # ([1.1, 1.2, 1.33], .7)

    CUT_VALIDATE, ZOOM_VALIDATE, ROT_VALIDATE = False, False, False
    # *** Hyperparams /> ***

    # df_prev_2 = [pd.concat([df_prev_chunks[0][~df_prev_chunks[0]['diagnosis'].isin([0, 2])].sample(frac=.3),
    #     df_prev_chunks[0][df_prev_chunks[0]['diagnosis'].isin([0, 2])].sample(frac=.05)], axis=0),
    #             pd.concat([df_prev_chunks[1][~df_prev_chunks[1]['diagnosis'].isin([0, 2])].sample(frac=.3),
    #     df_prev_chunks[1][df_prev_chunks[1]['diagnosis'].isin([0, 2])].sample(frac=.05)], axis=0),
    # ]

    # df_prev_chunks
    df_2, train_idx_2, valid_idx_2 = glue_df_gen2(df_source, df_prev_chunks, replace_pairs, OVERSAMPLE_NEW_DATASET,
                                                  IS_ADD_PREV_DATA, REMOVE_DUPLICATES_FROM_TRAIN, is_stratified=False,
                                                  debug_percentage=DEBUG_PERCENTAGE,
                                                  oversample_rectangular=OVERSAMPLE_RECTANGULAR, fr_prev=.1,
                                                  df_sub=df_sub)

    # train_idx_2 = df_2.index
    # pd.concat([df_sub, messidor_df[['id_code', 'diagnosis', 'train', 'oversampled']]], axis=0)
    # .sample(frac=.01)

    channel_stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ResizeMethod.SQUISH
    list_default_transformations = [transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                    transforms.Normalize(**channel_stats)]
    list_train_tfms = [
                          transforms.RandomHorizontalFlip(),
                          #     transforms.RandomRotation(degrees=90),
                          transforms.RandomVerticalFlip(),
                          # 135 ?worse , transforms.ColorJitter(saturation=0.05)  # transforms.RandomCrop(img_size, padding_mode='reflect'),
                          #      RandomChoiceAndApply([
                          #         transforms.ColorJitter(brightness=0.2),
                          #         transforms.ColorJitter(contrast=(0.9, 1.3)),
                          # #         transforms.ColorJitter(saturation=0.1)
                          #      ], p=.05)
                      ] + list_default_transformations

    tsfm_train = transforms.Compose(list_train_tfms)
    tsfm_valid = transforms.Compose(list_default_transformations)

    # TODO check !!!!!!!!!!!!!!!!!!!11
    train_dataset = RetinopathyDatasetTrain(df_2, transform=tsfm_train, cut=False, zoom=ZOOM, rot=ROT, mode='train',
                                            zoom_oversampled=ZOOM_OVERSAMPLED, cut_oversampled=CUT_OVERSAMPLED,
                                            jitter_p=0.1, jitter_p2=0.7, jitter_large_p=0.1, jitter_large_p2=0.7,
                                            shift_x_p=.25, shift_y_p=.25, is_circle_crop=False)
    valid_dataset = RetinopathyDatasetTrain(df_2, transform=tsfm_valid, cut=CUT_VALIDATE, zoom=ZOOM_VALIDATE,
                                            rot=ROT_VALIDATE, mode='valid')
    # valid_dataset = RetinopathyDatasetTest2(messidor_df, transform=tsfm_valid)

    test_dataset = RetinopathyDatasetTest(csv_file=SUBMISSION_FILE, transform=tsfm_valid)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training
    fold_predictions = np.zeros((len(test_dataset), n_folds))
    train_losses_hist, all_metrics_hist = [], []
    for fold in np.arange(n_folds):
        show_batch_info(train_dataset, train_idx_2[fold], valid_dataset, valid_idx_2[fold], test_dataset)
        device = torch.device("cuda:0")
        model = get_model(model_type, NUM_CLASSES)
        model.load_state_dict(torch.load('checkpoint.pt.bkp'))

        if IS_CUSTOM_LOSS:
            criterion = torch.nn.MSELoss(reduction='none')
        else:
            criterion = torch.nn.MSELoss()  # criterion = torch.nn.L1Loss() - +- the same as MSE

            #     criterion = AdjMSELoss()
            # https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
            # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no?rq=1

            #     plist = [
            #         {'params': model.conv1.parameters(), 'lr': lr/100},
            #         {'params': model.bn1.parameters(), 'lr': lr/100},
            #         {'params': model.layer1.parameters(), 'lr': lr},
            #         {'params': model.layer2.parameters(), 'lr': lr/2},
            #         {'params': model.layer3.parameters(), 'lr': lr/2},
            #         {'params': model.layer4.parameters(), 'lr': lr/2},
            #         {'params': model.fc.parameters(), 'lr': lr}
            #     ]

        plist = [{'params': model.parameters(), 'lr': lr}]
        optimizer = optim.Adam(plist, lr=lr)
        #     optimizer = RAdam(plist, lr=lr)
        #     optimizer = optim.SGD(plist, lr=lr, momentum=0.9)
        #     optimizer = optim.RMSprop(plist, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.01, momentum=0, centered=False)
        #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, )
        #     scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        # !!!  TODO revert if needed factor=0.1
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.15, patience=1,
                                                   verbose=True, threshold=0.0001, threshold_mode='rel',
                                                   cooldown=0, min_lr=0, eps=1e-08)
        #     scheduler = CyclicLR(optimizer, base_lr=lr / 20, max_lr=lr * 5, cycle_momentum=False, step_size_up=1000)
        #     eta_min = 1e-6 T_max = 10 T_mult = 1 restart_decay = 0.97
        #     scheduler = lr_scheduler.CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)

        # TODO revert
        if IS_AMP:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        train_sampler = SubsetRandomSampler(train_idx_2[fold])
        valid_sampler = SubsetSampler(valid_idx_2[fold])  # messidor_df.index

        data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                                        sampler=train_sampler)
        data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4,
                                                        sampler=valid_sampler)

        model, _train_losses_hist, _all_metrics = train_model(model, patience, n_epochs, IS_AMP, IS_CUSTOM_LOSS)
        train_losses_hist.append(_train_losses_hist)
        all_metrics_hist.append(_all_metrics)
        model.eval()
        freeze_model(model)

        if DEBUG_PERCENTAGE is None:
            fold_predictions[:, fold] = predict(model, data_loader_test).reshape(fold_predictions.shape[0])

            #     del(data_loader_train, data_loader_valid) # model
        gc.collect()
        torch.cuda.empty_cache()
        if fold >= n_stop_fold - 1:
            break

    if DEBUG_PERCENTAGE is None:
        test_preds_avg_fods = np.mean(fold_predictions[:, :n_stop_fold], axis=1)
        fold_predictions_avg = res_to_classes(test_preds_avg_fods, bound=bound, shift=0).astype(int)
        plt.hist(fold_predictions_avg);
    else:
        print('submission data is not predicted! Set DEBUG_PERCENTAGE to None')

        # Epoch 5/14 Train Loss: 0.183  Kappa 0.905 0.824 0.953 MSE 0.241 0.632
        # {0: '0.03 [448]', 1: '0.49 [92]', 2: '0.22 [253]', 3: '0.21 [42]', 4: '1.17 [81]'}

        # Epoch 3/14 Train Loss: 0.212  Kappa 0.906 0.815 0.959 MSE 0.251 0.653
        # {0: '0.03 [448]', 1: '0.39 [92]', 2: '0.26 [253]', 3: '0.22 [42]', 4: '1.32 [81]'}
        # All 0.251 m: 0.248 STD:  0.022 Expected LB:  0.5063068509836438
