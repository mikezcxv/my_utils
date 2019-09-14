

def denormalize_tensor(img: torch.Tensor) -> torch.Tensor:
    return img * torch.Tensor(channel_stats['std'])[:, None, None] + \
           torch.Tensor(channel_stats['mean'])[:, None, None]


# df.loc[valid_idx[0]]
def show_batch_sample(dataset, idx=None, n=8, mode='train', n_rows=1, limit=2 ** 16):
    if mode == 'train':
        tsfm_train = transforms.Compose(list_train_tfms)
        sampler = SubsetRandomSampler(idx[:limit])
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler)
    elif mode == 'valid':
        tsfm_valid = transforms.Compose(list_default_transformations)
        sampler = SubsetRandomSampler(idx[:limit])
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler)

    if mode == 'test':
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for i, (a, c) in enumerate(dl):
            w = c.numpy().squeeze()
            print(np.round(sorted(c.numpy()), 3))
            f, ax = plt.subplots(1, n, figsize=(18, 5))
            #     imshow(torchvision.utils.make_grid(images))
            #     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
            for el_idx in range(n):
                ax[el_idx].set_title(' [r: ' + str(np.round(c.numpy()[el_idx], 2)) + '] ')
                im = F.to_pil_image(denormalize_tensor(a[el_idx]))
                ax[el_idx].imshow(im)

            plt.show()
            if i >= n_rows - 1:
                break
    else:
        for i, (a, b, c, d) in enumerate(dl):
            w = c.numpy().squeeze()
            print(np.round(sorted(d.numpy()), 3))
            f, ax = plt.subplots(1, n, figsize=(18, 5))
            for el_idx in range(n):
                ax[el_idx].set_title(str(b.numpy()[el_idx]) +
                                     ' [r: ' + str(np.round(d.numpy()[el_idx], 2)) + '] ' +
                                     'w: ' + str(w[el_idx]))
                im = F.to_pil_image(denormalize_tensor(a[el_idx]))
                ax[el_idx].imshow(im)

            plt.show()
            if i >= n_rows - 1:
                break


# TODO show_batch_sample(n=8, mode='test')

def show_batch_info(train_dataset, idx_train, valid_dataset, idx_valid, test_dataset):
    print('In a batch on average we will see:')
    print('Classes:', dict(np.round(df.iloc[train_idx[0]]['diagnosis'].value_counts() / len(df), 1) * batch_size))
    r = {v: np.round(c * batch_size / len(df), 2) for v, c in
         dict(Counter(np.round(df.iloc[train_idx[0]]['size_ratio'], 2))).items() if c > 10}
    print('Img ratios:', sorted(r.items(), key=operator.itemgetter(1), reverse=True)[:5])

    print('Fold:', fold)
    print('Train imgs')
    show_batch_sample(train_dataset, idx_train, n=8, mode='train')
    print('Validation imgs:')
    show_batch_sample(valid_dataset, idx_valid, n=8, mode='valid')
    print('Test imgs:')
    show_batch_sample(test_dataset, n=8, mode='test')
