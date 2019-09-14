

class SimpleLenDataset(Dataset):
    def __len__(self):
        return len(self.data)


class RetinopathyDatasetTrain(SimpleLenDataset):
    def __init__(self, prepared_df, mode, transform=None,
                 cut=False, zoom=False, rot=False,
                 cut_oversampled=False, zoom_oversampled=False,
                 jitter_p=.8, jitter_p2=.7, jitter_large_p=.8, jitter_large_p2=.7,
                 shift_x_p=0, shift_y_p=0,
                 is_circle_crop=True):
        #         self.data = pd.read_csv(csv_file)
        self.data = prepared_df
        self.transform = transform
        self.cut = cut
        self.zoom = zoom
        self.rot = rot
        self.mode = mode
        self.cut_oversampled = cut_oversampled
        self.zoom_oversampled = zoom_oversampled
        self.jitter_p = jitter_p
        self.jitter_p2 = jitter_p2
        self.jitter_large_p = jitter_large_p
        self.jitter_large_p2 = jitter_large_p2
        self.shift_x_p = shift_x_p
        self.shift_y_p = shift_y_p
        self.is_circle_crop = is_circle_crop

    def __getitem__(self, idx):
        img_name = os.path.join('../input', self.data.loc[idx, 'id_code'])
        this_img = self.data.loc[idx]

        # pil_image = PIL.Image.open('Image.jpg').convert('RGB')
        # img = cv2.imread(path)
        #         image = crop_image_from_gray(np.array(image), tol=7)
        #         image = my_transform_img(image, tol = 7, inertia_px = 30, padding_px = 15, last_n_rows_smoothing=30, crop_x=3, crop_y=5, crop_y_last=10)
        #         image = np.array(image)
        #         image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image , (0,0) , sigmaX = 10) ,-4 ,128)
        #         image = Image.fromarray(image)
        # 1.333333    1403 .7         # 1.321757     134 .02         # 1.000000      86         # 1.301843      56 .01

        #         if self.img_size:
        #             image = Image.open(img_name)
        #             image = make_square_img(image, (self.img_size, self.img_size))

        #         if self.is_enable_zoom and np.random.rand() < self.any_trans_p:
        #             image = pil2tensor(image, np.float32)
        #             image.div_(255)
        #             image = fastai.vision.image.Image(image)
        # #             image = open_image(img_name)
        #             image = image.apply_tfms(fastai.vision.zoom(scale=(1.25, 1.3), p=1, row_pct=.5, col_pct=.5))
        #             image = transforms.ToPILImage()(image.data)

        if self.mode == 'train':
            if self.is_circle_crop:
                image = circle_crop(img_name)
                image = Image.fromarray(image)
            else:
                image = PIL.Image.open(img_name)
            # image = make_square_img(Image.fromarray(circle_crop(img_name)))

            cut_vars, cut_p = [], 0
            z_lower, z_lower_p, z_upper, z_upper_p = False, 0, False, 0
            rot_max_deg, rot_p = 0, 0

            if self.cut: cut_vars, cut_p = self.cut
            if self.zoom: z_lower, z_lower_p, z_upper, z_upper_p = self.zoom
            if self.rot: rot_max_deg, rot_p = self.rot

            if self.rot:
                image = gaussian_img_rotation(image, max_deg=rot_max_deg, p=rot_p)
            if this_img['oversampled']:
                if self.zoom_oversampled:
                    z_lower, z_lower_p, z_upper, z_upper_p = self.zoom_oversampled
                if self.cut_oversampled:
                    cut_vars, cut_p = self.cut_oversampled

            cropped = False
            if self.data.loc[idx, 'size_ratio'] < 1.05:
                if self.jitter_p:
                    image = my_jitter_img(image, p=self.jitter_p, p2=self.jitter_p2, max_cut_percent=.1)
            elif self.jitter_large_p:
                image = my_jitter_img(image, p=self.jitter_large_p, p2=self.jitter_large_p2, max_cut_percent=.02)
            # if np.random.rand() < cut_p:
            #                     image = cut_img(image, 1.05, np.random.choice(cut_vars))
            #                     cropped = True

            # Shifts section
            if self.shift_x_p or self.shift_y_p:
                image = shift_image(np.array(image), p_x=self.shift_x_p, p_y=self.shift_y_p)
                image = Image.fromarray(image)

            # Zoom section
            if (self.data.loc[idx, 'size_ratio'] < 1.05) and not cropped:
                if np.random.rand() < z_lower_p:
                    image = pil2tensor(image, np.float32)
                    image.div_(255)
                    image = fastai.vision.image.Image(image)
                    image = image.apply_tfms(fastai.vision.zoom(scale=z_lower, p=1, row_pct=.5, col_pct=.5))
                    image = transforms.ToPILImage()(image.data)
            else:
                if np.random.rand() < z_upper_p:
                    image = pil2tensor(image, np.float32)
                    image.div_(255)
                    image = fastai.vision.image.Image(image)
                    image = image.apply_tfms(fastai.vision.zoom(scale=z_upper, p=1, row_pct=.5, col_pct=.5))
                    image = transforms.ToPILImage()(image.data)
        else:
            #             image = circle_crop(img_name)
            image = PIL.Image.open(img_name)
        # image = Image.fromarray(image)
        #             image = crop_image_from_gray(np.array(image), tol=7)
        #             image = make_square_img(Image.fromarray(image))

        this_img = self.data.loc[idx]
        label = torch.tensor(this_img['diagnosis'])
        #         classes_stat = {0: 1.0, 2: 1.4, 1: 3.0, 4: 7.0, 3: 7.3} {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.5, 4: 2.0}
        classes_weights = {0: 1.0, 1: 1.2, 2: 1.1, 3: 1.2,
                           4: 2}  # 0: '0.05 [448]', 1: '0.25 [92]', 2: '0.35 [253]', 3: '0.58 [42]', 4: '2.83 [81]'}
        sample_weights = torch.tensor(this_img[['size_ratio']].apply(
            lambda x: classes_weights[this_img[target]] * (1.05 if x > 1.25 else 1.0)
        ))
        ratio_info = torch.tensor(this_img['size_ratio'])

        image = self.transform(image)
        return (image, label, sample_weights, ratio_info)


class RetinopathyDatasetTest(SimpleLenDataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(TEST_IMG_FOLDER, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = np.array(image)
        ratio_info = torch.tensor(float(image.shape[1] / image.shape[0]))

        image = crop_image_from_gray(image, tol=7)
        #         image = np.array(make_square_img(Image.fromarray(image)))

        image = Image.fromarray(image)
        image = self.transform(image)
        return (image, ratio_info)


TEST2_SUB_FOLDER = 'messidor/aptos_new_data_png/'
TEST2_FOLDER = '../input/' + TEST2_SUB_FOLDER
messidor_df = pd.read_csv(TEST2_FOLDER + 'messidor_labels.csv')
messidor_df.rename({'image_id': 'id_code', 'adjudicated_dr_grade': target}, axis=1, inplace=True)
messidor_df['id_code'] = messidor_df['id_code'].apply(lambda x: TEST2_SUB_FOLDER + x)


class RetinopathyDatasetTest2(SimpleLenDataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join('../input/', self.data.loc[idx, 'id_code'])
        image = Image.open(img_name)
        image = crop_image_from_gray(np.array(image), tol=7)

        ratio = float(image.shape[1] / image.shape[0])
        ratio_info = torch.tensor(ratio)

        this_img = self.data.loc[idx]
        label = torch.tensor(this_img['diagnosis'])
        classes_weights = {0: 1.0, 1: 1.2, 2: 1.1, 3: 1.2, 4: 2}
        sample_weights = torch.tensor(classes_weights[this_img[target]] * (1.05 if ratio > 1.25 else 1.0))

        image = Image.fromarray(image)
        image = self.transform(image)
        return (image, label, sample_weights, ratio_info)
