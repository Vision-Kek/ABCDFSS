r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetSUIM(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num_val=600):
        self.split = split
        self.benchmark = 'suim'
        self.shot = shot
        self.num_val = num_val

        self.base_path = os.path.join(datapath)
        self.img_path = os.path.join(self.base_path, 'images')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['FV','HD','PF','RI','RO','SR','WR']

        self.class_ids = range(len(self.categories))
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        # if it is the target domain, then also test on entire dataset
        return self.num_images if self.split !='val' else self.num_val

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_set': (support_imgs, support_masks),
                 'support_classes': torch.tensor([class_sample]), # adapt to Nway

                 'query_name': query_name, # REMOVE
                 'support_imgs': support_imgs, # REMOVE
                 'support_masks': support_masks, # REMOVE
                 'support_names': support_names, # REMOVE
                 'class_id': torch.tensor(class_sample)} # REMOVE

        return batch


    def load_frame(self, query_mask_path, support_mask_paths):
        def maskpath_to_imgpath(maskpath):
            filename, imgext = maskpath.split('/')[-1].split('.')[0], '.jpg'
            return os.path.join(self.img_path, filename) + imgext

        query_img = Image.open(maskpath_to_imgpath(query_mask_path)).convert('RGB')

        support_imgs = [Image.open(maskpath_to_imgpath(s_mask_path)).convert('RGB') for s_mask_path in support_mask_paths]

        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(s_mask_path) for s_mask_path in support_mask_paths]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id


    # def build_img_metadata(self):
    #     img_metadata = []
    #     for cat in self.categories:
    #         os.path.join(self.base_path, cat)
    #         img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat, 'test', 'origin'))])
    #         for img_path in img_paths:
    #             if os.path.basename(img_path).split('.')[1] == 'jpg':
    #                 img_metadata.append(img_path)
    #     return img_metadata

    def build_img_metadata_classwise(self):
        num_images=0
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            mask_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'masks', cat))])
            for mask_path in mask_paths:
                if self.read_mask(mask_path).count_nonzero() > 0: #no empty masks
                    img_metadata_classwise[cat] += [mask_path]
                    num_images += 1
        return img_metadata_classwise, num_images