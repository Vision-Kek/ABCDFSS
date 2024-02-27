r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetDeepglobe(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num_val=600):
        self.split = split
        self.benchmark = 'deepglobe'
        self.shot = shot
        self.num_val = num_val

        self.base_path = os.path.join(datapath)
        self.to_annpath = lambda p: p.replace('jpg', 'png').replace('origin', 'groundtruth')

        self.categories = ['1','2','3','4','5','6']

        self.class_ids = range(0, 6)
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


    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        ann_path = os.path.join(self.base_path, query_name.split('/')[-4], 'test', 'groundtruth')
        query_name = os.path.join(ann_path, query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(ann_path, sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

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
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat, 'test', 'origin'))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
                    num_images += 1
        return img_metadata_classwise, num_images
