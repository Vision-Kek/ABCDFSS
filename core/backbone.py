from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

class Backbone(nn.Module):

    def __init__(self, typestr):
        super(Backbone, self).__init__()

        self.backbone = typestr

        # feature extractor initialization
        if typestr == 'resnet50':
            self.feature_extractor = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        else:
            raise Exception('Unavailable backbone: %s' % typestr)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []
        bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
        # Layer 0
        feat = self.feature_extractor.conv1.forward(img)
        feat = self.feature_extractor.bn1.forward(feat)
        feat = self.feature_extractor.relu.forward(feat)
        feat = self.feature_extractor.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
            res = feat
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.feat_ids:
                feats.append(feat.clone())

            feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats
