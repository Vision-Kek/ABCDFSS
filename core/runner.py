from data.dataset import FSSDataset
from core.backbone import Backbone
from eval.logger import Logger, AverageMeter
from eval.evaluation import Evaluator
from utils import commonutils as utils
import utils.segutils as segutils
import utils.crfhelper as crfutils
import core.contrastivehead as ctrutils
import core.denseaffinity as dautils
import torch

def set_args(_args):
    global args
    # _args should write benchmark, datapath, nshot, adapt-to, postprocessing, logpath, verbosity
    args = _args

    # then some more args are appended
    args.backbone = 'resnet50'
    args.nworker = 0
    args.bsz = 1 # the method works on a single task, hence bsz=1
    args.fold = 0


def makeDataloader():
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    return dataloader


def makeConfig():
    config = ctrutils.ContrastiveConfig()
    config.fitting.protoloss = False
    config.fitting.o_t_contr_proto_loss = True
    config.fitting.selfattentionloss = False
    config.fitting.keepvarloss = True
    config.fitting.symmetricloss = False
    config.fitting.q_nceloss = True
    config.fitting.s_nceloss = True
    config.fitting.num_epochs = 25
    config.fitting.lr = 1e-2
    config.fitting.debug = args.verbosity > 2
    config.model.out_channels = 64
    config.model.debug = args.verbosity > 0
    config.featext.fit_every_episode = False
    config.aug.blurkernelsize = [1]
    config.aug.n_transformed_imgs = 2
    config.aug.maxjitter = 0.0
    config.aug.maxangle = 0
    config.aug.maxscale = 1
    config.aug.maxshear = 20
    config.aug.apply_affine = True
    config.aug.debug = args.verbosity > 2
    return config


def makeFeatureMaker(dataset, config, device='cpu', randseed=2, feat_extr_method=None):
    utils.fix_randseed(randseed)
    if feat_extr_method is None:
        feat_extr_method = Backbone(args.backbone).to(device).extract_feats
    feat_maker = ctrutils.FeatureMaker(feat_extr_method, dataset.class_ids, config)
    utils.fix_randseed(randseed)
    feat_maker.norm_bb_feats = False
    return feat_maker


# Motivation of this class: Handle every task individually -> create an object for each task, example: see main.py
class SingleSampleEval:
    def __init__(self, batch, feat_maker):
        self.damat_comp = dautils.DAMatComparison()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.feat_maker = feat_maker
        self.thresh_method = 'pred_mean'
        self.post_proc_method = 'off'
        self.verbosity = args.verbosity

    def taskAdapt(self, detach=True):
        b = self.batch
        if self.device.type == 'cuda': b = utils.to_cuda(b)
        self.q_img, self.s_img, self.s_mask, self.class_id = b['query_img'], b['support_imgs'], b['support_masks'], b[
            'class_id'].item()
        self.task_adapted = self.feat_maker.taskAdapt(self.q_img, self.s_img, self.s_mask, self.class_id)

    def compare_feats(self):
        if self.task_adapted is None:
            print("error, do task adaption first")
            return None
        self.logit_mask = self.damat_comp.forward(self.task_adapted[0], self.task_adapted[1], self.s_mask)
        return self.logit_mask

    def threshold(self, method=None):
        if self.logit_mask is None:
            print("error, calculate logit mask first (do forward pass)")
        if method is None:
            method = self.thresh_method
        self.thresh = segutils.calcthresh(self.logit_mask, self.s_mask, method)
        self.pred_mask = (self.logit_mask > self.thresh).float()
        return self.thresh, self.pred_mask

    def postprocess(self):
        if self.post_proc_method == 'off':
            apply = False
        elif self.post_proc_method == 'always':
            apply = True
        elif self.post_proc_method == 'dynamic':
            apply = crfutils.crf_is_good(self)
        else:
            apply = False
            print(f'Unknown postproc method: {self.post_proc_method=}')
        return crfutils.apply_crf(self.q_img, self.logit_mask, segutils.thresh_fn(self.thresh_method)).to(self.device) if apply else self.pred_mask

    # this method calls above components sequentially
    def forward(self):
        self.taskAdapt()

        self.logit_mask = self.compare_feats()

        self.thresh, self.pred_mask = self.threshold()

        self.pred_mask = self.postprocess()

        return self.logit_mask, self.pred_mask

    def calc_metrics(self):
        # assert torch.logical_or(self.logit_mask<0, self.logit_mask>1).sum()==0, display(tensor_table(logit_mask=self.logit_mask))
        self.area_inter, self.area_union = Evaluator.classify_prediction(self.pred_mask, self.batch)
        self.fgratio_pred = self.pred_mask.float().mean()
        self.fgratio_gt = self.batch['query_mask'].float().mean()
        return self.area_inter[1] / self.area_union[1]  # fg-iou

    def plots(self):
        display(pilImageRow(norm(self.logit_mask[0]), (self.logit_mask[0] > self.thresh).float(), self.pred_mask,
                            self.batch['query_mask'][:1], norm(self.q_img[0]), norm(self.s_img[0, 0])))
        display(segutils.tensor_table(probs=self.logit_mask))

        print('s_mask.mean, pred_mask.mean, thresh:', self.s_mask.mean().item(), self.logit_mask.mean().item(),
              self.thresh.item())


class AverageMeterWrapper:
    def __init__(self, dataloader, device='cpu', initlogger=True):
        if initlogger: Logger.initialize(args, training=False)
        self.average_meter = AverageMeter(dataloader.dataset, device)
        self.device = device
        self.dataloader = dataloader
        self.write_batch_idx = 50

    def update(self, sseval):
        self.average_meter.update(sseval.area_inter, sseval.area_union, torch.tensor(sseval.class_id).to(self.device),
                                  loss=None)

    def update_manual(self, area_inter, area_union, class_id):
        if isinstance(class_id, int): class_id = torch.tensor(class_id).to(self.device)
        self.average_meter.update(area_inter, area_union, class_id, loss=None)

    def write(self, i):
        self.average_meter.write_process(i, len(self.dataloader), 0, self.write_batch_idx)

