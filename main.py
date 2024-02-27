from core import runner
import torch
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Adapt Before Comparison - A New Perspective on Cross-Domain Few-Shot Segmentation')

    # common
    parser.add_argument('--benchmark', type=str, default='lung', choices=['fss', 'deepglobe', 'lung', 'isic', 'fss', 'lung'])
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--nshot', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_opts()
    print(args)
    runner.args.benchmark = args.benchmark
    runner.args.datapath = args.datapath
    runner.args.nshot = args.nshot

    dataloader = runner.makeDataloader()
    config = runner.makeConfig()
    feat_maker = runner.makeFeatureMaker(dataloader.dataset, config, device=device)
    average_meter = runner.AverageMeterWrapper(dataloader, device)

    for idx, batch in enumerate(dataloader):
        sseval = runner.SingleSampleEval(batch, feat_maker)
        sseval.forward()
        sseval.calc_metrics()
        average_meter.update(sseval)
        average_meter.write(idx)
    print('Result m|FB:', average_meter.average_meter.compute_iou())