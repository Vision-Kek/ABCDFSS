
## Dataset Prepration

In general, I followed the evaluation procedure from [1][2], using 5 datasets
- [1] Lei et al.`benchmark={Deepglobe, ISIC, Chest-XRay, FSS-1000}`
- [2] Wang et al.: `SUIM`

**But**, it is still cumbersome to make the data ready to be loaded with the dataloader, so I provide my readily usable datasets, which you can download and put into a `datasets` directory:

## [ISIC](https://www.kaggle.com/datasets/heyoujue/isic2018-classwise)
pass `datapath=./datasets/isic2018-classwise`

## [Chest-Xray (aka Lung)](https://www.kaggle.com/datasets/heyoujue/lungsegmentation)
pass `datapath=./datasets/lungsegmentation`

## [FSS-1000](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/fss1000-a-1000-class-fewshot-segmentation)
pass `datapath=./datasets/fss1000-a-1000-class-fewshot-segmentation`

## [Deepglobe](https://www.kaggle.com/datasets/heyoujue/deepglobe)
pass `datapath=./datasets/deepglobe`

Lei et al. suggested to preprocess the orginial Deepglobe dataset. They have uploaded their processed dataset only in 2023.
I have run the preprocessing before that.
The preprocessing code they provided needed to be fixed before it was runnable, [here](https://www.kaggle.com/code/heyoujue/preprocessing-inputs-for-patnet/notebook) is my fixed procedure.
You don't need to run it though, you can use its [output](https://www.kaggle.com/datasets/heyoujue/deepglobe).

## [SUIM](https://www.kaggle.com/datasets/heyoujue/suim-merged)
pass `datapath=./datasets//kaggle/input/suim-merged/suim_merged`

SUIM comes usually with a train and test split, but because all data serves as test for CD-FSS, the entire dataset needs to be considered.
If you only predict the images in the TEST folder of the original SUIM, you will get better results because these images seem to be easier, don't be fooled by it. On the other hand, 3859 episodes can take a bit long, I sampled 1650 only.

## Your own dataset
You can integrate your own dataset by copying and adjusting the procedure from the other datasets in the data/ directory
1. create a module `your_dataset.py` with a class `DatasetYours` inheriting from `torch.utils.data.Dataset` just like e.g. `DatasetLung` in `lung.py`
2. adjust the paths to your images and masks in the code below
2. register it in `dataset.py`'s `FSSDataset.datasets` dictionary, then you can use the key you chose as `args.benchmark` for inference.

## Further Considerations
Overall, the dataset composition from previous work might not be optimal due to the following issues
- Deepglobe: Zoomed crops have severe annotation issues. Please see Supplementary Material.
- FSS-1000: is not cross-domain w.r.t either ImageNet or PASCAL
- ISIC: There are three classes, the samples are unbalanced`70%, 22%, 8%`, the 8% class is the hardest, please do not cheat like [3] who ignored classes and calculated mIoU thus by `70%*71IoU + 22%*31IoU, 8%*21IoU%` instead of `(71+31+21)/3`. Moreover, some issues with ISIC are (i) if you look at a few examples from each class, they look pretty much the same, the class disntinction is not really meaningful for the few-shot setting,  (ii) at the same time, the high intra-class variance makes it unrealistically difficult (presumably nobody would provide a support image that looks entirely different from all the queries that are to be segmented) (iii) given (ii) combined with the fact that almost always the object is centered and salient, a salient object detector disregarding the support image would likely perform better, it disincentives developers from creating models that semantically match the support object
- SUIM: as in ISIC(ii), high intra-class variations make it unrealistically ill-posed for FSS, e.g. turtle and dolphin are the same category, but an underwater robot that looks like a turtle is a different category, so if support image masks a turtle and in the query image there is both robot and dolphin, it is ambiguous.

As a suggestion, some recent work testing SAM etc. across domains could be a reference for alternatives.

## References

[1] Lei et al.: Cross-Domain Few-Shot Semantic Segmentation, ECCV22

[2] Wang et al.: Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer, CVPR22

[3] Chen et al.: Pixel Matching Network for Cross-Domain Few-Shot Segmentation, WACV24
