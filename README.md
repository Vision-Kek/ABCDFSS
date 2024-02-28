# Adapt Before Comparision: A New Perspective on Cross-Domain Few-Shot Segmentation

[[`Paper`](https://arxiv.org/abs/2402.17614)]

## Preparing Data
Because we follow the evaluation procedure of PATNet and Remember the Difference (RtD), please refer to their work for prepration of the following datasets:
- Deepglobe (PAT)
- ISIC (PAT)
- Chest X-Ray (Lung) (PAT)
- FSS-1000 (PAT)
- SUIM (RtD)

You do not need to get all datasets. Just prepare the one you want to test our method with.

## Python package prerequisites
1. torch
2. torchvision
3. cv2
4. numpy
5. for others, follow the console output

## Run it
Call
`python main.py --benchmark {} --datapath {} --nshot {}`

for example
`python main.py --benchmark deepglobe --datapath ./datasets/deepglobe/ --nshot 1`

Available `benchmark` strings: `deepglobe`,`isic`,`lung`,`fss`,`suim`. Easiest to prepare should be `lung` or `fss`.

Default is quick-infer mode.
To change this, set `config.featext.fit_every_episode=True` in the main file.
You can change all other parameters likewise, check the available parameters in `core/runner->makeConfig()`.

## Await it

You can experiment with this code. Before opening issues, I suggest awaiting nicer demonstrations and documentation to be added. 

## Citing
If you use ABCDFSS in your research, please use the following BibTeX entry.
```
@article{herzog2024cdfss,
      title={Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation}, 
      author={Jonas Herzog},
      journal={arXiv:2304.02643},
      year={2024}
}
```

