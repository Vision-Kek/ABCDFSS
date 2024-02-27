# Adapt Before Comparision - A New Perspective on Cross-Domain Few-Shot Segmentation

Code for the Reproducing the Paper

## Preparing Data
Because we follow PATNet and RtD, please refer to their work for prepration of the following datasets:
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

Available `benchmark` strings: deepglobe,isic,lung,fss,suim
Easiest to prepare should be Lung or FSS.

Default is quick-infer mode.
To change this, set `config.featext.fit_every_episode = True` in the main file.
You can change all other parameters likewise, check the available parameters in runner.makeConfig.
