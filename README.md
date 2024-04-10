# Adapt Before Comparision for Cross-Domain Few-Shot Segmentation (ABCDFSS)

[[`Paper`](https://arxiv.org/abs/2402.17614)] accepted for CVPR'24.


**Two options:**
1. **Predict an image**
2. **Predict a dataset**

## Predict an image

1. Prepare your files for the task: `Task = {query image, support image, binary support mask}`.
2. Upload in the [`DEMO`](https://huggingface.co/spaces/heyoujue/ABCDFSS) OR `git clone` the [`huggingface repo`](https://huggingface.co/spaces/heyoujue/ABCDFSS) to either (a) call `from_model(Task) -> prediction` or (b) run the gradio app locally to let it use your GPU.

## Predict a dataset
Prepare the dataset: `data/README.md`.

Call
`python main.py --benchmark {} --datapath {} --nshot {}`

for example
`python main.py --benchmark deepglobe --datapath ./datasets/deepglobe/ --nshot 1`

Available `benchmark` strings: `deepglobe`,`isic`,`lung`,`fss`,`suim`.

Default is quick-infer mode.
To change this, set `config.featext.fit_every_episode=True` in the main file.
You can change all other parameters likewise, check the available parameters in `core/runner.py makeConfig()`.
Consult `eval/README.md` for notes on reproducing results.

## Limitations
This work might give you inspiration to try some adaption before comparison for CD-FSS. You might be interested in my opinion that
1. It is quite possible that there is a better specific adaption algorithm that you can find in your research.
2. It is also reasonable to replace the part after the comparison with a learned network, this work only demonstrated that even without such, one can get better results than previous works.
3. Lastly, for the latest best performance, you might want to refer to the other concurrent CD-FSS works.

## Citation
If this work finds use in your research, please cite:
```
@article{herzog2024cdfss,
      title={Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation}, 
      author={Jonas Herzog},
      journal={arXiv preprint arXiv:2402.17614},
      year={2024}
}
```
