# Adapt Before Comparision for Cross-Domain Few-Shot Segmentation (ABCDFSS)

[[`Paper`](https://arxiv.org/abs/2402.17614)] accepted for CVPR'24.


**Two options:**
1. **Predict an individual task**
2. **Predict tasks sampled from a dataset**

`task = {query image, support image(s), binary support mask(s)}`

## Predict an individual task

1. Prepare your files for the task.
2. Upload in the [`DEMO`](https://huggingface.co/spaces/heyoujue/ABCDFSS) OR `git clone` the [`huggingface repo`](https://huggingface.co/spaces/heyoujue/ABCDFSS) to either (a) call `from_model(task)` in `app.py` or (b) run the gradio app locally to let it use your GPU.

## Predict tasks sampled from a dataset
1. Prepare the dataset: [data/README.md](data/README.md).

2. Call
`python main.py --benchmark {} --datapath {} --nshot {}`,<br>
    for example
    `python main.py --benchmark deepglobe --datapath ./datasets/deepglobe/ --nshot 1`<br>
    Available `benchmark` strings: `deepglobe`,`isic`,`lung`,`fss`,`suim`.

Default is quick-infer mode.<br>
To change this, pass `--adapt-to every-episode`.<br>
To turn on post-processing, pass `--postprocessing [always|dynamic]`.<br>
To change other parameters, check the available parameters in [core/runner.py](core/runner.py) `makeConfig()`.<br>
Select `--verbosity 1` to get printed what's currently happening while runnning the loop.<br>
Consult [eval/README.md](eval/README.md) for notes on reproducing results.

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
