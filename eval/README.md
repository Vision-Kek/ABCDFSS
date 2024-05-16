# Reproduce

## How to do in one step


Handle quick-infer mode: ❌`OFF`:   `--adapt-to every-episode` ✅`ON`: `--adapt-to first-episode`

Handle no-pp mode: ❌`OFF`:   `--postprocessing dynamic` ✅`ON`: `--postprocessing off`

For the terms quick-infer and no-pp, please refer to the paper. For the main table both are off.
Note that turning them off will make the process dramatically slower, ~x50 for 1. Postprocessing in 2. is running only on CPU currently and thus slow. I am working on implementing the algorithms more efficiently and will update the repo then. You can get faster results with choosing postprocessing to be done `always` instad of `dynamic`, but that will worsen some predictions, for Chest-Xray for instance. If you prefer, you can also run it in two steps, to save time and resources, since only Part 1 makes use of GPU currently:

## How I did it

I have obtained the results in a process with two sequential parts:

**Part 1** [Notebook](https://www.kaggle.com/code/heyoujue/resusable-reproducable-bias-free-modular-testing)
1. Sample episodes, each equals to query and support set (i.e. the file names to images and masks). Save.
2. For each episode, estimate the attached layer's parameters from scratch. Save.
3. For each episode, forward pass to generate coarse prediction. Save.

At this stage, we already have the unrefined results (no-pp).

Saving episodes means saving their file name arrays, saving parameters means saving their state dict (for each episode).
Then, to refine:

**Part 2**  [Notebook](https://www.kaggle.com/code/heyoujue/postprocessing)
1. Load episodes from Part 1.1
2. For each episode, load coarse predictions from Part 1.3
3. For each episode, run postprocessing(=refinement). Save.

With this, we would have also the post-processed results.

However, applying postprocessing `always` can make the results worse in some cases and we can predict whether this will happen by pseudo-predicting the support image (described in supplementary). With this, Part 2 changes to:

**Part 2** - with `dynamic` refinement decision
1. Load episodes from Part 1.1
2. For each episode, load parameters from Part 1.2
3. For each episode, load coarse predictions from Part 1.3
4. For each episode, forward-pass pseudo-episode to make the yes/no decision whether to refine.
5. For each episode, if yes, run postprocessing(=refinement), if not, use coarse prediction from 1.3. Save.

Please contact me if you want to access the notebooks.
