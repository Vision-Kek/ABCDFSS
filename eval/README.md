# Reproduce

## How to do in one step
Using the code, you need to

1. set `fit_every_episode = True` to simulate the scenario that each episode is adapted independently from scratch
2. append `self.pred_mask = self.apply_crf()` in the last line of the `SingleSampleEval.forward` function before the`return` statement, to run post-processing

Note that both steps will make the process dramatically slower, ~x50 for 1. Postprocessing in 2. is running only on CPU currently and thus slow. Moreover, the dynamic decision mentioned in the paper is not covered here, so that it will actually worsen predictions for Chest-Xray for instance. I am working on it to provide you a faster and more convenient way to reproduce the paper algorithms. In the meantime, I show you how I did it with some more code fragments than this repo

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

However, the postprocessing can make the results worse in some cases and we can predict whether this will happen by pseudo-predicting the support image (described in supplementary). Admittedly, it makes things a little less clear. Anyway, here's how it looks like:

**Part 2** - with dynamic refinement decision
1. Load episodes from Part 1.1
2. For each episode, load parameters from Part 1.2
3. For each episode, load coarse predictions from Part 1.3
4. For each episode, forward-pass pseudo-episode to make the yes/no decision whether to refine.
5. For each episode, if yes, run postprocessing(=refinement), if not, use coarse prediction from 1.3. Save.

I understand you want a way to reproduce the results more quickly, which I am working on for you.
