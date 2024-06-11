# Project Title: Analysis of Non-Periodic Activation Functions in Coordinate-MLPs

## Introduction
This project investigates the generalization capabilities of non-periodic activation functions in coordinate multilayer perceptrons (MLPs), focusing on overcoming the limitations of ReLU activations in representing continuous signals with high fidelity. Based on the foundational work by Sitzmann et al. and others, we explore several activation functions, including Gaussian, Quadratic, Laplacian, and ExpSin, for their potential in video fitting tasks and dynamic reconstruction.

| Gaussian | Laplacian | Quadratic| ExpSin|
|----------|-----------|----------|-------|
| $e^{\frac{-0.5x^2}{a^2}}$  | $e^{\frac{-\lvert x \rvert}{a}}$ | $\frac{1}{\sqrt{1 + (ax)^2}}$ |   $e^{-\sin(ax)}$ |


## Goals
1. Evaluate the fitting and generalization capabilities of selected activation functions on the video fitting task in ResFields.
2. Adapt the best-performing activation function for dynamic reconstruction tasks.

## Achievements  

- Apply all proposed activations to 2D Video approximation tasks
- Geometric initialization for the gaussian activation function 
- Apply the gaussian activation function to Dynamic NeRFs from 4 RGB views	

## Instructions
- See [Here](/docs/installation.md) to install all the required packages
- See [Here](/docs/data.md) to set up the datasets
- See [Here](/docs/benchmarking.md) on how to run various experiments and reproduce results from the project

## Results
Here we illustrate the outputs of our trained models and make give a description of what to look out for.


### 2D Video Approximation 

Below we have the cat video with diffent activations using resflied layers
1. Gaussian
2. Quadratic
3. Laplacian
4. Siren

![Description of GIF](./images/cat_resized.gif)
Upon closer observation we notice that the first 3 activations give a more visually appealing output compared to the siren. 

**NOTE** The cat gif above was compressed so that it could be uploaded to github. It was compressed using 
the following command
```bash
ffmpeg -i cat_resized.gif -vf "scale=-1:320" cat_resized.gif
```
### Dynamic NeRF (tNerf) + Geometric initialization

In order for the model to give any meaningful outputs it's weights must be initialized geometrically. That is, we initialize the weights such that the output of the initialized network is a sphere. Below we illustarte the loss metric as a function of time:

![Description of GIF](images/cat_resized.gif)

Since the gaussian function isn't as "nice" as the ReLU function some additional steps are taken such that the network actually outputs a sphere. 

![Description of GIF](images/cat_resized.gif)

We then trained the model for 400k iterations on different datasets to get the following results


![Description of GIF](images/cat_resized.gif)

## References
- [ResFields, ICLR24](https://markomih.github.io/ResFields)
- [Siren, NeurIPS22](https://www.vincentsitzmann.com/siren/)
- [Beyond Periodicity, ECCV22](https://arxiv.org/pdf/2111.15135.pdf)
- [SAL, CVPR20](https://arxiv.org/pdf/1911.10414)
- The activation function code was inspired by the following [github](https://github.com/kwea123/Coordinate-MLPs/blob/master/models.py) repo

## Team Members
- Bruce Balfour, [balfourb@student.ethz.ch](mailto:balfourb@student.ethz.ch)
- Ruben Schenk, [ruben.schenk@inf.ethz.ch](mailto:ruben.schenk@inf.ethz.ch)
- Alexandra Trofimova, [atrofimo@student.ethz.ch](mailto:atrofimo@student.ethz.ch)
