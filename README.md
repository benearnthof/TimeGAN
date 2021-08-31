# TimeGAN
A pytorch implementation of  Time-series Generative Adversarial Networks (https://github.com/jsyoon0823/TimeGAN) 

## Project Description
The Goal was to create smoothed time series data via a GAN. This should be achieved via a combination of https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf and https://arxiv.org/pdf/2106.10414.pdf in PyTorch.

## Summary 
A recreation of the results of the original Time GAN paper is very hard to achieve. This is possibly due to a number of reasons. With the training time and computational power that was within our reach, it seems like our Generator tended strongly to learning one specific simple curve, often shaped like a hook, a right angle or a straight line. This was verified during multiple training runs, and can be seen in the demo notebooks. The reason might also lie in the nature of GANs, as the training process is often very fiddly and unstable. (Thanks to Tobias Webers numerous hints in his mail.). However, during one run, our Generator managed to escape the basic shape that occured most of the times, and started producing two different smoothed out curves. This progress was also visible in the visual comparison. Please refer to the best demo run notebook for these results. The third and final reason for the difficulties we faced, might lie in the paper itself, as many people struggled to reproduce the claimed results as can be seen in the github issues of the original implementation. For one example, the Authors claim they used a Wiener Process to sample data for the paper, but their implementation uses a uniform generator instead.

In general, we think we might be able to produce better results with careful selection of hyper parameters and more extensive training. As we struggled to reproduce the Time GAN results, we did not conduct the implementation of the ada FNN layer as we did not expect a positive result on the outcome. 
Additionally, the AdaFNN code was already fully available in Pytorch, so there it would have made no sense to "translate" the code. In general it should be possible to swap out the Embedding network in TimeGAN for an AdaFNN layer. Working with functional data should greatly benefit from this added "compression" step and lead to better learning in the Generator. 

## Repository Structure: 
requirements.txt contains all dependencies and can be run with pip: 
pip install -r requirements.txt 

utils.py contains all helper functions, mostly from the original repository

preprocess_eeg_data.R is used to preprocess the eeg data provided to us and make it usable for TimeGAN

modules_and_training contains the main implementation of TimeGAN. The Network blocks are defined there, aswell as a function that runs training and returns the trained networks. 

demo_multivariat.ipynb and demo_univariat.ipynb contain demos for the respective cases, aswell as analyses of their results

best_demo_run.ipynb contains the best training run we obtained while experimenting with different hyperparameter settings. 