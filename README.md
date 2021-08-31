# TimeGAN
A pytorch implementation of  Time-series Generative Adversarial Networks (https://github.com/jsyoon0823/TimeGAN) 

## Project Description
The Goal was to create smoothed time series data via a GAN. This should be achieved via a combination of https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf and https://arxiv.org/pdf/2106.10414.pdf in PyTorch.

## Summary 
A recreation of the results of the original Time GAN paper is very hard to achieve. This is possibly due to a number of reasons. With the training time and computational power that was within our reach, it seems like our Generator tended strongly to learning one specific simple curve, often shaped as a hook, a right angle or a straight line. This was verified during multiple training runs, and can be seen in the demo notebooks. The reason might also lie in the nature of GANs, as the training process is often very fiddly and unstable. (Also See Tobias Webers Mail.). However, during one run, our Generator managed to escape the basic shape that occured most of the times, and started producing two different smoothed out curves. This progress was also visible in the visual comparison. Please referr two the best demo run notebook for these results. The third and final reason for the difficulties we faced, might lie in the paper itself, as many people struggle to reproduce the claimed results as can be seen in the github issues of the original implementation. 
