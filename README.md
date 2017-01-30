# student deeplearning setup
Deep Learning Setup for Student or on a Budget

I'm studying machine learning at [The George Tech](http://www.cc.gatech.edu/). In my free time I also play around with trying to reproduce various deep learning papers.

This page is a brief description of my setup which aimed to be affordable but decent. It is not designed to compete with 'professional' 8-card server machines or the like.

Base equipment (purchased 2015):
* GIGABYTE GA-X99-UD3P motherboard (supports 40 pci lanes)
* Intel Core i7-5930K Haswell-E 6-Core 3.5 GHz LGA 2011 (40 pci lanes)
* 32 GB of DDR4 2400 (32 is not needed for these tests, I simply have that much for my large numpy models)
* Used GeForce GT 730 for running displays only.

Originally I also purchased a GTX-980ti SC which had 6GB 384-Bit GDDR5. This 'superclocked' model has twin fans that vent out the side of the card as opposed to a 'blower' model which vents out the rear of the case. This become relevant lower down.

In early 2017 I decided it was time to upgrade and rather than purchase one GTX-1080 which were selling for approximately $600, I opted for two GTX-1070s which I acquired for about $380 each. My reasoning for this is so that I could used data parallelism to reduce training time and also experiment with model parallelism.

I ran a series of test on CIFAR-10 using torch code for a [wide-residual-network](https://github.com/szagoruyko/wide-residual-networks). This code provides an ability to use model parallelism. Below are some of the results of what I learned and what I might do differently.

TL;DR - heat management is a real issue on home gear, the number of PCI lanes seriously affects training time on small mini batches (n <= 64). 3 cards is only marginally better than 2. 2 cards are better than 1.

### In progress 2017-01-28 - Writing up my test data which will go here ###

