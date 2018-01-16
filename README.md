# student deep learning setup - GTX-1070s & GTX-980ti #
Deep Learning Setup for Student or on a Budget

I'm studying machine learning at [The George Tech](http://www.cc.gatech.edu/). In my free time I also play around with trying to reproduce various deep learning papers.

This page is a brief description of my setup which aimed to be affordable but decent. It is not designed to compete with 'professional' 8-card server machines or the like.

TL;DR - heat management is a real issue on home gear, the number of PCI lanes seriously affects training time on small mini batches (n <= 64). 3 cards is only marginally better than 2. 2 cards are better than 1.

## Equipment ##
Base equipment (purchased 2015):
* GIGABYTE GA-X99-UD3P motherboard (supports 40 pci lanes)
* Intel Core i7-5930K Haswell-E 6-Core 3.5 GHz LGA 2011 (40 pci lanes)
* 32 GB of DDR4 2400 (32 is not needed for these tests, I simply have that much for my large numpy models)
* Used GeForce GT 730 for running displays only.(this allows the entire memory of the 1070s to be used)

Originally I also purchased a GTX-980ti SC which had 6GB 384-Bit GDDR5. This 'superclocked' model has twin fans that vent out the side of the card as opposed to a 'blower' model which vents out the rear of the case. This become relevant lower down.

In early 2017 I decided it was time to upgrade and rather than purchase one GTX-1080 which were selling for approximately $600, I opted for two GTX-1070s which I acquired for about $380 each. My reasoning for this is so that I could used data parallelism to reduce training time and also experiment with model parallelism.

I ran a series of test on CIFAR-10 using torch code for a [wide-residual-network](https://github.com/szagoruyko/wide-residual-networks). This code provides an ability to use model parallelism. Below are some of the results of what I learned and what I might do differently.

### Test Results for Training a wide-residual-network ###

I ran the following repeated for the various configurations.
'''
model=wide-resnet widen_factor=10 depth=22 batchSize=64 dropout=0.3 nGPU=2 ./scripts/train_cifar.sh
'''

My ASUS X99 board can run 16x pci on PCI_1 and PCI_2. Based on how I've got my BIOS set up that works out to:

Device 0 = PCI_1 = 16x PCIe
Device 1 = PCI_3 = 8x PCIe
Device 2 = PCI_2 = 16x PCIe
Device 3 = PCI_4 = 8x PCIe

My GT 730 which drive the Ubuntu Desktop (2 monitors) resides in PCI_3 and registers as Device 1.

My test is against a WRN with depth 20 and width 10. I varied my batch size as indicated below.

#Batch Size = 64#

Test # | Setup | Seconds per Epoch
-------|------|--------
Test 1 |1 GTX-1070 in 16x PCI slot| 155.8 seconds
Test 2 |1 GTX-1070 in 16x, 1 GTX-980ti in 16x| 102.26 seconds
Test 3 |2 GTX-1070s both in 16x slots | 106.17 seconds
Test 4 |2 GPUs 1070 & 1 GTX-980ti, all @ 8x PCI  | 103.1 seconds. 
Test 5 |2 GTX-1070s in 8x slots|118.05 seconds

Test 1 shows the performance of a single GTX-1070 in a 16 lane PCI slot.

Test 2 & 3 show data-parallel performance using either two 1070s or one 1070 and one GTX-980ti. In this comparison we see that running part of the training on the 980ti makes the training slightly faster. If there were two 980tis rather than two 1070s, my belief is that the two 980ti would be faster than two 1070s. When I run the various sample scripts provided by NVIDIA, in particular matrixMulCUBLAS, I found that the 980ti was just a bit faster.

Test 4 show the performance of all three cards using data parallelism. On my system once there are 4 cards in the system, all lanes downgrade to 8x. (Unfortunately I didn't run the test by removing the GT 730 which would have allowed two of the cards to stay at 16x with only on at 8x)

Test 5 shows the two 1070s training but with all lanes forced to 8x PCI (simply by haveing the GT 730 and 980 in the slots, my system drops all lanes down to 8x PCI.

Interesting Notes: At least on my motherboard/CPU combination, running small batch sizes in a data parallel environment sees the bigest gain when moving from 1 card to 2 cards. The addition of the third card caused lanes to downshipt to 8x PCI and was actually a bit slower than just the two cards used in Test 2.

Also Test 5 seems to illustrate the performance penalty of small batch sizes relative to bus speed. The two 1070s running at 16x was definitely faster than when they were forced to operate at 8x.

#Batch Size = 128#

Test # | Setup | Seconds per Epoch
-------|------|--------
Test 1|1 GTX-1070 in 16x|148.49 seconds
Test 2|1 GTX-1070 in 16x and 980ti in 16x| 86 seconds
Test 3|2 GTX-1070s in 16x slots|87.1 seconds
Test 4|2 GTX-1070s & 1 GTX-980ti|76 seconds
Test 5|2 GTX-1070s in 8x slots| 97 seconds

With a larger batch size, we start to see the benefits of the 3rd card in Test 4. Test 2 and Test 5 reflect the performance difference between having 2 cards at 16x vs 2 cards at 8x PCI.

#Batch Size = 256#

Test # | Setup | Seconds per Epoch
-------|------|--------
Test 1|1 GTX-1070 in 16x |165.0 seconds 
Test 2|1 GTX-1070 in 16x and 980ti in 16x | 78.22 seconds 
Test 3|2 GTX-1070s in 16x slots | 79.3 seconds
Test 4|2 GTX-1070s & 1 GTX-980ti | 63 seconds
Test 5|2 GTX-1070s in 8x slots| 85.4 seconds

As the batch size increases, we're getting better utilizatio on the GPU when using data parallelism. 

## System Heat ##
When Runing with all three cards, my system got very close to termal limits of the cards. I attribute this largely to the particular cards I purchased. All of the EVGA cards are the SC class which has two side mounted fans and which vent vent out around the edges of the cooling fins and back into the case. 

Because the cards are stacked so close to each other, with only a few millimeters of clearnce, the heat coming off of one card feeds directly into the intake of the card above it. The cards very quickly reached termal limits and would not be suitable for sustained training sessions.

I suspect that either the 'Blower' Model or the Hydro Models would have eliminated this problem. My understanding now, after the fact, is that the 'Blower' models of the cards, (For example the reference models) have one fan mounted towards the back of the PCI car and blow the air across the cooling fins and out the back of the card. This would probably have made a substantial differnce in the temperatures. 

Water cooling would definitely have solved this problem. Given that I have a fairly large case, I had to buy these cards again, I would have opted for the Hydro Models or some combination of Hydro/Blower.


## Overall Impressions ##
I realize this is kind of sloppy testing. It was originally just for my own curiosity. 

That said, my conculsions are: Two Card is probably thw sweet spot for reducing training time if you're working on something where data-parallelism can be applied. The Addition of a 3rd (in my case) didn't seem to add much and had a negative effect on the thermal load on the cards. 

I think two GTX-1070s is probably a better deal than one GTX-1080. From what I've read the GTX-1080 is about 25% faster than the 1070. With data parallelism, two 1070s would likely be faster than one 1080 on most work loads.

It was also interesting to see the effects of running on 16x PCI lanes vs 8x PCI lanes. Using consumer gear, you will end up in an '8x' situation as soon as you move to 3 or 4 cards. This had a performance impact when using small batch sizes. It's possble, that I could have removed the GT-730 and place two GPUs in 16x slots and had only the thrid in an 8x slot. I didn't test this.
