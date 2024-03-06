## Table of contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Data Prepration](#data_prepration)
4. [Model](#model)
    - [Wav2Vec](#wav2vec)
    - [Whisper](#whisper)

5. [Results](#results)


## <a name='introduction'></a> Introduction
In the past ten years, thanks to the Stochastic Gradient Descent, deep learning has achieved remarkable success with a vast amount of data and parameters. In backpropagation, parameters are updated using the gradients computed through backpropagation. One of the significant drawbacks of backpropagation is that it is completely dependent on the computation information of the derivatives in the forward pass. When a black box is introduced into the forward pass, backpropagation becomes unfeasible unless a differentiable model of the black box is developed. That's why in December 2022, Geoffrey Hinton introduced a proper alternative for Backpropagation called Forward-Forward algorithm that was a breakthrough in this field.

The Forward-Forward algorithm is a greedy multi-layer learning approach that is inspired by Boltzmann machines and Noise Contrastive Estimation. In this approach, the forward and backward passes of backpropagation are replaced with two forward passes that have the same operations but work on different data and have opposite objectives. The positive pass adjusts the weights to improve the goodness in every hidden layer using "positive data", while the negative pass adjusts the weights to reduce the goodness in every hidden layer using "negative data".

The Forward-Forward algorithm is somewhat slower that Backpropagation but their speed is comparable. One of the advantages of Forward-Forward algorithm is that it can be utilized even when the exact details of the forward computation are unknown. Moreover, it has the benefit of being able to learn while piplining sequential data through a neural network without having to store neural activities or pause to propagate error derivatives. One of the main drawback of this approach is that it cannot generalize very well on several of the toy problems investigated in this project. The Forward-Forward algorithm has potential superiority over backpropagation in two areas: as a model for learning in the cortex and as a means of utilizing low-power analog hardware without resorting to reinforcement learning.

In this project, the application and goodness of Forward-Forward algorithm is supposed to be investigated in the field of Speech Processing. All of the implementations are based on Convolutional Neural Networks (CNNs). Morover, the effectiveness of this approach in case of training a CNN model from scratch and training a CNN model using Transfer Learnign method is explored. It should be noted that, in this project differernt datasets with assorted number of samples are employed with the aim of demonstrating the effectiveness of this approach in the field of speech processing and considering various scenarios. Accordingly, three kinds of public datasets were employed named "Ausio MNIST", "Human Speech Emotion Classification", and "Google Speech Command".

There are several innovations in this project that were adopted in order to achieve promising results: 1) The way of generating the "Nagative Samples" 2) True labes were injected into the model, in the first layer of the Fully-Connected part of the CNN with the aim of preventing from forgetting the labels.

In the following, we will discuss more about our datasets, methods, and results. In the Section 2, the important steps have taken to achieve our goal is investigated. Then in the Section 3, A list of the machine learning techniques used to solve our problem are introduced and the corresponding hyperparameters are reported. Finally, in the Section 4 and 5, the obtained results are presented and analyzed and a short sumurization of the chosen approach is carried out.


## <a name='methodology'></a> Methodology
Consider a network that consists of $k$ feed-forward layers with non-linearity activation functions such as ReLU. In normal supervised training, for an input image, an objective function (typically cross-entropy) is calculated on the outputs of the network and the ground truth label to measure the goodness of the network. This signal is then backpropagated throughout the network, and all layers are updated accordingly, to show a better goodness score.

The Forward-Forward algorithm eliminates the need for backpropagation by using two forward passes, namely, a positive pass and a negative pass. Consider an input image $I$ where the task is to classify it from $k$ target classes. In Forward-Forward, the first $k$ pixels of the first row of the image (top-left) is replaced with the one-hot encoding of a  label $0 <= L < k$. If the one-hot encoding $L$ corresponds to the ground-truth label if the input image $I$, the resulting image is considered a positive sample $I_P$, and if $L$ is different from the ground-truth label, the corresponding image is considered a negative sample $I_N$. For each batch of data, we will get one $I_P$ image, and we sample a single negative sample $I_N$ randomly. At step 0, the outputs of the first layer are calculated for the set of positive $I_P$ and negative $I_N$ samples. Let $g_{pos}$ and $g_{neg}$ represent the feature representations of this layer for a batch of data. The following objective function is then calculated as the loss:

    `loss = torch.log(1 + torch.exp(torch.cat([-g_pos.pow(2) + self.threshold, 
                                                 g_neg.pow(2) - self.threshold]))).mean()`

This loss function encourages the positive data to have a high norm, and the negative data to have a smaller norm, using a threshold. If trained properly, for images with correct labels encoded inside them, the network outputs highly activated neurons, whereas for images with wrong labels inside them, the network outputs would be close to zero. Therefore, for a target image during test time, they encode it $L$ times, using each one-hot encoding of each label. This gives us a set of $L$ images corresponding to the same test image. Then, they pass the resulting encoded images through the network and calculate $L$ activation values. The arg max of these activation values is considered the predicted label for the input test image.

Another distinction between Forward-Forward and normal training is that at each step $i$, they optimize only the $i$th layer of the network (the target layer), and the gradients are prohibited from getting backpropagated through that layer, and the layer is fully optimized, before moving on to optimizing the next layer.

In this work, we extend this method to Convolutional Neural Networks in the audio and speech classification domain. We first extract FBANK features of the raw waveform inputs. Then, we augment the resulting FBANKs using the same label encoding procedure to get $I_P$ and $I_N$ data, for each sample in the batch. Then, use a CNN to extract features from $I_P$ and $I_N$ that give us the $g_{pos}$ and $g_{neg}$ feature representations for a target layer, and we use the same loss function to update the weights of the target convolutional layer.

The receptive field of a feed-forward layer is the whole input, and thus it does not matter where the one-hot encoded label is located. However, that is not the case in convolutional layers where the receptive field is only a few pixels in the input space (especially in the first layer where the receptive field is the kernel size). Therefore, for encoding the image with the one-hot-encoding of the label, instead of only the first layer, we propose encoding it in the top left square of the image. As mentioned above, the one-hot-encoding of a label will have $L$ values. We put this one-hot-encoding in the first $\sqrt{L}$ rows of the image. For instance, if we have $L=25$ target labels, encode the label in a square of $5 \times 5$ and put it on the top-left of the image. We observe that just doing so increases the accuracy by 5%.

Finally, we observe that the forward-forward algorithm is extremely effective in a transfer learning setting. Consider a pre-trained and frozen neural network (of any kind) $F$, getting transferred to a target classification dataset with $L$ labels. For every input $I$, we first calculate the outputs of the pre-trained model $z = F(I)$, where $z$ is the output of the model. Inspired by the Forward-Forward algorithm, we concatenate the one-hot encoding of labels to $z$, which leads to a positive embedding $z_P$ if the added label-encoding corresponded to the ground-truth of the input $I$, and a negative embedding $z_N$ otherwise. We then train a few feed-forward layers on top of the pre-trained network $F$ using the same procedure as mentioned above. In some settings, doing so outperforms normal training by a large margin, unraveling the great potential of the Forward-Forward algorithm in transfer learning.

In what follows, the most important implementation steps such as preprocessing, feature extraction, and model architecture are further explained.
