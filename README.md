## Table of contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
    - [Preprocessing](#preprocessing)
    - [Model](#model)
4. [Experimental Setup](#ٍexperimental_setup)
    - [Dataset](#dataset)
    - [List of Machine Learning techniques and their corresponding hyperparameters](#ML_setting)

6. [Experimental Results](#experimental_results)
    - [Audio MNIST](#AUDIOMNIST)
    - [Human Speech Emotion Classification (RAVDESS)](#ravdess)
    - [Google Speech Dataset](#google_speech)



## <a name='introduction'></a> Introduction
In the past ten years, thanks to the Stochastic Gradient Descent, deep learning has achieved remarkable success with a vast amount of data and parameters. In backpropagation, parameters are updated using the gradients computed through backpropagation. One of the significant drawbacks of backpropagation is that it is completely dependent on the computation information of the derivatives in the forward pass. When a black box is introduced into the forward pass, backpropagation becomes unfeasible unless a differentiable model of the black box is developed. That's why in December 2022, Geoffrey Hinton introduced a proper alternative for Backpropagation called Forward-Forward algorithm which was a breakthrough in this field.

The Forward-Forward algorithm is a greedy multi-layer learning approach that is inspired by Boltzmann machines and Noise Contrastive Estimation. In this approach, the forward and backward passes of backpropagation are replaced with two forward passes that have the same operations but work on different data and have opposite objectives. The positive pass adjusts the weights to improve the goodness in every hidden layer using "positive data", while the negative pass adjusts the weights to reduce the goodness in every hidden layer using "negative data".

The Forward-Forward algorithm is somewhat slower than Backpropagation but their speed is comparable. One of the advantages of Forward-Forward algorithm is that it can be utilized even when the exact details of the forward computation are unknown. Moreover, it has the benefit of being able to learn while pipelining sequential data through a neural network without having to store neural activities or pause to propagate error derivatives. One of the main drawbacks of this approach is that it cannot generalize very well on several of the toy problems investigated in this project. The Forward-Forward algorithm has potential superiority over backpropagation in two areas: as a model for learning in the cortex and as a means of utilizing low-power analog hardware without resorting to reinforcement learning.

In this project, the application and goodness of Forward-Forward algorithm are supposed to be investigated in the field of Speech Processing. All of the implementations are based on Convolutional Neural Networks (CNNs). Moreover, the effectiveness of this approach in the case of training a CNN model from scratch and training a CNN model using the Transfer learning method is explored. It should be noted that, in this project, different datasets with assorted numbers of samples are employed with the aim of demonstrating the effectiveness of this approach in the field of speech processing and considering various scenarios. Accordingly, three kinds of public datasets were employed named "Ausio MNIST", "Human Speech Emotion Classification", and "Google Speech Command".

There are several innovations in this project that were adopted in order to achieve promising results: 1) The way of generating the "Negative Samples" 2) True labels were injected into the model, in the first layer of the Fully-Connected part of the CNN with the aim of preventing from forgetting the labels.

In the following, we will discuss more about our datasets, methods, and results. In Section 2, the important steps taken to achieve our goal are investigated. Then in Section 3, A list of the machine-learning techniques used to solve our problem are introduced and the corresponding hyperparameters are reported. Finally, in Sections 4 and 5, the obtained results are presented and analyzed and a short summarization of the chosen approach is carried out.


## <a name='methodology'></a> Methodology
Consider a network that consists of $k$ feed-forward layers with non-linearity activation functions such as ReLU. In normal supervised training, for an input image, an objective function (typically cross-entropy) is calculated on the outputs of the network and the ground truth label to measure the goodness of the network. This signal is then backpropagated throughout the network, and all layers are updated accordingly, to show a better goodness score.

The Forward-Forward algorithm eliminates the need for backpropagation by using two forward passes, namely, a positive pass and a negative pass. Consider an input image $I$ where the task is to classify it from $k$ target classes. In Forward-Forward, the first $k$ pixels of the first row of the image (top-left) is replaced with the one-hot encoding of a  label $0 <= L < k$. If the one-hot encoding $L$ corresponds to the ground-truth label if the input image $I$, the resulting image is considered a positive sample $I_P$, and if $L$ is different from the ground-truth label, the corresponding image is considered a negative sample $I_N$. For each batch of data, we will get one $I_P$ image, and we sample a single negative sample $I_N$ randomly. At step 0, the outputs of the first layer are calculated for the set of positive $I_P$ and negative $I_N$ samples. Let $g_{pos}$ and $g_{neg}$ represent the feature representations of this layer for a batch of data. The following objective function is then calculated as the loss:

    loss = torch.log(1 + torch.exp(torch.cat([-g_pos.pow(2) + self.threshold, 
                                                 g_neg.pow(2) - self.threshold]))).mean()

This loss function encourages the positive data to have a high norm, and the negative data to have a smaller norm, using a threshold. If trained properly, for images with correct labels encoded inside them, the network outputs highly activated neurons, whereas for images with wrong labels inside them, the network outputs would be close to zero. Therefore, for a target image during test time, they encode it $L$ times, using each one-hot encoding of each label. This gives us a set of $L$ images corresponding to the same test image. Then, they pass the resulting encoded images through the network and calculate $L$ activation values. The arg max of these activation values is considered the predicted label for the input test image.

Another distinction between Forward-Forward and normal training is that at each step $i$, they optimize only the $i$th layer of the network (the target layer), and the gradients are prohibited from getting backpropagated through that layer, and the layer is fully optimized, before moving on to optimizing the next layer.

In this work, we extend this method to Convolutional Neural Networks in the audio and speech classification domain. We first extract FBANK features of the raw waveform inputs. Then, we augment the resulting FBANKs using the same label encoding procedure to get $I_P$ and $I_N$ data, for each sample in the batch. Then, use a CNN to extract features from $I_P$ and $I_N$ that give us the $g_{pos}$ and $g_{neg}$ feature representations for a target layer, and we use the same loss function to update the weights of the target convolutional layer.

The receptive field of a feed-forward layer is the whole input, and thus it does not matter where the one-hot encoded label is located. However, that is not the case in convolutional layers where the receptive field is only a few pixels in the input space (especially in the first layer where the receptive field is the kernel size). Therefore, for encoding the image with the one-hot-encoding of the label, instead of only the first layer, we propose encoding it in the top left square of the image. As mentioned above, the one-hot-encoding of a label will have $L$ values. We put this one-hot-encoding in the first $\sqrt{L}$ rows of the image. For instance, if we have $L=25$ target labels, encode the label in a square of $5 \times 5$ and put it on the top-left of the image. We observe that just doing so increases the accuracy by 5%.

Finally, we observe that the forward-forward algorithm is extremely effective in a transfer learning setting. Consider a pre-trained and frozen neural network (of any kind) $F$, getting transferred to a target classification dataset with $L$ labels. For every input $I$, we first calculate the outputs of the pre-trained model $z = F(I)$, where $z$ is the output of the model. Inspired by the Forward-Forward algorithm, we concatenate the one-hot encoding of labels to $z$, which leads to a positive embedding $z_P$ if the added label-encoding corresponds to the ground truth of the input $I$, and a negative embedding $z_N$ otherwise. We then train a few feed-forward layers on top of the pre-trained network $F$ using the same procedure as mentioned above. In some settings, doing so outperforms normal training by a large margin, unraveling the great potential of the Forward-Forward algorithm in transfer learning.

In what follows, the most important implementation steps such as preprocessing, feature extraction, and model architecture are further explained.


### <a name='preprocessing'></a> Preprocessing
After downloading the datasets and splitting them into training, testing, and validation, some preprocessing and feature extraction methods are required in order to make the datasets ready for passing into the model. First of all, the raw input signals are resampled by a frequency sampling of 16000, and then their features are extracted. There are two ways by which features can be extracted. The first one is by means of Convolutional layers and the second one is by means of standard feature extraction methods like MFCC, Spectrogram, or FBANKs. The first one needs a deeper neural network and this is not what we wanted to get into. That's why the second method was adopted and FBANKs features were extracted by means of the function in the speechbrain. It's worth bearing in mind that, I was of the opinion that FBANKs can result in better results because their features are more informative and when I compared it to those two methods, FBANKs outperformed them. As can be seen in the following, the number of mels is equal to 40 which is the common value in most of the projects.

    compute_features: !new:speechbrain.lobes.features.Fbank
                      n_mels: 40


Secondly, our dataset must be normalized in order to reduce the computational load of the network and increase its speed. To do so, the normalization method of SpeechBrain called "**InputNormalization**" was used. This method normalizes the mean and standard deviation of the model to zero and one. It should be noted that different normalization methods were tested like normalizing the values between 0 and 1 and normalizing the values between -1 and 1, but the best results were achieved by means of InputNormalization. As can be seen, the norm_type argument of this normalizer has been set to "**global**" which means, it computes a single normalization vector for all the sentences in the dataset.

    mean_var_norm: !new:speechbrain.processing.features.InputNormalization
                   norm_type: global

Finally, they have to be grouped into a different number of batches in order to be fed into the model. Due to the fact that these features have different sizes, it would be problematic if we want to feed them directly into the model. There were two ways by which it would be possible to address this issue. The first one is by means of padding, and the second one is by means of resizing. Both methods were evaluated and Resizing outperformed the padding method by 9%. Moreover, I printed the size of the output of the feature extractor in order to see the range of their sizes with the aim of selecting an appropriate size for the input of the model. The features have the same number of columns (40) but their rows were different and I tried to choose a size that both rows and columns have a close scale.

    wavs = torchvision.transforms.Resize((84, 56))(wavs)

Till now, the format of the size of our dataset is [Batch Number, Row, Column]. While the input of the Convolutional layer must be 4D. That's why a singleton dimension has been inserted at index 1. The final format of the input of the model would be [Batch Number, 1, Row, Column].

    wavs = wavs.unsqueeze(1)

As I mentioned earlier, positive and negative samples must be created and fed into our model. In the original paper the class labels are represented in the first n pixels of each image (n refers to the number of classes). For the positive image, the right pixel would be equal to the maximum pixel of the image while the others would be equal to the minimum pixel of the image. In the case of a negative sample, a random pixel among those n ones would be equal to the maximum pixel of the image while the others would be equal to the minimum pixel of the image. In this way, the positive samples are those with the right labels and negative samples are those with the wrong labels. The first innovation I mentioned in the introduction part is that, In some cases, it would be better to insert the class labels in a square in the right-left corner of the image instead of the first row of it. In some cases, it could result in better results up to 5%.



### <a name='model'></a> Model
As I mentioned earlier, in this project, the implementations are based on CNN architecture, and these models were investigated in terms of training a CNN model from scratch and training by means of transfer learning. In the following image, the overall structure of the CNN model that is supposed to be trained from scratch has presented. The first figure in the following is the model used for the Forward-Forward algorithm and the second figure is the model used for Backpropagation. It should be noted that this is the structure of the model used for the "Audio MNIST" dataset but for the other datasets the architectures would be similar with a little difference like their size of input, activation functions and so on that we will talk about it in the next section. Moreover, as can be seen, the architectures of the Forward-Forward and Backpropagation algorithms have been chosen as similar as possible in order to be able to compare them fairly.

![CNN_FF.png](https://drive.google.com/uc?export=view&id=1qt_8H1zIQ1oSdOrCawg_CaxGqG9m8tir)

![CNN_BP.png](https://drive.google.com/uc?export=view&id=1Y8aIdlqe0QdN7ikGpC5Hq1hfpOJTESXA)


Here you can see my second innovation in which after extracting the features the labels are concatenated with the flattened vector that is the output of the convolution parts. That's why the size of the flattened vector is 2058 instead of 2048. I came up with this idea because the Forward-Forward algorithm is extremely dependent on the positive and negative samples and it could assist the model to remember the labels and be able to distinguish better between the positive and negative samples. I tried concatenating labels in different layers, even the convolutional parts, but this one surpassed the others. The reason why I chose this architecture is that at the beginning of this project, I could not get any promising results which is why I made a decision to try the transfer algorithm. I evaluated various Transfer Learning methods and came up with the decision that ALexNet is better than the others in terms of performance and the size of the model. Afterward, I thought that maybe the architecture of the AlexNet would work for training a CNN model from scratch. That is why I came up with this model and most of the hyperparameters are based on the AlexNet. However, AlexNet has 5 convolutional layers, in my proposed architecture there are just 3 convolutional layers. It should be noted that I even tried models with more and less convolutional layers, but this architecture was the best among them.

In the following, the architectures of the Transformer-based models have been presented (The first one belongs to the Forward-Forward and the second one to the Backpropagation). As I indicated earlier, I evaluated all of the Transfer Learning models and then decided to use AlexNet. there is a difference between training from scratch and the Transfer Learning approach. In the latter one, the extracted features are fed into the pre-trained convolutional layers, and then in the Fully-Connected layers that are supposed to be fine-tuned, positive and negative samples are generated while in the first one, the extracted features are converted into positive and negative at the beginning of the model (input of the convolutional layers). This innovation improved the performance of the model by up to 33%.

![TransferLearning_FF.png](https://drive.google.com/uc?export=view&id=1S6yNFGsf_ULaLyNE2xw2jPP3EEzv-VQw)
![TransferLearning_BP.png](https://drive.google.com/uc?export=view&id=1KPXWlLKdqlUrYuSFhbv3_aavjD53s_BB)


## <a name='experimental_setup'></a> Experimental Setup
In this section, the details about the datasets employed in your experiments are provided. Enumerate the machine learning approaches utilized to address your problem and specify the corresponding hyperparameters used in each approach.

### <a name='dataset'></a> Dataset
Evaluating the performance of our model on different speech datasets was one of the tasks assigned by the professor. To fulfill this task three speech datasets with different specifications (in terms of the number of available data, number of classes, and so forth) were investigated. In the following, we will talk about them:

#### <a name='audio_mnist'></a> AudioMNIST
This is a Large public dataset of Audio MNIST, 30000 audio samples of spoken digits (0-9) of 60 different speakers. The dataset consists of 30000 audio samples of spoken digits (0-9) of 60 folders and 500 files each. There is one directory per speaker holding the audio recordings. It should be noted that writing the JSON file of this number of datasets was pretty time-consuming, which is why some parts of datasets have been selected in order to be used for training, testing, and validation. In the following table, the number of data allocated for training, testing, and validation has been determined. It should be noted that the first 10 speakers have been selected for training, the next 5 speakers have been selected for validation, and the next 5 speakers are chosen for testing.

| Data Type   |  # Number |
|:-----------:|:---------:|
| Training    |  5000 |
| Validation  |  2000 |
| Testing     |  2000 |


#### <a name='ravdess'></a> Human Speech Emotion Classification (RAVDESS)
The second database that has been employed is called the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) database  which is an open-access database. This dataset is composed of 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS dataset comprises 24 proficient performers, split evenly between male and female, who articulate two lexically similar statements in a neutral North American accent. The dataset includes seven different speech emotions, namely calm, happy, sad, angry, fearful, surprise, and disgust. Each emotion is portrayed at two levels of emotional intensity, normal and strong, alongside an additional neutral expression. In the following table, the number of data allocated for training, testing, and validation has been determined. This dataset is one of the most challenging databases because it does not have enough data for training our deep model. That's why the results of the Transfer Learning method have been provided in the results section.


| Data Type   |  # Number |
|:-----------:|:---------:|
| Training    |  1005 |
| Validation  |  218  |
| Testing     |  217  |


#### <a name='google_speech'></a> Google Speech Command Database
The last database that has been employed is called Google Speech Command Dataset which is a public database. The Speech Commands Dataset is composed of 65,000 utterances that are one second long, featuring 30 brief words. These utterances are contributed by thousands of individuals from the public through the AIY website. In the following table, the number of data allocated for training, testing, and validation has been determined. It should be noted that this is also a huge database, but I couldn't use the whole data because creating the JSON file was pretty time-consuming, that is why just 10 classes of this dataset were chosen to be used for training, testing, and validation.

| Data Type   |  # Number |
|:-----------:|:---------:|
| Training    |  5285 |
| Validation  |  1133 |
| Testing     |  1134 |



### <a name='ML_setting'></a> List of Machine Learning techniques and their corresponding hyperparameters
#### <a name='n_layers'></a> Number of layers of the model
As I indicated previously, the adopted Architecture for the CNN models has been drawn from the AlexNet architecture and I just revise it to be more compatible with training from scratch. For example, I just used 3 out of 5 convolutional layers because it was the optimum number of layers.

#### <a name='cnn_layers'></a> Number of input and output channels, kernel size, stride, and padding
Similar to what I told you about the number of layers, these hyperparameters were set according to the AlexNet architecture.

#### <a name='fc_layers'></a> Number of Linear layers and their corresponding number of neurons
As can be seen, here there is no need for the last layer of Fully-Connected layer of the Forwar-Forward algorithm to have number of classes output neurons because it computes the goodness of each layer for each label and predicts the output by applying argmax on it. These number of neurons are the best among the others because I tested different numbers of neurons and the number of linear layers and these values are the best.

#### <a name='activation_func'></a> Activation Function
The activation functions used in this project are ReLU and LeakyReLU. Using ReLU was our preference because it is a simpler and faster activation function but in some cases like the RAVDESS or Transfer Learning model of AudioMNIST, using LEakyReLU led to better performance.

#### <a name='norm'></a> Normalization
In this project, BatchNormalization was used for the convolutional layers but using BatchNormalization for the linear layers was deteriorating the performance of the model. That's why the original normalizer of this approach (the one used for the classification of MNIST via MLP) that divides the input tensor by its norm was used.

#### <a name='th'></a> Threshold
This is one of the most crucial hyperparameters in the implementation of the Forward-Forward algorithm. Setting the right value for this hyperparameter has a dramatic influence on the convergence of the model to the extent that a tiny change in this parameter can significantly improve the performance of the model. This hyperparameter is different from dataset to dataset and architecture to architecture and its values have been acquired by means of several tries and errors.

#### <a name='lr'></a> Learning rate
This hyperparameter locates the second rank in terms of the importance and effect on the convergence of the model. For example, during training on the Google Speech dataset, I couldn't achieve any convergence till I increased the learning of convolutional layers by a factor of 10. Moreover, in most of my implementations the learning rate of linear layers was equal to 0.01 or 0.05, while for the AudioMNISt dataset, my best results were achieved with a learning rate equal to 0.001.

#### <a name='lr_schedular'></a> Learning rate Scheduler
There are numerous ways by which it would be possible to change the values of the learning rate during training which can lead to better performance and results. The Learning rate scheduler that has been used in this project is based on the PLATEAU approach in which the learning rate is reduced whenever a metric (here is validation loss) has stopped improving. In the case of a lack of improvement for some specific number of epochs (patient), the learning rate is changed by a factor. It should be noted that in some of the implementations I achieved better performance without the usage of Learning Rate Scheduler. I am of the opinion that it's been because of the hyperparameter initialization of this scheduler.


## <a name='experimental_results'></a> Experimental Results
In this part, the experimental results of our model is presented and discussed. There are numerous evaluation metrics by which we can assess the performance of our model. In the following, we will enumerate them:

**_Accuracy_:** It is a metric mainly used when the distribution of different classes is uniform.

**_Precision_:** It is another metric representing the performance of our model on the positive classes. To put it another way, as the precision gets larger, more positive class identifications will be correct. Actually, Precision tries to answer this question that what proportion of positive identifications was actually correct.

**_Recall_:** Recall is an important metric when recognizing positive samples is essential. Actually, Precision tries to answer this question what proportion of actual positives was identified correctly?

**_F1-score_:** It is a metric that is used to strike a balance between Recall and Precision.

**_Confusion Matrix_:** In the Confusion Matrix, the model’s number of data assigned to each class can be obtained. TP, FP, TN, and FN can be obtained through this amount of data which shows the strength of our model in terms of classifying our datasets into their corresponding classes.

**_Receiver Operating Characteristic (ROC)_:** The area under the curve (AUC) is a metric for comparing different classifiers regarding the ROC curve. As the ROC curve becomes closer to the step function, the AUC grows and close to 1, and consequently, a better classification is done.

After introducing the employed evaluation metric, we will get through different datasets in order to present the obtained results of each of them individually and compare their results with the BackPropagation algorithm.

#### <a name='AUDIOMNIST'></a> Audio MNIST
In this section, the results of our Forward-Forward (FF) and BackPropagation (BP)algorithm have been provided. Among the following tables, the first and second ones are the results of our Forward-Forward algorithm on the Fine-tuned model (using AlexNet), respectively. Moreover, the third and fourth ones are the comparison between Forward-Forward and backpropagation algorithms on a CNN model (training from scratch) and Fine-tuned model (using AlexNet), respectively.

According to the achieved results, our Forward-Forward algorithm performs very well compared to the BackPropagation and their results are thoroughly comparable. Our obtained results for this dataset are inferior compared to the previous dataset. However, it is rational because the number of employed data for Audio MNIST is more than the RAVDESS and that's why this data set was pretty challenging and I had to train it for more number of epochs.


_Table 5: The result of Forward Forward Algorithm on a Fine-tuned Model_

| Metric               |  value |
|:-----------:         |:------:|
| Train loss           |  0.889 |
| Validation Accuracy  |  66.97 |
| Testing Accuracy     |  64.52 |
| Recall               |  64.04 |
| Precision            |  63.93 |
| F1-Score             |  62.96 |


_Table 6: Comparison between Forward-Forward and BackPropagation algorithm on a Fine-tuned Model_

| FF train loss | FF Valid Accuracy | FF test Accuracy | BP train loss | BP Valid Accuracy | BP test Accuracy|
|:-----------:         |:------:| :------:| :------:| :------:| :------:|
| 0.889 | 66.97 | 64.52 | 0.105 | 66.05 | 67.28|

#### <a name='ravdess'></a> Human Speech Emotion Classification (RAVDESS)
In this section, the results of our Forward-Forward (FF) and BackPropagation (BP)algorithm have been provided. Among the following tables, the first and second ones are the results of our Forward-Forward algorithm on the Fine-tuned model (using AlexNet), respectively. Moreover, the third and fourth ones are the comparison between Forward-Forward and backpropagation algorithms on a CNN model (training from scratch) and Fine-tuned model (using AlexNet), respectively.

According to the achieved results, our Forward-Forward algorithm performs very well compared to the BackPropagation and their results are thoroughly comparable. Our obtained results for this dataset are inferior compared to the previous dataset. However, it is rational because the number of employed data for Audio MNIST is more than the RAVDESS and that's why this data set was pretty challenging and I had to train it for more number of epochs.


_Table 5: The result of Forward Forward Algorithm on a Fine-tuned Model_

| Metric               |  value |
|:-----------:         |:------:|
| Train loss           |  0.889 |
| Validation Accuracy  |  66.97 |
| Testing Accuracy     |  64.52 |
| Recall               |  64.04 |
| Precision            |  63.93 |
| F1-Score             |  62.96 |


_Table 6: Comparison between Forward-Forward and BackPropagation algorithm on a Fine-tuned Model_

| FF train loss | FF Valid Accuracy | FF test Accuracy | BP train loss | BP Valid Accuracy | BP test Accuracy|
|:-----------:         |:------:| :------:| :------:| :------:| :------:|
| 0.889 | 66.97 | 64.52 | 0.105 | 66.05 | 67.28|

#### <a name='google_speech'></a> Google Speech Dataset
In this section, the results of our Forward-Forward (FF) and BackPropagation (BP)algorithm have been provided. Among the following tables, the first and second ones are the results of our Forward-Forward algorithm on a CNN model (training from scratch) and Fine-tuned model (using AlexNet), respectively. Moreover, the third and fourth ones are the comparison between Forward-Forward and backpropagation algorithms on a CNN model (training from scratch) and Fine-tuned model (using AlexNet), respectively.

In this dataset, our Forward-Forward algorithm results are much better than the Backpropagation method compared to the previous datasets. Again, the results of our CNN model are close to the Backpropagation but our Fine-tuned model completely outperforms the Backpropagation. I am of the opinion that the results were because of the selected structure (because I aimed to use a close architecture for all of the datasets) and by choosing another architecture the results of both Backpropagation and Forward-Forward outperform the current one.



_Table 7: The result of Forward Forward Algorithm on a CNN Model_

| Metric               |  value |
|:-----------:         |:------:|
| Train loss           |  2.99  |
| Validation Accuracy  |  42.81 |
| Testing Accuracy     |  45.16 |
| Recall               |  48.43 |
| Precision            |  48.31 |
| F1-Score             |  47.64 |


_Table 8: The result of Forward Forward Algorithm on a Fine-tuned Model_

| Metric               |  value |
|:-----------:         |:------:|
| Train loss           |  0.513 |
| Validation Accuracy  |  83.64 |
| Testing Accuracy     |  80.31 |
| Recall               |  92.79 |
| Precision            |  93.83 |
| F1-Score             |  93.02 |


_Table 9: Comparison between Forward-Forward and BackPropagation algorithm on a CNN Model_

| FF train loss | FF Valid Accuracy | FF test Accuracy | BP train loss | BP Valid Accuracy | BP test Accuracy|
|:-----------:         |:------:| :------:| :------:| :------:| :------:|
| 2.99 | 42.81 | 45.16 | 6.49 | 48.61 | 49.61|

_Table 10: Comparison between Forward-Forward and BackPropagation algorithm on a Fine-tuned Model_

| FF train loss | FF Valid Accuracy | FF test Accuracy | BP train loss | BP Valid Accuracy | BP test Accuracy|
|:-----------:         |:------:| :------:| :------:| :------:| :------:|
| 0.513  |  83.64  | 80.31 | 8.06 | 51.51 | 50.38 |

