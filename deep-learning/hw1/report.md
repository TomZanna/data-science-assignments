# Task 1
## AI disclosure
The followign prompts were given to Gemini 3.1 Pro:
- regarding task 1: write a PyTorch Lightning Module to train the following AppleEmbeddingNet. Use TripletMarginLoss from pytorch_metric_learning as loss function and TripletMarginMiner from pytorch_metric_learning with parameter type_of_triplets="all" to compute loss only on hard and semi-hard samples. At the end of each epoch log the training and validation loss to an array in self.history
- regarding task 2: write snippet to display confusion metrix given seaborn.cm and classes
- regarding task 2: write Python snippet to plot the family confusion matrix of a dataset with item idx. Use class_to_idx and family_to_idx to map item to corresponding family.
- regarding task 4: write a PyTorch Lightning Module to train the following ConvolutionalAutoencoder. Use MSELoss as loss function and at the end of each epoch log the training and validation loss to self.history


## Model architecture and loss function
#### Describe the neural network architecture that fits the task. Describe the different components of your architecture and explain how the relevant characteristics and symmetries of the data are accommodated for by your design choices.
We implemented a convolutional embedding network. Unlike a standard image classifier, the model does not output a probability distribution over the apple items, but learns an embedding function $f_\theta: \mathcal{X}\to \mathbb{R}^D$ which maps each input image $x$ to a dense vector representation $ z = f_\theta(x)$.

The model therefore learns how to represent each image in the embedding space. The distance function used in that space, such as Euclidean distance or cosine similarity, is then applied to the learned embeddings. This gives a similarity measure between the original images that is induced by the learned representation. Thus, item-level similarity can be reflected geometrically in the embedding space: images of the same apple item should be mapped to nearby vectors, while images of different apple items should be mapped farther apart.

The architecture has three main components: a convolutional feature extractor, a linear projection head, and a final L2 normalization.

The feature extractor consists of three convolutional blocks. Each block applies a convolution, batch normalization, ReLU nonlinearity, and max-pooling.

The convolutional feature extractor is appropriate because the inputs are images, which have local spatial structure. Nearby pixels are related, and relevant visual information appears in local patterns such as edges, color transitions, textures, and shape details, which can be learned by convolutional filters. Stacking several convolutional layers then allows the network to combine simple features into more abstract visual features that are more specific to the identity of the apple item.

Batch normalization is used to stabilize intermediate activations and make optimization more reliable, while ReLU introduces nonlinearity, allowing the model to represent more complex mappings than a pure linear model. As the network becomes deeper, max pooling reduces the spatial resolution of the feature maps, while the number of channels is increased. This gradually reduces dependence on exact pixel positions and shifts the representation from detailed spatial information to a more compact and abstract description of the image.

Convolutions also use parameter sharing: the same filter is applied at every spatial position. This makes the model more parameter-efficient than a fully connected network and helps generalization, which is especially useful because the training set contains a limited number of apple items. It also reflects the structure of image data: the same local visual feature can be meaningful even if it appears in a slightly different position in the image.

The architecture also accommodates the translation structure of images. Images are signals on a two-dimensional grid, and small shifts of an apple item in the image should not change its identity. Convolutional layers alone are translation equivariant: if a pattern shifts in the input image, the corresponding activation shifts in the feature map.
Then, max-pooling gives local robustness to small shifts, while adaptive average pooling aggregates each feature map into a single value, making the final representation less sensitive to exact pixel locations. As a result, a feature vector is produced, which is approximately invariant, or at least robust, to small translations and changes in the apple’s position in the image.

A linear projection head then maps this visual feature vector into the embedding space optimized by the metric learning loss.

Finally, the embedding vector is L2-normalized, so all embeddings lie on the unit sphere. This fixes the scale of the embeddings and makes comparisons depend mainly on their direction. For normalized vectors, minimizing Euclidean distance is equivalent to maximizing cosine similarity, so the embeddings can be compared consistently using either metric.

The dataset also contains multiple frames of the same apple item captured from different camera angles, corresponding to changes in viewpoint or 3D rotation of the object. The item identity should remain the same under these viewpoint changes. However, a standard CNN does not encode full rotation or viewpoint invariance by construction, so this robustness is mainly learned from the data and the metric-learning objective. Since different frames of the same apple item are treated as positive examples during training, the model is encouraged to map them to nearby embeddings.

#### What loss function is appropriate for the requirements of task 1? Give and explain the formula associated to that function

A metric-learning loss is an appropriate loss function for this task because the goal of the model is to learn an embedding space where images of the same apple item are close together and images of different apple items are farther apart. In our implementation, we used triplet margin loss.

Triplet loss is based on triplets of images: an anchor image $x_a$, a positive image $x_p$, which shows the same apple item as the anchor, and a negative image $x_n$, which shows a different apple item. The same embedding network $f_\theta$ is applied to all three images, giving embeddings $f_\theta(x_a)$, $f_\theta(x_p)$, $f_\theta(x_n)$. The objective is then to make the distance between the anchor and the positive image smaller than the distance between the anchor and the negative in the embedding space: 
$$\|f_\theta(x_a) -f_\theta(x_p)\|_2 < \|f_\theta(x_a) -f_\theta(x_n)\|_2$$
To make this separation more robust, a margin $m>0$ is added that specifies how much farther the negative should be from the anchor compared with the anchor-positive distance. The formula for the triplet loss we used is: $$\mathcal{L}(x_a, x_p, x_n) = \max \left\{d(f_\theta(x_a) , f_\theta(x_p))- d(f_\theta(x_a), f_\theta(x_n)) + m, 0\right\},$$ where $d(u,v) = \|u-v\|_2$ is the Euclidean distance between normalized embedding vectors.

The loss is positive when the negative example is closer to the anchor than the positive embedding, or when the margin requirement is not satisfied; i.e., the negative example is not sufficiently separated from the positive example. In such cases, minimizing the loss encourages the model to pull the anchor and positive embeddings closer together and/or push the negative embedding farther away. If the negative is already farther away by at least the margin, the triplet already satisfies the desired ordering, and the loss is zero.

This loss matches the requirements of the task, especially because it directly addresses the relative nature of similarity in the embedding space. Compared with contrastive loss, which is another valid metric-learning approach, the model is not only trained to set similar pairs closer together or push dissimilar pairs apart, but also learns to satisfy a relative ordering of distances between different types of data samples.

In practice, not all triplets are equally useful. Easy triplets already satisfy the margin and do not contribute much to learning, as they have zero loss.

For this reason, we use hard and semi-hard triplets to train the model: the former are cases where the negative is closer to the anchor than the positive, while in the latter, the negative is farther from the anchor than the positive, but still not far enough to satisfy the margin. These triplets produce a positive loss and therefore provide a useful training signal.


## Implementation
#### Implement and describe the model architecture.
The model was implemented in PyTorch by defining an AppleEmbeddingNet class inheriting from torch.nn.Module. Overall, it is a compact model with about $26$K trainable parameters.

Before feeding them to the network, the input images are resized to $96\times96$ pixels and converted to tensors, giving RGB inputs of shape 3×96×96. The model then outputs embeddings in $32$-dimensional space.

The network consists of a convolutional feature extractor, an adaptive average pooling layer, a linear projection layer, and a final L2 normalization.

The convolutional feature extractor consists of three blocks. The convolutions use $3\times3$ kernels with padding 1, so they preserve the spatial resolution inside each block while increasing the number of channels. After applying Batch normalization and adding a ReLU nonlinearity to the activations, the spatial resolution of the activations is then halved by a $2\times 2$ max-pooling layer.
So after each block the number of channels increases from 3 to 16, then 32, and finally 64, while the spatial resolution goes from $96\times96$ to $48\times 48$, then $24\times24$, and finally $12\times 12$.

After the convolutional blocks, AdaptiveAvgPool2d(1,1) aggregates each of the 64 final feature maps into one value, returning a tensor of shape $64\times 1\times 1$. The tensor is then flattened into a $64$-dimensional feature vector. The linear layer then maps this vector to the final $32$-dimensional embedding space. 

Finally, we L2-normalize the output embedding using the PyTorch function. This places all embeddings on the unit sphere.

#### Implement and describe the training procedure.
We trained the model from scratch %using the triplet margin loss defined -above-.
by implementing a PyTorch Lightning module called 'TripletEmbeddingLit' to structure the training loop. The model definition, training step, validation step, optimizer, and logging are defined explicitly. The learning objective is still the triplet margin loss described above, which is implemented by the TripletMarginLoss class from pytorch-metric-learning.

In each forward pass of the training:

- Compute embeddings for the batch: Batches of images and labels are passed through the network, obtaining normalized embeddings. These are built using MPerClassSampler from pytorch-metric-learning. We set: m = 6, batch_size = 60, length_before_new_iter = len(train_data); each batch contains 60 images, with 6 images for each selected apple item. This is important for triplet loss because a batch must contain multiple positive and negative examples to form triplets. 

- Embeddings and labels are passed to TripletMarginMiner(margin=0.2, type_of_triplets="all"), which selects hard and semi-hard triplets from the batch.
Given the embeddings and labels in a batch, the miner computes pairwise distances and identifies the anchor-positive-negative combinations. It then filters them to keep non-easy triplets.
- Finally, given embeddings, labels, and selected triplet indices, the (average) triplet margin loss is computed. Lightning uses this loss to compute gradients and update the model during the backpropagation step.

We trained for 10 epochs using the standard adaptive AdamW optimizer, which has a default learning rate of 3e-4, and weight decay of 1e-2.

During validation, embeddings are computed without contributing to the weight update, but only used to monitor the loss, and the M-per-class batch structure is not needed. The validation loader uses batch_size=128 and no shuffling.

## Classification
#### Given an image of one of the apples captured at a camera angle not seen during training, how can you use your model and the training set to predict which item the image corresponds to?
Since the model is trained with a metric-learning objective, it is not trained to directly predict fixed class probabilities. Instead, it learns an embedding function that should map different frames of the same apple item close to each other in the embedding space. Therefore, an unseen camera angle of a known apple item is expected to have an embedding close to the embeddings of training images from that same item.

To classify a new test image $x_q$, we first extract the embeddings of the training set images using the trained model. Then, the query image is encoded, and prediction takes place via a nearest-neighbor classifier: the test sample inherits the label of the closest neighbor in the embedding space. Because the embeddings are L2-normalized, this closest match can be found by either minimizing Euclidean distance or maximizing cosine similarity.

#### Evaluate and report the accuracy of your solution on the provided test set (test_data).
Using the training set as the labeled support set and test data as the query set, the nearest-neighbour classifier achieved an accuracy of $0.9480$.
This means that $94.8\%$ of the test images from the seen apple items were assigned to the correct item.

#### Given a support set, how can you classify items not seen during training?
Given a labeled support set, the classification method remains the same, without the need to retrain the network.

As explained before, we first compute and store the embeddings and labels of all images in the support set. Then, for each query image, we compute its embedding too and assign it the label of the closest support embedding using nearest neighbor classification. 
In this case, the support set may contain apple items that were not used to train the model. Since the model was trained to organize apple images according to visual similarity, different frames of a new apple are still expected to be embedded close to each other, making the nearest neighbor approach still effective. 

This can be safely assumed when the new support items come from a similar domain; if the support set were visually very different from the training data, the learned embedding space might not generalize as well.

#### Evaluate and report the accuracy of your solution on the provided test set (test_new_data).
Using \texttt{support_new_data} as the labelled support set and \texttt{test_new_data} as the query set, the nearest-neighbour classifier achieved an accuracy of $0.9040$, showing that the model can still achieve good generalization even when tested on unseen items.

# Task 2
#### Did you update your model or training procedure for task 2? If yes give all necessary information for someone to adapt the implementation of task 1 for task 2, if not you explain why. 
No, we maintained the same configuration of the previous task. This is appropriate because the core objective remains the same: mapping similar items close together in the metric space, while pushing dissimilar items far apart.

#### Evaluate and report the accuracies of your model in the 4 different scenarios
The embedding model, trained using the triplet objective, was evaluated in the four specified scenarios. The model achieved 100% accuracy across all testing conditions. By analyzing the datasets, we determined that this performance is likely due to the minor variations between the test samples and those used during training. Thus, the model extracts nearly identical representations, and the nearest-neighbor algorithm easily identifies the correct identity in the embedding space during inference.

## Visualization
#### [to-load] In scenario 1, compute the item classification accuracy for the different families and upload a visualization (e.g. bar plot) of your results.
#### Which families have the lowest accuracy?
Since the overall accuracy is 100%, the confusion matrix of the test results grouped by family shows that no family is identified incorrectly more than any other.

#### [to-load] In scenario 2, compute the confusion matrix and upload an image of the matrix (you can use tools like ConfusionMatrixDisplay from sklearn for a nice visualization).
#### Which families are most likely to be confused by your model?
As in the previous scenario, there is no family (or group of families) that is more likely to be confused.

# Task 3
#### What loss should be added to the loss function of task 2 to satisfy the requirements of task 3? Provide the complete loss function (including the one of task2) and explain the formula.
Since task 3 asks to improve performance on family classification, the previous loss function can be extended by adding a new term, computed based on how the model performs at labelling families. Thus, the joint objective is computed as the (weighted) sum of the item-level triplet loss and the family-level triplet loss (both computed as triplet margin loss): `total_loss = α * family_level_loss + (1 - α) * item_level_loss`, where `α` is a hyperparameter balancing the trade-off between coarse-grained (family) and fine-grained (item) feature representations.

#### Implement and describe your training procedure. Only explain the differences with the training procedure in the previous task
The training procedure remains identical to the previous task, except for the batch selection and loss computation, which now account for two distinct supervision levels. We sample the first triplet based on the item label, matching the exact difficulty and selection criteria from the previous task. A second triplet is sampled based on the family label; this allows the network to pair an anchor with a positive frame from a completely different item, provided they share the same overarching family. During the forward pass, embeddings are generated for both sets of triplets.

#### Evaluate and report the accuracies of your model in the 4 different scenarios
Since the accuracy was already 100%, it couldn’t be improved. However, the new loss function has maintained the previous score, resulting in no errors across the 4 scenarios.

# Task 4
#### Describe a suitable anomaly detection method using the model obtained in task 3 and the training set
We use the embedding model from task 3 as a nearest-neighbor anomaly detector in feature space. First, we embed the clean training set and keep those embeddings as the support set. For a query image, we compute its embedding and measure its anomaly score as $1 - \max_j \cos(z, z_j)$, where $z$ is the query embedding and $z_j$ are the support embeddings. The threshold is chosen on a clean calibration set so that about 10% of normal samples are flagged as abnormal; samples above the threshold are rejected as anomalies.

#### Given that we tolerate 10% of the test sample to be predicted as abnormal, evaluate and report the percentage of anomalies detected by your method with 1%, 5% and 10% of black pixels.
Using the clean calibration set, the embedding detector threshold was set to 0.0005 and the false-positive rate on normal samples was 0.1001. The detection rates for corrupted test images were 1.0000 for 1% black pixels, 1.0000 for 5% black pixels, and 1.0000 for 10% black pixels.

#### What would happen if the anomaly you are trying to detect resembles a data augmentation used during training of the embedding mode
If the anomaly resembles a transformation the embedding model has already seen during training, the detector becomes less sensitive to it. The perturbed sample may still land inside the normal embedding cluster, so its distance to the support set remains small and it may fall below the threshold. In that case, false negatives increase and the anomaly detection rate drops.

#### Explain what an autoencoder is, how it is trained and how it could be used to detect anomalies.
An autoencoder is made of an encoder and a decoder. The encoder compresses an input image into a latent representation, and the decoder reconstructs the original image from that code. When trained only on normal data with a reconstruction loss, the model learns the normal data manifold; anomalous inputs usually produce a larger reconstruction error and can be flagged using a threshold on that error.

#### Implement your autoencoder and give all necessary instruction for someone to implement your architecture. As in task 1, think of an architecture that suits the data.
We use a fully convolutional autoencoder for 96x96 RGB images because the anomalies we inject are local pixel corruptions and the spatial layout matters. The encoder applies Conv2d(3, 16, kernel_size=3, padding=1) -> BatchNorm2d(16) -> ReLU -> MaxPool2d(2), then Conv2d(16, 32, kernel_size=3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2). This reduces the spatial size from 96x96 to 24x24 while preserving local structure. BatchNorm is useful here because it stabilizes optimization and usually yields sharper reconstructions instead of blurry averages. The decoder mirrors the encoder with ConvTranspose2d(32, 16, kernel_size=2, stride=2) -> BatchNorm2d(16) -> ReLU, followed by ConvTranspose2d(16, 3, kernel_size=2, stride=2) -> Sigmoid. The sigmoid output matches the [0, 1] range produced by ToTensor.

#### Implement a training method and provide all necessary information for someone to reproduce the training of the model.
The autoencoder is trained only on clean training images from the GardenDataset, resized to 96x96 and converted to tensors. This matches the standard anomaly-detection assumption that the training set contains only normal samples. We minimize mean squared error between the input and the reconstruction, use AdamW optimization, batch size 64 for training and 128 for validation, and train for 5 epochs in the notebook. At inference, the anomaly score is the per-image reconstruction MSE, and the threshold is set on clean validation scores so that about 10% of normal samples are flagged.

#### Given that we tolerate 10% of the test sample to be predicted as abnormal, evaluate and report the percentage of anomalies detected by your method with 1%, 5% and 10% of black pixels.
With the reconstruction-error threshold chosen on clean data, the autoencoder achieved a false-positive rate of 0.1001 on normal samples. The detection rates were 0.9997 for 1% black pixels, 1.0000 for 5% black pixels, and 1.0000 for 10% black pixels, corresponding to 99.97%, 100.00%, and 100.00% detection.
