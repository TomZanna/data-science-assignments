# Intro

## 1a: Upload the Jupyter notebook (.ipynb) here!

## 1b: Upload the PDF version of the notebook (.pdf) with results here!

## 1c: In case you used an LLM for coding, provide the model(s) and the prompts used.

# Task 1

## 2a: What properties of the graph-classification task on the DrowsyFaceGraphDataset can be leveraged for neural-network design?

## 2b: Describe your neural-network architecture components/layers 
> Your description should explain:
> - how node representations are updated
> - how graph-level representations are obtained from node representations
> - how the final prediction is produced

## 2c: Upload the training and test loss curves

## 2d: Name two methods that could potentially prevent overfitting on DrowsyFaceGraphDataset

## 2e: Upload your test-set evaluation metrics.
> The metric names and the required baseline threshold of 0.6 must be clearly shown.


## 2f: In this driver-monitoring setting:
- which type of prediction error is more dangerous (FP or FN)?
- which evaluation metric should therefore be prioritized when choosing a model for deployment?

# Task 2
## 3a: What properties of the image data does the PatchShuffle corruption process alter? Why can this cause standard convolutional neural networks to fail?

## 3b: Describe your neural-network architecture layer-by-layer and explain how each component addresses the challenges introduced by the PatchShuffle corruption process.
> Your description should explain:
> - how image patches are converted into input representations
> - how information is exchanged between patches
> - why the intermediate layers are permutation equivariant with respect to shuffled patch locations
> - how the final object prediction is produced

## 3c: Upload the figure with the per-class metrics

## 3d: Give one reason why evaluation metrics should also be computed separately for each class in a multi-class classification problem.

## 3e: Report the aggregate metrics for the clean/corrupted test set. Your table must be 2x2 where each row corresponds to a metric and each column to a dataset.

# Task 3

## 4a: Describe the pretrained model you selected for this task
> Your description should include:
> - the exact name of the pretrained model and the meaning of its naming convention
> - the dataset on which the model was pretrained
> - the preprocessing operations (transformations) required to match the input format expected by the pretrained model

## 4b: Describe the conceptual differences between linear probing and k-nearest neighbor (kNN) classification as evaluation protocols for pretrained embeddings.
> In your answer, discuss:
> - how the two methods differ in terms of learnable parameters
> - how they differ in the type of structure they evaluate in the representation/embedding space

## 4c: Provide the code used to freeze the pretrained encoder weights before training the linear classifier

## 4d: Upload your metrics comparing the model from scratch, the pretrained model with linear probing and the pretrained model with kNN classification.

## 4e: Looking at the performance of the model trained from scratch compared to the pretrained frozen encoder, what can you conclude about:
- the usefulness of pretrained visual representations for downstream object classification tasks
- the amount of training data typically required to train strong visual representations from scratch
- the necessity of end-to-end fine-tuning when strong pretrained embeddings are already available?

