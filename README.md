# What is "model accuracy", really?

In the book [Responsible Machine Learning](https://www.h2o.ai/resources/ebook/responsible-machine-learning/),
when discussing trust and risk, the authors recommend a list of questions to ask to understand the
risk of a machine learning (ML) deployment.

One of the questions is **"What is the quality of the model? (Accuracy, AUC/ROC, F1)"**. These
metrics compare correct and incorrect predictions of a model.

But how exactly a model determines what a correct prediction is?

Here we will analyze the effect of an important factor a model uses to decide what the correct
prediction (label) is for classification problems, the **decision threshold**. We will see that
without understanding how a model decides what "correct" is, talking about the model accuracy
is premature.

We use _accuracy_ in this text as the number of correct predictions on the test set, divided by the
number of instances in the test set.

```text
             Number of correct predictions
accuracy = ----------------------------------
                  Number of instances
```

To illustrate the discussion, we will use an image classification model (the code is in this
[Jupyter notebook](./softmax-thresholds.ipynb)).

(_Simplification disclaimer: there are other types of problems, e.g. regression, and other types of
models -- we are making simplifications to expose the main concept._)

A typical image classification problem, taught early in machine learning, is digit classification with the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/). The dataset looks like this (a small sample -
the dataset has 70,000 images):

&nbsp;&nbsp;&nbsp;&nbsp;![MNIST](./pics/mnist.png)

In an image classification problem, we train a model to identify the class (label) of an image.
In this case, there are ten classes, one for each digit (from zero to nine).

&nbsp;&nbsp;&nbsp;&nbsp;![Digit classification model](./pics/digit-classification-model.png)

This is an actual digit from MNIST. The model correctly classifies it as the digit "2".

&nbsp;&nbsp;&nbsp;&nbsp;![Classification example](./pics/digit-classification-example.png)

A neural network has several hidden layers to extract ("learn") features from the images. The very
last layer is the one that classifies the image. In this case, we are classifying ten classes (ten
digits), therefore the last layer has ten neurons, one for each digit.

&nbsp;&nbsp;&nbsp;&nbsp;![Classification layer](./pics/classification-layer.png)

Because we want to know what digit it is, we use [softmax activation](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax)
in the last layer to give us a probability distribution of each class. In the case below, the model
is certain that the image is a number "2".

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification certain](./pics/model-classification-certain.png)

For other images, the model may not be so certain.

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification not certain](./pics/model-classification-not-certain.png)

In those cases, how should we decide what the label is?

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - how to decide](./pics/model-classification-how-to-decide.png)

Most of the time, the class with the largest probability is used as the label. In this example, the
model classifies the image as the digit "2".

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - largest probability](./pics/model-classification-use-largest.png)

But what should the model do when the largest probability is not that high and is close to the
probability of other classes?

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - uncertain](./pics/model-classification-uncertain.png)

In the example below, the largest probability is for the class "9", but it is not even 50% and the
probability for class "4" is not too far behind. The model does not have high confidence in this
prediction.

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - uncertain](./pics/model-classification-uncertain2.png)

What should we do in these cases?

To solve those cases we usually pick a threshold for the decision. Instead of simply using the class
with the maximum probability, we pick the largest probability above the threshold we chose. If we
choose 50% as the threshold, in the number "2" example above we are still able to classify the image
as the number "2".

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - above threshold](./pics/model-classification-threshold-above.png)

But now we no longer classify the ambiguous image as a number "9". In this case, we would not make
a decision at all.

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - below threshold](./pics/model-classification-threshold-below.png)

But what threshold do we pick?

It depends. For high-stakes applications, where wrong decisions have severe consequence, we want to
be very confident in the model's prediction.

For example, for an automatic check deposit application, we want the model to be at least 99%
certain of the prediction. Any image below that threshold is sent to human review.

&nbsp;&nbsp;&nbsp;&nbsp;![Model classification - high stakes](./pics/model-classification-high-stakes.png)

## Effect of different thresholds

The higher the threshold for the decision, the fewer images the model can classify. For the model
used in these examples, this is the effect of different thresholds on the model's accuracy.

| Threshold | Accuracy |
| --------: | -------: |
|    99.99% |    45.3% |
|     99.9% |    72.4% |
|       99% |    89.8% |
|       90% |    96.6% |
|       80% |    97.7% |
|       75% |    97.9% |
|       67% |    98.3% |
|       50% |    98.7% |

## Asking questions about "accuracy"

The one-line takeaway: _to use a model responsibly **we must ask questions** about how its accuracy
was measured and not just accept published numbers_.

1. How predictions are being made: is it probability-based (as in the examples above)? Something
   else?
1. What factors control the predictions: is it threshold-based or some other decision (e.g. argmax)?
   If it is threshold-based, what are the thresholds?

_"We achieved 99.9% accuracy on [some task here]"_ means nothing if it's not accompanied by a
detailed description of what a "correct prediction" is for the model.
