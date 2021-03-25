"""Draw digits and prediction histograms to help understand how the model is predicting."""
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def _draw_one_digit(ax, digit_index, digit_images, title=None):
    """Draw the image of a digit, given its index into an array of images."""
    ax.axis('off')
    ax.imshow(digit_images[digit_index], cmap=plt.get_cmap('Greys'))
    ax.set_title(title)


def draw_digits(digit_indices, test_label, predicted_label, test_image):
    """Draw digits selected from the test dataset, given their indices.

    Inspired by https://medium.com/@mjbhobe/mnist-digits-classification-with-keras-ed6c2374bd0e
    """
    number_of_digits = len(digit_indices)
    _, ax = plt.subplots(1, number_of_digits, figsize=(8, 8))
    for c in range(number_of_digits):
        digit_index = digit_indices[c]
        title = "{}: {}/{}".format(digit_index, test_label[digit_index],
                                   predicted_label[digit_index])
        _draw_one_digit(ax[c], digit_index, test_image, title)
    plt.tight_layout()
    plt.show()


def _draw_prediction_histogram(ax, class_probabilities, hide_y_labels, num_classes=10):
    """Draw a histogram of the predicted probabilities for each class and format for easier reading."""
    classes = np.arange(num_classes)

    # Show all graphs in the same scale, to allow comparison
    ax.set_ylim(0, 100)

    # Draw the bars and show the class (digit) above each one
    # Show in % (nicer labels for the y axis)
    bars = ax.bar(classes, class_probabilities*100)
    for digit, bar in zip(classes, bars):
        yval = bar.get_height()
        ax.text(bar.get_x() - 0.25, yval + 3, digit)

    # Remove all tick marks, the bottom labels (already show class above the bar) and suppress
    # the y label for zero on the left corner to avoid confusion with the class - also zero
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([25, 50, 75, 100])

    # Hide the y axis label if so requested (to make it look cleaner, show y labels only for
    # the first graph [left] in the row)
    if hide_y_labels:
        ax.yaxis.set_major_formatter(NullFormatter())

    # Show faint grid lines behind the bars
    ax.yaxis.grid(color='grey', alpha=0.25)
    ax.set_axisbelow(True)

    # Leave only the bottom spine visible, so the bars aren't "floating in space"
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


def draw_digits_histogram(digit_indices, test_label, predicted_label, test_image,
                          predicted_probability):
    """Draw digits and histogram of probaility predictions.

    Assumes that arrays with predicted classes and predicted probabilities have already
    been populated elsewhere .
    """
    number_of_digits = len(digit_indices)
    num_cols = 10
    num_rows = math.ceil(number_of_digits / num_cols)

    # Subplots with twice as many rows as we calculated above because we will show a histogram
    # of the predictions below each digit. The histogram is twice as tall as the digit (the
    # gridspec_kw values).
    _, ax = plt.subplots(num_rows*2, num_cols, figsize=(12, 3*num_rows),
                         gridspec_kw={'height_ratios': np.tile([1, 2], num_rows)})

    for r in range(num_rows):
        for c in range(num_cols):
            image_index = r * num_cols + c
            if image_index < number_of_digits:
                digit_index = digit_indices[image_index]
                title = "{}: {}/{}".format(digit_index, test_label[digit_index],
                                           predicted_label[digit_index])
                _draw_one_digit(ax[r*2, c], digit_index, test_image, title)
                _draw_prediction_histogram(
                    ax[r*2+1, c], predicted_probability[digit_index], c > 0)
            else:
                # Turn off the spines to show an empty space in incomplete rows
                ax[r*2, c].axis('off')
                ax[r*2+1, c].axis('off')

    plt.tight_layout()
    plt.show()
