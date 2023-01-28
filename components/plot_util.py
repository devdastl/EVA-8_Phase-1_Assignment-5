# Utility function to plot graphs and misclassified image.

import matplotlib.pyplot as plt
import os

#function to plot loss & accuracy graph.
def plot_loss_accuracy(test1, test2, test3):

    # Model 1
    loss1, accuracy1 = test1.test_losses, test1.test_acc
    # Model 2
    loss2, accuracy2 = test2.test_losses, test2.test_acc
    # Model 3
    loss3, accuracy3 = test3.test_losses, test3.test_acc

    # Plot loss and accuracy as subplots
    fig, ax = plt.subplots(2,1, figsize=(8,8))
    ax[0].plot(loss1, label='Model 1')
    ax[0].plot(loss2, label='Model 2')
    ax[0].plot(loss3, label='Model 3')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(accuracy1, label='Model 1')
    ax[1].plot(accuracy2, label='Model 2')
    ax[1].plot(accuracy3, label='Model 3')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.show()


#function to plot misclassified images.
def plot_misclassified(test_misc_img, test_misc_label, subtitle='misclassified images'):
# Set the number of rows and columns for the plot
  num_rows = 5
  num_cols = 2

  # Create a figure and axes for the plot
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

  # Iterate over the mis-classified images and labels
  for i, (img, (pred_label, true_label)) in enumerate(zip(test_misc_img, test_misc_label)):
      # Get the row and column index for the current image
      row = i // num_cols
      col = i % num_cols

      # Plot the image and label on the current axes
      axes[row, col].imshow(img.to('cpu').squeeze(), cmap='gray')
      axes[row, col].set_title(f'Pred: {pred_label}, True: {true_label}')
      # remove axis labels
      axes[row, col].axis('off')
      
  plt.suptitle(subtitle)
  if not os.path.exists("report"):
      os.makedirs("report")
  plt.savefig("report/" + subtitle + ".png")
  plt.show()
