import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, epoch, save_path):
    plt.figure(figsize=(60, 60))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Fraction'})

    plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}/epoch_{epoch + 1}_confusion_matrix.png")
    plt.close()