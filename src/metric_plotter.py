import pandas as pd
import matplotlib.pyplot as plt
import os 
from parsers import MetricPlotterConfigParser


def visualize_metric_per_class(csv_file_path, output_img_path, classes, metric="Recall"):
    df = pd.read_csv(csv_file_path)
  
    plt.figure(figsize=(12, 8))

    for class_name in classes:
        class_data = df[df['Class Name'] == class_name]
        plt.plot(class_data['Epoch'], class_data[metric], label=class_name)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f"{metric} per Class over Epochs")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_img_path)
    plt.close()


def plot_confusion_trends(csv_file_path, true_class_name, top_n=3, output_img_path=None, class_names=None, epoch_until=None):
    stats_df = pd.read_csv(csv_file_path)
    class_df = stats_df[stats_df["Class Name"] == true_class_name]
    #epochs = class_df["Epoch"].unique()
    epochs = class_df[class_df["Epoch"] <= epoch_until]["Epoch"].unique() \
        if epoch_until is not None \
        else class_df["Epoch"].unique()

    top_confused_classes = class_df.iloc[:, 6:].mean().sort_values(ascending=False).head(top_n).index
    top_confused_classes = [int(col.split()[-1]) for col in top_confused_classes]

    plt.figure(figsize=(10, 6))
    for confused_class in top_confused_classes:
        confusion_trend = class_df[class_df["Epoch"] <= epoch_until][f"Misclassification {confused_class}"] \
            if epoch_until is not None \
            else  class_df[f"Misclassification {confused_class}"]
        plt.plot(epochs, confusion_trend, label=f'{class_names[confused_class-1]}')

    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Rate')
    plt.title(f'Misclassification Trends for True Class {true_class_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.savefig(output_img_path)
    plt.close()


def main():
    parser = MetricPlotterConfigParser()
    csv_file_path = parser.get_csv_file_path()
    output_img_folder_path = parser.get_output_img_folder_path()

    if not os.path.exists(output_img_folder_path):
        os.makedirs(output_img_folder_path)

    metric = parser.get_metric()

    if metric in ["Recall", "Precision", "F1_Score"]:
        classes = parser.get_classes()
        category = parser.get_category()
        if category is None:
            output_img_path = os.path.join(output_img_folder_path, f"{metric}_{'_'.join(classes)}.png")
        else:
            output_img_path = os.path.join(output_img_folder_path, f"{metric}_{category}.png")
        visualize_metric_per_class(csv_file_path, output_img_path, classes, metric)
    if metric == "confusion_trends":
        true_class = parser.get_true_class()
        top_n = parser.get_top_N()
        output_img_path = os.path.join(output_img_folder_path, f"{metric}_{true_class}.png")
        plot_confusion_trends(csv_file_path,
                            true_class,
                            top_n, 
                            output_img_path=output_img_path,
                            class_names=parser.get_class_names(),
                            epoch_until=parser.get_epoch_until())

if __name__ == "__main__":
    main()
    