import pandas as pd
import matplotlib.pyplot as plt
import os 
from parsers import MetricPlotterConfigParser


def visualize_recall_per_class(csv_file_path, output_img_path, classes):
    df = pd.read_csv(csv_file_path)
  
    plt.figure(figsize=(12, 8))

    for class_name in classes:
        class_data = df[df['Class Name'] == class_name]
        plt.plot(class_data['Epoch'], class_data['Recall'], label=class_name)

    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall per Class over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_img_path)
    plt.close()


def main():
    parser = MetricPlotterConfigParser()
    csv_file_path = parser.get_csv_file_path()
    output_img_folder_path = parser.get_output_img_folder_path()

    if not os.path.exists(output_img_folder_path):
        os.makedirs(output_img_folder_path)

    metric = parser.get_metric()
    classes = parser.get_classes()
    category = parser.get_category()
    if category is None:
        output_img_path = os.path.join(output_img_folder_path, f"{metric}_{'_'.join(classes)}.png")
    else:
        output_img_path = os.path.join(output_img_folder_path, f"{metric}_{category}.png")
    visualize_recall_per_class(csv_file_path, output_img_path, classes)

if __name__ == "__main__":
    main()
    