import numpy as np
import sklearn.metrics as metrics
from parsers import TrainConfigParser


def one_iter(model, criterion, loader, device, train=True, optimizer=None, scheduler=None, monitoring_metrics=list()):
    parser = TrainConfigParser()
    model_name = parser.get_model_name()
    running_loss = 0.0
    count = 0.0
    if train:
        model.train()
    else:
        model.eval()

    preds_arr = []
    targets_arr = []
    for data, labels in (loader):
        batch_size = len(labels)
        data = data.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        outputs = model(data)
        if (model_name == "vit"):
            outputs = outputs.logits
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        preds = outputs.max(dim=1)[1]
        count += batch_size
        running_loss += loss.item() * batch_size

        targets_arr.append(labels.cpu().numpy())
        preds_arr.append(preds.detach().cpu().numpy())

        if train and scheduler is not None:
            scheduler.step()

    targets = np.concatenate(targets_arr)
    preds = np.concatenate(preds_arr)

    loss = running_loss*1.0/count

    result = {"loss": loss}
    if "accuracy" in monitoring_metrics:
        accuracy = metrics.accuracy_score(targets, preds)
        result["accuracy"] = accuracy
    if "balanced_accuracy" in monitoring_metrics:
        balanced_accuracy = metrics.balanced_accuracy_score(targets, preds)
        result["balanced_accuracy"] = balanced_accuracy
    if "macro_avg_precision" in monitoring_metrics:
        macro_avg_precision = metrics.precision_score(targets, preds, average="macro")
        result["macro_avg_precision"] = macro_avg_precision
    if "macro_avg_f1_score" in monitoring_metrics:
        macro_avg_f1_score = metrics.f1_score(targets, preds, average="macro")
        result["macro_avg_f1_score"] = macro_avg_f1_score
    if "confusion_matrix" in monitoring_metrics:
        cm = metrics.confusion_matrix(targets, preds)
        result["confusion_matrix"] = cm
    if "recall_per_class" in monitoring_metrics:
        recall_per_class = metrics.recall_score(targets, preds, average=None)
        result["recall_per_class"] = recall_per_class
    if "precision_per_class" in monitoring_metrics:
        precision_per_class = metrics.precision_score(targets, preds, average=None)
        result["precision_per_class"] = precision_per_class
    if "f1_score_per_class" in monitoring_metrics:
        f1_score_per_class = metrics.f1_score(targets, preds, average=None)
        result["f1_score_per_class"] = f1_score_per_class
    return result