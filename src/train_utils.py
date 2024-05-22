import numpy as np
import sklearn.metrics as metrics


def one_iter(model, criterion, loader, device, train=True, optimizer=None, scheduler=None, monitoring_metrics=list()):
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
    if "confusion_matrix" in monitoring_metrics:
        cm = metrics.confusion_matrix(targets, preds)
        result["confusion_matrix"] = cm
    if "recall_per_class" in monitoring_metrics:
        recall_per_class = metrics.recall_score(targets, preds, average=None)
        result["recall_per_class"] = recall_per_class

    return result