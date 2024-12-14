import numpy as np
import sklearn.metrics as metrics
import ipdb
import torch

#def one_iter(model, criterion, loader, device, train=True, optimizer=None, scheduler=None, monitoring_metrics=list()):
    #parser = TrainConfigParser()
    #model_name = parser.get_model_name()
 #   model_name = model.__class__.__name__ 
 #   running_loss = 0.0
 #   count = 0.0
 #   if train:
 #       model.train()
 #   else:
 #       model.eval()

 #   preds_arr = []
 #   targets_arr = []
 #   for data, labels in (loader):
 #       batch_size = len(labels)
 #       data = data.to(device)
 #       labels = labels.to(device)
 #       if optimizer is not None:
 #           optimizer.zero_grad()
 #       outputs = model(data)
 #       if hasattr(model.module, 'config') and hasattr(model.module.config, 'model_type') \
 #           and (model.module.config.model_type in ["vit", "deit", "swin", "vit_mae"]):
 #           outputs = outputs.logits
 #       loss = criterion(outputs, labels)
 #       if train:
 #           loss.backward()
 #           optimizer.step()

 #       preds = outputs.max(dim=1)[1]
 #       count += batch_size
 #       running_loss += loss.item() * batch_size

 #       targets_arr.append(labels.cpu().numpy())
 #       preds_arr.append(preds.detach().cpu().numpy())

 #       if train and scheduler is not None:
 #           scheduler.step()

  #  targets = np.concatenate(targets_arr)
  #  preds = np.concatenate(preds_arr)

  #  loss = running_loss*1.0/count

  #  result = {"loss": loss}
  #  if "accuracy" in monitoring_metrics:
  #      accuracy = metrics.accuracy_score(targets, preds)
  #      result["accuracy"] = accuracy
  #  if "balanced_accuracy" in monitoring_metrics:
  #      balanced_accuracy = metrics.balanced_accuracy_score(targets, preds)
  #      result["balanced_accuracy"] = balanced_accuracy
  #  if "macro_avg_precision" in monitoring_metrics:
  #      macro_avg_precision = metrics.precision_score(targets, preds, average="macro")
  #      result["macro_avg_precision"] = macro_avg_precision
  #  if "macro_avg_f1_score" in monitoring_metrics:
  #      macro_avg_f1_score = metrics.f1_score(targets, preds, average="macro")
  #      result["macro_avg_f1_score"] = macro_avg_f1_score
  #  if "confusion_matrix" in monitoring_metrics:
  #      cm = metrics.confusion_matrix(targets, preds)
  #      result["confusion_matrix"] = cm
  #  if "recall_per_class" in monitoring_metrics:
  #      recall_per_class = metrics.recall_score(targets, preds, average=None)
  #      result["recall_per_class"] = recall_per_class
  #  if "precision_per_class" in monitoring_metrics:
  #      precision_per_class = metrics.precision_score(targets, preds, average=None)
  #      result["precision_per_class"] = precision_per_class
  #  if "f1_score_per_class" in monitoring_metrics:
  #      f1_score_per_class = metrics.f1_score(targets, preds, average=None)
  #      result["f1_score_per_class"] = f1_score_per_class
  #  return result

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

def one_iter(model, criterion, loader, device, train=True, optimizer=None, scheduler=None, monitoring_metrics=list(), mixup_alpha=0.0):
    model_name = model.__class__.__name__ 
    running_loss = 0.0
    count = 0.0
    running_corrects = 0.0
    if train:
        model.train()
    else:
        model.eval()

    preds_arr = []
    targets_arr = []
    with torch.set_grad_enabled(train):
        for data, labels in loader:
            batch_size = len(labels)
            data = data.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            if train and mixup_alpha > 0.0:
                # Apply mixup
                data, targets_a, targets_b, lam = mixup_data(data, labels, alpha=mixup_alpha)
                outputs = model(data)
                if hasattr(model.module, 'config') and hasattr(model.module.config, 'model_type') \
                    and (model.module.config.model_type in ["vit", "deit", "swin", "vit_mae"]):
                    outputs = outputs.logits
                # Compute mixup loss
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                # For accuracy calculation, we can use targets_a and targets_b
                preds = outputs.max(dim=1)[1]
                correct_a = preds.eq(targets_a).float()
                correct_b = preds.eq(targets_b).float()
                correct = lam * correct_a + (1 - lam) * correct_b
                running_corrects += correct.sum().item()
            else:
                # No mixup
                outputs = model(data)
                if hasattr(model.module, 'config') and hasattr(model.module.config, 'model_type') \
                    and (model.module.config.model_type in ["vit", "deit", "swin", "vit_mae"]):
                    outputs = outputs.logits
                loss = criterion(outputs, labels)
                preds = outputs.max(dim=1)[1]
                correct = preds.eq(labels).float()
                running_corrects += correct.sum().item()

            if train:
                loss.backward()
                optimizer.step()

            count += batch_size
            running_loss += loss.item() * batch_size

            preds_arr.append(preds.detach().cpu().numpy())
            targets_arr.append(labels.cpu().numpy())

            if train and scheduler is not None:
                scheduler.step()

    loss = running_loss / count
    accuracy = running_corrects / count

    result = {"loss": loss}
    if "accuracy" in monitoring_metrics:
        result["accuracy"] = accuracy

    # For non-training iterations or when mixup is not used, compute additional metrics
    if not train or mixup_alpha == 0.0:
        targets = np.concatenate(targets_arr)
        preds = np.concatenate(preds_arr)
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