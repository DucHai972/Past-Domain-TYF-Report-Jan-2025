import torch
import time
from utils.metric import compute_metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum()
        total += labels.size(0)

    precision, recall, f1 = compute_metrics(all_labels, all_preds)
    return total_loss / len(train_loader), correct / total, precision, recall, f1

def validate_one_epoch(model, validation_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum()
            total += labels.size(0)

    precision, recall, f1 = compute_metrics(all_labels, all_preds)
    return total_loss / len(validation_loader), correct / total, precision, recall, f1

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, time_, history):
    best_val_loss = float('inf')
    best_model_path = f'./model/trained_model_{time_}_best.pth'
    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device)
        
        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['precision'].append(train_prec)
        history['recall'].append(train_rec)
        history['f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.2f}, ' 
              f'Train accuracy: {train_acc:.2f}, '
              f'Train precision: {train_prec:.2f}, '
              f'Train recall: {train_rec:.2f}, '
              f'Train F1: {train_f1:.2f}, '
              f'Validation Loss: {val_loss:.2f}, '
              f'Val accuracy: {val_acc:.2f}, '
              f'Val precision: {val_prec:.2f}, '
              f'Val recall: {val_rec:.2f}, '
              f'Val F1: {val_f1:.2f}'
        )
        
        end_time = time.time()
        print(f"Time one epoch: {end_time - start_time:.2f} seconds")
        
        # Optional: Implement early stopping or save best model here

    return model