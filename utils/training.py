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

def train_model(config, model, train_loader, val_loader, device):
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_epochs = config.epochs

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
              f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        print(f"Train Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        print(f"Time taken for epoch: {time.time() - start_time:.2f}s")

    return history