#TODO Import libraries

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Early stopping class to prevent overfitting and stop training when validation loss stops improving
class EarlyStopping():
    def __init__(self, patience, delta=0, path='checkpoint.pt'):
        self.patience = patience  # Number of epochs to wait before stopping after no improvement
        self.delta = delta  # Minimum change in loss to qualify as an improvement
        self.path = path  # Path to save the best model
        self.counter = 0  # Counts the number of epochs with no improvement
        self.best_score = None  # Tracks the best validation score
        self.early_stop = False  # Flag to indicate if training should stop

    def __call__(self, val_loss, model):
        score = -val_loss  # Convert loss to a score (minimization problem)
        
        # Initialize best score in the first call
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            # No improvement, increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training
        else:
            # Improvement found, save model and reset counter
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        print(f'Model saved with validation loss: {val_loss}')


# Training function
def train(dataloader, model, loss, optimizer):
    model.train()  # Set model to training mode
    predictions = []
    truths = []
    losses = []
    counter = [] #Tensor of shape (num_xprtz) tracking the number of samples processed by each expert.
    loads = [] #Load distribution tensor of shape (num_xprtz), representing the computational load for each expert.

    for tensor in dataloader:
        optimizer.zero_grad()  # Reset gradients
        prediction = model(tensor.to(device))  # Forward pass
        divergence = loss(prediction, tensor.y)  # Compute loss
        
        divergence.backward()  # Backpropagation
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update model weights

        # Store results for analysis
        predictions.append(prediction[0])
        truths.append(tensor.y)
        losses.append(divergence)
        counter.append(prediction[2])
        loads.append(prediction[1])

    return predictions, truths, loads, counter


# Evaluation function (similar to training but without gradient updates)
def evaluate(dataloader, model, loss):
    model.eval()  # Set model to evaluation mode
    predictions = []
    truths = []
    losses = []
    loads = [] #Load distribution tensor of shape (num_xprtz), representing the computational load for each expert.
    counter = [] #Tensor of shape (num_xprtz) tracking the number of samples processed by each expert.
    wx = [] #Tensor tracking the assignment of nodes to experts across all layers.

    with torch.no_grad():  # Disable gradient computation for efficiency
        for tensor in dataloader:
            prediction = model(tensor.to(device))  # Forward pass
            divergence = loss(prediction, tensor.y)  # Compute loss

            # Store results for analysis
            predictions.append(prediction[0])
            truths.append(tensor.y)
            losses.append(divergence)
            loads.append(prediction[1])
            counter.append(prediction[2])
            wx.append(prediction[3])  

    return predictions, truths, loads, counter, wx


# Training and evaluation loop
def train_evaluate(trainloader, evaloader, model, criterion, loss, optimizer, scheduler, patience, epochs):
    train_history, eval_history = [], []
    early_stopping = EarlyStopping(patience=epochs)  # Initialize early stopping
    temp = 0

    # Metrics tracking
    train_accu, train_precision, train_recall, train_load = [], [], [], []
    eval_accu, eval_precision, eval_recall, eval_load = [], [], [], []

    for epoch in range(epochs):
        running_loss = 0.0

        # Perform training and evaluation
        train_res = train(dataloader=trainloader, model=model, loss=loss, optimizer=optimizer)
        eval_res = evaluate(dataloader=evaloader, model=model, loss=loss)

        # Store history
        train_history.append(train_res)
        eval_history.append(eval_res)

        # Learning rate scheduling (if applicable)
        if scheduler is not None:
            if temp == patience:
                model.load_state_dict(torch.load('checkpoint.pt'))  # Load best model
                print("Best model loaded before stepping")
            scheduler.step()
        temp += 1

        # Compute overall losses for training and evaluation
        train_loss = criterion(torch.cat(train_res[0]), torch.cat(train_res[1]))
        eval_loss = criterion(torch.cat(eval_res[0]), torch.cat(eval_res[1]))

        # Convert predictions to class labels
        train_pred_epoch = torch.argmax(torch.cat(train_res[0]), dim=1)
        eval_pred_epoch = torch.argmax(torch.cat(eval_res[0]), dim=1)

        # Compute classification metrics
        train_accu.append(accuracy_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_accu.append(accuracy_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))

        train_precision.append(precision_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_precision.append(precision_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))

        train_recall.append(recall_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_recall.append(recall_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))

        # Compute load balancing statistics
        train_load.append(torch.mean(torch.tensor(train_res[3])))
        eval_load.append(torch.mean(torch.tensor(eval_res[3])))

        # Print training progress
        print("Training loss:", train_loss, "Evaluation loss:", eval_loss)
        print("Epoch:", epoch + 1)

        # Check for early stopping
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model before returning
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Organize training and evaluation metrics
    train_lists = [train_accu, train_precision, train_recall, train_load]
    eval_lists = [eval_accu, eval_precision, eval_recall, eval_load]

    return train_history, eval_history, train_lists, eval_lists

