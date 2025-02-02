#TODO Import libraries

class EarlyStopping():
    def __init__(self, patience, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Model saved with validation loss: {val_loss}')


def train(dataloader, model, loss, optimizer):
    model.train()
    predictions = []
    truths = []
    losses = []
    counter = []
    loads = []
    for tensor in dataloader:
        optimizer.zero_grad()
        prediction = model(tensor.to(device))
        divergence = loss(prediction, tensor.y)
        divergence.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        predictions.append(prediction[0])
        truths.append(tensor.y)
        losses.append(divergence)
        counter.append(prediction[2])
        loads.append(prediction[1])

    return predictions, truths, loads, counter


def evaluate(dataloader, model, loss):
    model.eval()
    predictions = []
    truths = []
    losses = []
    loads = []
    counter = []
    wx = []
    with torch.no_grad():
        for tensor in dataloader:
            prediction = model(tensor.to(device))
            divergence = loss(prediction, tensor.y)
            predictions.append(prediction[0])
            truths.append(tensor.y)
            losses.append(divergence)
            loads.append(prediction[1])
            counter.append(prediction[2])
            wx.append(prediction[3])

    return predictions, truths, loads, counter, wx



def train_evaluate(trainloader, evaloader, model, criterion, loss, optimizer, scheduler, patience, epochs):
    #track functions
    train_history, eval_history = [],[]
    early_stopping = EarlyStopping(patience=epochs)
    temp = 0
    train_accu=[]
    train_precision=[]
    train_recall=[]
    train_load=[]
    eval_accu=[]
    eval_precision=[]
    eval_recall=[]
    eval_load=[]
    for epoch in range(epochs):
        running_loss = 0.0
        train_res = train(dataloader=trainloader, model=model, loss=loss, optimizer=optimizer)
        eval_res = evaluate(dataloader=evaloader, model=model, loss=loss)
        train_history.append(train_res)
        eval_history.append(eval_res)
        if scheduler is not None:
            if temp == patience:
                model.load_state_dict(torch.load('checkpoint.pt'))
                print("best model loaded before stepping")
            scheduler.step()
        temp +=1
        train_loss = criterion(torch.cat(train_res[0]), torch.cat(train_res[1]))
        eval_loss = criterion(torch.cat(eval_res[0]), torch.cat(eval_res[1]))
        train_pred_epoch = torch.argmax(torch.cat(train_res[0]),dim=1)
        eval_pred_epoch = torch.argmax(torch.cat(eval_res[0]),dim=1)
        train_accu.append(accuracy_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_accu.append(accuracy_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))
        train_precision.append(precision_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_precision.append(precision_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))
        train_recall.append(recall_score(torch.cat(train_res[1]).cpu(), train_pred_epoch.cpu()))
        eval_recall.append(recall_score(torch.cat(eval_res[1]).cpu(), eval_pred_epoch.cpu()))
        train_load.append(torch.mean(torch.tensor(train_res[3])))
        eval_load.append(torch.mean(torch.tensor(eval_res[3])))

        print("training loss ", (train_loss), "evaluation loss ", (eval_loss))
        print("epoch: " + str(epoch+1))

        early_stopping(eval_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    train_lists = [train_accu, train_precision, train_recall, train_load]
    eval_lists = [eval_accu, eval_precision, eval_recall, eval_load]


    return train_history, eval_history, train_lists, eval_lists
