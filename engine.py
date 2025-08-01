from torch import device, nn, optim
import torch
from torch.cuda import is_available
from torch.optim import optimizer
from torch.utils.data import DataLoader
#from setup import device
from tqdm.auto import tqdm
from torch.utils import tensorboard
def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer):
    """Function.

    model: torch.nn.Module model which needs to be trained
    dataloader : torch.utils.data.DataLoader Train data
    loss_fn:torch.nn.Module The loss function
    optimizer : torch.optim.Optimizer The optimizer used for trainig the modelnn.CrossEntropyLoss()

    
    Returns:
    Description.
    """
        
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    train_loss , train_acc = 0,0
    for BATCH , (X,y) in enumerate(dataloader):
        X , y = X.to(device) , y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              ):
    """Function.

    Args:
        model: torch.nn.Module model which needs to be trained
        dataloader : torch.utils.data.DataLoader Train data
        loss_fn:torch.nn.Module The loss function

        
    Returns:
        Description.
    """
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    test_loss, test_acc = 0,0
    with torch.inference_mode():
        for Batch , (X,y) in enumerate(dataloader):
            X , y = X.to(device) , y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(test_pred_logits,dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)


    return test_loss,test_acc


def train(model:torch.nn.Module ,
          train_dataloader : torch.utils.data.DataLoader,
          test_dataloader :torch.utils.data.DataLoader,
          optimizer : torch.optim,
          writer : torch.utils.tensorboard.SummaryWriter,
          loss_fn:torch.nn.Module = nn.CrossEntropyLoss(),
          epochs : int = 5,):
          
          
    """train. Trains a Neural Model

    Args:
        model: torch.nn.Module model which needs to be trained
        train_dataloader : torch.utils.data.DataLoader Train data
        test_dataloader : torch.utils.data.DataLoader test data
        optimizer : torch.optim.Optimizer The optimizer used for trainig the modelnn.CrossEntropyLoss()
        writer : torch.utils.tensorboard.SummaryWriter object
        loss_fn:torch.nn.Module The loss function
        epochs (int): number of epochs 
    Returns:
        .None
    """
    
        
    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]
               }

    for epoch in tqdm(range(epochs)):
        train_loss , train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer)

        test_loss , test_acc = test_step(model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn
                                         )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss : {test_loss:.4f} | "
            f"test_acc : {test_acc:.4f} | "
        )

        results["train_loss"].append(train_loss.item() if isinstance(train_loss,torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc,torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss,torch.Tensor) else test_loss) 
        results["test_acc"].append(test_acc.item() if isinstance(test_acc,torch.Tensor) else test_acc)
        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss":train_loss,
                                            "test_loss":test_loss},
                           global_step = epoch)
        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict = {"train_acc":train_acc,
                                              "test_acc":test_acc},
                           global_step=epoch)
    writer.close()

    return results


    


        

        

        
    
    

    
    
