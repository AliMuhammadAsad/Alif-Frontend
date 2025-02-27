import torch
from tqdm import tqdm
from .config import *
from .data_loader import TextDataLoader
from .model import GPTLanguageModel
import math

# max_lr = 6e-4
# min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50
# def get_lr(it):
#     if it < warmup_steps:
#         return max_lr * (it+1) / warmup_steps
    
#     if it > max_steps:
#         return min_lr
    
#     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 +math.cos(math.pi * decay_ratio))
#     return min_lr + coeff * (max_lr - min_lr)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from DataLoader import create_dataloader



def train(file_path, tokenizer, model=None, optimizer=None, vocab_size=10000, platform='none', checkpoint=None):

    torch.set_float32_matmul_precision('high')  #hammad added this line (need to check if it is necessary)
    if model is None:
        model = GPTLanguageModel(vocab_size=vocab_size)
        print("Model Initialised")
        if checkpoint != None:
            print("loading checkpoint........")
            model.load(checkpoint)
            print("Model loaded from checkpoint", checkpoint)

        if platform == 'kaggle':
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
            model = model.to(DEVICE)
        else:
            model = model.to(DEVICE)
            model = torch.compile(model) #hammad added this line
        # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas = (0.9, 0.95), eps = 1e-8)
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=LEARNING_RATE, device=DEVICE) #hammad added this line

    # # Initialize the data loader
    # loader = TextDataLoader(file_path, BATCH_SIZE, BLOCK_SIZE, tokenizer)

    loader = create_dataloader(tokenizer, file_path, BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE) #hammad added this line
    
    # Set up a tqdm progress bar for the epoch
    for epoch in range(MAX_ITERS):
        print(f"Epoch {epoch}")
        epoch_loss = None  # Track loss for the epoch
        
        # Create a progress bar for batch processing
        batch_progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Batch Progress", unit="batch", ncols=100)
        count = 0
        for xb, yb in batch_progress_bar:
            if xb is None:
                break  # No more batches, stop the epoch
            
            # Forward pass and loss computation
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16): #hammad added this line
                logits, loss = model(xb, yb)
            optimizer.zero_grad()
            if platform == 'kaggle':
                loss.mean().backward()
            else:
                loss.backward()  # Backpropagate the loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #hammad added this line
            # lr = get_lr(count) #need to check if this is correct
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            optimizer.step()  # Update model parameters
            
            # Update epoch_loss to the most recent loss value
            if platform == 'kaggle':
                epoch_loss = loss.mean().item()
            else:
                epoch_loss = loss.item()
            
            # Update tqdm with the latest loss value
            batch_progress_bar.set_postfix(loss=epoch_loss)

            count+=1
            if count%5000 == 0:
                if platform == 'kaggle':
                    torch.save(model.module.state_dict(), f"model_weights_checkpoint_{count}.pth")
                else:
                    torch.save(model.state_dict(), f"model_weights_checkpoint_{count}.pth")
                print(f"Model weights saved at checkpoint {count}")
        
        # Save model weights after each chunk or epoch
        if platform == 'kaggle':
            torch.save(model.module.state_dict(),
                       f"model_weights_epoch_{epoch}_{file_path[-6:-4]}.pth")
        else:
            torch.save(model.state_dict(),
                       f"model_weights_epoch_{epoch}_{file_path[-6:-4]}.pth")
        print(f"Model weights saved at epoch {epoch}")
        
        # Print the loss at the end of the epoch
        if epoch_loss is not None:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")
        else:
            print(f"Epoch {epoch}, No data available for loss calculation.")
        
        # Reset the loader for a new epoch
    #     loader.reset()
    
    # loader.close()  # Ensure the file is properly closed at the end

    return model, optimizer


#before parallelizing the model
# def train(file_path, tokenizer, model=None, optimizer=None, vocab_size=10000, platform='none'):
#     if model is None:
#         model = GPTLanguageModel(vocab_size=vocab_size)
#         if platform == 'kaggle':
#             model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(DEVICE)
#         else:
#             model = model.to(DEVICE)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

#     # Initialize the data loader
#     loader = TextDataLoader(file_path, BATCH_SIZE, BLOCK_SIZE, tokenizer, DEVICE)
    
#     # Set up a tqdm progress bar for the epoch
#     for epoch in range(MAX_ITERS):
#         print(f"Epoch {epoch}")
#         epoch_loss = None  # Track loss for the epoch
        
#         # Create a progress bar for batch processing
#         batch_progress_bar = tqdm(loader, total=loader.num_batches(), desc=f"Epoch {epoch+1} Batch Progress", unit="batch", ncols=100)
        
#         for xb, yb in batch_progress_bar:
#             if xb is None:
#                 break  # No more batches, stop the epoch
            
#             # Forward pass and loss computation
#             logits, loss = model(xb, yb)
#             optimizer.zero_grad()
#             loss.backward()  # Backpropagate the loss
#             optimizer.step()  # Update model parameters
            
#             # Update epoch_loss to the most recent loss value
#             epoch_loss = loss.item()
            
#             # Update tqdm with the latest loss value
#             batch_progress_bar.set_postfix(loss=epoch_loss)
        
#         # Save model weights after each chunk or epoch
#         model.save(f"model_weights_epoch_{epoch}.pth")
#         print(f"Model weights saved at epoch {epoch}")
        
#         # Print the loss at the end of the epoch
#         if epoch_loss is not None:
#             print(f"Epoch {epoch}, Loss: {epoch_loss}")
#         else:
#             print(f"Epoch {epoch}, No data available for loss calculation.")
        
#         # Reset the loader for a new epoch
#         loader.reset()
    
#     loader.close()  # Ensure the file is properly closed at the end

#     return model, optimizer

# def train(file_path, tokenizer, model = None, optimizer = None, vocab_size=10000, platform='none'):
#     if model is None:
#         model = GPTLanguageModel(vocab_size=vocab_size)
#         if platform == 'kaggle':
#             model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(DEVICE)
#         else:
#             model = model.to(DEVICE)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     loader = TextDataLoader(file_path, BATCH_SIZE, BLOCK_SIZE, tokenizer, DEVICE)


#     for epoch in range(MAX_ITERS):  # Iterate over the file chunks
#         print(f"Epoch {epoch}")
#         epoch_loss = None  # Track loss for the epoch
#         while not loader.end_of_file:
#             xb, yb = loader.get_batch()
#             if xb is None:
#                 break  # No more batches, stop the epoch

#             # Forward pass and loss computation
#             # print("This is xb", xb)
#             # print("This is yb", yb)
#             logits, loss = model(xb, yb)
#             optimizer.zero_grad()
#             loss.backward()   #2 gpus pe masla kr rraha (krna for n gpus hai)
#             optimizer.step()
            
#             # Update epoch_loss to the most recent loss value
#             epoch_loss = loss.item()

#         # Save model weights after each chunk or epoch
#         model.save(f"model_weights_epoch_{epoch}.pth")
#         print(f"Model weights saved at epoch {epoch}")
        
#         # Print the loss only if it was computed
#         if epoch_loss is not None:
#             print(f"Epoch {epoch}, Loss: {epoch_loss}")
#         else:
#             print(f"Epoch {epoch}, No data available for loss calculation.")

#         # Reset the loader for a new epoch
#         loader.reset()

#     loader.close()  # Ensure file is properly closed at the end

#     return model, optimizer


# def train(file_path, tokenizer, model=None, optimizer=None, vocab_size=10000):
#     # Check if multiple GPUs are available
#     device = DEVICE
#     if model is None:
#         if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#             print(f"Training on {torch.cuda.device_count()} GPUs")
#             model = GPTLanguageModel(vocab_size=vocab_size).to(device)
#             model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Wrap the model for multi-GPU training
#         else:
#             print("Training on a single GPU or CPU.")
            
#             model = GPTLanguageModel(vocab_size=vocab_size).to(device)
        
#     if optimizer is None:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
#     loader = TextDataLoader(file_path, BATCH_SIZE, BLOCK_SIZE, tokenizer, device)

#     for epoch in range(MAX_ITERS):  # Iterate over the file chunks
#         print(f"Epoch {epoch}")
#         epoch_loss = None  # Track loss for the epoch

#         xb, yb = loader.get_batch()
#         if xb is None:
#             break  # No more batches, stop the epoch

#         # Forward pass and loss computation
#         logits, loss = model(xb, yb)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Update epoch_loss to the most recent loss value
#         epoch_loss = loss.item()

#         # Save model weights after each chunk or epoch
#         model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model  # Get the underlying model if using DataParallel
#         model_to_save.save(f"model_weights_epoch_{epoch}.pth")
#         print(f"Model weights saved at epoch {epoch}")
        
#         # Print the loss only if it was computed
#         if epoch_loss is not None:
#             print(f"Epoch {epoch}, Loss: {epoch_loss}")
#         else:
#             print(f"Epoch {epoch}, No data available for loss calculation.")

#         # Reset the loader for a new epoch
#         loader.reset()

#     loader.close()  # Ensure file is properly closed at the end

#     return model, optimizer
