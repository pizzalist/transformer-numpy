from tqdm import tqdm
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from transformer import build_transformer
# from validation import run_validation
from dataset import get_ds

def get_model(config, vocab_src_len, vocab_tgt_len):
    
    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def get_config():
    return{
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512, # Dimensions of the embeddings in the Transformer. 512 like in the "Attention Is All You Need" paper.
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }
    

# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder'] # Extracting model folder from the config
    model_basename = config['model_basename'] # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt" # Building filename
    return str(Path('.')/ model_folder/ model_filename) # Combining current directory, the model folder, and the model filename

def train_model(config):
    # Setting up device to run on GPU to train faster
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    # Creating model directory to store weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Initializing model on the GPU using the 'get_model' function
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    
    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0
    
    # Checking if there is a pre-trained model to load
    # If true, loads it
    # if config['preload']:
    #     model_filename = get_weights_file_path(config, config['preload'])
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename) # Loading model
        
    #     # Sets epoch to the saved in the state plus one, to resume from where it stopped
    #     initial_epoch = state['epoch'] + 1
    #     # Loading the optimizer state from the saved model
    #     # optimizer.load_state_dict(state['optimizer_state_dict'])
    #     # Loading the global step state from the saved model
    #     global_step = state['global_step']
        
    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    # loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
    
    # Initializing training loop 
    
    # Iterating over each epoch from the 'initial_epoch' variable up to
    # the number of epochs informed in the config
    for epoch in range(initial_epoch, config['num_epochs']):
        
        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        
        # For each batch...
        for batch in batch_iterator:
            # data 로더 과정에서 encoder decoder 무슨 차이 여기서? 
            #   -> encoder는 src, decoder는 tgt이다.
            # 왜 한 batch안에 8개 문장이 들어가는지?
            #   -> config에 'batch_size': 8로 설정 해놔서
            # 350의 의미는?
            #  -> 350개의 단어를 의미한다.
            # model.train() # Train the model
            
            # Loading input data and masks onto the GPU
            encoder_input = batch['encoder_input'] # torch.Size([8, 350])
            decoder_input = batch['decoder_input'] # torch.Size([8, 350])
            encoder_mask = batch['encoder_mask']   # torch.Size([8, 1, 1, 350])
            decoder_mask = batch['decoder_mask']   # torch.Size([8, 1, 350, 350])
            
            # Running tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask) 
            # encoder_output.shape = (8, 350, 512)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # torch.Size([8, 350, 22463]) (batch_size, seq_len, target vocab_size)
            # output에 나오는 softmax의 확률은 어떤 의미를 가질까? vocab_size는 왜 target vocab_size인가?
            #  -> decoder output에 나오는 softmax의 확률은 encoder input에 들어간 단어에 대한 target vocab의 확률이다.
            
            # Loading the target labels onto the GPU
            # label = batch['label'].to(device) # label.shape = torch.Size([8, 350])
            label = batch['label']
            # Computing loss between model's output and true labels
            # loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # # Updating progress bar
            # batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()
            
            # # Performing backpropagation
            # loss.backward()
            
            # # Updating parameters based on the gradients
            # optimizer.step()
            
            # # Clearing the gradients to prepare for the next batch
            # optimizer.zero_grad()
            
            global_step += 1 # Updating global step count
            
        # # We run the 'run_validation' function at the end of each epoch
        # # to evaluate model performance
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
         
        # # Saving model
        # model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # # Writting current model state to the 'model_filename'
        # torch.save({
        #     'epoch': epoch, # Current epoch
        #     'model_state_dict': model.state_dict(),# Current model state
        #     'optimizer_state_dict': optimizer.state_dict(), # Current optimizer state
        #     'global_step': global_step # Current global step 
        # }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore') # Filtering warnings
    config = get_config() # Retrieving config settings
    train_model(config) # Training model with the config arguments
