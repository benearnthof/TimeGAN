import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import random_generator
from utils import extract_time
import matplotlib.pyplot as plt

class Time_GAN_module(nn.Module):
    """
    Class from which a module of the Time GAN Architecture can be constructed, 
    consisting of a n_layer stacked RNN layers and a fully connected layer
    
    input_size = dim of data (depending if module operates on latent or non-latent space)
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, activation=torch.sigmoid, rnn_type="gru"):
        super(Time_GAN_module, self).__init__()

        # Parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sigma = activation
        self.rnn_type = rnn_type

        #Defining the layers
        # RNN Layer
        if self.rnn_type == "gru":
          self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        elif self.rnn_type == "rnn":
          self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first = True) 
        elif self.rnn_type == "lstm": # input params still the same for lstm
          self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first = True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
    
            batch_size = x.size(0)

            # Initializing hidden state for first input using method defined below
            if self.rnn_type in ["rnn", "gru"]:
              hidden = self.init_hidden(batch_size)
            elif self.rnn_type == "lstm": # additional initial cell state for lstm
              h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).float()
              c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).float()
              hidden = (h0, c0)
            # Passing in the input and hidden state into the model and obtaining outputs
            out, hidden = self.rnn(x, hidden)
        
            # Reshaping the outputs such that it can be fit into the fully connected layer
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
            
            if self.sigma == nn.Identity:
                idendity = nn.Identity()
                return idendity(out)
                
            out = self.sigma(out)
            
            # HIDDEN STATES WERDEN IN DER PAPER IMPLEMENTIERUNG AUCH COMPUTED, ALLERDINGS NICHT BENUTZT?
            
            return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
def TimeGAN(data, parameters):
  """
  Main workhorse function of timegan
  Args: 
    - data = Time series data
    - parameters = dictionary of training parameters
  
  hidden_dim = dimension of hidden layers, integer
  num_layers = number of recurrent layers, integer
  iterations = number of training iterations every epoch, integer  
  batch_size = number of samples passed in every training batch, integer [32 - 128 works best]
  epoch = number of training epochs
  """
  hidden_dim = parameters["hidden_dim"]
  num_layers = parameters["num_layers"]
  iterations = parameters["iterations"]
  batch_size = parameters["batch_size"]
  module = parameters["module"]
  epoch = parameters["epoch"]
  no, seq_len, dim = np.asarray(data).shape
  z_dim = dim
  gamma = 1

  checkpoints = {}

  # instantiating every module we're going to train
  Embedder = Time_GAN_module(input_size=z_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers)
  Recovery = Time_GAN_module(input_size=hidden_dim, output_size=dim, hidden_dim=hidden_dim, n_layers=num_layers)
  Generator = Time_GAN_module(input_size=dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers)
  Supervisor = Time_GAN_module(input_size=hidden_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=num_layers)
  Discriminator = Time_GAN_module(input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim, n_layers=num_layers, activation=nn.Identity)

  # instantiating all optimizers, 
  # learning rates chosen through experimentation and comparison with results in the paper
  embedder_optimizer = optim.Adam(Embedder.parameters(), lr=0.0035)
  recovery_optimizer = optim.Adam(Recovery.parameters(), lr=0.01)
  supervisor_optimizer = optim.Adam(Recovery.parameters(), lr=0.001)
  discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=0.01)
  generator_optimizer = optim.Adam(Generator.parameters(), lr=0.01)
  
  # instantiating mse loss & Data Loader
  binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
  MSE_loss = nn.MSELoss()
  loader = DataLoader(data, parameters['batch_size'], shuffle=False)
  random_data = random_generator(batch_size=parameters['batch_size'], z_dim=dim, 
                                       T_mb=extract_time(data)[0], max_seq_len=extract_time(data)[1])
  
  # Embedding Network Training
  # Here we train embedding & Recovery network jointly
  print('Start Embedding Network Training')
  for e in range(epoch): 
    for batch_index, X in enumerate(loader):
                
        H, _ = Embedder(X.float())
        H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

        X_tilde, _ = Recovery(H)
        X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

        # constants chosen like in the paper
        E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))  

        Embedder.zero_grad()
        Recovery.zero_grad()

        E_loss0.backward(retain_graph=True)

        embedder_optimizer.step()
        recovery_optimizer.step()

        if e in range(1,epoch) and batch_index == 0:
            print('step: '+ str(e) + '/' + str(epoch) + ', e_loss: ' + str(np.sqrt(E_loss0.detach().numpy())))

  print('Finish Embedding Network Training')

  # Training only with supervised loss
  # here the embedder and supervisor are trained jointly
  print('Start Training with Supervised Loss Only')
  for e in range(epoch): 
    for batch_index, X in enumerate(loader):

        H, _ = Embedder(X.float())
        H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

        H_hat_supervise, _ = Supervisor(H)
        H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))  

        G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])


        Embedder.zero_grad()
        Supervisor.zero_grad()

        G_loss_S.backward(retain_graph=True)

        embedder_optimizer.step()
        supervisor_optimizer.step()

        if e in range(1,epoch) and batch_index == 0:
            print('step: '+ str(e) + '/' + str(epoch) + ', s_loss: ' + str(np.sqrt(G_loss_S.detach().numpy())))

  print('Finish Training with Supervised Loss Only')
  
  
  # Joint Training
  print('Start Joint Training')
  for itt in range(epoch):
    # the generator, supervisor and discriminator are trained for an extra two steps
    # in the issues on github the author mentions, the marked constant of 0.15 (below)
    # had been chosen because it worked well in experiments, and to keep the balance in
    # training the generator and discriminator.
    for kk in range(2):
      X = next(iter(loader))
      random_data #= random_generator(batch_size=batch_size, z_dim=dim, 
                                       #T_mb=extract_time(data)[0], max_seq_len=extract_time(data)[1])
        
      # Generator Training 
      ## Train Generator
      z = torch.tensor(random_data)
      z = z.float()
        
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

      Generator.zero_grad()
      Supervisor.zero_grad()
      Discriminator.zero_grad()
      Recovery.zero_grad()

      # line 267 of original implementation: 
      # G_loss_U, G_loss_S, G_loss_V
      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
      # logits first, then targets
      # D_loss_real(Y_real, torch.ones_like(Y_real))
      G_loss_U = binary_cross_entropy_loss(Y_fake, torch.ones_like(Y_fake))
        
      G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
      G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X, [0])))))
      G_loss_V = G_loss_V1 + G_loss_V2
        
      # doing a backward step for each loss should result in gradients accumulating 
      # so we should be able to optimize them jointly
      G_loss_S.backward(retain_graph=True)#
      G_loss_U.backward(retain_graph=True)
      G_loss_V.backward(retain_graph=True)#


      generator_optimizer.step()
      supervisor_optimizer.step()
      discriminator_optimizer.step()
      # Train Embedder 
      ## line 270: we only optimize E_loss_T0
      ## E_loss_T0 = just mse of x and x_tilde
      # but it calls E_solver which optimizes E_loss, which is a sum of 
      # E_loss0 and 0.1* G_loss_S
      MSE_loss = nn.MSELoss()
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      X_tilde, _ = Recovery(H)
      X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

      E_loss_T0 = MSE_loss(X, X_tilde)
      E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))  
        
      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))  

      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      E_loss = E_loss0  + 0.1 * G_loss_S
        
      G_loss_S.backward(retain_graph=True)
      E_loss_T0.backward()
        
      Embedder.zero_grad()
      Recovery.zero_grad()
      Supervisor.zero_grad()
        
      embedder_optimizer.step()
      recovery_optimizer.step()
      supervisor_optimizer.step()
    # train Discriminator
    for batch_index, X in enumerate(loader):
      random_data #= random_generator(batch_size=batch_size, z_dim=dim, 
                                       #T_mb=extract_time(data)[0], max_seq_len=extract_time(data)[1])
      
      z = torch.tensor(random_data)
      z = z.float()

      H, _ = Embedder(X)
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      Y_real = Discriminator(H)
      Y_real = torch.reshape(Y_real, (batch_size, seq_len, 1))
      
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))

      Y_fake_e = Discriminator(e_hat)
      Y_fake_e = torch.reshape(Y_fake_e, (batch_size, seq_len, 1))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))

      Generator.zero_grad()
      Supervisor.zero_grad()
      Discriminator.zero_grad()
      Recovery.zero_grad()
      Embedder.zero_grad()

      # logits first, then targets
      # D_loss_real(Y_real, torch.ones_like(Y_real))
      D_loss_real = nn.BCEWithLogitsLoss()
      DLR = D_loss_real(Y_real, torch.ones_like(Y_real))

      D_loss_fake = nn.BCEWithLogitsLoss()
      DLF = D_loss_fake(Y_fake, torch.zeros_like(Y_fake))

      D_loss_fake_e = nn.BCEWithLogitsLoss()
      DLF_e = D_loss_fake_e(Y_fake_e, torch.zeros_like(Y_fake_e))

      D_loss = DLR + DLF + gamma * DLF_e

      # check discriminator loss before updating
      check_d_loss = D_loss
      # This is the magic number 0.15 we mentioned above. Set exactly like in the original implementation
      if (check_d_loss > 0.15):
        D_loss.backward(retain_graph=True)
        discriminator_optimizer.step()        
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim)) 
        
      X_tilde, _ = Recovery(H)
      X_tilde = torch.reshape(X_tilde, (batch_size, seq_len, dim))

      
      z = torch.tensor(random_data)
      z = z.float()
        
      e_hat, _ = Generator(z)
      e_hat = torch.reshape(e_hat, (batch_size, seq_len, hidden_dim))
        
      H_hat, _ = Supervisor(e_hat)
      H_hat = torch.reshape(H_hat, (batch_size, seq_len, hidden_dim))
        
      Y_fake = Discriminator(H_hat)
      Y_fake = torch.reshape(Y_fake, (batch_size, seq_len, 1))
        
      x_hat, _ = Recovery(H_hat)
      x_hat = torch.reshape(x_hat, (batch_size, seq_len, dim))
        
      H, _ = Embedder(X.float())
      H = torch.reshape(H, (batch_size, seq_len, hidden_dim))

      H_hat_supervise, _ = Supervisor(H)
      H_hat_supervise = torch.reshape(H_hat_supervise, (batch_size, seq_len, hidden_dim))

      G_loss_S = MSE_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])
      binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
      # logits first then targets
      G_loss_U = binary_cross_entropy_loss(Y_fake, torch.ones_like(Y_fake))
        
      G_loss_V1 = torch.mean(torch.abs((torch.std(x_hat, [0], unbiased = False)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
      G_loss_V2 = torch.mean(torch.abs((torch.mean(x_hat, [0]) - (torch.mean(X, [0])))))
      G_loss_V = G_loss_V1 + G_loss_V2
    
      E_loss_T0 = MSE_loss(X, X_tilde)
      E_loss0 = 10 * torch.sqrt(MSE_loss(X, X_tilde))  
      E_loss = E_loss0  + 0.1 * G_loss_S
        
      # doing a backward step for each loss should result in gradients accumulating 
      # so we should be able to optimize them jointly
      G_loss_S.backward(retain_graph=True)#
      G_loss_U.backward(retain_graph=True)
      G_loss_V.backward(retain_graph=True)#
      E_loss.backward()

      generator_optimizer.step()
      supervisor_optimizer.step()
      embedder_optimizer.step()
      recovery_optimizer.step()
            
      print('step: '+ str(itt) + '/' + str(epoch) + 
            ', D_loss: ' + str(D_loss.detach().numpy()) +
            ', G_loss_U: ' + str(G_loss_U.detach().numpy()) + 
            ', G_loss_S: ' + str(G_loss_S.detach().numpy()) + 
            ', E_loss_t0: ' + str(np.sqrt(E_loss0.detach().numpy())))
         

      
      random_test = random_generator(1, dim, extract_time(data)[0], extract_time(data)[1])        
      test_sample = Generator(torch.tensor(random_generator(1, dim, extract_time(data)[0], extract_time(data)[1])).float())[0]      
      test_sample = torch.reshape(test_sample, (1, seq_len, hidden_dim))
      test_recovery = Recovery(test_sample)
      test_recovery = torch.reshape(test_recovery[0], (1, seq_len, dim))
      fig, ax = plt.subplots()
      ax1 = plt.plot(test_recovery[0].detach().numpy())
      plt.show()
      
      if itt % 2:
        checkpoints[itt] = [Generator.state_dict(), Discriminator.state_dict(), Embedder.state_dict(), Recovery.state_dict,
                    Supervisor.state_dict()]
             
  print('Finish Joint Training')
                
  return Generator, Embedder, Supervisor, Recovery, Discriminator, checkpoints