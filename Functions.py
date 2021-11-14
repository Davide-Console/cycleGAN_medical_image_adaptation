# Packages
import matplotlib.pyplot as plt

import numpy as np

import torch


def image_viewer(image):
    # It plots a numpy image
    # INPUT: torch.tensor image
    image = image.detach().numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))


# Least square loss for discriminator
def LSGAN_D(real, fake):
    return (torch.mean((real - 1)**2) + torch.mean(fake**2))

#least square loss for generator
def LSGAN_G(fake):
    return (torch.mean((fake - 1)**2))


# Train
def Train(num_epochs, bs, G_A2B, G_B2A,optimizer_G_A2B, optimizer_G_B2A, D_A, D_B, 
          optimizer_D_A, optimizer_D_B, dataloader_train_CT, 
          dataloader_train_MR, criterion_Im, device):
    
    # Lists to keep track of progress
    # img_list = []
    G_losses = []  # array with generator losses
    D_A_losses = []  # array with discriminator A's Least Square Losses
    D_B_losses = []  # array with discriminator B's Least Square Losses
    
    
    iters = 0
    Full_Disc_Losses_A2B = []
    Full_Disc_Losses_B2A = []
    Cycle_Losses_A = []
    Cycle_Losses_B = []
    Identity_Losses_B2A = []
    Identity_Losses_A2B = []
    disc_A = []
    disc_B = []
    
    
    Full_Disc_Losses_A2B_t = []
    Full_Disc_Losses_B2A_t = []
    Cycle_Losses_A_t = []
    Cycle_Losses_B_t = []
    Identity_Losses_B2A_t = []
    Identity_Losses_A2B_t = []
    disc_A_t = []
    disc_B_t = []
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        print("epoch", epoch)
        # For each batch in the dataloader
        for i, (data_CT, data_MR) in enumerate(zip(dataloader_train_CT, dataloader_train_MR),0):
            print('batch', i)
            # Set model input
            A_real = data_CT[0].to(device=device)
            B_real = data_MR[0].to(device=device)

            # Genrated images using not updated generators
            B_fake = G_A2B(A_real)
            A_rec = G_B2A(B_fake)
            A_fake = G_B2A(B_real)
            B_rec = G_A2B(A_fake)


            # Discriminator A training
            # Computes discriminator loss by feeding real A and fake A samples in discriminator A
            optimizer_D_A.zero_grad() #sets gradient to zero
            Disc_loss_A = LSGAN_D(D_A(A_real), D_A(A_fake.detach()))  # computes Least Square Loss for discriminator A
            D_A_losses.append(Disc_loss_A.item())  # puts it into array
            
            Disc_loss_A.backward(retain_graph=True) #calculates backpropagation-derivative of Loss function with attention to weights
            optimizer_D_A.step() #updates computed values
    
            
            # Discriminator B training
            # Compute discriminator loss by feeding real B and fake B samples in discriminator B
            optimizer_D_B.zero_grad()
            Disc_loss_B = LSGAN_D(D_B(B_real), D_B(B_fake.detach()))
            D_B_losses.append(Disc_loss_B.item())
    
            Disc_loss_B.backward(retain_graph=True)
            optimizer_D_B.step()
    
            # Generators' gradients set to zero otherwise it accumulates them
            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()

            # least square loss for generators: Loss based on how many samples the discriminator has discovered
            Full_disc_loss_A2B = LSGAN_G(D_B(B_fake))
            Full_disc_loss_B2A = LSGAN_G(D_A(A_fake))
    
            # Cycle Consistency: Loss based on how much similar the starting image and the reconstructed images are
            Cycle_loss_A = criterion_Im(A_rec, A_real)*5  # lambda set to 5 because of Torch convention
            Cycle_loss_B = criterion_Im(B_rec, B_real)*5
    
            # Identity loss: Loss based on how much similar the starting image and the transformed images are
            Identity_loss_B2A = criterion_Im(G_B2A(A_real), A_real)*10
            Identity_loss_A2B = criterion_Im(G_A2B(B_real), B_real)*10
    
            # generator losses: sum of previous generator
            Loss_G = Full_disc_loss_A2B+Full_disc_loss_B2A+Cycle_loss_A+Cycle_loss_B+Identity_loss_B2A+Identity_loss_A2B
            G_losses.append(Loss_G)
    
            # Backward propagation: computes derivative of loss function based oh weights
            Loss_G.backward()

            # Optimization step: values are updated
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()

            # Puts calculated values in previously created matrixes
            Full_Disc_Losses_A2B.append(Full_disc_loss_A2B)
            Full_Disc_Losses_B2A.append(Full_disc_loss_B2A)
            Cycle_Losses_A.append(Cycle_loss_A)
            Cycle_Losses_B.append(Cycle_loss_B)
            Identity_Losses_B2A.append(Identity_loss_B2A)
            Identity_Losses_A2B.append(Identity_loss_A2B)
            disc_A.append(Disc_loss_A)
            disc_B.append(Disc_loss_B)

            # Saves old samples for next iterations and epochs

            iters += 1
            
            del data_MR, data_CT, A_real, B_real, A_fake, B_fake
    
    
            print('[%d/%d]\tFull_Disc_Losses_A2B: %.4f\tFull_Disc_Losses_B2A: %.4f\tCycle_Losses_A: %.4f\tCycle_Losses_B: %.4f\tIdentity_Losses_B2A: %.4f\tIdentity_Losses_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_B: %.4f'
                          % (epoch+1, num_epochs, Full_disc_loss_A2B, Full_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Identity_loss_B2A,
                              Identity_loss_A2B, Disc_loss_A.item(), Disc_loss_B.item()))
            
    
        Full_Disc_Losses_A2B_t.append(sum(Full_Disc_Losses_A2B)/len(Full_Disc_Losses_A2B))
        Full_Disc_Losses_B2A_t.append(sum(Full_Disc_Losses_B2A)/len(Full_Disc_Losses_B2A))
        Cycle_Losses_A_t.append(sum(Cycle_Losses_A)/len(Cycle_Losses_A))
        Cycle_Losses_B_t.append(sum(Cycle_Losses_B)/len(Cycle_Losses_B))
        Identity_Losses_B2A_t.append(sum(Identity_Losses_B2A)/len(Identity_Losses_B2A))
        Identity_Losses_A2B_t.append(sum(Identity_Losses_A2B)/len(Identity_Losses_A2B))
        disc_A_t.append(sum(disc_A)/len(disc_A))
        disc_B_t.append(sum(disc_B)/len(disc_B))
    
        Full_Disc_Losses_A2B = []
        Full_Disc_Losses_B2A = []
        Cycle_Losses_A = []
        Cycle_Losses_B = []
        Identity_Losses_B2A = []
        Identity_Losses_A2B = []
        disc_B = []
        disc_A = []

