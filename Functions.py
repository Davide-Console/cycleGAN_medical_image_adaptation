# Packages
import torch
import numpy as np
import matplotlib.pyplot as plt

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
          dataloader_train_MR, criterion_Im, device, old=True):
    
    # Lists to keep track of progress
    # img_list = []
    G_losses = []
    D_A_losses = []
    D_B_losses = []
    
    
    iters=0
    FDL_A2B = []
    FDL_B2A = []
    CL_A = []
    CL_B = []
    ID_B2A = []
    ID_A2B = []
    disc_A = []
    disc_B = []
    
    
    FDL_A2B_t = []
    FDL_B2A_t = []
    CL_A_t = []
    CL_B_t = []
    ID_B2A_t = []
    ID_A2B_t = []
    disc_A_t = []
    disc_B_t = []
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        print("epoch")
        print(epoch)
        # For each batch in the dataloader
        for i,(data_CT, data_MR) in enumerate(zip(dataloader_train_CT, dataloader_train_MR),0):
            print(i)
            # Set model input
            A_real = data_CT[0].to(device=device)
            B_real = data_MR[0].to(device=device)

            #tensor_ones = torch.ones([A_real.shape[0],1,14,14])
            #tensor_zeros = torch.zeros([A_real.shape[0],1,14,14])

            # Genrated images using not updated generators
            B_fake = G_A2B(A_real)
            A_rec = G_B2A(B_fake)
            A_fake = G_B2A(B_real)
            B_rec = G_A2B(A_fake)
    
            
            # Discriminator A training
            # Computes discriminator loss by feeding real A and fake A samples in discriminator A
            optimizer_D_A.zero_grad() #sets gradient to zero
            if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
                rand_int = torch.randint(5, old_A_fake.shape[0]-1, (1,1))
                Disc_loss_A = LSGAN_D(D_A(A_real), D_A(old_A_fake[rand_int-5:rand_int].detach()))
                D_A_losses.append(Disc_loss_A.item()) #puts into matrix
            else:
                Disc_loss_A = LSGAN_D(D_A(A_real), D_A(A_fake.detach())) #computes Least Square Loss for discriminator A
                D_A_losses.append(Disc_loss_A.item()) #puts it into matrix
            
            Disc_loss_A.backward() #calculates backpropagation-derivative of Loss function with attention to weights
            optimizer_D_A.step() #updates computed values
    
            
            # Discriminator B training
            # Compute discriminator loss by feeding real B and fake B samples in discriminator B
            optimizer_D_B.zero_grad() #sets gradient descent to zero
            if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
              rand_int = torch.randint(5, old_B_fake.shape[0]-1, (1,1))
              Disc_loss_B = LSGAN_D(D_B(B_real), D_B(old_B_fake[rand_int-5:rand_int].detach()))
              D_B_losses.append(Disc_loss_B.item())
            else:
              Disc_loss_B = LSGAN_D(D_B(B_real), D_B(B_fake.detach()))
              D_B_losses.append(Disc_loss_B.item())
    
            Disc_loss_B.backward() #calculates backprop
            optimizer_D_B.step()   #updates values
    
            # Generators' gradients set to zero otherwise it accumulates them
            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()
    
            # least square loss for generators: Loss based on how many samples the discriminator has discovered
            Fool_disc_loss_A2B = LSGAN_G(D_B(B_fake))
            Fool_disc_loss_B2A = LSGAN_G(D_A(A_fake))
    
            # Cycle Consistency: Loss based on how much similar the starting image and the reconstructed images are
            Cycle_loss_A = criterion_Im(A_rec, A_real)*5 #lambda set to 5 because of Torch convention
            Cycle_loss_B = criterion_Im(B_rec, B_real)*5
    
            # Identity loss: Loss based on how much similar the starting image and the transformed images are
            Id_loss_B2A = criterion_Im(G_B2A(A_real), A_real)*10
            Id_loss_A2B = criterion_Im(G_A2B(B_real), B_real)*10
    
            # generator losses: sum of previous generator
            Loss_G = Fool_disc_loss_A2B+Fool_disc_loss_B2A+Cycle_loss_A+Cycle_loss_B+Id_loss_B2A+Id_loss_A2B
            G_losses.append(Loss_G)
    
            # Backward propagation: computes derivative of loss function based oh weights
            Loss_G.backward()
            
            
            # Optimization step: values are updated
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()
            #puts calculated values in previously created matrixes
            FDL_A2B.append(Fool_disc_loss_A2B)
            FDL_B2A.append(Fool_disc_loss_B2A)
            CL_A.append(Cycle_loss_A)
            CL_B.append(Cycle_loss_B)
            ID_B2A.append(Id_loss_B2A)
            ID_A2B.append(Id_loss_A2B)
            disc_A.append(Disc_loss_A)
            disc_B.append(Disc_loss_B)

            # saves old samples for next iterations and epochs
            if(iters == 0 and epoch == 0):
              old_B_fake = B_fake.clone()
              old_A_fake = A_fake.clone()
            elif (old_B_fake.shape[0] == bs*5 and B_fake.shape[0]==bs):
              rand_int = torch.randint(5, 24, (1,1))
              old_B_fake[rand_int-5:rand_int] = B_fake.clone()
              old_A_fake[rand_int-5:rand_int] = A_fake.clone()
            elif(old_B_fake.shape[0]< 25):
              old_B_fake = torch.cat((B_fake.clone(),old_B_fake))
              old_A_fake = torch.cat((A_fake.clone(),old_A_fake))
    
            iters += 1
            
            del data_MR, data_CT, A_real, B_real, A_fake, B_fake
    
    
            print('[%d/%d]\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_B: %.4f'
                          % (epoch+1, num_epochs, Fool_disc_loss_A2B, Fool_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                              Id_loss_A2B, Disc_loss_A.item(), Disc_loss_B.item()))
            
    
        FDL_A2B_t.append(sum(FDL_A2B)/len(FDL_A2B))
        FDL_B2A_t.append(sum(FDL_B2A)/len(FDL_B2A))
        CL_A_t.append(sum(CL_A)/len(CL_A))
        CL_B_t.append(sum(CL_B)/len(CL_B))
        ID_B2A_t.append(sum(ID_B2A)/len(ID_B2A))
        ID_A2B_t.append(sum(ID_A2B)/len(ID_A2B))
        disc_A_t.append(sum(disc_A)/len(disc_A))
        disc_B_t.append(sum(disc_B)/len(disc_B))
    
        FDL_A2B = []
        FDL_B2A = []
        CL_A = []
        CL_B = []
        ID_B2A = []
        ID_A2B = []
        disc_B = []
        disc_A = []
    
#        iters = 0             
#        save_models(G_A2B, G_B2A, D_A, D_B, name)
#        if (epoch % 5 == 0):
#            plot_images_test(dataloader_test_horses, dataloader_zebra_test)
#         #plot_all_images(4, dataloader_test_horses, dataloader_zebra_test)
#        return(FDL_A2B_t,FDL_B2A_t,CL_A_t,CL_B_t,ID_B2A_t,ID_A2B_t,disc_A_t,disc_B_t)