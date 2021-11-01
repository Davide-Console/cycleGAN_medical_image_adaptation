# Packages
import torch

# Loss Functions
def LSGAN_D(real, fake):
  return (torch.mean((real - 1)**2) + torch.mean(fake**2))

def LSGAN_G(fake):
  return  torch.mean((fake - 1)**2)

# Train

def Train(num_epochs, bs, G_A2B, G_B2A,optimizer_G_A2B, optimizer_G_B2A, D_A, D_B, 
          optimizer_D_A, optimizer_D_B, dataloader_train_CT, 
          dataloader_train_MR, criterion_Im, old = True):
    
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
    
        # For each batch in the dataloader
        for  i,(data_CT, data_MR) in enumerate(zip(dataloader_train_CT, dataloader_train_MR),0):
        
            # Set model input
            A_real = data_CT[0]
            B_real = data_MR[0]
          
            #tensor_ones = torch.ones([A_real.shape[0],1,14,14])
            #tensor_zeros = torch.zeros([A_real.shape[0],1,14,14])
    
            # Genrated images
            B_fake = G_A2B(A_real)
            A_rec = G_B2A(B_fake)
            A_fake = G_B2A(B_real)
            B_rec = G_A2B(A_fake)
    
            
            # Discriminator A
            optimizer_D_A.zero_grad()
            if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
              rand_int = torch.randint(5, old_A_fake.shape[0]-1)
              Disc_loss_A = LSGAN_D(D_A(A_real), D_A(old_A_fake[rand_int-5:rand_int].detach()))
              D_A_losses.append(Disc_loss_A.item())
            else:
              Disc_loss_A = LSGAN_D(D_A(A_real), D_A(A_fake.detach()))
              D_A_losses.append(Disc_loss_A.item())
            
            Disc_loss_A.backward()
            optimizer_D_A.step()
    
            
            # Discriminator B
    
            optimizer_D_B.zero_grad()
            if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
              rand_int = torch.randint(5, old_B_fake.shape[0]-1)
              Disc_loss_B =  LSGAN_D(D_B(B_real), D_B(old_B_fake[rand_int-5:rand_int].detach()))
              D_B_losses.append(Disc_loss_B.item())
            else:
              Disc_loss_B =  LSGAN_D(D_B(B_real), D_B(B_fake.detach()))
              D_B_losses.append(Disc_loss_B.item())
    
            Disc_loss_B.backward()
            optimizer_D_B.step()   
    
            # Generator
    
            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()
    
            # Fool discriminator
            Fool_disc_loss_A2B = LSGAN_G(D_B(B_fake))
            Fool_disc_loss_B2A = LSGAN_G(D_A(A_fake))
    
            # Cycle Consistency    both use the two generators
            Cycle_loss_A = criterion_Im(A_rec, A_real)*5
            Cycle_loss_B = criterion_Im(B_rec, B_real)*5
    
            # Identity loss
            Id_loss_B2A = criterion_Im(G_B2A(A_real), A_real)*10
            Id_loss_A2B = criterion_Im(G_A2B(B_real), B_real)*10
    
            # generator losses
            Loss_G = Fool_disc_loss_A2B+Fool_disc_loss_B2A+Cycle_loss_A+Cycle_loss_B+Id_loss_B2A+Id_loss_A2B
            G_losses.append(Loss_G)
    
            # Backward propagation
            Loss_G.backward()
            
            
            # Optimization step
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()
    
            FDL_A2B.append(Fool_disc_loss_A2B)
            FDL_B2A.append(Fool_disc_loss_B2A)
            CL_A.append(Cycle_loss_A)
            CL_B.append(Cycle_loss_B)
            ID_B2A.append(Id_loss_B2A)
            ID_A2B.append(Id_loss_A2B)
            disc_A.append(Disc_loss_A)
            disc_B.append(Disc_loss_B)
    
            if(iters == 0 and epoch == 0):
              old_B_fake = B_fake.clone()
              old_A_fake = A_fake.clone()
            elif (old_B_fake.shape[0] == bs*5 and B_fake.shape[0]==bs):
              rand_int = random.randint(5, 24)
              old_B_fake[rand_int-5:rand_int] = B_fake.clone()
              old_A_fake[rand_int-5:rand_int] = A_fake.clone()
            elif(old_B_fake.shape[0]< 25):
              old_B_fake = torch.cat((B_fake.clone(),old_B_fake))
              old_A_fake = torch.cat((A_fake.clone(),old_A_fake))
    
            iters += 1
            del data_MR, data_CT, A_real, B_real, A_fake, B_fake
    
    
            if iters % 50 == 0:
          
              print('[%d/%d]\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_A: %.4f'
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