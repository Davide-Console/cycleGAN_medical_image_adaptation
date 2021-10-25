# Packages
import torch


def Train(num_epochs, discriminator, optimizer_discriminator, generator, 
          optimizer_generator):
    for epoch in range(num_epochs):
        
        ### Data preparation
        
        # Loading real samples
        
        real_labels = torch.ones(batch_size, 1)
        
        # Loading generated samples
        latent_space_samples = torch.randn((batch_size, 100))
        fake_img = generator(latent_space_samples)
    
        fake_labels = torch.zeros(batch_size, 1)
        
        # Unifiy samples
        all_img = torch.cat((real_img, fake_img))
        all_labels = torch.cat((real_labels, 
                                        fake_labels))
    
    
        ### Training
        
        # Train Discriminator
        discriminator.zero_grad()
        
        output_discriminator = discriminator(all_img)
        
        loss_discriminator = loss_function(output_discriminator, 
                                           all_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # Train Generator
        latent_space_samples = torch.randn(batch_size, 100)
        generator.zero_grad()
        
        generated_img = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_img)
        
        # Loss
        loss_generator = loss_function(output_discriminator_generated, 
                                       real_labels)
        loss_generator.backward()
        optimizer_generator.step()
        
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")

