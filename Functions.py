import torch

def Train(num_epochs, discriminator, optimizer_discriminator, generator, 
          optimizer_generator):
    for epoch in range(num_epochs):
        
        ## Loading real samples
        # foo for loading batch
        
        real_labels = torch.ones(batch_size, 1)
        
        ## Loading generated samples
        latent_space_samples = torch.randn((batch_size, 100))
        generated_samples = generator(latent_space_samples)
    
        fake_labels = torch.zeros(batch_size, 1)
        
        ## Unifiy samples
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_labels, 
                                        generated_samples_labels))
    
    
        ## Training
        
        # Train Discriminator
        discriminator.zero_grad()
        
        output_discriminator = discriminator(all_samples)
        
        loss_discriminator = loss_function(output_discriminator, 
                                           all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # Train Generator
        latent_space_samples = torch.randn(batch_size, 100)
        generator.zero_grad()
        
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        
        loss_generator = loss_function(output_discriminator_generated, 
                                       real_labels)
        loss_generator.backward()
        optimizer_generator.step()
        
        # Show loss
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")

