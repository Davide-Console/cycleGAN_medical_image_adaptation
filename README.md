# Domain Adaptation for Medical Images
## 2021/2022 NECSTLab Project
- ###  Davide Console ([@Davide-Console](https://github.com/Davide-Console)) <br> davide.console@mail.polimi.it
- ###  Laura Ginestretti ([@lauraginestretti](https://github.com/lauraginestretti)) <br> laura.ginetretti@mail.polimi.it

## Dependencies

This project requires the following libraries:

- NumPy
- Pandas
- Scikit-learn

To install all the requirements, execute on the command prompt:
```bash
pip install requirements.txt
```

## Dataset
The dataset must be positioned into the project folder. It must be organized into the following structure.
```
project
└── Data
    ├── Domain A DICOM images
    │   ├── image000.dcm
    │   ├── image001.dcm
    │   └── ...
    └── Domain B DICOM images
        ├── image000.dcm
        ├── image001.dcm
        └── ...

```

## Offline preprocessing
All the DICOM images go through the following steps:
- conversion to PNG images
- resized to 256x256 images
- gantry removal
- converted to 8-bit unsigned integers images

To perform this process, execute on the command prompt:
```bash
python data_preparation.py
```
This command will work with a dataset presented as specified above.
Otherwise, execute:
```bash
python data_preparation.py --path 'Data'
```
changing `--path` according to your needs.

## CycleGAN training

To train the models of the cycleGAN, execute:
```bash
python training.py
```
In alternative, you can execute:
```bash
python training.py --batch_size 8 --learning_rate 0.0002 --epochs 50 --discriminators_epochs 5 --lambda_gp 5 --test_split 0.1 --validation_split 0.2 --save_figs False --save_all False
```
All these parameters can be changed according to your needs.

- `--discriminators_epochs` refers to the number of epochs through which the discriminators are trained during each epoch (e.g., having `--epochs == 50` and `--discriminators_epochs == 5`, discriminators will be trained for 250 epochs).


- `--save_figs`: if set to `True`, saves some images for each epoch. After the training, you will have:
  - `batches` folder with comparison between generated images and real ones.
  - `images` folder with single images of generated and real images


- `--save_all`: if set to `True`, saves networks and decoders for each epoch. After the training, you will have:
  - `models` folder with best models and possibly models from each epoch
  - `optimizers` folder with the optimizers of the saved models

## CycleGan testing

To test the models of the cycleGAN for inference, execute:

```bash
python testing.py
```