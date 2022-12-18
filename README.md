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
The dataset must be organized into the following folders.
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

## CycleGan testing

To test the models of the cycleGAN for inference, execute:

```bash
python testing.py
```