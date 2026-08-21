# Experiment 4: Transfer Learning on CIFAR-10

CS3807 Deep Learning Lab. This covers Tasks 1 to 5, the hyperparameter study, and the additional exercises (VGG16, ResNet50, AlexNet from scratch).

## What's in this project

- `Experiment_4_Transfer_Learning.ipynb`, the actual experiment. Loads CIFAR-10, builds a MobileNetV2 transfer learning model, trains it, fine-tunes it, evaluates it, runs the hyperparameter study, then repeats the pipeline with VGG16 and ResNet50 and trains an AlexNet from scratch for comparison.
- `report_bundle/` (`Experiment_4_Report.tex` + `figures/`), the write-up built from an actual run. Compiles to the submission PDF.
- `Exp4_Study_Notes.md`, the plain-English notes on what's actually going on, for understanding the material rather than just having working code.

## Running the notebook

Needs a GPU. In Colab: Runtime menu, change runtime type, GPU.

The Setup cell checks GPU availability and sets up the plot styling. Times New Roman isn't installed in Colab by default, so there's a commented out font install line right there. Uncomment it, run it once, and if plots were already generated earlier in the session, force a font rescan instead of restarting (restarting wipes the trained model out of memory):

```
!apt-get install -y msttcorefonts -qq > /dev/null
!rm -rf ~/.cache/matplotlib
```

then, in a fresh cell, without restarting:

```
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
```

After that, run the notebook top to bottom. Two things take a while:

- the Hyperparameter Study cell trains 8 separate models
- the Additional Exercises section trains VGG16, ResNet50, and an AlexNet from scratch, each with its own training run

Everything downloads its own pretrained weights the first time it's called (MobileNetV2, VGG16, ResNet50), so the first run of each will pause briefly to fetch them.

At the end, zip and download the figures:

```
!zip -r figures.zip figures/
from google.colab import files
files.download('figures.zip')
```

