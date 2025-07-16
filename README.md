# Odyssey
![Odyssey](Odyssey.png)
*"Cast off these garments, and leave the raft to drift before the winds; but swim with thy hands and strive to reach the coast of the Phaiakians."*



The original implementation of Odyssey 1.0, Anthrogen's protein language model.  Odyssey 1.0 features:
- Consensus transformer blocks
- Finite scalar quantization
- 14-atom structure prediction
- Discrete diffusion transformer trunk training

# Setup 
To install all dependencies, and to initialize `odyssey` as a pip package, run:
```
cd /path/to/Odyssey
pip install -e .
```

# Training Odyssey
Create a training configuration file following the directions of Odyssey/configs/configuration_constructor.md.

Next, execute your experiment:
```
cd odyssey/train/
python train.py --config path/to/config.yaml
```
If your config file is stored in `Odyssey/configs/`, then your command line argument may look like `../../configs/config.yaml`.


The model checkpoints and training history will be stored to the `checkpoints_path` directory (as specified in the configuration YAML script).
