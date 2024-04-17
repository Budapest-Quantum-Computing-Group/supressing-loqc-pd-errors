# Suppressing photon detection errors in LOQC

This project aims to simulate and optimize the six mode KLM controlled sign flip gate to achieve higher output fidelity for preparing the state
$$\frac{1}{2}(\ket{0, 1, 0, 1} + \ket{1, 0, 0, 1}) + \ket{0, 1, 1, 0} - \ket{1, 0, 1, 0}$$
from the dual-rail encoded $\ket{+}\ket{+}$ states with imperfect detectors

## Training and generating plots
Makes sure to install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

You can execute the optimizations by running the `trainer.py` script.

Example
```bash
python trainer.py --output-dir ./results --starting-s-star 0.0650 --ending-s-star 0.0850 --enhanced --use-jit
```
This script will save the generated losses and weights in the specified output directory.

The plots can be generated using the script `make-plots.py`.

Example
```bash
python make_plots.py --input-dir ./results
```
