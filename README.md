# Eye tracking based on EOG signal processing

Instituto de Ingeniería Biomédica (IIBM)

Universidad de Buenos Aires

2020

---------------------------------------------

This repository contains the software for a BCI designed to control a virtual d-pad using EOG signals as input. It includes Arduino firmware, a legacy MATLAB interface, and a Python application and trainer.

## Arduino

Software to control an IC ADS1299 through an Arduino NANO. It handles the startup sequence (configuration, setup) of the device and triggers the data conversion. The processing loop reads data continuously, processes it and sends it to a computer through SPI.

See the [ADS1299 configuration and control digest](docs/ads1299/README.md) for the datasheet-derived command, register, timing, and sample-decoding reference.

## Python application

The simple signal recorder is implemented in `eog_app/`. Install
[uv](https://docs.astral.sh/uv/) and run these commands from the repository
root:

```bash
uv sync
uv run eog-bci
```

`uv sync` creates the project environment from `pyproject.toml` and the
committed `uv.lock`. To run the tests, use `uv run pytest`.

The serial-port field defaults to `/dev/ttyUSB0` on Linux. Choose the ADS1299
sample rate (250, 500, or 1000 SPS), channel gain (1, 2, 4, 6, 8, 12, or
24), and a recording directory. **Connect, configure & record** sends the
selected settings to the Arduino and starts acquisition after it acknowledges
the configuration. The Arduino streams input-referred differential voltage in
millivolts; the window plots both channels in microvolts and saves the device
millivolt values alongside the converted microvolt values. The
default output directory is `EOG_data/` on the local drive; choose another
directory if you want recordings stored on a particular SSD or mounted drive.

The configuration handshake requires the updated Arduino sketch in
`arduino/ads1299/ads1299.ino` to be flashed to the board. The serial link
runs at 500000 baud (exact on a 16 MHz Arduino); the firmware, Python app and
MATLAB controller must all use the same rate. 1000 SPS is the highest rate
offered by the recorder for two channels.
The legacy MATLAB controller's one-byte `1` command remains supported and
starts acquisition with the original 250 SPS, gain-24 defaults.

The previous calibration/live BCI interface is still available separately:

```bash
uv run python -m eog_app.app
```

That interface retains the BCI prompts, training workflow, audio cues, and
separate raw-signal viewer.

The legacy-compatible PyTorch trainer is part of `eog_app`:

```bash
uv run python -m eog_app.trainer SESSION_ID
```

It reads `Pytorch/input_data/input_SESSION_ID_filled.txt` and
`output_SESSION_ID_filled.txt` by default, and writes predictions and trained
weight matrices to `Pytorch/output/`. Optional `--data-dir`, `--output-dir`,
and `--epochs` arguments allow using other locations or training lengths.

## Matlab (legacy)

The original MATLAB implementation is retained for reference. The MATLAB
script reads data transmitted by the Arduino and executes the complete test:

1. Calibration stage
    - Presents the calibration interface.
    - Records eog data from vertical & horizontal channel and stores them into a file.
    - Processes the data to extract the features to train the NN (input & desired output).

2. Training of the NN
    - Calls the Python trainer in `eog_app/trainer.py`.

3. Live test
    - Presents the testing interface, which reads and processes data continuously, online.

## Training data and outputs

`Pytorch/input_data/` stores training input and target files, and
`Pytorch/output/` stores predictions and the trained weight and bias matrices.
The 2-8-2 multilayer perceptron trainer is maintained in `eog_app/trainer.py`;
see the Python application section for its command and options.
