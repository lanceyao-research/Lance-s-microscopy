# Lance's Microscopy

A lightweight electron microscopy simulator with a fake FEI AutoScript-compatible API, designed for developing and testing automation, AI, and ML algorithms without requiring access to real microscope hardware.

## Overview

This project provides:

- **A simulated TEM/STEM microscope** with realistic stage movement, sample geometry (TEM grid with holey carbon film), and configurable noise models
- **A TCP server** that accepts commands to control the virtual microscope
- **An AutoScript API doppelganger** (`autoscript_tem_microscope_client_doppelganger.py`) that mimics the real FEI/Thermo Fisher AutoScript Python API

This enables researchers and developers to:

- Prototype automation scripts before running them on expensive instruments
- Generate synthetic training data for RL/AI models
- Test real-time image processing pipelines with controllable noise parameters
- Develop stage navigation algorithms in a safe sandbox environment

## Features

- **Dual imaging modes**: TEM (parallel beam) and STEM (scanning) with realistic line-by-line acquisition
- **Realistic sample geometry**: Simulated TEM grid with square mesh, holey carbon support film, and random low-poly "sample" features
- **Configurable noise models**:
  - Poisson noise (shot noise from electron counting)
  - Gaussian noise (readout noise)
  - Hot pixels (salt noise)
- **Stage controls**: Absolute/relative moves, continuous jogging (drift), velocity control
- **Zoom/magnification**: Discrete magnification steps from 1Kx to 1Mx
- **Live GUI**: Real-time visualization with adjustable FPS, noise parameters, and display normalization
- **Persistent configuration**: Settings saved to JSON and restored on restart

## Installation

### Dependencies

```bash
pip install numpy scikit-image matplotlib
```

Tkinter is included with most Python distributions. If not available:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (with Homebrew Python)
brew install python-tk
```

## Usage

### Starting the Simulator

```bash
python LancesMicroscopy.py
```

This launches the GUI and starts a TCP server on `127.0.0.1:9999`.

### Connecting via the AutoScript Doppelganger

```python
from autoscript_tem_microscope_client_doppelganger import TemMicroscopeClient, StagePosition

# Connect to the simulator
microscope = TemMicroscopeClient()
microscope.connect()

# Move stage to an absolute position (SI units: meters)
microscope.specimen.stage.absolute_move(StagePosition(x=100e-6, y=-50e-6))

# Relative move
microscope.specimen.stage.relative_move(StagePosition(x=10e-6))

# Get current position
pos = microscope.specimen.stage.position
print(f"Stage at x={pos.x}, y={pos.y}")

# Change magnification
available_mags = microscope.optics.magnification.available_values
microscope.optics.magnification.value = available_mags[10]  # Pick from list

# Start continuous drift (jogging)
from autoscript_tem_microscope_client_doppelganger import StageVelocity
microscope.specimen.stage.start_jogging(StageVelocity(x=1e-6))  # 1 µm/s in x
microscope.specimen.stage.stop_jogging()
```

### GUI Controls

- **Mode**: Switch between TEM and STEM imaging
- **FPS Limit**: Control frame rate (STEM scan speed)
- **Poisson k**: Higher values = more signal, less shot noise
- **Gaussian σ**: Readout noise standard deviation
- **Low/High %**: Percentile-based display normalization
- **Arrow keys/buttons**: Navigate the sample
- **Zoom In/Out**: Change magnification

## API Compatibility

The doppelganger API mirrors key parts of the real `autoscript_tem_microscope_client`:

| Real AutoScript | Doppelganger | Status |
|-----------------|--------------|--------|
| `TemMicroscopeClient` | ✅ | Implemented |
| `specimen.stage.absolute_move()` | ✅ | Implemented |
| `specimen.stage.relative_move()` | ✅ | Implemented |
| `specimen.stage.position` | ✅ | Implemented |
| `specimen.stage.start_jogging()` | ✅ | Implemented |
| `specimen.stage.stop_jogging()` | ✅ | Implemented |
| `optics.magnification.value` | ✅ | Implemented |
| `optics.magnification.available_values` | ✅ | Implemented |
| `optics.magnification.increase()` | ✅ | Implemented |
| Image acquisition | ❌ | Not yet implemented |

## Configuration

Settings are persisted in `microscopy_config.json`:

```json
{
  "mode": "TEM",
  "fps": 30.0,
  "poisson_k": 100.0,
  "gaussian_std": 0.02,
  "percentile_low": 1.0,
  "percentile_high": 99.0
}
```

## License

MIT License - feel free to use and modify for your research and development needs.

## Contributing

Contributions welcome! Some areas that could use work:

- Image acquisition API (`microscope.imaging.acquire_image()`)
- Beam controls (brightness, focus simulation)
- Detector simulation
- More realistic sample geometries
- Unit tests
