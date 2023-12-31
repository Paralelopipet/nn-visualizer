# nn-visualizer
Project for visualisation of pytorch training process using arduino + MAX7219 8x32 LED  matrix.

![Alt NN-Visualizer](imgs/gaf.gif)
### Installation

This package can be installed in any pytorch project by cloning the repository and running the following command:
```bash
pip install -r requirements.txt
```

### Usage

In order to use this amazing nn training visualizes tool you need to follow next steps:|

Import the class in following fashion
```python
from main import Arduino_Visualizer
```
Make an instance by providing your model, baudrate (prefeferably 9600) and COM port - whichever the amazing device is connected to:

```python
arduino_visualizer = Arduino_Visualizer(list(model.parameters()), baudrate=9600, port='COM3')
```
Finally add the following line in your training process - recommended to be added after a number of batches and not on every call because of potential performance impacts.

```python
arduino_visualizer.show(model)
```

Example is in mnist_train.py. Enjoy luck owner of the nn-visualiser! :D


