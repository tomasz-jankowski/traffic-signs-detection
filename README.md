# traffic-sign-detector
Deep learning university project to detect and classify traffic signs from given images.

[![Generic badge](https://img.shields.io/badge/python-3.7.7-blue.svg)](https://shields.io/)   [![Generic badge](https://img.shields.io/badge/anaconda-2019.10-green.svg)](https://shields.io/)   [![Generic badge](https://img.shields.io/badge/tensorflow-2.1.0-red.svg)](https://shields.io/)
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

## Usage

### program2.py
Artificial neural network architecture and training process.

### classify.py
Main script used to show neural network predictions on given images.

### object_detector.py
Main program of this project, uses trained model, sliding window and image pyramid techniques to find and classify traffic signs.

## Results
![znaki1_007](https://user-images.githubusercontent.com/49961031/110140990-f9a7e200-7dd4-11eb-875a-e02ae2925d74.jpg)
![znaki4_002](https://user-images.githubusercontent.com/49961031/110141036-07f5fe00-7dd5-11eb-94b0-13c8d261e6d0.jpg)
![znaki5_002](https://user-images.githubusercontent.com/49961031/110141042-09272b00-7dd5-11eb-83ec-4daaa0479e63.jpg)

## Anaconda packages being used
* numpy 1.18.1
* matplotlib 3.1.3
* opencv 3.4.2
* tensorflow 2.1.0
* tqdm 4.46.0
* tabulate 0.8.3

## Credits
The project has been developed by:
- [Michał Kliczkowski](https://github.com/michal090497)
- [Tomasz Jankowski](https://github.com/tomasz-jankowski)

## License
 
The MIT License (MIT)

Copyright (c) 2021 Michał Kliczkowski Tomasz Jankowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
