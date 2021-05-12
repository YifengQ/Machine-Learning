#!/usr/bin/env python3

import compress as c
import os


data = c.load_data('Data/Train/')
c.compress_images(data,10)
os.rename('Output','OutputK10')

data = c.load_data('Data/Train/')
c.compress_images(data,100)
os.rename('Output','OutputK100')

data = c.load_data('Data/Train/')
c.compress_images(data,500)
os.rename('Output','OutputK500')

data = c.load_data('Data/Train/')
c.compress_images(data,1000)
os.rename('Output','OutputK1000')

data = c.load_data('Data/Train/')
c.compress_images(data,2000)
os.rename('Output','OutputK2000')
