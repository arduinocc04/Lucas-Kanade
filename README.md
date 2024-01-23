This project depends on `Eigen` and `openCV`.  
`Eigen` is included.  
`openCV` is used only for debug. If you use `DO_NOT_TEST=ON` or `DO_NOT_USE_OPENCV=ON` flags while configuring cmake, you don't need to install `openCV`.

# Acknowledgement
I wrote these codes while working at [viuron](https://github.com/viuron/). This repository is public version of the codes.  
Thanks to [JunHyok Kong](https://github.com/happie827) for helps while debugging and understanding Lucas-Kanade.

# Documentation
used `doxygen`.  
To generate documentation, go to `doc` and
```
doxygen doxy.conf
```

# Coordinate System
```
0/0---X--->
 |
 |
 Y
 |
 |
 v
```
(x, y, 1)
## Attention
The way accessing value in eigen is `(row, column)` but in calculating image, the way is `(column, row, 1)`. 

# References
1. https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
