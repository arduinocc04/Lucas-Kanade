This project depends on `Eigen` and `openCV`.
`Eigen` is included.
`openCV` is used only for debug. If you use `DO_NOT_TEST=ON` or `DO_NOT_USE_OPENCV=ON` flags while configure cmake, you don't need to install `openCV`.

# Acknowledgement
I wrote these code while working at @viuron. This repository is public version.  
Thanks to @happie827(JunHyok Kong) for helps.

# Documentation
I used `doxygen`.  
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
We need to access value in eigen with (row, column) but in calculating image we need to use (column, row, 1). 

# References
1. https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
