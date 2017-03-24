=====================================
LineSfM
=====================================

------------
Introduction
------------

This code is mainly linked with the following publication, which was subject to peer review: 

"Robust and Accurate Line- and/or Point-Based Pose Estimation without Manhattan Assumptions", Yohann Salaun, Renaud Marlet, and Pascal Monasse, ECCV 2016

It computes the calibration between two cameras using both lines and points. It gives performant results in textured scenes and in indoor scenes.

------------
Notes
------------

This code has been tested on linux 64 bits. It might not work well on other os. If you have tested it on Windows, I would be glad to know if it works.

------------
Building
------------

Requirement:

- CMake (available for free at https://cmake.org/download/)

- openCV (available for free at http://opencv.org/) > 2.x

- Eigen library > 3

- Blas library

- Lapack library

- Ceres library (http://ceres-solver.org/installation.html)

- Arpack++ with SuperLU (https://github.com/m-reuter/arpackpp)

- C++ compiler

I am working on simplifying the required libraries, do not hesitate to contact me if you have any issue or any solution for simplification.

------------
Use
------------

Look at the folder TEST_IMAGES for some information on how the code works

------------
License
------------

MPL2 for the main code

Different for third party

------------
Authors
------------

Yohann Salaun <yohann.salaun AT imagine.enpc.fr>

Renaud Marlet <renaud.marlet AT enpc.fr>

Pascal Monasse <pascal.monasse AT enpc.fr>

------------
Citations
------------

If you used our code in your publication, please cite the following:

- Line detection

[1] Yohann Salaun, Renaud Marlet, and Pascal Monasse, [Multiscale line segment detector for robust and accurate SfM](https://drive.google.com/file/d/0B96kyL2SBsmzOFY0b2hnSm54eTQ/view),  ICPR 2016

[2] Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall, [LSD: a Line Segment Detector](http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd), Image Processing On Line, 2 (2012), pp. 35–55.

- Line Matching

[3] Lilian Zhang, and Reinhard Koch, [An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency](http://www.sciencedirect.com/science/article/pii/S1047320313000874), JVCI 2013

- Vanishing point detection

[4] Almansa, A. and Desolneux, A. and Vamech, S., [Vanishing point detection without any a priori information](http://desolneux.perso.math.cnrs.fr/papers/ADV_vpoint_03.pdf), TPAMI 2003

- Calibration

[5] Yohann Salaun, Renaud Marlet, and Pascal Monasse, [Robust and Accurate Line- and/or Point-Based Pose Estimation without Manhattan Assumptions](https://drive.google.com/file/d/0B96kyL2SBsmzWDc4ekd2VElKY0U/view), ECCV 2016

[6] Yohann Salaun, Renaud Marlet, and Pascal Monasse, [Robust SfM with Little Image Overlap](https://arxiv.org/pdf/1703.07957.pdf), arXiv.org 2017



