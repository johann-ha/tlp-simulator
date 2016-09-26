# -*- coding: utf-8 -*-
""" 
    A short interface for exploring TLP Diffusion paths simulations
    parameter space.
"""
import numpy as np
from SimTLP_DiffusionPaths import ModelFun
from fipy.viewers.tsvViewer import TSVViewer

# define the parameters space (dgf, ebb, exb, exx)
t_max = 1800.
param_space = np.ogrid[-70000:-70000:1j, 0.0:0.0:1j, -5:5:5j, 0:75:4j, 0.0:0.0:1j, 0.0:0.0:1j, 0.0:0.0:1j]

# loop over the parameter space to explore it
loop_index = 0

for dgf, ebb, exb, exx, bbb, bxb, bxx in np.nditer(param_space):
    print "dgf: %.3f, ebb: %.3f, exb: %.3f, exx: %.3f, bbb: %.3f, bxb: %.3f, bxx: %.3f" % (dgf, ebb, exb, exx, bbb, bxb, bxx)
    # calculate the diffusion field
    res = ModelFun(t_max, dgf, ebb, exb, exx, bbb, bxb, bxx)
    # save the data to a file
    fname = "test_new_" + str(loop_index) + ".tsv"
    title = "dgf: %.3f, ebb: %.3f, exb: %.3f, exx: %.3f, bbb: %.3f, bxb: %.3f, bxx: %.3f" % (dgf, ebb, exb, exx, bbb, bxb, bxx)
    TSVViewer(vars=res, title=title).plot(filename=fname)
    loop_index += 1
