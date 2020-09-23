CONTRE
======
**CONT** inuum **RE** weighting.

This is the documentation for ``CONTRE``,
a package to be used in combination with the software framework of the Belle II experiment ``basf2``.
It can be used to improve the Continuum Monte Carlo (MC) samples, using Continuum Reweighting.

Continuum Reweighting is a data-driven, event-wise method to reweight the Continuum MC samples.
A classifier can be used to distinguish between off-resonance MC and off-resonance data.
Event weights can be calculated from the classifier output to improve the on-resonance MC samples.

More information about the method can be found in this note_ (access for Belle II members, only).

.. _note: https://docs.belle2.org/record/1978?ln=en 



.. attention::
    The ``CONTRE`` package, and this documentation are still under construction.
    I appreciate comments and suggestions.

Features
--------

``CONTRE`` includes the following features:

- Reweighting of arbitrary ntuple files using off-resonance ntuple with the same selection
- Changing Parameters (trainings- and test size of the samples, BDT training parameters)
- Example: Jupyter notebook with detailed explanation (see ``example/``)
- Systematic uncertainties via bootstrapping

Modules
-------

The package contains the following modules

.. toctree::
    modules




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
