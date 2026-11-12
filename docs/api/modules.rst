Modules
=======

MLP
---

.. automodule:: rsl_rl.modules.mlp
   :members:
   :undoc-members:


RNN
---

.. automodule:: rsl_rl.modules.rnn
   :members:
   :undoc-members:


CNN
---

.. automodule:: rsl_rl.modules.cnn
   :members:
   :undoc-members:


Normalization
-------------

.. automodule:: rsl_rl.modules.normalization
   :members:
   :undoc-members:


Distribution
------------

Distribution updates promote float16 and bfloat16 network outputs to float32 before forming probability parameters.
Samples and probabilistic statistics then use at least float32 precision, while float64 inputs retain their precision.
Deterministic output and export paths retain the network output dtype.

.. automodule:: rsl_rl.modules.distribution
   :members:
   :undoc-members:

