====
YAML
====

Finally, all of the customized grid-objects and functions defined in the last
pages can be used in conjunction with the yaml configuration files to make the
creation of custom environments extremely simple.  All that is requires is that
a custom module is created and contains all the custom code (grid-objects and
functions), that all custom components are appropriately registered, and that
the module name is used in the yaml when indicating the grid-objects and
functions to be used.

Practical Example
-----------------

.. note::

  The examples shown here can be found in the ``examples/`` folder.

We are going to create a custom environment where the agent needs to collect
all the coins scattered around.  For this purpose, we will first define a new
custom module ``coin_env.py`` which contains all the necessary components:  A new
``Coin`` grid-object, and appropriate reset, transition, reward, and
terminating functions.

.. literalinclude:: /../examples/coin_env.py
  :language: python

Next, we are going to create a YAML configuration file ``coin_env.yaml`` which
combines these new custom components with some predefined ones.  To use the
customly defined components, we just need to prepend their names with the name
of the modules where they are found.

.. literalinclude:: /../examples/coin_env.yaml
  :language: yaml

.. important::
  
  For the custom module to be useable in the YAML configuration, you'll need to
  make sure that its directory is in the ``PYTHONPATH``, and is therefore
  findable by the python interpreter.

.. note::

  This example can be run with the ``gv_viewer.py`` script::

    cd <path/to/examples/folder>
    PYTHONPATH="$PYTHONPATH:$PWD" gv_viewer.py coin_env.yaml
