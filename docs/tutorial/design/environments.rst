Environments
============

In GV, there is an explicit distinction between `inner` and `outer`
environments.  Both types provide similar interfaces, with some key
differences:  primarily in the data formats returned by the respective methods.
`Inner` environments return data in the form of python objects, while `outer`
environments return data in a raw numeric form.  Additionally, `inner`
environments also provide functional interfaces to some of their methods, which
tend to be more useful for planning methods.  To bridge `inner` and `outer`
environments, a conversion method from python objects to raw numeric data is
needed;  we call the classes responsible for this conversion `representations`.

.. note::
  A typical RL agent will likely want to interact with an `outer` environment,
  since that is the one which provides states and observations as raw numeric
  data, suitable to be processed using neural network models.  However, other
  forms of control methods (e.g. certain forms of planning) may not need the
  data to be in a numeric format, and may want to interact with the `inner`
  environment directly.

.. toctree::
  :hidden:
  :maxdepth: 2

  environments_inner
  environments_representations
  environments_outer
  environments_gym
