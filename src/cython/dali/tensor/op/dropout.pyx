cpdef Tensor dropout(Tensor t, double drop_prob, bint normalize=True):
    """
    dropout(t, drop_prob, normalize=True)

    Multiply tensor t by bernoulli noise with probability `drop_prob`
    of being 0. If normalize is true then the product of the noise and
    the tensor is then divided by `drop_prob` so that the expected value
    of the output equals that of the input.

    Parameters
    ----------
    t : an N-Dimensional Tensor
    drop_prob : a continuous value between 0 and 1
    normalize : optional, a boolean controlling whether normalization
                should be applied. Defaults to True.

    Returns
    -------

    out : Tensor
        the tensor t with noise applied to it, post-normalization.

    See Also
    --------
    dropout_unnormalized : dropout with normalization off by default.
    fast_dropout : gaussian noise applied to a tensor.

    Apply bernoulli noise to a tensor (e.g. to regularize a
    neural network). Randomly sets elements in the input
    tensor to 0.

    When normalized:
    ----------------
    When normalized the non-zeroed out elements
    are scaled up by `alpha = 1.0 / (1 - drop_prob)` to preserve
    the same distributional statistics.

    When unnormalized:
    ------------------
    Elements that were not dropped are kept the same.
    In this approach the network is trained with a noise distribution
    during training, and typically during inference (test time)
    dropout is switched off (drop_prob = 0), and the units are
    multiplied by `alpha = 1 - drop_prob` to recover the same
    distributional statistics. Since this is not automatic, the
    user must correct for this change themselves.

    Paper Abstract
    --------------
    Deep neural nets with a large number of parameters are
    very powerful machine learning systems. However, overfitting
    is a serious problem in such networks. Large networks are
    also slow to use, making it difficult to deal with overfitting
    by combining the predictions of many different large neural
    nets at test time. Dropout is a technique for addressing
    this problem. The key idea is to randomly drop units (along
    with their connections) from the neural network during training.
    This prevents units from co-adapting too much. During training,
    dropout samples from an exponential number of different
    “thinned” networks. At test time, it is easy to approximate
    the effect of averaging the predictions of all these thinned
    networks by simply using a single unthinned network that has
    smaller weights. This significantly reduces overfitting and
    gives major improvements over other regularization methods.

    - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
      Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way
      to Prevent Neural Networks from Overfitting," JMLR 2014
    """
    if normalize:
        return Tensor.wrapc(c_dropout(t.o, drop_prob))
    else:
        return Tensor.wrapc(c_dropout_unnormalized(t.o, drop_prob))

cpdef Tensor dropout_unnormalized(Tensor t, double drop_prob, bint normalize=False):
    """
    dropout_unnormalized(t, drop_prob, normalize=False)

    Multiply tensor t by bernoulli noise with probability `drop_prob`
    of being 0. If normalize is true then the product of the noise and
    the tensor is then divided by `drop_prob` so that the expected value
    of the output equals that of the input.

    Parameters
    ----------
    t : an N-Dimensional Tensor
    drop_prob : a continuous value between 0 and 1
    normalize : optional, a boolean controlling whether normalization
                should be applied. Defaults to False.

    Returns
    -------

    out : Tensor
        the tensor t with noise applied to it, post-normalization.

    See Also
    --------
    dropout : dropout with normalization by default.
    fast_dropout : gaussian noise applied to a tensor.

    Apply bernoulli noise to a tensor (e.g. to regularize a
    neural network). Randomly sets elements in the input
    tensor to 0.

    When normalized:
    ----------------
    When normalized the non-zeroed out elements
    are scaled up by `alpha = 1.0 / (1 - drop_prob)` to preserve
    the same distributional statistics.

    When unnormalized:
    ------------------
    Elements that were not dropped are kept the same.
    In this approach the network is trained with a noise distribution
    during training, and typically during inference (test time)
    dropout is switched off (drop_prob = 0), and the units are
    multiplied by `alpha = 1 - drop_prob` to recover the same
    distributional statistics. Since this is not automatic, the
    user must correct for this change themselves.

    Paper Abstract
    --------------
    Deep neural nets with a large number of parameters are
    very powerful machine learning systems. However, overfitting
    is a serious problem in such networks. Large networks are
    also slow to use, making it difficult to deal with overfitting
    by combining the predictions of many different large neural
    nets at test time. Dropout is a technique for addressing
    this problem. The key idea is to randomly drop units (along
    with their connections) from the neural network during training.
    This prevents units from co-adapting too much. During training,
    dropout samples from an exponential number of different
    “thinned” networks. At test time, it is easy to approximate
    the effect of averaging the predictions of all these thinned
    networks by simply using a single unthinned network that has
    smaller weights. This significantly reduces overfitting and
    gives major improvements over other regularization methods.

    - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
      Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way
      to Prevent Neural Networks from Overfitting," JMLR 2014
    """
    if normalize:
        return Tensor.wrapc(c_dropout(t.o, drop_prob))
    else:
        return Tensor.wrapc(c_dropout_unnormalized(t.o, drop_prob))

cpdef Tensor fast_dropout(Tensor t):
    """
    fast_dropout(t)

    Apply Gaussian Noise a standard deviation of 1 and a
    mean of 1 to a tensor (e.g. to regularize it)

    Parameters
    ----------
    t : an N-Dimensional Tensor

    Returns
    -------

    out : Tensor
        the tensor t with noise applied to it

    See Also
    --------
    dropout : dropout with normalization by default.
    dropout_unnormalized : dropout with normalization off by default.

    Paper Abstract
    --------------
    Preventing feature co-adaptation by encouraging independent
    contributions from different features often improves
    classification and regression performance. Dropout training
    (Hinton et al., 2012) does this by randomly dropping out
    (zeroing) hidden units and input features during training
    of neural networks. However, repeatedly sampling a random
    subset of input features makes training much slower. Based
    on an examination of the implied objective function of dropout
    training, we show how to do fast dropout training by sampling
    from or integrating a Gaussian approximation, instead of
    doing Monte Carlo optimization of this objective. This
    approximation, justified by the central limit theorem and
    empirical evidence, gives an order of magnitude speedup and
    more stability.
    - Sida I. Wang, Christopher D. Manning, "Fast dropout training",
      ICML 2013

    Relevant reading:
    -----------------
    https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc
    """
    return Tensor.wrapc(c_fast_dropout(t.o))

