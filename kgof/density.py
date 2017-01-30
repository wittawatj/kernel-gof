"""
Module containing implementations of some unnormalized probability density 
functions.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kgof.config as config
#import kgof.util as util
import numpy as np
import scipy.stats as stats
import tensorflow as tf


class UnnormalizedDensity(object):
    """
    An abstract class for an unnormalized differentiable density function.
    Subclasses should call __init__() of this base class.

    - Subclasses should implement build_graph() with a singleton pattern.
    This method constructs a TensorFlow graph only once and stores all 
    TensorFlow variables in tf_vars (a class variable). build_graph() will likely
    call tf_log_den().

    - Subclasses should have tf_vars as their class variables.
    - Subclasses should have an instance variable called "params". A dictionary:
        parameter name |-> parameter value. The parameter name keys must match 
        that of the TensorFlow variables in tf_parameters (class variable).
    - Subclasses must implement tf_params() return a dictionary with parameter 
        names as the keys.
    - The constructor of subclasses msut take in all the parameters as keyword
      arguments.
    """
    __metaclass__ = ABCMeta

    # TensorFlow variables for this class. Need to build them only once.
    # A dictionary: variable name |-> TensorFlow variable.
    tf_vars = None
    tf_parameters = None

    @classmethod
    def build_graph(cls):
        """
        Populate the class variables tf_vars (a dictionary of TensorFlow
        variables representing the model), and tf_parameters (a dictionary of
        TensorFlow variables for parameters.) 
        
        params: a list of TensorFlow variables for parameters of the model.
        """
        if cls.tf_vars is None and cls.tf_parameters is None:
            # This should create the TensorFlow variables.
            tf_params = cls.tf_params()
            cls.tf_parameters = tf_params

            tf_vars = {}
            # tf_X is a TensorFlow variable representing the input 
            tf_X = tf.placeholder(config.tensorflow_config['default_float'])
            # prebuild TensorFlow graph for evaluating the log density
            tf_logp = cls.tf_log_den(tf_X, **tf_params)
            # prebuild TensorFlow graph for evaluating the gradient of the log density
            tf_logp_dx = tf.gradients(tf_logp, [tf_X])[0]

            tf_vars['tf_X'] = tf_X
            tf_vars['tf_logp'] = tf_logp
            tf_vars['tf_logp_dx'] = tf_logp_dx
            cls.tf_vars = tf_vars

    def __init__(self, **params):
        """
        params: a dictionary of variable names to parameter values (numpy arrays or
            values). The variable names should match the keys of tf_params().
        """
        self.build_graph()
        self.params = params

        self.tf_X = self.tf_vars['tf_X']
        self.tf_logp = self.tf_vars['tf_logp']
        self.tf_logp_dx = self.tf_vars['tf_logp_dx']

    def get_params(self):
        """
        Return the dictionary of parameters passed to the constructor.
        """
        return self.params

    @classmethod
    def from_params(cls, **params):
        """
        Construct an UnnormalizedDensity from the class cls and the dictionary 
        of parameters. params can be obtained from get_params().
        """
        return cls(**params)

    @classmethod
    def tf_params(cls):
        """
        Return a dictionary whose keys are variables names, and values are 
        TensorFlow variables representing parameters of the unnormalized density,
        creating the TensorFlow variables in the process.
        """
        raise NotImplementedError()

    @classmethod
    def tf_log_den(cls, X, **tf_params):
        """
        TensorFlow version of the log_den(.) method. Build a subgraph to 
        evaluate the log density.  This method is intended to be used with
        TensorFlow to find the derivative of the density with respect to the
        input vector.

        X: n x d TensorFlow variable 
        tf_params: a dictionary of TensorFlow variables for parameters of the model.
        """
        raise NotImplementedError()

    #@abstractmethod
    #def dim(self):
    #    """Return the expected input dimension of this density."""
    #    pass

    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.

        This implementation is done by automatically evaluating the TensorFlow
        graph through tf_log_den(). Subclasses may override this if necessary.

        X: n x d numpy array

        Return a one-dimensional numpy array of length n.
        """
        tf_X = self.tf_X
        tf_logp = self.tf_logp
        return self._run_with_params(tf_logp, X)

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X.

        This implementation is done by automatically computing the gradient 
        through tf_log_den() using TensorFlow. Subclasses may override this if
        necessary.

        X: n x d numpy array.

        Return an n x d numpy array of gradients.
        """
        tf_X = self.tf_X
        tf_logp_dx = self.tf_logp_dx
        return self._run_with_params(tf_logp_dx, X)

    def _run_with_params(self, tf_var, X):
        """
        Run the specified TensorFlow variable in a new Session with appropriate 
        parameter values in the feed_dict.
        """
        tf_X = self.tf_X
        to_feed = {tf_X: X}
        if self.tf_parameters is not None:
            for var_name, tfv in self.tf_parameters.iteritems():
                if var_name not in self.params:
                    raise ValueError('The parameter "{0}" is not specified.'.format(var_name))
                to_feed[tfv] = self.params[var_name]
        #print 'tf_parameters: {0}'.format(self.tf_parameters)
        #print 'params: {0}'.format(self.params)
        #print 'to_feed: {0}'.format(to_feed)

        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #init = tf.initialize_variables([tf_X])
            #sess.run(init)
            var_value = sess.run(tf_var, feed_dict=to_feed)
        return var_value



# end of UnnormalizedDensity

class IsotropicNormal(UnnormalizedDensity):
    """
    Unnormalized density of an isotropic multivariate normal distribution.
    """
    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        self.mean = mean 
        self.variance = variance
        # This is required.
        #self.params = {'mean': mean, 'variance': variance}
        super(IsotropicNormal, self).__init__(mean=mean, variance=variance)

    @classmethod
    def tf_log_den(cls, X, mean, variance):
        """
        X: TensorFlow variable representing the data 
        mean: TensorFlow variable for the mean 
        variance: TensorFlow variable (scalar) for the variance
        """
        tf_unden = -tf.reduce_sum(tf.square(X - mean), 1)/(2.0*variance)
        return tf_unden

    @classmethod
    def tf_params(cls):
        tf_vars = {
                'mean':
                tf.placeholder(config.tensorflow_config['default_float'],
                    name='mean'), 
                'variance':
                tf.placeholder(config.tensorflow_config['default_float'],
                    name='variance'),                
                }
        return tf_vars

    #def dim(self):
    #   return len(self.mean)

# end of IsotropicNormal
    

