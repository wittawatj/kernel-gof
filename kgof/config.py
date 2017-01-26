
"""
This file defines global configuration of the project.
Casual usage of the package should not need to change this. 
"""

import tensorflow as tf

tensorflow_config = {
    # The default TensorFlow floating-point type to use.
    'default_float': tf.float64,

}

expr_configs = {
        # a directory to store temporary files
        'scratch_dir': '/nfs/data3/wittawat/',
    }
