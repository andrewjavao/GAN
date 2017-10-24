'''
Building full connect net
'''

import tensorflow as tf

        
global_rand_seed = 2

def GLOBAL_RAND_SEED():
    global global_rand_seed
    global_rand_seed += 1
    
    return global_rand_seed


def flatten(tensor, start_dim=1):
    input_shape = tensor.get_shape().as_list()
    length = 1
    for d in range(start_dim, len(input_shape)):
        length = length * input_shape[d]
        
    output_shape = input_shape[:start_dim]
    output_shape.append(length)
    
    return tf.reshape(tensor, output_shape), length
    
    
class full_connected(object):
    def __init__(self, net_struct):
        self.net_struct = net_struct
        self.weights = []
        self._gen = False
        self.name = None
        
    def gen(self, input_tensor, input_node_num, name="full_connected"):
        assert not self._gen
        
        self.name = name
        self._gen = True
        
        layer_output = None
        prev_layer = input_tensor
        prev_layer_node_num = input_node_num
        
        weights_list = []
        bias_list = []
        outputs_list = []
        
        layer_index = 0
        
        for layer_struct in self.net_struct:
            layer_index += 1
            
            with tf.name_scope("{}_L{}".format(name, layer_index)):
                node_num = layer_struct[0]
                activation = layer_struct[1]

                weights_shape = [prev_layer_node_num, node_num]
                
                weights = tf.get_variable(
                    "weights",
                    weights_shape,
                    initializer=tf.random_normal_initializer(seed=GLOBAL_RAND_SEED())
                )
                bias = tf.get_variable(
                    "bias",
                    [node_num],
                    initializer=tf.random_normal_initializer(seed=GLOBAL_RAND_SEED())
                )
                weights_list.append(weights)
                bias_list.append(bias)

                self.weights.append(weights)

                layer_output = tf.add(tf.matmul(prev_layer, weights), bias, name="W_mul_X_add_B")
                if activation is not None:
                    layer_output = activation(layer_output, name="activation")
                               
                outputs_list.append(layer_output)

                prev_layer = layer_output
                prev_layer_node_num = node_num
        
        return layer_output, weights_list, bias_list, outputs_list
    
    def regularized(self, regularizer):
        assert self._gen
        assert regularizer is not None
        
        with tf.name_scope("{}_Regulize".format(self.name)):
            regs = [regularizer(weight) for weight in self.weights]
            return tf.add_n(regs)
    
  
    