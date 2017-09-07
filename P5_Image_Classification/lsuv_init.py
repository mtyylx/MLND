from __future__ import print_function
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Convolution2D

# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q
def get_activations(model, layer, X_batch):
    intermediate_layer_model = Model(input=model.get_input_at(0), output=layer.get_output_at(0))
    activations = intermediate_layer_model.predict(X_batch)
    return activations

def LSUVinit(model,batch):
    # only these layer classes considered for LSUV initialization; add more if needed
    classes_to_consider = (Dense, Convolution2D)
    
    margin = 0.1
    max_iter = 10
    layers_inintialized = 0
    for layer in model.layers:
        print(layer.name)
        if not any([type(layer) is class_name for class_name in classes_to_consider]):
            continue
        # avoid small layers where activation variance close to zero, esp. for small batches
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            print(layer.name, 'too small')
            continue
        print('LSUV initializing', layer.name)
        layers_inintialized += 1
        w_all=layer.get_weights();
        weights = np.array(w_all[0])
        weights = svd_orthonormal(weights.shape)
        biases = np.array(w_all[1])
        w_all_new = [weights,biases]
        layer.set_weights(w_all_new)
        acts1=get_activations(model, layer, batch)
        var1=np.var(acts1)
        iter1=0
        needed_variance = 1.0
        print(var1)
        while (abs(needed_variance - var1) > margin):
            w_all=layer.get_weights();
            weights = np.array(w_all[0])
            biases = np.array(w_all[1])
            if np.abs(np.sqrt(var1)) < 1e-7: break      # avoid zero division
            weights /= np.sqrt(var1)/np.sqrt(needed_variance)
            w_all_new = [weights,biases]
            layer.set_weights(w_all_new)
            acts1=get_activations(model, layer, batch)
            var1=np.var(acts1)
            iter1+=1
            print(var1)
            if iter1 > max_iter:
                break
    print('LSUV: total layers initialized', layers_inintialized)   
    return model
