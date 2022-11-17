def exp_relu(x, beta=0.001):
    return K.relu(K.exp(.1*x)-1)

def log(x):
    return K.log(K.abs(x) + 1e-10)

def log_relu(x):
    return K.relu(K.log(K.abs(x) + 1e-10))

def shift_scale_tanh(x):
    return K.tanh(x-6.0)*500 + 500

def shift_scale_sigmoid(x):
    return K.sigmoid(x-8.0)*4000

def shift_scale_relu(x):
    return K.relu(K.pow(x-0.2, 3))

def shift_tanh(x):
    return K.tanh(x-6.0)

def shift_sigmoid(x):
    return K.sigmoid(x-8.0)

def shift_relu(x):
    return K.relu(x-0.2)

def scale_tanh(x):
    return K.tanh(x)*500 + 500

def scale_sigmoid(x):
    return K.sigmoid(x)*4000

def scale_relu(x):
    return K.relu((x)**3)