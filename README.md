# keras-CosineSoftmax-Layer
Implementation of CosineSoftmax Layer for keras (softmax with normalized weights and normalized activations):


I changed keras' Dense layer to implement Cosine Metric Learning as proposed in:

Wojke, Nicolai and Bewley, Alex (2018) Deep Cosine Metric Learning for Person Re-Identification. In: IEEE Winter Conference on Applications of Computer Vision (WACV). [Manuscript accepted for publication]

Implementation details: I changed the following function:

__output = K.dot(inputs, self.kernel)__

to

__output = K.dot(K.l2_normalize(inputs), K.l2_normalize(self.kernel))__

and set the default activation function to "softmax" and "use_bias" to False
