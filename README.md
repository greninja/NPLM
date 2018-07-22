This is an implementation of a **Feedforward neural network** for building a language model. Though, I took inspiration from Y. Bengio's 2003 classic paper [published](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  in JMLR :  *"A Neural Probabilistic Language Model"*, I tweaked the word embeddings generation procedure to the current SOTA method : skip-gram model with Noise Contrastive Estimation. 

I have also employed the use of [Xavier algorithm](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) to initialize weights of the neural network. It determines the variance of the distribution of the weights based on the number of input and output neurons. It helps to propagate the signal deep into the network. This is because if the weights are initialized with a small value, it starts to diminish as it passes through layers and drops off to a really low value. Considering the activation function to be sigmoid, the low value of input makes the activations almost linear which beats the point of introducing non-linearity in the network. Vice versa, in the case the weights are initialized with a large value, the variance of input increases with each passing layer. Eventually, the activations again become linear since sigmoid function becomes flat for large values. Hence the need to initialize the weights with the right amount of variance :

```
Var(Wi) = 2/ (Ninp + Nout)      
```
*Kindly check the paper for detailed description*

I am using [TensorFlow](https://www.tensorflow.org/) for generating the word embeddings as well as for the language model. At the time, I wrote this code, the latest version was 1.3!



## Corpus:

As of now, I am using a basic text corpus of a synopsis of a movie.
