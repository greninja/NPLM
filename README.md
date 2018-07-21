This is an implementation of a **Feedforward neural network** for building a language model. Though, I took inspiration from Y. Bengio's 2003 classic paper [published](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  in JMLR :  *"A Neural Probabilistic Language Model"*, I tweaked the word embeddings generation procedure to the current SOTA method : skip-gram model with Noise Contrastive Estimation.

I am using [TensorFlow](https://www.tensorflow.org/) for generating the word embeddings as well as for the language model. At the time, I wrote this code, the latest version was 1.3!
## Corpus:

As of now, I am using a basic text corpus of a synopsis of a movie.
