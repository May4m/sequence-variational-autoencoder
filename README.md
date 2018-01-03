# Sequence Based Autoencoding Variational Bayes Model

A keras implementation of sequence based variational autoencoder with the encoder and the decoder being recurrent layers and the reconstruction loss being the loss over time. Implementation based on https://arxiv.org/abs/1511.06349

The purpose of this project was to test whether auto-encoders can find differences between signals in the latent space

```
python3 sequence_vae.py
```
should train, save the encoder .h5 file and plot the latent space