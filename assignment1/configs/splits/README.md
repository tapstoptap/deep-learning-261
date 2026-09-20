# Saved split manifests

The main protocol uses
`fashion_mnist_train50000_val10000_seed42.json`. It records a stratified split
of the official Fashion-MNIST training set into 50,000 training and 10,000
validation samples using split seed `42`.

Changing a model run seed must not recreate or modify this split.
