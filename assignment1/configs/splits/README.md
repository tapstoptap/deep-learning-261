# Saved split manifests

Store the reproducible train/validation split manifest here. The main protocol
currently reserves `fashion_mnist_seed42.json`; the data pipeline will create it
once from the official Fashion-MNIST training set using split seed `42`.

Changing a model run seed must not recreate or modify this split.
