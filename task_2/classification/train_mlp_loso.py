"""MLP compact sur les dix sujets, avec LOSO et validation interne par sujet."""
from .train_cnn_loso import main

if __name__ == '__main__':
    main(model_type='mlp')
