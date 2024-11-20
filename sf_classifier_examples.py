"""
Examples of running tf_classifers.
"""
import numpy as np
import sf_classifiers
import matplotlib.pyplot as plt


# ----------------------------------------------
# Functions for generating synthetic test data
# ----------------------------------------------

def two_clusters(num_points, noise=.3, show=False):
    """ Synthetic two-class dataset. """
    features = np.random.randint(2, size=(num_points, 1))
    features = np.append(features, features, axis=1)
    labels = np.array(np.logical_or(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + np.random.normal(0, noise, features.shape),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


def noisy_xor(num_points, show=False):
    """ Synthetic Dataset that is not linearly separable. """

    features = np.random.randint(2, size=(num_points, 2))
    labels = np.array(np.logical_xor(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + (np.random.random(features.shape) - .5),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


# ----------------------------------------------
# Examples of training models
# ----------------------------------------------


def logistic_regression_clusters(lr=.05, epochs=100):
    dataset_train_x, dataset_train_y = two_clusters(500)
    dataset_test_x, dataset_test_y = two_clusters(500)

    classifier = sf_classifiers.LogisticRegression(2)
    classifier.graph.gen_dot("logres.dot")

    classifier.train(dataset_train_x,
                     dataset_train_y,
                     epochs=epochs, learning_rate=lr)

    classifier.score(dataset_test_x, dataset_test_y)
    classifier.plot_2d_predictions(dataset_train_x, dataset_train_y)


def mlp_xor(lr=.01, epochs=50, activation='sigmoid'):
    all_losses = []
    for run in range(5):
        dataset_train_x, dataset_train_y = noisy_xor(500)
        dataset_test_x, dataset_test_y = noisy_xor(500)

        classifier = sf_classifiers.MLP(2, [10, 10, 10, 10, 10], activation=activation)

        losses = classifier.train(dataset_train_x, dataset_train_y, epochs=epochs,
                        learning_rate=lr)
        classifier.score(dataset_test_x, dataset_test_y)
        all_losses.append(losses)

    for losses in all_losses:
        plt.plot(range(epochs), losses, label=f"Run {all_losses.index(losses) + 1}")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{activation.capitalize()} Activation Learning Curve")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    mlp_xor()
    #logistic_regression_clusters()
