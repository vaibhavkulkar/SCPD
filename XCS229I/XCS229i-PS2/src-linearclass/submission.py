import numpy as np
import util


def main_GDA(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    x_eval = util.add_intercept(x_eval)

    # Use np.savetxt to save outputs from validation set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        n_examples, n_features = x.shape
        phi = np.sum(y) / n_examples

        mu0 = np.dot(x.T, 1 - y) / np.sum(1 - y)
        mu0r = np.reshape(mu0, (-1, 1))

        mu1 = np.dot(x.T, y) / np.sum(y)
        mu1r = np.reshape(mu1, (-1, 1))

        y_r = np.reshape(y, (1, -1))
        mu = np.dot(mu0r, 1 - y_r) + np.dot(mu1r, y_r)

        sigma = np.dot(x.T - mu, x - mu.T) / n_examples
        sigma_inv = np.linalg.inv(sigma)

        theta = np.dot(sigma_inv, mu1 - mu0)
        theta_0 = 0.5 * mu0 @ sigma_inv @ mu0 - 0.5 * mu1 @ sigma_inv @ mu1 + np.log((1 - phi) / phi)
        self.theta = np.insert(theta, 0, theta_0)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(- np.dot(x, self.theta)))
        # *** END CODE HERE

def main_LogReg(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def hessian(x):
          index_max = self.theta.shape[0]
          mat = np.zeros((index_max, index_max))
          for k in range(index_max):
            for j in range(index_max):
              if j <= k:
                M_kj = 0
                for x_i in x:
                    M_kj += self.predict([x_i]) * (1 - self.predict([x_i])) * x_i[j] * x_i[k]
                mat[k, j] = M_kj
                mat[j, k] = M_kj
              else:
                break
          return mat / len(x)

        def grad(x, y):
          vector = np.zeros_like(self.theta)
          for j in range(len(vector)):
            vector_j = 0
            for x_i, y_i in zip(x, y):
              vector_j += (self.predict([x_i]) - y_i) * x_i[j]
              vector[j] = vector_j
          return vector / len(x)

        if self.theta is None:
          self.theta = np.zeros(x.shape[-1])
        while True:
          gradient = grad(x, y)
          matrix = hessian(x)
          inverse_hessian = np.linalg.inv(matrix)
          theta_update = self.theta - inverse_hessian @ gradient
          if np.linalg.norm(theta_update - self.theta, ord=1) < self.eps:
            break
          else:
            self.theta = theta_update
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        for x_i in x:
            assert x_i[0] == 1
        z = np.array([self.theta @ x_i for x_i in x])
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main_LogReg(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main_LogReg(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

    main_GDA(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main_GDA(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
