import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
  """Base class for linear models.

  Attributes:
    theta: np.ndarray, dtype=np.float64, shape=(n_features,). Weights vector for
      the model.
  """

  def __init__(self, theta=None):
    """
    Args:
      theta: (See class definition)
    """
    self.theta = theta

  def fit(self, x, y):
    """Fits the linear model to x -> y using np.linalg.solve.

    Remember to update self.theta with the fitted model parameters.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Inputs.
      y: np.ndarray, dtype=np.float64, shape=(n_examples,). Outputs.

    Returns: Nothing
    """
    pass
    # *** START CODE HERE ***
    #mat_x = x
    #mat_y = y
    X = np.dot(np.transpose(x), x)
    Y = np.dot(np.transpose(x), y)
    self.theta = np.linalg.solve(X, Y)
    # *** END CODE HERE ***

  def predict(self, x):
    """ Makes a prediction given a new set of input features.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Model input.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples,). Model output.
    """
    pass
    # *** START CODE HERE ***
    return np.dot(x, self.theta)
    # *** END CODE HERE ***

  @staticmethod
  def create_poly(k, x):
    """ Generates polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples,). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+1). Polynomial
      features of x with powers 0 to k (inclusive).
    """
    pass
    # *** START CODE HERE ***
    poly = np.ones((len(x), k+1), np.float64)
    for xi in range (0,len(x)):
      for ki in range(1, k+1):
        xpower = np.power(x[xi], ki)
        poly[xi, ki] = xpower
    return poly
    # *** END CODE HERE ***

  @staticmethod
  def create_sin(k, x):
    """ Generates sine and polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples,). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+2). Sine (column
      0) and polynomial (columns 1 to k+1) features of x with powers 0 to k
      (inclusive).
    """
    pass
    # *** START CODE HERE ***
    poly = np.ones((len(x), k+2), np.float64)
    for xi in range (0, len(x)):
      for ki in range(0, k+2):
        if ki == 0:
          xpower = np.sin(x[xi])
        else:
          xpower = np.power(x[xi], ki-1)
        poly[xi, ki] = xpower
    return poly
  # *** END CODE HERE ***

def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
  train_x,train_y=util.load_dataset(train_path,add_intercept=False)
  plot_x = np.ones([1000, 1])
  plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
  plt.figure()
  plt.scatter(train_x, train_y)

  for k in ks:
      '''
      Our objective is to train models and perform predictions on plot_x data
      '''
      # *** START CODE HERE ***
      model = LinearModel()
      if sine:
        model.fit(LinearModel.create_sin(k, train_x), train_y)
        plot_y = model.predict(LinearModel.create_sin(k, plot_x))
      else:
        model.fit(LinearModel.create_poly(k, train_x), train_y)
        plot_y = model.predict(LinearModel.create_poly(k, plot_x))
      # *** END CODE HERE ***
      '''
      Here plot_y are the predictions of the linear model on the plot_x data
      '''
      plt.ylim(-2, 2)
      plt.plot(plot_x[:, 0], plot_y, label='k=%d' % k)

  plt.legend()
  plt.savefig(filename)
  plt.clf()


def main(train_path, small_path, eval_path):
  '''
  Run all experiments
  '''
  run_exp(train_path, False, [3], 'large-poly3.png')
  run_exp(train_path, True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
  run_exp(train_path, False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
  run_exp(small_path, True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
  run_exp(small_path, False, [1, 2, 3, 5, 10, 20], 'small-poly.png')

if __name__ == '__main__':
  main(train_path='train.csv',
      small_path='small.csv',
      eval_path='test.csv')
