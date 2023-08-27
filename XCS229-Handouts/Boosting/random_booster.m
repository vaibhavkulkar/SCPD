function [theta, feature_inds, thresholds] = random_booster(X, y, T)
% RANDOM_BOOSTER Uses random thresholds and indices to train a classifier
%
% [theta, feature_inds, thresholds] = random_booster(X, y, T)
%  performs T rounds of boosted decision stumps to classify the data X,
%  which is an m-by-n matrix of m training examples in dimension n.
%
%  The returned parameters are theta, the parameter vector in T dimensions,
%  the feature_inds, which are indices of the features (a T dimensional vector
%  taking values in {1, 2, ..., n}), and thresholds, which are real-valued
%  thresholds. The resulting classifier may be computed on an n-dimensional
%
%   theta' * sgn(x(feature_inds) - thresholds).

[mm, nn] = size(X);
p_dist = ones(mm, 1);
p_dist = p_dist / sum(p_dist);

theta = [];
feature_inds = [];
thresholds = [];

for iter = 1:T
  ind = ceil(rand * nn);
  thresh = X(ceil(rand * mm), ind) + 1e-8 * randn;
  Wplus = p_dist' * (sign(X(:, ind) - thresh) == y);
  Wminus = p_dist' * (sign(X(:, ind) - thresh) ~= y);
  theta = [theta; .5 * log(Wplus / Wminus)];
  feature_inds = [feature_inds; ind];
  thresholds = [thresholds; thresh];
  p_dist = exp(-y .* (...
    sign(X(:, feature_inds) - repmat(thresholds', mm, 1)) * theta));
  fprintf(1, 'Iter %d, empirical risk = %1.4f, empirical error = %1.4f\n', ...
          iter, sum(p_dist), sum(p_dist >= 1));
  p_dist = p_dist / sum(p_dist);
end

function s = sgn(v)
s = 2 * (v >= 0) - 1;
