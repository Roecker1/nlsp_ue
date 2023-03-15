# Nonlinear Signal Processing
# Assignment 1

| First Name | Last Name  | Matr. Number |
| ---------- | ---------- | ------------ |
| Fritz      | Hierzegger | 11729768     |
| Thomas     | Röck       | 11727563     |


| Problem | a)           | b)           | c)           | d)           | e)           |
| ------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| 1       | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| 2       | $\checkmark$ | $\checkmark$ |              |              |              |
| 3       |              |              |              |              |              |
| 4       | $\checkmark$ | $\checkmark$ |              |              |              |
| 5       | $\checkmark$ | $\checkmark$ | $\checkmark$ |              |              |
<div style="page-break-after: always;"></div>

\pagebreak

## Problem 1: Universal Approximators and System Identification


### Task a) (3 Points) Implement a polynomial model
On top of the provided code skeleton we extended the `fit`-function of the `PolynomialModel` class:

```python
class PolynomialModel:
    def __init__(self, order: int):
        self.order: int = order
        self.alpha: np.ndarray

    def predict(self, x):
        return np.array([sum(self.pbf(x_n, self.order) * self.alpha)
                         for x_n in x])

    def fit(self, data, targets):
        A = self._make_feature_matrix(data, self.order)
        alpha = np.linalg.pinv(A) @ targets.reshape(-1, 1)
        self.alpha = alpha.ravel()

        return self._mse(targets, self.predict(data))

    def pbf(self, x, k):
        return np.tile(x, (k + 1,)) ** np.arange(k+1)

    def _make_feature_matrix(self, x, N):
        A = np.array([self.pbf(x_n, N) for x_n in x])
        return A

    @staticmethod
    def _mse(y_true, y_pred):
        N = len(y_true)
        assert N == len(y_pred)

        return np.mean((y_true - y_pred) ** 2)
``` 
where the maximum order $P$ was saved as an instance-attribute of the class. The function class $\phi _k(x)$ for polynomials is defined as 

$$
\phi_k(x) = x ^ {k - 1}.
$$

The feature-matrix $A$ for data $X = \{(x_i,y_i\}_{i=1...N_{train}}$ is then computed using the method `_make_feature_matrix`. The parameters $\alpha$ can then be computed using the relation
$$
\alpha = (A^T A)^{-1}A^Ty
$$
where $(A^T A)^{-1}A^T$ is the pseudo-inverse of the feature matrix $A$.

A prediction $\hat{f}(x)$ on a new sample $x$ can be computed using the equation
$$
\hat{f}(x) = \sum_{p=0}^{P}\alpha_p \cdot x^p 
$$
for which the mean-squared-error (MSE) can be calculated. For an order $P \in [1, 15]$ a Polynomial Model is trained on the training-data. The MSE for both the tran- and testset are shown in the following table.

| Order $P$ | Train MSE | Test MSE    |
| ----- | --------- | ----------- |
| 1 | 1.975 | 1.645 |
| 2 | 1.946 | 1.571 |
| 3 | 0.986 | 0.677 |
| 4 | 0.736 | 0.527 |
| 5 | 0.202 | 0.998 |
| 6 | 0.164 | 0.251 |
| 7 | 0.107 | 0.404 |
| 8 | 0.082 | 1.464 |
| 9 | 0.081 | 0.414 |
| 10 | 0.076 | 27.591 |
| 11 | 0.076 | 61.323 |
| 12 | 0.075 | 2037.45 |
| 13 | 0.052 | 543525.33 |
| 14 | 0.051 | 1576728.601 |
| 15 | 0.102 | 13076714.76 |

For the training data, a model with order $P=14$ minimizes the MSE ($MSE_{train} = 0.051, MSE_{test} = 1583011.914$). This result is interesting since we would have expected the maximum order ($P = 15$) to yield the lowest MSE. This could be due to numerical resolution problems when dealing with such high order polynomials.

The model with order $P=6$ results in the lowest MSE for the test-data ($MSE_{train} = 0.164, MSE_{test} = 0.251$). When comparing the MSE curves for the train- and test over increasing model orders we can observe that with a higher order $P$, the MSE for the training set decreases. This also applies for the test set but only until order 4. For higher orders, the MSE fluctuates and, for an order of $p \geq 10$, results in enormous values due to overfitting.

The model one should choose for data that is neither in the train-set nor the test-set should be the model that resulted in the lowest MSE for the test-set.

\pagebreak

![](https://i.imgur.com/p4I0E7G.png)
*Mean squared errors for a maximum model order $P = 9$*


![](https://i.imgur.com/0sE4O7G.png)
*Mean squared errors for a maximum model order $P = 15$*

![](https://i.imgur.com/UizYiFf.png)
*Approximation of the system on support $x \in \{-10, 10\}$ using the order $k\leq 10$ that minimizes the MSE for the train- and testset respectively*


### Task b) (3 Points)
In this task we split the trainset into a train- and validation set with a trainsize of 70% by shuffling the trainset and slicing it.

```python
train_data = pd.read_csv(data_dir + 'training-set.csv').to_numpy()
np.random.shuffle(train_data)
np.random.seed(100)
split = round(0.7*np.shape(train_data)[0])
x_train, y_train = train_data[0:split,0], train_data[0:split,1]
x_val, y_val = train_data[split::,0], train_data[split::,1]
```

The distribution of the samples can be seen here

![](https://i.imgur.com/DSjdJo1.png)

The MSE for train, test and validation set are given in the next table
| Order $P$ | Train MSE | Test MSE      | Validation MSE |
| --------- | --------- | ------------- | -------------- |
| 1         | 1.968     | 1.65          | 2.046          |
| 2         | 1.968     | 1.646         | 2.036          |
| 3         | 0.749     | 1.004         | 2.088          |
| 4         | 0.651     | 0.655         | 1.356          |
| 5         | 0.231     | 0.877         | 0.194          |
| 6         | 0.183     | 0.22          | 0.16           |
| 7         | 0.114     | 0.411         | 0.102          |
| 8         | 0.088     | 1.952         | 0.075          |
| 9         | 0.079     | 2.277         | 0.237          |
| 10        | 0.076     | 28.735        | 0.144          |
| 11        | 0.076     | 10.945        | 0.132          |
| 12        | 0.07      | 36661.82      | 2.175          |
| 13        | 0.032     | 2963295.72    | 68.006         |
| 14        | 0.031     | 4068576.683   | 57.991         |
| 15        | 0.034     | 142616469.776 | 322.361        |

The model with order $P=8$ results in the lowest MSE for the validation set (MSE=0.075).

In the following figure, the approximations over the support $x \in \{-10, 10\}$ are shown.
![](https://i.imgur.com/ZWnEUAE.png)

As already mentioned, we chose 30% of the samples originally contained in the training-data for the validation set. Choosing a larger portion for the validation set would result in a better generalization for new, unknown data whereas a larger training set (smaller validation set) would approximate the whole system better.


### Task c) (3 Points) Implement a Gaussian radial basis function (RBF) model $\hat{f}$

In this task we implemented a Gaussian radial basis function (RBF) model $\hat{f}$
$$
\hat{f}(x) = \alpha_0 + \sum_{p=1}^{P} \alpha_p \text{e}^{-\frac{(x-c_p)^2}{w_p}}
$$
of order $P$. the centers were equally spaced on the support of the input x and the lengthscale was chosen as a single parameter $w_1 = w_2 = \cdots = w$.
Again, we extended the `fit`-function of the `RBFModel` class:

```python
class RBFModel:
    def __init__(self, order: int, lengthscale: float = 10.):
        self.order = order
        self.lengthscale = lengthscale
        self.alpha: np.ndarray
        self.centers: np.ndarray

    def __str__(self):
        return "RBFModel"

    def predict(self, x):
        return np.array([
            sum(
                self.rbf(x_n, self.centers, self.lengthscale) *
                self.alpha[1:],
                self.alpha[0]) for x_n in x])

    def fit(self, data, targets):
        self.centers = np.linspace(data.min(), data.max(), self.order-1)
        return PolynomialModel.fit(self, data, targets)

    @staticmethod
    def rbf(x, c, w):
        return np.e ** -((x - c) ** 2 / w)

    def _make_feature_matrix(self, x, N):
        def phi_x(x): return np.append(
            np.array([1]), self.rbf(x, self.centers, self.lengthscale))

        A = np.array([phi_x(x_n) for x_n in x])
        return A

    def _mse(self, y_true, y_pred):
        return PolynomialModel._mse(y_true, y_pred)
```


The following table shows the resulting MSE for the train- and testset with the default lengthscale (`10.0`). An order of $P=8$ results in the lowest MSE (0.157) for the testset.

| Order $P$ | Train MSE | Test MSE    |
| ----- | --------- | ----------- |
| 1 | 2.068 | 1.779 |
| 2 | 2.037 | 1.994 |
| 3 | 1.973 | 1.877 |
| 4 | 1.898 | 2.118 |
| 5 | 0.165 | 0.316 |
| 6 | 0.699 | 0.713 |
| 7 | 0.169 | 0.221 |
| 8 | 0.102 | 0.093 |
| 9 | 0.08 | 0.208 |
| 10 | 0.079 | 0.24 |
| 11 | 0.078 | 0.114 |
| 12 | 0.073 | 0.25 |
| 13 | 0.073 | 0.13 |
| 14 | 0.061 | 144.283 |
| 15 | 0.054 | 2606.77 |

![](https://i.imgur.com/DGl7kQt.png)


#### Questions

- How does the width parameter influence the RBF models’ performance? Is it better to choose a small or large width?
    - We cannot clearly say that a model's improved performance is a direct result of, let's say, a larger width since the choice of the lengthscale parameter does also depend on the model order.
- How does the choice of width and model order interact/depend on each other?
    - Generally we can say that we have to choose a lower width when having a higher order, otherwise the RBF's would overlap too much. A theroetical solution on the optimal choice for the width parameter is rather tricky to derive and has been a central topics in works on RBF's (for example [here](http://www.math.iit.edu/~fass/Dolomites.pdf))
- Think about a multi-variate system, e.g. $x \in Rn$ with n = 100, what could be potential problems when we try to choose the RBF centers and widths by hand?
    - Time (for the user to find these parameters)
    - Accuracy
    - Optimization

### Task d) (3 Points)

This network is built upon the `SimpleMLP` class that is provided in the course-repository where changed only the `__init__` and `forward` method according to the assignment's description.


```python
class MLPModel(nn.Module):
    def __init__(self, hidden_sizes: List[int], activation_function=torch.tanh):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.input_layer = nn.Linear(1, hidden_sizes[0])
        self.hidden_layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(
            hidden_sizes[:-1], hidden_sizes[1:])]
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.activation_function = activation_function

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)

        x = x.view((-1, 1))

        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))

        x = self.output_layer(x)
        return x
```
For the network we chose one input- and one output layer as well as a number of hidden layers with a certain number of neurons respecitvely. In the following table, the hidden layers are given in the column 'Hidden Layers', where each element in the list corresponds to the number of neurons in that layer. We testet two activation functions: 
1. Hyperbolic tangent (tanh) 
2. Rectified linear unit (relu) with
    $$
    (x)^+ = \begin{cases} 0 & \text{if} \; x \leq 0 \\ x & \text{if} \; x > 0 \end{cases}
    $$
    
    
| Layers | $MSE_{train}$ (tanh) | $MSE_{train}$ (relu) | $MSE_{test}$ (tanh)| $MSE_{test}$ (relu) |
| - |- |- |- |- |
| [0] | 2.156 | 2.156 | 1.784 | 1.782 |
| [1] | 1.337 | 1.847 | 1.325 | 1.659 |
| [5] | 0.766 | 0.322 | 1.092 | 0.776 |
| [10] | 0.365 | 0.312 | 0.783 | 0.721 |
| [32] | 0.177 | 0.262 | 0.472 | 0.688 |
| [64] | 0.169 | 0.322 | 0.464 | 0.933 |
| [1, 1] | 1.336 | 1.919 | 1.320 | 1.681 |
| [5, 5] | 0.858 | 0.392 | 0.944 | 0.699 |
| [10, 10] | 0.171 | 0.311 | 0.469 | 0.696 |
| [32, 32] | **0.166** | **0.133** | 0.510 | **0.417** |
| [64, 64] | 0.170 | 0.199 | **0.460** | 0.616 |
| [10, 10, 10] | 0.248 | 0.337 | 0.641 | 0.704 |
| [32, 32, 32] | 0.197 | 0.199 | 0.571 | 0.509 |
| [64, 32, 16] | 0.178 | 0.213 | 0.503 | 0.634 |
| [32, 64, 32] | 0.195 | 0.207 | 0.873 | 0.556 |

And reported the MSE for both the training- and testset for the two activation functions. The lowest MSE for each scenario is highlighted in a bold font. When comparing the activation functions we cannot clearly say which performs better since there are cases where either one outperforms the other. However, both for the train- and testset the lowest MSE was achieved with the relu activation function. We tested the FF-NN with 0 to 3 hidden layers of up to 64 neurons. Looking at the table that the maximum number of layers and neurons does not necessarily mean a lower MSE. For the testset, the best approximation for the system is achieved with the hidden layer structure of `[32, 32]`, i.e. two layers with 32 neurons each. This also applied for the testset when predicting with a relu activation. However, for the tanh activation function a network with more neurons performed better (`[64, 64]` hidden layer structure).

For the final model we chose a network with two hidden layers containing 32 neurons and a tanh activation function. The approximation of the network on the support x can be seen in the following figure. With an MSE of 0.166 we can 


![](https://i.imgur.com/6oMhg1J.png)
*Approximation of the system on support $x∈{−10,10}$ by the MLP with the structure that minimizes the MSE for the train- and testset respectively*


Comparing the network to the MSE of the testset of the previous implementations from task a) and c), the network approximates the system better than the polynomial model but has a higher MSE than the one with radial basis functions.


### Task e) (3 Points)
For this simple system we would certainly choose an universal approximator with radial basis functions because it was easy to implement and we considered it to be a very elegant solution to the problem that did also result in the overall best approximation score for the test data. However, for any other system it would probably be the safest approach to choose the MLP in terms of accuracy and true approximation of the system.

## Problem 2: Harmonic Analysis and Equalization

### Task a) (6 Points) Analytically derive the coefficients $\alpha_0 \text{...} \alpha_3$

This involves plugging the input signal into the Taylor series representation of a general system $f(z)$.
The Taylor expansion $\hat{f}$ of a system $f$ for an input signal $x[n]$ is defined as follows:
$$
\begin{equation}
\hat{f}(x[n]) = \sum_{k=0}^{K} \left.\frac{1}{k!}\frac{d^{(k)}f(z)}{dz^{(k)}}\right\vert_{z=c}(x[n]-c)^k
\end{equation}
$$

Defining $x[n]=A\cdot cos(\theta_0 n)$ and $K=3$:
$$
\begin{aligned}
\hat{f}(x[n]) &= \sum_{k=0}^{3} \left.\frac{1}{k!}\frac{d^{(k)}f(z)}{dz^{(k)}}\right\vert_{z=c}(A\cdot cos(\theta_0 n)-c)^k\\\\
&=\hat{f}_0(x[n])+\hat{f}_1(x[n])+\hat{f}_2(x[n])+\hat{f}_3(x[n])
\end{aligned}
$$
Let's now look at the sum terms $\hat{f}_0(x[n]),\hat{f}_1(x[n]),\hat{f}_2(x[n]),\hat{f}_3(x[n])$ separately.
For $k=0$:

$$
\begin{aligned}
\hat{f}_0(x[n]) &= \left.\frac{1}{0!}\frac{d^{(0)}f(z)}{dz^{(0)}}\right\vert_{z=c}(A\cdot cos(\theta_0 n)-c)^0 \\
&= f(c).
\end{aligned}
$$


For $k=1$:
$$
\begin{aligned}
\hat{f}_1(x[n]) &= \left.\frac{1}{1!}\frac{d^{(1)}f(z)}{dz^{(1)}}\right\vert_{z=c}(A\cdot cos(\theta_0 n)-c)^1 \\\\
&=f'(c)(A\cdot cos(\theta_0 n)-c).
\end{aligned}
$$

For $k=2$:
\begin{aligned}
\hat{f}_2(x[n]) &= \left.\frac{1}{2!}\frac{d^{(2)}f(z)}{dz^{(2)}}\right\vert_{z=c}(A\cdot cos(\theta_0 n)-c)^2 \\\\
&=\frac{f''(c)}{2}(A\cdot cos(\theta_0 n)-c)^2 \\\\
&= \frac{f''(c)}{2}(A^2\cdot cos^2(\theta_0 n) - 2Acos(\theta_0 n) + c^2).
\end{aligned}
Since $cos^2(\phi) = \frac{1}{2}(1+cos(2\phi))$:
\begin{aligned}
\hat{f}_2(x[n]) &= \frac{f''(c)}{2}(A^2\cdot cos^2(\theta_0 n) - 2Acos(\theta_0 n) + c^2) \\\\
&= \frac{f''(c)}{2}\left(\frac{A^2}{2}\cdot (1+cos(2\theta_0 n)) - 2Acos(\theta_0 n) + c^2 \right )
\end{aligned}

For $k=3$:
\begin{aligned}
\hat{f}_3(x[n]) &= \left.\frac{1}{3!}\frac{d^{(3)}f(z)}{dz^{(3)}}\right\vert_{z=c}(A\cdot cos(\theta_0 n)-c)^3 \\\\
&=\frac{f'''(c)}{6}(A^3\cdot cos^3(\theta_0 n) - 3A^2c\cdot cos^2(\theta_0 n) + 3Ac^2\cdot cos(\theta_0 n)
-c^3)
\end{aligned}
Since $cos^2(\phi) = \frac{1}{2}(1+cos(2\phi))$ and $cos^3(\phi)=\frac{1}{4}(3cos(\phi)+cos(3\phi))$:
\begin{aligned}
\hat{f}_3(x[n]) &=\frac{f'''(c)}{6}(A^3\cdot cos^3(\theta_0 n) - 3A^2c\cdot cos^2(\theta_0 n) + 3Ac^2\cdot cos(\theta_0 n) -c^3) \\
&=\frac{f'''(c)}{6}\left ( \frac{A^3}{4}(3cos(\phi)+cos(3\phi)) - \frac{3A^2c}{2}(1+cos(2\phi)) + 3Ac^2\cdot cos(\theta_0 n) -c^3 \right )
\end{aligned}
After reassembling $\hat{f}(x[n])$ and rearranging to find the coefficients for the cosine terms:
\begin{aligned}
\hat{f}(x[n]) =& \underbrace{ f(c) - c \cdot f'(c) + \frac{f''(c)}{2}\cdot \left( \frac{A^2}{2} + c^2 \right ) - \frac{f'''(c)}{6} \left( \frac{3A^2c}{2} + c^3 \right )}_{\Large{\alpha_0}}\\\\
+cos(\theta_0 n) &\underbrace{ \left( A\cdot f'(c) - Ac\cdot f''(c) + \frac{f'''(c)}{6} \left ( \frac{3A^3}{4} + 3Ac^2 \right ) \right)}_{\Large{\alpha_1}}  \\\\
+cos(2\theta_0 n)&\underbrace{\left(\frac{A^2\cdot f''(c)}{4} - \frac{3A^2c\cdot f'''(c)}{12} \right )}_{\Large{\alpha_2}} \\\\
+cos(3\theta_0 n) &\underbrace{\left(\frac{A^3\cdot f'''(c)}{24}\right)}_{\Large{\alpha_3}}
\end{aligned} 

Thus:
\begin{aligned}
\alpha_0 =& f(c) - c \cdot f'(c) + \frac{f''(c)}{2}\cdot \left( \frac{A^2}{2} + c^2 \right ) - \frac{f'''(c)}{6} \left( \frac{3A^2c}{2} + c^3 \right ),\\\\
\alpha_1 =&  A\cdot f'(c) - Ac\cdot f''(c) + \frac{f'''(c)}{2} \left ( \frac{A^3}{4} + Ac^2  \right),  \\\\
\alpha_2 =&\frac{A^2}{4}\left(f''(c) - c\cdot f'''(c) \right ),\\\\
\alpha_3 =&\frac{A^3\cdot f'''(c)}{24}.
\end{aligned} 

### Task b) (3 Points) Derive the third order Taylor approximation around the centers $c \in \{0, \ln 2\}$
Now the third order Taylor approximation was to be derived for a given system $y(x[n])$:\
$$
y(x[n]) = \sigma(x[n]) = \frac{1}{1+e^{-x[n]}}.
$$
First the first 3 derivatives have to be determined. Since $\sigma(x[n]) = ({1+e^{-x[n]}})^{-1}$ the first derivative can be easily determied using the chain rule: 
$$
\begin{aligned}
\sigma'(x[n]) &= -(1+e^{-x[n]})^{-2}\cdot (-e^{-x[n]}) = \frac{e^{-x[n]}}{(1+e^{-x[n]})^2} \\\\
&=\underbrace{\frac{1}{1+e^{-x[n]}}}_{\Large{\sigma(x[n])}} \cdot \frac{e^{-x[n]}}{1+e^{-x[n]}} \\\\
&=\sigma(x[n]) \cdot \underbrace{\left( \frac{1+e^{-x[n]}}{1+e^{-x[n]}} - \frac{1}{1+e^{-x[n]}} \right)}_{\Large{1-\sigma(x[n])}} \\\\
\sigma'(x[n]) &= \sigma(x[n]) \cdot (1-\sigma(x[n])).
\end{aligned}
$$
Since the exact analytical derivatives are not needed, but are just used to be evaluated at the given centers, the second and third derivatives are now derived in terms of nested first derivatives.
This is done as follows by using the product rule:
$$
\sigma''(x[n]) = \sigma'(x[n])\cdot(1-\sigma(x[n])) -\sigma(x[n])\cdot \sigma'(x[n]) = \sigma'(x[n])\cdot(1-2\sigma(x[n])).
$$
For the third derivative:
$$
\sigma'''(x[n]) = \sigma''(x[n]) \cdot (1-2\sigma(x[n])) - 2(\sigma'(x[n]))^2.
$$
These can now be evaluated for $x[n] \in \{0,\ln 2\}$.
For $x[n]=0$:
$$
\begin{aligned}
\sigma(0) &= \frac{1}{1+e^0} = \frac{1}{2}, \\\\
\sigma'(0) &= \frac{1}{2} \cdot \left (1-\frac{1}{2} \right) = \frac{1}{4}, \\\\
\sigma''(0) &= \frac{1}{4} \cdot \left(1-2\cdot\frac{1}{2}\right) = 0, \\\\
\sigma'''(0) &= 0 \cdot \left(1-2\cdot\frac{1}{2}\right) - 2\left(\frac{1}{4}\right)^2 =-\frac{1}{8}.
\end{aligned}
$$
For $x[n] = \ln 2$:
$$
\begin{aligned}
\sigma(\ln 2) &= \frac{1}{1+e^{-\ln 2}} = \frac{1}{\frac{3}{2}} = \frac{2}{3}, \\\\
\sigma'(\ln 2) &= \frac{2}{3} \cdot \left (1-\frac{2}{3} \right) = \frac{2}{9}, \\\\
\sigma''(\ln 2) &= \frac{2}{9} \cdot \left(1-2\cdot\frac{2}{3}\right) = -\frac{2}{27}, \\\\
\sigma'''(\ln 2) &= -\frac{2}{27} \cdot \left(1-2\cdot\frac{2}{3}\right) - 2\left(\frac{2}{9}\right)^2 =-\frac{6}{81} = -\frac{2}{27}.
\end{aligned}
$$
Now these can be used in the derived equations for $\alpha_{0\text{...}3}$ to get the coefficients of the harmonics of the input signal.
For $x[n]=0$:
$$
\begin{aligned}
\alpha_0 &= \frac{1}{2}, \\\\
\alpha_1 &= \frac{A}{4}-\frac{A^3}{64}, \\\\
\alpha_2 &= 0, \\\\
\alpha_3 &= -\frac{A^3}{192}.
\end{aligned}
$$

For $x[n]=\ln2$:
$$
\begin{aligned}
\alpha_0 &= \frac{2}{3}-\frac{2}{9}\ln2-\frac{1}{27}\cdot \left( \frac{A^2}{2}+\ln^2 2 \right)+\frac{1}{81}\left(\frac{3A^2 \ln 2}{2}+ln^3 2\right), \\\\
\alpha_1 &= \frac{2A}{9}+\frac{2A\ln 2}{27} - \frac{1}{27}\left( \frac{A^3}{4}+A\ln^2 2\right), \\\\
\alpha_2 &= \frac{A^2}{54}\cdot\left(\ln2 -1 \right), \\\\
\alpha_3 &= -\frac{A^3}{324}.
\end{aligned}
$$

### Task c) (6 Points) Verify both approximations from Task b) numerically

## Problem 4: Normalizing Flows


### Task a) (5 Points) Implement a coupling flow using Pyro and PyTorch

The used ground truth is a bivariate destribution given by means of a black and white image depicting an upwards pointing arrow. This is the distribution that is to be learned by the coupling flow architecture. The image is given in the following Figure.

![](https://i.imgur.com/bYaIvan.png)

The given distribution assigns a probability of 0 to all white pixels and equal probabilty to all black pixels.

Then 5000 samples were drawn from this distribution and plottes using a scatter plot . This can be seen in the following Figure.


![](https://i.imgur.com/kwtPCvb.png)

Those were the samples used in the following implementation of a normalizing flow using Spline coupling.
It was implemented using the Pyro function <tt> spline_coupling </tt> and the Pytorch implementation of the Adam optimizer. An excerpt of the used code that implements the training process follows:

```python
base_dist = dist.Normal(torch.Tensor([0,0]), torch.ones(2))
num_components = 3
transforms = [dist.transforms.spline_coupling(2, count_bins=32, bound=3) for _ in range(num_components)]
flow_dist = dist.TransformedDistribution(base_dist, transforms)
# configure optimizier
modules = torch.nn.ModuleList(transforms)
optimizer = torch.optim.Adam(modules.parameters(), lr=2e-2)
print(flow_dist)
# start training
steps = 900
for step in range(steps+1):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(data).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()
```
As can be seen from the code excerpt, 3 cascaded spline coupling flow stages were used, as given by the <tt>num_components</tt> variable. The learning rate was set to be 0.02, and the training was done for 900 steps.
<tt>count_bins</tt>, which defines the number of Spline Parameters was set to 32, and <tt>bound</tt> was set to 3. <tt>bound</tt> defines the range, or bounding box of the parametrized Spline. Since the given datapoints were scaled to range from -0.5 to 0.5, <tt>bound</tt> had to be at least 0.5. All parameters were chosen based on trial and error, within their respective sensible ranges. After training, the achieved logarithmic loss was **-0.9169**. Then 5000 samples were drawn from the learned distribution and plotted. This can be seen in the following Figure.

![](https://i.imgur.com/nbpJMjY.png)

Clearly the learned distribution approximates the given ground truth quite well.
This can be further examined by plotting the log-likelyhood of the learned distribution.
This was done by evaluating the learned distribution at a predefined grid, spanning across the entire ground truth distribution. The result can be seen in the following Figure.

![](https://i.imgur.com/0jKubxM.png)

### Task b) (5 Points) Implement an autoregressive coupling flow using Pyro and PyTorch

For this task, the training part of the code was changed slightly to use autoregressive Spline coupling.
This was achieved using Pyro's <tt>spline_autoregressive</tt> function.
This excerpt shows how the autoregressive coupling was implemented:

``` python
data = torch.Tensor(data)
base_dist = dist.Normal(torch.Tensor([0,0]), torch.ones(2))
num_components = 3
transforms = [dist.transforms.spline_autoregressive(2, count_bins=32, bound=3) for _ in range(num_components)]
flow_dist = dist.TransformedDistribution(base_dist, transforms)
# configure optimizier
modules = torch.nn.ModuleList(transforms)
optimizer = torch.optim.Adam(modules.parameters(), lr=2e-2)
print(flow_dist)
# start training
steps = 900
for step in range(steps+1):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(data).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()
```
The training was repeated using the same parameters as listed above. This yielded a logarithmic loss of **-0.9100**. The following figure shows the result of drawing 5000 samples from the learned distribution:
![](https://i.imgur.com/nhRAWJe.png)
As can be seen from the plot aswell as the achieved loss, the results are quite similar to those achieved when using simple Spline coupling.
The following plot shows the log likelihood, evaluated at the entire range of the ground truth distribution. 
![](https://i.imgur.com/Oi9IKKn.png)
While the achieved loss might be slightly worse than when using spline coupling, visually the ground truth distribution seems to be separated better, especially the regions to the left and right of the arrow.
The used parameters however were the ones that were optimized for spline coupling, so tweaking the parameters could definitely improve the achieved loss aswell as 


## Problem 5: Speech Command Recognition with CNNs
### Task a)
We implemented a simple CNN with one convolution layer, one hidden layer and a log-softmax output layer. To overcome the proplem with the variable length of the sequences we calculated averaged along the entire sequence using `torch.mean`.
```python
class KeyWordCNN1d(nn.Module):
    def __init__(self, num_classes, num_features, num_kernels, mem_depth, num_hidden=20):
        super().__init__()
        self.conv_layer = nn.Conv1d(num_features, num_kernels, mem_depth)
        self.hidden_layer = nn.Linear(num_kernels, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_classes)

    def forward(self, x:torch.Tensor):
        x = self.conv_layer(x))
        x = F.relu(x)
        x = torch.mean(x, dim=2)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        output = F.log_softmax(x, dim=-1)
        return output
```

With `num_kernels=80` (we chose such a high number of kernels to provide a fair comparison to the network in task b)), `mem_depth=20` and `num_hidden=60` we can report an overall accuracy of *92%* on the testset after 6 test epochs. The model has *69245* trainable parameters. 

**TODO: Output?**
```
Training epoch: 1 [0/15530 samples(0%)]	Loss: 2.0321
Training epoch: 1 [5120/15530 samples(33%)]	Loss: 1.4145
Training epoch: 1 [10240/15530 samples(66%)]	Loss: 0.9148
Training epoch: 1 [10200/15530 samples(98%)]	Loss: 0.6969

Test Epoch: 1	Accuracy: 1470/2037 (72%)

Training epoch: 2 [0/15530 samples(0%)]	Loss: 0.7829
Training epoch: 2 [5120/15530 samples(33%)]	Loss: 0.5837
Training epoch: 2 [10240/15530 samples(66%)]	Loss: 0.4475
Training epoch: 2 [10200/15530 samples(98%)]	Loss: 0.4325

Test Epoch: 2	Accuracy: 1663/2037 (82%)

Training epoch: 3 [0/15530 samples(0%)]	Loss: 0.4904
Training epoch: 3 [5120/15530 samples(33%)]	Loss: 0.4347
Training epoch: 3 [10240/15530 samples(66%)]	Loss: 0.3757
Training epoch: 3 [10200/15530 samples(98%)]	Loss: 0.2887

Test Epoch: 3	Accuracy: 1799/2037 (88%)

Training epoch: 4 [0/15530 samples(0%)]	Loss: 0.3067
Training epoch: 4 [5120/15530 samples(33%)]	Loss: 0.3714
Training epoch: 4 [10240/15530 samples(66%)]	Loss: 0.3317
Training epoch: 4 [10200/15530 samples(98%)]	Loss: 0.3656

Test Epoch: 4	Accuracy: 1833/2037 (90%)

Training epoch: 5 [0/15530 samples(0%)]	Loss: 0.2681
Training epoch: 5 [5120/15530 samples(33%)]	Loss: 0.2791
Training epoch: 5 [10240/15530 samples(66%)]	Loss: 0.3060
Training epoch: 5 [10200/15530 samples(98%)]	Loss: 0.2535

Test Epoch: 5	Accuracy: 1857/2037 (91%)

Training epoch: 6 [0/15530 samples(0%)]	Loss: 0.2650
Training epoch: 6 [5120/15530 samples(33%)]	Loss: 0.2205
Training epoch: 6 [10240/15530 samples(66%)]	Loss: 0.2518
Training epoch: 6 [10200/15530 samples(98%)]	Loss: 0.2397

Test Epoch: 6	Accuracy: 1870/2037 (92%)
```
The following figure shows the training loss over batch updates


![](https://i.imgur.com/MnlUR9G.png)
*Training loss over batch updates*



The confusion matrix is shown here
![](https://i.imgur.com/5hKZ3Ni.png)
*Confusion Matrix*

The network predicts the five classes very well. There are some mixups between `four` and `go` and between `go` and `no`. These confusions are to be expected due to these keywords' phonetic similarities.



### Task b)

Setting the paramter `groups=num_features` for the `conv1d` layer results in *6845* trainable parameters.

The group parameter in the `Conv1d` function is an integer that defines the groupsize that the input channels are divided into. When we set the groupsize to the `num_features` value, we must set the number of kernels to a value that can be divided by the `group` size. The convolution kernels themself then are divided into the same number of groups and only convolve with their corresponding input-channel group. If we don't set a groupsize (in this case the `group` value is `1`), all input channels are convolved with all kernels. 
Here the output of the training process is shown
<!-- 

In PyTorch, the group parameter in a Conv1d layer determines the number of groups that the input channels are divided into. When the group parameter is set to 1 (the default value), it means that all input channels are convolved with all kernels, resulting in a standard convolution operation.

When the group parameter is set to a value greater than 1, it means that the input channels are divided into groups, and the convolution kernels are also divided into the same number of groups. Each group of kernels convolves with only the channels in its corresponding input group. This results in a depthwise convolution operation.

In terms of trainable parameters, a standard convolution operation has $[\texttt{num_input_channels} \times \texttt{num_kernels}]$ parameters, while a depthwise convolution operation has $[\texttt{num_input_channels} \times  \texttt{num_kernels_per_group}]$ parameters, where $\texttt{num_kernels_per_group}$ is the number of kernels in each group. The number of kernels in the depthwise convolution operation is equal to the number of input channels.
 -->
<!-- For example, consider a Conv1d layer with 3 input channels and 4 output channels, and a group parameter value of 2. This means that the input channels are divided into 2 groups, with each group containing 1.5 channels. The convolution kernels are also divided into 2 groups, with each group containing 2 kernels. In this case, the convolution operation would be a depthwise convolution, and there would be 3 * 2 = 6 trainable parameters. -->

```
Training epoch: 1 [0/15530 samples(0%)]	Loss: 1.7615
Training epoch: 1 [5120/15530 samples(33%)]	Loss: 1.0595
Training epoch: 1 [10240/15530 samples(66%)]	Loss: 0.9649
Training epoch: 1 [10200/15530 samples(98%)]	Loss: 0.9380

Test Epoch: 1	Accuracy: 1329/2037 (65%)

Training epoch: 2 [0/15530 samples(0%)]	Loss: 0.8927
Training epoch: 2 [5120/15530 samples(33%)]	Loss: 0.8683
Training epoch: 2 [10240/15530 samples(66%)]	Loss: 0.7974
Training epoch: 2 [10200/15530 samples(98%)]	Loss: 0.7256

Test Epoch: 2	Accuracy: 1416/2037 (70%)

Training epoch: 3 [0/15530 samples(0%)]	Loss: 0.7183
Training epoch: 3 [5120/15530 samples(33%)]	Loss: 0.8184
Training epoch: 3 [10240/15530 samples(66%)]	Loss: 0.7156
Training epoch: 3 [10200/15530 samples(98%)]	Loss: 0.7136

Test Epoch: 3	Accuracy: 1449/2037 (71%)

Training epoch: 4 [0/15530 samples(0%)]	Loss: 0.7740
Training epoch: 4 [5120/15530 samples(33%)]	Loss: 0.7512
Training epoch: 4 [10240/15530 samples(66%)]	Loss: 0.6959
Training epoch: 4 [10200/15530 samples(98%)]	Loss: 0.7681

Test Epoch: 4	Accuracy: 1460/2037 (72%)

Training epoch: 5 [0/15530 samples(0%)]	Loss: 0.5515
Training epoch: 5 [5120/15530 samples(33%)]	Loss: 0.7007
Training epoch: 5 [10240/15530 samples(66%)]	Loss: 0.6294
Training epoch: 5 [10200/15530 samples(98%)]	Loss: 0.6939

Test Epoch: 5	Accuracy: 1453/2037 (71%)

Training epoch: 6 [0/15530 samples(0%)]	Loss: 0.6634
Training epoch: 6 [5120/15530 samples(33%)]	Loss: 0.7337
Training epoch: 6 [10240/15530 samples(66%)]	Loss: 0.6429
Training epoch: 6 [10200/15530 samples(98%)]	Loss: 0.5862

Test Epoch: 6	Accuracy: 1472/2037 (72%)
```
After 6 trianing epochs we achieve an accuracy on the testset of 72% and a training loss of 0.5862. The loss over batch updates is shown in the following figure
![](https://i.imgur.com/nE20J5T.png)

<!-- Test Accuracy: 72%  
Training Loss: 0.5862 -->
### Task c)

The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:


- a single int – in which case the same value is used for the height and width dimension
- a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension


The model has 69,245 trainable parameters and is implemented as follows:

```python
class KeyWordCNN2d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels, mem_depth, num_hidden=20):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, num_kernels, (num_features, mem_depth))
        self.hidden_layer = nn.Linear(num_kernels, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = torch.mean(x, dim=3).squeeze()
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        output = F.log_softmax(x, dim=-1)
        return output
```

The only difference to the model with a 1d-convolution layer is that now that number of input channels corresponds to `1` because we set a two dimensional `kernel_size` of `(height, width)` with `height=num_features` and `width=mem_depth`. The input data is indeed two-dimensional and has been for task a) as well where we set the number of input channels to the number of resulting features from the mfcc-transform.



```
Training epoch: 1 [0/15530 samples(0%)]	Loss: 3.2825
Training epoch: 1 [5120/15530 samples(33%)]	Loss: 1.3496
Training epoch: 1 [10240/15530 samples(66%)]	Loss: 0.9715
Training epoch: 1 [10200/15530 samples(98%)]	Loss: 0.8921

Test Epoch: 1	Accuracy: 1284/2037 (63%)

Training epoch: 2 [0/15530 samples(0%)]	Loss: 0.8333
Training epoch: 2 [5120/15530 samples(33%)]	Loss: 0.8149
Training epoch: 2 [10240/15530 samples(66%)]	Loss: 0.7455
Training epoch: 2 [10200/15530 samples(98%)]	Loss: 0.5723

Test Epoch: 2	Accuracy: 1599/2037 (78%)

Training epoch: 3 [0/15530 samples(0%)]	Loss: 0.4834
Training epoch: 3 [5120/15530 samples(33%)]	Loss: 0.4771
Training epoch: 3 [10240/15530 samples(66%)]	Loss: 0.4294
Training epoch: 3 [10200/15530 samples(98%)]	Loss: 0.3976

Test Epoch: 3	Accuracy: 1717/2037 (84%)

Training epoch: 4 [0/15530 samples(0%)]	Loss: 0.3603
Training epoch: 4 [5120/15530 samples(33%)]	Loss: 0.3780
Training epoch: 4 [10240/15530 samples(66%)]	Loss: 0.2906
Training epoch: 4 [10200/15530 samples(98%)]	Loss: 0.2634

Test Epoch: 4	Accuracy: 1806/2037 (89%)

Training epoch: 5 [0/15530 samples(0%)]	Loss: 0.2867
Training epoch: 5 [5120/15530 samples(33%)]	Loss: 0.3090
Training epoch: 5 [10240/15530 samples(66%)]	Loss: 0.2777
Training epoch: 5 [10200/15530 samples(98%)]	Loss: 0.2077

Test Epoch: 5	Accuracy: 1852/2037 (91%)

Training epoch: 6 [0/15530 samples(0%)]	Loss: 0.1851
Training epoch: 6 [5120/15530 samples(33%)]	Loss: 0.1668
Training epoch: 6 [10240/15530 samples(66%)]	Loss: 0.3036
Training epoch: 6 [10200/15530 samples(98%)]	Loss: 0.2600

Test Epoch: 6	Accuracy: 1871/2037 (92%)
```

![](https://i.imgur.com/c6phJk8.png)


