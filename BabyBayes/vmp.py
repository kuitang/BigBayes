from __future__ import division
#import abc
import numpy as np
import scipy.special as sp

LOG2PI = np.log(2*np.pi)

class NonConjugate(Exception):
    pass

class Edge(object):
    pass

# PRIOR GETS UPDATED EVERY TIME!

class Distribution(object):
    def __init__(self, *args):
        self.parents = args
        self.update()

    def ex(self):
        if self.observation:
            return self.observation
        else:
            return self._ex()

    def update(self):
        self.prior = self.naturalParameter(*( a.ex() for a in self.parents ))
        self.posterior = self.prior + sum(self.msgsFromChildren)
        self.recoverParams()

        if self.observation:
            self.msgToChild = self.naturalStatistic(self.observation)
        else:
            self.msgToChild = self.calcMsgToChild()

    @classmethod
    def baseMeasure(*args):
        return 0

    def lowerBound(self):
        if self.observation:
            param = self.prior
            norm  = self.priorNorm()
        else:
            param = self.prior - self.posterior
            norm  = self.priorNorm() - self.posteriorNorm()
        return np.dot(param.transpose(), self.msgToChild) + norm

class Delta(Distribution):
    def __init__(self, x):
        self.observation = x

class Gamma(Distribution):
    def __init__(self, shape, rate):
        super(self, Gamma).__init__(shape, rate)

    @classmethod
    def naturalParameter(shape, rate):
        # Equation (A.12)
        return np.array([ -rate, shape - 1 ])

    @classmethod
    def naturalStatistic(x):
        return np.array([ x, x**2 ])

    @classmethod
    def normalizer(shape, rate):
        # Equation (A.12)
        return shape * np.log(rate) - sp.gammaln(shape)

    def recoverParams(self):
        """Called by updatePosterior. Calculates parameters from natural
           parameter."""
        self.shape, self.rate = self.posterior[1] + 1, -self.posterior[0]

    def _ex(self):
        return self.msgToChild[0]
        #return self.shape / self.rate

    def calcMsgToChild(self):
        """Message to children is the expectation of the natural statistic vector.
        Also, messages to all children are the same.
        """
        # Equation (A.13)
        return np.array([ self.shape / self.rate,
                          sp.psi(self.shape) - np.log(self.rate) ])

class Gaussian(Distribution):
    def __init__(self, mean, precision):
        super(self. Gaussian).__init__(mean, precision)

    def _ex(self):
        return self.msgToChild[0]

    def ex2(self):
        return self.msgToChild[1]

    @classmethod
    def naturalParameter(mean, precision):
        # Equation (A.3)
        return np.array([ precision * mean, -precision / 2 ])

    @classmethod
    def naturalStatistic(x):
        return np.array([ x, np.log(x) ])

    @classmethod
    def normalizer(mean, precision):
        # Equation (A.3)
        return 0.5 * (np.log(precision)
                      - precision*mean**2
                      - LOG2PI)

    def recoverParams(self):
        meanTimesPrecision, minusPrecisionOverTwo = self.posterior
        self.precision = -2 * minusPrecisionOverTwo
        self.mean = meanTimesPrecision / self.precision

    def calcMsgToChild(self):
        # Equation (A.4)
        return np.array([ self.mean, self.mean**2 + 1/self.precision ])

    # CORRECT -- verified against VIBES source.
    def msgToParent(self, parent):
        """Child sends each parents the parent's natural parameter. Requires
           coparents to send messages to the child.

           This is the heart of Bayesian conjugate updates.
           """
        if isinstance(parent, Gaussian):
            # Equation (2.25)
            return np.array([ self.precision.ex() * self.ex()
                             -self.precision.ex() / 2 ])
        elif isinstance(parent, Gamma):
            Emean, EmeanSquared = self.msgsFromChildren[0]
            # Equation (2.27)
            return np.array([ -1/2*(self.ex2() + self.mean.ex2() - 2*self.ex()*self.mean.ex()),
                              1/2 ])
        else:
            raise NonConjugate()

def infer(nodes):
    pass
