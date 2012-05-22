from __future__ import division
#import abc
import numpy as np
import scipy.special as sp

np.seterr('raise')
LOG2PI = np.log(2*np.pi)

class NonConjugate(Exception):
    pass

class Edge(object):
    pass

# PRIOR GETS UPDATED EVERY TIME!

class Distribution(object):
    def __init__(self, *args):
        if not hasattr(self, 'observation'):
            self.observation = None
        self.msgsFromChildren = []
        self.parents = args
        self.children = set()

        # Initialize the "moments"
        self.update()

    def ex(self):
        if self.observation is not None:
            return self.observation
        else:
            return self._ex()

    def ex2(self):
        if self.observation is not None:
            return self.observation**2
        else:
            return self._ex2()

    def exLn(self):
        if self.observation is not None:
            return np.log(self.observation)
        else:
            return self._exLn()

    def update(self):
        #print self.msgsFromChildren
        self.updatePrior()
        #print self.prior, sum(self.msgsFromChildren)
        self.posterior = self.prior + sum(self.msgsFromChildren)
        self.recoverParams()
        if self.observation is not None:
            self.msgToChild = self.naturalStatistic(self.observation)
        else:
            self.updateMsgToChild()

    @staticmethod
    def baseMeasure(*args):
        return 0

    def lowerBound(self):
        if self.observation is not None:
            param = self.prior
            norm  = self.priorNorm()
        #    print 'lower bound terms: ', param, self.msgToChild, norm
        else:
            param = self.prior - self.posterior
            norm  = self.priorNorm() - self.posteriorNorm()
        #    print 'lower bound terms: ', param, self.msgToChild, norm
        return np.dot(param.transpose(), self.msgToChild) + norm

class Delta(Distribution):
    def __init__(self, x):
        self.observation = x
        super(Delta, self).__init__()

    def updateMsgToChild(self):
        pass

    def updatePrior(self):
        self.prior = 0

    def recoverParams(self):
        pass

    @staticmethod
    def naturalStatistic(x):
        return np.array([x])

class Gamma(Distribution):
    def __init__(self, shape, rate):
        super(Gamma, self).__init__(shape, rate)

    def updatePrior(self):
        self.prior = self.naturalParameter(*( a.ex() for a in self.parents ))

    def _exLn(self):
        return self.msgToChild[1]

    @staticmethod
    def naturalParameter(shape, rate):
        # Equation (A.12)
        return np.array([ -rate, shape - 1 ])

    @staticmethod
    def naturalStatistic(x):
        return np.array([ x, np.log(x) ])

    @staticmethod
    def normalizer(shape, rate):
        # Equation (A.12)
        return shape * np.log(rate) - sp.gammaln(shape)

    def priorNorm(self):
        shape, rate = self.parents
        return shape.ex() * rate.exLn() - sp.gammaln(shape.ex())

    def posteriorNorm(self):
        return self.normalizer(self.shape, self.rate)

    def recoverParams(self):
        """Called by updatePosterior. Calculates parameters from natural
           parameter."""
        self.shape, self.rate = self.posterior[1] + 1, -self.posterior[0]

    def _ex(self):
        return self.msgToChild[0]
        #return self.shape / self.rate

    def updateMsgToChild(self):
        """Message to children is the expectation of the natural statistic vector.
        Also, messages to all children are the same.
        """
        # Equation (A.13)
        self.msgToChild = np.array([ self.shape / self.rate,
                                    sp.psi(self.shape) - np.log(self.rate) ])

class Gaussian(Distribution):
    def __init__(self, mean, precision):
        super(Gaussian, self).__init__(mean, precision)

    def _ex(self):
        return self.msgToChild[0]

    def _ex2(self):
        return self.msgToChild[1]

    def updatePrior(self):
        self.prior = self.naturalParameter(*( a.ex() for a in self.parents ))

    @staticmethod
    def naturalParameter(mean, precision):
        # Equation (A.3)
        return np.array([ precision * mean, -precision / 2 ])

    @staticmethod
    def naturalStatistic(x):
        return np.array([ x, x**2 ])

    @staticmethod
    def normalizer(mean, precision):
        # Equation (A.3)
        return 0.5 * (np.log(precision)
                      - precision*mean**2
                      - LOG2PI)

    def priorNorm(self):
        mean, precision = self.parents
        return 0.5 * (precision.exLn()
                      - precision.ex() * mean.ex2()
                      - LOG2PI)

    def posteriorNorm(self):
        return self.normalizer(self.mean, self.precision)

    def recoverParams(self):
        meanTimesPrecision, minusPrecisionOverTwo = self.posterior
        self.precision = -2 * minusPrecisionOverTwo
        self.mean = meanTimesPrecision / self.precision

    def updateMsgToChild(self):
        # Equation (A.4)
        self.msgToChild = np.array([ self.mean, self.mean**2 + 1/self.precision ])

    # CORRECT -- verified against VIBES source.
    def msgToParent(self, parent):
        """Child sends each parents the parent's natural parameter. Requires
           coparents to send messages to the child.

           This is the heart of Bayesian conjugate updates.
           """
        if isinstance(parent, Gaussian):
            # Equation (2.25)
            precision = self.parents[1]
            return np.array([ precision.ex() * self.ex(),
                             -precision.ex() / 2 ])
        elif isinstance(parent, Gamma):
            # Equation (2.27)
            mean = self.parents[0]
            return np.array([ -1/2*(self.ex2() + mean.ex2() - 2*self.ex()*mean.ex()),
                              1/2 ])
        else:
            raise NonConjugate()

def setChildren(root):
    for p in root.parents:
        if root not in p.children:
            p.children.add(root)
            setChildren(p)

def setAllChildren(nodes):
    for root in nodes:
        setChildren(root)

# Random scheduler.
def run(nodes, tol=1e-3):
    oldlb, lb = np.infty, 0.0
    while np.abs(oldlb - lb) > tol:
        oldlb = lb
        lb = 0.0
        for node in nodes:
            for p in node.parents:
                p.updateMsgToChild()
            for c in node.children:
                coparents = ( cp for cp in c.parents if node != cp )
                for cp in coparents:
                    cp.updateMsgToChild()
            node.msgsFromChildren = [ c.msgToParent(node) for c in node.children ]
            node.update()
            lb += node.lowerBound()
        print "Lower bound: %g"%lb

if __name__ == '__main__':
    # Incorrect priors
    mu   = Gaussian(Delta(-20), Delta(1))
    beta = Gamma(Delta(.01), Delta(.01))
    X    = [ Gaussian(mu, beta) for _ in xrange(10000) ]
    # We want a mean-2, variance-16 variate. Let's see how close we get.
    for x in X:
        x.observation = np.random.normal(2, 4)

    nodes = [mu, beta] + X
    setAllChildren(nodes)
    run(nodes)

    print "Posterior mu mean: %g"%(mu.ex())
    print "Posterior beta mean: %g"%(beta.ex())

