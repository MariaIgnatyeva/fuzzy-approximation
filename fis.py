import numpy as np
import scipy.integrate as integrate
import numexpr as ne


class Params:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and \
               self.c == other.c and self.d == other.d

    def __str__(self):
        return str(self.a) + ', ' + str(self.b) + ', ' + str(self.c) + ', ' + str(self.d)

    def areValid(self):
        return self.a <= self.b <= self.c <= self.d and \
               not self.a == self.b == self.c == self.d

    def areInBounds(self, bounds):
        return not (self.a >= bounds[1] or self.d <= bounds[0])

    def getBoundParams(self, bounds):
        a = max(self.a, bounds[0])
        d = min(self.d, bounds[1])
        return a, d


class MembershipFunction:
    def __init__(self, params):
        self.params = params
        self.function = np.vectorize(self.getFunction, otypes=[float])

    def getFunction(self, x):
        if x < self.params.a or x > self.params.d:
            return 0

        if x >= self.params.a and x < self.params.b:
            return (x - self.params.a) / (self.params.b - self.params.a)

        if x >= self.params.b and x <= self.params.c:
            return 1

        return (self.params.d - x) / (self.params.d - self.params.c)

    def getModifiedFunction(self, x, hedge):
        if hedge == '':
            return self.getFunction(x)

        if hedge == 'very':
            return self.very(x)

        if hedge == 'somewhat':
            return self.somewhat(x)

        if hedge == 'definitely':
            return self.definitely(x)

        if hedge == 'generally':
            return self.generally(x)

    def very(self, x):
        return self.getFunction(x) ** 2

    def somewhat(self, x):
        return self.getFunction(x) ** 0.5

    def definitely(self, x):
        f = self.getFunction(x)
        if f <= 0.5:
            return 2 * f ** 2
        return 1 - 2 * (1 - f) ** 2

    def generally(self, x):
        f = self.getFunction(x)
        if f <= 0.5:
            return (f / 2) ** 0.5
        return 1 - ((1 - f) / 2) ** 0.5

    def __eq__(self, other):
        return self.params == other.params


class Rule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def __eq__(self, other):
        return self.antecedent == other.antecedent

    def toMamdaniStr(self):
        return 'If x is ' + self.antecedent + ', then f(x) is ' + self.consequent

    def toSugenoStr(self):
        return 'If x is ' + str(self.antecedent) + ', then f(x) is ' + \
               str(self.consequent[0]) + '*x' + '{:+.2f}'.format(self.consequent[1])

    def getConstsFromSugenoString(self, s):
        ind1 = s.rfind('f(x) is ') + len('f(x) is ')
        ind2 = s.rfind('*x')
        const = float(s[ind2 + 2:])
        constX = float(s[ind1:ind2])
        return constX, const

    def getPart(self, var):
        if var.lower() == 'input':
            return self.antecedent

        return self.consequent


class Implication:
    @staticmethod
    def minOperator(const, func):
        return lambda x: min(const, func(x))

    @staticmethod
    def productOperator(const, func):
        return lambda x: const * func(x)


class Aggregation:
    @staticmethod
    def maxOperator(functions, x):
        max = functions[0](x)
        for i in range(1, len(functions)):
            res = functions[i](x)
            if res > max:
                max = res
        return max

    @staticmethod
    def sumOperator(functions, x):
        res = functions[0](x)
        for i in range(1, len(functions)):
            res = res + functions[i](x) - res * functions[i](x)
        return res


class Defuzzification:
    @staticmethod
    def centroid(func, domain):
        return integrate.quad(lambda y: func(y) * y, domain[0], domain[-1])[0] / \
               integrate.quad(func, domain[0], domain[-1])[0]

    @staticmethod
    def meanOfMaximum(func, domain):
        weights = np.vectorize(func, otypes=[float])(domain)
        max = np.max(weights)
        maxInd = np.argwhere(weights == max)
        return (domain[maxInd[0]] + domain[maxInd[-1]]) / 2


class Error:
    @staticmethod
    def smape(true, approximated):
        n = len(true)
        notEqual = np.argwhere(approximated != true)
        approximated = approximated[notEqual]
        true = true[notEqual]
        return {'smape': 1 / n * np.sum(
            np.fabs(approximated - true) / (np.fabs(true) / 2 + np.fabs(approximated) / 2))}

    @staticmethod
    def mae(true, approximated):
        return {'mae': 1 / len(true) * np.sum(np.fabs(approximated - true))}

    @staticmethod
    def rmse(true, approximated):
        return {'rmse': 1 * np.sqrt(np.sum((approximated - true) ** 2) / len(true))}


class FIS:
    def __init__(self):
        self.func = None
        self.inputType = None
        self.model = 0
        self.funcBound = {'input': [0, 0], 'output': [0, 0]}
        self.trueValues = None
        self.approximatedValues = None
        self.clearMFuncAndRules()

    def addMFunc(self, name, params, var):
        self.membershipFunc[var.lower()][name] = MembershipFunction(params)

    def deleteMFunc(self, name, var):
        func = self.membershipFunc[var.lower()].pop(name)
        del func

    def clearOutputMFuncs(self):
        self.membershipFunc['output'] = {}

    def clearMFuncAndRules(self):
        self.membershipFunc = {'input': {}, 'output': {}}
        self.rules = []

    def getMFuncNames(self, var):
        return self.membershipFunc[var.lower()].keys()

    def getMFunc(self, name, var):
        return self.membershipFunc[var.lower()][name]

    def hasMFuncName(self, name, var):
        return name.lower() in map(str.lower, self.membershipFunc[var.lower()].keys())

    def hasMFuncParams(self, params, var):
        name = ''
        for key, value in self.membershipFunc[var.lower()].items():
            if value.params == params:
                name = key
        return name != '', name

    def getRule(self, ind):
        return self.rules[ind]

    def getRulesIndWithName(self, name, var):
        indices = []
        for i in range(len(self.rules)):
            if self.rules[i].getPart(var) == name:
                indices.append(i)
        return indices

    def areRulesEmpty(self):
        return len(self.rules) == 0

    def hasRule(self, rule, ind=None):
        for i in range(len(self.rules)):
            if i == ind:
                continue
            if rule == self.rules[i]:
                return True
        return False

    def addRule(self, rule):
        self.rules.append(rule)

    def updateRule(self, ind, rule):
        self.rules[ind] = rule

    def deleteRule(self, ind):
        rule = self.rules.pop(ind)
        del rule

    def fuzzifyInput(self, rule, input, hedge=None):
        degree = self.membershipFunc['input'][rule.antecedent].getFunction(input)
        if degree > 0:
            self.degrees = np.append(self.degrees, degree)
            self.firedRules = np.append(self.firedRules, rule)

    def findIntersection(self, rule, input, hedge):
        antecedentF = self.membershipFunc['input'][rule.antecedent]
        inputF = self.membershipFunc['input'][input]

        if antecedentF.params == inputF.params or (
                antecedentF.params.a <= inputF.params.a and antecedentF.params.d >= inputF.params.d) or (
                antecedentF.params.a >= inputF.params.a and antecedentF.params.d <= inputF.params.d):
            self.degrees = np.append(self.degrees, 1)
            self.firedRules = np.append(self.firedRules, rule)
            return

        left = None
        right = None

        if antecedentF.params.d > inputF.params.a and antecedentF.params.a < inputF.params.a:
            left = inputF.params.a
            right = antecedentF.params.d
        elif antecedentF.params.a < inputF.params.d and antecedentF.params.d > inputF.params.d:
            left = antecedentF.params.a
            right = inputF.params.d

        if left is not None and right is not None:
            arguments = np.arange(left, right + 0.001, 0.01)
            arguments = np.around(arguments, 2)

            antecedentValues = np.vectorize(antecedentF.getModifiedFunction, otypes=[float])(arguments, hedge)
            inputValues = np.vectorize(inputF.getModifiedFunction, otypes=[float])(arguments, hedge)

            degree = np.max(inputValues[inputValues == antecedentValues])
            self.degrees = np.append(self.degrees, degree)
            self.firedRules = np.append(self.firedRules, rule)

    def applyImplication(self, rule, degree):
        return self.implication(degree, self.membershipFunc['output'][rule.consequent].function)

    def aggregateConclusions(self, conclusions):
        return lambda x: self.aggregation(conclusions, x)

    def approximateInputMamdani(self, input, hedge):
        self.degrees = np.array([])
        self.firedRules = np.array([])

        np.vectorize(self.getDegree, otypes=[float])(self.rules, input, hedge)
        if len(self.firedRules) == 0:
            return None

        conclusions = np.vectorize(self.applyImplication)(self.firedRules, self.degrees)

        aggregationRes = self.aggregateConclusions(conclusions)

        defuzRes = self.defuzzification(aggregationRes, self.y)
        return defuzRes

    def approximateInputSugeno(self, input, hedge):
        self.degrees = np.array([])
        self.firedRules = np.array([])

        np.vectorize(self.getDegree, otypes=[float])(self.rules, input, hedge)
        if len(self.firedRules) == 0:
            return None

        crispOutput = 0
        degreesSum = np.sum(self.degrees)
        for i in range(len(self.firedRules)):
            constX = self.firedRules[i].consequent[0]
            const = self.firedRules[i].consequent[1]
            crispOutput = (crispOutput + self.degrees[i] * (constX * input + const)) / degreesSum

        return crispOutput

    def start(self, input, operators, defuzzification, progress):
        if self.model == 0:
            if operators == 'min-max':
                self.implication = Implication.minOperator
                self.aggregation = Aggregation.maxOperator
            else:
                self.implication = Implication.productOperator
                self.aggregation = Aggregation.sumOperator

            if defuzzification == 'centroid':
                self.defuzzification = Defuzzification.centroid
            else:
                self.defuzzification = Defuzzification.meanOfMaximum

            self.approximateInput = self.approximateInputMamdani

        else:
            self.approximateInput = self.approximateInputSugeno

        self.y = np.arange(self.funcBound['output'][0], self.funcBound['output'][1] + 0.001, 0.01)
        self.y = np.around(self.y, 2)

        self.getDegree = self.fuzzifyInput
        hedge = None

        if self.inputType == 0:
            self.x = [input]

        elif self.inputType == 1:
            self.x = [input[1]]
            self.getDegree = self.findIntersection
            hedge = input[0]

        else:
            self.x = np.arange(input[0], input[1] + 0.001, 0.01)
            self.x = np.around(self.x, 2)

        progress.show()
        progress.setValue(1)

        self.approximatedValues = np.empty(len(self.x))

        for i in range(len(self.x)):
            res = self.approximateInput(self.x[i], hedge)

            if res is None:
                progress.close()
                return False, self.x[i]

            self.approximatedValues[i] = res

            if progress.wasCanceled():
                return False, None
            progress.setValue((i + 1) / len(self.x) * 100)

        progress.close()

        if self.inputType == 1:
            a, d = self.membershipFunc['input'][input[1]].params.getBoundParams(self.funcBound['input'])
            self.x = np.arange(a, d + 0.001, 0.01)
            self.x = np.around(self.x, 2)
            self.approximatedValues = np.full(len(self.x), self.approximatedValues[0])

        self.trueInd = int((round(self.x[0], 2) - self.funcBound['input'][0]) / 0.01)

        errors = {}
        for method in [Error.mae, Error.rmse, Error.smape]:
            errors.update(method(np.array(self.trueValues[self.trueInd:self.trueInd + len(self.x)]),
                                 self.approximatedValues))

        return True, errors
