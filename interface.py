from PyQt5.QtGui import QIcon

import fis
import sys
import numexpr as ne
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as lines

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QPushButton, QComboBox, QStackedLayout, QCheckBox,
                             QSizePolicy, QLineEdit, QHBoxLayout, QVBoxLayout, QGroupBox, QWidget, QLabel,
                             QFormLayout, QScrollArea, QListWidget, QDoubleSpinBox, QMessageBox, QDesktopWidget,
                             QProgressDialog, QRadioButton, QMainWindow, QAction, QDialog, QTextEdit, QToolButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg \
    as FigureCanvas
from matplotlib.figure import Figure

fuzzySys = fis.FIS()


class Window(QMainWindow):
    plotCanvas = None
    stackedLayout = QStackedLayout()

    def __init__(self):
        super().__init__()
        self.title = '2D-function Approximation Using Fuzzy Rule Base'
        self.setSize()
        self.initUI()

    def setSize(self):
        desktop = QDesktopWidget()
        size = desktop.screenGeometry()
        self.left = size.width() / 9
        self.top = size.height() / 20
        self.width = size.width() - self.left * 2
        self.height = size.height() - self.top * 2 - 100

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        helpAction = QAction('&Help', self)
        helpAction.setShortcut('F1')
        helpAction.triggered.connect(self.showHelp)

        aboutAction = QAction('&About', self)
        aboutAction.triggered.connect(self.showAbout)

        mainMenu = self.menuBar()
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(helpAction)
        helpMenu.addAction(aboutAction)

        Window.plotCanvas = PlotCanvas()

        fourthStepForm = FourthStepForm()
        thirdStepForm = ThirdStepForm(fourthStepForm)
        secondStepForm = SecondStepForm(thirdStepForm)
        firstStepForm = FirstStepForm(secondStepForm)
        Window.stackedLayout.addWidget(firstStepForm)
        Window.stackedLayout.addWidget(secondStepForm)
        Window.stackedLayout.addWidget(thirdStepForm)
        Window.stackedLayout.addWidget(fourthStepForm)
        Window.stackedLayout.setCurrentIndex(0)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(Window.plotCanvas, stretch=5)
        mainLayout.addLayout(Window.stackedLayout, stretch=3)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.show()

    def showHelp(self):
        help = Help()
        help.exec()

    def showAbout(self):
        about = About()
        about.exec()

    @staticmethod
    def buttonBackOnClick():
        Window.stackedLayout.setCurrentIndex(Window.stackedLayout.currentIndex() - 1)

    @staticmethod
    def buttonNextOnClick():
        Window.stackedLayout.setCurrentIndex(Window.stackedLayout.currentIndex() + 1)


class FirstStepForm(QWidget):
    def __init__(self, nextStep):
        super().__init__()
        self.nextStep = nextStep
        self.prevFunc = None
        self.prevXLeft = None
        self.prevXRight = None
        self.initUI()

    def initUI(self):
        self.buttonPlot = QPushButton('Plot')
        self.buttonPlot.setAutoDefault(True)
        self.buttonPlot.clicked.connect(self.buttonPlotOnClick)

        self.createFunctionFormGroupBox()

        self.createModelFormGroupBox()

        self.buttonNext = QPushButton('Next >')
        self.buttonNext.setAutoDefault(True)
        self.buttonNext.setEnabled(False)
        self.buttonNext.clicked.connect(self.buttonNextOnClick)

        layout = QVBoxLayout()
        layout.addWidget(self.functionFormGroupBox)
        layout.addWidget(self.buttonPlot)
        self.buttonPlot.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.setAlignment(self.buttonPlot, Qt.AlignRight)
        layout.addWidget(self.modelFormGroupBox)
        layout.setAlignment(self.modelFormGroupBox, Qt.AlignTop)
        layout.addWidget(self.buttonNext)
        self.buttonNext.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.setAlignment(self.buttonNext, Qt.AlignRight)

        self.setLayout(layout)

    def createFunctionFormGroupBox(self):
        self.functionFormGroupBox = QGroupBox('Function for approximation')

        formLayout = QFormLayout()
        formLayout.setLabelAlignment(Qt.AlignRight)

        self.editorFunc = QLineEdit()
        self.editorFunc.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.infoButton = QToolButton()
        self.infoButton.setStyleSheet('background-color: transparent')
        self.infoButton.setEnabled(False)
        self.infoButton.setIcon(QIcon('./files/info.png'))
        self.infoButton.setToolTip('Function depends on the argument x\n' +
                                   '\nSupported operators:\n' +
                                   '+, -, *, /\n' +
                                   '^, ** - exponentiation\n' +
                                   '% - the modulo operation\n' +
                                   '\nSupported functions:\n' +
                                   'sin, cos, tan, arcsin, arccos, arctan\n' +
                                   'sinh, cosh, tanh, arcsinh, arccosh, arctanh\n' +
                                   'arctan2(a, b) – trigonometric inverse tangent of a/b\n' +
                                   'log, log10, log1p - natural, base-10 and log(1+x) logarithms\n' +
                                   'exp, expm1 – exponential and exponential minus one\n' +
                                   'sqrt – square root\n' +
                                   'abs – absolute value\n' +
                                   '\nThe integer and decimal parts of a number are separated by a dot\n')

        hBox = QHBoxLayout()
        hBox.addWidget(self.editorFunc)
        hBox.addWidget(self.infoButton)
        formLayout.addRow('f(x):', hBox)

        editorDomain = QHBoxLayout()
        self.editorDomainLeft = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        self.editorDomainRight = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        min = QLabel('min:')
        min.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        max = QLabel('max:')
        max.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        editorDomain.addWidget(min)
        editorDomain.addWidget(self.editorDomainLeft)
        editorDomain.addWidget(max)
        editorDomain.addWidget(self.editorDomainRight)
        formLayout.addRow('D(x):', editorDomain)

        layout = QVBoxLayout()
        layout.addLayout(formLayout)

        self.functionFormGroupBox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.functionFormGroupBox.setLayout(layout)

    def createModelFormGroupBox(self):
        self.modelFormGroupBox = QGroupBox('FIS model')

        self.mamdaniModel = QRadioButton('Mamdani model')
        self.mamdaniModel.setChecked(True)
        self.sugenoModel = QRadioButton('Takagi-Sugeno-Kang model')

        layout = QVBoxLayout()
        layout.addWidget(self.mamdaniModel)
        layout.addWidget(self.sugenoModel)

        self.modelFormGroupBox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.modelFormGroupBox.setLayout(layout)

    def buttonPlotOnClick(self):
        func = self.editorFunc.text()
        func = func.replace(' ', '').replace('^', '**').lower()

        xLeft = self.editorDomainLeft.value()
        xRight = self.editorDomainRight.value()
        if not self.isDomainValid(xLeft, xRight):
            WarningMessage('Domain params are not valid: min >= max')
            return

        self.changed = False
        if xLeft != self.prevXLeft or xRight != self.prevXRight or func != self.prevFunc:
            self.changed = True
            if Window.plotCanvas.plotFunction((xLeft, xRight), func):
                self.prevFunc = func
                self.prevXLeft = xLeft
                self.prevXRight = xRight
                self.buttonNext.setEnabled(True)

    def isDomainValid(self, left, right):
        return left < right

    def setNextStepsParams(self):
        fuzzySys.clearMFuncAndRules()
        self.nextStep.clearFunctionsLists()
        self.nextStep.nextStep.clearRules()
        self.nextStep.nextStep.nextStep.setInitialState()
        self.nextStep.nextStep.nextStep.clearResults()

    def buttonNextOnClick(self):
        fuzzySys.model = 0 if self.mamdaniModel.isChecked() else 1
        if fuzzySys.model == 0:
            self.nextStep.createOutputBox()
        else:
            self.nextStep.removeOutputBox()

        if self.changed:
            self.setNextStepsParams()
            self.changed = not self.changed

        Window.buttonNextOnClick()


class SecondStepForm(QWidget):
    def __init__(self, next):
        super().__init__()
        self.nextStep = next
        self.inputBox = None
        self.outputBox = None
        self.initUI()

    def initUI(self):
        self.groupBox = QGroupBox('Membership functions')
        self.vBox = QVBoxLayout()
        self.inputBox = MembershipFunctionsBox('Input', self.nextStep)
        self.vBox.addWidget(self.inputBox)
        self.createOutputBox()
        self.groupBox.setLayout(self.vBox)

        self.buttonNext = QPushButton('Next >')
        self.buttonNext.setAutoDefault(True)
        self.buttonNext.clicked.connect(self.buttonNextOnClick)
        self.buttonBack = QPushButton('< Back')
        self.buttonBack.setAutoDefault(True)
        self.buttonBack.clicked.connect(Window.buttonBackOnClick)

        navigateButtons = QHBoxLayout()
        navigateButtons.addWidget(self.buttonBack)
        self.buttonBack.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        navigateButtons.setAlignment(self.buttonBack, Qt.AlignRight)
        navigateButtons.addWidget(self.buttonNext)
        self.buttonNext.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        formLayout = QVBoxLayout()
        formLayout.addWidget(self.groupBox)
        formLayout.addLayout(navigateButtons)

        self.setLayout(formLayout)

    def clearFunctionsLists(self):
        self.inputBox.clearFunctionsList()
        if fuzzySys.model == 0:
            self.outputBox.clearFunctionsList()

    def removeOutputBox(self):
        if self.outputBox is not None:
            self.vBox.removeWidget(self.outputBox)
            self.outputBox.deleteLater()
            self.outputBox = None
            fuzzySys.clearOutputMFuncs()
            Window.plotCanvas.deleteOutputMFPlots()

            self.nextStep.clearRules()

    def createOutputBox(self):
        if self.outputBox is None:
            self.outputBox = MembershipFunctionsBox('Output', self.nextStep)
            self.vBox.addWidget(self.outputBox)

            self.nextStep.clearRules()

    def buttonNextOnClick(self):
        if fuzzySys.model == 0:
            self.nextStep.makeMamdaniTypeForm()
        else:
            self.nextStep.makeSugenoTypeForm()

        self.nextStep.setNamesSet()
        Window.buttonNextOnClick()


class MembershipFunctionsBox(QGroupBox):
    def __init__(self, title, rulesForm):
        super().__init__(title)
        self.rulesForm = rulesForm
        self.prevMFuncs = None
        self.initUI()

    def initUI(self):
        self.createScrollArea()
        self.createFunctionForm()
        self.buttonAdd = QPushButton('Add')
        self.buttonAdd.setAutoDefault(True)
        self.buttonAdd.clicked.connect(self.buttonAddOnClick)
        self.buttonChange = QPushButton('Change')
        self.buttonChange.setAutoDefault(True)
        self.buttonChange.setEnabled(False)
        self.buttonChange.clicked.connect(self.buttonChangeOnClick)
        self.buttonDelete = QPushButton('Delete')
        self.buttonDelete.setAutoDefault(True)
        self.buttonDelete.setEnabled(False)
        self.buttonDelete.clicked.connect(self.buttonDeleteOnClick)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignRight)
        buttons.addWidget(self.buttonAdd)
        buttons.addWidget(self.buttonChange)
        buttons.addWidget(self.buttonDelete)

        layout = QVBoxLayout()
        layout.addWidget(self.scrollArea)
        layout.addLayout(self.functionForm)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def createScrollArea(self):
        self.functionsList = QListWidget()
        self.functionsList.itemClicked.connect(self.functionItemOnClick)
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.functionsList)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def createFunctionForm(self):
        self.functionForm = QFormLayout()
        self.functionForm.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)

        self.editorName = QLineEdit()
        self.editorName.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.functionForm.addRow('Name:', self.editorName)

        a = QLabel('a:')
        a.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        b = QLabel('b:')
        b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        c = QLabel('c:')
        c.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        d = QLabel('d:')
        d.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.a = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        self.b = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        self.c = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        self.d = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
        editorParams = QVBoxLayout()
        hBox = QHBoxLayout()
        hBox.addWidget(a)
        hBox.addWidget(self.a)
        hBox.addWidget(b)
        hBox.addWidget(self.b)
        editorParams.addLayout(hBox)
        hBox = QHBoxLayout()
        hBox.addWidget(c)
        hBox.addWidget(self.c)
        hBox.addWidget(d)
        hBox.addWidget(self.d)
        editorParams.addLayout(hBox)
        self.functionForm.addRow('Params:', editorParams)

    def inverseButtonsState(self):
        self.buttonAdd.setEnabled(not self.buttonAdd.isEnabled())
        self.buttonChange.setEnabled(not self.buttonChange.isEnabled())
        self.buttonDelete.setEnabled(not self.buttonDelete.isEnabled())

    def clearFunctionsList(self):
        self.functionsList.clear()

    def functionItemOnClick(self):
        if self.buttonAdd.isEnabled():
            self.currentName = self.getNameFromText(self.functionsList.currentItem().text())
            self.editorName.setText(self.currentName)

            params = fuzzySys.getMFunc(self.currentName, self.title()).params
            self.a.setValue(params.a)
            self.b.setValue(params.b)
            self.c.setValue(params.c)
            self.d.setValue(params.d)
        else:
            self.emptyFields()

        self.inverseButtonsState()

    def buttonAddOnClick(self):
        self.addMembershipFunc()

    def buttonChangeOnClick(self):
        fuzzySys.deleteMFunc(self.currentName, self.title())
        ind = self.functionsList.currentRow()
        Window.plotCanvas.deleteMFPlot(self.title(), ind)
        res = self.addMembershipFunc(ind)

        if not fuzzySys.areRulesEmpty():
            rulesInd = fuzzySys.getRulesIndWithName(self.getNameFromText(self.functionsList.currentItem().text()),
                                                    self.title())
            for ind in rulesInd:
                Window.plotCanvas.changeRulePatch(ind)

        if res:
            self.inverseButtonsState()

    def buttonDeleteOnClick(self):
        if not fuzzySys.areRulesEmpty():
            rulesInd = fuzzySys.getRulesIndWithName(self.getNameFromText(self.functionsList.currentItem().text()),
                                                    self.title())
            for ind in rulesInd:
                self.rulesForm.deleteRule(ind)

        self.deleteMembershipFunc()

        self.emptyFields()
        self.inverseButtonsState()

    def addMembershipFunc(self, ind=None):
        name = self.editorName.text().strip()
        if len(name) == 0:
            WarningMessage('Invalid name for membership function')
            return False
        if fuzzySys.hasMFuncName(name, self.title()):
            WarningMessage('Membership function with name ' + name +
                           ' for ' + self.title().lower() + ' already exists')
            return False

        params = fis.Params(self.a.value(), self.b.value(), self.c.value(), self.d.value())
        if not params.areValid():
            WarningMessage('Params ' + str(params) + ' must be in non-decreasing order')
            return False
        if not params.areInBounds(fuzzySys.funcBound[self.title().lower()]):
            WarningMessage('All params are out of the function domain')
            return False
        hasParams, existedName = fuzzySys.hasMFuncParams(params, self.title())
        if hasParams:
            WarningMessage('Membership function called ' + existedName + ' already has params ' + str(params))
            return False

        fuzzySys.addMFunc(name, params, self.title())

        item = name + ' Params: ' + str(params)
        if ind is None:
            self.functionsList.addItem(item)
        else:
            self.functionsList.item(ind).setText(item)

        Window.plotCanvas.plotMF(name, self.title(), ind)

        self.emptyFields()

        return True

    def deleteMembershipFunc(self):
        fuzzySys.deleteMFunc(self.currentName, self.title())
        Window.plotCanvas.deleteMFPlot(self.title(), self.functionsList.currentRow())
        item = self.functionsList.takeItem(self.functionsList.currentRow())
        del item

        if self.functionsList.count() > 0:
            self.currentName = self.getNameFromText(self.functionsList.currentItem().text())
        else:
            self.currentName = None

    def areMFuncsChanged(self):
        return self.prevMFuncs != fuzzySys.membershipFunc[self.title()]

    def getNameFromText(self, text):
        return text[: text.find(' Params: ')]

    def emptyFields(self):
        self.editorName.clear()
        self.a.setValue(0)
        self.b.setValue(0)
        self.c.setValue(0)
        self.d.setValue(0)


class ThirdStepForm(QWidget):
    def __init__(self, nextStep):
        super().__init__()
        self.nextStep = nextStep
        self.initUI()

    def initUI(self):
        self.createGroupBox()

        self.buttonNext = QPushButton('Next >')
        self.buttonNext.setAutoDefault(True)
        self.buttonNext.clicked.connect(self.buttonNextOnClick)
        self.buttonBack = QPushButton('< Back')
        self.buttonNext.setAutoDefault(True)
        self.buttonBack.clicked.connect(Window.buttonBackOnClick)
        navigateButtons = QHBoxLayout()
        navigateButtons.addWidget(self.buttonBack)
        self.buttonBack.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        navigateButtons.setAlignment(self.buttonBack, Qt.AlignRight)
        navigateButtons.addWidget(self.buttonNext)
        self.buttonNext.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        formLayout = QVBoxLayout()
        formLayout.addWidget(self.groupBox)
        formLayout.addLayout(navigateButtons)

        self.setLayout(formLayout)

    def createGroupBox(self):
        self.groupBox = QGroupBox('Rules')

        self.createScrollArea()
        self.createRuleForm()
        self.createEditButtons()

        layout = QVBoxLayout()
        layout.addWidget(self.scrollArea)
        layout.addLayout(self.ruleForm)
        layout.addLayout(self.editButtons)
        self.groupBox.setLayout(layout)

        self.groupBox.setLayout(layout)

    def createScrollArea(self):
        self.rulesList = QListWidget()
        self.rulesList.itemClicked.connect(self.ruleItemOnClick)
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.rulesList)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def createRuleForm(self):
        self.ruleForm = QHBoxLayout()
        self.ruleForm.setContentsMargins(0, 20, 0, 0)

        antecedent = QLabel('If x is')
        antecedent.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ruleForm.addWidget(antecedent)

        self.comboBoxes = {'input': QComboBox(), 'output': QComboBox()}

        self.ruleForm.addWidget(self.comboBoxes['input'])

        consequent = QLabel(', then f(x) is')
        consequent.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ruleForm.addWidget(consequent)

        self.ruleForm.addWidget(self.comboBoxes['output'])

        self.constX = None

    def setNamesSet(self):
        for key, value in self.comboBoxes.items():
            self.updateComboBox(key)

    def updateComboBox(self, var):
        if self.comboBoxes[var] is not None:
            self.comboBoxes[var].clear()
            names = fuzzySys.getMFuncNames(var)
            for name in names:
                self.comboBoxes[var].addItem(name)

    def makeSugenoTypeForm(self):
        if self.comboBoxes['output'] is not None:
            self.ruleForm.removeWidget(self.comboBoxes['output'])
            self.comboBoxes['output'].deleteLater()
            self.comboBoxes['output'] = None

            self.constX = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
            self.const = DoubleSpinBox(-sys.maxsize - 1, sys.maxsize)
            self.xLabel = QLabel('x +')
            self.xLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            self.ruleForm.addWidget(self.constX)
            self.ruleForm.addWidget(self.xLabel)
            self.ruleForm.addWidget(self.const)

    def makeMamdaniTypeForm(self):
        if self.constX is not None:
            self.ruleForm.removeWidget(self.constX)
            self.ruleForm.removeWidget(self.xLabel)
            self.ruleForm.removeWidget(self.const)
            self.constX.deleteLater()
            self.xLabel.deleteLater()
            self.const.deleteLater()
            self.constX = None
            self.xLabel = None
            self.const = None

            self.comboBoxes['output'] = QComboBox()
            self.ruleForm.addWidget(self.comboBoxes['output'])

    def createEditButtons(self):
        self.buttonAdd = QPushButton('Add')
        self.buttonAdd.setAutoDefault(True)
        self.buttonAdd.clicked.connect(self.buttonAddOnClick)
        self.buttonChange = QPushButton('Change')
        self.buttonChange.setAutoDefault(True)
        self.buttonChange.setEnabled(False)
        self.buttonChange.clicked.connect(self.buttonChangeOnClick)
        self.buttonDelete = QPushButton('Delete')
        self.buttonDelete.setAutoDefault(True)
        self.buttonDelete.setEnabled(False)
        self.buttonDelete.clicked.connect(self.buttonDeleteOnClick)

        self.editButtons = QHBoxLayout()
        self.editButtons.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.editButtons.addWidget(self.buttonAdd)
        self.editButtons.addWidget(self.buttonChange)
        self.editButtons.addWidget(self.buttonDelete)

    def inverseButtonsState(self):
        self.buttonAdd.setEnabled(not self.buttonAdd.isEnabled())
        self.buttonChange.setEnabled(not self.buttonChange.isEnabled())
        self.buttonDelete.setEnabled(not self.buttonDelete.isEnabled())

    def ruleItemOnClick(self):
        if self.buttonAdd.isEnabled():
            self.currentIndex = self.rulesList.currentRow()

            rule = fuzzySys.getRule(self.currentIndex)
            self.comboBoxes['input'].setCurrentIndex(self.comboBoxes['input'].findText(rule.antecedent))
            if self.comboBoxes['output'] is not None:
                self.comboBoxes['output'].setCurrentIndex(self.comboBoxes['output'].findText(rule.consequent))
            else:
                constX, const = rule.getConstsFromSugenoString(self.rulesList.currentItem().text())
                self.constX.setValue(constX)
                self.const.setValue(const)

        else:
            self.emptyFields()

        self.inverseButtonsState()

    def buttonAddOnClick(self):
        self.addRule()

    def buttonChangeOnClick(self):
        self.changeRule()

    def buttonDeleteOnClick(self):
        self.deleteRule()
        self.emptyFields()
        self.inverseButtonsState()

    def getRuleString(self):
        if self.comboBoxes['output'] is not None:
            return self.currentRule.toMamdaniStr()
        else:
            return self.currentRule.toSugenoStr()

    def addRule(self):
        if self.isRuleValid():
            fuzzySys.addRule(self.currentRule)
            self.rulesList.addItem(self.getRuleString())
            Window.plotCanvas.plotRulePatch()
            self.emptyFields()

    def changeRule(self):
        if self.isRuleValid(self.currentIndex):
            fuzzySys.updateRule(self.currentIndex, self.currentRule)
            Window.plotCanvas.changeRulePatch(self.currentIndex)
            self.rulesList.item(self.currentIndex).setText(self.getRuleString())
            self.emptyFields()
            self.inverseButtonsState()

    def deleteRule(self, ind=None):
        if ind is None:
            ind = self.currentIndex

        fuzzySys.deleteRule(ind)
        Window.plotCanvas.deleteRulePatch(ind)
        item = self.rulesList.takeItem(ind)
        del item

        if self.rulesList.count() > 0:
            self.currentIndex = self.rulesList.currentRow()
        else:
            self.currentIndex = None

    def isRuleValid(self, ind=None):
        if self.comboBoxes['output'] is not None:
            isValid = self.isMamdaniRuleValid()
        else:
            isValid = self.isSugenoRuleValid()

        if not isValid[0]:
            WarningMessage('Invalid rule')
            return False

        self.currentRule = fis.Rule(isValid[1], isValid[2])
        if fuzzySys.hasRule(self.currentRule, ind):
            WarningMessage('Rule with such condition already exists')
            return False

        return True

    def isMamdaniRuleValid(self):
        antecedent = self.comboBoxes['input'].currentText()
        consequent = self.comboBoxes['output'].currentText()

        if antecedent == '' or consequent == '':
            return False, antecedent, consequent

        return True, antecedent, consequent

    def isSugenoRuleValid(self):
        antecedent = self.comboBoxes['input'].currentText()
        consequent = [self.constX.value(), self.const.value()]

        func = fuzzySys.getMFunc(antecedent, 'input')
        a = func.params.a
        d = func.params.d
        if antecedent == '':
            return False, antecedent, consequent

        self.constX.setValue(0)
        self.const.setValue(0)

        return True, antecedent, consequent

    def emptyFields(self):
        if self.comboBoxes['output'] is None:
            self.constX.setValue(0)
            self.const.setValue(0)

    def clearRules(self):
        fuzzySys.rules = []
        self.rulesList.clear()
        Window.plotCanvas.deleteRulePatches()

    def buttonNextOnClick(self):
        self.nextStep.setFuzzyNames()
        if Window.plotCanvas.approximatedPlot is not None and (
                Window.plotCanvas.mFuncPlots['input'] != self.nextStep.curMFuncPlots['input'] or
                Window.plotCanvas.mFuncPlots['output'] != self.nextStep.curMFuncPlots['output'] or
                Window.plotCanvas.rulesPatches != self.nextStep.curRulePatches):
            Window.plotCanvas.deleteApproximated()

        if fuzzySys.model == 0:
            self.nextStep.addDefuzzificationField()
        else:
            self.nextStep.removeDefuzzificationField()

        Window.plotCanvas.hideOrShowApproximated()
        Window.buttonNextOnClick()


class FourthStepForm(QWidget):
    def __init__(self):
        super().__init__()
        self.curMFuncPlots = {'input': [], 'output': []}
        self.curRulePatches = []
        self.initUI()

    def initUI(self):
        self.createParamsGroupBox()
        self.createInputGroupBox()
        self.createResultsGroupBox()

        self.buttonBack = QPushButton('< Back')
        self.buttonBack.setAutoDefault(True)
        self.buttonBack.clicked.connect(self.buttonBackOnClick)
        self.buttonBack.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout()
        layout.addWidget(self.paramsFormGroupBox)
        layout.addWidget(self.inputGroupBox)
        layout.addWidget(self.resultsGroupBox)
        layout.addWidget(self.buttonBack)
        layout.setAlignment(self.resultsGroupBox, Qt.AlignTop)
        layout.setAlignment(self.buttonBack, Qt.AlignRight)

        self.setLayout(layout)

    def createParamsGroupBox(self):
        self.paramsFormGroupBox = QGroupBox('FIS params')

        self.operators = QComboBox()
        self.operators.addItem('min-max')
        self.operators.addItem('product-sum')

        self.defuzzification = None

        self.paramsFormLayout = QFormLayout()
        self.paramsFormLayout.setLabelAlignment(Qt.AlignRight)
        self.paramsFormLayout.addRow('Operators:', self.operators)
        self.addDefuzzificationField()

        self.paramsFormGroupBox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.paramsFormGroupBox.setLayout(self.paramsFormLayout)

    def addDefuzzificationField(self):
        if self.defuzzification is None:
            self.operators.setEnabled(True)
            self.defuzzification = QComboBox()
            self.defuzzification.addItem('centroid')
            self.defuzzification.addItem('mean of maximum')
            self.paramsFormLayout.addRow('Deffuzification', self.defuzzification)

    def removeDefuzzificationField(self):
        if self.defuzzification is not None:
            self.operators.setCurrentIndex(0)
            self.operators.setEnabled(False)
            label = self.paramsFormLayout.labelForField(self.defuzzification)
            self.paramsFormLayout.removeWidget(label)
            self.paramsFormLayout.removeWidget(self.defuzzification)
            label.deleteLater()
            self.defuzzification.deleteLater()
            self.defuzzification = None

    def createInputGroupBox(self):
        self.inputGroupBox = QGroupBox('Input')

        self.crispButton = QRadioButton('crisp value')
        self.crispButton.clicked.connect(self.activateCrispValue)
        self.crispEdit = DoubleSpinBox()

        self.fuzzyButton = QRadioButton('fuzzy value')
        self.fuzzyButton.clicked.connect(self.activateFuzzyValue)
        self.hedgeEdit = QComboBox()
        self.hedgeEdit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hedgeEdit.addItem('')
        self.hedgeEdit.addItem('very')
        self.hedgeEdit.addItem('somewhat')
        self.hedgeEdit.addItem('definitely')
        self.hedgeEdit.addItem('generally')
        self.nameEdit = QComboBox()
        self.nameEdit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        fuzzy = QHBoxLayout()
        fuzzy.addWidget(self.hedgeEdit)
        fuzzy.addWidget(self.nameEdit)

        self.intervalButton = QRadioButton('crisp interval')
        self.intervalButton.clicked.connect(self.activateIntervalValue)
        self.domainCheckBox = QCheckBox('Whole domain')
        self.domainCheckBox.stateChanged.connect(self.setDomainValues)
        self.intervalLEdit = DoubleSpinBox()
        self.intervalLEdit.valueChanged.connect(self.uncheckDomain)
        self.intervalREdit = DoubleSpinBox()
        self.intervalREdit.valueChanged.connect(self.uncheckDomain)
        hBox = QHBoxLayout()
        self.label1 = QLabel('[')
        self.label1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hBox.addWidget(self.label1)
        hBox.addWidget(self.intervalLEdit)
        self.label2 = QLabel(',')
        self.label2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hBox.addWidget(self.label2)
        hBox.addWidget(self.intervalREdit)
        self.label3 = QLabel(']')
        self.label3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hBox.addWidget(self.label3)
        interval = QVBoxLayout()
        interval.addLayout(hBox)
        interval.addWidget(self.domainCheckBox)

        self.buttonApproximate = QPushButton('Approximate')
        self.buttonApproximate.setAutoDefault(True)
        self.buttonApproximate.clicked.connect(self.buttonApproximateOnClick)
        self.buttonApproximate.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        formLayout = QFormLayout()
        formLayout.setLabelAlignment(Qt.AlignLeft)
        formLayout.addRow(self.crispButton, self.crispEdit)
        formLayout.addRow(self.fuzzyButton, fuzzy)
        formLayout.addRow(self.intervalButton, interval)

        layout = QVBoxLayout()
        layout.addLayout(formLayout)
        layout.addWidget(self.buttonApproximate, alignment=Qt.AlignRight)

        self.inputGroupBox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.inputGroupBox.setLayout(layout)

    def uncheckDomain(self):
        if self.domainCheckBox.isChecked() and (self.intervalLEdit.value() != fuzzySys.funcBound['input'][0] or
                                                self.intervalREdit.value() != fuzzySys.funcBound['input'][1]):
            self.domainCheckBox.setChecked(False)

    def setDomainValues(self):
        if self.domainCheckBox.isChecked():
            self.intervalLEdit.setValue(fuzzySys.funcBound['input'][0])
            self.intervalREdit.setValue(fuzzySys.funcBound['input'][1])

    def setCrispEnabled(self, flag):
        self.crispEdit.setEnabled(flag)

    def setFuzzyEnabled(self, flag):
        self.hedgeEdit.setEnabled(flag)
        self.nameEdit.setEnabled(flag)

    def setIntervalEnabled(self, flag):
        self.intervalLEdit.setEnabled(flag)
        self.intervalREdit.setEnabled(flag)
        self.label1.setEnabled(flag)
        self.label2.setEnabled(flag)
        self.label3.setEnabled(flag)
        self.domainCheckBox.setEnabled(flag)

    def activateCrispValue(self):
        self.setCrispEnabled(True)
        self.setFuzzyEnabled(False)
        self.setIntervalEnabled(False)

    def activateFuzzyValue(self):
        self.setCrispEnabled(False)
        self.setFuzzyEnabled(True)
        self.setIntervalEnabled(False)

    def activateIntervalValue(self):
        self.setCrispEnabled(False)
        self.setFuzzyEnabled(False)
        self.setIntervalEnabled(True)

    def setCrispBoundaryValues(self):
        self.crispEdit.setMinimum(fuzzySys.funcBound['input'][0])
        self.crispEdit.setMaximum(fuzzySys.funcBound['input'][1])
        self.setCrispDefaultValue()

        self.intervalLEdit.setMinimum(fuzzySys.funcBound['input'][0])
        self.intervalLEdit.setMaximum(fuzzySys.funcBound['input'][1])
        self.intervalREdit.setMinimum(fuzzySys.funcBound['input'][0])
        self.intervalREdit.setMaximum(fuzzySys.funcBound['input'][1])
        self.setIntervalDefaultValues()

    def setCrispDefaultValue(self):
        self.crispEdit.setValue(self.crispEdit.minimum())

    def setIntervalDefaultValues(self):
        self.intervalLEdit.setValue(self.intervalLEdit.minimum())
        self.intervalREdit.setValue(self.intervalREdit.minimum())

    def setFuzzyNames(self):
        self.nameEdit.clear()
        for name in fuzzySys.getMFuncNames('input'):
            self.nameEdit.addItem(name)

    def setInitialState(self):
        self.crispButton.setChecked(True)
        self.activateCrispValue()
        self.setCrispBoundaryValues()
        self.setFuzzyNames()

    def createResultsGroupBox(self):
        self.resultsGroupBox = QGroupBox('Approximation Results')

        self.trueOutput = QLabel()
        self.approximatedOutput = QLabel()
        self.mae = QLabel()
        self.rmse = QLabel()
        self.smape = QLabel()

        self.resFormLayout = QFormLayout()
        self.resFormLayout.setLabelAlignment(Qt.AlignRight)
        self.resFormLayout.addRow('True output:', self.trueOutput)
        self.resFormLayout.addRow('Approximated output:', self.approximatedOutput)
        self.resFormLayout.addRow('MAE:', self.mae)
        self.resFormLayout.addRow('RMSE:', self.rmse)
        self.resFormLayout.addRow('SMAPE:', self.smape)

        self.resultsGroupBox.setLayout(self.resFormLayout)

    def buttonApproximateOnClick(self):
        self.clearResults()
        self.createProgressDialog()

        if self.defuzzification is not None:
            defMethod = self.defuzzification.currentText()
        else:
            defMethod = None

        input = self.defineInput()
        res = fuzzySys.start(input, self.operators.currentText(), defMethod, self.progress)
        if res[0] == False:
            if res[1] is not None:
                WarningMessage('Input ' + self.getInputString(res[1]) + ' does not fire any rule')
            return

        errors = res[1]

        fuzzyInput = None if fuzzySys.inputType != 1 else input[1]
        self.showResults(errors, fuzzyInput)

    def getInputString(self, input):
        return input if fuzzySys.inputType == 1 else '{:.2f}'.format(input)

    def defineInput(self):
        if self.crispButton.isChecked():
            fuzzySys.inputType = 0
            return self.crispEdit.value()

        if self.fuzzyButton.isChecked():
            fuzzySys.inputType = 1
            return [self.hedgeEdit.currentText(), self.nameEdit.currentText()]

        if self.intervalButton.isChecked():
            fuzzySys.inputType = 2
            return [self.intervalLEdit.value(), self.intervalREdit.value()]

    def showResults(self, errors, fuzzyInput=None):
        if fuzzySys.inputType != 2:
            if fuzzySys.inputType == 0:
                self.trueOutput.setText('{:.2}'.format(fuzzySys.trueValues[fuzzySys.trueInd]))
            else:
                self.trueOutput.setText('-')
            self.approximatedOutput.setText('{:.2}'.format((fuzzySys.approximatedValues[0])))
        else:
            self.trueOutput.setText('-')
            self.approximatedOutput.setText('-')

        self.mae.setText('{:.3}'.format(errors['mae']))
        self.rmse.setText('{:.3}'.format(errors['rmse']))
        self.smape.setText('{:.3}'.format(errors['smape']))

        if fuzzySys.inputType == 1:
            Window.plotCanvas.plotApproximated(fuzzyInput)
        else:
            Window.plotCanvas.plotApproximated()

    def clearResults(self):
        self.trueOutput.clear()
        self.approximatedOutput.clear()

        self.mae.clear()
        self.rmse.clear()
        self.smape.clear()

        Window.plotCanvas.deleteApproximated()

    def createProgressDialog(self):
        self.progress = QProgressDialog('Approximating function...', 'Cancel', 0, 100)
        self.progress.setWindowFlags(Qt.WindowCloseButtonHint)
        self.progress.setWindowTitle('Approximate')
        self.progress.setModal(True)
        self.progress.setAutoReset(True)
        self.progress.setAutoClose(True)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.progress.resize(500, 100)

    def buttonBackOnClick(self):
        self.curMFuncPlots = {'input': [], 'output': []}
        self.curRulePatches = []

        for name in ['input', 'output']:
            for i in range(len(Window.plotCanvas.mFuncPlots[name])):
                self.curMFuncPlots[name].append(Window.plotCanvas.mFuncPlots[name][i])

        for i in range(len(Window.plotCanvas.rulesPatches)):
            self.curRulePatches.append(Window.plotCanvas.rulesPatches[i])

        Window.plotCanvas.hideOrShowApproximated()

        Window.buttonBackOnClick()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=900, height=800, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.initMembers()

        super().__init__(self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)

    def initMembers(self):
        self.mFuncLabels = {'input': [], 'output': []}
        self.mFuncPlots = {'input': [], 'output': []}
        self.rulesPatches = []
        self.sugenoRulesLims = {'right': [], 'left': []}
        self.truePlot = None
        self.truePoint = None
        self.trueLegend = None
        self.approximatedPlot = None
        self.legend = None
        self.alphaApproximated = None
        self.xLeft = None
        self.xRight = None
        self.yLeft = None
        self.yRight = None

    def grid(self):
        self.axes.grid()
        self.draw()

    def plotFunction(self, domain, function, isShowGrid=None):
        self.axes.clear()
        self.initMembers()

        try:
            x = np.arange(domain[0], domain[1] + 0.001, 0.01)
            x = np.around(x, 2)
        except Exception:
            WarningMessage('Domain is too large')
            return

        try:
            if function.find('x') < 0:
                raise Exception()
            y = ne.evaluate(function, local_dict={'x': x})
            fuzzySys.func = function
        except Exception:
            WarningMessage('Function expression is invlaid')
            return False

        nanIndices = np.argwhere(np.isnan(y))
        if len(nanIndices) > 0:
            x = np.delete(x, nanIndices)
            y = np.delete(y, nanIndices)

        infIndices = np.argwhere((y > sys.maxsize) | (y < (-sys.maxsize - 1)))

        if len(infIndices) == len(x):
            WarningMessage('Function has too large values on the specified domain')
            return

        length = len(y)
        if len(infIndices) > 0 and (infIndices[0] == 0 or infIndices[-1] == length - 1):
            x = np.delete(x, infIndices)
            y = np.delete(y, infIndices)
        if len(infIndices) > 0 and infIndices[0] > 0 and infIndices[-1] < length - 1:
            y[infIndices] = np.nan
            self.truePlot = self.axes.plot(x, y, linewidth=2.5)[0]
            x = np.delete(x, infIndices)
            y = np.delete(y, infIndices)

        if self.truePlot is None:
            self.truePlot = self.axes.plot(x, y, linewidth=2.5)[0]

        fuzzySys.funcBound['input'][0] = x[0]
        fuzzySys.funcBound['input'][1] = x[-1]
        fuzzySys.funcBound['output'][0] = np.min(y)
        fuzzySys.funcBound['output'][1] = np.max(y)
        fuzzySys.trueValues = y

        self.moveAxes(xLeft=fuzzySys.funcBound['input'][0], yLeft=fuzzySys.funcBound['output'][0])
        self.xRight = fuzzySys.funcBound['input'][1]
        self.yRight = fuzzySys.funcBound['output'][1]

        self.draw()

        return True

    def moveAxes(self, yLeft, xLeft=None):
        if xLeft != self.xLeft or yLeft != self.yLeft:
            if len(self.mFuncPlots['input']) > 0:
                self.moveInputMFuncs(yLeft - self.yLeft)

            self.axes.spines['right'].set_color('none')
            self.axes.spines['top'].set_color('none')
            self.axes.xaxis.set_ticks_position('bottom')
            self.axes.spines['bottom'].set_position(('data', yLeft))
            self.axes.yaxis.set_ticks_position('left')
            if xLeft is not None:
                self.axes.spines['left'].set_position(('data', xLeft))
                self.xLeft = xLeft
            self.yLeft = yLeft

    def updateYLims(self):
        self.axes.relim()
        self.axes.autoscale_view(tight=False)

    def plotApproximated(self, fuzzyInput=None):
        self.deleteApproximated()

        self.alphaApproximated = 1

        if fuzzySys.inputType != 2:
            if fuzzySys.inputType == 0:
                x = fuzzySys.x
                colorApproximated = 'orange'
                marker = '.'
                self.truePoint = self.axes.scatter(x, fuzzySys.trueValues[fuzzySys.trueInd],
                                                   color='#1f77b4', linewidths=2)
            else:
                params = fuzzySys.getMFunc(fuzzyInput, 'input').params
                a, d = fuzzySys.getMFunc(fuzzyInput, 'input').params.getBoundParams(fuzzySys.funcBound['input'])
                x = (a + d) / 2
                colorApproximated = self.mFuncPlots['input'][self.findInputMFLabel(fuzzyInput)].get_color()
                marker = None

            self.approximatedPlot = self.axes.scatter(x, fuzzySys.approximatedValues[0],
                                                      color=colorApproximated, linewidths=2)
            trueLegend = lines.Line2D([], [], color='#1f77b4', marker=marker)
            plots = [trueLegend, self.approximatedPlot]

        else:
            self.approximatedPlot, = self.axes.plot(fuzzySys.x, fuzzySys.approximatedValues,
                                                    linewidth=2.5, color='orange')
            plots = [self.truePlot, self.approximatedPlot]

        self.legend = self.axes.legend(plots, ['true', 'approximated'], loc='best')

        self.draw()

    def findInputMFLabel(self, name):
        for i in range(len(self.mFuncLabels['input'])):
            if self.mFuncLabels['input'][i].get_text() == name:
                return i

    def deleteApproximated(self):
        if self.approximatedPlot is not None:
            self.legend.remove()

            plot = self.approximatedPlot
            plot.remove()
            self.approximatedPlot = None

            if self.truePoint is not None:
                point = self.truePoint
                point.remove()
                self.truePoint = None

            self.draw()

    def hideOrShowApproximated(self):
        if self.approximatedPlot is not None:
            self.alphaApproximated = 1 - self.alphaApproximated

            self.approximatedPlot.set_alpha(self.alphaApproximated)
            if self.truePoint is not None:
                self.truePoint.set_alpha(self.alphaApproximated)
            self.legend.set_visible(self.alphaApproximated)

            self.draw()

    def plotMF(self, name, axis, ind=None, delta=0):
        if ind is None:
            ind = len(self.mFuncPlots[axis.lower()])

        mFunc = fuzzySys.getMFunc(name, axis)
        bounds = fuzzySys.funcBound[axis.lower()]
        a = max(mFunc.params.a, bounds[0])
        b = max(mFunc.params.b, bounds[0])
        c = max(mFunc.params.c, bounds[0])
        d = mFunc.params.d
        if d > bounds[1]:
            d = min(d, bounds[1])
            c = min(c, bounds[1])
            b = min(b, bounds[1])

        x = np.arange(a, d + 0.001, 0.01)
        x[-1] = d
        scale = -1 / 15
        center = (b + c) / 2 if b != c else b

        if axis.lower() == 'input':
            k = scale * (self.yRight - (self.yLeft + delta))
            coef = self.yLeft + delta
            y = mFunc.function(x) * k + coef
            self.mFuncLabels[axis.lower()].insert(ind, self.axes.text(center, np.min(y) + 0.3 * k, name,
                                                                      horizontalalignment='center',
                                                                      verticalalignment='center'))

        else:
            k = scale * (self.xRight - self.xLeft)
            coef = self.xLeft
            x, y = mFunc.function(x) * k + coef, x
            self.mFuncLabels[axis.lower()].insert(ind, self.axes.text(np.min(x) + 0.3 * k, center, name,
                                                                      horizontalalignment='center',
                                                                      verticalalignment='center',
                                                                      rotation='vertical'))

        self.mFuncPlots[axis.lower()].insert(ind, self.axes.plot(x, y)[0])
        self.draw()

    def moveInputMFuncs(self, delta):
        for i in range(len(self.mFuncPlots['input'])):
            name = self.mFuncLabels['input'][i].get_text()
            color = self.mFuncPlots['input'][i].get_color()

            self.deleteMFPlot('input', i)
            self.plotMF(name, 'input', i, delta)
            self.mFuncPlots['input'][i].set_color(color)

    def deleteMFPlot(self, axis, ind):
        self.mFuncLabels[axis.lower()].pop(ind).remove()
        self.mFuncPlots[axis.lower()].pop(ind).remove()
        self.draw()

    def deleteOutputMFPlots(self):
        while len(self.mFuncPlots['output']) > 0:
            self.mFuncPlots['output'].pop().remove()
            self.mFuncLabels['output'].pop().remove()
        self.draw()

    def plotRulePatch(self, ind=None):
        if ind is None:
            ind = len(self.rulesPatches)

        if fuzzySys.model == 0:
            self.plotMamdaniRule(ind)
        else:
            self.plotSugenoRule(ind)

    def plotMamdaniRule(self, ind=None):
        rule = fuzzySys.getRule(ind)

        xa, xd = fuzzySys.getMFunc(rule.antecedent, 'input').params.getBoundParams(fuzzySys.funcBound['input'])
        ya, yd = fuzzySys.getMFunc(rule.consequent, 'output').params.getBoundParams(fuzzySys.funcBound['output'])
        self.rulesPatches.insert(ind,
                                 self.axes.add_patch(patches.Rectangle((xa, ya),
                                                                       xd - xa, yd - ya,
                                                                       edgecolor='black', facecolor='gray', alpha=0.2)))
        self.draw()

    def plotSugenoRule(self, ind=None):
        rule = fuzzySys.getRule(ind)

        a, d = fuzzySys.getMFunc(rule.antecedent, 'input').params.getBoundParams(fuzzySys.funcBound['input'])
        x = np.arange(a, d + 0.001, 0.01)
        x[-1] = d

        consts = rule.consequent
        y = consts[0] * x + consts[1]

        self.rulesPatches.insert(ind, self.axes.plot(x, y, color='gray', alpha=0.6)[0])

        left = y[0] if y[0] < y[-1] else y[-1]
        right = y[-1] if y[-1] > y[0] else y[0]
        self.sugenoRulesLims['right'].insert(ind, right)
        self.sugenoRulesLims['left'].insert(ind, left)

        yLeft = left if left < self.yLeft else self.yLeft
        self.yRight = right if right > self.yRight else self.yRight
        self.moveAxes(yLeft=yLeft)
        self.updateYLims()

        self.draw()

    def deleteRulePatch(self, ind):
        self.rulesPatches.pop(ind).remove()
        if len(self.sugenoRulesLims['right']) > 0:
            self.deleteSugenoRulePatch(ind)
        self.draw()

    def deleteSugenoRulePatch(self, ind):
        right = self.sugenoRulesLims['right'][ind]
        left = self.sugenoRulesLims['left'][ind]
        max = np.max(self.sugenoRulesLims['right'])
        min = np.min(self.sugenoRulesLims['left'])

        self.sugenoRulesLims['right'].pop(ind)
        self.sugenoRulesLims['left'].pop(ind)

        if right == max and max >= fuzzySys.funcBound['output'][1]:
            newMax = np.max(self.sugenoRulesLims['right']) if len(self.sugenoRulesLims['right']) > 0 \
                else fuzzySys.funcBound['output'][1]
            topLim = newMax if newMax > fuzzySys.funcBound['output'][1] else fuzzySys.funcBound['output'][1]
            self.yRight = topLim
            self.moveInputMFuncs(0)
            self.updateYLims()

        if left == min and min <= fuzzySys.funcBound['output'][0]:
            newMin = np.min(self.sugenoRulesLims['left']) if len(self.sugenoRulesLims['left']) > 0 \
                else fuzzySys.funcBound['output'][0]
            bottomLim = newMin if newMin < fuzzySys.funcBound['output'][0] else fuzzySys.funcBound['output'][0]
            self.moveAxes(yLeft=bottomLim)
            self.updateYLims()

    def deleteRulePatches(self):
        n = len(self.rulesPatches)
        if n > 0:
            for i in range(n):
                self.deleteRulePatch(i)
            self.draw()

    def changeRulePatch(self, ind):
        self.deleteRulePatch(ind)
        self.plotRulePatch(ind)

    def prepareFunction(self, func):
        func = func.replace(' ', '').replace('^', '**').lower()


class WarningMessage(QMessageBox):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle('Warning')
        self.setText(message)
        self.setIcon(QMessageBox.Warning)
        self.exec()


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, min=0, max=99.9, value=0):
        super().__init__()
        self.setWrapping(True)
        self.setSingleStep(0.01)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setBounds(min, max)
        self.setValue(value)

    def setBounds(self, min, max):
        self.setMinimum(min)
        self.setMaximum(max)


class Help(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Help')
        self.setSize()
        self.initUI()

    def setSize(self):
        size = QDesktopWidget().screenGeometry()
        left = size.width() / 4
        top = size.height() / 4.2
        width = size.width() - left * 2
        height = size.height() - top * 2
        self.setGeometry(left, top, width, height)

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        file = open('./files/help.txt')
        text = file.read()
        file.close()
        textEdit.setText(text)

        button = QPushButton('OK')
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(textEdit)
        layout.addWidget(button)
        layout.setAlignment(button, Qt.AlignRight)

        self.setLayout(layout)


class About(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('About')
        self.setSize()
        self.initUI()

    def setSize(self):
        size = QDesktopWidget().screenGeometry()
        left = size.width() / 2.8
        top = size.height() / 2.4
        width = size.width() - left * 2
        height = size.height() - top * 2
        self.setGeometry(left, top, width, height)

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        file = open('./files/about.txt')
        text = file.read()
        file.close()
        textEdit.setText(text)

        layout = QVBoxLayout()
        layout.addWidget(textEdit)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
