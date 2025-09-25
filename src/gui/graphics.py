# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'graphics.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QSizePolicy, QWidget)

class Ui_Graphics(object):
    def setupUi(self, Graphics):
        if not Graphics.objectName():
            Graphics.setObjectName(u"Graphics")
        Graphics.resize(636, 471)
        self.gridLayout = QGridLayout(Graphics)
        self.gridLayout.setObjectName(u"gridLayout")
        self.widget = QWidget(Graphics)
        self.widget.setObjectName(u"widget")

        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)


        self.retranslateUi(Graphics)

        QMetaObject.connectSlotsByName(Graphics)
    # setupUi

    def retranslateUi(self, Graphics):
        Graphics.setWindowTitle(QCoreApplication.translate("Graphics", u"Visualisation des r\u00e9sultats", None))
    # retranslateUi

