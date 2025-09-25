# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDoubleSpinBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QSpinBox,
    QSplitter, QTabWidget, QTableView, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1228, 817)
        self.actiongo = QAction(MainWindow)
        self.actiongo.setObjectName(u"actiongo")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_16 = QGridLayout(self.centralwidget)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.pb_import_data = QPushButton(self.widget)
        self.pb_import_data.setObjectName(u"pb_import_data")

        self.verticalLayout.addWidget(self.pb_import_data)

        self.le_path_data_import = QLineEdit(self.widget)
        self.le_path_data_import.setObjectName(u"le_path_data_import")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_path_data_import.sizePolicy().hasHeightForWidth())
        self.le_path_data_import.setSizePolicy(sizePolicy)
        self.le_path_data_import.setReadOnly(True)

        self.verticalLayout.addWidget(self.le_path_data_import)

        self.gb_sim_param = QGroupBox(self.widget)
        self.gb_sim_param.setObjectName(u"gb_sim_param")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.gb_sim_param.sizePolicy().hasHeightForWidth())
        self.gb_sim_param.setSizePolicy(sizePolicy1)
        self.gridLayout_24 = QGridLayout(self.gb_sim_param)
        self.gridLayout_24.setObjectName(u"gridLayout_24")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_4 = QLabel(self.gb_sim_param)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_2.addWidget(self.label_4)

        self.rb_type_freq_selec_dur = QRadioButton(self.gb_sim_param)
        self.rb_type_freq_selec_dur.setObjectName(u"rb_type_freq_selec_dur")

        self.horizontalLayout_2.addWidget(self.rb_type_freq_selec_dur)

        self.rb_type_freq_selec_doux = QRadioButton(self.gb_sim_param)
        self.rb_type_freq_selec_doux.setObjectName(u"rb_type_freq_selec_doux")

        self.horizontalLayout_2.addWidget(self.rb_type_freq_selec_doux)


        self.gridLayout_24.addLayout(self.horizontalLayout_2, 0, 0, 1, 6)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_2 = QLabel(self.gb_sim_param)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.sb_seuil_freq_select = QDoubleSpinBox(self.gb_sim_param)
        self.sb_seuil_freq_select.setObjectName(u"sb_seuil_freq_select")
        self.sb_seuil_freq_select.setMaximum(100.000000000000000)
        self.sb_seuil_freq_select.setSingleStep(0.100000000000000)
        self.sb_seuil_freq_select.setValue(50.000000000000000)

        self.horizontalLayout.addWidget(self.sb_seuil_freq_select)


        self.gridLayout_24.addLayout(self.horizontalLayout, 1, 0, 1, 4)

        self.line_3 = QFrame(self.gb_sim_param)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line_3, 2, 0, 1, 8)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_5 = QLabel(self.gb_sim_param)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.sb_taux_sans_risque = QDoubleSpinBox(self.gb_sim_param)
        self.sb_taux_sans_risque.setObjectName(u"sb_taux_sans_risque")
        self.sb_taux_sans_risque.setMaximum(10.000000000000000)
        self.sb_taux_sans_risque.setSingleStep(0.100000000000000)
        self.sb_taux_sans_risque.setValue(5.000000000000000)

        self.horizontalLayout_4.addWidget(self.sb_taux_sans_risque)


        self.gridLayout_24.addLayout(self.horizontalLayout_4, 3, 0, 1, 4)

        self.line_2 = QFrame(self.gb_sim_param)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line_2, 4, 0, 1, 8)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(self.gb_sim_param)
        self.label.setObjectName(u"label")

        self.horizontalLayout_3.addWidget(self.label)

        self.rb_export_results_yes = QRadioButton(self.gb_sim_param)
        self.buttonGroup = QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.rb_export_results_yes)
        self.rb_export_results_yes.setObjectName(u"rb_export_results_yes")

        self.horizontalLayout_3.addWidget(self.rb_export_results_yes)

        self.rb_export_results_no = QRadioButton(self.gb_sim_param)
        self.buttonGroup.addButton(self.rb_export_results_no)
        self.rb_export_results_no.setObjectName(u"rb_export_results_no")

        self.horizontalLayout_3.addWidget(self.rb_export_results_no)


        self.gridLayout_24.addLayout(self.horizontalLayout_3, 5, 0, 1, 4)

        self.label_export_path = QLabel(self.gb_sim_param)
        self.label_export_path.setObjectName(u"label_export_path")

        self.gridLayout_24.addWidget(self.label_export_path, 6, 0, 1, 4)

        self.pb_select_export_repo = QPushButton(self.gb_sim_param)
        self.pb_select_export_repo.setObjectName(u"pb_select_export_repo")

        self.gridLayout_24.addWidget(self.pb_select_export_repo, 6, 5, 1, 2)

        self.le_path_data_export = QLineEdit(self.gb_sim_param)
        self.le_path_data_export.setObjectName(u"le_path_data_export")
        self.le_path_data_export.setReadOnly(True)

        self.gridLayout_24.addWidget(self.le_path_data_export, 7, 0, 1, 2)

        self.line = QFrame(self.gb_sim_param)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line, 8, 0, 1, 8)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_7 = QLabel(self.gb_sim_param)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 0, 0, 1, 1)

        self.sp_ca = QDoubleSpinBox(self.gb_sim_param)
        self.sp_ca.setObjectName(u"sp_ca")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.sp_ca.sizePolicy().hasHeightForWidth())
        self.sp_ca.setSizePolicy(sizePolicy2)
        self.sp_ca.setMaximum(100.000000000000000)
        self.sp_ca.setSingleStep(0.010000000000000)
        self.sp_ca.setValue(5.000000000000000)

        self.gridLayout.addWidget(self.sp_ca, 1, 0, 1, 1)


        self.horizontalLayout_5.addLayout(self.gridLayout)

        self.gridLayout_17 = QGridLayout()
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.label_8 = QLabel(self.gb_sim_param)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_17.addWidget(self.label_8, 0, 0, 1, 1)

        self.sp_gross_margin = QDoubleSpinBox(self.gb_sim_param)
        self.sp_gross_margin.setObjectName(u"sp_gross_margin")
        sizePolicy2.setHeightForWidth(self.sp_gross_margin.sizePolicy().hasHeightForWidth())
        self.sp_gross_margin.setSizePolicy(sizePolicy2)
        self.sp_gross_margin.setMaximum(100.000000000000000)
        self.sp_gross_margin.setSingleStep(0.010000000000000)
        self.sp_gross_margin.setValue(15.000000000000000)

        self.gridLayout_17.addWidget(self.sp_gross_margin, 1, 0, 1, 1)


        self.horizontalLayout_5.addLayout(self.gridLayout_17)

        self.gridLayout_18 = QGridLayout()
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.label_9 = QLabel(self.gb_sim_param)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_18.addWidget(self.label_9, 0, 0, 1, 1)

        self.sp_roic = QDoubleSpinBox(self.gb_sim_param)
        self.sp_roic.setObjectName(u"sp_roic")
        sizePolicy2.setHeightForWidth(self.sp_roic.sizePolicy().hasHeightForWidth())
        self.sp_roic.setSizePolicy(sizePolicy2)
        self.sp_roic.setMaximum(100.000000000000000)
        self.sp_roic.setSingleStep(0.010000000000000)
        self.sp_roic.setValue(20.000000000000000)

        self.gridLayout_18.addWidget(self.sp_roic, 1, 0, 1, 1)


        self.horizontalLayout_5.addLayout(self.gridLayout_18)

        self.gridLayout_19 = QGridLayout()
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.label_12 = QLabel(self.gb_sim_param)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_19.addWidget(self.label_12, 0, 0, 1, 1)

        self.sp_debt = QDoubleSpinBox(self.gb_sim_param)
        self.sp_debt.setObjectName(u"sp_debt")
        sizePolicy2.setHeightForWidth(self.sp_debt.sizePolicy().hasHeightForWidth())
        self.sp_debt.setSizePolicy(sizePolicy2)
        self.sp_debt.setMaximum(100.000000000000000)
        self.sp_debt.setSingleStep(0.010000000000000)
        self.sp_debt.setValue(5.000000000000000)

        self.gridLayout_19.addWidget(self.sp_debt, 1, 0, 1, 1)


        self.horizontalLayout_5.addLayout(self.gridLayout_19)


        self.gridLayout_24.addLayout(self.horizontalLayout_5, 9, 0, 1, 7)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.gridLayout_23 = QGridLayout()
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.label_10 = QLabel(self.gb_sim_param)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_23.addWidget(self.label_10, 0, 0, 1, 1)

        self.sp_piotroski = QDoubleSpinBox(self.gb_sim_param)
        self.sp_piotroski.setObjectName(u"sp_piotroski")
        sizePolicy.setHeightForWidth(self.sp_piotroski.sizePolicy().hasHeightForWidth())
        self.sp_piotroski.setSizePolicy(sizePolicy)
        self.sp_piotroski.setMaximum(100.000000000000000)
        self.sp_piotroski.setSingleStep(0.010000000000000)
        self.sp_piotroski.setValue(10.000000000000000)

        self.gridLayout_23.addWidget(self.sp_piotroski, 1, 0, 1, 1)


        self.horizontalLayout_7.addLayout(self.gridLayout_23)

        self.gridLayout_22 = QGridLayout()
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.label_11 = QLabel(self.gb_sim_param)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_22.addWidget(self.label_11, 0, 0, 1, 1)

        self.sp_buyback = QDoubleSpinBox(self.gb_sim_param)
        self.sp_buyback.setObjectName(u"sp_buyback")
        sizePolicy.setHeightForWidth(self.sp_buyback.sizePolicy().hasHeightForWidth())
        self.sp_buyback.setSizePolicy(sizePolicy)
        self.sp_buyback.setMaximum(100.000000000000000)
        self.sp_buyback.setSingleStep(0.010000000000000)
        self.sp_buyback.setValue(5.000000000000000)

        self.gridLayout_22.addWidget(self.sp_buyback, 1, 0, 1, 1)


        self.horizontalLayout_7.addLayout(self.gridLayout_22)

        self.gridLayout_21 = QGridLayout()
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.label_6 = QLabel(self.gb_sim_param)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_21.addWidget(self.label_6, 0, 0, 1, 1)

        self.sp_value = QDoubleSpinBox(self.gb_sim_param)
        self.sp_value.setObjectName(u"sp_value")
        sizePolicy.setHeightForWidth(self.sp_value.sizePolicy().hasHeightForWidth())
        self.sp_value.setSizePolicy(sizePolicy)
        self.sp_value.setMaximum(100.000000000000000)
        self.sp_value.setSingleStep(0.010000000000000)
        self.sp_value.setValue(40.000000000000000)

        self.gridLayout_21.addWidget(self.sp_value, 1, 0, 1, 1)


        self.horizontalLayout_7.addLayout(self.gridLayout_21)

        self.gridLayout_20 = QGridLayout()
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.label_13 = QLabel(self.gb_sim_param)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_20.addWidget(self.label_13, 0, 0, 1, 1)

        self.l_sum_weights = QLabel(self.gb_sim_param)
        self.l_sum_weights.setObjectName(u"l_sum_weights")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.l_sum_weights.sizePolicy().hasHeightForWidth())
        self.l_sum_weights.setSizePolicy(sizePolicy3)
        self.l_sum_weights.setMidLineWidth(-3)

        self.gridLayout_20.addWidget(self.l_sum_weights, 1, 0, 1, 1)


        self.horizontalLayout_7.addLayout(self.gridLayout_20)


        self.gridLayout_24.addLayout(self.horizontalLayout_7, 10, 0, 1, 7)

        self.line_4 = QFrame(self.gb_sim_param)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line_4, 11, 0, 1, 8)

        self.label_3 = QLabel(self.gb_sim_param)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_24.addWidget(self.label_3, 12, 0, 2, 1)

        self.label_14 = QLabel(self.gb_sim_param)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_24.addWidget(self.label_14, 12, 1, 2, 1)

        self.sp_start_year = QSpinBox(self.gb_sim_param)
        self.sp_start_year.setObjectName(u"sp_start_year")
        sizePolicy.setHeightForWidth(self.sp_start_year.sizePolicy().hasHeightForWidth())
        self.sp_start_year.setSizePolicy(sizePolicy)
        self.sp_start_year.setMinimum(1900)
        self.sp_start_year.setMaximum(9999)
        self.sp_start_year.setValue(2014)

        self.gridLayout_24.addWidget(self.sp_start_year, 12, 3, 2, 1)

        self.label_17 = QLabel(self.gb_sim_param)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_24.addWidget(self.label_17, 12, 4, 2, 1)

        self.sp_end_year = QSpinBox(self.gb_sim_param)
        self.sp_end_year.setObjectName(u"sp_end_year")
        sizePolicy.setHeightForWidth(self.sp_end_year.sizePolicy().hasHeightForWidth())
        self.sp_end_year.setSizePolicy(sizePolicy)
        self.sp_end_year.setMinimum(1900)
        self.sp_end_year.setMaximum(9999)
        self.sp_end_year.setValue(2024)

        self.gridLayout_24.addWidget(self.sp_end_year, 12, 5, 2, 2)

        self.line_5 = QFrame(self.gb_sim_param)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line_5, 14, 0, 2, 8)

        self.sb_nbe_monte_carlo = QSpinBox(self.gb_sim_param)
        self.sb_nbe_monte_carlo.setObjectName(u"sb_nbe_monte_carlo")
        sizePolicy.setHeightForWidth(self.sb_nbe_monte_carlo.sizePolicy().hasHeightForWidth())
        self.sb_nbe_monte_carlo.setSizePolicy(sizePolicy)
        self.sb_nbe_monte_carlo.setMinimum(1)
        self.sb_nbe_monte_carlo.setMaximum(10000)
        self.sb_nbe_monte_carlo.setSingleStep(100)
        self.sb_nbe_monte_carlo.setValue(2500)

        self.gridLayout_24.addWidget(self.sb_nbe_monte_carlo, 15, 6, 2, 1)

        self.label_15 = QLabel(self.gb_sim_param)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_24.addWidget(self.label_15, 16, 0, 1, 6)

        self.line_6 = QFrame(self.gb_sim_param)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.Shape.HLine)
        self.line_6.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_24.addWidget(self.line_6, 17, 0, 1, 8)

        self.label_16 = QLabel(self.gb_sim_param)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_24.addWidget(self.label_16, 18, 0, 2, 2)

        self.sb_nb_stocks = QSpinBox(self.gb_sim_param)
        self.sb_nb_stocks.setObjectName(u"sb_nb_stocks")
        sizePolicy.setHeightForWidth(self.sb_nb_stocks.sizePolicy().hasHeightForWidth())
        self.sb_nb_stocks.setSizePolicy(sizePolicy)
        self.sb_nb_stocks.setMinimum(1)
        self.sb_nb_stocks.setMaximum(100)
        self.sb_nb_stocks.setSingleStep(1)
        self.sb_nb_stocks.setValue(20)

        self.gridLayout_24.addWidget(self.sb_nb_stocks, 19, 2, 1, 2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalSpacer_3 = QSpacerItem(108, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_3)

        self.pb_run = QPushButton(self.gb_sim_param)
        self.pb_run.setObjectName(u"pb_run")
        sizePolicy2.setHeightForWidth(self.pb_run.sizePolicy().hasHeightForWidth())
        self.pb_run.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.pb_run)

        self.pb_reset = QPushButton(self.gb_sim_param)
        self.pb_reset.setObjectName(u"pb_reset")
        sizePolicy2.setHeightForWidth(self.pb_reset.sizePolicy().hasHeightForWidth())
        self.pb_reset.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.pb_reset)

        self.pb_export_results = QPushButton(self.gb_sim_param)
        self.pb_export_results.setObjectName(u"pb_export_results")
        sizePolicy2.setHeightForWidth(self.pb_export_results.sizePolicy().hasHeightForWidth())
        self.pb_export_results.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.pb_export_results)


        self.gridLayout_24.addLayout(self.horizontalLayout_6, 20, 0, 1, 7)


        self.verticalLayout.addWidget(self.gb_sim_param)

        self.verticalSpacer = QSpacerItem(154, 46, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.splitter.addWidget(self.widget)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.gridLayout_2 = QGridLayout(self.layoutWidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.tw_results = QTabWidget(self.layoutWidget)
        self.tw_results.setObjectName(u"tw_results")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.tw_results.sizePolicy().hasHeightForWidth())
        self.tw_results.setSizePolicy(sizePolicy4)
        self.tab_final_ranking = QWidget()
        self.tab_final_ranking.setObjectName(u"tab_final_ranking")
        self.gridLayout_3 = QGridLayout(self.tab_final_ranking)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.tableView_final_ranking = QTableView(self.tab_final_ranking)
        self.tableView_final_ranking.setObjectName(u"tableView_final_ranking")

        self.gridLayout_3.addWidget(self.tableView_final_ranking, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_final_ranking, "")
        self.tab_ranking_detailed = QWidget()
        self.tab_ranking_detailed.setObjectName(u"tab_ranking_detailed")
        self.gridLayout_4 = QGridLayout(self.tab_ranking_detailed)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.tableView_ranking_detailed = QTableView(self.tab_ranking_detailed)
        self.tableView_ranking_detailed.setObjectName(u"tableView_ranking_detailed")

        self.gridLayout_4.addWidget(self.tableView_ranking_detailed, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_ranking_detailed, "")
        self.tab_revenues = QWidget()
        self.tab_revenues.setObjectName(u"tab_revenues")
        self.gridLayout_5 = QGridLayout(self.tab_revenues)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.tableView_revenues = QTableView(self.tab_revenues)
        self.tableView_revenues.setObjectName(u"tableView_revenues")

        self.gridLayout_5.addWidget(self.tableView_revenues, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_revenues, "")
        self.tab_gross_margin = QWidget()
        self.tab_gross_margin.setObjectName(u"tab_gross_margin")
        self.gridLayout_6 = QGridLayout(self.tab_gross_margin)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.tableView_gross_margin = QTableView(self.tab_gross_margin)
        self.tableView_gross_margin.setObjectName(u"tableView_gross_margin")

        self.gridLayout_6.addWidget(self.tableView_gross_margin, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_gross_margin, "")
        self.tab_fcf = QWidget()
        self.tab_fcf.setObjectName(u"tab_fcf")
        self.gridLayout_7 = QGridLayout(self.tab_fcf)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.tableView_fcf = QTableView(self.tab_fcf)
        self.tableView_fcf.setObjectName(u"tableView_fcf")

        self.gridLayout_7.addWidget(self.tableView_fcf, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_fcf, "")
        self.tab_roic = QWidget()
        self.tab_roic.setObjectName(u"tab_roic")
        self.gridLayout_8 = QGridLayout(self.tab_roic)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.tableView_roic = QTableView(self.tab_roic)
        self.tableView_roic.setObjectName(u"tableView_roic")

        self.gridLayout_8.addWidget(self.tableView_roic, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_roic, "")
        self.tab_debt = QWidget()
        self.tab_debt.setObjectName(u"tab_debt")
        self.gridLayout_9 = QGridLayout(self.tab_debt)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.tableView_debt = QTableView(self.tab_debt)
        self.tableView_debt.setObjectName(u"tableView_debt")

        self.gridLayout_9.addWidget(self.tableView_debt, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_debt, "")
        self.tab_piotroski = QWidget()
        self.tab_piotroski.setObjectName(u"tab_piotroski")
        self.gridLayout_10 = QGridLayout(self.tab_piotroski)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.tableView_piotroski = QTableView(self.tab_piotroski)
        self.tableView_piotroski.setObjectName(u"tableView_piotroski")

        self.gridLayout_10.addWidget(self.tableView_piotroski, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_piotroski, "")
        self.tab_buyback = QWidget()
        self.tab_buyback.setObjectName(u"tab_buyback")
        self.gridLayout_11 = QGridLayout(self.tab_buyback)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.tableView_buyback = QTableView(self.tab_buyback)
        self.tableView_buyback.setObjectName(u"tableView_buyback")

        self.gridLayout_11.addWidget(self.tableView_buyback, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_buyback, "")
        self.tab_value = QWidget()
        self.tab_value.setObjectName(u"tab_value")
        self.gridLayout_12 = QGridLayout(self.tab_value)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.tableView_value = QTableView(self.tab_value)
        self.tableView_value.setObjectName(u"tableView_value")

        self.gridLayout_12.addWidget(self.tableView_value, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_value, "")
        self.tab_quality = QWidget()
        self.tab_quality.setObjectName(u"tab_quality")
        self.gridLayout_13 = QGridLayout(self.tab_quality)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.tableView_quality = QTableView(self.tab_quality)
        self.tableView_quality.setObjectName(u"tableView_quality")

        self.gridLayout_13.addWidget(self.tableView_quality, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_quality, "")
        self.tab_sensibility_analysis = QWidget()
        self.tab_sensibility_analysis.setObjectName(u"tab_sensibility_analysis")
        self.gridLayout_14 = QGridLayout(self.tab_sensibility_analysis)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.tableView_sensibility_analysis = QTableView(self.tab_sensibility_analysis)
        self.tableView_sensibility_analysis.setObjectName(u"tableView_sensibility_analysis")

        self.gridLayout_14.addWidget(self.tableView_sensibility_analysis, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_sensibility_analysis, "")
        self.tab_robust_quality = QWidget()
        self.tab_robust_quality.setObjectName(u"tab_robust_quality")
        self.gridLayout_15 = QGridLayout(self.tab_robust_quality)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.tableView_robust_quality = QTableView(self.tab_robust_quality)
        self.tableView_robust_quality.setObjectName(u"tableView_robust_quality")

        self.gridLayout_15.addWidget(self.tableView_robust_quality, 0, 0, 1, 1)

        self.tw_results.addTab(self.tab_robust_quality, "")

        self.gridLayout_2.addWidget(self.tw_results, 0, 0, 1, 2)

        self.horizontalSpacer_2 = QSpacerItem(378, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 1, 0, 1, 1)

        self.pb_show_graphics_window = QPushButton(self.layoutWidget)
        self.pb_show_graphics_window.setObjectName(u"pb_show_graphics_window")
        sizePolicy2.setHeightForWidth(self.pb_show_graphics_window.sizePolicy().hasHeightForWidth())
        self.pb_show_graphics_window.setSizePolicy(sizePolicy2)

        self.gridLayout_2.addWidget(self.pb_show_graphics_window, 1, 1, 1, 1)

        self.splitter.addWidget(self.layoutWidget)

        self.gridLayout_16.addWidget(self.splitter, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tw_results.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actiongo.setText(QCoreApplication.translate("MainWindow", u"go", None))
        self.pb_import_data.setText(QCoreApplication.translate("MainWindow", u"Importation des donn\u00e9es", None))
        self.gb_sim_param.setTitle(QCoreApplication.translate("MainWindow", u"Param\u00e8tres de simulation", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Type de seuil pour la fr\u00e9quence de s\u00e9lections", None))
        self.rb_type_freq_selec_dur.setText(QCoreApplication.translate("MainWindow", u"Dur", None))
        self.rb_type_freq_selec_doux.setText(QCoreApplication.translate("MainWindow", u"Doux", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Seuil de fr\u00e9quence de s\u00e9lections", None))
        self.sb_seuil_freq_select.setSuffix(QCoreApplication.translate("MainWindow", u" %", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Taux sans risque", None))
        self.sb_taux_sans_risque.setSuffix(QCoreApplication.translate("MainWindow", u" %", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Exporter les r\u00e9sultats", None))
        self.rb_export_results_yes.setText(QCoreApplication.translate("MainWindow", u"Oui", None))
        self.rb_export_results_no.setText(QCoreApplication.translate("MainWindow", u"Non", None))
        self.label_export_path.setText(QCoreApplication.translate("MainWindow", u"R\u00e9pertoire d'exportation des r\u00e9sultats :", None))
        self.pb_select_export_repo.setText(QCoreApplication.translate("MainWindow", u"S\u00e9lection du r\u00e9pertoire", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Chiffre d'Affaires", None))
        self.sp_ca.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Marges brutes", None))
        self.sp_gross_margin.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"ROIC et FCF", None))
        self.sp_roic.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Endettement", None))
        self.sp_debt.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Piotroski", None))
        self.sp_piotroski.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Rachat d'actions", None))
        self.sp_buyback.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Valorisation", None))
        self.sp_value.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Somme des poids", None))
        self.l_sum_weights.setText("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"P\u00e9riode d'analyse :", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"D\u00e9but", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"- Fin", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Nombre de simulations pour l'analyse de sensibilit\u00e9 Monte-Carlo :", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Nombre d'actions souhait\u00e9 :", None))
        self.pb_run.setText(QCoreApplication.translate("MainWindow", u"Lancer les calculs", None))
        self.pb_reset.setText(QCoreApplication.translate("MainWindow", u"R\u00e9initialiser", None))
        self.pb_export_results.setText(QCoreApplication.translate("MainWindow", u"Exporter les r\u00e9sultats", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_final_ranking), QCoreApplication.translate("MainWindow", u"Classement final", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_ranking_detailed), QCoreApplication.translate("MainWindow", u"Classement d\u00e9taill\u00e9", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_revenues), QCoreApplication.translate("MainWindow", u"Chiffre d'Affaires", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_gross_margin), QCoreApplication.translate("MainWindow", u"Marges brutes", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_fcf), QCoreApplication.translate("MainWindow", u"Flux de tr\u00e9sorerie disponible", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_roic), QCoreApplication.translate("MainWindow", u"Retour sur Capitaux Investis", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_debt), QCoreApplication.translate("MainWindow", u"Endettement", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_piotroski), QCoreApplication.translate("MainWindow", u"Score de Piotroski", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_buyback), QCoreApplication.translate("MainWindow", u"Rachat d'actions", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_value), QCoreApplication.translate("MainWindow", u"Valorisation", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_quality), QCoreApplication.translate("MainWindow", u"Qualit\u00e9", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_sensibility_analysis), QCoreApplication.translate("MainWindow", u"Analyse de sensibilit\u00e9", None))
        self.tw_results.setTabText(self.tw_results.indexOf(self.tab_robust_quality), QCoreApplication.translate("MainWindow", u"Qualit\u00e9 Robuste", None))
        self.pb_show_graphics_window.setText(QCoreApplication.translate("MainWindow", u"Visualiser les graphiques", None))
    # retranslateUi

