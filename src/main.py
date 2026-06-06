import os
import sys

import sys
import logging

# Configuration d'un logger minimaliste
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger("FATAL_DEBUG")
#
# def handle_exception(exc_type, exc_value, exc_traceback):
#     logger.critical("Exception non gérée", exc_info=(exc_type, exc_value, exc_traceback))
#
# sys.excepthook = handle_exception

print("--- Début du chargement de l'IHM ---")

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from pathlib import Path
import matplotlib.pyplot as plt
from PySide6.QtGui import QIcon, QAction, QPixmap, QActionGroup

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from gui.graphics import Ui_Graphics
from PySide6.QtWidgets import QHeaderView, QTableView, QProgressBar, QToolBar
from gui.MainWindow import Ui_MainWindow
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                               QMessageBox, QDialog, QSizePolicy)

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect, QEasingCurve, QPropertyAnimation

from matplotlib.figure import Figure

from model.Classes.data_manager import RobustFinancialLoader, RobustTargetGenerator
from model.ModelSelection.HAEN_Engine import HuberAdaptiveElasticNetPipeline
from model.RegressionAlgorithms.Solveurs.BaseAdaptativeEngine import RobustRegressorFactory
from model.RegressionAlgorithms.Solveurs.BaseRobustRegressors import SolverType
from model.RegressionAlgorithms.Solveurs.Robust_utils import ADMMWarmStartPath
from model.parameters import Parameters
from model.tikr_2016_2024 import Analysis, monte_carlo_sensitivity_analysis, write_excel
from model.portfolio_metrics_engine import PortfolioMetricsEngine

# Pour la compilation du .ui
# --> pyside6-uic C:\Users\guill\Desktop\Dev\Quality-Value-App\src\gui\MainWindow.ui -o C:\Users\guill\Desktop\Dev\Quality-Value-App\src\gui\MainWindow.py
# --> pyside6-uic C:\Users\guill\Desktop\Dev\Quality-Value-App\src\gui\graphics.ui -o C:\Users\guill\Desktop\Dev\Quality-Value-App\src\gui\graphics.py


class PandasTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            # Formatage des nombres
            if isinstance(value, (int, float)):
                return f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
            return str(value)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, float)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

class GraphicsWindow(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = [self.fig.add_subplot(111), self.fig.add_subplot(312), self.fig.add_subplot(313)]
        self.axes = self.fig.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})

        super(GraphicsWindow, self).__init__(self.fig)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        self.ax = self.fig.add_subplot(111)

    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Affiche la matrice de corrélation avec ajustement automatique"""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        # Créer un masque pour n'afficher que la moitié inférieure
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Dessiner le heatmap
        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": 0.8, "aspect": 30},
                    annot_kws={"size": 8},
                    ax=self.ax)

        self.ax.set_title('Corrélation des rangs entre tous les facteurs du modèle (Spearman)\n',
                          fontweight='bold', fontsize=12)

        # Rotation des étiquettes
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        # Ajustement AUTOMATIQUE de la mise en page
        self.fig.tight_layout()

        # Forcer le redessin et l'ajustement
        self.draw()
        self.adjust_size()

    def adjust_size(self):
        """Ajuste automatiquement la taille du canvas à son contenu"""
        # Obtenir la taille recommandée par la figure
        self.fig.canvas.draw()
        rect = self.fig.get_tightbbox(self.fig.canvas.get_renderer())

        # Convertir en pixels
        width = rect.width / self.fig.dpi
        height = rect.height / self.fig.dpi

        # Ajuster la taille de la figure
        self.fig.set_size_inches(width * 1.1, height * 1.1)  # Marge de 10%

        # Forcer le recalcul du layout
        self.fig.tight_layout()
        self.draw()

    def resizeEvent(self, event):
        """Redéfinition pour l'ajustement lors du redimensionnement"""
        super().resizeEvent(event)
        self.fig.tight_layout()
        self.draw()

class MainWindow(QMainWindow):
    QUALITY_FACTORS = ['Revenues', 'GrossMargin', 'FCF', 'ROIC', 'Debt', 'Buyback']

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # charge l'IHM générée par Qt Designer

        self.data_file_path: Path | None = None
        self.parameters: Parameters | None = None

        self.results_storage: dict[str, pd.DataFrame | pd.Series] = {}

        self.all_ranks_df: pd.DataFrame | None = None
        self.quality_results_series: pd.Series | None = None
        self.robust_quality_results: pd.DataFrame | None = None
        self.quality_weights_sum: float = 0.

        self.factorial_model_results: pd.DataFrame | None = None
        self.final_value_score_results: pd.DataFrame | None = None
        self.weights_allocation: None  = None

        self.revenues_results: pd.DataFrame | None = None
        self.gross_margin_results: pd.DataFrame | None = None
        self.cfoa_results: pd.DataFrame | None = None
        self.groc_results: pd.DataFrame | None = None
        self.buyback_results: pd.DataFrame | None = None
        self.piotroski_results: pd.DataFrame | None = None
        self.piotroski_results_reduced: pd.DataFrame | None = None
        self.value_results: pd.DataFrame | None = None
        self.value_sector_stats: pd.DataFrame | None = None
        self.debt_series: pd.DataFrame | None = None
        self.detailed_ranking_df: pd.DataFrame | None = None

        # Backtest
        self.backtest_engine: PortfolioMetricsEngine | None = None

        # Fenêtre graphique
        self.graphics_window: GraphicsWindow | None = None
        self.correlation_matrix: pd.DataFrame | None = None

        # Initialisation fenêtre + connections
        self.__update_gui_from_model()
        self.__initialize_connections()

        self.setWindowIcon(QIcon(str(self.parameters.script_path.parent.parent / "assets" / "logo.png")))
        self.setWindowTitle("Outil Fondamental d'Analyse Boursière")

        self.load_stylesheet()
        self.setup_menu()
        self.setup_theme_selector()
        self.setup_animations()


    def load_stylesheet(self):
        try:
            with open(self.parameters.script_path.parent.parent / "assets" / "style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            # Style par défaut si le fichier n'existe pas
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QPushButton { 
                    background-color: #007acc; 
                    color: white; 
                    border-radius: 4px; 
                    padding: 5px;
                }
            """)

    def setup_theme_selector(self):
        self.theme_actions = QActionGroup(self)

        light_theme = QAction("Thème clair", self, checkable=True)
        dark_theme = QAction("Thème sombre", self, checkable=True)
        system_theme = QAction("Thème système", self, checkable=True, checked=True)

        self.theme_actions.addAction(light_theme)
        self.theme_actions.addAction(dark_theme)
        self.theme_actions.addAction(system_theme)

        light_theme.triggered.connect(lambda: self.change_theme('light'))
        dark_theme.triggered.connect(lambda: self.change_theme('dark'))
        system_theme.triggered.connect(lambda: self.change_theme('system'))

        theme_menu = self.menuBar().addMenu("&Thème")
        theme_menu.addAction(light_theme)
        theme_menu.addAction(dark_theme)
        theme_menu.addAction(system_theme)

    def change_theme(self, theme_name):
        if theme_name == 'dark':
            self.apply_dark_theme()
        elif theme_name == 'light':
            self.apply_light_theme()
        else:
            self.apply_system_theme()

    def apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow {
            background-color: #2D2D30;
            color: #FFFFFF;
        }
        /* Ajoutez d'autres styles sombres ici */
        """
        self.setStyleSheet(dark_stylesheet)

    def setup_animations(self):
        # Animation pour le bouton d'analyse
        self.analysis_animation = QPropertyAnimation(self.ui.pb_run, b"geometry")
        self.analysis_animation.setDuration(1000)
        self.analysis_animation.setLoopCount(1)
        self.analysis_animation.setEasingCurve(QEasingCurve.OutBounce)

    def animate_run_button(self):
        original_geometry = self.ui.pb_run.geometry()
        self.analysis_animation.setStartValue(QRect(original_geometry.x() - 10,
                                                    original_geometry.y(),
                                                    original_geometry.width(),
                                                    original_geometry.height()))
        self.analysis_animation.setEndValue(original_geometry)
        self.analysis_animation.start()

    def show_about(self):
        about_text = """
        <h2>Financial Analysis Tool</h2>
        <p>Version 1.0</p>
        <p>Outil d'analyse financière basé sur les stratégie Value et Quality pour l'évaluation d'entreprises côtées sur les marchés US.</p>
        <p>© 2024 . Tous droits réservés.</p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("À propos")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)

        # Ajouter un logo
        pixmap = QPixmap(str(self.parameters.script_path.parent.parent / "assets" / "logo.png")).scaled(
            64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        msg.setIconPixmap(pixmap)

        msg.exec()

    def setup_menu(self):
        # Créer une barre de menu
        menubar = self.menuBar()

        # Menu Fichier
        file_menu = menubar.addMenu("&Fichier")

        open_action = QAction("Ouvrir", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Quitter", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menu Aide
        help_menu = menubar.addMenu("&Aide")

        about_action = QAction("À propos", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Barre de statut
        # self.statusBar().showMessage("Prêt")

        # Indicateur de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        # self.statusBar().addPermanentWidget(self.progress_bar)

    def __initialize_connections(self):
        self.ui.pb_import_data.clicked.connect(self.__open_file)

        self.ui.pb_reset.clicked.connect(self.__update_gui_from_model)

        self.ui.rb_type_freq_selec_dur.toggled.connect(self.toggle_seuil_widgets)
        self.ui.rb_type_freq_selec_doux.toggled.connect(self.toggle_seuil_widgets)

        self.ui.rb_export_results_yes.toggled.connect(self.toggle_export_widgets)
        self.ui.rb_export_results_no.toggled.connect(self.toggle_export_widgets)
        self.ui.pb_select_export_repo.clicked.connect(self.__open_dir)
        self.ui.pb_export_results.clicked.connect(self.__export_data)

        self.ui.sb_taux_sans_risque.valueChanged.connect(self.__update_model_from_gui)

        self.ui.sp_ca.valueChanged.connect(lambda value: self.__update_weights('Revenues', value))
        self.ui.sp_gross_margin.valueChanged.connect(lambda value: self.__update_weights('GrossMargin', value))
        self.ui.sp_roic.valueChanged.connect(lambda value: self.__update_weights('ROIC_&_FCF', value))
        self.ui.sp_debt.valueChanged.connect(lambda value: self.__update_weights('Debt', value))
        self.ui.sp_piotroski.valueChanged.connect(lambda value: self.__update_weights('Piotroski', value))
        self.ui.sp_buyback.valueChanged.connect(lambda value: self.__update_weights('Buyback', value))
        self.ui.sp_value.valueChanged.connect(lambda value: self.__update_weights('Value', value))

        self.ui.sp_start_year.valueChanged.connect(lambda value: self.__update_period('start_year', value))
        self.ui.sp_end_year.valueChanged.connect(lambda value: self.__update_period('end_year', value))

        self.ui.pb_run.clicked.connect(self.__run_analysis)
        self.ui.pb_run_backtest.clicked.connect(self.__run_backtest)
        self.ui.pb_show_correlation_window.clicked.connect(self.__show_graphics_window)

    def toggle_seuil_widgets(self):
        if self.ui.rb_type_freq_selec_dur.isChecked():
            self.ui.label_2.show()
            self.ui.label_2.show()
            self.ui.sb_seuil_freq_select.show()
        else:
            self.ui.label_2.hide()
            self.ui.label_2.hide()
            self.ui.sb_seuil_freq_select.hide()

        self.__update_model_from_gui()

    def toggle_export_widgets(self):
        if self.ui.rb_export_results_yes.isChecked():
            self.ui.le_path_data_export.show()
            self.ui.pb_select_export_repo.show()
            self.ui.label_export_path.show()
        else:
            self.ui.le_path_data_export.hide()
            self.ui.pb_select_export_repo.hide()
            self.ui.label_export_path.hide()

        self.__update_model_from_gui()

    def __update_period(self, attribute_name: str, value: float):
        setattr(self.parameters, attribute_name, value)
        self.__update_model_from_gui()

    def __update_weights(self, attribute_name: str, value: float):
        if ('FCF' in attribute_name) or ('ROIC' in attribute_name):
            new_weight = value / 2
            self.parameters.weights['FCF'] = new_weight * 1e-2
            self.parameters.weights['ROIC'] = new_weight * 1e-2
        else:
            self.parameters.weights[attribute_name] = value*1e-2

        self.parameters.weights_sum = sum(self.parameters.weights.values())
        self.ui.l_sum_weights.setText(f"{np.round(self.parameters.weights_sum * 1e2, decimals=2)} %")

    def __update_gui_from_model(self):
        """ Méthode appelée lors de l'initialisation ou d'une réinitialisation"""
        # Réinitialisation des paramètres
        weights = {'Revenues': 0.05,
                   'GrossMargin': 0.15,
                   'FCF': 0.1,
                   'ROIC': 0.1,
                   'Debt': 0.05,
                   'Piotroski': 0.1,
                   'Buyback': 0.05,
                   'Value': 0.4
                   }

        weights_sum = sum(weights.values())
        seuil_freq_select = 0.5
        taux_sans_risque = 0.05

        start_year = 2017
        end_year = 2025

        nbe_monte_carlo = 2500
        nbe_stocks = 20

        if self.data_file_path:
            analysis = Analysis(filepath=self.data_file_path)

            self.parameters.weights = weights
            self.parameters.weights_sum = weights_sum
            self.parameters.seuil_weights = seuil_freq_select
            self.parameters.taux_sans_risque = taux_sans_risque
            self.parameters.start_year = start_year
            self.parameters.end_year = end_year

            self.parameters.analysis = analysis
            self.parameters.nbe_monte_carlo = nbe_monte_carlo
            self.parameters.nbe_stocks = nbe_stocks

        else:
            analysis = None

            self.parameters = Parameters(script_path=Path(os.path.abspath(__file__)),
                                         weights=weights,
                                         weights_sum=weights_sum,
                                         seuil_freq_select=seuil_freq_select,
                                         taux_sans_risque=taux_sans_risque,
                                         start_year=start_year,
                                         end_year=end_year,
                                         analysis=analysis,
                                         nbe_monte_carlo=nbe_monte_carlo,
                                         nbe_stocks=nbe_stocks
                                         )

        # Réinitialisation des résultats d'analyse
        self.results_storage = {}

        self.all_ranks_df = None
        self.quality_results_series = None
        self.robust_quality_results = None
        self.quality_weights_sum = 0.0

        self.factorial_model_results = None
        self.final_value_score_results = None
        self.weights_allocation = None

        self.revenues_results = None
        self.gross_margin_results = None
        self.cfoa_results = None
        self.groc_results = None
        self.buyback_results = None
        self.piotroski_results = None
        self.piotroski_results_reduced = None
        self.value_results = None
        self.value_sector_stats = None
        self.debt_series = None

        self.detailed_ranking_df = None

        # Réinitialisation des fenêtres graphiques
        self.graphics_window = None
        self.correlation_matrix = None

        # Réinitialisation de l'interface
        self.ui.rb_type_freq_selec_dur.setChecked(True)
        self.ui.sb_seuil_freq_select.setValue(self.parameters.seuil_freq_select)
        self.ui.sb_taux_sans_risque.setValue(self.parameters.taux_sans_risque)

        self.ui.rb_export_results_no.setChecked(True)
        self.ui.le_path_data_export.hide()
        self.ui.pb_select_export_repo.hide()
        self.ui.label_export_path.hide()

        self.ui.pb_export_results.setEnabled(False)

        self.ui.sb_seuil_freq_select.setValue(self.parameters.seuil_freq_select * 1e2)
        self.ui.sb_taux_sans_risque.setValue(self.parameters.taux_sans_risque * 1e2)

        self.ui.sp_ca.setValue(self.parameters.weights['Revenues'] * 1e2)
        self.ui.sp_debt.setValue(self.parameters.weights['Debt'] * 1e2)
        self.ui.sp_buyback.setValue(self.parameters.weights['Buyback'] * 1e2)
        self.ui.sp_gross_margin.setValue(self.parameters.weights['GrossMargin'] * 1e2)
        self.ui.sp_roic.setValue(self.parameters.weights['ROIC'] * 1e2 + self.parameters.weights['FCF'] * 1e2)
        self.ui.sp_piotroski.setValue(self.parameters.weights['Piotroski'] * 1e2)
        self.ui.sp_value.setValue(self.parameters.weights['Value'] * 1e2)

        self.ui.l_sum_weights.setText(f"{np.round(sum(self.parameters.weights.values()) * 1e2, decimals=2)} %")

        self.ui.sp_start_year.setValue(self.parameters.start_year)
        self.ui.sp_end_year.setValue(self.parameters.end_year)

        self.ui.sb_nbe_monte_carlo.setValue(self.parameters.nbe_monte_carlo)
        self.ui.sb_nb_stocks.setValue(self.parameters.nbe_stocks)

        # Réinitialisation des tableaux d'affichage
        self.__clear_all_table_views()

    def __update_model_from_gui(self):
        self.parameters.seuil_freq_select = self.ui.sb_seuil_freq_select.value()
        self.parameters.taux_sans_risque = self.ui.sb_taux_sans_risque.value()

        self.parameters.analysis.start_year = self.ui.sp_start_year.value()
        self.parameters.analysis.end_year = self.ui.sp_end_year.value()

        self.parameters.nbe_monte_carlo = self.ui.sb_nbe_monte_carlo.value()
        self.parameters.nbe_stocks = self.ui.sb_nb_stocks.value()

    def __open_file(self):
        data_filepath = self.open_file()
        self.ui.le_path_data_import.setText(data_filepath)

    def open_file(self) -> str:
        # Ouvre une boîte de dialogue pour choisir un fichier
        self.data_file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner un fichier",
            str(self.parameters.script_path.parent.parent / "data"),  # répertoire par défaut (vide = dossier courant)
            "Fichiers Excel (*.xlsx);;Fichiers CSV (*.csv)"
        )

        if Path(self.data_file_path).suffix.lower() == ".xlsx":
            # Chargement via ton ExcelLoader
            self.parameters.analysis = Analysis(filepath=self.data_file_path)
            # QMessageBox.information(self, "Succès", "Fichier Excel chargé avec succès.")

        elif Path(self.data_file_path).suffix.lower() == ".csv":
            self.csv_data = pd.read_csv(self.data_file_path)
            # QMessageBox.information(self, "Succès", "Fichier CSV chargé avec succès.")

        else:
            QMessageBox.warning(self, "Format non supporté",
                                f"Extension {self.data_file_path.suffix} non reconnue.")

        try:
            # Création du loader et de l’analyse
            self.parameters.analysis = Analysis(filepath=self.data_file_path)

            QMessageBox.information(self, "Succès", "Fichier chargé avec succès.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier.\n{e}")

        return self.data_file_path

    def __open_dir(self):
        """Sélectionne le répertoire d'exportation"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Sélectionnez le répertoire d'exportation",
            f"{self.ui.le_path_data_import.parent}",
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.ui.le_path_data_export.setText(directory)

            if self.results_storage:
                self.ui.pb_export_results.setEnabled(True)

    def __run_analysis(self):
        self.revenues_results = self.parameters.analysis.check_revenues_growth_and_stability()
        self.gross_margin_results = self.parameters.analysis.check_gross_margin_growth_and_stability()
        self.cfoa_results, self.groc_results = self.parameters.analysis.check_fcf_and_roic_gr()
        self.buyback_results = self.parameters.analysis.check_shares_buyback()
        self.piotroski_results, self.piotroski_results_reduced = self.parameters.analysis.compute_piotroski_score()
        self.value_results, self.value_sector_stats, _ = self.parameters.analysis.find_cheapest_stocks(risk_free_rate=0.045)
        self.debt_series = self.parameters.analysis.check_debt_solvency()

        # Initialisation de la classe RobustFinancialLoader pour normaliser les données d'entrée avant d'être
        # passées dans l'algorithme de sélection de variables robuste LASSO.
        df_list = [self.revenues_results, self.gross_margin_results,
                   self.cfoa_results, self.groc_results, self.buyback_results,
                   self.piotroski_results, self.value_results, self.debt_series ]

        loader_lasso = RobustFinancialLoader(dfs=df_list, mode='Z-score', feature_names=['Revenues', 'Gross Margin',
                                                                                         'CFOA', 'ROIC', 'Buyback',
                                                                                         'Piotroski', 'Value',
                                                                                         'Solvency'])
        X_df_lasso = loader_lasso.get_dataframe()

        corr_matrix = X_df_lasso.corr()
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        masked_corr = corr_matrix.where(mask)

        # 4. Trouver la corrélation max en valeur absolue (pour détecter les corrélations négatives fortes)
        abs_corr = masked_corr.abs()
        max_corr_val = abs_corr.max().max()

        idx = abs_corr.stack().idxmax()
        print(f"Corrélation maximale absolue : {max_corr_val:.4f} entre {idx[0]} et {idx[1]}")

        self.results_storage = {'Revenues': self.revenues_results,
                                'GrossMargin': self.gross_margin_results,
                                'FCF': self.cfoa_results,
                                'ROIC': self.groc_results,
                                'Buyback': self.buyback_results,
                                'Piotroski': self.piotroski_results,
                                'Value': self.value_results,
                                'Debt': self.debt_series
                                }
        start_year = 2017
        start_year_dt = f'{start_year}-01-01'
        end_year_dt = f'{start_year + 2}-01-01'
        tickers_list = self.revenues_results.index.to_list()

        # Téléchargement par batch pour éviter les timeouts
        batch_size = 20
        all_prices = []
        all_volumes = []

        for i in range(0, len(tickers_list), batch_size):
            batch = tickers_list[i:i + batch_size]
            print(f"   Batch {i // batch_size + 1}/{(len(tickers_list) - 1) // batch_size + 1}...")

            try:
                data = yf.download(
                    batch,
                    start=start_year_dt,
                    end=end_year_dt,
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )

                if len(batch) == 1:
                    prices_batch = data['Close'].to_frame(name=batch[0])
                    volumes_batch = data['Volume'].to_frame(name=batch[0])
                else:
                    prices_batch = data['Close']
                    volumes_batch = data['Volume']

                all_prices.append(prices_batch)
                all_volumes.append(volumes_batch)

            except Exception as e:
                print(f"   ⚠️  Erreur batch {i // batch_size + 1}: {e}")
                continue

        # Concatenation
        prices = pd.concat(all_prices, axis=1)
        volumes = pd.concat(all_volumes, axis=1)
        sector_mapping = self.revenues_results['Sector']

        # Récupération du vecteur d'observation y : rendements excédentaires par secteur
        robust_y_gen = RobustTargetGenerator(start_date=start_year_dt, price_df=prices, sector_mapping=sector_mapping)
        df_returns, y = robust_y_gen.get_robust_y(start_year_dt, method='Option_A')

        # Alignement des tickers
        common_tickers = X_df_lasso.index.intersection(y.index)
        X_aligned = X_df_lasso.loc[common_tickers]
        y_aligned = y.loc[common_tickers]

        """
            Utilisation des méthodes abstraites
        """
        # Un simple changement de cette variable modifie l'intégralité du comportement mathématique
        CHOIX_SOLVEUR = SolverType.ADMM_ADAPTIVE_HUBER_LASSO

        # Instanciation dynamique via la Factory
        screener_regressor = RobustRegressorFactory.create_solver(
            solver_type=CHOIX_SOLVEUR,
            lambda_=0.5,
            delta=1.35,
            fit_intercept=True,
            verbose=True
        )

        # Utilisation standardisée (l'intercept n'est pas pénalisé en interne)
        screener_regressor.fit(X_aligned.to_numpy(), y_aligned.to_numpy())

        # Accès propre aux données financières finales
        facteurs_clefs = screener_regressor.coef_
        alpha_moyen = screener_regressor.intercept_

        # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
        # AFFICHAGE ET EXPLOITATION DES FACTEURS SÉLECTIONNÉS
        # =====================================================================
        feature_names = X_aligned.columns.tolist()
        threshold_selection = 1e-4
        facteurs_retenus = [name for name, coef in zip(feature_names, facteurs_clefs) if
                            abs(coef) > threshold_selection]

        print("\n" + "=" * 50)
        print(f"   RÉSULTATS DE LA SÉLECTION FACTORIELLE ({CHOIX_SOLVEUR.name})   ")
        print("=" * 50)
        print(f"Intercept (Alpha moyen sectoriel) : {alpha_moyen:.5f}")
        print(f"Facteurs conservés pour le portefeuille : {facteurs_retenus}")
        print("-" * 50)

        for nom, coef in zip(feature_names, facteurs_clefs):
            statut = "[CONSERVÉ]" if abs(coef) > threshold_selection else "[EXCLU]"
            print(f"  -> {statut:<11} Facteur {nom:<15} : Coefficient β = {coef:.5f}")
        print("=" * 50)

        """
            Test du code par chemin de régularisation
        """
        optimizer = ADMMWarmStartPath(n_lambdas=50)
        optimizer.fit(X_aligned.to_numpy(), y_aligned.to_numpy())
        optimizer.plot_path(feature_names=X_aligned.columns.tolist())

        # Utilisation des meilleurs paramètres trouvés
        best_beta = optimizer.best_coef_
        best_intercept = optimizer.best_intercept_

        print(f"\n[INFO] Modèle final sélectionné via BIC Robuste.")
        print(f"Optimal Lambda : {optimizer.best_lambda_:.6f}")

        """
            Appel au code fonctionnel
        """

        # 1. Instanciation de la pipeline (n_jobs=-1 utilise automatiquement tous tes cœurs CPU)
        pipeline = HuberAdaptiveElasticNetPipeline(criterion="BIC", delta=1.345, n_jobs=5)

        # 2. Entraînement : On passe le DataFrame X et la Series y.
        # La classe va aligner les Tickers communs en interne, standardiser de façon robuste et lancer le Grid Search MP.
        pipeline.fit(X_df_lasso, y, n_l1=100)

        # =====================================================================
        # EXTRACTION ET EXPLOITATION DES FACTEURS SÉLECTIONNÉS
        # =====================================================================

        # 3. Récupération des facteurs alpha statistiquement robustes
        facteurs_retenus = pipeline.get_selected_factors(threshold=1e-4)

        print("\n" + "=" * 50)
        print("       RÉSULTATS DE LA SÉLECTION FACTORIELLE HAEN      ")
        print("=" * 50)
        print(f"Facteurs conservés pour le portefeuille : {facteurs_retenus}")
        print("-" * 50)

        # Affichage des coefficients beta pour l'IHM ou le débugging
        for nom, coef in zip(pipeline.feature_names, pipeline.beta_final):
            print(f"  -> Facteur {nom:<15} : Coefficient β = {coef:.5f}")
        print("=" * 50)

        pipeline.print_factor_significance(pipeline, X_df_lasso, y)

        pipeline.plot_regularization_path()

        self.all_ranks_df = pd.DataFrame({'Revenues': self.revenues_results['Rank'],
                                          'GrossMargin': self.gross_margin_results['Rank'],
                                          'FCF': self.cfoa_results['Rank'],
                                          'ROIC': self.groc_results['Rank'],
                                          'Buyback': self.buyback_results['Rank'],
                                          'Debt': self.debt_series,
                                          'Piotroski': self.piotroski_results['Rank'],
                                          'Value': self.value_results['Rank'],
                                          })

        self.quality_results_series = self.__make_quality_ranking()
        self.robust_quality_results_series = self.__make_sensibility_analysis()

        self.detailed_ranking_df = pd.DataFrame({'Revenues': self.revenues_results['Rank'],
                                                 'Gross Margin': self.gross_margin_results['Rank'],
                                                 'Free Cash Flow': self.cfoa_results['Rank'],
                                                 'Return On Capital': self.groc_results['Rank'],
                                                 'Debt': self.debt_series,
                                                 'Piotroski': self.piotroski_results['Rank'],
                                                 'Buyback': self.buyback_results['Rank'],
                                                 'Value': self.value_results['Rank'],
                                                 'Quality': self.quality_results_series,
                                                 'Robust Quality': self.robust_quality_results_series
                                                 })

        self.__compute_factorial_model()
        self.__display_results_in_tabs()

        if self.ui.le_path_data_export.text() != '' and self.results_storage:
            self.ui.pb_export_results.setEnabled(True)

    def __make_quality_ranking(self):
        dfs = [self.results_storage[name] for name in MainWindow.QUALITY_FACTORS]

        self.quality_weights_sum = sum([self.parameters.weights[name] for name in MainWindow.QUALITY_FACTORS])
        new_quality_weights = [self.parameters.weights[name] / self.quality_weights_sum for name in MainWindow.QUALITY_FACTORS]

        # Trouver les index communs à tous les DataFrames
        common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))

        # Mettre à jour chaque DataFrame en ne gardant que ces index
        dfs = [df.loc[common_index] for df in dfs]

        # Création du classement final Quality (i.e. sans la valorisation EBIT/TEV).
        quality_results = (new_quality_weights[0] * dfs[0]['Rank'] +
                           new_quality_weights[1] * dfs[1]['Rank'] +
                           new_quality_weights[2] * dfs[2]['Rank'] +
                           new_quality_weights[3] * dfs[3]['Rank'] +
                           new_quality_weights[4] * dfs[4] +
                           new_quality_weights[5] * dfs[5]['Rank']
                           )
        quality_results = quality_results.sort_values(ascending=False)
        quality_results.name = 'Final Quality Ranking'


        return quality_results

    def __make_sensibility_analysis(self) -> pd.Series:
        threshold = self.ui.sb_seuil_freq_select.value()
        sensitivity_results = monte_carlo_sensitivity_analysis(self.parameters,
                                                               num_simulations=self.ui.sb_nbe_monte_carlo.value())

        # Pondération du classement Quality par la fréquence de sélection dans le top X% après N simulations
        above_threshold_sel_freq = list(
            sensitivity_results[
                sensitivity_results['selection_frequency'] > threshold*1e-2].index)

        # Soit on prend toutes les actions au-dessus d'un certain seuil
        if self.ui.rb_type_freq_selec_dur.isChecked():
            robust_quality_results = self.quality_results_series.loc[self.quality_results_series.index.isin(above_threshold_sel_freq)]
        else:
            robust_quality_results = self.quality_results_series * sensitivity_results['Rank'] * 1e-2
            robust_quality_results = robust_quality_results[sensitivity_results['selection_frequency'] != 0]

        return robust_quality_results

    def __compute_factorial_model(self):
        dfs = [self.all_ranks_df['Value'], self.quality_results_series, self.all_ranks_df['Piotroski']]
        common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))
        dfs = [df.loc[common_index] for df in dfs]

        # Création du classement final (Value + Quality + Piotroski) basé sur le modèle factoriel ci-dessus.
        self.factorial_model_results = (self.parameters.weights['Value'] * dfs[0] +
                                   self.quality_weights_sum * dfs[1] +
                                   self.parameters.weights['Piotroski'] * dfs[2]
                                   )
        self.factorial_model_results = self.factorial_model_results.sort_values(ascending=False)
        self.factorial_model_results.name = 'Factor Model Ranking'

        factorial_model_sector_results = self.parameters.analysis.data_general['Sector'].loc[self.factorial_model_results.index]
        self.factorial_model_results = pd.DataFrame({"Factorial Model Score": self.factorial_model_results,
                                                  "Sector": factorial_model_sector_results})

        # Calcul des poids à allouer pour les 20 premières actions
        nb_stocks = self.ui.sb_nb_stocks.value()
        top_20_sum = sum(self.factorial_model_results.iloc[:nb_stocks]['Factorial Model Score'].values)
        self.weights_allocation = self.factorial_model_results.iloc[:nb_stocks]['Factorial Model Score'].values / top_20_sum * 1e2

        total_nbe_stocks = len(self.factorial_model_results.index.to_list())
        allocation_series_weight = pd.Series(np.zeros(total_nbe_stocks), index=self.factorial_model_results.index)
        allocation_series_amount = pd.Series(np.zeros(total_nbe_stocks), index=self.factorial_model_results.index)

        allocation_series_weight.iloc[:nb_stocks] = self.weights_allocation
        allocation_series_amount.iloc[:nb_stocks] = self.ui.sb_capital.value() * self.weights_allocation * 1e-2

        self.factorial_model_results['Weights (%)'] = allocation_series_weight
        self.factorial_model_results['Amount (€)'] = allocation_series_amount

        # Récupération des 20% meilleures actions seulement
        top_value_results_threshold = self.factorial_model_results['Factorial Model Score'].quantile(0.8)
        final_value_score_results = self.factorial_model_results[self.factorial_model_results['Factorial Model Score']
                                                                 > top_value_results_threshold]
        final_value_sector_results = self.parameters.analysis.data_general['Sector'].loc[final_value_score_results.index]

        self.final_value_score_results = pd.DataFrame({"Score": final_value_score_results['Factorial Model Score'],
                                                       "Secteur": final_value_sector_results
                                                       })

    def __show_graphics_window(self):
        """Affiche la fenêtre graphique avec la matrice de corrélation"""
        if self.all_ranks_df is None:
            QMessageBox.warning(self, "Aucune donnée",
                                "Veuillez d'abord lancer l'analyse avant de visualiser les graphiques.")
            return

        if self.graphics_window is None:
            self.graphics_window = GraphicsWindow(self)

        if self.correlation_matrix is None:
            # Calculer et afficher la matrice de corrélation
            self.correlation_matrix = self.__check_factorial_correlation()

        self.graphics_window.plot_correlation_matrix(self.correlation_matrix)
        self.graphics_window.show()

    def __check_factorial_correlation(self):
        """Calcule et retourne la matrice de corrélation des facteurs"""
        # Préparer les dataframes individuels pour chaque facteur
        rank_list = [self.all_ranks_df[col] for col in self.all_ranks_df.columns]

        # L'index est déjà commun à tous les dataframes
        common_index = rank_list[0].index
        num_stocks = len(common_index)

        print(f"Nombre d'actions dans l'analyse de corrélation : {num_stocks}")

        # 3. Vérifier qu'il y a suffisamment de données
        if num_stocks < 2:
            QMessageBox.warning(self, "Données insuffisantes",
                                f"Seulement {num_stocks} action(s) commune(s) trouvée(s). "
                                "Au moins 2 actions sont nécessaires pour calculer la corrélation.")
            return None, num_stocks

        correlation_matrix = self.all_ranks_df.corr(method='spearman')

        return correlation_matrix

    def __display_results_in_tabs(self):
        """Affiche tous les résultats dans les QTableView correspondants"""        # Mapping des données vers les QTableView spécifiques
        table_data_mapping = [
            (self.ui.tableView_revenues, self.revenues_results),
            (self.ui.tableView_gross_margin, self.gross_margin_results),
            (self.ui.tableView_fcf, self.cfoa_results),
            (self.ui.tableView_roic, self.groc_results),
            (self.ui.tableView_debt, self.debt_series.to_frame(name='Debt Rank')),
            (self.ui.tableView_buyback, self.buyback_results),
            (self.ui.tableView_piotroski, self.piotroski_results),
            (self.ui.tableView_value, self.value_results),
            (self.ui.tableView_quality, self.quality_results_series.to_frame(name='Quality Score')),
            (self.ui.tableView_final_ranking, self.factorial_model_results),
            (self.ui.tableView_robust_quality, self.robust_quality_results),
            (self.ui.tableView_ranking_detailed, self.detailed_ranking_df)
        ]

        # Afficher chaque tableau
        for tableView, data in table_data_mapping:
            if data is not None:
                self.__setup_table_view(tableView, data)

    def __setup_table_view(self, tableView: QTableView, data: pd.DataFrame):
        """Configure un QTableView avec les données"""

        # Convertir en DataFrame si nécessaire
        if isinstance(data, pd.Series):
            df = data.to_frame()
        else:
            df = data

        # Créer et définir le modèle
        model = PandasTableModel(df)
        tableView.setModel(model)

        # Configuration de l'affichage
        tableView.setSortingEnabled(True)
        tableView.setAlternatingRowColors(True)
        tableView.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)

        # Configuration des en-têtes
        header = tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setDefaultSectionSize(120)

        # Ajuster les colonnes après un court délai
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.__resize_table_columns(tableView))

    def __resize_table_columns(self, tableView: QTableView):
        """Ajuste automatiquement les colonnes d'un QTableView"""
        header = tableView.horizontalHeader()
        for column in range(tableView.model().columnCount()):
            header.resizeSection(column, header.sectionSizeHint(column))

    def __clear_all_table_views(self):
        """Vide tous les tableaux d'affichage"""
        table_views = [
            self.ui.tableView_revenues,
            self.ui.tableView_gross_margin,
            self.ui.tableView_fcf,
            self.ui.tableView_roic,
            self.ui.tableView_debt,
            self.ui.tableView_buyback,
            self.ui.tableView_piotroski,
            self.ui.tableView_value,
            self.ui.tableView_quality,
            self.ui.tableView_final_ranking,
            self.ui.tableView_robust_quality,
            self.ui.tableView_ranking_detailed
        ]

        for tableView in table_views:
            tableView.setModel(None)

    def __export_data(self):
        df_list = [self.revenues_results,
                   self.gross_margin_results,
                   self.cfoa_results,
                   self.groc_results,
                   self.piotroski_results,
                   self.buyback_results,
                   self.value_results,
                   self.value_sector_stats,
                   self.quality_results_series,
                   self.robust_quality_results_series,
                   self.final_value_score_results
                   ]

        df_list_names = ['Revenues', 'Gross Profit', 'Free Cash Flow', 'Return On Capital', 'Piotroski Score',
                         'Shares Buyback', 'EBIT_ov_TEV (Full)', 'EBIT_ov_TEV (Sector)',
                         'Quality Ranking', 'Robust Quality', 'Factor Model Ranking']

        if self.ui.rb_export_results_yes.isChecked():
            today = datetime.today().strftime("%d_%m_%Y")
            # filename = f"{today}_{'US' if US_stocks else 'EU'}_value_results.xlsx"
            filename = f"{today}_US_value_results.xlsx"

            write_excel(filepath=Path(self.ui.le_path_data_export.text()) / filename,
                        dfs=df_list,
                        sheet_names=df_list_names
                        )

    def __run_backtest(self):
        tickers = self.final_value_score_results.index.to_list()
        start_date = f"{self.ui.sb_start_year_backtest.value()}-01-01"
        end_date = f"{self.ui.sb_end_year_backtest.value()}-01-01"

        # Initialisation
        self.backtest_engine = PortfolioMetricsEngine(parameters=self.parameters,
                                                      tickers_list=tickers,
                                                      start_date=start_date,
                                                      end_date=end_date
                                                      )

        # TODO : faire le backtest en se basant sur la date de début + 8 ans pour le classement
        # 2016 - 2022 pour acheter de 2023 à 2024.
        # 2017 - 2023 pour acheter de 2024 à 2025.
        # Décaler d'1 an tout en stockant les résultats à chaque fois
        # Peut-être faire un dataframe avec comme index les différentes années et en colonne les métriques

        # Lancement du backtest
        self.backtest_engine.compute_metrics(capital=self.ui.sb_capital.value(),
                                             stock_weights=self.weights_allocation,
                                             risk_free_rate=self.ui.sb_taux_sans_risque.value()
                                             )

        self.backtest_engine.compute_benchmark_metrics(is_US=True,
                                                       risk_free_rate=self.ui.sb_taux_sans_risque.value()
                                                       )

    # def __compute_allocation(self) -> tuple[pd.Series, pd.Series]:
    #     nbe_stocks = len(self.revenues_results.index.to_list())
    #     allocation_series_weight = pd.Series(np.zeros(nbe_stocks), index=self.revenues_results.index)
    #     allocation_series_amount = pd.Series(np.zeros(nbe_stocks), index=self.revenues_results.index)
    #
    #     wanted_nbe_stocks = int(self.ui.sb_nb_stocks.value())
    #
    #     alloc_sum = sum(self.final_value_score_results.iloc[:wanted_nbe_stocks]['Value Score'])
    #     alloc_weights = self.final_value_score_results.iloc[:wanted_nbe_stocks]['Value Score'] / alloc_sum
    #     amount_by_stock = self.ui.sb_capital.value() * alloc_weights
    #
    #     allocation_series_weight.iloc[:wanted_nbe_stocks] = alloc_weights
    #     allocation_series_amount.iloc[:wanted_nbe_stocks] = amount_by_stock
    #
    #     return allocation_series_weight, allocation_series_amount




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Fondamental Screener Tool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    # TODO : terminer l'export au format .XLSX des résultats
