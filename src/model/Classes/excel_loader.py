import pandas as pd


class ExcelLoader:
    def __init__(self, filepath):
        """
        Initialise le chargeur avec le chemin du fichier Excel.

        :param filepath: Chemin vers le fichier Excel (.xlsx ou .xls)
        """
        self.filepath = filepath
        self.sheet_names = self._get_sheet_names()
        self.dataframes = {}

    def reset_dataframes(self):
        self.dataframes = {}

    def _get_sheet_names(self):
        """
        Retourne la liste des noms de feuilles dans le fichier Excel.
        """
        try:
            xls = pd.ExcelFile(self.filepath)
            return xls.sheet_names
        except Exception as e:
            print(f"Erreur lors de l'accès aux feuilles : {e}")
            return []

    def load_sheet(self, sheet_name=None, usecols=None, index_col=None, skiprows=None) -> pd.DataFrame:
        """
        Charge une feuille spécifique dans un DataFrame.

        :param sheet_name: Nom de la feuille (ou indice) à charger. Si None, charge la première feuille.
        :param usecols: Colonnes à charger (ex: "A:D" ou ["A", "C"]).
        :param index_col: Colonne à utiliser comme index (numéro ou nom).
        :param skiprows: Lignes à ignorer au début (entier ou liste).
        :return: DataFrame chargé.
        """
        try:
            df = pd.read_excel(
                self.filepath,
                sheet_name=sheet_name,
                usecols=usecols,
                index_col=index_col,
                skiprows=skiprows,
                engine='openpyxl',  # explicite pour .xlsx
                na_values=["NaN", "-", "n/a", "NA"]
            )
            key = sheet_name if sheet_name else self.sheet_names[0]
            self.dataframes[key] = df
            return df
        except Exception as e:
            print(f"Erreur lors du chargement de la feuille '{sheet_name}': {e}")
            return None

    def get_loaded_dataframes(self):
        """
        Retourne les DataFrames déjà chargés.
        """
        return self.dataframes

    def get_sheet_names(self):
        """
        Retourne à nouveau les noms de feuilles.
        """
        return self.sheet_names