import pandas as pd
import requests


class DataImport:

    @staticmethod
    def imoprt_url(url):
        """Import data from a URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame.from_dict(data, orient="index")
        except (requests.exceptions.RequestException, ValueError, Exception) as e:
            print(f"Failed to import data from {url}: {e}")
        return None

    @staticmethod
    def import_file(path, sep=None):
        """Import data from a local file in CSV, EXCEL, TXT, etc."""
        try:
            if path.endswith(".csv"):
                return pd.read_csv(path)
            elif path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt")):
                return pd.read_excel(path)
            elif path.endswith((".txt")):
                return pd.read_csv(path, sep="\t")
        except Exception as e:
            print(f"Failed to import data from {path}: {e}")
        return None

    