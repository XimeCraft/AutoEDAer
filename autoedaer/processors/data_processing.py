import pandas as pd
import numpy as np

class FeatureSelection:

    @staticmethod
    def select_columns_by_name(data: pd.DataFrame, names: list) -> pd.DataFrame:
        """Select by names"""
        return data[names]
    
    @staticmethod
    def select_columns_by_type(data: pd.DataFrame, dtype: type) -> pd.DataFrame:
        """Select by data type"""
        return data.select_dtypes(include=[dtype])
    
    @staticmethod
    def select_columns_by_condition(data: pd.DataFrame, condition: callable) -> pd.DataFrame:
        """Select by condition
        example: select_columns_by_condition(df, lambda col: col.sum() > 100)
        """
        return data.loc[:, condition(data)]

class DataProcess:

    @staticmethod
    def infer_numeric_dtype(series: pd.Series) -> type:
        """Infer numeric data to the most suitable numeric type, reduce the memory usage."""
        if series.dtype.kind in 'iu':
            if series.min() >= np.iinfo(np.int8).min and series.max() <= np.iinfo(np.int8).max:
                return np.int8
            elif series.min() >= np.iinfo(np.int16).min and series.max() <= np.iinfo(np.int16).max:
                return np.int16
            elif series.min() >= np.iinfo(np.int32).min and series.max() <= np.iinfo(np.int32).max:
                return np.int32
            elif series.min() >= np.iinfo(np.int64).min and series.max() <= np.iinfo(np.int64).max:
                return np.int64
        elif series.dtype.kind == 'f':
            return np.float32 if np.all(series.dropna() == series.dropna().astype(np.float32)) else np.float64
        return series.dtype

    @staticmethod
    def convert_data_type(df: pd.DataFrame, cols: list, dtype: type) -> pd.DataFrame:
        """Convert single or multiple columns to specified data type"""
        for col in cols:
            try:
                if dtype in ('int', 'float'):
                    df[col] = df[col].astype(dtype).apply(DataProcess.infer_numeric_dtype)
                elif dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
            except ValueError as e:
                print(f"Failed to convert column {col} to {dtype}: {e}")
        return df
        
    @staticmethod
    def remove_constant_or_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove constant or unique columns"""
        return df.loc[:, ~df.nunique().isin([1, df.shape[0]])]
    
    @staticmethod
    def remove_duplicate_rows_by_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Remove duplicate rows in specific columns"""
        return df.drop_duplicates(subset=columns, inplace=True)
    
    @staticmethod
    def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, on: list, how: str = 'outer', direction: str = 'horizontal') -> pd.DataFrame:
        """
        Merge two dataframes horizontally or vertically, return combined dataframe.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            on: Column(s) to join on for horizontal merge. Defaults to None.
            how: Type of merge to be performed. Defaults to 'outer'.
            direction: Direction of merge ('vertical' or 'horizontal'). Defaults to 'vertical'.
        """

        if direction == 'horizontal':
            if on:
                return pd.merge(df1, df2, on=on, how=how)
            else:
                raise ValueError("'on' parameter must be specified for horizontal merge")
        elif direction == 'vertical':
            return pd.concat([df1, df2], axis=0, join='outer', ignore_index=True)
        else:
            raise ValueError("Invalid direction. Choose 'vertical' or 'horizontal'")

    @staticmethod
    def bin_columns(df: pd.DataFrame, bins_meta: dict) -> pd.DataFrame:
        """Bin columns into bins.
        bins_meta: dict, e.g. {'height': {'bins': 3, 'labels': ['low', 'medium', 'high']}}
        """

        for col, meta in bins_meta.items():
            bins, labels = meta['bins'], meta['labels']
            df[col] = pd.cut(df[col], bins=bins, labels=labels)

        return df
    
class MissingValue:
    @staticmethod
    def remove_missing_columns(df: pd.DataFrame, columns: list, max_missing_rate: float) -> pd.DataFrame:
        """Remove columns contains at most rate of threshold missing values.
        
        Args:
            columns: List of columns to check for missing values. If None, check all columns.
            missing_rate: Maximum allowed proportion of missing values to keep the column, 0 to 1.
                * 0.0 means keep all columns regardless of missing values
                * 1.0 means keep only columns with all non-missing values
        """

        if not 0 <= max_missing_rate <= 1:
            raise ValueError("max_missing_rate must be between 0 and 1")
        
        columns = columns or df.columns
        if max_missing_rate == 1:
            return df
        
        min_non_missing = int(df.shape[0] * (1 - max_missing_rate))
        return df.dropna(axis=1, thresh=min_non_missing, subset=columns)
    
    @staticmethod
    def fill_na_by_statistics(df: pd.DataFrame, columns: list = None, method: str = 'mean') -> pd.DataFrame:
        """Fill missing values in specified columns with statistical methods, or custom value.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
            method: Method to fill missing values. Defaults to 'mean'.
        """

        if method not in ('mean', 'median', 'mode'):
            raise ValueError("Invalid method. Choose 'mean', 'median', or 'mode'")
        
        columns = columns or df.columns

        if method == 'median':
            return df.fillna(df.median())
        elif method == 'mode':
            return df.fillna(df.mode())
        else:
            return df.fillna(df.mean())
    
    @staticmethod
    def fill_na_by_rolling(df: pd.DataFrame, columns: list, method: str = 'mean', window_size: int = None, center: bool = False) -> pd.DataFrame:
        """Fill missing values in specified columns with rolling methods. 
        The window size and center set to None is set to the whole series by default. Keep fill_na_by_statistics for more effient usage.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
            method: Method to fill missing values. Defaults to 'mean'.
            window_size: Window size for rolling. Defaults to None as the whole series.
            center: Whether to set the window at the center of the rolling. Defaults to False to the left.
        """

        if method not in ('mean', 'median', 'mode'):
            raise ValueError("Invalid method. Choose 'mean', 'median', or 'mode'")
        
        columns = columns or df.columns
        window_size = window_size or df.shape[0]

        return df[columns].fillna(df.rolling(window=window_size, center=center).transform(method))
        

    @staticmethod
    def fill_na_by_group(df: pd.DataFrame, columns: list, by: list, method: str = 'mean') -> pd.DataFrame:
        """Fill missing values in specified columns with statistical methods, or custom value.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
            method: Method to fill missing values. Defaults to 'mean'.
        """

        if method not in ('mean', 'median', 'mode'):
            raise ValueError("Invalid method. Choose 'mean', 'median', or 'mode'")
        
        columns = columns or df.columns

        return df[columns].fillna(df.groupby(by)[columns].transform(method))
        
    @staticmethod
    def fill_na_by_custom_value(df: pd.DataFrame, columns: list = None, value: any = None) -> pd.DataFrame:
        """Fill missing values in specified columns with custom value.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
            value: Value to fill missing values. Defaults to None.
        """
        
        columns = columns or df.columns
        return df.fillna(value)
    
    @staticmethod
    def fill_na_forward_or_backward(df: pd.DataFrame, columns: list = None, method: str = 'ffill') -> pd.DataFrame:
        """Fill missing values in specified columns with forward or backward value.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
        """
        
        if method not in ('ffill', 'bfill'):
            raise ValueError("Invalid method. Choose 'ffill', 'bfill'")
        
        columns = columns or df.columns
        return df.fillna(method=method)
        
    @staticmethod
    def fill_na_by_interpolate(df: pd.DataFrame, columns: list = None, method: str = 'linear') -> pd.DataFrame:
        """Fill missing values in specified columns with linear interpolation.
        
        Args:
            columns: List of columns to fill missing values. If None, fill all columns.
            method: Method of interpolation. Defaults to 'linear'.
        """

        if method not in ('linear', 'quadratic', 'cubic'):
            raise ValueError("Invalid method. Choose 'linear', 'quadratic', or 'cubic'")
        
        columns = columns or df.columns
        return df.interpolate(method=method)
    
    #TODO: fill na by KNN，ML prediction, etc.

class 
        

        