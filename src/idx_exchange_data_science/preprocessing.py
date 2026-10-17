import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin



def log_removed_rows(step, before, after):
    """Summarize the number and percentage of rows removed.

    Args:
        step (str): Name of the cleaning step.
        before (int): Number of rows before cleaning.
        after (int): Number of rows afer cleaning.
    
    Returns: 
        dict: Step name, initial count, removed count and percentage, and remaining count.
    """
    removed = before - after
    percentage = ((removed / before) * 100) if before > 0 else 0
    return {"step": step, "Before": before, "Removed": removed ,
            "Removed (%)": round(percentage, 4), "Remaining": after}


def initial_cleaning(df):
    """Apply fixed cleaning rules: 
    Keeps California single-family residential records, converts listing
    and closing dates to datetime, removes duplicate transactions with
    valid addresses, removes closing dates that precede listing dates,
    and removes missing or nonpositive ClosePrice values.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Cleaned records and a log
            summarizing row removals for each filtering step.    
    """
    # Create a copy of the original DateFrame
    df = df.copy()
    # Create an empty list to store a record of each cleaning step
    cleaning_log = []

    def log_step(step, before):
        # Add information about the cleaning step to the log
        cleaning_log.append(log_removed_rows(step, before, len(df)))
        
    # Filter CA records
    before = len(df)
    df = df[df["StateOrProvince"] == "CA"].copy()
    log_step("California filtering", before)

    # Convert dates
    for column in ["ListingContractDate", "CloseDate"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # Remove duplicate transactions using valid addresses
    before = len(df)
    invalid_addresses = ["0 0", "0 .", "N/A", "0 ,", "0 x", "E", "1 -", "0 --", "/", "0 I"]

    addresses = df["UnparsedAddress"].astype("string").str.strip()
    
    valid_address = (addresses.notna() &\
                ~addresses.isin(invalid_addresses) & addresses.ne(""))

    duplicates = (valid_address &\
                  df.assign(UnparsedAddress=addresses).duplicated(
                      subset=["UnparsedAddress", "CloseDate"], keep="first"))
    df = df.loc[~duplicates].copy()
    log_step("Duplicate removal", before)
    
    # Keep single-family residential properties
    before = len(df)

    # Keep single-family residential properties
    df = df[(df["PropertyType"] == "Residential") &\
             (df["PropertySubType"] == "SingleFamilyResidence")].copy()

    log_step("Property-type filtering", before)

    # Remove inconsistent closing dates
    before = len(df)

    invalid_date_order = (df["CloseDate"] < df["ListingContractDate"])

    df = df.loc[~invalid_date_order].copy()
    log_step("Date consistency", before)
    

    # Remove missing or invalid target values
    before = len(df)

    df = df[df["ClosePrice"].notna() &
            (df["ClosePrice"] > 0)].copy()

    log_step("Target cleaning", before)

    return df, pd.DataFrame(cleaning_log)

def select_training_window(df, n_months, validation_month):
    """Select the months immediately before the validation month.

    Args:
        df (pd.DataFrame): Training records containing SourceMonth.
        n_months (int): Number of calendar months to include.
        validation_month (str): Validation month.

    Returns:
        pd.DataFrame: A copy of records within the selected training window.
    """
    if n_months < 1:
        raise ValueError("n_months must be positive.")

    months = pd.PeriodIndex(df["SourceMonth"], freq="M")
    validation_month = pd.Period(validation_month, freq="M")

    mask = (
        (months >= validation_month - n_months) &
        (months < validation_month)
    )
    return df.loc[mask].copy()


def clean_living_area(df):
    """Remove records with zero or negative living area and retains missing values.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        tuple: Cleaned DataFrame and a dictionary summarizing
            the number and percentage of rows removed.
    """
    df = df.copy()

    df["LivingArea"] = pd.to_numeric(df["LivingArea"], errors="coerce")
    before = len(df)

    # Remove zero and negative LivingArea values, but retain missing values
    df = df[df["LivingArea"].isna() | (df["LivingArea"] > 0)].copy()

    log = log_removed_rows("Zero and negative LivingArea", before, len(df))

    return df, log

def filter_close_price(df, lower_threshold, upper_threshold):
    """Remove ClosePrice values outside the specified thresholds.

    Args:
        df (pd.DataFrame): Contains property records.
        lower_threshold (float): Minimum accepted ClosePrice.
        upper_threshold (float): Maximum accepted ClosePrice.

    Returns:
        tuple: Filtered DataFrame and a dictionary summarizing
            the number and percentage of rows removed.
    """
    df = df.copy()

    before = len(df)

    df = df[df["ClosePrice"].between(lower_threshold, upper_threshold)].copy()
    log = log_removed_rows("ClosePrice outlier filtering", before, len(df))
    
    return df, log

def clean_lot_size(df):
    """Clean lot-size values and recover missing square footage.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        tuple: Cleaned DataFrame and the number of records
            with recovered lot-size square footage.
    """
    df = df.copy()

    # Convert both lot-size columns to numeric
    lot_columns = ["LotSizeSquareFeet", "LotSizeAcres"]

    for column in lot_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Treat zeros and negative values as missing
        df.loc[df[column] <= 0, column] = np.nan

    # Identify records where square footage is missing but acres is valid
    missing_lotsize_sqft = (df["LotSizeSquareFeet"].isna() & df["LotSizeAcres"].notna())

    # Recover missing square footage using acres
    recovered_rows = missing_lotsize_sqft.sum()
    df.loc[missing_lotsize_sqft , "LotSizeSquareFeet"] = ( 
           df.loc[missing_lotsize_sqft, "LotSizeAcres"] * 43560)

    # Drop the redundant acreage feature
    df = df.drop(columns=["LotSizeAcres"])

    return df, recovered_rows

def clean_postal_code(df):
    """Standardize postal codes to five-digit ZIP code strings.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        pd.DataFrame: Dataset with standardized postal codes.
    """
    df = df.copy()

    # Convert postal codes to strings and remove whitespace
    postal = df["PostalCode"].astype("string").str.strip()

    # Extract the first five digits from ZIP+4 codes
    postal = postal.str.replace(r"^(\d{5})-\d{4}$", r"\1", regex=True)

    # Keep only valid five-digit formats
    df["PostalCode"] = postal.where(postal.str.fullmatch(r"\d{5}"))

    return df

def clean_coordinates(df):
    """Identify and invalidate coordinates outside California bounds.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        pd.DataFrame: Dataset with invalid coordinates
            replaced by NaN.
    """
    df = df.copy()
    # Convert lag and lat to numeric
    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Identify latitude values outside CA bounds
    invalid_latitude = (
        df["Latitude"].notna() &
        ~df["Latitude"].between(32.5, 42.1)
    )

    # Identify longitude values outside CA bounds
    invalid_longitude = (
        df["Longitude"].notna() &
        ~df["Longitude"].between(-124.5, -114.0)
    )

    # Mark both coordinates missing if either is invalid
    invalid_pair = invalid_latitude | invalid_longitude

    df.loc[invalid_pair, ["Latitude", "Longitude"]] = np.nan

    return df


def impute_coordinates(df, zip_coordinates, county_coordinates, overall_coordinates):
    """Impute missing coordinates using training-derived medians.
    Fills missing latitude and longitude values using ZIP-level
    medians, followed by county-level and overall medians.

    Args:
        df (pd.DataFrame): Contains property records.
        zip_coordinates (pd.DataFrame): Median coordinates by ZIP.
        county_coordinates (pd.DataFrame): Median coordinates by county.
        overall_coordinates (pd.Series): Overall median coordinates.

    Returns:
        tuple: DataFrame with imputed coordinates and the number
            of records with missing coordinates recovered.
    """
    df = df.copy()

    missing_before = df["Latitude"].isna() | df["Longitude"].isna()

    # Look up training-derived median coordinates
    for column in ["Latitude", "Longitude"]:
        # Zip fallback
        zip_medians = df["PostalCode"].map(zip_coordinates[column])
        df[column] = df[column].fillna(zip_medians)
        # County fallback
        county_medians = df["CountyOrParish"].map(county_coordinates[column])
        df[column] = df[column].fillna(county_medians)
        # Overall fallback
        df[column] = df[column].fillna(overall_coordinates[column])

    missing_after = df["Latitude"].isna() | df["Longitude"].isna()

    recovered = int(missing_before.sum() - missing_after.sum())

    return df, recovered



class CoordinateImputer(BaseEstimator, TransformerMixin):
    """Fill missing coordinates using ZIP, county, then overall medians."""

    def fit(self, X, y=None):
        coordinates = ["Latitude", "Longitude"]

        self.zip_medians_ = (
            X.groupby("PostalCode")[coordinates].median()
        )

        self.county_medians_ = (
            X.groupby("CountyOrParish")[coordinates].median()
        )

        self.overall_medians_ = X[coordinates].median()

        return self

    def transform(self, X):
        imputed_df, _ = impute_coordinates(
            X,
            self.zip_medians_,
            self.county_medians_,
            self.overall_medians_,
        )

        return imputed_df

def add_binary_missing_indicators(df):
    """Add missingness indicators for each binary property feature.
    Preserve original missing values for later imputation.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        pd.DataFrame: Dataset with missingness indicators added.
    """
    df = df.copy()

    binary_columns = ["AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"] 

    for column in binary_columns:
        # Flag missing values: 1 = missing, 0 = non-missing
        df[column + "_Missing"] = df[column].isna().astype(int)

    return df

def add_property_age_at_sale(df):
    """Calculate property age at the time of sale.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        pd.DataFrame: Dataset with PropertyAgeAtSale added.
    """
    df = df.copy()

    # Convert YearBuilt to numeric
    year_built = pd.to_numeric(df["YearBuilt"], errors="coerce")
    # Calculate age using the sale year and construction year 
    df["PropertyAgeAtSale"] = df["CloseDate"].dt.year - year_built
    # Mark negative ages as missing 
    df.loc[df["PropertyAgeAtSale"] < 0, "PropertyAgeAtSale"] = np.nan

    return df


def clean_property_characteristics(df):
    """Identify invalid property counts using provisional ranges.
    Converts property characteristics to numeric and replaces
    out-of-range or non-integer values with NaN. 

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        tuple: Cleaned DataFrame and a dictionary containing
            the number of invalid values for each feature.
    """
    df = df.copy()

    property_columns = [ "BedroomsTotal", "BathroomsTotalInteger",  "GarageSpaces", "ParkingTotal"]

    # Convert property characteristics to numeric
    for column in property_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Define provisional valid ranges for property counts
    valid_ranges = {"BedroomsTotal": (0, 20),
                    "BathroomsTotalInteger": (1, 20),
                    "GarageSpaces": (0, 20),
                    "ParkingTotal": (0, 100)}

    invalid_logs = {}

    # Replace invalid property counts with NaN
    for column, (lower, upper) in valid_ranges.items():

        # Identify non-missing values below the minimum or non-integers
        invalid = (df[column].notna() &
            ((df[column] < lower) | (df[column] % 1 != 0)))

        if upper is not None:
            invalid |= df[column] > upper

        invalid_logs[column] = int(invalid.sum())
        df.loc[invalid, column] = np.nan

    return df, invalid_logs

def clean_categorical_features(df):
    """Standardize categorical features and handle missing values.

    Args:
        df (pd.DataFrame): Contains property records.

    Returns:
        pd.DataFrame: Dataset with standardized categorical features.
    """
    df = df.copy()

    categorical_columns = ["City", "CountyOrParish", "PostalCode"]

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")\
                               .astype(str)\
                               .str.strip()
        

    return df


