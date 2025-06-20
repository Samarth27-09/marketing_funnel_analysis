"""
Marketing Funnel Drop-Off Analysis - Data Exploration
====================================================
Script: 01_data_exploration.py
Purpose: Initial exploration and profiling of Olist marketing funnel datasets
Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*60)
print("OLIST MARKETING FUNNEL - DATA EXPLORATION")
print("="*60)

# Define file paths
RAW_DATA_PATH = "data/raw/"
CLOSED_DEALS_FILE = "olist_closed_deals_dataset.csv"
MQL_FILE = "olist_marketing_qualified_leads_dataset.csv"

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================

print("\n1. LOADING DATASETS")
print("-" * 30)

try:
    # Load closed deals dataset
    deals_df = pd.read_csv(RAW_DATA_PATH + CLOSED_DEALS_FILE)
    print(f"✓ Closed deals dataset loaded: {deals_df.shape[0]} rows, {deals_df.shape[1]} columns")
    
    # Load MQL dataset
    mql_df = pd.read_csv(RAW_DATA_PATH + MQL_FILE)
    print(f"✓ MQL dataset loaded: {mql_df.shape[0]} rows, {mql_df.shape[1]} columns")
    
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please ensure CSV files are in the data/raw/ folder")
    exit()

# ============================================================================
# STEP 2: BASIC DATASET INFORMATION
# ============================================================================

print("\n2. BASIC DATASET INFORMATION")
print("-" * 30)

print("\n📊 CLOSED DEALS DATASET:")
print(f"Shape: {deals_df.shape}")
print(f"Memory usage: {deals_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nColumn datatypes:")
print(deals_df.dtypes)

print("\n📊 MQL DATASET:")
print(f"Shape: {mql_df.shape}")
print(f"Memory usage: {mql_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nColumn datatypes:")
print(mql_df.dtypes)

# ============================================================================
# STEP 3: FIRST LOOK AT THE DATA
# ============================================================================

print("\n3. FIRST LOOK AT THE DATA")
print("-" * 30)

print("\n📋 CLOSED DEALS - First 5 rows:")
print(deals_df.head())

print("\n📋 MQL - First 5 rows:")
print(mql_df.head())

# ============================================================================
# STEP 4: MISSING VALUES ANALYSIS
# ============================================================================

print("\n4. MISSING VALUES ANALYSIS")
print("-" * 30)

def analyze_missing_values(df, dataset_name):
    """Analyze missing values in a dataset"""
    print(f"\n🔍 {dataset_name} - Missing Values:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✓ No missing values found!")
    
    return missing_df

deals_missing = analyze_missing_values(deals_df, "CLOSED DEALS")
mql_missing = analyze_missing_values(mql_df, "MQL")

# ============================================================================
# STEP 5: DUPLICATE VALUES ANALYSIS
# ============================================================================

print("\n5. DUPLICATE VALUES ANALYSIS")
print("-" * 30)

print(f"\n🔍 CLOSED DEALS - Duplicates:")
deals_duplicates = deals_df.duplicated().sum()
print(f"Total duplicate rows: {deals_duplicates}")

# Check for duplicate MQL IDs in deals
deals_mql_duplicates = deals_df['mql_id'].duplicated().sum()
print(f"Duplicate MQL IDs: {deals_mql_duplicates}")

print(f"\n🔍 MQL - Duplicates:")
mql_duplicates = mql_df.duplicated().sum()
print(f"Total duplicate rows: {mql_duplicates}")

# Check for duplicate MQL IDs
mql_id_duplicates = mql_df['mql_id'].duplicated().sum()
print(f"Duplicate MQL IDs: {mql_id_duplicates}")

# ============================================================================
# STEP 6: UNIQUE VALUES ANALYSIS
# ============================================================================

print("\n6. UNIQUE VALUES ANALYSIS")
print("-" * 30)

def analyze_categorical_columns(df, dataset_name):
    """Analyze categorical columns in a dataset"""
    print(f"\n📊 {dataset_name} - Categorical Columns Analysis:")
    
    # Identify object/categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    
    for col in cat_columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        
        # Show top values for columns with reasonable number of unique values
        if unique_count <= 20:
            print(f"  Values: {df[col].value_counts().index.tolist()}")
        elif unique_count <= 100:
            print(f"  Top 5: {df[col].value_counts().head().index.tolist()}")
        print()

analyze_categorical_columns(deals_df, "CLOSED DEALS")
analyze_categorical_columns(mql_df, "MQL")

# ============================================================================
# STEP 7: NUMERICAL COLUMNS ANALYSIS
# ============================================================================

print("\n7. NUMERICAL COLUMNS ANALYSIS")
print("-" * 30)

print("\n📈 CLOSED DEALS - Numerical Summary:")
numeric_deals = deals_df.select_dtypes(include=[np.number])
if len(numeric_deals.columns) > 0:
    print(numeric_deals.describe())
else:
    print("No numerical columns found")

print("\n📈 MQL - Numerical Summary:")
numeric_mql = mql_df.select_dtypes(include=[np.number])
if len(numeric_mql.columns) > 0:
    print(numeric_mql.describe())
else:
    print("No numerical columns found")

# ============================================================================
# STEP 8: DATE COLUMNS ANALYSIS
# ============================================================================

print("\n8. DATE COLUMNS ANALYSIS")
print("-" * 30)

def analyze_date_columns(df, dataset_name):
    """Analyze date columns in a dataset"""
    print(f"\n📅 {dataset_name} - Date Columns Analysis:")
    
    # Find potential date columns
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
    
    if not date_columns:
        print("No date columns detected")
        return
    
    for col in date_columns:
        print(f"\n{col}:")
        print(f"  Data type: {df[col].dtype}")
        print(f"  Sample values: {df[col].dropna().head(3).tolist()}")
        
        # Try to analyze date range if possible
        try:
            # Convert to datetime if not already
            if df[col].dtype != 'datetime64[ns]':
                temp_dates = pd.to_datetime(df[col], errors='coerce')
            else:
                temp_dates = df[col]
            
            valid_dates = temp_dates.dropna()
            if len(valid_dates) > 0:
                print(f"  Date range: {valid_dates.min()} to {valid_dates.max()}")
                print(f"  Valid dates: {len(valid_dates)} / {len(df)}")
        except:
            print(f"  Could not parse as dates")

analyze_date_columns(deals_df, "CLOSED DEALS")
analyze_date_columns(mql_df, "MQL")

# ============================================================================
# STEP 9: RELATIONSHIP ANALYSIS
# ============================================================================

print("\n9. RELATIONSHIP ANALYSIS")
print("-" * 30)

# Check MQL ID overlap between datasets
deals_mql_ids = set(deals_df['mql_id'].dropna())
mql_ids = set(mql_df['mql_id'].dropna())

print(f"\n🔗 MQL ID Relationship Analysis:")
print(f"Unique MQL IDs in deals dataset: {len(deals_mql_ids)}")
print(f"Unique MQL IDs in MQL dataset: {len(mql_ids)}")
print(f"Common MQL IDs: {len(deals_mql_ids.intersection(mql_ids))}")
print(f"MQL IDs only in deals: {len(deals_mql_ids - mql_ids)}")
print(f"MQL IDs only in MQL dataset: {len(mql_ids - deals_mql_ids)}")

# Calculate basic conversion rate
if len(mql_ids) > 0:
    conversion_rate = len(deals_mql_ids.intersection(mql_ids)) / len(mql_ids) * 100
    print(f"Basic conversion rate: {conversion_rate:.2f}%")

# ============================================================================
# STEP 10: DATA QUALITY SUMMARY
# ============================================================================

print("\n10. DATA QUALITY SUMMARY")
print("-" * 30)

print("\n📋 OVERALL DATA QUALITY ASSESSMENT:")
print(f"✓ Datasets loaded successfully")
print(f"✓ Total records: {len(deals_df)} deals, {len(mql_df)} MQLs")
print(f"✓ Key relationship: {len(deals_mql_ids.intersection(mql_ids))} matched MQL IDs")

# Identify potential issues
issues = []
if deals_duplicates > 0:
    issues.append(f"Duplicate rows in deals dataset: {deals_duplicates}")
if mql_duplicates > 0:
    issues.append(f"Duplicate rows in MQL dataset: {mql_duplicates}")
if len(deals_missing) > 0:
    issues.append(f"Missing values in deals dataset: {len(deals_missing)} columns affected")
if len(mql_missing) > 0:
    issues.append(f"Missing values in MQL dataset: {len(mql_missing)} columns affected")

if issues:
    print(f"\n⚠️  ISSUES TO ADDRESS IN CLEANING:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print(f"✓ No major data quality issues detected")

print("\n" + "="*60)
print("DATA EXPLORATION COMPLETED")
print("Next step: Run 02_data_cleaning.py")
print("="*60)
