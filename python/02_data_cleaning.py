"""
Marketing Funnel Drop-Off Analysis - Data Cleaning
==================================================
Script: 02_data_cleaning.py
Purpose: Clean and preprocess Olist marketing funnel datasets
Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OLIST MARKETING FUNNEL - DATA CLEANING")
print("="*60)

# Define file paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
CLOSED_DEALS_FILE = "olist_closed_deals_dataset.csv"
MQL_FILE = "olist_marketing_qualified_leads_dataset.csv"

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# ============================================================================
# STEP 1: LOAD RAW DATASETS
# ============================================================================

print("\n1. LOADING RAW DATASETS")
print("-" * 30)

try:
    # Load datasets
    deals_df = pd.read_csv(RAW_DATA_PATH + CLOSED_DEALS_FILE)
    mql_df = pd.read_csv(RAW_DATA_PATH + MQL_FILE)
    
    print(f"✓ Deals dataset loaded: {deals_df.shape}")
    print(f"✓ MQL dataset loaded: {mql_df.shape}")
    
    # Store original shapes for comparison
    original_deals_shape = deals_df.shape
    original_mql_shape = mql_df.shape
    
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit()

# ============================================================================
# STEP 2: CLEAN CLOSED DEALS DATASET
# ============================================================================

print("\n2. CLEANING CLOSED DEALS DATASET")
print("-" * 30)

# Create a copy for cleaning
deals_clean = deals_df.copy()

# Remove exact duplicates
print(f"🧹 Removing duplicates...")
duplicates_before = deals_clean.duplicated().sum()
deals_clean = deals_clean.drop_duplicates()
duplicates_removed = duplicates_before - deals_clean.duplicated().sum()
print(f"   Removed {duplicates_removed} duplicate rows")

# Handle missing values
print(f"🧹 Handling missing values...")
missing_before = deals_clean.isnull().sum().sum()

# Fill missing categorical values with 'unknown'
categorical_cols = ['business_segment', 'lead_type', 'lead_behaviour_profile', 
                   'business_type', 'sdr_id', 'sr_id']

for col in categorical_cols:
    if col in deals_clean.columns:
        deals_clean[col] = deals_clean[col].fillna('unknown')

# Fill missing numerical values with 0 or median
numerical_cols = ['has_company', 'has_gtin', 'average_stock', 
                 'declared_product_catalog_size', 'declared_monthly_revenue']

for col in numerical_cols:
    if col in deals_clean.columns:
        if col in ['has_company', 'has_gtin']:
            # Boolean columns - fill with 0
            deals_clean[col] = deals_clean[col].fillna(0)
        else:
            # Numerical columns - fill with median
            median_val = deals_clean[col].median()
            deals_clean[col] = deals_clean[col].fillna(median_val)

missing_after = deals_clean.isnull().sum().sum()
print(f"   Missing values reduced from {missing_before} to {missing_after}")

# Convert date columns
print(f"🧹 Converting date columns...")
if 'won_date' in deals_clean.columns:
    deals_clean['won_date'] = pd.to_datetime(deals_clean['won_date'], errors='coerce')
    invalid_dates = deals_clean['won_date'].isnull().sum()
    print(f"   Converted won_date to datetime ({invalid_dates} invalid dates)")

# Standardize categorical values
print(f"🧹 Standardizing categorical values...")

# Standardize business segment
if 'business_segment' in deals_clean.columns:
    deals_clean['business_segment'] = deals_clean['business_segment'].str.lower().str.strip()

# Standardize lead type
if 'lead_type' in deals_clean.columns:
    deals_clean['lead_type'] = deals_clean['lead_type'].str.lower().str.strip()

# Standardize lead behavior profile
if 'lead_behaviour_profile' in deals_clean.columns:
    deals_clean['lead_behaviour_profile'] = deals_clean['lead_behaviour_profile'].str.lower().str.strip()

# Clean numerical columns
print(f"🧹 Cleaning numerical columns...")

# Ensure boolean columns are 0 or 1
boolean_cols = ['has_company', 'has_gtin']
for col in boolean_cols:
    if col in deals_clean.columns:
        deals_clean[col] = deals_clean[col].astype(int)

# Ensure non-negative values for stock and revenue
if 'average_stock' in deals_clean.columns:
    deals_clean['average_stock'] = deals_clean['average_stock'].clip(lower=0)

if 'declared_monthly_revenue' in deals_clean.columns:
    deals_clean['declared_monthly_revenue'] = deals_clean['declared_monthly_revenue'].clip(lower=0)

print(f"✓ Deals dataset cleaned: {deals_clean.shape}")

# ============================================================================
# STEP 3: CLEAN MQL DATASET
# ============================================================================

print("\n3. CLEANING MQL DATASET")
print("-" * 30)

# Create a copy for cleaning
mql_clean = mql_df.copy()

# Remove exact duplicates
print(f"🧹 Removing duplicates...")
duplicates_before = mql_clean.duplicated().sum()
mql_clean = mql_clean.drop_duplicates()
duplicates_removed = duplicates_before - mql_clean.duplicated().sum()
print(f"   Removed {duplicates_removed} duplicate rows")

# Handle missing values
print(f"🧹 Handling missing values...")
missing_before = mql_clean.isnull().sum().sum()

# Fill missing categorical values
if 'origin' in mql_clean.columns:
    mql_clean['origin'] = mql_clean['origin'].fillna('unknown')

if 'landing_page_id' in mql_clean.columns:
    mql_clean['landing_page_id'] = mql_clean['landing_page_id'].fillna('unknown')

missing_after = mql_clean.isnull().sum().sum()
print(f"   Missing values reduced from {missing_before} to {missing_after}")

# Convert date columns
print(f"🧹 Converting date columns...")
if 'first_contact_date' in mql_clean.columns:
    mql_clean['first_contact_date'] = pd.to_datetime(mql_clean['first_contact_date'], errors='coerce')
    invalid_dates = mql_clean['first_contact_date'].isnull().sum()
    print(f"   Converted first_contact_date to datetime ({invalid_dates} invalid dates)")

# Standardize categorical values
print(f"🧹 Standardizing categorical values...")

# Standardize origin
if 'origin' in mql_clean.columns:
    mql_clean['origin'] = mql_clean['origin'].str.lower().str.strip()

print(f"✓ MQL dataset cleaned: {mql_clean.shape}")

# ============================================================================
# STEP 4: DATA QUALITY VALIDATION
# ============================================================================

print("\n4. DATA QUALITY VALIDATION")
print("-" * 30)

def validate_dataset(df, name):
    """Validate cleaned dataset quality"""
    print(f"\n🔍 {name} Validation:")
    
    # Check for remaining duplicates
    total_duplicates = df.duplicated().sum()
    print(f"   Duplicates: {total_duplicates}")
    
    # Check for missing values
    total_missing = df.isnull().sum().sum()
    print(f"   Missing values: {total_missing}")
    
    # Check date column validity
    date_cols = df.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        valid_dates = df[col].notna().sum()
        total_dates = len(df)
        print(f"   {col}: {valid_dates}/{total_dates} valid dates")
    
    # Check for negative values in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in ['average_stock', 'declared_monthly_revenue', 'declared_product_catalog_size']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   ⚠️  {col}: {negative_count} negative values")

validate_dataset(deals_clean, "DEALS")
validate_dataset(mql_clean, "MQL")

# ============================================================================
# STEP 5: RELATIONSHIP VALIDATION
# ============================================================================

print("\n5. RELATIONSHIP VALIDATION")
print("-" * 30)

# Validate MQL ID relationships
deals_mql_ids = set(deals_clean['mql_id'].dropna())
mql_ids = set(mql_clean['mql_id'].dropna())

print(f"🔗 MQL ID Relationships:")
print(f"   Deals MQL IDs: {len(deals_mql_ids)}")
print(f"   MQL IDs: {len(mql_ids)}")
print(f"   Common IDs: {len(deals_mql_ids.intersection(mql_ids))}")

# Check for orphaned records
orphaned_deals = deals_mql_ids - mql_ids
orphaned_mqls = mql_ids - deals_mql_ids

print(f"   Orphaned deals (no MQL): {len(orphaned_deals)}")
print(f"   Orphaned MQLs (no deal): {len(orphaned_mqls)}")

# ============================================================================
# STEP 6: CREATE SUMMARY STATISTICS
# ============================================================================

print("\n6. SUMMARY STATISTICS")
print("-" * 30)

def create_summary_stats(df, name):
    """Create summary statistics for dataset"""
    print(f"\n📊 {name} Summary:")
    
    # Basic stats
    print(f"   Total records: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"   Categorical columns: {len(cat_cols)}")
    
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f"   Numerical columns: {len(num_cols)}")
    
    # Date columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    print(f"   Date columns: {len(date_cols)}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True
