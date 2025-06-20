"""
Marketing Funnel Drop-Off Analysis - Data Merging
=================================================
Script: 03_data_merging.py
Purpose: Merge cleaned datasets to create master dataset for analysis
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
print("OLIST MARKETING FUNNEL - DATA MERGING")
print("="*60)

# Define file paths
PROCESSED_DATA_PATH = "data/processed/"
CLEANED_DEALS_FILE = "cleaned_deals.csv"
CLEANED_MQL_FILE = "cleaned_leads.csv"
MASTER_FILE = "olist_master.csv"

# ============================================================================
# STEP 1: LOAD CLEANED DATASETS
# ============================================================================

print("\n1. LOADING CLEANED DATASETS")
print("-" * 30)

try:
    # Load cleaned datasets
    deals_df = pd.read_csv(PROCESSED_DATA_PATH + CLEANED_DEALS_FILE)
    mql_df = pd.read_csv(PROCESSED_DATA_PATH + CLEANED_MQL_FILE)
    
    print(f"✓ Cleaned deals loaded: {deals_df.shape}")
    print(f"✓ Cleaned MQL loaded: {mql_df.shape}")
    
    # Convert date columns back to datetime (they become strings after CSV save)
    if 'won_date' in deals_df.columns:
        deals_df['won_date'] = pd.to_datetime(deals_df['won_date'])
    
    if 'first_contact_date' in mql_df.columns:
        mql_df['first_contact_date'] = pd.to_datetime(mql_df['first_contact_date'])
    
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit()

# ============================================================================
# STEP 2: ANALYZE MERGE SCENARIOS
# ============================================================================

print("\n2. ANALYZING MERGE SCENARIOS")
print("-" * 30)

# Analyze MQL ID overlap
deals_mql_ids = set(deals_df['mql_id'].dropna())
mql_ids = set(mql_df['mql_id'].dropna())

print(f"🔍 Merge Analysis:")
print(f"   Deals with MQL ID: {len(deals_mql_ids)}")
print(f"   Total MQLs: {len(mql_ids)}")
print(f"   Matching MQL IDs: {len(deals_mql_ids.intersection(mql_ids))}")
print(f"   MQLs without deals: {len(mql_ids - deals_mql_ids)}")
print(f"   Deals without MQL: {len(deals_mql_ids - mql_ids)}")

# ============================================================================
# STEP 3: CREATE MASTER DATASET
# ============================================================================

print("\n3. CREATING MASTER DATASET")
print("-" * 30)

print("🔄 Merging datasets...")

# Strategy: Left join MQL with deals to include all leads
# This preserves all MQLs and shows which ones converted to deals
master_df = mql_df.merge(deals_df, on='mql_id', how='left')

print(f"✓ Initial merge completed: {master_df.shape}")

# ============================================================================
# STEP 4: CREATE FUNNEL STAGE INDICATORS
# ============================================================================

print("\n4. CREATING FUNNEL STAGE INDICATORS")
print("-" * 30)

# Create funnel stage indicators
print("🏗️ Creating funnel stage indicators...")

# Stage 1: Lead Generated (all records have this)
master_df['stage_1_lead_generated'] = 1

# Stage 2: First Contact Made (has first_contact_date)
master_df['stage_2_first_contact'] = (~master_df['first_contact_date'].isna()).astype(int)

# Stage 3: Sales Qualified (has SDR assigned)
master_df['stage_3_sales_qualified'] = (~master_df['sdr_id'].isna()).astype(int)

# Stage 4: Sales Consultation (has SR assigned)
master_df['stage_4_consultation'] = (~master_df['sr_id'].isna()).astype(int)

# Stage 5: Deal Closed (has won_date)
master_df['stage_5_deal_closed'] = (~master_df['won_date'].isna()).astype(int)

print("✓ Funnel stage indicators created")

# ============================================================================
# STEP 5: CREATE CONVERSION FLAGS
# ============================================================================

print("\n5. CREATING CONVERSION FLAGS")
print("-" * 30)

print("🏗️ Creating conversion flags...")

# Overall conversion flag
master_df['is_converted'] = master_df['stage_5_deal_closed']

# Drop-off flags (inverse of reaching next stage)
master_df['dropout_after_lead'] = 1 - master_df['stage_2_first_contact']
master_df['dropout_after_contact'] = ((master_df['stage_2_first_contact'] == 1) & 
                                     (master_df['stage_3_sales_qualified'] == 0)).astype(int)
master_df['dropout_after_qualification'] = ((master_df['stage_3_sales_qualified'] == 1) & 
                                           (master_df['stage_4_consultation'] == 0)).astype(int)
master_df['dropout_after_consultation'] = ((master_df['stage_4_consultation'] == 1) & 
                                          (master_df['stage_5_deal_closed'] == 0)).astype(int)

print("✓ Conversion flags created")

# ============================================================================
# STEP 6: CREATE TIME-BASED FEATURES
# ============================================================================

print("\n6. CREATING TIME-BASED FEATURES")
print("-" * 30)

print("🏗️ Creating time-based features...")

# Time to first contact (days from lead generation to first contact)
# Note: We don't have lead generation date, so we'll use first_contact_date as baseline
master_df['days_to_first_contact'] = 0  # Placeholder since we don't have lead gen date

# Time to close (days from first contact to deal close)
master_df['days_to_close'] = (master_df['won_date'] - master_df['first_contact_date']).dt.days

# Extract date components for seasonal analysis
master_df['first_contact_year'] = master_df['first_contact_date'].dt.year
master_df['first_contact_month'] = master_df['first_contact_date'].dt.month
master_df['first_contact_quarter'] = master_df['first_contact_date'].dt.quarter
master_df['first_contact_weekday'] = master_df['first_contact_date'].dt.dayofweek

# Close date components
master_df['won_year'] = master_df['won_date'].dt.year
master_df['won_month'] = master_df['won_date'].dt.month
master_df['won_quarter'] = master_df['won_date'].dt.quarter

print("✓ Time-based features created")

# ============================================================================
# STEP 7: CREATE BUSINESS INTELLIGENCE FEATURES
# ============================================================================

print("\n7. CREATING BUSINESS INTELLIGENCE FEATURES")
print("-" * 30)

print("🏗️ Creating business intelligence features...")

# Lead source categorization
master_df['lead_source_category'] = master_df['origin'].apply(
    lambda x: 'organic' if 'organic' in str(x).lower() else
              'paid' if 'paid' in str(x).lower() else
              'direct' if 'direct' in str(x).lower() else
              'other'
)

# Business size categorization based on declared revenue
def categorize_business_size(revenue):
    if pd.isna(revenue):
        return 'unknown'
    elif revenue == 0:
        return 'startup'
    elif revenue <= 1000:
        return 'micro'
    elif revenue <= 10000:
        return 'small'
    elif revenue <= 100000:
        return 'medium'
    else:
        return 'large'

master_df['business_size_category'] = master_df['declared_monthly_revenue'].apply(categorize_business_size)

# Catalog size categorization
def categorize_catalog_size(catalog_size):
    if pd.isna(catalog_size):
        return 'unknown'
    elif catalog_size <= 10:
        return 'small'
    elif catalog_size <= 100:
        return 'medium'
    elif catalog_size <= 1000:
        return 'large'
    else:
        return 'enterprise'

master_df['catalog_size_category'] = master_df['declared_product_catalog_size'].apply(categorize_catalog_size)

print("✓ Business intelligence features created")

# ============================================================================
# STEP 8: DATA QUALITY VALIDATION
# ============================================================================

print("\n8. DATA QUALITY VALIDATION")
print("-" * 30)

print("🔍 Validating master dataset...")

# Check for logical consistency
print(f"✓ Total records: {len(master_df)}")
print(f"✓ Total columns: {len(master_df.columns)}")

# Validate funnel logic
funnel_validation = {
    'Stage 1 (Lead Generated)': master_df['stage_1_lead_generated'].sum(),
    'Stage 2 (First Contact)': master_df['stage_2_first_contact'].sum(),
    'Stage 3 (Sales Qualified)': master_df['stage_3_sales_qualified'].sum(),
    'Stage 4 (Consultation)': master_df['stage_4_consultation'].sum(),
    'Stage 5 (Deal Closed)': master_df['stage_5_deal_closed'].sum()
}

print(f"\n📊 Funnel Validation:")
for stage, count in funnel_validation.items():
    percentage = (count / len(master_df)) * 100
    print(f"   {stage}: {count} ({percentage:.1f}%)")

# Check for logical errors (later stages should not exceed earlier stages)
logic_errors = []
if master_df['stage_2_first_contact'].sum() > master_df['stage_1_lead_generated'].sum():
    logic_errors.append("Stage 2 > Stage 1")
if master_df['stage_3_sales_qualified'].sum() > master_df['stage_2_first_contact'].sum():
    logic_errors.append("Stage 3 > Stage 2")
if master_df['stage_4_consultation'].sum() > master_df['stage_3_sales_qualified'].sum():
    logic_errors.append("Stage 4 > Stage 3")
if master_df['stage_5_deal_closed'].sum() > master_df['stage_4_consultation'].sum():
    logic_errors.append("Stage 5 > Stage 4")

if logic_errors:
    print(f"⚠️  Logic errors found: {logic_errors}")
else:
    print(f"✓ Funnel logic validated")

# ============================================================================
# STEP 9: CREATE SUMMARY STATISTICS
# ============================================================================

print("\n9. CREATING SUMMARY STATISTICS")
print("-" * 30)

print("📊 Master Dataset Summary:")
print(f"   Total MQLs: {len(master_df)}")
print(f"   Converted Deals: {master_df['is_converted'].sum()}")
print(f"   Overall Conversion Rate: {(master_df['is_converted'].sum() / len(master_df) * 100):.2f}%")
print(f"   Average Days to Close: {master_df['days_to_close'].mean():.1f} days")

# Business segment performance
if 'business_segment' in master_df.columns:
    segment_performance = master_df.groupby('business_segment').agg({
        'is_converted': ['count', 'sum', 'mean']
    }).round(3)
    print(f"\n📊 Performance by Business Segment:")
    print(segment_performance)

# Lead source performance
source_performance = master_df.groupby('lead_source_category').agg({
    'is_converted': ['count', 'sum', 'mean']
}).round(3)
print(f"\n📊 Performance by Lead Source:")
print(source_performance)

# ============================================================================
# STEP 10: SAVE MASTER DATASET
# ============================================================================

print("\n10. SAVING MASTER DATASET")
print("-" * 30)

try:
    # Save master dataset
    master_output_file = PROCESSED_DATA_PATH + MASTER_FILE
    master_df.to_csv(master_output_file, index=False)
    print(f"✓ Master dataset saved to: {master_output_file}")
    
    # Save column list for reference
    columns_file = PROCESSED_DATA_PATH + "master_columns.txt"
    with open(columns_file, 'w') as f:
        f.write("OLIST MASTER DATASET COLUMNS\n")
        f.write("="*40 + "\n\n")
        f.write("ORIGINAL MQL COLUMNS:\n")
        for col in mql_df.columns:
            f.write(f"  - {col}\n")
        f.write("\nORIGINAL DEALS COLUMNS:\n")
        for col in deals_df.columns:
            f.write(f"  - {col}\n")
        f.write("\nENGINEERED FEATURES:\n")
        engineered_features = [col for col in master_df.columns 
                             if col not in list(mql_df.columns) + list(deals_df.columns)]
        for col in engineered_features:
            f.write(f"  - {col}\n")
        f.write(f"\nTOTAL COLUMNS: {len(master_df.columns)}\n")
    
    print(f"✓ Column reference saved to: {columns_file}")
    
except Exception as e:
    print(f"❌ Error saving files: {e}")

print("\n" + "="*60)
print("DATA MERGING COMPLETED")
print(f"Master dataset created with {len(master_df)} records and {len(master_df.columns)} columns")
print("Next step: Run 04_feature_engineering.py")
print("="*60)
