# PHASE 6: FEATURE ENGINEERING
# Marketing Funnel Drop-Off Analysis - Olist Dataset
# Created for: Beginner Data Analytics Portfolio Project

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_master_data():
    """
    Load the master dataset created in Phase 5
    """
    print("Loading master dataset...")
    try:
        # Load the merged master dataset from Phase 5
        master_df = pd.read_csv('data/processed/olist_master.csv')
        print(f"Master dataset loaded successfully with {len(master_df)} rows and {len(master_df.columns)} columns")
        return master_df
    except FileNotFoundError:
        print("Error: Master dataset not found. Please run Phase 5 first.")
        return None

def create_delivery_delay_feature(df):
    """
    Create delivery delay feature
    - Calculate days between estimated and actual delivery
    - Flag orders with delivery delays
    """
    print("\n--- Creating Delivery Delay Features ---")
    
    # Convert date columns to datetime if not already
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 
                   'order_estimated_delivery_date']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate delivery delay in days
    df['delivery_delay_days'] = (df['order_delivered_customer_date'] - 
                                df['order_estimated_delivery_date']).dt.days
    
    # Fill NaN values with 0 (orders not yet delivered or no delay)
    df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0)
    
    # Create delay flag: 1 if delayed (positive days), 0 if on time or early
    df['is_delayed'] = (df['delivery_delay_days'] > 0).astype(int)
    
    # Create delay severity categories
    def categorize_delay(days):
        if days <= 0:
            return 'On Time/Early'
        elif days <= 7:
            return 'Minor Delay (1-7 days)'
        elif days <= 30:
            return 'Major Delay (8-30 days)'
        else:
            return 'Severe Delay (30+ days)'
    
    df['delay_category'] = df['delivery_delay_days'].apply(categorize_delay)
    
    print(f"✓ Delivery delay features created")
    print(f"  - Average delay: {df['delivery_delay_days'].mean():.2f} days")
    print(f"  - Delayed orders: {df['is_delayed'].sum():,} ({df['is_delayed'].mean()*100:.1f}%)")
    
    return df

def create_review_score_features(df):
    """
    Create review score features and negative review flags
    """
    print("\n--- Creating Review Score Features ---")
    
    # Create review score labels
    def categorize_review_score(score):
        if pd.isna(score):
            return 'No Review'
        elif score >= 4:
            return 'Positive (4-5)'
        elif score == 3:
            return 'Neutral (3)'
        else:
            return 'Negative (1-2)'
    
    df['review_score_label'] = df['review_score'].apply(categorize_review_score)
    
    # Create negative review flag (1-2 stars)
    df['has_negative_review'] = ((df['review_score'] >= 1) & (df['review_score'] <= 2)).astype(int)
    
    # Create no review flag (missing reviews can indicate poor experience)
    df['has_no_review'] = df['review_score'].isna().astype(int)
    
    print(f"✓ Review score features created")
    print(f"  - Negative reviews: {df['has_negative_review'].sum():,} ({df['has_negative_review'].mean()*100:.1f}%)")
    print(f"  - No reviews: {df['has_no_review'].sum():,} ({df['has_no_review'].mean()*100:.1f}%)")
    
    return df

def create_repeat_customer_feature(df):
    """
    Create repeat customer feature
    Identifies customers who have made multiple orders
    """
    print("\n--- Creating Repeat Customer Features ---")
    
    # Count orders per customer
    customer_order_counts = df.groupby('customer_unique_id').size().reset_index(name='total_orders')
    
    # Merge back to main dataframe
    df = df.merge(customer_order_counts, on='customer_unique_id', how='left')
    
    # Create repeat customer flag (2+ orders)
    df['is_repeat_customer'] = (df['total_orders'] >= 2).astype(int)
    
    # Create customer category
    def categorize_customer(orders):
        if orders == 1:
            return 'One-time Customer'
        elif orders <= 3:
            return 'Occasional Customer (2-3 orders)'
        else:
            return 'Frequent Customer (4+ orders)'
    
    df['customer_category'] = df['total_orders'].apply(categorize_customer)
    
    print(f"✓ Repeat customer features created")
    print(f"  - Repeat customers: {df['is_repeat_customer'].sum():,} ({df['is_repeat_customer'].mean()*100:.1f}%)")
    print(f"  - Average orders per customer: {df['total_orders'].mean():.2f}")
    
    return df

def create_cancellation_features(df):
    """
    Create order cancellation features
    """
    print("\n--- Creating Cancellation Features ---")
    
    # Create cancellation flag based on order status
    df['is_canceled'] = (df['order_status'] == 'canceled').astype(int)
    
    # Create order status categories for analysis
    def categorize_order_status(status):
        if status in ['delivered']:
            return 'Completed'
        elif status in ['canceled']:
            return 'Canceled'
        elif status in ['shipped', 'processing', 'approved']:
            return 'In Progress'
        else:
            return 'Other'
    
    df['order_status_category'] = df['order_status'].apply(categorize_order_status)
    
    print(f"✓ Cancellation features created")
    print(f"  - Canceled orders: {df['is_canceled'].sum():,} ({df['is_canceled'].mean()*100:.1f}%)")
    
    return df

def create_purchase_timing_features(df):
    """
    Create features related to purchase timing
    """
    print("\n--- Creating Purchase Timing Features ---")
    
    # Extract time components from purchase timestamp
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    
    # Create day of week labels
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['purchase_day_name'] = df['purchase_day_of_week'].map(day_names)
    
    # Create time of day categories
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return 'Morning (6-11)'
        elif 12 <= hour < 18:
            return 'Afternoon (12-17)'
        elif 18 <= hour < 22:
            return 'Evening (18-21)'
        else:
            return 'Night (22-5)'
    
    df['purchase_time_category'] = df['purchase_hour'].apply(categorize_hour)
    
    print(f"✓ Purchase timing features created")
    
    return df

def create_drop_off_flags(df):
    """
    Create comprehensive drop-off flags for funnel analysis
    These flags identify potential reasons why customers might drop off
    """
    print("\n--- Creating Drop-off Flags ---")
    
    # Primary drop-off indicators
    df['dropped_due_to_delay'] = df['is_delayed']
    df['dropped_due_to_cancellation'] = df['is_canceled']  
    df['dropped_due_to_negative_review'] = df['has_negative_review']
    
    # Combined drop-off flag (any negative experience)
    df['has_negative_experience'] = (
        (df['is_delayed'] == 1) | 
        (df['is_canceled'] == 1) | 
        (df['has_negative_review'] == 1)
    ).astype(int)
    
    # Severe drop-off flag (multiple issues)
    df['has_severe_issues'] = (
        df['is_delayed'] + df['is_canceled'] + df['has_negative_review'] >= 2
    ).astype(int)
    
    print(f"✓ Drop-off flags created")
    print(f"  - Customers with negative experience: {df['has_negative_experience'].sum():,} ({df['has_negative_experience'].mean()*100:.1f}%)")
    print(f"  - Customers with severe issues: {df['has_severe_issues'].sum():,} ({df['has_severe_issues'].mean()*100:.1f}%)")
    
    return df

def create_order_value_features(df):
    """
    Create features related to order value and payment
    """
    print("\n--- Creating Order Value Features ---")
    
    # Create order value categories
    def categorize_order_value(value):
        if pd.isna(value):
            return 'Unknown'
        elif value < 50:
            return 'Low Value (<$50)'
        elif value < 200:
            return 'Medium Value ($50-200)'
        else:
            return 'High Value ($200+)'
    
    df['order_value_category'] = df['price'].apply(categorize_order_value)
    
    # Create high-value order flag
    df['is_high_value_order'] = (df['price'] >= 200).astype(int)
    
    print(f"✓ Order value features created")
    
    return df

def generate_feature_summary(df):
    """
    Generate a summary of all created features
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    # List all new features created
    new_features = [
        'delivery_delay_days', 'is_delayed', 'delay_category',
        'review_score_label', 'has_negative_review', 'has_no_review',
        'total_orders', 'is_repeat_customer', 'customer_category',
        'is_canceled', 'order_status_category',
        'purchase_year', 'purchase_month', 'purchase_day_of_week', 
        'purchase_day_name', 'purchase_hour', 'purchase_time_category',
        'dropped_due_to_delay', 'dropped_due_to_cancellation', 
        'dropped_due_to_negative_review', 'has_negative_experience', 
        'has_severe_issues', 'order_value_category', 'is_high_value_order'
    ]
    
    print(f"Total new features created: {len(new_features)}")
    print(f"Final dataset shape: {df.shape}")
    
    # Key metrics summary
    print(f"\nKey Funnel Metrics:")
    print(f"- Total unique customers: {df['customer_unique_id'].nunique():,}")
    print(f"- Total orders: {len(df):,}")
    print(f"- Repeat customers: {df['is_repeat_customer'].sum():,} ({df['is_repeat_customer'].mean()*100:.1f}%)")
    print(f"- Delayed orders: {df['is_delayed'].sum():,} ({df['is_delayed'].mean()*100:.1f}%)")
    print(f"- Canceled orders: {df['is_canceled'].sum():,} ({df['is_canceled'].mean()*100:.1f}%)")
    print(f"- Negative reviews: {df['has_negative_review'].sum():,} ({df['has_negative_review'].mean()*100:.1f}%)")
    print(f"- Orders with negative experience: {df['has_negative_experience'].sum():,} ({df['has_negative_experience'].mean()*100:.1f}%)")
    
    return new_features

def save_engineered_dataset(df):
    """
    Save the dataset with engineered features
    """
    print(f"\n--- Saving Engineered Dataset ---")
    
    # Save to processed data folder
    output_path = 'data/processed/olist_master_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Engineered dataset saved to: {output_path}")
    
    # Also create a features documentation file
    feature_docs = """# Feature Engineering Documentation - Olist Marketing Funnel Analysis

## New Features Created in Phase 6

### Delivery Delay Features
- `delivery_delay_days`: Number of days between estimated and actual delivery (negative = early)
- `is_delayed`: Binary flag (1 = delayed, 0 = on time/early)
- `delay_category`: Categorical delay severity (On Time/Early, Minor, Major, Severe)

### Review Score Features  
- `review_score_label`: Categorical review rating (Positive 4-5, Neutral 3, Negative 1-2, No Review)
- `has_negative_review`: Binary flag for poor reviews (1-2 stars)
- `has_no_review`: Binary flag for missing reviews

### Customer Behavior Features
- `total_orders`: Total number of orders per customer
- `is_repeat_customer`: Binary flag for customers with 2+ orders
- `customer_category`: Customer segmentation (One-time, Occasional, Frequent)

### Order Status Features
- `is_canceled`: Binary flag for canceled orders
- `order_status_category`: Simplified order status (Completed, Canceled, In Progress, Other)

### Purchase Timing Features
- `purchase_year/month`: Time components from purchase timestamp
- `purchase_day_of_week`: Day of week (0=Monday, 6=Sunday)
- `purchase_day_name`: Day name (Monday, Tuesday, etc.)
- `purchase_hour`: Hour of purchase (0-23)
- `purchase_time_category`: Time of day category (Morning, Afternoon, Evening, Night)

### Drop-off Analysis Flags
- `dropped_due_to_delay`: Same as is_delayed
- `dropped_due_to_cancellation`: Same as is_canceled
- `dropped_due_to_negative_review`: Same as has_negative_review
- `has_negative_experience`: Combined flag for any negative experience
- `has_severe_issues`: Flag for customers with multiple issues

### Order Value Features
- `order_value_category`: Order value segments (Low <$50, Medium $50-200, High $200+)
- `is_high_value_order`: Binary flag for orders $200+

## Usage Notes
- These features are designed for funnel drop-off analysis
- All binary flags use 1/0 encoding for easy aggregation
- Missing values are handled appropriately for each feature type
- Features support both customer-level and order-level analysis
"""
    
    with open('data/processed/feature_engineering_documentation.md', 'w') as f:
        f.write(feature_docs)
    print(f"✓ Feature documentation saved to: data/processed/feature_engineering_documentation.md")

def main():
    """
    Main function to run the complete feature engineering process
    """
    print("PHASE 6: FEATURE ENGINEERING")
    print("Marketing Funnel Drop-Off Analysis - Olist Dataset")
    print("="*60)
    
    # Load master dataset
    df = load_master_data()
    if df is None:
        return
    
    print(f"Starting with {len(df)} rows and {len(df.columns)} columns")
    
    # Apply all feature engineering steps
    df = create_delivery_delay_feature(df)
    df = create_review_score_features(df)
    df = create_repeat_customer_feature(df)
    df = create_cancellation_features(df)
    df = create_purchase_timing_features(df)
    df = create_drop_off_flags(df)
    df = create_order_value_features(df)
    
    # Generate summary and save
    new_features = generate_feature_summary(df)
    save_engineered_dataset(df)
    
    print(f"\n✅ PHASE 6 COMPLETED SUCCESSFULLY!")
    print(f"Ready for Phase 7: Funnel Metric Calculation")

if __name__ == "__main__":
    main()
