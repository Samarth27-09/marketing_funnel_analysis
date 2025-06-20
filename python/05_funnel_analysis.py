# PHASE 7: FUNNEL METRIC CALCULATION
# Marketing Funnel Drop-Off Analysis - Olist Dataset
# Created for: Beginner Data Analytics Portfolio Project

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def load_engineered_data():
    """
    Load the engineered dataset from Phase 6
    """
    print("Loading engineered dataset...")
    try:
        df = pd.read_csv('data/processed/olist_master_engineered.csv')
        print(f"✓ Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Engineered dataset not found. Run Phase 6 first.")
        return None

def calculate_customer_level_metrics(df):
    """
    Calculate customer-level metrics for funnel analysis
    """
    print("\n--- Calculating Customer-Level Metrics ---")
    
    # Create customer summary table
    customer_summary = df.groupby('customer_unique_id').agg({
        'order_id': 'count',  # Total orders per customer
        'is_delayed': 'max',  # 1 if customer ever experienced delay
        'is_canceled': 'max', # 1 if customer ever had cancellation
        'has_negative_review': 'max', # 1 if customer ever gave negative review
        'has_negative_experience': 'max', # 1 if customer had any negative experience
        'price': 'sum', # Total spending per customer
        'customer_state': 'first', # Customer location
        'order_purchase_timestamp': 'min' # First order date
    }).reset_index()
    
    # Rename columns for clarity
    customer_summary.columns = [
        'customer_unique_id', 'total_orders', 'experienced_delay',
        'experienced_cancellation', 'gave_negative_review', 
        'had_negative_experience', 'total_spent', 'customer_state',
        'first_order_date'
    ]
    
    # Create customer categories
    customer_summary['customer_type'] = customer_summary['total_orders'].apply(
        lambda x: 'One-time' if x == 1 else 'Repeat'
    )
    
    print(f"✓ Customer summary created: {len(customer_summary)} unique customers")
    return customer_summary

def calculate_core_funnel_metrics(customer_summary):
    """
    Calculate the core funnel metrics: Acquisition, Activation, Retention
    """
    print("\n--- Calculating Core Funnel Metrics ---")
    
    # ACQUISITION: Total unique customers who entered the funnel
    acquisition = len(customer_summary)
    
    # ACTIVATION: Customers with at least 1 order (all customers in our dataset)
    activation = len(customer_summary[customer_summary['total_orders'] >= 1])
    
    # RETENTION: Customers with 2+ orders (repeat customers)
    retention = len(customer_summary[customer_summary['total_orders'] >= 2])
    
    # Calculate conversion rates
    activation_rate = (activation / acquisition) * 100 if acquisition > 0 else 0
    retention_rate = (retention / activation) * 100 if activation > 0 else 0
    
    # Drop-off calculations
    acquisition_to_activation_dropoff = acquisition - activation
    activation_to_retention_dropoff = activation - retention
    
    activation_dropoff_rate = (acquisition_to_activation_dropoff / acquisition) * 100 if acquisition > 0 else 0
    retention_dropoff_rate = (activation_to_retention_dropoff / activation) * 100 if activation > 0 else 0
    
    # Store metrics in dictionary
    core_metrics = {
        'acquisition_customers': acquisition,
        'activation_customers': activation,
        'retention_customers': retention,
        'activation_rate_percent': round(activation_rate, 2),
        'retention_rate_percent': round(retention_rate, 2),
        'acquisition_to_activation_dropoff': acquisition_to_activation_dropoff,
        'activation_to_retention_dropoff': activation_to_retention_dropoff,
        'activation_dropoff_rate_percent': round(activation_dropoff_rate, 2),
        'retention_dropoff_rate_percent': round(retention_dropoff_rate, 2)
    }
    
    print(f"✓ Core funnel metrics calculated:")
    print(f"  - Acquisition: {acquisition:,} customers")
    print(f"  - Activation: {activation:,} customers ({activation_rate:.1f}%)")
    print(f"  - Retention: {retention:,} customers ({retention_rate:.1f}%)")
    print(f"  - Activation → Retention Drop-off: {retention_dropoff_rate:.1f}%")
    
    return core_metrics

def calculate_dropoff_reason_metrics(customer_summary):
    """
    Calculate drop-off metrics by reason (delay, cancellation, negative review)
    """
    print("\n--- Calculating Drop-off Reason Metrics ---")
    
    total_customers = len(customer_summary)
    
    # Customers who experienced each type of negative experience
    delay_customers = len(customer_summary[customer_summary['experienced_delay'] == 1])
    cancellation_customers = len(customer_summary[customer_summary['experienced_cancellation'] == 1])
    negative_review_customers = len(customer_summary[customer_summary['gave_negative_review'] == 1])
    any_negative_customers = len(customer_summary[customer_summary['had_negative_experience'] == 1])
    
    # Calculate percentages
    delay_percent = (delay_customers / total_customers) * 100 if total_customers > 0 else 0
    cancellation_percent = (cancellation_customers / total_customers) * 100 if total_customers > 0 else 0
    negative_review_percent = (negative_review_customers / total_customers) * 100 if total_customers > 0 else 0
    any_negative_percent = (any_negative_customers / total_customers) * 100 if total_customers > 0 else 0
    
    # Analyze impact on retention
    delay_retention_impact = calculate_retention_impact(customer_summary, 'experienced_delay')
    cancellation_retention_impact = calculate_retention_impact(customer_summary, 'experienced_cancellation')
    review_retention_impact = calculate_retention_impact(customer_summary, 'gave_negative_review')
    
    dropoff_metrics = {
        'customers_with_delays': delay_customers,
        'customers_with_cancellations': cancellation_customers,
        'customers_with_negative_reviews': negative_review_customers,
        'customers_with_any_negative_experience': any_negative_customers,
        'delay_rate_percent': round(delay_percent, 2),
        'cancellation_rate_percent': round(cancellation_percent, 2),
        'negative_review_rate_percent': round(negative_review_percent, 2),
        'any_negative_experience_rate_percent': round(any_negative_percent, 2),
        'delay_retention_impact': delay_retention_impact,
        'cancellation_retention_impact': cancellation_retention_impact,
        'review_retention_impact': review_retention_impact
    }
    
    print(f"✓ Drop-off reason metrics calculated:")
    print(f"  - Customers with delays: {delay_customers:,} ({delay_percent:.1f}%)")
    print(f"  - Customers with cancellations: {cancellation_customers:,} ({cancellation_percent:.1f}%)")
    print(f"  - Customers with negative reviews: {negative_review_customers:,} ({negative_review_percent:.1f}%)")
    print(f"  - Customers with any negative experience: {any_negative_customers:,} ({any_negative_percent:.1f}%)")
    
    return dropoff_metrics

def calculate_retention_impact(customer_summary, negative_experience_column):
    """
    Calculate how negative experiences impact customer retention
    """
    # Retention rate for customers without negative experience
    no_negative = customer_summary[customer_summary[negative_experience_column] == 0]
    retention_no_negative = (len(no_negative[no_negative['total_orders'] >= 2]) / len(no_negative)) * 100 if len(no_negative) > 0 else 0
    
    # Retention rate for customers with negative experience
    with_negative = customer_summary[customer_summary[negative_experience_column] == 1]
    retention_with_negative = (len(with_negative[with_negative['total_orders'] >= 2]) / len(with_negative)) * 100 if len(with_negative) > 0 else 0
    
    # Impact (difference in retention rates)
    impact = retention_no_negative - retention_with_negative
    
    return {
        'retention_without_negative_percent': round(retention_no_negative, 2),
        'retention_with_negative_percent': round(retention_with_negative, 2),
        'retention_impact_percent': round(impact, 2)
    }

def calculate_segment_metrics(customer_summary):
    """
    Calculate funnel metrics by customer segments (state, order value, etc.)
    """
    print("\n--- Calculating Segment Metrics ---")
    
    # Top 10 states by customer count
    state_metrics = customer_summary.groupby('customer_state').agg({
        'customer_unique_id': 'count',
        'total_orders': 'mean',
        'experienced_delay': 'mean',
        'experienced_cancellation': 'mean',
        'gave_negative_review': 'mean',
        'total_spent': 'mean'
    }).round(2)
    
    state_metrics.columns = [
        'customer_count', 'avg_orders_per_customer', 'delay_rate',
        'cancellation_rate', 'negative_review_rate', 'avg_spending'
    ]
    
    # Get top 10 states
    top_states = state_metrics.nlargest(10, 'customer_count')
    
    # Customer type breakdown
    customer_type_metrics = customer_summary.groupby('customer_type').agg({
        'customer_unique_id': 'count',
        'total_orders': 'mean',
        'total_spent': 'mean',
        'experienced_delay': 'mean',
        'experienced_cancellation': 'mean',
        'gave_negative_review': 'mean'
    }).round(2)
    
    print(f"✓ Segment metrics calculated")
    print(f"  - Analyzed {len(state_metrics)} states")
    print(f"  - Top state: {top_states.index[0]} ({top_states.iloc[0]['customer_count']:,} customers)")
    
    return {
        'state_metrics': state_metrics,
        'top_states': top_states,
        'customer_type_metrics': customer_type_metrics
    }

def calculate_time_based_metrics(df):
    """
    Calculate funnel metrics over time
    """
    print("\n--- Calculating Time-based Metrics ---")
    
    # Convert timestamp to datetime if not already
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    
    # Monthly metrics
    monthly_metrics = df.groupby('purchase_month').agg({
        'customer_unique_id': 'nunique',  # Unique customers per month
        'order_id': 'count',  # Total orders per month
        'is_delayed': 'mean',  # Delay rate per month
        'is_canceled': 'mean',  # Cancellation rate per month
        'has_negative_review': 'mean',  # Negative review rate per month
        'price': 'mean'  # Average order value per month
    }).round(3)
    
    monthly_metrics.columns = [
        'unique_customers', 'total_orders', 'delay_rate',
        'cancellation_rate', 'negative_review_rate', 'avg_order_value'
    ]
    
    # Convert percentages
    monthly_metrics['delay_rate'] *= 100
    monthly_metrics['cancellation_rate'] *= 100
    monthly_metrics['negative_review_rate'] *= 100
    
    print(f"✓ Time-based metrics calculated for {len(monthly_metrics)} months")
    
    return monthly_metrics

def save_funnel_metrics(core_metrics, dropoff_metrics, segment_metrics, monthly_metrics, customer_summary):
    """
    Save all calculated metrics to files
    """
    print("\n--- Saving Funnel Metrics ---")
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    # Save core metrics as JSON
    all_metrics = {
        'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'core_funnel_metrics': core_metrics,
        'dropoff_reason_metrics': dropoff_metrics
    }
    
    with open('outputs/metrics/funnel_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("✓ Core metrics saved to: outputs/metrics/funnel_metrics.json")
    
    # Save segment metrics
    segment_metrics['state_metrics'].to_csv('outputs/metrics/state_metrics.csv')
    segment_metrics['top_states'].to_csv('outputs/metrics/top_states_metrics.csv')
    segment_metrics['customer_type_metrics'].to_csv('outputs/metrics/customer_type_metrics.csv')
    print("✓ Segment metrics saved to: outputs/metrics/")
    
    # Save monthly metrics
    monthly_metrics.to_csv('outputs/metrics/monthly_metrics.csv')
    print("✓ Monthly metrics saved to: outputs/metrics/monthly_metrics.csv")
    
    # Save customer summary for further analysis
    customer_summary.to_csv('outputs/metrics/customer_summary.csv', index=False)
    print("✓ Customer summary saved to: outputs/metrics/customer_summary.csv")

def generate_metrics_report(core_metrics, dropoff_metrics, segment_metrics):
    """
    Generate a comprehensive metrics report
    """
    report = f"""# MARKETING FUNNEL ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY

### Core Funnel Performance
- **Total Customers (Acquisition)**: {core_metrics['acquisition_customers']:,}
- **Active Customers (Activation)**: {core_metrics['activation_customers']:,} ({core_metrics['activation_rate_percent']}%)
- **Repeat Customers (Retention)**: {core_metrics['retention_customers']:,} ({core_metrics['retention_rate_percent']}%)

### Key Drop-off Points
- **Activation → Retention Drop-off**: {core_metrics['retention_dropoff_rate_percent']}%
- **Total customers lost at retention stage**: {core_metrics['activation_to_retention_dropoff']:,}

## DROP-OFF ANALYSIS BY REASON

### Negative Experience Rates
- **Delivery Delays**: {dropoff_metrics['delay_rate_percent']}% of customers
- **Order Cancellations**: {dropoff_metrics['cancellation_rate_percent']}% of customers  
- **Negative Reviews**: {dropoff_metrics['negative_review_rate_percent']}% of customers
- **Any Negative Experience**: {dropoff_metrics['any_negative_experience_rate_percent']}% of customers

### Impact on Customer Retention
- **Delivery Delays Impact**: {dropoff_metrics['delay_retention_impact']['retention_impact_percent']:.1f}% reduction in retention
- **Cancellation Impact**: {dropoff_metrics['cancellation_retention_impact']['retention_impact_percent']:.1f}% reduction in retention
- **Negative Review Impact**: {dropoff_metrics['review_retention_impact']['retention_impact_percent']:.1f}% reduction in retention

## TOP PERFORMING STATES
{segment_metrics['top_states'].head().to_string()}

## CUSTOMER SEGMENTS
{segment_metrics['customer_type_metrics'].to_string()}

## KEY INSIGHTS
1. **Primary Drop-off Point**: {core_metrics['retention_dropoff_rate_percent']}% of activated customers do not return for a second purchase
2. **Biggest Retention Killer**: {'Delivery delays' if dropoff_metrics['delay_retention_impact']['retention_impact_percent'] == max(dropoff_metrics['delay_retention_impact']['retention_impact_percent'], dropoff_metrics['cancellation_retention_impact']['retention_impact_percent'], dropoff_metrics['review_retention_impact']['retention_impact_percent']) else 'Cancellations' if dropoff_metrics['cancellation_retention_impact']['retention_impact_percent'] == max(dropoff_metrics['delay_retention_impact']['retention_impact_percent'], dropoff_metrics['cancellation_retention_impact']['retention_impact_percent'], dropoff_metrics['review_retention_impact']['retention_impact_percent']) else 'Negative reviews'} have the highest impact on retention
3. **Geographic Concentration**: Top state ({segment_metrics['top_states'].index[0]}) represents {(segment_metrics['top_states'].iloc[0]['customer_count'] / core_metrics['acquisition_customers'] * 100):.1f}% of all customers

## RECOMMENDATIONS
1. Focus on reducing delivery delays to improve retention
2. Investigate cancellation reasons in top states
3. Implement proactive customer service for at-risk segments
4. Develop retention campaigns for one-time customers
"""
    
    # Save report
    with open('outputs/funnel_analysis_report.md', 'w') as f:
        f.write(report)
    print("✓ Comprehensive report saved to: outputs/funnel_analysis_report.md")

def main():
    """
    Main function to run the complete funnel metrics calculation
    """
    print("PHASE 7: FUNNEL METRIC CALCULATION")
    print("Marketing Funnel Drop-Off Analysis - Olist Dataset")
    print("="*60)
    
    # Load data
    df = load_engineered_data()
    if df is None:
        return
    
    # Calculate customer-level metrics
    customer_summary = calculate_customer_level_metrics(df)
    
    # Calculate core funnel metrics
    core_metrics = calculate_core_funnel_metrics(customer_summary)
    
    # Calculate drop-off reason metrics
    dropoff_metrics = calculate_dropoff_reason_metrics(customer_summary)
    
    # Calculate segment metrics
    segment_metrics = calculate_segment_metrics(customer_summary)
    
    # Calculate time-based metrics
    monthly_metrics = calculate_time_based_metrics(df)
    
    # Save all metrics
    save_funnel_metrics(core_metrics, dropoff_metrics, segment_metrics, monthly_metrics, customer_summary)
    
    # Generate comprehensive report
    generate_metrics_report(core_metrics, dropoff_metrics, segment_metrics)
    
    print(f"\n✅ PHASE 7 COMPLETED SUCCESSFULLY!")
    print(f"📊 All metrics calculated and saved to outputs/ folder")
    print(f"📋 Comprehensive report available at: outputs/funnel_analysis_report.md")
    print(f"🎯 Ready for Phase 9: Exploratory Data Analysis!")

if __name__ == "__main__":
    main()
