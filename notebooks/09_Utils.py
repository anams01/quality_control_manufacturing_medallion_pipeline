"""
Shared utilities for the production inference and label enrichment pipeline.
"""


###############################################################################
# Imports
###############################################################################

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


###############################################################################
# Table configuration
###############################################################################

spine_table = f"{catalog}.{database}.gold_fraud_spine"
customer_profile_table = f"{catalog}.{database}.gold_customer_profile"
customer_agg_table = f"{catalog}.{database}.gold_customer_aggregations"
inference_enriched_table = f"{catalog}.{database}.gold_fraud_inference_enriched"
fraud_labels_table = f"{catalog}.{database}.silver_fraud_events"


###############################################################################
# Feature store configuration
###############################################################################

entity_key = "customer_id"
timestamp_key = "timestamp"

# Static or slowly-changing customer profile features.
# Must match exactly the feature_names used in 05_Training_Dataset_Generation
# to guarantee that the enrichment is identical to training.
profile_feature_names = [
    # Demographic
    "age",
    "age_group",
    "gender",
    "occupation",
    "city_tier",
    # Financial profile
    "income_bracket",
    "income_group",
    "customer_segment",
    "card_type",
    # Account security and behaviour
    "num_cards_issued",
    "two_fa_enabled",
    "email_verified",
    "phone_verified",
    "preferred_channel",
    "loyalty_points_balance",
    # Geography
    "country"
]

# Behavioral aggregations over rolling windows.
aggregation_feature_names = [
    # 1-hour window (very short-term velocity)
    "count_tx_1h",
    "sum_amount_1h",
    "avg_amount_1h",
    "distinct_merchants_1h",
    "count_cross_border_1h",
    # 24-hour window (intra-day behaviour)
    "count_tx_24h",
    "sum_amount_24h",
    "avg_amount_24h",
    "max_amount_24h",
    "distinct_merchants_24h",
    "distinct_countries_24h",
    "count_tor_vpn_24h",
    "count_3ds_failed_24h",
    # 7-day window (weekly pattern)
    "count_tx_7d",
    "sum_amount_7d",
    "avg_amount_7d",
    "distinct_merchants_7d",
    "distinct_countries_7d",
    "distinct_devices_7d",
    # 30-day window (monthly baseline)
    "count_tx_30d",
    "sum_amount_30d",
    "avg_amount_30d",
    "max_amount_30d",
    "min_amount_30d",
    "distinct_merchants_30d",
    "distinct_countries_30d",
    "num_fraud_confirmed_30d",
    "spend_24h_vs_avg_30d_ratio"
]

profile_lookup = FeatureLookup(
    table_name = customer_profile_table,
    feature_names = profile_feature_names,
    lookup_key = entity_key,
    timestamp_lookup_key = timestamp_key
)

aggregations_lookup = FeatureLookup(
    table_name = customer_agg_table,
    feature_names = aggregation_feature_names,
    lookup_key = entity_key,
    timestamp_lookup_key = timestamp_key
)

feature_lookups = [profile_lookup, aggregations_lookup]

exclude_columns = ["label_available_date"]

print(f"Profile features ({len(profile_feature_names)}): {profile_feature_names}")
print(f"Aggregation features ({len(aggregation_feature_names)}): {aggregation_feature_names}")
print(f"Total feature columns: {len(profile_feature_names) + len(aggregation_feature_names)}")
print()


print("09_Utils.py script loaded successfully.")