import great_expectations as ge
from typing import Tuple, List

# This module is responsible for validating the raw data before it enters the ML pipeline.
# We use Great Expectations to define a comprehensive set of expectations that the data must meet.
# this function is for QUALITY CHECKS

# this step is after loading the data (which is done in src/data/load_data.py) 
# and before preprocessing (which is done in src/data/preprocess.py)
# for example:
#
#      1.       in the "gender" column, we expect only "Male" and "Female" values, and no nulls. 
#               If we find any other values or nulls, the validation will fail.
#
#      2.       in the "tenure" column, we expect numeric values between 0 and 120 (months).   
#               If we find negative values, or values greater than 120, or nulls, the validation will fail.

# So here we set what we expect to receive in the raw data, and if the data doesn't meet these expectations, 
# we can catch it early and prevent bad data from entering our ML pipeline.

# This is done for each feature, that's why the code is a bit long, we have over 20 features to validate, 
# and each has its own set of expectations based on the original dataset and business logic.


# Working as a function that can be called from the main training script (src/train.py) after loading the data and before preprocessing it
# START here:

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Core demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Financial features (key churn predictors)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    # Yes/No fields must have valid values
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Contract types must be valid (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )
    
    # Internet service types (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    # Monthly charges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    # Total charges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
    # No missing values in critical numeric features  
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    )
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations