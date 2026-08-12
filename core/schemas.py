REQUIRED_FIELDS = [
    "annual_income",
    "monthly_debt",
    "credit_score",
    "employment_status",
    "employment_years",
    "property_value",
    "requested_loan_amount",
    "property_type",
    "down_payment",
]

EXTRACTION_SCHEMA = {
    "annual_income": "number (>0)",
    "monthly_debt": "number (>=0)",
    "credit_score": "integer (300-850)",
    "employment_status": "string (e.g. employed, self-employed, unemployed)",
    "employment_years": "number (>=0) — years at current job/business",
    "property_value": "number (>0) — appraised or purchase price of the property",
    "requested_loan_amount": "number (>0) — amount the applicant wants to borrow",
    "property_type": "string (residential|commercial) — is the property for a home or a commercial property",
    "down_payment": "number (>=0) — cash the applicant will pay upfront",
}

FIELD_LABELS = {
    "annual_income": "Annual Income",
    "monthly_debt": "Monthly Debt Payments",
    "credit_score": "Credit Score",
    "employment_status": "Employment Status",
    "employment_years": "Years at Current Job",
    "property_value": "Property Value",
    "requested_loan_amount": "Requested Loan Amount",
    "property_type": "Property Type",
    "down_payment": "Down Payment",
}