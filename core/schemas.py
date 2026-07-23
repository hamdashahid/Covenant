REQUIRED_FIELDS = [
    "annual_income",
    "monthly_debt",
    "credit_score",
    "employment_status",
]

EXTRACTION_SCHEMA = {
    "annual_income": "number (>0)",
    "monthly_debt": "number (>=0)",
    "credit_score": "integer (300-850)",
    "employment_status": "string (e.g. employed, self-employed, unemployed)",
    "requested_loan_amount": "number (>0, optional)",
}
