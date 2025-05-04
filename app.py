# Example 1: Generate a full business report
from business_report_generator import BusinessReportGenerator

# Initialize the generator (API key from environment variable)
generator = BusinessReportGenerator()

# Generate a full report
report_path = generator.generate_full_report(
    data_path="quarterly_sales_data.csv",
    output_dir="./reports/quarterly_sales",
    report_title="Q2 2025 Sales Performance Analysis"
)

print(f"Report generated at: {report_path}")


# Example 2: Generate a focused marketing report
generator = BusinessReportGenerator()

marketing_report_path = generator.generate_focused_report(
    data_path="marketing_campaign_results.xlsx",
    output_dir="./reports/marketing",
    focus_area="marketing"
)

print(f"Marketing report generated at: {marketing_report_path}")


# Example 3: Command line usage
"""
# Generate a full sales report
python -m business_report_generator --data sales_data.csv --output ./reports/sales --title "Annual Sales Report 2025"

# Generate a focused operations report
python -m business_report_generator --data operations_metrics.xlsx --output ./reports/operations --report-type focused --focus-area operations
"""
