import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import openai
from typing import Dict, List, Any, Union, Optional
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

class BusinessReportGenerator:
    """
    A class to generate business reports using LLMs and data analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the report generator with API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it either as parameter or OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.model = "gpt-4"  # Default model, can be changed

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats (CSV, Excel, JSON).
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic analysis on the data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'summary': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Add basic time series analysis if date columns exist
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
        if date_columns:
            analysis['time_period'] = {
                'start': str(df[date_columns[0]].min()),
                'end': str(df[date_columns[0]].max()),
                'duration': str(df[date_columns[0]].max() - df[date_columns[0]].min())
            }
            
        # Add basic numeric analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            analysis['numeric_columns'] = numeric_cols
            analysis['correlations'] = df[numeric_cols].corr().to_dict()
            
        return analysis
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Generate basic visualizations based on the data.
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = []
        
        # Time series plot if date columns exist
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if date_columns and numeric_cols:
            plt.figure(figsize=(12, 6))
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                plt.plot(df[date_columns[0]], df[col], label=col)
            plt.title(f"Time Series: {', '.join(numeric_cols[:3])}")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            time_series_path = os.path.join(output_dir, 'time_series.png')
            plt.savefig(time_series_path)
            plt.close()
            visualization_paths.append(time_series_path)
        
        # Correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            
            heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()
            visualization_paths.append(heatmap_path)
        
        # Distribution plots for key numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            
            dist_path = os.path.join(output_dir, f'distribution_{col}.png')
            plt.savefig(dist_path)
            plt.close()
            visualization_paths.append(dist_path)
            
        # Bar chart for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if df[col].nunique() < 15:  # Only if there aren't too many categories
                plt.figure(figsize=(10, 6))
                counts = df[col].value_counts().sort_values(ascending=False)
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f"Counts by {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                bar_path = os.path.join(output_dir, f'bar_{col}.png')
                plt.savefig(bar_path)
                plt.close()
                visualization_paths.append(bar_path)
                
        return visualization_paths
    
    def generate_llm_insights(self, analysis: Dict[str, Any], data_sample: pd.DataFrame) -> Dict[str, str]:
        """
        Generate insights using LLM based on data analysis.
        
        Args:
            analysis: Analysis results dictionary
            data_sample: Sample of the data to show to the LLM
            
        Returns:
            Dictionary with different sections of insights
        """
        # Convert the analysis to a readable format for the LLM
        analysis_str = json.dumps(analysis, indent=2)
        data_sample_str = data_sample.to_string()
        
        # Define prompts for different sections
        prompts = {
            "executive_summary": """
            You are an expert business analyst. Based on the following data analysis and sample data, 
            write a concise executive summary (3-4 paragraphs) that highlights the most important findings.
            
            Data Analysis:
            {analysis}
            
            Data Sample:
            {data_sample}
            
            Executive Summary:
            """,
            
            "key_insights": """
            You are an expert business analyst. Based on the following data analysis and sample data,
            provide 5-7 key insights that would be valuable for business decision makers. 
            Be specific and focus on actionable findings.
            
            Data Analysis:
            {analysis}
            
            Data Sample:
            {data_sample}
            
            Key Insights:
            """,
            
            "recommendations": """
            You are an expert business consultant. Based on the following data analysis and sample data,
            provide 3-5 strategic recommendations for the business. Each recommendation should be 
            specific, actionable, and directly supported by the data.
            
            Data Analysis:
            {analysis}
            
            Data Sample:
            {data_sample}
            
            Strategic Recommendations:
            """
        }
        
        insights = {}
        
        # Generate content for each section using the LLM
        for section, prompt in prompts.items():
            formatted_prompt = prompt.format(
                analysis=analysis_str, 
                data_sample=data_sample_str
            )
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=1000,
                temperature=0.2  # Lower temperature for more focused responses
            )
            
            insights[section] = response.choices[0].message.content.strip()
            
        return insights
    
    def generate_full_report(self, data_path: str, output_dir: str, report_title: str = "Business Analysis Report") -> str:
        """
        Generate a complete business report from data.
        
        Args:
            data_path: Path to the data file
            output_dir: Directory to save the report and visualizations
            report_title: Title of the report
            
        Returns:
            Path to the generated report
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, 'visualizations')
        
        # Load and analyze data
        print("Loading data...")
        df = self.load_data(data_path)
        
        print("Analyzing data...")
        analysis_results = self.analyze_data(df)
        
        print("Generating visualizations...")
        vis_paths = self.generate_visualizations(df, vis_dir)
        
        print("Generating LLM insights...")
        # Use a sample of the data for the LLM to process
        sample_size = min(100, len(df))
        insights = self.generate_llm_insights(analysis_results, df.sample(sample_size))
        
        # Prepare the report content
        report_content = f"""# {report_title}
Generated on {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
{insights['executive_summary']}

## Data Overview
- Number of records: {df.shape[0]}
- Number of features: {df.shape[1]}
- Time period: {analysis_results.get('time_period', {}).get('start', 'N/A')} to {analysis_results.get('time_period', {}).get('end', 'N/A')}
- Missing values: {sum(analysis_results['missing_values'].values())} across all columns

## Key Insights
{insights['key_insights']}

## Detailed Analysis
### Data Summary
```
{df.describe().to_string()}
```

### Missing Values
```
{pd.Series(analysis_results['missing_values']).to_string()}
```

## Recommendations
{insights['recommendations']}

## Appendix: Visualizations
"""
        # Add visualization references
        for i, path in enumerate(vis_paths, 1):
            filename = os.path.basename(path)
            report_content += f"\n### Visualization {i}: {filename}\n"
            report_content += f"![{filename}](visualizations/{filename})\n"
        
        # Save the report
        report_path = os.path.join(output_dir, 'business_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        print(f"Report saved to {report_path}")
        return report_path
    
    def generate_focused_report(self, data_path: str, output_dir: str, focus_area: str) -> str:
        """
        Generate a report focused on a specific business area.
        
        Args:
            data_path: Path to the data file
            output_dir: Directory to save the report
            focus_area: Business area to focus on (e.g., 'sales', 'marketing', 'operations')
            
        Returns:
            Path to the generated report
        """
        # Load data
        df = self.load_data(data_path)
        
        # Create a focused prompt for the LLM
        focus_prompt = f"""
        You are an expert {focus_area} analyst. Based on the following data sample,
        create a detailed analysis focusing specifically on {focus_area} insights and recommendations.
        
        The report should include:
        1. Executive summary specific to {focus_area}
        2. Key {focus_area} metrics and their performance
        3. Detailed analysis of {focus_area} trends and patterns
        4. Strategic recommendations for improving {focus_area} performance
        
        Data Sample:
        {df.sample(min(100, len(df))).to_string()}
        
        Please format the report in Markdown.
        """
        
        # Generate the focused report using LLM
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are an expert {focus_area} analyst."},
                {"role": "user", "content": focus_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        report_content = response.choices[0].message.content.strip()
        
        # Save the report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f'{focus_area}_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        return report_path


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description='Generate business reports using LLMs and data analysis')
    parser.add_argument('--data', required=True, help='Path to data file (CSV, Excel, or JSON)')
    parser.add_argument('--output', default='./reports', help='Output directory for reports')
    parser.add_argument('--api-key', help='OpenAI API key (optional if set as environment variable)')
    parser.add_argument('--report-type', choices=['full', 'focused'], default='full', 
                        help='Type of report to generate')
    parser.add_argument('--focus-area', help='Business area to focus on (required if report-type is "focused")')
    parser.add_argument('--title', default='Business Analysis Report', help='Title for the report')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.report_type == 'focused' and not args.focus_area:
        parser.error('--focus-area is required when report-type is "focused"')
    
    try:
        # Initialize the report generator
        generator = BusinessReportGenerator(api_key=args.api_key)
        
        # Generate the appropriate report
        if args.report_type == 'full':
            report_path = generator.generate_full_report(args.data, args.output, args.title)
        else:
            report_path = generator.generate_focused_report(args.data, args.output, args.focus_area)
            
        print(f"Report successfully generated at: {report_path}")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main()
