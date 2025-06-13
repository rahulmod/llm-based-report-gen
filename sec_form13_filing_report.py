import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
import os
import json


class SECFilingsFetcher:
    """Class to fetch Form 13G and 13D filings from the SEC EDGAR database."""

    def __init__(self, user_agent):
        """
        Initialize the SEC Filings Fetcher.

        Args:
            user_agent (str): Email address to identify yourself to the SEC
        """
        self.headers = {
            'User-Agent': user_agent
        }
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.search_url = "https://www.sec.gov/cgi-bin/browse-edgar"

    def search_filings(self, form_type, start_date=None, end_date=None, company=None,
                       owner=None, limit=100):
        """
        Search for SEC filings based on form type and other criteria.

        Args:
            form_type (str): Form type (SC 13G, SC 13D, etc.)
            start_date (str, optional): Start date in YYYYMMDD format
            end_date (str, optional): End date in YYYYMMDD format
            company (str, optional): Company name or ticker
            owner (str, optional): Filing owner/entity name
            limit (int, optional): Maximum number of results to return

        Returns:
            pandas.DataFrame: DataFrame containing filing information
        """
        params = {
            'action': 'getcompany',
            'count': limit,
            'output': 'atom',
        }

        if form_type:
            params['type'] = form_type.replace(' ', '')
        if company:
            params['company'] = company
        if owner:
            params['owner'] = 'only'
            params['nameinst'] = owner
        if start_date:
            params['dateb'] = start_date
        if end_date:
            params['datea'] = end_date

        print(f"Searching for {form_type} filings...")

        try:
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')

            results = []
            for entry in entries:
                title = entry.find('title').text
                # Extract accession number and CIK from the URL
                filing_link = entry.find('link')['href']
                accession_match = re.search(r'accession_number=(\d+-\d+-\d+)', filing_link)
                cik_match = re.search(r'CIK=(\d+)', filing_link)

                if accession_match and cik_match:
                    accession_number = accession_match.group(1)
                    cik = cik_match.group(1)

                    # Get filing date
                    filing_date = entry.find('updated').text.split('T')[0]

                    # Get filing details
                    file_info = {
                        'Title': title,
                        'CIK': cik,
                        'Accession Number': accession_number.replace('-', ''),
                        'Form Type': form_type,
                        'Filing Date': filing_date,
                        'URL': filing_link,
                        'Raw Accession': accession_number
                    }
                    results.append(file_info)

            if not results:
                print(f"No {form_type} filings found matching the criteria.")
                return pd.DataFrame()

            df = pd.DataFrame(results)
            print(f"Found {len(df)} {form_type} filings.")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching SEC filings: {e}")
            return pd.DataFrame()

    def download_filing(self, cik, accession_number, output_dir='filings'):
        """
        Download the complete filing document.

        Args:
            cik (str): CIK number of the company/filer
            accession_number (str): Accession number of the filing
            output_dir (str): Directory to save downloaded files

        Returns:
            str: Path to the downloaded file or None if failed
        """
        # Format CIK and accession number for URL
        cik = cik.zfill(10)
        acc_no_dash = accession_number.replace('-', '')
        acc_no = accession_number.replace('-', '')

        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct URL for the complete submission file
        url = f"{self.base_url}/{cik}/{acc_no_dash}/{acc_no}.txt"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            # Save the filing to a file
            file_path = os.path.join(output_dir, f"{cik}_{acc_no}.txt")
            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"Filing downloaded successfully: {file_path}")
            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading filing: {e}")
            return None

    def extract_filing_data(self, filing_path):
        """
        Extract key information from a downloaded filing.

        Args:
            filing_path (str): Path to the downloaded filing

        Returns:
            dict: Extracted data from the filing
        """
        try:
            with open(filing_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Find the XML section with the key data
            xml_start = content.find('<XML>')
            xml_end = content.find('</XML>', xml_start)

            if xml_start != -1 and xml_end != -1:
                xml_content = content[xml_start:xml_end + 6]
                soup = BeautifulSoup(xml_content, 'xml')

                # Extract basic filing information
                filing_info = {}

                # Look for common elements in 13D/G filings
                reporting_owners = soup.find_all('reportingOwner')
                if reporting_owners:
                    filing_info['reporting_owners'] = []
                    for owner in reporting_owners:
                        owner_info = {}
                        if owner.find('rptOwnerName'):
                            owner_info['name'] = owner.find('rptOwnerName').text
                        if owner.find('rptOwnerCik'):
                            owner_info['cik'] = owner.find('rptOwnerCik').text
                        filing_info['reporting_owners'].append(owner_info)

                # Extract issuer information
                issuer = soup.find('issuer')
                if issuer:
                    if issuer.find('issuerName'):
                        filing_info['issuer_name'] = issuer.find('issuerName').text
                    if issuer.find('issuerCik'):
                        filing_info['issuer_cik'] = issuer.find('issuerCik').text
                    if issuer.find('issuerTradingSymbol'):
                        filing_info['issuer_ticker'] = issuer.find('issuerTradingSymbol').text

                # Try to extract percentage owned
                percent_owned = re.search(r'PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW[\s\S]*?(\d+\.?\d*)\s*%',
                                          content)
                if percent_owned:
                    filing_info['percent_owned'] = float(percent_owned.group(1))

                return filing_info
            else:
                # Parse the text version if XML is not available
                filing_info = {}

                # Regular expressions to find common 13D/G data points
                issuer_match = re.search(r'NAME OF ISSUER:\s*(.*?)[\r\n]', content)
                if issuer_match:
                    filing_info['issuer_name'] = issuer_match.group(1).strip()

                security_match = re.search(r'TITLE OF CLASS OF SECURITIES:\s*(.*?)[\r\n]', content)
                if security_match:
                    filing_info['security_type'] = security_match.group(1).strip()

                cusip_match = re.search(r'CUSIP NUMBER:\s*(.*?)[\r\n]', content)
                if cusip_match:
                    filing_info['cusip'] = cusip_match.group(1).strip()

                # Try to extract percentage owned
                percent_owned = re.search(r'PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW[\s\S]*?(\d+\.?\d*)\s*%',
                                          content)
                if percent_owned:
                    filing_info['percent_owned'] = float(percent_owned.group(1))

                return filing_info

        except Exception as e:
            print(f"Error extracting data from filing: {e}")
            return {}


def main():
    # Replace with your email address for the SEC
    user_agent = "your_email@example.com"
    fetcher = SECFilingsFetcher(user_agent)

    # Example: Search for 13G filings from the last 30 days
    form_13g_results = fetcher.search_filings("SCHEDULE 13G", limit=10, company='CITADEL ADVISORS LLC')

    # Example: Search for 13D filings from the last 30 days
    form_13d_results = fetcher.search_filings("SCHEDULE 13G", limit=10, company='CITADEL ADVISORS LLC')

    # Combine results
    all_results = pd.concat([form_13g_results, form_13d_results], ignore_index=True)

    if not all_results.empty:
        # Save results to CSV
        all_results.to_csv('sec_13d_13g_filings.csv', index=False)
        print("Results saved to sec_13d_13g_filings.csv")

        # Download and analyze a few filings as examples
        for i, row in all_results.head(3).iterrows():
            print(f"\nProcessing filing {i + 1}/{min(3, len(all_results))}")
            file_path = fetcher.download_filing(row['CIK'], row['Raw Accession'])

            if file_path:
                filing_data = fetcher.extract_filing_data(file_path)
                print(f"Filing data for {row['Title']}:")
                print(json.dumps(filing_data, indent=2))

                # Add a small delay to avoid hitting SEC rate limits
                time.sleep(0.1)


if __name__ == "__main__":
    main()
