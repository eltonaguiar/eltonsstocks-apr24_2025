"""
Google Sheets API handler for the stock update application.
"""
import time
import logging
from functools import wraps
import gspread
from google.oauth2.service_account import Credentials
from gspread_formatting import set_column_width as gspread_set_column_width
import base64
from datetime import datetime

logger = logging.getLogger('sheets_handler')

MAIN_SHEET_NAME = "Sheet1"
BACKUP_SHEET_PREFIX = "Backup_"
SUMMARY_SHEET_NAME = "Summary"

def set_column_width(worksheet, column, width):
    try:
        if isinstance(column, int):
            column = chr(64 + column)  # Convert column number to letter (1 -> A, 2 -> B, etc.)
        gspread_set_column_width(worksheet, column, width)
    except Exception as e:
        logger.error(f"Error setting column width: {str(e)}")

def retry_on_quota_exceeded(max_retries=3, initial_wait=1):
    """
    Decorator to retry operations when quota is exceeded.
    
    Args:
        max_retries (int, optional): Maximum number of retry attempts
        initial_wait (int, optional): Initial wait time in seconds
        
    Returns:
        function: Decorated function
    """
    logger.debug(f"retry_on_quota_exceeded called with max_retries={max_retries}, initial_wait={initial_wait}")
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Executing {func.__name__} with retry_on_quota_exceeded")
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except gspread.exceptions.APIError as e:
                    if e.response.status_code == 429 and attempt < max_retries:
                        wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Google Sheets API rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Google Sheets API error: {e}")
                        raise
            
            logger.error(f"Retry limit exceeded for {func.__name__}")
            raise Exception(f"Retry limit exceeded for {func.__name__}")
            
        return wrapper
    return decorator

class SheetsHandler:
    """
    Handler for Google Sheets operations with batch processing and error handling.
    """
    
    def __init__(self, spreadsheet_id, service_account_file, batch_size=20):
        """
        Initialize the SheetsHandler.
        
        Args:
            spreadsheet_id (str): Google Sheets spreadsheet ID
            service_account_file (str): Path to service account credentials JSON file
            batch_size (int, optional): Batch size for Google Sheets API calls
        """
        self.spreadsheet_id = spreadsheet_id
        self.service_account_file = service_account_file
        self.batch_size = batch_size
        self.client = None
        self.spreadsheet = None
        self.main_sheet = None
        self.summary_sheet = None

    def backup_main_sheet(self):
        """
        Create a backup of the main sheet if it has at least 30 data points.
        
        Returns:
            str: Name of the backup sheet if created, None otherwise
        """
        if not self.main_sheet:
            self.main_sheet = self.get_or_create_worksheet(MAIN_SHEET_NAME)
        
        values = self.main_sheet.get_all_values()
        if len(values) >= 31:  # 30 data points + header row
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_sheet_name = f"{BACKUP_SHEET_PREFIX}{timestamp}"
            backup_sheet = self.spreadsheet.add_worksheet(title=backup_sheet_name, rows=len(values), cols=len(values[0]))
            backup_sheet.update(values)
            logger.info(f"Backup created: {backup_sheet_name}")
            return backup_sheet_name
        return None

    @retry_on_quota_exceeded()
    def update_sheet_with_results(self, worksheet, results):
        """
        Update the worksheet with the stock analysis results.

        Args:
            worksheet (gspread.Worksheet): Worksheet to update
            results (list): List of dictionaries containing stock analysis results

        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            backup_sheet_name = self.backup_main_sheet()

            # Get existing symbols
            existing_symbols = worksheet.col_values(1)[1:]  # Exclude header

            # Prepare the header row
            header = ['Symbol', 'Score', 'Verdict', 'Total Return', 'Sharpe Ratio', 'Sortino Ratio',
                      'Calmar Ratio', 'Max Drawdown', 'Win Rate', 'Average Win', 'Average Loss']

            # Prepare the data rows
            data = [header]
            for result in results:
                row = [
                    result['symbol'],
                    f"{result['score']:.2f}",
                    result['verdict'],
                    f"{result['total_return']:.2%}",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['sortino_ratio']:.2f}",
                    f"{result['calmar_ratio']:.2f}",
                    f"{result['max_drawdown']:.2%}",
                    f"{result['win_rate']:.2%}",
                    f"{result['avg_win']:.2%}",
                    f"{result['avg_loss']:.2%}"
                ]
                data.append(row)

            # Update the worksheet with the new data
            worksheet.update('A1', data)

            # Append "No results to display" for symbols without results
            processed_symbols = [result['symbol'] for result in results]
            for symbol in existing_symbols:
                if symbol not in processed_symbols:
                    worksheet.append_row([symbol] + ['No results'] * 10)

            # Apply formatting
            header_format = {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.8, "green": 0.8, "blue": 0.8}
            }
            self.format_cells(worksheet, "A1:K1", header_format)

            # Auto-resize columns
            for col in range(1, 12):  # A to K
                set_column_width(worksheet, col, 150)  # Set width to 150 pixels

            logger.info("Sheet updated with results successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating sheet with results: {str(e)}")
            worksheet.update('A1', [['Error updating results']])
            return False

    @retry_on_quota_exceeded()
    def update_summary_tab(self, worksheet):
        """
        Update the summary tab with key advantages of our algorithm and information about sheets.

        Args:
            worksheet (gspread.Worksheet): Summary worksheet to update

        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            summary_content = [
                ["Stock Spike Replicator: The Ultimate Stock Prediction Algorithm"],
                [""],
                ["Key Advantages:"],
                ["1. Hybrid Approach: Combines ML predictions with technical and fundamental analysis"],
                ["2. Advanced ML Models: Utilizes ensemble methods and deep learning for superior accuracy"],
                ["3. Dynamic Scoring: Adapts to different market regimes for reliable recommendations"],
                ["4. Comprehensive Analysis: Incorporates wide range of data points, including sentiment"],
                ["5. Continuous Learning: Daily model retraining to adapt to evolving market conditions"],
                ["6. Focus on Undiscovered Potential: Specializes in high-growth stocks under $1"],
                ["7. User-Centric: Processes user-provided lists and identifies outperformers"],
                ["8. Rigorous Backtesting: Ensures strategy effectiveness across various scenarios"],
                [""],
                ["Why Choose Stock Spike Replicator:"],
                ["- Unparalleled accuracy through advanced models and comprehensive analysis"],
                ["- Unique focus on undiscovered, high-potential stocks trading under $1"],
                ["- Personalized insights by enhancing user-provided stock lists"],
                ["- Adaptive strategies that evolve with changing market conditions"],
                ["- Transparent methodology with extensive backtesting and validation"],
                ["- Continuous improvement through daily automated model retraining"],
                [""],
                ["Important Information:"],
                [f"- Main results are in the '{MAIN_SHEET_NAME}' sheet"],
                [f"- Backup sheets are created with the prefix '{BACKUP_SHEET_PREFIX}' when there are 30+ data points"],
                ["- The scheduler runs independently within the application and does not require Windows Task Scheduler"],
                ["- Daily updates are performed automatically when the application is running"],
                [""],
                ["Machine Learning Status:"],
                ["- We have a backtested predictive model for each stock"],
                ["- Models are retrained daily to adapt to new market conditions"],
                ["- Each model uses a combination of technical indicators and historical price data"],
                [""],
                ["Recommendation Explanation:"],
                ["- Strong Buy: High ML prediction score (>0.8) and strong technical/fundamental indicators"],
                ["- Buy: Positive ML prediction (>0.6) and favorable market conditions"],
                ["- Hold: Neutral ML prediction or mixed technical/fundamental signals"],
                ["- Sell: Negative ML prediction (<0.4) or deteriorating market conditions"],
                ["- Strong Sell: Low ML prediction score (<0.2) and weak technical/fundamental indicators"],
                [""],
                ["Using the Web Interface vs Running main.py:"],
                ["1. Web Interface:"],
                ["   - User-friendly dashboard for real-time analysis"],
                ["   - Interactive charts and visualizations"],
                ["   - Easy filtering and sorting of stock recommendations"],
                ["   - Manual backtesting and custom date range analysis"],
                ["   - Watchlist management and alerts"],
                ["   - No need for local Python environment setup"],
                [""],
                ["2. Running main.py:"],
                ["   - Suitable for automated scheduled runs"],
                ["   - Can be integrated into other Python scripts or workflows"],
                ["   - Allows for easy customization of analysis parameters"],
                ["   - Requires Python environment and dependencies installed locally"],
                [""],
                ["For most users, we recommend using the web interface for its ease of use and additional features."],
                ["However, advanced users may prefer running main.py for more granular control and automation capabilities."],
            ]

            # Update the summary content
            worksheet.update('A1', summary_content)

            # Apply formatting
            title_format = {
                "textFormat": {"bold": True, "fontSize": 14},
                "horizontalAlignment": "CENTER"
            }
            subtitle_format = {
                "textFormat": {"bold": True, "fontSize": 12},
                "horizontalAlignment": "LEFT"
            }
            body_format = {
                "textFormat": {"fontSize": 11},
                "horizontalAlignment": "LEFT"
            }

            self.format_cells(worksheet, "A1:A1", title_format)
            self.format_cells(worksheet, "A3:A3", subtitle_format)
            self.format_cells(worksheet, "A13:A13", subtitle_format)
            self.format_cells(worksheet, "A21:A21", subtitle_format)
            self.format_cells(worksheet, "A26:A26", subtitle_format)
            self.format_cells(worksheet, "A31:A31", subtitle_format)
            self.format_cells(worksheet, "A38:A38", subtitle_format)
            self.format_cells(worksheet, "A4:A55", body_format)

            # Adjust column width
            set_column_width(worksheet, 1, 500)  # Set width of column A to 500 pixels

            logger.info("Summary tab updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating summary tab: {str(e)}")
            try:
                worksheet.update('A1', [['Error updating summary tab']])
            except Exception as inner_e:
                logger.error(f"Error updating error message: {str(inner_e)}")
            return False
        
    def initialize(self):
        """
        Initialize the connection to Google Sheets.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Set up credentials and authorize the client
            creds = Credentials.from_service_account_file(
                self.service_account_file,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
            self.client = gspread.authorize(creds)
            
            # Open the spreadsheet
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets connection: {str(e)}")
            return False
    
    def get_or_create_worksheet(self, title, rows=100, cols=26):
        """
        Get a worksheet by title, creating it if it doesn't exist.
        
        Args:
            title (str): Worksheet title
            rows (int, optional): Number of rows for new worksheet
            cols (int, optional): Number of columns for new worksheet
            
        Returns:
            gspread.Worksheet: The worksheet
            
        Raises:
            Exception: If the worksheet couldn't be retrieved or created
        """
        if not self.spreadsheet:
            if not self.initialize():
                raise Exception("Could not initialize Google Sheets connection")
                
        try:
            # Try to get the existing worksheet
            worksheet = self.spreadsheet.worksheet(title)
            logger.info(f"Found existing worksheet: {title}")
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            # Create a new worksheet if it doesn't exist
            logger.info(f"Worksheet '{title}' not found. Creating it.")
            worksheet = self.spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
            return worksheet
    
    @staticmethod
    def retry_on_quota_exceeded(max_retries=3, initial_wait=1):
        """
        Decorator to retry operations when quota is exceeded.
        
        Args:
            max_retries (int, optional): Maximum number of retry attempts
            initial_wait (int, optional): Initial wait time in seconds
            
        Returns:
            function: Decorated function
        """
        logger.debug(f"retry_on_quota_exceeded called with max_retries={max_retries}, initial_wait={initial_wait}")
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                logger.debug(f"Executing {func.__name__} with retry_on_quota_exceeded")
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except gspread.exceptions.APIError as e:
                        if e.response.status_code == 429 and attempt < max_retries:
                            wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Google Sheets API rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Google Sheets API error: {e}")
                            raise
                
                logger.error(f"Retry limit exceeded for {func.__name__}")
                raise Exception(f"Retry limit exceeded for {func.__name__}")
                
            return wrapper
        return decorator
    
    @retry_on_quota_exceeded()
    def batch_update(self, worksheet, updates):
        """
        Update cells in batches to avoid exceeding API limits.
        
        Args:
            worksheet (gspread.Worksheet): Worksheet to update
            updates (list): List of update dictionaries with 'range' and 'values' keys
            
        Returns:
            bool: True if all updates were successful
        """
        if not updates:
            return True
            
        for i in range(0, len(updates), self.batch_size):
            batch = updates[i:i + self.batch_size]
            try:
                worksheet.batch_update(batch)
                # Short delay between batches to avoid rate limits
                if i + self.batch_size < len(updates):
                    time.sleep(0.5)
            except gspread.exceptions.APIError as e:
                logger.error(f"Batch update API error: {str(e)}")
                raise  # Re-raise the APIError for the decorator to handle
            except Exception as e:
                logger.error(f"Batch update error: {str(e)}")
                return False
                
        return True
    
    @retry_on_quota_exceeded()
    def batch_clear(self, worksheet, ranges):
        """
        Clear ranges in a worksheet.
        
        Args:
            worksheet (gspread.Worksheet): Worksheet to clear
            ranges (list): List of range strings to clear
            
        Returns:
            bool: True if all clears were successful
        """
        if not ranges:
            return True
            
        try:
            worksheet.batch_clear(ranges)
            return True
        except gspread.exceptions.APIError as e:
            logger.error(f"Batch clear API error: {str(e)}")
            raise  # Re-raise the APIError for the decorator to handle
        except Exception as e:
            logger.error(f"Batch clear error: {str(e)}")
            return False
    
    def get_all_values(self, worksheet):
        """
        Get all values from a worksheet.
        
        Args:
            worksheet (gspread.Worksheet): Worksheet to read
            
        Returns:
            list: List of rows, each with a list of values
        """
        try:
            return worksheet.get_all_values()
        except Exception as e:
            logger.error(f"Error getting worksheet values: {str(e)}")
            return []
    
    def get_range(self, worksheet, range_name, value_render_option='FORMATTED_VALUE'):
        """
        Get values from a specific range.
        
        Args:
            worksheet (gspread.Worksheet): Worksheet to read
            range_name (str): A1 notation range to get
            value_render_option (str, optional): How to render the values
            
        Returns:
            list: List of rows, each with a list of values
        """
        try:
            return worksheet.get(range_name, value_render_option=value_render_option)
        except Exception as e:
            logger.error(f"Error getting range values: {str(e)}")
            return []
    
    def format_cells(self, worksheet, cell_range, format_dict):
        """
        Apply formatting to a range of cells.
        
        Args:
            worksheet (gspread.Worksheet): Worksheet to format
            cell_range (str): A1 notation range to format
            format_dict (dict): Format specifications
            
        Returns:
            bool: True if formatting was successful
        """
        try:
            worksheet.format(cell_range, format_dict)
            return True
        except Exception as e:
            logger.error(f"Error formatting cells: {str(e)}")
            return False

    @retry_on_quota_exceeded()
    def insert_image(self, worksheet, cell, image_buffer):
        """
        Insert an image into a specific cell in the worksheet.

        Args:
            worksheet (gspread.Worksheet): Worksheet to insert the image into
            cell (str): A1 notation of the cell to insert the image
            image_buffer (BytesIO): BytesIO object containing the image data

        Returns:
            bool: True if the image was inserted successfully, False otherwise
        """
        try:
            # Convert the image buffer to base64
            image_data = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

            # Prepare the image formula
            image_formula = f'=IMAGE("data:image/png;base64,{image_data}")'

            # Update the cell with the image formula
            worksheet.update(cell, image_formula)

            # Adjust column width to fit the image
            col = gspread.utils.a1_to_rowcol(cell)[1]
            set_column_width(worksheet, col, 400)  # Set width to 400 pixels

            logger.info(f"Image inserted successfully in cell {cell}")
            return True
        except Exception as e:
            logger.error(f"Error inserting image: {str(e)}")
            return False

    @retry_on_quota_exceeded()
    def update_summary_tab(self, worksheet):
        """
        Update the summary tab with key advantages of our algorithm.

        Args:
            worksheet (gspread.Worksheet): Summary worksheet to update

        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            summary_content = [
                ["Stock Spike Replicator: The Ultimate Stock Prediction Algorithm"],
                [""],
                ["Key Advantages:"],
                ["1. Hybrid Approach: Combines ML predictions with technical and fundamental analysis"],
                ["2. Advanced ML Models: Utilizes ensemble methods and deep learning for superior accuracy"],
                ["3. Dynamic Scoring: Adapts to different market regimes for reliable recommendations"],
                ["4. Comprehensive Analysis: Incorporates wide range of data points, including sentiment"],
                ["5. Continuous Learning: Daily model retraining to adapt to evolving market conditions"],
                ["6. Focus on Undiscovered Potential: Specializes in high-growth stocks under $1"],
                ["7. User-Centric: Processes user-provided lists and identifies outperformers"],
                ["8. Rigorous Backtesting: Ensures strategy effectiveness across various scenarios"],
                [""],
                ["Why Choose Stock Spike Replicator:"],
                ["- Unparalleled accuracy through advanced models and comprehensive analysis"],
                ["- Unique focus on undiscovered, high-potential stocks trading under $1"],
                ["- Personalized insights by enhancing user-provided stock lists"],
                ["- Adaptive strategies that evolve with changing market conditions"],
                ["- Transparent methodology with extensive backtesting and validation"],
                ["- Continuous improvement through daily automated model retraining"],
            ]

            # Update the summary content
            worksheet.update('A1', summary_content)

            # Apply formatting
            title_format = {
                "textFormat": {"bold": True, "fontSize": 14},
                "horizontalAlignment": "CENTER"
            }
            subtitle_format = {
                "textFormat": {"bold": True, "fontSize": 12},
                "horizontalAlignment": "LEFT"
            }
            body_format = {
                "textFormat": {"fontSize": 11},
                "horizontalAlignment": "LEFT"
            }

            self.format_cells(worksheet, "A1:A1", title_format)
            self.format_cells(worksheet, "A3:A3", subtitle_format)
            self.format_cells(worksheet, "A13:A13", subtitle_format)
            self.format_cells(worksheet, "A4:A19", body_format)

            # Adjust column width
            set_column_width(worksheet, 1, 500)  # Set width of column A to 500 pixels

            logger.info("Summary tab updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating summary tab: {str(e)}")
            try:
                worksheet.update('A1', [['Error updating summary tab']])
            except Exception as inner_e:
                logger.error(f"Error updating error message: {str(inner_e)}")
            return False
        finally:
            # Ensure the worksheet is saved
            try:
                worksheet.spreadsheet.client.session.close()
            except Exception as close_e:
                logger.error(f"Error closing spreadsheet session: {str(close_e)}")

    @retry_on_quota_exceeded()
    def update_sheet_with_results(self, worksheet, results):
        """
        Update the worksheet with the stock analysis results.

        Args:
            worksheet (gspread.Worksheet): Worksheet to update
            results (list): List of dictionaries containing stock analysis results

        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            backup_sheet_name = self.backup_main_sheet()

            # Get existing symbols
            existing_symbols = worksheet.col_values(1)[1:]  # Exclude header

            # Prepare the header row
            header = ['Symbol', 'Source', 'Date', 'Price in 1 week', 'Price in 1 day', 'Price now', 'Last updated',
                      'Support (20-day low)', 'Resistance (20-day high)', 'Support & Resistance', 'Price in date pulled',
                      'Price diff', 'Verdict', "Today's date", 'RSI (0-100)', 'Sharpe Ratio (annualized)',
                      'Bollinger Squeeze?', 'Max Drawdown %', 'Beta', 'Composite Score (0-1)', 'MA50/200 Ratio',
                      '20d Momentum %', 'P/E Ratio', 'Debt/Equity', 'Explanation', 'Total Return', 'Sortino Ratio',
                      'Calmar Ratio', 'Win Rate', 'Average Win', 'Average Loss']

            # Prepare the data rows
            data = [header]
            processed_symbols = []
            for result in results:
                row = [
                    result['symbol'],
                    'Stock Spike Replicator',
                    datetime.now().strftime('%Y-%m-%d'),
                    'N/A',  # Price in 1 week (prediction not implemented)
                    'N/A',  # Price in 1 day (prediction not implemented)
                    result.get('current_price', 'N/A'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    result.get('support', 'N/A'),
                    result.get('resistance', 'N/A'),
                    f"{result.get('support', 'N/A')} - {result.get('resistance', 'N/A')}",
                    result.get('price_date_pulled', 'N/A'),
                    result.get('price_diff', 'N/A'),
                    result['verdict'],
                    datetime.now().strftime('%Y-%m-%d'),
                    result.get('rsi', 'N/A'),
                    f"{result['sharpe_ratio']:.2f}",
                    result.get('bollinger_squeeze', 'N/A'),
                    f"{result['max_drawdown']:.2%}",
                    result.get('beta', 'N/A'),
                    f"{result['score']:.2f}",
                    result.get('ma50_200_ratio', 'N/A'),
                    result.get('momentum_20d', 'N/A'),
                    result.get('pe_ratio', 'N/A'),
                    result.get('debt_equity', 'N/A'),
                    result.get('explanation', 'N/A'),
                    f"{result['total_return']:.2%}",
                    f"{result['sortino_ratio']:.2f}",
                    f"{result['calmar_ratio']:.2f}",
                    f"{result['win_rate']:.2%}",
                    f"{result['avg_win']:.2%}",
                    f"{result['avg_loss']:.2%}"
                ]
                data.append(row)
                processed_symbols.append(result['symbol'])

            # Add rows for symbols without results
            for symbol in existing_symbols:
                if symbol not in processed_symbols:
                    data.append([symbol] + ['N/A'] * (len(header) - 1))

            # Update the worksheet with the new data
            worksheet.update('A1', data)

            # Apply formatting
            header_format = {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.8, "green": 0.8, "blue": 0.8}
            }
            self.format_cells(worksheet, f"A1:{chr(64 + len(header))}1", header_format)

            # Auto-resize columns
            for col in range(1, len(header) + 1):
                set_column_width(worksheet, col, 150)  # Set width to 150 pixels

            logger.info("Sheet updated with results successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating sheet with results: {str(e)}")
            return False