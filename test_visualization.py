import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from updatesheet import StockUpdateApp
from visualizations import plot_top_stock_pick

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.app = StockUpdateApp()
        self.app.symbol_data_cache = {
            'AAPL': {
                'score': 0.8,
                'price': 150.0,
                'support': 145.0,
                'resistance': 155.0,
                'explanation': 'Strong buy based on technical indicators'
            }
        }
        self.app.historical_data = {
            'AAPL': MagicMock()  # Mock historical data
        }

    @patch('updatesheet.plot_top_stock_pick')
    @patch('updatesheet.SheetsHandler')
    def test_plot_top_pick(self, mock_sheets_handler, mock_plot_top_stock_pick):
        # Mock the plot_top_stock_pick function to return a BytesIO object
        mock_plot_top_stock_pick.return_value = BytesIO(b'mock image data')

        # Mock the SheetsHandler methods
        mock_sheets_handler_instance = mock_sheets_handler.return_value
        mock_sheets_handler_instance.get_worksheet.return_value = MagicMock()
        mock_sheets_handler_instance.insert_image.return_value = None
        mock_sheets_handler_instance.batch_update.return_value = None

        # Set the mocked SheetsHandler to the app
        self.app.sheets_handler = mock_sheets_handler_instance

        # Call the method we're testing
        self.app.plot_top_pick()

        # Assert that plot_top_stock_pick was called with the correct arguments
        mock_plot_top_stock_pick.assert_called_once_with(
            'AAPL',
            self.app.historical_data['AAPL'],
            150.0,
            145.0,
            155.0
        )

        # Assert that the image was inserted into the sheet
        mock_sheets_handler_instance.insert_image.assert_called_once()

        # Assert that the description was added to the sheet
        mock_sheets_handler_instance.batch_update.assert_called_once()

if __name__ == '__main__':
    unittest.main()