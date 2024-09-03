import sys
import unittest
from unittest.mock import patch,MagicMock
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import download_images
sys.path.append('C:/Users/laran/PycharmProjects/ASX/stock_market/fl_multimodal_knowledge_graph/data_collection/data_collection_tools')

class TestDownloadImages(unittest.TestCase):
    @patch('builtins.input',return_value='stock market')
    @patch('download_images.create_search_uri')
    @patch('download_images.requests.get')

    def test_download_images(self,mock_create_search_uri,mock_requests_get,mock_input):
        # Mocking the responses
        mock_create_search_uri.return_value = ['/mocked_uri']
        mock_response = MagicMock
        mock_response.content = b'mocked image data'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        # Defining the text and save path
        text = 'stock market'
        save_path = './downloads'

        # Calling the function
        record = download_images.download_images(text,save_path)

        # Assertions
        self.assertEqual(len(record),1)
if __name__ == '__main__':
    unittest.main()
