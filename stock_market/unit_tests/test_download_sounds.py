import unittest
from unittest.mock import patch, MagicMock
import os
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import download_sounds
import sys
print(sys.path)
sys.path.append('C:/Users/laran/PycharmProjects/ASX/stock_market/fl_multimodal_knowledge_graph/data_collection/data_collection_tools')
class TestDownloadSound(unittest.TestCase):
    @patch('download_sounds.requests.get')  # Using @patch, a decorator to temporarily replace classes, methods, or functions with mock objects during testing
    @patch('download_sounds.get_content')
    @patch('download_sounds.parser_content')
    @patch('download_sounds.get_file_size')
    @patch('download_sounds.download_file')
    def test_download_sound(self,mock_download_file,mock_get_file_size,mock_parser_content,mock_get_content,mock_requests_get):
        # Mocking the responses
        mock_get_content.return_value = 'mocked content'
        mock_parser_content.return_value = ['/mocked_url']
        mock_get_file_size.return_value = 2.0 # MiB
        mock_requests_get.return_value = b'mocked audio data'

        # Defining the search term and save path
        search_term = 'stock market'
        save_path = './downloads'

        # Calling the function
        downloaded_files = download_sounds.download_sound(search_term,save_path)

        # Assertions
        self.assertEqual(len(downloaded_files),0)  # An error exception to check if the length of the audio file downloaded is 0 or not
if __name__ == '__main__':
    unittest.main()