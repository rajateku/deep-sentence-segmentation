import os

PORT = os.getenv('PORT', 8096)

HOST = os.getenv('HOST', '0.0.0.0')

DEBUG = os.getenv('DEBUG', False)

NER_URL = os.getenv('NER_URL', 'http://localhost:8096/sentence_detection')
