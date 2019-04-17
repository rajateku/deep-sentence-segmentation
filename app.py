from config import PORT, HOST, DEBUG
from flask_setup import app
from routes import *

if __name__ == '__main__':
    app.debug = DEBUG
    app.run(host=HOST, port=int(PORT))
