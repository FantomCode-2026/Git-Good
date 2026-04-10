import urllib.request
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import threading, http.server, socketserver
import time

PORT = 8003
Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(('', PORT), Handler)
t = threading.Thread(target=httpd.serve_forever)
t.daemon = True
t.start()

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

driver = webdriver.Chrome(options=chrome_options)
driver.get(f'http://localhost:{PORT}/digitaltwin/temple_simulation (3).html')
time.sleep(2)

logs = driver.get_log('browser')
for log in logs:
    if log['level'] in ['SEVERE', 'WARNING']:
        print(f"[{log['level']}] {log['message']}")

driver.quit()
