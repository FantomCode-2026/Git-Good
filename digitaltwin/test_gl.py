import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

PORT = 8003

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)
driver.get(f'http://localhost:{PORT}/digitaltwin/temple_simulation.html')
time.sleep(2)

try:
    width = driver.execute_script('return document.querySelector("#canvas-container canvas").width')
    height = driver.execute_script('return document.querySelector("#canvas-container canvas").height')
    print(f'Canvas size: {width}x{height}')

    num_meshes = driver.execute_script('return window.__app.sceneManager.scene.children.length')
    print(f'Meshes in scene: {num_meshes}')

    cam = driver.execute_script('return [window.__app.sceneManager.camera.position.x, window.__app.sceneManager.camera.position.y, window.__app.sceneManager.camera.position.z]')
    print(f'Camera: {cam}')

    frames = driver.execute_script('return window.__app._frameCount')
    print(f'Frames rendered: {frames}')
except Exception as e:
    print(f"Error querying DOM: {e}")

driver.quit()
