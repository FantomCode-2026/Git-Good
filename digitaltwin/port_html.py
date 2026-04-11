import re

with open('digitaltwin/stadium_simulation.html', 'r', encoding='utf-8') as f:
    stadium_content = f.read()

stadium_html_match = re.search(r'(<div class="ui-overlay">.*?)</script>', stadium_content, re.DOTALL)
stadium_html = stadium_html_match.group(1) if stadium_html_match else ''
stadium_html = stadium_html.rsplit('<script', 1)[0] # remove importmap

with open('digitaltwin/temple_simulation.html', 'r', encoding='utf-8') as f:
    temple_content = f.read()

# Replace the HTML block between <div id="canvas-container"></div> and <script type="importmap">
temple_content = re.sub(r'<div id="ui-overlay">.*?(?=<script type="importmap">)', stadium_html, temple_content, flags=re.DOTALL)

with open('digitaltwin/temple_simulation.html', 'w', encoding='utf-8') as f:
    f.write(temple_content)

print('Replaced HTML')
