from pathlib import Path
import re
content = Path('sales_model.pkl').read_bytes()
for pattern in [b'1.5.2', b'1.6.1', b'1.7', b'1.8', b'scikit-learn', b'sklearn']:
    print(pattern.decode('utf-8'), content.find(pattern))
text = ''.join(chr(b) if 32 <= b < 127 else ' ' for b in content)
for m in re.finditer(r'[A-Za-z0-9_.+-]{6,80}', text):
    s = m.group(0)
    if any(x in s for x in ['sklearn', 'scikit', '1.', 'pipeline', 'ColumnTransformer', 'RandomForest']):
        print(m.start(), s)
