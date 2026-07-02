import json

with open('3dd_tta.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modify Adım 6: Qualitative Evaluation to use demo_gsd_tta.py
        if 'demo_3dd_tta.py' in source and '--weight_spectral' not in source:
            new_source = source.replace('demo_3dd_tta.py', 'demo_gsd_tta.py')
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            # Since the original lines might have \n, we just replace in string
            
        # Adım 7: GSDTTA Quantitative Test
        elif 'main_3dd_tta.py' in source and '--weight_spectral 1.0' in source:
            new_source = source.replace('main_3dd_tta.py', 'main_gsd_tta.py')
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            
        # Adım 9: Baseline Qualitative
        elif 'demo_3dd_tta.py' in source and '--weight_spectral 0.0' in source:
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if '--weight_spectral' in line or '--weight_chamfer' in line:
                    continue
                if '--sample_id=11 \\' in line:
                    new_lines.append(line.replace(' \\', '')) # Remove trailing slash from previous line
                else:
                    new_lines.append(line)
            cell['source'] = [line + '\n' for line in new_lines if line]
            
        # Adım 9: Baseline Quantitative
        elif 'main_3dd_tta.py' in source and '--weight_spectral 0.0' in source:
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if '--weight_spectral' in line or '--weight_chamfer' in line:
                    continue
                if '--batch_size 16 \\' in line:
                    new_lines.append(line.replace(' \\', ''))
                else:
                    new_lines.append(line)
            cell['source'] = [line + '\n' for line in new_lines if line]

with open('3dd_tta.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook updated successfully.")
