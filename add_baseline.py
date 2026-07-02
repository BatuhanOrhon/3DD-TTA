import json

with open('3dd_tta.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### Adım 9: Orijinal 3DD-TTA (Baseline) Karşılaştırma Testleri\n',
        'Bu adımda yazarların makalede kullandığı orijinal Chamfer Distance güdümlü (GSDTTA spektral loss olmadan) TTA algoritmasını çalıştırıyoruz.\n',
        'Böylece kendi önerdiğimiz Graph Spectral (GSDTTA) yöntemiyle orijinal yöntemin sonuçlarını (hem görsel hem sayısal olarak) kıyaslayabileceksiniz.'
    ]
}

code_cell_1 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 1. Baseline Qualitative (Görsel) Testi\n',
        '# Sadece Chamfer Distance weight kullanıyoruz, Spectral weight = 0.0\n',
        '!conda run --no-capture-output -n 3dd_tta_env python demo_3dd_tta.py \\\n',
        '  --diff_ckpt=./lion_ckpts/epoch_10999_iters_2100999.pt \\\n',
        '  --denoising_step=35 \\\n',
        '  --dataset_root=./data/modelnet40_c \\\n',
        '  --corruption=background \\\n',
        '  --sample_id=11 \\\n',
        '  --weight_spectral 0.0 \\\n',
        '  --weight_chamfer 1.0\n'
    ]
}

code_cell_2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 2. Baseline Quantitative (Sayısal) Testi on ModelNet40-C\n',
        '# Yine aynı şekilde Spectral = 0.0, Chamfer = 1.0 ile accuracy ölçümü yapıyoruz.\n',
        '!conda run --no-capture-output -n 3dd_tta_env python main_3dd_tta.py \\\n',
        '  --dataset_name modelnet-c \\\n',
        '  --dataset_root ./data/modelnet40_c \\\n',
        '  --label_path ./data/modelnet40_c/label.npy \\\n',
        '  --pointmae_config ./cfgs/tta_modelnet.yaml \\\n',
        '  --pointmae_ckpt ./pointnet_ckpts/modelnet_jt.pth \\\n',
        '  --batch_size 16 \\\n',
        '  --weight_spectral 0.0 \\\n',
        '  --weight_chamfer 1.0\n'
    ]
}

nb['cells'].append(md_cell)
nb['cells'].append(code_cell_1)
nb['cells'].append(code_cell_2)

with open('3dd_tta.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Notebook updated successfully with baseline comparison cells.')
