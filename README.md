
# Gibi Style Transfer

Treinamento simples de transferência de estilo para transformar imagens comuns em estilo GIBI (HQ, gibi, quadrinhos).

## Estrutura

- `datasets/content/` → imagens normais
- `datasets/style/` → imagens com estilo gibi
- `models/` → onde os modelos salvos e exportados ficarão

## Instalação

```bash
pip install tensorflow tensorflowjs pillow numpy
```

## Como usar

1. Coloque imagens nas pastas `datasets/content/` e `datasets/style/`
2. Treine o modelo:

```bash
python train.py 
# 
python train_ghibli_style.py
```

3. Teste uma imagem de exemplo:

```bash
python test_image.py
# 
python apply_ghibli.py
```

4. Exporte para TensorFlow.js:

```bash
python export_tfjs.py
```

Resultado final estará em `models/gibi_tfjs/` com `model.json` e `.bin`.
