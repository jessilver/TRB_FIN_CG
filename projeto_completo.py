
# Generated from notebook Projeto2.ipynb

# Imports
import os
import numpy as np
import torch
import torchvision
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_metric
from datasets import load_dataset
import warnings
from PIL import Image
from IPython.display import display
import json

# Setup
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Automatizando a Carga dos Seus Próprios Dados Para Ajuste do Modelo
print("Carregando dataset...")
dados = load_dataset("imagefolder", data_files = "data/dataset.zip")
print("Dataset carregado:")
print(dados)
print(type(dados))

# Explorando os Dados
print("\nExplorando os dados...")
print("Features do dataset de treino:")
print(dados["train"].features)
print("\nDetalhes da imagem de índice 1142:")
print(dados['train'][1142])
imagem = dados["train"][1142]
print("\nObjeto da imagem:")
print(imagem)
print("Label da imagem:")
print(imagem['label'])
print("Visualizando a imagem (uma janela pode abrir):")
# imagem['image'].show() # Descomente para visualizar

# Criando Mapeamentos Índice/Nome de Classe
print("\nCriando mapeamentos de labels...")
labels = dados["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print("Mapeamento id2label:")
print(id2label)
print("Mapeamento label2id:")
print(label2id)

# Pré-Processamento das Imagens
print("\nIniciando pré-processamento...")
dsa_modelo_hf = "google/vit-base-patch16-224"
dsa_image_processor = AutoImageProcessor.from_pretrained(dsa_modelo_hf)
print("Processador de imagem carregado:")
print(dsa_image_processor)

dsa_normalize = Normalize(mean = dsa_image_processor.image_mean, std = dsa_image_processor.image_std)
print("Tipo do objeto de normalização:")
print(type(dsa_normalize))

if "height" in dsa_image_processor.size:
    size = (dsa_image_processor.size["height"], dsa_image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in dsa_image_processor.size:
    size = dsa_image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = dsa_image_processor.size.get("longest_edge")

transformacoes_treino = Compose([RandomResizedCrop(crop_size), RandomHorizontalFlip(), ToTensor(), dsa_normalize])
transformacoes_valid = Compose([Resize(size), CenterCrop(crop_size), ToTensor(), dsa_normalize])

def dsa_preprocessa_treino(lote_dados):
    lote_dados["pixel_values"] = [transformacoes_treino(image.convert("RGB")) for image in lote_dados["image"]]
    return lote_dados

def dsa_preprocessa_valid(lote_dados):
    lote_dados["pixel_values"] = [transformacoes_valid(image.convert("RGB")) for image in lote_dados["image"]]
    return lote_dados

splits = dados["train"].train_test_split(test_size = 0.1)
dados_treino = splits['train']
dados_treino.set_transform(dsa_preprocessa_treino)
dados_valid = splits['test']
dados_valid.set_transform(dsa_preprocessa_valid)
print("Dados de treino e validação pré-processados.")

# Definindo Argumentos e Hiperparâmetros do Fine-Tuning
print("\nDefinindo argumentos de fine-tuning...")
modelo_salvar = dsa_modelo_hf.split("/")[-1]

batch_size = 32
taxa_aprendizado = 5e-5
accumulation_steps = 4
num_epochs = 3
wratio = 0.1
lsteps = 10

dsa_args = TrainingArguments(f"{modelo_salvar}-dsa-p2-finetuned",
                             remove_unused_columns = False,
                             evaluation_strategy = "epoch",
                             save_strategy = "epoch",
                             learning_rate = taxa_aprendizado,
                             per_device_train_batch_size = batch_size,
                             gradient_accumulation_steps = accumulation_steps,
                             per_device_eval_batch_size = batch_size,
                             num_train_epochs = num_epochs,
                             warmup_ratio = wratio,
                             logging_steps = lsteps,
                             load_best_model_at_end = True,
                             metric_for_best_model = "accuracy",
                             push_to_hub = False)

dsa_metrica = load_metric("accuracy")

def dsa_compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis = 1)
    return dsa_metrica.compute(predictions = predictions, references = eval_pred.label_ids)

def dsa_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Treinamento do Modelo
print("\nConfigurando o treinamento...")
modelo_salvo_path = "models/oregon-wild-life-trained-model"

# Verificamos se o modelo já existe
if os.path.exists(os.path.join(modelo_salvo_path, "model.safetensors")) or os.path.exists(os.path.join(modelo_salvo_path, "pytorch_model.bin")):
    print("Modelo já treinado encontrado. Carregando modelo salvo...")
    modelo = AutoModelForImageClassification.from_pretrained(modelo_salvo_path)
    # Como o trainer não foi salvo, precisamos recriá-lo para a avaliação
    dsa_trainer = Trainer(modelo,
                          dsa_args,
                          train_dataset = dados_treino,
                          eval_dataset = dados_valid,
                          tokenizer = dsa_image_processor,
                          compute_metrics = dsa_compute_metrics,
                          data_collator = dsa_collate_fn)
else:
    print("Modelo salvo não encontrado. Treinando modelo...")
    modelo = AutoModelForImageClassification.from_pretrained(
        dsa_modelo_hf,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    dsa_trainer = Trainer(
        modelo,
        dsa_args,
        train_dataset=dados_treino,
        eval_dataset=dados_valid,
        tokenizer=dsa_image_processor,
        compute_metrics=dsa_compute_metrics,
        data_collator=dsa_collate_fn
    )
    resultados_treino = dsa_trainer.train()
    print("Salvando modelo e métricas...")
    dsa_trainer.save_model(modelo_salvo_path)
    dsa_trainer.log_metrics("train", resultados_treino.metrics)
    dsa_trainer.save_metrics("train", resultados_treino.metrics)
    dsa_trainer.save_state()

# Avaliação do Modelo
print("\nAvaliando o modelo...")
avaliacao_salva_path = os.path.join(modelo_salvo_path, "eval_results.json")

if os.path.exists(avaliacao_salva_path):
    print("Resultados de avaliação encontrados. Carregando resultados salvos...")
    with open(avaliacao_salva_path, "r") as f:
        avaliador = json.load(f)
else:
    print("Resultados de avaliação não encontrados. Avaliando modelo...")
    avaliador = dsa_trainer.evaluate()
    dsa_trainer.log_metrics("eval", avaliador)
    dsa_trainer.save_metrics("eval", avaliador)

print("Métricas de avaliação:")
print(avaliador)

# Usando o Modelo Para Previsões com Novas Imagens
print("\nUsando o modelo para previsão...")

try:
    # Carrega a imagem
    image_path = 'imagem01.jpeg'
    image = Image.open(image_path)
    print(f"Carregando imagem: {image_path}")

    # Mostra a imagem redimensionada
    img_pequena = image.resize((224, 224))
    # img_pequena.show() # Descomente para visualizar

    # Preprocessa a imagem
    image_processed = transformacoes_valid(image.convert("RGB")).unsqueeze(0)

    # Move a imagem para o mesmo dispositivo do modelo
    image_processed = image_processed.to(modelo.device)

    # Faz a previsão
    with torch.no_grad():
        logits = modelo(image_processed).logits

    # Pega o label previsto
    id_label_previsto = logits.argmax(-1).item()
    nome_label_previsto = id2label[id_label_previsto]

    print(f"\nLabel Previsto para '{image_path}': {nome_label_previsto}")

except FileNotFoundError:
    print(f"Erro: Arquivo de imagem não encontrado em '{image_path}'. Certifique-se de que a imagem está no diretório correto.")
except Exception as e:
    print(f"Ocorreu um erro durante a previsão: {e}")

print("\nScript finalizado.")

# Comandos do Watermark (informativo, não executável em .py padrão)
# %reload_ext watermark
# %watermark -a "Universidade Federal do Tocantins"
# %watermark -v -m
# %watermark --iversions
