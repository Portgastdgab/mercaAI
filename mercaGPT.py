import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# 1. Cargar el dataset
data = pd.read_csv('amazon_cells_labelled.csv', header=None, names=['comment', 'label'])

# 2. Eliminar filas con valores nulos (si existen)
data.dropna(inplace=True)

# 3. Dividir el dataset en conjunto de entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# 4. Inicializar el tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# 5. Tokenización
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 6. Crear un dataset compatible con PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# 7. Cargar el modelo (actualizar el modelo a uno con 2 etiquetas)
model = RobertaForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment', 
    num_labels=2,  # Ajustar aquí si decides cambiar a 3 etiquetas
    ignore_mismatched_sizes=True
)

# 8. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=500,                 
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)

# 9. Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10. Entrenar el modelo
trainer.train()

# 11. Hacer predicciones
def predict_sentiment(comment):
    encoding = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=128)
    encoding = encoding.to(model.device)  # Mover a la GPU si está disponible
    with torch.no_grad():
        output = model(**encoding)
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]  # Obtener probabilidades
    return probabilities

# Ejemplo de uso
if __name__ == "__main__":
    example_comment = "I didn't like the product"
    probabilities = predict_sentiment(example_comment)
    labels = ['Negative', 'Positive']  # Ajustar etiquetas según tu dataset

    # Mostrar resultados
    for label, probability in zip(labels, probabilities):
        print(f"{label}: {probability:.4f}")
