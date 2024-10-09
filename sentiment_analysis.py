from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Cargar el tokenizador y el modelo preentrenado de RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Texto para analizar
text = "my math grade was 19"

# Tokenizar el texto
inputs = tokenizer(text, return_tensors="pt").to(device)

# Hacer predicción
with torch.no_grad():
    outputs = model(**inputs)



# Obtener las probabilidades
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
labels = ['Negative', 'Neutral', 'Positive']

# Mostrar los resultados
for label, probability in zip(labels, probabilities[0]):
    print(f"{label}: {probability.item():.4f}")
