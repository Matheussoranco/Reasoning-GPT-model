import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Função para criar o bloco Transformer
def transformer_block(embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(embed_dim)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    
    return Model(inputs, ff_output, name="TransformerBlock")

# Parâmetros do modelo
vocab_size = 20000  # Aumentar o vocabulário para raciocínio mais complexo
embed_dim = 256     # Dimensão dos embeddings ampliada
num_heads = 8       # Número de cabeças de atenção aumentado
ff_dim = 512        # Dimensão da feed-forward network aumentada
num_blocks = 6      # Mais blocos Transformer para maior profundidade
sequence_length = 50  # Comprimento máximo da sequência

# Entrada e Embedding
inputs = Input(shape=(sequence_length,), dtype=tf.int32)
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

# Aplicar vários blocos Transformer
x = embedding_layer
for _ in range(num_blocks):
    x = transformer_block(embed_dim, num_heads, ff_dim)(x)

# Camada final
x = Dense(ff_dim, activation="relu")(x)
x = Dense(vocab_size, activation="softmax")(x)

# Modelo
model = Model(inputs, x, name="AdvancedMiniGPT")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Dados sintéticos de exemplo (para demonstração de estrutura)
def create_synthetic_data(num_samples, sequence_length, vocab_size):
    x_data = np.random.randint(0, vocab_size, size=(num_samples, sequence_length))
    y_data = np.random.randint(0, vocab_size, size=(num_samples, sequence_length))
    return x_data, y_data

# Gerar dados sintéticos para treinamento inicial
num_samples = 5000
x_train, y_train = create_synthetic_data(num_samples, sequence_length, vocab_size)

# Ajuste para um conjunto de dados real de raciocínio abstrato
# Exemplo: carregar um dataset como GSM8K, AQuA ou CommonsenseQA (tokenizado e padronizado)
# Aqui colocamos uma estrutura placeholder para integração futura
# x_train, y_train = load_real_dataset("path_to_dataset")

# Treinamento
model.fit(x_train, y_train, batch_size=32, epochs=10)

# Salvar o modelo para experimentação
model.save("advanced_mini_gpt_model.h5")
