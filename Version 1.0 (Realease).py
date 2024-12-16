import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def preprocess_text(text, seq_length):
    """Подготовка текста для обучения модели."""
    # Разделяем текст на слова
    words = text.lower().split(' ')
    
    # Создаем токенизатор
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    
    total_words = len(tokenizer.word_index) + 1
    
    # Генерация входных последовательностей
    input_sequences = []
    for i in range(len(words) - seq_length):
        seq = words[i:i + seq_length + 1]
        encoded_seq = tokenizer.texts_to_sequences([' '.join(seq)])[0]
        input_sequences.append(encoded_seq)
        
    # Проверка на пустой список перед вычислением max
    assert input_sequences, f"Не удалось создать входные последовательности. Проверьте параметр seq_length ({seq_length}) и текст."
    
    # Приведение всех последовательностей к одинаковой длине
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Создание обучающих данных
    X = input_sequences[:, :-1]
    y = to_categorical(input_sequences[:, -1], num_classes=total_words)
    
    return X, y, tokenizer, total_words, max_sequence_len


def create_model(total_words, max_sequence_len, embedding_dim=256, lstm_units=(256, 512)):
    """Создание и компиляция модели."""
    # Определение модели
    model = Sequential([
        Embedding(total_words, embedding_dim, input_length=max_sequence_len - 1),
        LSTM(lstm_units[0], return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.2),
        LSTM(lstm_units[1],
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(total_words, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    
    # Компиляция модели
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=50, temperature=1.0):
    """Генерация текста на основе обученной модели."""
    generated_words = set(seed_text.split())
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Применение температуры для регулирования случайности
        predicted /= temperature
        exp_pred = np.exp(predicted)
        next_index = np.argmax(np.random.multinomial(1, exp_pred / np.sum(exp_pred)))
        
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break
                
        if output_word and output_word not in generated_words:
            seed_text += ' ' + output_word
            generated_words.add(output_word)
            
    return seed_text


# Пример текста для обучения
text = """
# Чем больше текст, тем качественнее результат
"""

# Параметры
seq_length = 500  # Длина последовательности
seed_text = " ".join(text.split(' ')[:seq_length])  # Первые 10 слов

# Подготовка данных
X, y, tokenizer, total_words, max_sequence_len = preprocess_text(text, seq_length)

# Создание и обучение модели
model = create_model(total_words, max_sequence_len)
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
history = model.fit(X, y, epochs=10, batch_size=128, validation_split=0.25, callbacks=[early_stopping], verbose=1)

# Генерация текста
generated_text = generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=500, temperature=0.8)

print(generated_text)
