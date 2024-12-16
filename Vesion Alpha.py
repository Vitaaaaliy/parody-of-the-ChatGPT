import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def preprocess_text(text, seq_length):
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
    if not input_sequences:
        raise ValueError(f"Не удалось создать входные последовательности. Проверьте параметр seq_length ({seq_length}) и текст.")
    
    # Приведение всех последовательностей к одинаковой длине
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Создание обучающих данных
    X = input_sequences[:, :-1]
    y = to_categorical(input_sequences[:, -1], num_classes=total_words)
    
    return X, y, tokenizer, total_words, max_sequence_len


def create_model(total_words, max_sequence_len):
    # Определение модели
    model = Sequential([
        Embedding(total_words, 128, input_length=max_sequence_len - 1),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dense(total_words, activation='softmax')
    ])
    
    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=40, temperature=1.0):
    generated_words = set(seed_text.split())
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Температура для регулирования случайности
        predicted = np.log(predicted) / temperature
        predicted = np.exp(predicted) / np.sum(np.exp(predicted))
        next_index = np.random.choice(range(total_words), p=predicted)
        
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

"""

# Параметры
seq_length = 10  # Длина последовательности
seed_text = " ".join(text.split(' ')[:seq_length])  # Первые 10 слов

# Подготовка данных
X, y, tokenizer, total_words, max_sequence_len = preprocess_text(text, seq_length)

# Создание и обучение модели
model = create_model(total_words, max_sequence_len)
early_stopping = EarlyStopping(monitor='loss', patience=5)
history = model.fit(X, y, epochs=100, verbose=1, callbacks=[early_stopping])

# Генерация текста
generated_text = generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=40)

print(generated_text)