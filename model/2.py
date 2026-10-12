import pandas as pd

df = pd.read_csv('annotations.csv', encoding='utf-8', sep='\t')
df = df[df['text'] != 'no_event']

# Сколько всего видео в каждой категории
letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')

df_letters = df[df['text'].isin(letters)]
df_words = df[~df['text'].isin(letters)]

print(f"Всего видео: {len(df)}")
print(f"Видео с буквами: {len(df_letters)}")
print(f"Видео со словами: {len(df_words)}")
print(f"Уникальных букв: {df_letters['text'].nunique()}")
print(f"Уникальных слов: {df_words['text'].nunique()}")

# Сколько видео на класс в среднем
print(f"\nСреднее видео на букву: {len(df_letters) / df_letters['text'].nunique():.1f}")
print(f"Среднее видео на слово: {len(df_words) / df_words['text'].nunique():.1f}")

# Проверка: сколько всего видеофайлов на диске
import os
train_files = os.listdir('train') if os.path.exists('train') else []
test_files = os.listdir('test') if os.path.exists('test') else []
print(f"\nФайлов в папке train: {len(train_files)}")
print(f"Файлов в папке test: {len(test_files)}")
