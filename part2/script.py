"""
Практическая работа 1. Виртуальные окружения Python.
Команда 2: Anaconda + линейная регрессия (пример GeeksforGeeks).

Скрипт тестирования виртуальной среды для второй команды аналитиков.
Использует датасет Bottle (bottle.csv).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # ============================================================
    # Шаг 1: Чтение датасета
    # ============================================================
    # ВАЖНО: скачайте bottle.csv со страницы Kaggle и положите рядом со скриптом.
    CSV_PATH = "bottle.csv"

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df_binary = df[['Salnty', 'T_degC']].copy()
    df_binary.columns = ['Sal', 'Temp']

    print("Первые 5 строк данных:")
    print(df_binary.head())
    print()

    # ============================================================
    # Шаг 2: Визуализация исходных данных
    # ============================================================
    sns.lmplot(x="Sal", y="Temp", data=df_binary, order=2, ci=None)
    plt.title("Исходные данные: Sal vs Temp")
    plt.show()

    # ============================================================
    # Шаг 3: Очистка данных
    # ============================================================
    df_binary.ffill(inplace=True)
    df_binary.dropna(inplace=True)

    # ============================================================
    # Шаг 4: Обучение модели на полном датасете
    # ============================================================
    X = np.array(df_binary['Sal']).reshape(-1, 1)
    y = np.array(df_binary['Temp']).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print(f"R^2 на полном датасете: {regr.score(X_test, y_test):.4f}")

    y_pred = regr.predict(X_test)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='b', label='Тестовые данные')
    plt.plot(X_test, y_pred, color='k', label='Предсказание')
    plt.xlabel("Sal")
    plt.ylabel("Temp")
    plt.title("Линейная регрессия на полном датасете")
    plt.legend()
    plt.show()

    # ============================================================
    # Шаг 5: Работа с уменьшенным датасетом (первые 500 строк)
    # ============================================================
    df_binary500 = df_binary.iloc[:500].copy()

    sns.lmplot(x="Sal", y="Temp", data=df_binary500, order=2, ci=None)
    plt.title("Первые 500 строк: Sal vs Temp")
    plt.show()

    df_binary500.ffill(inplace=True)
    df_binary500.dropna(inplace=True)

    X = np.array(df_binary500['Sal']).reshape(-1, 1)
    y = np.array(df_binary500['Temp']).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print(f"R^2 на 500 строках: {regr.score(X_test, y_test):.4f}")

    y_pred = regr.predict(X_test)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='b', label='Тестовые данные')
    plt.plot(X_test, y_pred, color='k', label='Предсказание')
    plt.xlabel("Sal")
    plt.ylabel("Temp")
    plt.title("Линейная регрессия на первых 500 строках")
    plt.legend()
    plt.show()

    # ============================================================
    # Шаг 6: Метрики оценки регрессии
    # ============================================================
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

    print("\nМетрики оценки модели (на 500 строках):")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()