![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)

# 🗑️ Garbage Classification

Классификация мусора на 10 категорий с помощью кастомного CNN.  
Проект находится в активной разработке - текущая версия является baseline.

## Результаты

| Модель | Accuracy | Тип |
|--------|----------|-----|
| Logistic Regression | 34.96% | Классический ML |
| SVM (RBF) | 57.15% | Классический ML |
| Random Forest | 58.21% | Классический ML |
| **CNN v3 (BatchNorm + ReduceLROnPlateau)** | **63.90%** | **Deep Learning** |

## Датасет

[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) — ~9800 изображений, 10 классов:
`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `white-glass`

Сплит: **80% train / 10% val / 10% test**, стратифицированный по классам.

## Архитектура CNN v3

```
Input (224×224×3)
    → Augmentation (Flip, Rotation, Zoom, Translation, Contrast, Brightness)
    → Conv(32)  + BatchNorm + ReLU + MaxPool   # 224→112
    → Conv(64)  + BatchNorm + ReLU + MaxPool   # 112→56
    → Conv(128) + BatchNorm + ReLU + MaxPool   # 56→28
    → Conv(256) + BatchNorm + ReLU + MaxPool   # 28→14
    → Conv(512) + BatchNorm + ReLU + MaxPool   # 14→7
    → GlobalAveragePooling
    → Dense(256) + Dropout(0.4)
    → Dense(128) + Dropout(0.3)
    → Dense(10, softmax)
```

Оптимизатор: `Adam(lr=1e-3)` + `ReduceLROnPlateau(factor=0.3, patience=3)`  
Loss: `SparseCategoricalCrossentropy`



## Структура проекта

```
├── garbage_classification_last.ipynb   # основной ноутбук
├── garbage_cnn_v3_6390.keras              # сохранённая модель
├── class_names.json                  # список классов
└── README.md
```

## Limitations

**Размер датасета.** ~9800 изображений - относительно мало для задачи классификации
с 10 классами. Модель видит ~780 уникальных примеров на класс, чего недостаточно
для устойчивого обобщения на реальные условия съёмки.

**Дисбаланс классов.** Классы представлены неравномерно, что смещает модель в сторону
частых категорий. Частично компенсируется аугментацией, но не устраняется полностью.

**Контролируемые условия съёмки.** Изображения сделаны в студийных условиях -
хорошее освещение, объект по центру, чистый фон. В реальных сценариях точность
предсказуемо снизится.

**Отсутствие класса "неизвестно".** Модель всегда возвращает один из 10 классов —
даже если на входе объект вне этих категорий.

**Платформенная зависимость аугментации.** На Apple Silicon m1 с
`tensorflow-metal` слои `RandomRotation`, `RandomZoom`, `RandomTranslation` без
явного `fill_mode='reflect'` заполняют пустые пиксели константой вместо зеркального
отражения - края заливаются чёрным или белым(иногда всё изображение становится черным или белым).

**Accuracy как основная метрика.** При дисбалансе классов accuracy вводит в
заблуждение. Более честные метрики для данной задачи - macro F1-score и per-class recall.

## Roadmap

Текущая версия — **baseline**. В планах:

- [ ] **Transfer Learning** — попробовать EfficientNetB0 / MobileNetV3 как backbone,
      заморозить веса и дообучить только голову. Ожидаемый прирост: +10–15% к accuracy (Обновление: попробовал с EfficientNetB0, результат залил в репозиторий garbage-classifier-2)
- [ ] **Другие архитектуры CNN** — добавить residual connections (mini-ResNet),
      протестировать `SeparableConv2D` для снижения числа параметров
- [ ] **Улучшенная аугментация** — попробовать CutMix, MixUp, GridDistortion
      из библиотеки `albumentations`
- [ ] **Веб-интерфейс** — простое приложение на Gradio или Streamlit:
      загрузить фото мусора → получить предсказанный класс и уверенность модели (Обновление: Сделал web app с Flask, репозиторий: garbage-classifier-2)
- [ ] **Метрики** — добавить macro F1-score и per-class recall как основные метрики
      вместо accuracy
      
## Технологии

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
