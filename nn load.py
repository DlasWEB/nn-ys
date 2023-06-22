import pandas as pd
import tensorflow as tf

# Загружаем ранее сохраненую модель
model_loaded_y2 = tf.keras.models.load_model('/full/path/to/dir/y2_model')

# Импортируем данные для передачи в сеть, можно не использовать если хотите вводить все значения сами
x_excel = pd.read_excel('/full/path/to/dir/with/X_for_test.xlsx')

# Передача значений в модель в скрипте
t_end_roll = 800
t_end_roll_min = 731
t_end_roll_max = 972
t_end_roll_norm = (t_end_roll - t_end_roll_min) / (t_end_roll_max - t_end_roll_min)

v_cool = 29
v_cool_min = 6
v_cool_max = 12
v_cool_norm = (v_cool - v_cool_min) / (v_cool_max - v_cool_min)

ti = 0.0117 * 0.5
ti_min = 0.0010
ti_max = 0.0200
ti_norm = (ti - ti_min) / (ti_max - ti_min)

nb = 0.0367 * 0.5
nb_min = 0.0010
nb_max = 0.0200
nb_norm = (nb - nb_min) / (nb_max - nb_min)

# Формируем выборку x-ов на основе исходных данных (все берем как среднее по столбцу, кроме определенных параметров)
x = [[x_excel['т сляб/т лист'].mean(), x_excel['ш лист/ш сляб'].mean(), x_excel['т пос 1 п/т пос 2 п'].mean(),
# Т к прок °С, Ск ох град/c
      t_end_roll_norm, v_cool_norm,
      x_excel['C %'].mean(), x_excel['MN %'].mean(), x_excel['SI %'].mean(), x_excel['S %'].mean(), x_excel['P %'].mean(),
      x_excel['CR %'].mean(), x_excel['CU %'].mean(), x_excel['NI %'].mean(), x_excel['V %'].mean(), x_excel['N %'].mean(),
        # TI %, NB %
      ti_norm, nb_norm,
      x_excel['AL %'].mean(), x_excel['MO %'].mean(), x_excel['CA %'].mean(), x_excel['H %'].mean()]]

# Скармливаем в модель Х-ы и получаем предсказанное значение
y2_norm = model_loaded_y2.predict(x)

# Переводим предсказанное значение из нормализованного вида в исходный

y2_max = 554
y2_min = 391
y2 = y2_norm * y2_max - y2_norm * y2_min + y2_min

print('Предел текучести = ' + f'{y2}' + ' МПа')