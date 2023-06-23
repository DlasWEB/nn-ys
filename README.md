# nn-ys
Нейронная сеть для предсказания предела текучести стали после горячей прокатки

<p align="center">
      <img src="img/1.png" alt="Лого проекта">
</p>

## Описание

Данный репозиторий содержит нейросетевую модель на основе многослойного перцептрона. Модель состоит из 3 слоех:
1. Входной слой с 21 нейронами на входе и выходе;
2. Скрытый слой с 21 нейронами на входе и 9 нейронами на выходе. Функция активации данного слоя - relu;
3. Выходной слой с 9 нейронами на входе и 1 нейроном на выходе (предсказанное значение предела текучести). Функция активации данного слоя - selu;

<p align="center">
      <img src="img/13_0.78_r_154_schem.png" alt="Архитектура модели">
</p>

Сеть построена на массиве данных, состоящем из 21 входного параметра (16 параметров - химический состав металла, 2 параметра - технологические параметры, 3 параметра - геометрические параметры заготовки и готового изделия).
Таблица 1. Входы сети
| №  |      Параметр      | Минимальное значение | Максимальное значение | Среднее значение |
|:--:|:------------------:|:--------------------:|:---------------------:|:----------------:|
| 1  |  Содержание С, %   |        0,0390        |         0,1140        |       0,0554     |
| 2  |  Содержание Mn, %  |        1,2580        |         1,6000        |       1,4771     |
| 3  |  Содержание Si, %  |        0,1500        |         0,4470        |       0,2000     |
| 4  |  Содержание S, %   |        0,0001        |         0,0070        |       0,0018     |
| 5  |  Содержание P, %   |        0,0040        |         0,0160        |       0,0070     |
| 6  |  Содержание Cr, %  |        0,0120        |         0,4410        |       0,0284     |
| 7  |  Содержание Cu, %  |        0,0150        |         0,2730        |       0,0309     |
| 8  |  Содержание Ni, %  |        0,0120        |         0,2930        |       0,2330     |
| 9  |  Содержание V, %   |        0,0010        |         0,0380        |       0,0086     |
| 10 |  Содержание N, %   |        0,0027        |         0,0060        |       0,0038     |
| 11 |  Содержание Ti, %  |        0,0010        |         0,0200        |       0,0117     |
| 12 |  Содержание Nb, %  |        0,0220        |         0,0460        |       0,0367     |
| 13 |  Содержание Al, %  |        0,0200        |         0,0500        |       0,0410     |
| 14 |  Содержание Mo, %  |        0,0010        |         0,0350        |       0,0037     |
| 15 |  Содержание Сa, %  |        0,0001        |         0,0029        |       0,0005     |
| 16 |  Содержание H, %   |        0,00007       |         0,0005        |       0,00017    |
| 17 |  Tконц.прокат,°C   |         731          |           972         |         800      |
| 18 |  V охл пол,°C/с    |           6          |            12         |         7,64     |
| 19 |       H0/H1*       |        4,970         |         8,859         |       7,1823     |
| 20 |       h1/h0**      |        1,0320        |         2,644         |       1,6058     |
| 21 |       b1/b0***     |        0,631         |         1,984         |       1,199      |

*отношение толщины сляба к толщине листа                                    
**отношение толщины заготовки после первого прохода к толщине заготовки после второго прохода                                                   
***отношение ширины листа к ширине сляба

