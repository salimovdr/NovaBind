# NovaBind
Ветка `snp` содержит упрощенную версию пайплайна для оценки силы связывания `TFs` и конвертацию датасета `rSNP` в совместимый с моделью формат. Это необходимо для оценки способности нашего подхода предсказывать эффект SNP.

## Environment
Если еще не стоит окружение:
```bash
conda env create -f environment.yml
```
Активация:
```bash
conda activate Keras
```

## Input data
Ожидается, что в директории `data` будет находиться по крайней мере архив с тестовым `rSNP` датасетом. Можно загрузить и все остальное:
```bash
cp /home/fds/ibis/NovaBind/data/*.zip .
```

## Reproduction

### Data preprocessing
Следующий скрипт разархивирует `data/ibis_rSNP.zip`, конвертирует `.fasta's` в `pd.DataFrame's`, отбросит нецелевые белки, сконкатенирует для параллельных предсказаний и сохранит подготовленные сиквенсы в `test/SNP.csv`:  
```bash
python prep_data.py
```
Теперь можно нарезать сиквенсы с шагом 1 и окном 60 и one-hot-энкоднуть их:
```bash
python encode_data.py
```
В директорию `SNP_w60s1` будет сохранено 64 "бача"

### Training
Не применимо. Ожидается, что модели уже были обучены согласно пайплайну из ветки `full`. Веса уже должны лежать в `models_*`. Может пригодиться предподготовленный тренировочный сабсет:
```bash
cp /home/fds/ibis/NovaBind/folds_* .
```

### Prediction

**Step 4.** To generate predictions, you need to run the script `make_predict.py` with the argument `--type_exp` set to 'PBM' or 'HTS', which specifies based on which experiments the prediction will be made.

| Prediction | Based on         | Discipline   |
|------------|------------------|--------------|
| PBM        | PBM              | Secondary    |
| GHTS       | PBM and HTS      | Primary      |
| CHS        | PBM and HTS      | Primary      |
| HTS        | HTS              | Secondary    |

If you want to run predictions based on PBM or HTS in parallel, please specify the device number to perform the calculations on:

```bash
python make_predict.py --device 0 --type_exp PBM
python make_predict.py --device 1 --type_exp HTS
```

As a result of predictions on different models, the sum of the predictions is calculated and min-max scaling is applied. To merge the prediction results, run the script:

```bash
python get_results.py
```
