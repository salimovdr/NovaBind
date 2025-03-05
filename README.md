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
Ожидается, что в директории `data` будет находиться по крайней мере архив с тестовым `rSNP` датасетом. Можно загрузить:
```bash
cp /home/fds/ibis/NovaBind/data/ibis_rSNP.zip .
```

## Reproduction
### Data preprocessing
Следующий скрипт разархивирует `data/ibis_rSNP.zip`, сконвертирует `.fasta's` в `pd.DataFrame's`, отбросит нецелевые белки, сконкатенирует для параллельных предсказаний и сохранит подготовленные сиквенсы в `test/SNP.csv`:  
```bash
python prep_data.py
```
Теперь можно нарезать сиквенсы с шагом 1 и окном 60 и one-hot-энкоднуть их:
```bash
python encode_data.py
```
В директорию `SNP_w60s1` будет сохранено 64 "бача".

### Training
Не применимо. Ожидается, что модели уже были обучены согласно пайплайну из ветки `full`. Веса уже должны лежать в `models_*`.

### Prediction
Предсказания для белков `GCM1, MKX, MSANTD1, MYPOP, SP140L, TPRX1, ZFTA` будут сделаны на основании тренированной на `PBM` модели, а для белков `ZNF831, ZNF780B, ZNF721, ZNF500, ZNF286B, ZBTB47, FIZ1, CREB3L3` – на `HTS`. При этом оценены будут все сиквенсы для каждого белка (проще отбросить избыточные предсказания). Рекомендуется запускать предсказания параллельно в двух терминальных сессиях:
```bash
python make_predict.py --device 1 --type_exp PBM
```
```bash
python make_predict.py --device 3 --type_exp HTS
```
Предсказания будут объеденены и сохранены в запрашиваемой форме (`plain text files keeping the file names the same`), а также в удобный `concated.tsv`, и заархивированы в `your_results_folder_name.zip`:
```bash
python get_results.py your_results_folder_name
```
