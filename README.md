## pytonh version
```
default=3.7
```
 
## Environment
```shell
pip install -r requirements.txt
```

## Download glove 300d
```shell
bash preprocess.sh
```

## Download model
```shell
bash download.sh
```
## preprocess+train
```shell
python intent_preprocess_train.py --train_file "${1}"  --eval_file "${2}"
python slot_preprocess_train.py --train_file "${1}"  --eval_file "${2}"
```

## predict
```shell
python intent.py --test_file "${1}"  --pred_file "${2}"
python slot.py --test_file "${1}"  --pred_file "${2}"
```