The codes for the LREC-COLING paper **"Model-Agnostic Cross-Lingual Training for Discourse
Representation Structure Parsing"** and the LREC-COLING paper **"Soft Well-Formed Semantic Parsing with Score-Based Selection"**

Clone the repository and obtain the directory `DRS-Cross-Lingual-Training`, and enter the directory.

## Data

Download the data pmb-3.0.0.zip and pmb-4.0.0.zip from [here](https://pmb.let.rug.nl/releases), and unzip them to obtain `DRS-Cross-Lingual-Training/pmb-3.0.0` and `DRS-Cross-Lingual-Training/pmb-4.0.0`

Enter the data folder, generate the train/dev/test data according to instructions in the data folder.

## Metric

Clone the DRS_parsing from [DRS_parsing](https://github.com/RikVN/DRS_parsing) and the [ud-boxer](https://github.com/WPoelman/ud-boxer).

Download and install mtool.

## Cross-Training

```
mkdir workspace_4.0.0_word_all #DRS-Cross-Lingual-Training/workspace_4.0.0_word_all
cd workspace_4.0.0_word_all
```

### data

```
mkdir data
cd data
cp ../../src/pretrain_data_bash.sh .
bash pretrain_data_bash.sh
cd ..
```

### training

For clause
```
mkdir mt0 
ln -s ../../src/train.py
ln -s ../data
python train.py --train_prefix data/pretrain\
                --dev_prefix data/dev \
                --model_name_or_path bigscience/mt0-large \
                --save_model_name mt0-large-pretrain \
                --rank 32 \
                --lr 1e-3 \
                --num_proc 1 \
                --num_epoches 20
cd ..
```
For graph
```
mkdir mt0-sbn
ln -s ../../src/train.py
ln -s ../data
python train.py --train_prefix data/pretrain-sbn\
                --dev_prefix data/dev-sbn \
                --model_name_or_path bigscience/mt0-large \
                --save_model_name mt0-large-pretrain \
                --rank 32 \
                --lr 1e-3 \
                --num_proc 1 \
                --num_epoches 20
cd ..
```

## Inference

Take German as an example

```
mkdir workspace_4.0.0_word_de #DRS-Cross-Lingual-Training/workspace_4.0.0_word_de
cd workspace_4.0.0_word_de
```

### data

```
mkdir data
cd data
cp ../../src/data_bash.sh .
bash data_bash.sh de
cd ..
```
### inference

For clause
```
mkdir mt0
ln -s ../../src/inference.py
ln -s ../data
python inference.py --input_file data/test.src \
                    --output_file test.pred \
                    --peft_model_id model_path
cd ..
```
For graph
```
mkdir mt0
ln -s ../../src/inference.py
ln -s ../data
python inference.py --input_file data/test-sbn.src \
                    --output_file test-sbn.pred \
                    --peft_model_id model_path
cd ..
```

## Evaluation

For clause
```
ln -s /the/path/pmb-4.0.0/src/clf_signature.yaml
ln -s /the/path/DRS_parsing/evaluation/counter.py
ln -s /the/path/Neural_DRS/src/postprocess.py
python postprocess.py -i test.pred -o test.post -ns
python counter.py -f1 test.post -f2 data/test.txt -g clf_signature.yaml -ill dummy -r 5 -p 1
```

For graph
```
ln -s DRS-Cross-Lingual-Training/src/sbn2penman.py
python sbn2penman.py -i test-sbn.pred -o test-sbn.post
mtool --read amr --score smatch --gold test.penman test-sbn.post
```

## Fine-tuning

Add the cross-lingual training model checkpoints to the LORA.




