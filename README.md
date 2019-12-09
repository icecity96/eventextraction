# EventExtraction

Extract Service-related Event From Social Media News. 

## Usage
### Preparation
1. Clone this repository
```bash
git clone --recursive http://192.168.1.104:12345/serviceecosystem/eventextraction.git
```
2. download albert pre-trained model
```bash
wget https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip
```
**Note: This is albert tiny. For more pre-trained albert model, please visit https://github.com/brightmart/albert_zh**
### Format your file
```bash
python src/preprocess.py
```
After this you need to split your data into train and test(dev) set.

### Train joint learning model
```bash
cd src/albert_zh
python joint_learning.py --do_train true \
    --data_dir your/path/to/data/directory/ \
    --vocab_file your/path/to/vocab.txt  \
    --bert_config_file your/path/to/albert_config_tiny.json \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --init_checkpoint your/path/to/albert_model.ckpt \
    --output_dir your/path/to/output_dir/
```
Normally you do not need modify the `max_predictions_per_seq`, `train_batch_size`, `learning_rate`.

### Using the model to do prediction or evaluation
**Do Predict**
```bash
cd src/albert_zh
python joint_learning.py --do_predict true \
    --data_dir your/path/to/data_dir/ \
    --vocab_file your/path/vocab.txt \
    --bert_config_file your/path/to/albert_config_tiny.json \
    --max_seq_length 128 \ 
    --output_dir your/path/to/output_dir/ \
    --predict_batch_size 32
```
If you need to do evaluation just change `--do_predict` to `--do_eval`.

## How to Use Other Pre-training models (BERT, RoBERT)?:
I use the albert as pre-trained model in my joint learning model. Because ALBERT is much smaller than BERT. 
So that it much faster to train/predict/eval.

**This code can be easily modified to suit BERT or RoBERT**. Only need to change some import in `src/albert_zh/joint_learning.py`.

## What should I do if I only want to do NER/Text-Pair-Classification?
You can modify the souce code. In `src/albert_zh/joint_learning.py`, you can find
```python
total_loss = task1_loss + task2_loss
```
where `task1_loss` is the NER job loss and `task2_loss` is the Text-Pair-Classification loss. You can modify the loss weight to 
control which job is more important. 

Specifically, when you set `task1_loss` weight to `0`, then this only do text pair classification task.

## Can I use this model to do single sentence classification?
Of course you can.

What you need to do is change `get_labelrs`. and make sure your each of your input sentence doesn't contain `,`.

## How can I use custom NER-tag and sentence-pair-label?
You can modify `get_labeles` and `get_labelrs` respectively.
