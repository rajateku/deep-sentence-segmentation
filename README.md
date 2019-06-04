Neural Network Based Sentence Segmentation 
=======

## Usage

```
python : 3.6
git clone https://github.com/rajateku/deep-sentence-segmentation.git
cd deep-sentence-segmentation
pip install -r requirements.txt

Training the model : 
Step 1: Generate the data
python data/data_gen.py

Step 2: Change Hyperparameters
File here : models/config.py

Step 2: Start Training
python  models/trainer.py

```

Problem Definition

```
Sentence segmentation or Sentence boundary Detection 
is breaking the text into its component sentences.
The above model breaks the sentences even it doesn't have and puctuations such 
as periods or question marks.
```
