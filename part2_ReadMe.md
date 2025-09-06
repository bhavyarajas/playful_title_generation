# Generating Paper Titles from Abstracts

## Goal: fine-tune a lightweight language model so that, given an abstract as input, it generates a reasonable "playful" academic paper title. 

### Model Choice
	•	I chose T5-small, an encoder–decoder model designed for text-to-text tasks.
	•	Reason:
	    •	It is compact enough to train on a Mac with Apple Silicon.
	    •	Pretrained for tasks like summarization, translation, and text generation, so it adapts well to title generation with little fine-tuning.

### Preprocessing
	•	Tokenization:
	    •	Abstracts are tokenized to a max length of 512 tokens to preserve compute.
	    •	Titles tokenized to a max length of 48 tokens to preserve compute.
	    •	The tokenizer outputs input_ids for abstracts and labels for titles.
    
### Training Details
	•	Used Hugging Face Trainer with DataCollatorForSeq2Seq (which pads dynamically and shifts labels correctly for seq2seq tasks).
	•	Batch size = 1 - 2
	•	Epochs = 2–3 for quick fine-tuning.
	•	Learning rate = 3e-4
	•	To reduce memory, I capped abstract length so only a small subset of weights are updated.

### Evaluation
	•	Split data into train (90%) and eval (10%).
	•	During/after training, generate titles for remaining abstracts

## How to Make it Better

There are other ways I could improve the fine-tuning if I had more time and compute:
- Use stronger inputs: instead of only the abstract, I could also feed in other available columns such as the ChatGPT summary (or even concatenate abstract + conclusion). This gives the model more context to generate precise and stylistically appropriate titles.
- Train longer: increasing the number of epochs would let the model converge better. I limited training for time/compute, but with more passes over the data, the model would likely produce more consistent titles.
- Bigger effective batch size: on my laptop I had to keep batch size very small. With more memory (or gradient accumulation tuned properly), I could scale up batch size, which usually stabilizes training.
- 	Hyperparameter tuning: experiment with learning rate schedules, dropout rates, and weight decay to prevent over/underfitting.
- 	Try other models:
- 		lan-T5-small or Flan-T5-base: already tuned for instruction-following, so they might learn title style faster.
- 		RT-base: another strong seq2seq model, well-suited for summarization.
