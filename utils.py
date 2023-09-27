from torch.utils.data import Dataset
import pandas as pd

class MakeData(Dataset):
    
    def __init__(self, path, tokenizer) -> None:
        super().__init__()

        self.data = pd.read_excel(path, names=['class','review'])
        self.reviews = self.data['review']
        self.label = self.data['class']
        idx_neu = self.label == 'neutral'
        idx_pos = self.label == 'positive'
        idx_neg = self.label == 'negative'
        self.label[idx_neu] = 1
        self.label[idx_pos] = 2
        self.label[idx_neg] = 0
        self.tokenizer = tokenizer

        self.max_length = max([len(sentence.split()) for sentence in self.reviews])

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):

        review = self.reviews[index]
        label = self.label[index]

        inputs = self.tokenizer(review, 
                                pad_to_max_length=True,
                                add_special_tokens=True, 
                                return_attention_mask=True, 
                                max_length=self.max_length,
                                return_tensors="pt")
        
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return (ids, mask, label)
        # return (label, inputs)
        