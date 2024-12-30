from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from fuzzywuzzy import fuzz
import datetime
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from typing import Dict
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from fuzzywuzzy import fuzz
import re
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import jellyfish

app = FastAPI()


SPECIAL_CHAR_DOT_REGEX = r"[.]"
SPECIAL_CHARS_REGEX = r"[-+.^:,_/\s]+" 
SALUTATION_REGEX = r"^(shree|shri|miss|smt|mrs|mr|ms|dr|master|hon|sir|madam|prof|capt|major|rev|fr|br)\s*"
PARENT_SPOUSE_NAME_REGEX = r"(?:\s*(?:s/o|d/o|w/o|so|do|wo|daughter of|son of|wife of|husband of)\s*)"
COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX = r"\b(mohammad|mohammed|muhamed|mohd|mohamed|mohamad|muhamad|muhammad|muhammed|muhammet|mohamud|mohummad|mohummed|mouhamed|muhamaad|mohammod|mouhamad|mo|md|mahmood|mahmud|ahmad|ahmed|hameed|hamid|hammed|mahd|mahmod|mohd|mouhammed|mohamad|muhmood|mohhammed|muhmamed|mohmed|mohmat|muhmat|mu|m|shaikh|mo)\b"
LAST_NAMES_AGARWAL_VARIANTS_REGEX = r"\b(aggarwal|agrawal|agarwal|aggrawal|agarwalla|agarwal)\b"


KEYWORDS = [
    "traders","trading", "enterprise", "garments", "collection", "food", "clothes", 
    "glass", "fittings", "digital", "kirana", "medical", "agency","tex",
    "security", "systems", "badges", "hospitality", "jewellers",'lic','agent',
    "ready-made", "store", "hospital", "restaurant", "auto", "center", 
    "dairy", "home", "products", "services", "furniture", "hardware", 
    "pharmacy", "stationery", "treatments", "nutrition", "wellness", 
    "sweets", "resort", "kitchen", "clothing", "market",'workshop','agency','consumer','amale' 
    "poultry", "seeds", "pesticides", "sales", "cafe", "clinic", 'project'
    "supermart", "distributors", "automobiles", "electricity", 
    "electronics", "general", "provision", "fertilizers", "agriculture", 
    "beverages", "textiles", "plumbing", "supplies", "handicrafts", 
    "construction", "medical", "bakery", "tissue", "cleaning", 
    "appliances", "homecare", "kitchenware", "decor", "glass and fittings",
    "interiors", "shopping", "crafts", "tools", "wholesale", 
    "retail", "outlet", "merchants", "trade", "distribution", 
    "solutions", "innovation", "consultancy", "services", "equipment", 
    "manufacturing", "exports", "imports", "packaging", "network", 
    "consultants", "transport", "moving", "storage", "logistics", 
    "construction", "real estate",'distributor','wines','hardware',
    'plywood','company','craft','soda','station','mobile'
    "brokerage", "management", 'handloom','co.','tvs',
    "finance", "investment", "funding", "support", "technology", 
    "software", "applications", "digital marketing", "advertising", 
    "communication", "entertainment", "events", "tourism", "travel", 
    "transportation", "automotive", "services", "supply chain", 
    "fashion", "cosmetics", "beauty", "spa", "wellness", "glass and fitting",
    "personal care", "gifts", "custom", "specialty", "craftsmanship", 
    "fashions", "motors", "enterprises", "garment", "cloth centre", "mart", 
    "foods", "silk and readymade", "wool centre", "jewellery", "mill", 
    "farms", "farm", "electrical", "egg centre", "centre", 
    "vegetable and fruits", "vegetables", "fruits", "pvt", "pvt ltd", 
    "limited", "solutions", "energies", "photo", "studio", "works", 
    "associates", "medico", "agencies", "diagnosis", "cool drinks", 
    "drinks", "care", "liquor", "automobiles", "materials", "diagnostics", 
    "provision", "trader", "farms", "farm", "stations", "restaurant", 
    "creations", "travels", "hardware", "printers", "graphics", 
    "fertilisers", "house", "studio", "private", "appliances", "steels", 
    "shop", "metals", "international", "jwellers", "corporation", 
    "dresses", "industries", "electricals", "company", "lim", "colddrinks", 
    "electron", "medicines", "llc", "computers", "hotel", "spa", 
    "cosmetics", "telecom", "sarees", "petroleums", "bhandar",'store','stores', 
    "surgical", "wines", "constructions", "shoppy", "lab", "builders", 
    "footwear", "wear", "shoe", "repair", "ventures", "paint", "depot",'cake','chinies',
    "tent", "decorators", "communications", "pharmacy", "products",'textile','CERAMIC','Pharmaceuticals','stores','sons',
]

# Loading the model fine tuned on the custom dataset over here
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class BertForSTS(torch.nn.Module):

    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output

PATH = 'bert-sts.pt'
model = BertForSTS()
model.load_state_dict(torch.load(PATH))
model.eval()



# Loading hugging face embedding models over here
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def convert_to_lower(name):
    return name.lower()

def replace_adjacent_duplicates(value):
    if isinstance(value, str):
        return re.sub(r'(.)\1+', r'\1', value)
    return value

def replace_characters(name):
    replacements = {'e': 'i', 'j': 'z', 'v': 'w', 'q': 'k'}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def replace_bigrams(name):
    replacements = {'ph': 'f', 'gh': 'g', 'th': 't', 'kh': 'k', 'dh': 'd', 'ch': 'c', 'sh': 's', 'au': 'o',
                    'bh': 'b', 'ks': 'x', 'ck': 'k', 'ah': 'h', 'wh': 'w', 'wr': 'r'}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def remove_extra_spaces(name):
    return re.sub(r'\s+', ' ', name).strip()

def remove_consonant_a(name):
    consonants = 'bcdfghjklmnpqrstvwxyz'
    new_name = ''.join([name[i] for i in range(len(name)) if not (i > 0 and name[i] == 'a' and name[i - 1].lower() in consonants)])
    return new_name

def remove_special_characters(text):
    text = re.sub(SPECIAL_CHAR_DOT_REGEX, '', text)
    text = re.sub(SPECIAL_CHARS_REGEX, '', text)
    return text.strip()

def remove_salutations(text):
    return re.sub(SALUTATION_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_parent_spouse_name(text):
    return re.sub(r'\s*(?:s/o|d/o|w/o|so|do|wo|daughter of|son of|wife of|husband of|daughter|son|child of)\s*[\w\s,.]*$', '', text, flags=re.IGNORECASE).strip()

def remove_common_muslim_variations(text):
    return re.sub(COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_agarwal_variants(text):
    return re.sub(LAST_NAMES_AGARWAL_VARIANTS_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_stop_words(text):
    stop_words = ['devi', 'dei', 'debi','singh','kumar', 'kmr', 'kumr','bhai', 'bhau', 'bai', 'ben', 'kaur', 'Md', 'Mohd', 'Mohammad', 'Mohamad','alam','shekh','sek']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)



def preprocess_layer1(name):
    name = convert_to_lower(name)
    name = remove_salutations(name)
    name = remove_stop_words(name)
    name = remove_extra_spaces(name)
    return name

def check_keywords_layer1(name1, name2):
    found_in_name1 = any(keyword in name1.lower() for keyword in KEYWORDS)
    found_in_name2 = any(keyword in name2.lower() for keyword in KEYWORDS)
    
    if found_in_name1 and found_in_name2:
        return 1  
    elif found_in_name1 or found_in_name2:
        return 0 
    return 1 

# def preprocess_layer1_PNC(name):
#     name = convert_to_lower(name)
#     name = name.strip() 
#     name = " ".join(name.split()) 
#     name = remove_stop_words(name)
#     name = remove_salutations(name)
#     return name

# def check_permutation_match(name1, name2):
#     name1 = preprocess_layer1_PNC(name1)
#     name2 = preprocess_layer1_PNC(name2)
    
#     words1 = set(name1.split())
#     words2 = set(name2.split())
    
#     return words1 == words2

def calculate_fuzzy_similarity_layer1(name1, name2):
    name1 = preprocess_layer1(name1)
    name2 = preprocess_layer1(name2)
    
    fuzzy_ratio = fuzz.ratio(name1, name2) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0

    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
    return fuzzy_similarity

# def check_substring_match(name1, name2):
#     name1 = preprocess_layer1(name1)
#     name2 = preprocess_layer1(name2)
#     words1 = set(name1.lower().split())
#     words2 = set(name2.lower().split())
    
#     if words1.issubset(words2) or words2.issubset(words1):
#         return True
#     return False


def fuzzy_layer1(row):
    name1 = row['name1']
    name2 = row['name2']
    
#     if check_substring_match(name1, name2):
#         keyword_flag = check_keywords_layer1(name1, name2)
#         return 1 if keyword_flag == 1 else 0  
    
#     if check_permutation_match(name1, name2):
#         keyword_flag = check_keywords_layer1(name1, name2)
#         return 1 if keyword_flag == 1 else 0  
    
    fuzzy_SS = calculate_fuzzy_similarity_layer1(name1, name2)
    fuzzy_flag = fuzzy_SS >= 0.80  

    if fuzzy_flag:
        keyword_flag = check_keywords_layer1(name1, name2)
        return 1 if keyword_flag == 1 else 0  

    return fuzzy_SS




def preprocess(name):
    name = remove_salutations(name)
    name = remove_parent_spouse_name(name)
    name = remove_common_muslim_variations(name)
    name = remove_agarwal_variants(name)
    name = convert_to_lower(name)
    name = replace_adjacent_duplicates(name)
    name = replace_characters(name)
    name = replace_bigrams(name)
    name = remove_consonant_a(name)
    name = remove_special_characters(name)
    name = remove_extra_spaces(name)
    name = remove_stop_words(name)
    return name

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2).item()

def calculate_levenshtein_similarity(name1, name2):
    lev_distance = levenshtein_distance(name1, name2)
    max_len = max(len(name1), len(name2))
    return (max_len - lev_distance) / max_len if max_len > 0 else 1.0

def calculate_phonetic_similarity(name1, name2):
    soundex1 = jellyfish.soundex(name1)
    soundex2 = jellyfish.soundex(name2)
    return jellyfish.jaro_winkler_similarity(soundex1, soundex2)

def calculate_jaccard_similarity(name1, name2):
    set1, set2 = set(name1), set(name2)
    intersection, union = set1.intersection(set2), set1.union(set2)
    return len(intersection) / len(union) if union else 1.0

def predict_similarity_embedding_model_finetuned(sentence_pair):
    """
    Predict similarity between a pair of sentences
    """
    test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']
    output = model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
    return sim

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")




def name_match(name1, name2):
    name1_processed = preprocess(name1)
    name2_processed = preprocess(name2)
    
    # uncomment the following statement in the case of hugging face embeddings models over here
    # embedding_similarity = predict_similarity_embedding_model_huggingface([name1_processed, name2_processed])
    
    
    # uncomment the following statement in the case of model fine tuned on custom dataset over here
    embedding_similarity = predict_similarity_embedding_model_finetuned([name1_processed, name2_processed])
    

    levenshtein_similarity = calculate_levenshtein_similarity(name1_processed, name2_processed)
    phonetic_similarity = calculate_phonetic_similarity(name1_processed, name2_processed)
    jaccard_similarity = calculate_jaccard_similarity(name1_processed, name2_processed)

    fuzzy_ratio = fuzz.ratio(name1_processed, name2_processed) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1_processed, name2_processed) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1_processed, name2_processed) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1_processed, name2_processed) / 100.0

    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0

    final_score = (
        embedding_similarity * 0.121630 +
        levenshtein_similarity * 0.205104 +
        phonetic_similarity * 0.022287 +
        jaccard_similarity * 0.262775 +
        fuzzy_similarity * 0.388205
    )
    return final_score 

def check_keywords_layer2(name1, name2):
    found_in_name1 = any(keyword in name1.lower() for keyword in KEYWORDS)
    found_in_name2 = any(keyword in name2.lower() for keyword in KEYWORDS)
    
    if found_in_name1 and found_in_name2:
        return 1  
    elif found_in_name1 or found_in_name2:
        return 0  
    return 1 

def preprocess_layer3(name):
    name = convert_to_lower(name)
    name = remove_stop_words(name)
    name = remove_common_muslim_variations(name)
    name = remove_agarwal_variants(name)
    return name

def calculate_fuzzy_similarity_layer3(name1, name2):
    name1 = preprocess_layer3(name1)
    name2 = preprocess_layer3(name2)
    
    fuzzy_ratio = fuzz.ratio(name1, name2) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0
    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
    return fuzzy_similarity

def check_keywords_layer3(name1, name2):
    found_in_name1 = any(keyword in name1.lower() for keyword in KEYWORDS)
    found_in_name2 = any(keyword in name2.lower() for keyword in KEYWORDS)
    
    if found_in_name1 and found_in_name2:
        return 1  
    elif found_in_name1 or found_in_name2:
        return 0  
    return 1  

def Fuzzy_Wuzzy_layer3(row):
    name1 = row['name1']
    name2 = row['name2']
    fuzzy_SS = calculate_fuzzy_similarity_layer3(name1, name2)
    fuzzy_flag = fuzzy_SS >= 0.75  
    
    if fuzzy_flag:
        fuzzy_flag = check_keywords_layer3(name1, name2)
        
    return fuzzy_SS

def process_name_matching_from_names(name1, name2):
    threshold = 0.80
    threshold1 = 0.65

    data = {'name1': [name1], 'name2': [name2]}
    df = pd.DataFrame(data)
    
    df["Prediction"] = 0 
    
#     df["First_Layer_Pass"] = df.apply(fuzzy_layer1, axis=1)
#     df["Prediction"] = df.apply(lambda x: 1 if x["First_Layer_Pass"] else x["Prediction"], axis=1)
    
    df["First_Layer_Score"] = df.apply(lambda x: calculate_fuzzy_similarity_layer1(x['name1'], x['name2']), axis=1)
    df["First_Layer_Pass"] = df["First_Layer_Score"] >= threshold
    
    if df["First_Layer_Pass"].iloc[0]:
        df["Prediction"] = 1
        return df

    df["Second_Layer_Score"] = df.apply(lambda x: name_match(x['name1'], x['name2']), axis=1)
    df["Second_Layer_Pass"] = df["Second_Layer_Score"] >= threshold1

    if df["Second_Layer_Pass"].iloc[0]:
        df["Prediction"] = 1
        return df

    df["Third_Layer_Pass"] = df.apply(Fuzzy_Wuzzy_layer3, axis=1)
    df["Prediction"] = 1 if df["Third_Layer_Pass"].iloc[0] else 0

    return df


@app.get("/match-names/")
async def match_names(name1: str, name2: str) -> Dict:
    df_result = process_name_matching_from_names(name1, name2)
    result = df_result.iloc[0].to_dict()
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, port=9000)