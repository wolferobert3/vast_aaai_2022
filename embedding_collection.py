from nltk.tag import pos_tag
import numpy as np
import pandas as pd
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from string import punctuation
from os import path, listdir
import pickle
import copy
from helper_functions import get_embeddings
from nltk import word_tokenize

#Model
MODEL_ID_GPT2 = 'gpt2'
MODEL_GPT2 = TFGPT2LMHeadModel.from_pretrained(MODEL_ID_GPT2, output_hidden_states = True, output_attentions = False)
MODEL_TOKENIZER_GPT2 = GPT2Tokenizer.from_pretrained(MODEL_ID_GPT2)

#Embedding Property Lists
pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))
dominant = sorted(list(set('power,command,control,master,rule,authority,strong,superior,dominant,confident,leader,king,victory,mighty,bravery,triumph,win,success,fame,glory,respect,honor,champion,advantage,capable'.split(','))))
submissive = sorted(list(set('subordinate,weak,disadvantage,helpless,insecure,failure,lonely,humiliate,coward,feeble,inferior,embarrassed,victim,afraid,timid,shame,defeat,panic,disappointment,impotence,shy,nervous,meek,fearful,distressed'.split(','))))
arousal = sorted(list(set('thrill,excitement,desire,sex,ecstasy,erotic,passion,infatuation,lust,flirt,murder,rage,assault,danger,terror,fight,scream,violent,startled,alert,anger,laughter,surprise,intruder,aroused'.split(','))))
indifference = sorted(list(set('relaxed,sleep,quiet,bored,subdued,peace,indifferent,secure,gentle,cozy,bland,reserved,slow,plain,solemn,polite,tired,weary,safe,comfort,protected,dull,soothing,leisure,placid'.split(','))))

#WEAT Names
ea_name_male = sorted(list(set('Adam,Harry,Josh,Roger,Alan,Frank,Justin,Ryan,Andrew,Jack,Matthew,Stephen,Brad,Greg,Paul,Jonathan,Peter,Brad,Brendan,Geoffrey,Greg,Brett,Matthew,Neil,Todd'.split(','))))
ea_name_female = sorted(list(set('Amanda,Courtney,Heather,Melanie,Katie,Betsy,Kristin,Nancy,Stephanie,Ellen,Lauren,Colleen,Emily,Megan,Rachel,Allison,Anne,Carrie,Emily,Jill,Laurie,Meredith,Sarah'.split(','))))
aa_name_male = sorted(list(set('Alonzo,Jamel,Theo,Alphonse,Jerome,Leroy,Torrance,Darnell,Lamar,Lionel,Tyree,Deion,Lamont,Malik,Terrence,Tyrone,Lavon,Marcellus,Wardell,Darnell,Hakim,Jermaine,Kareem,Jamal,Leroy,Rasheed,Tyrone'.split(','))))
aa_name_female = sorted(list(set('Nichelle,Shereen,Ebony,Latisha,Shaniqua,Jasmine,Tanisha,Tia,Lakisha,Latoya,Yolanda,Malika,Yvette,Aisha,Ebony,Keisha,Kenya,Lakisha,Latoya,Tamika,Tanisha'.split(','))))

#Full WEAT
pleasant = ['caress','freedom','health','love','peace','cheer','friend','heaven','loyal','pleasure','diamond','gentle','honest','lucky','rainbow','diploma','gift','honor','miracle','sunrise','family','happy','laughter','paradise','vacation']
unpleasant = ['abuse','crash','filth','murder','sickness','accident','death','grief','poison','stink','assault','disaster','hatred','pollute','tragedy','divorce','jail','poverty','ugly','cancer','kill','rotten','vomit','agony','prison']
flower = ['aster','clover','hyacinth','marigold','poppy','azalea','crocus','iris','orchid','rose','bluebell','daffodil','lilac','pansy','tulip','buttercup','daisy','lily','peony','violet','carnation','gladiola','magnolia','petunia','zinnia']
insect = ['ant','caterpillar','flea','locust','spider','bedbug','centipede','fly','maggot','tarantula','bee','cockroach','gnat','mosquito','termite','beetle','cricket','hornet','moth','wasp','blackfly','dragonfly','horsefly','roach','weevil']
instrument = ['bagpipe','cello','guitar','lute','trombone','banjo','clarinet','harmonica','mandolin','trumpet','bassoon','drum','harp','oboe','tuba','bell','fiddle','harpsichord','piano','viola','bongo','flute','horn','saxophone','violin']
weapon = ['arrow','club','gun','missile','spear','axe','dagger','harpoon','pistol','sword','blade','dynamite','hatchet','rifle','tank','bomb','firearm','knife','shotgun','teargas','cannon','grenade','mace','slingshot','whip']
ea_name = ['Adam','Harry','Josh','Roger','Alan','Frank','Justin','Ryan','Andrew','Jack','Matthew','Stephen','Brad','Greg','Paul','Jonathan','Peter','Amanda','Courtney','Heather','Melanie','Katie','Betsy','Kristin','Nancy','Stephanie','Ellen','Lauren','Colleen','Emily','Megan','Rachel']
aa_name = ['Alonzo','Jamel','Theo','Alphonse','Jerome','Leroy','Torrance','Darnell','Lamar','Lionel','Tyree','Deion','Lamont','Malik','Terrence','Tyrone','Lavon','Marcellus','Wardell','Nichelle','Shereen','Ebony','Latisha','Shaniqua','Jasmine','Tanisha','Tia','Lakisha','Latoya','Yolanda','Malika','Yvette']
ea_name_2 = ['Brad','Brendan','Geoffrey','Greg','Brett','Matthew','Neil','Todd','Allison','Anne','Carrie','Emily','Jill','Laurie','Meredith','Sarah']
aa_name_2 = ['Darnell','Hakim','Jermaine','Kareem','Jamal','Leroy','Rasheed','Tyrone','Aisha','Ebony','Keisha','Kenya','Lakisha','Latoya','Tamika','Tanisha']
pleasant_2 = ['joy','love','peace','wonderful','pleasure','friend','laughter','happy']
unpleasant_2 = ['agony','terrible','horrible','nasty','evil','war','awful','failure']
career = ['executive','management','professional','corporation','salary','office','business','career']
domestic = ['home','parents','children','family','cousins','marriage','wedding','relatives']
male_name = ['John','Paul','Mike','Kevin','Steve','Greg','Jeff','Bill']
female_name = ['Amy','Joan','Lisa','Sarah','Diana','Kate','Ann','Donna']
male = ['male','man','boy','brother','he','him','his','son']
female = ['female','woman','girl','sister','she','her','hers','daughter']
mathematics = ['math','algebra','geometry','calculus','equations','computation','numbers','addition']
art = ['poetry','art','dance','literature','novel','symphony','drama','sculpture']
male_2 = ['brother','father','uncle','grandfather','son','he','his','him']
female_2 = ['sister','mother','aunt','grandmother','daughter','she','hers','her']
science = ['science','technology','physics','chemistry','Einstein','NASA','experiment','astronomy']
art_2 = ['poetry','art','Shakespeare','dance','literature','novel','symphony','drama']
temporary = ['impermanent','unstable','variable','fleeting','short-term','brief','occasional']
permanent = ['stable','always','constant','persistent','chronic','prolonged','forever']
mental = ['sad','hopeless','gloomy','tearful','miserable','depressed']
physical = ['sick','illness','influenza','disease','virus','cancer']
young = ['Tiffany','Michelle','Cindy','Kristy','Brad','Eric','Joey','Billy']
old = ['Ethel','Bernice','Gertrude','Agnes','Cecil','Wilbert','Mortimer','Edgar']

#Scripting Area

weat_terms = list(set(flower + insect + instrument + weapon + ea_name + aa_name + ea_name_2 + aa_name_2 + pleasant + unpleasant + pleasant_2 + unpleasant_2 + young + old + male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2))
pleasant_weat = list(set(flower + instrument + ea_name + ea_name_2 + pleasant + pleasant_2 + young))
unpleasant_weat = list(set(insect + weapon + aa_name + aa_name_2 + unpleasant + unpleasant_2 + old))
neutral_weat = list(set(male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2))

CURRENT_MODEL = MODEL_GPT2
CURRENT_TOKENIZER = MODEL_TOKENIZER_GPT2
WRITE_MODEL = 'gpt2'
TEMPLATE = 'This is WORD'
DUMP_PATH = f'D:\\cwe_dictionaries'
TENSOR_TYPE = 'tf'

#Load in lexica
bellezza = pd.read_csv('Bellezza_Lexicon.csv')
bellezza_terms = bellezza['word'].to_list()
bellezza_valence = bellezza['combined_pleasantness'].to_list()
bellezza_valence_dict = {bellezza_terms[idx]: bellezza_valence[idx] for idx in range(len(bellezza_terms))}

anew = pd.read_csv('ANEW.csv')
anew_terms = anew['Description'].to_list()
anew_valence = anew['Valence Mean'].to_list()
anew_dominance = anew['Dominance Mean'].to_list()
anew_arousal = anew['Arousal Mean'].to_list()
anew_sd_valence = anew['Valence SD'].to_list()
anew_sd_dominance = anew['Dominance SD'].to_list()
anew_sd_arousal = anew['Arousal SD'].to_list()
anew_valence_dict = {anew_terms[idx]: anew_valence[idx] for idx in range(len(anew_terms))}

warriner = pd.read_csv('Warriner_Lexicon.csv')
warriner_terms = warriner['Word'].to_list()
warriner_terms[8289] = 'null'
warriner_valence = warriner['V.Mean.Sum'].to_list()
warriner_dominance = warriner['D.Mean.Sum'].to_list()
warriner_arousal = warriner['A.Mean.Sum'].to_list()
warriner_sd_valence = warriner['V.SD.Sum'].to_list()
warriner_sd_dominance = warriner['D.SD.Sum'].to_list()
warriner_sd_arousal = warriner['A.SD.Sum'].to_list()
warriner_valence_dict = {warriner_terms[idx]: warriner_valence[idx] for idx in range(len(warriner_terms))}

term_list = list(set(bellezza_terms + anew_terms + warriner_terms + weat_terms + arousal + dominant + indifference + submissive))
missing = list(term_list)
context_dict = {}

dir_ = f'D:\\new_contexts'
dir_list = list(listdir(dir_))

for target_file in dir_list:
    print(target_file)
    with open(path.join(dir_,target_file), 'rb') as pkl_reader:
        contexts = pickle.load(pkl_reader)[0]
    for context in contexts:
        if type(context) != str:
            continue
        pop_idx = []
        for idx, term in enumerate(missing):
            if term in context:
                print(context)
                try:
                    tokenized_context = word_tokenize(context)
                    tokenized_term = word_tokenize(term)
                except:
                    continue
                try:
                    pos = tokenized_context.index(term[0])
                    start = max(0, pos - 10)
                    end = min(len(tokenized_context), pos + 10)
                    context_dict[term] = ' '.join(tokenized_context[start:end])
                    pop_idx.append(idx)
                    print(term)
                    with open(f'D:\\cwe_dictionaries\\updated_random_context_dictionary.pkl', 'wb') as pkl_writer:
                        pickle.dump(context_dict, pkl_writer)
                    continue
                except:
                    continue
        missing = [missing[i] for i in range(len(missing)) if i not in pop_idx]
        print(len(missing))
        if not missing:
            break
    if not missing:
        break

print(missing)

with open(f'D:\\cwe_dictionaries\\random_context_dictionary.pkl', 'wb') as pkl_writer:
    pickle.dump(context_dict, pkl_writer)


SETTING = 'aligned'
#Set valence for aligned contexts
if SETTING == 'aligned' or SETTING == 'misaligned':

    term_valence_dict = copy.deepcopy(warriner_valence_dict)

    for idx, term in enumerate(anew_terms):
        if term not in term_valence_dict:
            term_valence_dict[term] = anew_valence[idx]

    #Rescale Bellezza for consistency with other lexica
    for idx, term in enumerate(bellezza_terms):
        if term not in term_valence_dict:
            term_valence_dict[term] = ((bellezza_valence[idx] - 1) * 2) + 1

    for term in pleasant_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 8.0

    for term in unpleasant_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 2.0

    for term in neutral_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 5.0

    term_class_dict = {}

    for term, valence in term_valence_dict.items():
        if valence <= 2.5:
            term_class_dict[term] = 0
        elif valence <= 4.0:
            term_class_dict[term] = 1
        elif valence <= 6.0:
            term_class_dict[term] = 2
        elif valence <= 7.5:
            term_class_dict[term] = 3
        else:
            term_class_dict[term] = 4

    aligned_context_dict = {0: 'It is very unpleasant to think about WORD',
        1: 'It is unpleasant to think about WORD',
        2: 'It is neither pleasant nor unpleasant to think about WORD',
        3: 'It is pleasant to think about WORD',
        4: 'It is very pleasant to think about WORD',}

    misaligned_context_dict = {4: 'It is very unpleasant to think about WORD',
        3: 'It is unpleasant to think about WORD',
        2: 'It is neither pleasant nor unpleasant to think about WORD',
        1: 'It is pleasant to think about WORD',
        0: 'It is very pleasant to think about WORD',}

#Collect embeddings and write to a dictionary
embedding_dict = {}

SETTING = 'bleached'
if SETTING == 'bleached':
    embedding_dict = {}

    for idx, term in enumerate(term_list):
        context = TEMPLATE.replace('WORD', term)
        embedding_dict[term] = get_embeddings(term, context, CURRENT_MODEL, CURRENT_TOKENIZER, tensor_type=TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

SETTING = 'random'
if SETTING == 'random':

    for idx, term in enumerate(term_list):
        embedding_dict[term] = get_embeddings(term, context_dict[term], CURRENT_MODEL, CURRENT_TOKENIZER, tensor_type=TENSOR_TYPE)
        print(f'{term} worked')

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

SETTING = 'aligned'
if SETTING == 'aligned':
    embedding_dict = {}

    for idx, term in enumerate(term_list):
        context = aligned_context_dict[term_class_dict[term]].replace('WORD', term)
        embedding_dict[term] = get_embeddings(term, context, CURRENT_MODEL, CURRENT_TOKENIZER, tensor_type=TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

SETTING = 'misaligned'
if SETTING == 'misaligned':
    embedding_dict = {}

    for idx, term in enumerate(term_list):
        context = misaligned_context_dict[term_class_dict[term]].replace('WORD', term)
        embedding_dict[term] = get_embeddings(term, context, CURRENT_MODEL, CURRENT_TOKENIZER, tensor_type=TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)



#Get CoLA Test Embeddings
k = pd.read_csv(f'D:\\in_domain_dev.tsv',sep='\t',header=None)

ids = k.index.to_list()
labels = k[1].to_list()
label_dict = {ids[idx]:labels[idx] for idx in range(len(ids))}

sentences = k[3].to_list()
sentence_dict = {ids[idx]:sentences[idx] for idx in range(len(ids))}

sentences = [i.strip() for i in sentences]
new_sentences = [i.rstrip(punctuation) for i in sentences]
new_sentence_dict = {ids[idx]:new_sentences[idx] for idx in range(len(ids))}

actual_last_word = [i.rsplit(' ',1)[1] for i in new_sentences]
trunced = [i.rsplit(' ',1)[0] for i in new_sentences]

last_dict = {}
trunc_dict = {}
gpt2_predictions = {}
gpt2_pos = {}
no_punct_dict = {}

for idx in range(len(ids)):
    sentence = trunced[idx]
    encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
    output = MODEL_GPT2(encoded)
    last_hs = np.array(output[-1][12][0][-1])
    trunc_dict[idx] = last_hs

    pred = np.argmax(np.squeeze(output[0])[-1])
    next_word = MODEL_TOKENIZER_GPT2.decode([pred])
    gpt2_predictions[idx] = next_word

    new_ = sentence + next_word
    pos = pos_tag(word_tokenize(new_))[-1]
    gpt2_pos[idx] = pos

with open(f'D:\\cola_test\\trunc_vectors_val.pkl','wb') as pkl_writer:
    pickle.dump(trunc_dict,pkl_writer)

with open(f'D:\\cola_test\\gpt2_preds_val.pkl','wb') as pkl_writer:
    pickle.dump(gpt2_predictions,pkl_writer)

with open(f'D:\\cola_test\\gpt2_pred_pos_val.pkl','wb') as pkl_writer:
    pickle.dump(gpt2_pos,pkl_writer)

print('done trunced')

for idx in range(len(ids)):
    sentence = sentences[idx]
    encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
    output = MODEL_GPT2(encoded)
    last_hs = np.array(output[-1][12][0][-1])
    last_dict[idx] = last_hs

with open(f'D:\\cola_test\\last_vectors_val.pkl','wb') as pkl_writer:
    pickle.dump(last_dict,pkl_writer)

print('done last')

for idx in range(len(ids)):
    sentence = new_sentences[idx]
    encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
    output = MODEL_GPT2(encoded)
    last_hs = np.array(output[-1][12][0][-1])
    no_punct_dict[idx] = last_hs

with open(f'D:\\cola_test\\no_punct_vectors_val.pkl','wb') as pkl_writer:
    pickle.dump(no_punct_dict,pkl_writer)

print('done everything')