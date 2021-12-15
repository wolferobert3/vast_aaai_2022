import numpy as np
import pandas as pd
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import pickle
import copy
import random
from WEAT import WEAT, SV_WEAT
from helper_functions import pca_transform, form_representations, cosine_similarity

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

#Greedily obtained from ANEW lexicon - multiply tokenized by GPT-2
multi_pleasant = ['masterful','dignified','politeness','easygoing','sailboat','blossom','dazzle','soothe','fascinate','jolly','refreshment','elated','luscious','carefree','untroubled','cuddle','christmas','caress','snuggle','rollercoaster','valentine','sweetheart']
multi_unpleasant = ['suffocate','syphilis','rabies','ulcer','mutilate','pollute','morgue','disloyal','toothache','seasick','unfaithful','maggot','leprosy','anguished','detest','stench','crucify','humiliate','gangrene','regretful','lice','roach','putrid']

multi_dominance = ['bathtub','glamour','carefree','nourish','valentine','garter','lightbulb','caress','detest','cuddle','sailboat','swimmer','zest','sweetheart','decorate','dignified','bouquet','fascinate','jolly','penthouse','masterful']
multi_submission = ['humiliate','shamed','unfaithful','flabby','syphilis','gangrene','mutilate','seasick','despairing','impotent','regretful','suffocate','anguished','scapegoat','ache','louse','sissy','morgue','meek','crucify','wasp','deserter']

multi_arousal = ['valentine','pollute','rabies','cockroach','ulcer','humiliate','unfaithful','elated','pervert','christmas','leprosy','dazzle','cyclone','mutilate','crucify','disloyal','guillotine','roach','infatuation','skijump','rollercoaster']
multi_indifferent = ['fatigued','dreary','nonchalant','hairpin','mantel','mucus','prairie','dustpan','kerchief','overcast','utensil','hairdryer','hydrant','golfer','slush','politeness','windmill','thermometer','cork','leisurely','meek','handicap']

#Scripting Area
weat_terms = list(set(flower + insect + instrument + weapon + ea_name + aa_name + ea_name_2 + aa_name_2 + pleasant + unpleasant + pleasant_2 + unpleasant_2 + young + old + male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2 + dominant + arousal + submissive + indifference))
pleasant_weat = list(set(flower + instrument + ea_name + ea_name_2 + pleasant + pleasant_2 + young))
unpleasant_weat = list(set(insect + weapon + aa_name + aa_name_2 + unpleasant + unpleasant_2 + old))
neutral_weat = list(set(male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2))

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
anew_arousal_dict = {anew_terms[idx]: anew_arousal[idx] for idx in range(len(anew_terms))}
anew_dominance_dict = {anew_terms[idx]: anew_dominance[idx] for idx in range(len(anew_terms))}

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
warriner_arousal_dict = {warriner_terms[idx]: warriner_arousal[idx] for idx in range(len(warriner_terms)) if warriner_terms[idx] not in arousal + indifference}
warriner_dominance_dict = {warriner_terms[idx]: warriner_dominance[idx] for idx in range(len(warriner_terms)) if warriner_terms[idx] not in dominant + submissive}
warriner_terms_valence = [key for key in warriner_valence_dict.keys()]
warriner_terms_arousal = [key for key in warriner_arousal_dict.keys()]
warriner_terms_dominance = [key for key in warriner_dominance_dict.keys()]
warriner_valence = [value for value in warriner_valence_dict.values()]
warriner_arousal = [value for value in warriner_arousal_dict.values()]
warriner_dominance = [value for value in warriner_dominance_dict.values()]

term_list = list(set(bellezza_terms + anew_terms + warriner_terms + weat_terms))

#Layerwise VAST by Experimental Setting
WRITE_MODEL = 'gpt2'
CHART_MODEL = 'GPT-2'
SETTING = 'random'
WRITE_SETTING = 'Random'

LAYERS = range(13)
SUBTOKEN_TYPE = 'Last'
TARGET_W = (bellezza_terms, anew_terms)
VALENCE_GROUND_TRUTH = (bellezza_valence, anew_valence)
DOMINANCE_GROUND_TRUTH = (bellezza_valence , anew_dominance)
AROUSAL_GROUND_TRUTH = (bellezza_valence, anew_arousal)
LEXICON = ('Bellezza', 'ANEW')
DIMENSION = ('Valence', 'Dominance', 'Arousal')

with open(f'D:\\cwe_dictionaries\\{WRITE_MODEL}_{SETTING}.pkl', 'rb') as pkl_reader:
    embedding_dict = pickle.load(pkl_reader)

if SETTING == 'misaligned':
    with open(f'D:\\cwe_dictionaries\\{WRITE_MODEL}_aligned.pkl', 'rb') as pkl_reader:
        weat_dict = pickle.load(pkl_reader)
else:
    weat_dict = {key: embedding_dict[key] for key in pleasant + unpleasant + dominant + submissive + arousal + indifference}

lexicon_valence = []
lexicon_dominance = []
lexicon_arousal = []

for idx, lexicon_target in enumerate(TARGET_W):

    layerwise_valence = []
    layerwise_dominance = []
    layerwise_arousal = []

    ground_truth_val = VALENCE_GROUND_TRUTH[idx]
    ground_truth_dom = DOMINANCE_GROUND_TRUTH[idx]
    ground_truth_aro = AROUSAL_GROUND_TRUTH[idx]

    for layer in LAYERS:

        A_vectors_val = form_representations([weat_dict[a][layer] for a in pleasant], rep_type = SUBTOKEN_TYPE)
        B_vectors_val = form_representations([weat_dict[b][layer] for b in unpleasant], rep_type = SUBTOKEN_TYPE)

        A_vectors_dom = form_representations([weat_dict[a][layer] for a in dominant], rep_type = SUBTOKEN_TYPE)
        B_vectors_dom = form_representations([weat_dict[b][layer] for b in submissive], rep_type = SUBTOKEN_TYPE)

        A_vectors_aro = form_representations([weat_dict[a][layer] for a in arousal], rep_type = SUBTOKEN_TYPE)
        B_vectors_aro = form_representations([weat_dict[b][layer] for b in indifference], rep_type = SUBTOKEN_TYPE)

        valence_associations = []
        dominance_associations = []
        arousal_associations = []

        for w in lexicon_target:
            w_vector = form_representations([embedding_dict[w][layer]], rep_type = SUBTOKEN_TYPE)[0]

            valence_association = SV_WEAT(w_vector, A_vectors_val, B_vectors_val)[0]
            valence_associations.append(valence_association)

            dominance_association = SV_WEAT(w_vector, A_vectors_dom, B_vectors_dom)[0]
            dominance_associations.append(dominance_association)

            arousal_association = SV_WEAT(w_vector, A_vectors_aro, B_vectors_aro)[0]
            arousal_associations.append(arousal_association)

        valence_corr = pearsonr(ground_truth_val, valence_associations)[0]
        dominance_corr = pearsonr(ground_truth_dom, dominance_associations)[0]
        arousal_corr = pearsonr(ground_truth_aro, arousal_associations)[0]

        print(f'{WRITE_MODEL} Layer {layer} VAST {SUBTOKEN_TYPE}: {valence_corr}')
        print(f'{WRITE_MODEL} Layer {layer} Dominance Correlation {SUBTOKEN_TYPE}: {dominance_corr}')
        print(f'{WRITE_MODEL} Layer {layer} Arousal Correlation {SUBTOKEN_TYPE}: {arousal_corr}')

        layerwise_valence.append(valence_corr)
        layerwise_dominance.append(dominance_corr)
        layerwise_arousal.append(arousal_corr)

    lexicon_valence.append(layerwise_valence)
    plt.plot(LAYERS, layerwise_valence, label = f'{LEXICON[idx]} Valence', marker = 'o')
    if idx > 0:
        lexicon_dominance.append(layerwise_dominance)
        plt.plot(LAYERS, layerwise_dominance, label = f'{LEXICON[idx]} Dominance', marker = 'o')
        lexicon_arousal.append(layerwise_arousal)
        plt.plot(LAYERS, layerwise_arousal, label = f'{LEXICON[idx]} Arousal', marker = 'o')

plt.xlabel('Layer')
plt.ylabel('Pearson\'s Correlation Coefficient')
plt.title(f'{CHART_MODEL} Valence, Arousal, and Dominance by Lexicon')
plt.legend()
plt.show()




#Tokenization Analysis
A = pleasant
B = unpleasant
POLAR_TOKENIZATION = 'multi'
LEXICON = 'Warriner'
SUBTOKEN_TYPES = ('First', 'Last', 'Mean', 'Max')

ground_truth_dict = warriner_valence_dict

subtoken_vasts = []

#Evolution of Subtoken Representations
with open(f'D:\\cwe_dictionaries\\tokenization_dictionary_{WRITE_MODEL}.pkl', 'rb') as pkl_reader:
    tokenization_dict = pickle.load(pkl_reader)

A_single = [i for i in A if tokenization_dict[i] == 1]
B_single = [i for i in B if tokenization_dict[i] == 1]
final_len = min(len(A_single),len(B_single))

random.shuffle(A_single)
random.shuffle(B_single)

A_single = A_single[:final_len]
B_single = B_single[:final_len]

print(len(A_single))

term_dict_single = {key: value for key, value in ground_truth_dict.items() if tokenization_dict[key] == 1}
term_dict_multi = {key: value for key, value in ground_truth_dict.items() if key not in term_dict_single}

target_single = list(sorted(list(term_dict_single.items()), key = lambda x: x[1]))
target_multi = list(sorted(list(term_dict_multi.items()), key = lambda x: x[1]))

print(len(target_multi))

random.shuffle(target_single)
random.shuffle(target_multi)
lexicon_length = min(len(target_single),len(target_multi))
target_single = target_single[:lexicon_length]
target_multi = target_multi[:lexicon_length]

target_single_terms = [term[0] for term in target_single]
target_single_valence = [term[1] for term in target_single]

target_multi_terms = [term[0] for term in target_multi]
target_multi_valence = [term[1] for term in target_multi]

#Layerwise VAST by Representation

if POLAR_TOKENIZATION == 'single':
    A = A_single
    B = B_single

if POLAR_TOKENIZATION == 'multi':
    #Note: obtained these sets by tokenizing ANEW words and selecting top and bottom k multiply tokenized
    A = multi_pleasant
    B = multi_unpleasant

    if SETTING =='misaligned':
        with open(f'D:\\cwe_dictionaries\\{WRITE_MODEL}_aligned.pkl', 'rb') as pkl_reader:
            weat_dict = pickle.load(pkl_reader)

TARGET_W = target_multi_terms
GROUND_TRUTH = target_multi_valence

vast_scores = []

for subtoken_type in SUBTOKEN_TYPES:

    subtoken_vasts = []

    for idx, layer in enumerate(LAYERS):

        A_vectors = form_representations([weat_dict[a][layer] for a in A], rep_type = subtoken_type)
        B_vectors = form_representations([weat_dict[b][layer] for b in B], rep_type = subtoken_type)

        associations = []

        for w in TARGET_W:
            w_vector = form_representations([embedding_dict[w][layer]], rep_type = subtoken_type)[0]
            association = SV_WEAT(w_vector, A_vectors, B_vectors)[0]
            associations.append(association)

        vast = pearsonr(GROUND_TRUTH, associations)[0]
        print(f'{WRITE_MODEL} Layer {layer} VAST {subtoken_type}: {vast}')

        subtoken_vasts.append(vast)
    
    vast_scores.append(subtoken_vasts)
    plt.plot(LAYERS, subtoken_vasts, label = f'Multi - {subtoken_type}', marker = 'o')

A = multi_pleasant
B = multi_unpleasant
TARGET_W = target_single_terms
GROUND_TRUTH = target_single_valence
subtoken_type = 'Last'

subtoken_vasts = []

for idx, layer in enumerate(LAYERS):

    A_vectors = form_representations([weat_dict[a][layer] for a in A], rep_type = subtoken_type)
    B_vectors = form_representations([weat_dict[b][layer] for b in B], rep_type = subtoken_type)

    associations = []

    for w in TARGET_W:
        w_vector = form_representations([embedding_dict[w][layer]], rep_type = subtoken_type)[0]
        association = SV_WEAT(w_vector, A_vectors, B_vectors)[0]
        associations.append(association)

    vast = pearsonr(GROUND_TRUTH, associations)[0]
    print(f'{WRITE_MODEL} Layer {layer} VAST {subtoken_type}: {vast}')

    subtoken_vasts.append(vast)

vast_scores.append(subtoken_vasts)

plt.plot(LAYERS, subtoken_vasts, label = 'Single Token', marker = 'o')
plt.xlabel('Layer')
plt.ylabel('VAST Score')
plt.title(f'{CHART_MODEL} Warriner Tokenization VASTs - {WRITE_SETTING} Setting - Multi-Token Polar Words')
plt.legend()
plt.show()


#Principal component removal analysis

LAYER = 12
PC_RANGE = list(range(13))
SUBTRACT_MEAN = True
lexica = ('warriner','anew','bellezza')
SUBTOKEN_TYPE = 'Last'
WRITE_MODEL = 'gpt2'
SETTING = 'bleached'
PLOT_TOP_PCS = False

bellezza_scores_val = {'Removed': [], 'Top': []}
anew_scores_val = {'Removed': [], 'Top': []}
warriner_scores_val = {'Removed': [], 'Top': []}
anew_scores_dom = {'Removed': [], 'Top': []}
anew_scores_aro = {'Removed': [], 'Top': []}
warriner_scores_dom = {'Removed': [], 'Top': []}
warriner_scores_aro = {'Removed': [], 'Top': []}

key_idx = ['Removed', 'Top']

term_list = list(embedding_dict.keys())
weat_terms = list(weat_dict.keys())

vector_arr = np.array(form_representations([embedding_dict[term][LAYER] for term in embedding_dict.keys()], rep_type = SUBTOKEN_TYPE))
weat_arr = np.array(form_representations([weat_dict[term][LAYER] for term in weat_dict.keys()], rep_type = SUBTOKEN_TYPE))

vector_arr = np.concatenate((vector_arr,weat_arr),axis=0)

for i in PC_RANGE:

    pca_arr = copy.deepcopy(vector_arr)
    pca_removed, pca_top = pca_transform(pca_arr, i, subtract_mean = SUBTRACT_MEAN)
    
    all_but_top_dict = {term_list[idx]: pca_removed[idx] for idx in range(len(term_list))}
    top_pc_dict = {term_list[idx]: pca_top[idx] for idx in range(len(term_list))}

    weat_rem_dict = {weat_terms[idx]: pca_removed[idx + len(term_list)] for idx in range(len(weat_terms))}
    weat_top_dict = {weat_terms[idx]: pca_top[idx + len(term_list)] for idx in range(len(weat_terms))}

    v_dicts = (all_but_top_dict, top_pc_dict)
    w_dicts = (weat_rem_dict, weat_top_dict)

    for idx, vector_dict in enumerate(v_dicts):

        if idx == 1 and i == 0:
            bellezza_scores_val[key_idx[idx]].append(0)
            anew_scores_val[key_idx[idx]].append(0)
            anew_scores_dom[key_idx[idx]].append(0)
            anew_scores_aro[key_idx[idx]].append(0)
            warriner_scores_val[key_idx[idx]].append(0)
            warriner_scores_dom[key_idx[idx]].append(0)
            warriner_scores_aro[key_idx[idx]].append(0)
            continue

        A_vectors_val = [w_dicts[idx][term] for term in pleasant]
        B_vectors_val = [w_dicts[idx][term] for term in unpleasant]

        A_vectors_dom = [w_dicts[idx][term] for term in dominant]
        B_vectors_dom = [w_dicts[idx][term] for term in submissive]

        A_vectors_aro = [w_dicts[idx][term] for term in arousal]
        B_vectors_aro = [w_dicts[idx][term] for term in indifference]

        if 'bellezza' in lexica:
            bellezza_associations_val = [SV_WEAT(vector_dict[w], A_vectors_val, B_vectors_val)[0] for w in bellezza_terms]
            bellezza_scores_val[key_idx[idx]].append(pearsonr(bellezza_associations_val, bellezza_valence)[0])
            print(f'{CHART_MODEL} Layer {LAYER} Bellezza VAST {i} PCs {key_idx[idx]}: {pearsonr(bellezza_associations_val, bellezza_valence)[0]}')

        if 'anew' in lexica:
            anew_associations_val = [SV_WEAT(vector_dict[w], A_vectors_val, B_vectors_val)[0] for w in anew_terms]
            anew_scores_val[key_idx[idx]].append(pearsonr(anew_associations_val, anew_valence)[0])
            print(f'{CHART_MODEL} Layer {LAYER} ANEW VAST {i} PCs {key_idx[idx]}: {pearsonr(anew_associations_val, anew_valence)[0]}')

            anew_associations_dom = [SV_WEAT(vector_dict[w], A_vectors_dom, B_vectors_dom)[0] for w in anew_terms]
            anew_scores_dom[key_idx[idx]].append(pearsonr(anew_associations_dom, anew_dominance)[0])
            print(f'{CHART_MODEL} Layer {LAYER} ANEW Dominance {i} PCs {key_idx[idx]}: {pearsonr(anew_associations_dom, anew_dominance)[0]}')

            anew_associations_aro = [SV_WEAT(vector_dict[w], A_vectors_aro, B_vectors_aro)[0] for w in anew_terms]
            anew_scores_aro[key_idx[idx]].append(pearsonr(anew_associations_aro, anew_arousal)[0])
            print(f'{CHART_MODEL} Layer {LAYER} ANEW Arousal {i} PCs {key_idx[idx]}: {pearsonr(anew_associations_aro, anew_arousal)[0]}')

        if 'warriner' in lexica:

            warriner_associations_val = [SV_WEAT(vector_dict[w], A_vectors_val, B_vectors_val)[0] for w in warriner_terms_valence]
            warriner_scores_val[key_idx[idx]].append(pearsonr(warriner_associations_val, warriner_valence)[0])
            print(f'{CHART_MODEL} Layer {LAYER} Warriner VAST {i} PCs {key_idx[idx]}: {pearsonr(warriner_associations_val, warriner_valence)[0]}')

            warriner_associations_dom = [SV_WEAT(vector_dict[w], A_vectors_dom, B_vectors_dom)[0] for w in warriner_terms_dominance]
            warriner_scores_dom[key_idx[idx]].append(pearsonr(warriner_associations_dom, warriner_dominance)[0])
            print(f'{CHART_MODEL} Layer {LAYER} Warriner Dominance {i} PCs {key_idx[idx]}: {pearsonr(warriner_associations_dom, warriner_dominance)[0]}')

            warriner_associations_aro = [SV_WEAT(vector_dict[w], A_vectors_aro, B_vectors_aro)[0] for w in warriner_terms_arousal]
            warriner_scores_aro[key_idx[idx]].append(pearsonr(warriner_associations_aro, warriner_arousal)[0])
            print(f'{CHART_MODEL} Layer {LAYER} Warriner Arousal {i} PCs {key_idx[idx]}: {pearsonr(warriner_associations_aro, warriner_arousal)[0]}')


if PC_RANGE[0] == 0:
    start = 1
else:
    start = PC_RANGE[0]

if PLOT_TOP_PCS:
    key = 'Top'
    if 'bellezza' in lexica:
        plt.plot(PC_RANGE[start:], bellezza_scores_val[key][start:], label = f'Bellezza Valence - {key} PCs', marker = 'o')

    if 'anew' in lexica:
        plt.plot(PC_RANGE[start:], anew_scores_val[key][start:], label = f'ANEW Valence - {key} PCs', marker = 'o')
        plt.plot(PC_RANGE[start:], anew_scores_dom[key][start:], label = f'ANEW Dominance - {key} PCs', marker = 'o')
        plt.plot(PC_RANGE[start:], anew_scores_aro[key][start:], label = f'ANEW Arousal - {key} PCs', marker = 'o')

    if 'warriner' in lexica:
        plt.plot(PC_RANGE[start:], warriner_scores_val[key][start:], label = f'Warriner Valence - {key} PCs', marker = 'o')
        plt.plot(PC_RANGE[start:], warriner_scores_dom[key][start:], label = f'Warriner Dominance - {key} PCs', marker = 'o')
        plt.plot(PC_RANGE[start:], warriner_scores_aro[key][start:], label = f'Warriner Arousal - {key} PCs', marker = 'o')

key = 'Removed'
if 'bellezza' in lexica:
    plt.plot(PC_RANGE, bellezza_scores_val[key], label = f'Bellezza Valence - {key} PCs', marker = 'o')

if 'anew' in lexica:
    plt.plot(PC_RANGE, anew_scores_val[key], label = f'ANEW Valence - {key} PCs', marker = 'o')
    plt.plot(PC_RANGE, anew_scores_dom[key], label = f'ANEW Dominance - {key} PCs', marker = 'o')
    plt.plot(PC_RANGE, anew_scores_aro[key], label = f'ANEW Arousal - {key} PCs', marker = 'o')

if 'warriner' in lexica:
    plt.plot(PC_RANGE, warriner_scores_val[key], label = f'Warriner Valence - {key} PCs', marker = 'o')
    plt.plot(PC_RANGE, warriner_scores_dom[key], label = f'Warriner Dominance - {key} PCs', marker = 'o')
    plt.plot(PC_RANGE, warriner_scores_aro[key], label = f'Warriner Arousal - {key} PCs', marker = 'o')

plt.xlabel('Principal Components')
plt.xticks(PC_RANGE)
plt.ylabel('Pearson\'s Correlation Coefficient')
plt.title(f'{CHART_MODEL} Layer {LAYER} by PCs Removed - {WRITE_SETTING} Setting')
plt.legend()
plt.show()

#Bias Tests
A = pleasant
B = unpleasant
X = ea_name
Y = aa_name
BIAS = 'Top Layer Flowers vs. Insects Bias'
LAYER = 12
SUBTRACT_MEAN = True

term_list = list(embedding_dict.keys())
vector_arr = np.array(form_representations([embedding_dict[term][LAYER] for term in embedding_dict.keys()], rep_type = SUBTOKEN_TYPE))

bias_pcs_removed = []
bias_top_pcs = []
biases = [bias_pcs_removed, bias_top_pcs]

for i in PC_RANGE:

    pca_arr = copy.deepcopy(vector_arr)
    pca_removed, pca_top = pca_transform(pca_arr, i, subtract_mean = SUBTRACT_MEAN)

    all_but_top_dict = {term_list[idx]: pca_removed[idx] for idx in range(len(term_list))}
    top_pc_dict = {term_list[idx]: pca_top[idx] for idx in range(len(term_list))}
    v_dicts = (all_but_top_dict, top_pc_dict)

    for idx, vector_dict in enumerate(v_dicts):

        if i == 0 and idx == 1:
            biases[idx].append(0)
            continue

        A_vectors = [vector_dict[term] for term in A]
        B_vectors = [vector_dict[term] for term in B]
        X_vectors = [vector_dict[term] for term in X]
        Y_vectors = [vector_dict[term] for term in Y]

        bias = WEAT(A_vectors, B_vectors, X_vectors, Y_vectors)[0]
        biases[idx].append(bias)

print(biases)

plt.plot(PC_RANGE, bias_pcs_removed, label = 'PCs Removed', marker = 'o')
plt.plot(PC_RANGE[1:], bias_top_pcs[1:], label = 'Top PCs', marker = 'o')
plt.xlabel('Principal Components')
plt.ylabel('Bias Effect Size')
plt.legend()
plt.title(f'{CHART_MODEL} {BIAS} Bias by PCs Nullified')
plt.show()


#Validation on Other Intrinsic Evaluations

ws353 = pd.read_csv(f'D:\\ws353.csv',sep=',')
word_1 = ws353['Word 1'].to_list()
word_2 = ws353['Word 2'].to_list()
human = ws353['Human (Mean)'].to_list()

with open(f'D:\\cwe_dictionaries\\gpt_2_ws353_dict.pkl','rb') as pkl_reader:
    emb_dict = pickle.load(pkl_reader)

LAYERS = list(range(13))
layer_perf = []

for layer in LAYERS:
    cos_sims = []
    for idx in range(len(word_1)):
        w1_emb = emb_dict[word_1[idx]][layer]
        w2_emb = emb_dict[word_2[idx]][layer]
        cs = cosine_similarity(w1_emb,w2_emb)
        cos_sims.append(cs)
    ws = spearmanr(cos_sims,human)[0]
    print(layer)
    print(ws)
    layer_perf.append(ws)

plt.plot(list(range(13)),layer_perf,marker='o')
plt.xlabel('Layer')
plt.ylabel('Spearman Coefficient')
plt.title('WS-353 Performance by Layer')
plt.show()

LAYER = 12
SUBTRACT_MEAN = True
PC_RANGE = range(1,13)

pc_perf = []

for pc_rem in PC_RANGE:
    cos_sims = []
    pca_dict = {}
    ws_arr = np.array([value[LAYER] for key, value in emb_dict.items()])
    ws_words = [key for key, value in emb_dict.items()]

    pc_arr, pc_top = pca_transform(ws_arr,pc_rem,subtract_mean=SUBTRACT_MEAN)

    for idx, word in enumerate(ws_words):
        pca_dict[word] = pc_arr[idx]
    
    for idx in range(len(word_1)):
        w1_emb = pca_dict[word_1[idx]]
        w2_emb = pca_dict[word_2[idx]]
        cs = cosine_similarity(w1_emb,w2_emb)
        cos_sims.append(cs)
    
    ws = spearmanr(cos_sims,human)[0]
    print(ws)

    pc_perf.append(ws)

plt.plot(list(PC_RANGE),pc_perf,marker='o',label='PCs Removed')
plt.xlabel('Principal Components Removed')
plt.ylabel('Spearman Coefficient')
plt.title('WS-353 Performance by PCs Removed')
plt.show()