import cancerrisknet.learn.train as train
from copy import deepcopy
from cancerrisknet.utils.learn import init_metrics_dictionary, get_dataset_loader, get_train_variables
from collections import defaultdict
from cancerrisknet.utils.parsing import get_code
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients, LayerGradientShap,  TokenReferenceBase, visualization
import torch
import pandas as pd
import numpy as np
from functools import partial


torch.backends.cudnn.enabled = False


def compute_attribution(attribute_data, model, args,class_index=0, only_positive=True,\
                        model_for_preds=None, attribution_method="absolute", pred_threshold=0, \
                        attribution_sum_method='sum'):

    """
    Computes the attribution of the given attribute_data using the given model and arguments.

    Args:
    - attribute_data: The data to compute the attribution for.
    - model: The model to use for computing the attribution.
    - args: The arguments to use for computing the attribution.
    - task_index: The index of the task to compute the attribution for.

    Returns:
    - word2attr: A defaultdict that maps each code to its attribution.
    - word2censor_attr: A defaultdict that maps each time bin to a defaultdict that maps each code to its attribution.
    """
    
    model = model.to(args.device)
    test_data_loader = get_dataset_loader(args, attribute_data)
    lig_code = LayerIntegratedGradients(model, model.model.code_embed)
    if hasattr(model.model, 'a_embed_add_fc') and hasattr(model.model, 'a_embed_scale_fc'):
        age_embeddings_layers = [model.model.a_embed_add_fc, model.model.a_embed_scale_fc]
        lig_age = LayerIntegratedGradients(model, age_embeddings_layers)
    else:
        lig_age = None
    test_iterator = iter(test_data_loader)
    word2attr = defaultdict(list)
    word2attr_y= defaultdict(list)
    word2censor_attr = defaultdict(partial(defaultdict, list))
    month_index=3
    #args.max_batches_per_dev_epoch=0
    for i, batch in enumerate(tqdm(test_iterator)):

        #2024-01-08: in the ClassRisk implementation, batch['y'] is not set and we need to set it here
        #2024-03-05: I am uncertain if the following line is correct, as batch['y'] refers to the gold
        #while batch['y_seq'] refers to a given time point. I will leave it as is for now.
        batch['y']=batch['y_seq'][:,month_index]==class_index
        #if batch['y'].sum() == 0 and only_positive:
        #    continue
        batch = train.prepare_batch(batch, args)        

        ###START DEBUGGING CODE
        #how many samples are in one batch?
        #print("Batch size: ", batch['x'].shape[0])

        codes, attr, ages, add_attr_ages, scale_attr_ages, combined_add_ages, preds = \
            attribute_batch(lig_code, lig_age, batch,class_index=class_index, month_idx=month_index, \
                            model_for_preds=model_for_preds, attribution_method=attribution_method,
                            attribution_sum_method=attribution_sum_method)

        #print('length of codes (corresponding to trajectories)', len(codes))
        #print(codes)


        #days_to_censor is a tensor of shape [num_classes,] containing the days to censor for each class
        #this might be changed to a scalar later
        for patient_codes, patient_attr, gold, days, pred in zip(codes, attr, batch['y'], batch['days_to_censor'][class_index], preds):
            #print('patient_codes', patient_codes)
            if(pred<pred_threshold):
                continue
            
            patient_codes = patient_codes.split()
            time_bin = int(days//30)

            codes_dictionary='SNOMED'

            for c, a in zip(patient_codes, patient_attr[-len(patient_codes):]):
                if(codes_dictionary=='SNOMED'):
                    #when using SNOMED, the values are already the codes
                    code=c
                else:
                    code = get_code(args, c)

                word2attr[code].append(a)
                if gold:
                    word2attr_y[code].append(a)
                    word2censor_attr[time_bin][code].append(a)
        for patient_age, patient_age_attr in zip(ages, add_attr_ages):
            word2attr["Add-Age-{}".format(patient_age)].append(patient_age_attr)
        for patient_age, patient_age_attr in zip(ages, scale_attr_ages):
            word2attr["Scale-Age-{}".format(patient_age)].append(patient_age_attr)
        for patient_age, patient_age_attr in zip(ages, combined_add_ages):
            word2attr["Combined-Age-{}".format(patient_age)].append(patient_age_attr)
        if i >= args.max_batches_per_dev_epoch:
            break
    return word2attr, word2attr_y, word2censor_attr


def attribute_batch(explain_code, explain_age, batch, class_index=0, month_idx=3, model_for_preds=None, \
                    attribution_method="absolute", attribution_sum_method='sum'):
    """
    Computes the attributions for the given batch of data using the provided explainers.

    Args:
        explain_code: The explainer for the code input.
        explain_age: The explainer for the age input.
        batch: The batch of data to compute attributions for.
        task_index: The index of the task to compute attributions for.
        month_idx: The index of the month to compute attributions for.

    Returns:
        A tuple containing the following:
        - The code string for the batch.
        - The attributions for the code input.
        - The age of the batch in years.
        - The attribution for the age input (additive).
        - The attribution for the age input (scaling).
        - The combined attribution for the age input (additive and scaling).
    """
    batch_age = deepcopy(batch)
    index=(class_index, month_idx)
    if(model_for_preds is not None):
        logits = model_for_preds(batch['x'],batch)
        probs = torch.sigmoid(logits).cpu().data.numpy() 
    else:
        probs=None
    if explain_code:
        attributions_code = explain_code.attribute(inputs=(batch['x'],batch['age_seq'],batch['time_seq'],batch['age']),
                                                   n_steps=5,
                                                   return_convergence_delta=False,
                                                   target=index,
                                                   additional_forward_args=batch)

        if(attribution_sum_method=='sum'):
            # If the method is 'sum', sum up the attribution values across the specified dimension (dim=2).
            # This collapses the third dimension and aggregates the attribution scores.
            attributions_code = attributions_code.sum(dim=2).squeeze(0)

            # In this case, we need to take the absolute value of the attribution (see bug E234 and E235)
            attributions_code = torch.abs(attributions_code)

            #normalize per sample.
            attributions_code = attributions_code / attributions_code.sum(dim=1, keepdim=True)

        elif(attribution_sum_method=='sum_norm'):
            # If the method is 'sum_norm', first compute the absolute value of all attribution values.
            # This ensures that all contributions are treated as positive quantities.
            # Then, sum up these absolute values across the specified dimension (dim=2),
            attributions_code = torch.abs(attributions_code).sum(dim=2).squeeze(0)
            
            #normalize per sample.
            attributions_code = attributions_code / attributions_code.sum(dim=1, keepdim=True)
        elif(attribution_sum_method=='L2'):
            attributions_code = attributions_code.sum(dim=2).squeeze(0)
            #normalize per sample with L2 normx
            attributions_code = attributions_code / torch.norm(attributions_code, p=2, dim=1, keepdim=True)


        attributions_code = attributions_code.cpu().detach().numpy()
        if(attribution_method=="relative"):
            attributions_code = attributions_code * probs[:,class_index,month_idx].reshape(-1, 1) #relative attribution
    
    else:
        attributions_code = []

    if explain_age:
        attributions_age = explain_age.attribute(inputs=(batch['x'],batch['age_seq'],batch['time_seq'],batch['age']),
                                                 n_steps=2,
                                                 return_convergence_delta=False,
                                                 target=index,
                                                 attribute_to_layer_input=True,
                                                 additional_forward_args=batch_age)
        attributions_age[0] = attributions_age[0].sum(dim=(-1, -2)).squeeze()
        attributions_age[0] = attributions_age[0]#/torch.norm(attributions_age[0])
        attributions_age[1] = attributions_age[1].sum(dim=(-1, -2)).squeeze()
        attributions_age[1] = attributions_age[1]#/torch.norm(attributions_age[1])

        age_attribution_add = attributions_age[0].cpu().detach().numpy()
        age_attribution_scale = attributions_age[1].cpu().detach().numpy()
        age_attribution_combined = (attributions_age[0] + attributions_age[1]).cpu().detach().numpy()
    else:
        age_attribution_add = []
        age_attribution_scale = []
        age_attribution_combined = []
    
    return batch['code_str'], attributions_code, (batch_age['age']//365).squeeze().tolist(), age_attribution_add,\
        age_attribution_scale, age_attribution_combined, probs[:,class_index,month_idx].reshape(-1, 1)
