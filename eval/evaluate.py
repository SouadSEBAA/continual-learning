import numpy as np
import torch
from visual import visual_plt
from visual import visual_visdom
from utils import get_data_loader,checkattr

from sklearn.metrics import confusion_matrix
from data.available import NUM_CLASSES
from prettytable import PrettyTable

####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def calc_metrics(cm):
    # per class performance
    per_class_performance = {
        # 'precision': confusion_matrix.diagonal()/confusion_matrix.sum(axis=0),
        'recall': cm.diagonal() / cm.sum(axis=1),
    }

    # average performance
    tp_attacks = cm[1:, 1:].sum()
    fp_attacks = cm[0, 1:].sum()
    fn_attacks = cm[1:, 0].sum()
    tn_attacks = cm[0, 0]
    brief_cm = {
        "Labeled Malicious": {
            "Predicted Malicious": tp_attacks,
            "Predicted Benign": fn_attacks,
        },
        "Labeled Benign": {
            "Predicted Malicious": fp_attacks,
            "Predicted Benign": tn_attacks,
        }
    }

    average_performance = {}
    average_performance['multiclass accuracy'] = cm.diagonal().sum() / cm.sum()
    average_performance['macro recall'] = per_class_performance['recall'].sum() / len(per_class_performance['recall'])
    average_performance['binary FPR'] = fp_attacks / (tn_attacks + fp_attacks)
    #average_performance['weighted recall'] = sum(per_class_performance['recall'] *  cm.sum(axis=1)) / cm.sum()
    average_performance['binary accuracy'] = (tp_attacks+tn_attacks) / (tp_attacks+tn_attacks+fp_attacks+fn_attacks)
    prec = average_performance['binary precision'] = tp_attacks / (tp_attacks + fp_attacks) 
    rec = average_performance['binary recall'] = tp_attacks / (tp_attacks + fn_attacks)
    average_performance['binary f1-score'] = 2*prec*rec / (prec + rec)

    return per_class_performance, average_performance, brief_cm

def log_context_results(confusion_matrix, average_performance, per_class_performance, classes=None):
    tbl = PrettyTable()
    tbl.field_names = [''] + [f"Predicted {i}" for i in classes]
    for i in range(len(confusion_matrix)):
        tbl.add_row([f"Labeled {classes[i]}"] + [int(confusion_matrix[i][j]) for j in range(len(confusion_matrix))])
    print(tbl)
    print("Per class perfomance:")
    tbl = PrettyTable()
    tbl.field_names = [''] + classes
    for metric in per_class_performance.keys():
        tbl.add_row([metric] + [round(per_class_performance[metric][i], 4) for i in range(len(classes))])
    print(tbl)
    print("Average perfomance:")
    tbl = PrettyTable()
    tbl.field_names = average_performance.keys()
    tbl.add_row([round(average_performance[metric], 4) for metric in tbl.field_names])
    print(tbl)
    print('\n')

def test_acc(model, dataset, batch_size=128, test_size=1024, verbose=True, context_id=None, allowed_classes=None,
             no_context_mask=False, cm=None, **kwargs):
    '''Evaluate accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model.device if hasattr(model, 'device') else model._device()
    cuda = model.cuda if hasattr(model, 'cuda') else model._is_on_cuda()

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply context-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_context_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(context=context_id+1)

    # Should output-labels be adjusted for allowed classes? (ASSUMPTION: [allowed_classes] has consecutive numbers)
    label_correction = 0 if checkattr(model, 'stream_classifier') or (allowed_classes is None) else allowed_classes[0]

    # If there is a separate network per context, select the correct subnetwork
    if model.label=="SeparateClassifiers":
        model = getattr(model, 'context{}'.format(context_id+1))
        allowed_classes = None

    # Loop over batches in [dataset]
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for x, y in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -if the model is a "stream-classifier", add context
        if checkattr(model, 'stream_classifier'):
            context_tensor = torch.tensor([context_id]*x.shape[0]).to(device)
        # -evaluate model (if requested, only on [allowed_classes])
        with torch.no_grad():
            if checkattr(model, 'stream_classifier'):
                scores = model.classify(x.to(device), context=context_tensor)
            else:
                scores = model.classify(x.to(device), allowed_classes=allowed_classes)
        _, predicted = torch.max(scores.cpu(), 1)
        if model.prototypes and max(predicted).item() >= model.classes:
            # -in case of Domain-IL (or Task-IL + singlehead), collapse all corresponding domains to same class
            predicted = predicted % model.classes
        # -update statistics
        y = y-label_correction
        if cm is not None:
            # confusion matrix with rows as real labels and columns as predictions
            cm += confusion_matrix(y, predicted, labels=kwargs["active_classes"])
            # cm = confusion_matrix(y, predicted, labels=range(NUM_CLASSES))
        total_correct += (predicted == y).sum().item()
        total_tested += len(x)
    accuracy = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    # if verbose:
    #     print('=> accuracy: {:.3f}'.format(accuracy))
    #     print('=> confusion matrix: {:.3f}'.format(accuracy))
    
    return accuracy if cm is None else (accuracy, cm)


def test_all_so_far(model, datasets, current_context, iteration, test_size=None, no_context_mask=False,
                    visdom=None, summary_graph=True, plotting_dict=None, verbose=False):
    '''Evaluate accuracy of a classifier (=[model]) on all contexts so far (= up to [current_context]) using [datasets].

    [visdom]      None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_contexts = len(datasets)

    # Evaluate accuracy of model predictions
    # - in the academic CL setting:  for all contexts so far, reporting "0" for future contexts
    # - in task-free stream setting (current_context==None): always for all contexts
    precs = []
    for i in range(n_contexts):
        if (current_context is None) or (i+1 <= current_context):
            allowed_classes = None
            if model.scenario=='task' and not checkattr(model, 'singlehead'):
                allowed_classes = list(range(model.classes_per_context * i, model.classes_per_context * (i + 1)))
            precs.append(test_acc(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, no_context_mask=no_context_mask, context_id=i))
        else:
            precs.append(0)
    if current_context is None:
        current_context = i+1
    average_precs = sum([precs[context_id] for context_id in range(current_context)]) / current_context

    # Print results on screen
    if verbose:
        print(' => ave accuracy: {:.3f}'.format(average_precs))

    # Add results to [plotting_dict]
    if plotting_dict is not None:
        for i in range(n_contexts):
            plotting_dict['acc per context']['context {}'.format(i+1)].append(precs[i])
        plotting_dict['average'].append(average_precs)
        plotting_dict['x_iteration'].append(iteration)
        plotting_dict['x_context'].append(current_context)

    # Send results to visdom server
    names = ['context {}'.format(i + 1) for i in range(n_contexts)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            precs, names=names, title="accuracy ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test accuracy"
        )
        if n_contexts>1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_precs], names=["ave"], title="ave accuracy ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test accuracy"
            )


def initiate_plotting_dict(n_contexts):
    '''Initiate <dict> with accuracy-measures to keep track of for plotting.'''
    plotting_dict = {}
    plotting_dict["acc per context"] = {}
    for i in range(n_contexts):
        plotting_dict["acc per context"]["context {}".format(i+1)] = []
    plotting_dict["average"] = []      # average accuracy over all contexts so far: Task-IL  -> only classes in context
                                       #                                            Class-IL -> all classes so far
    plotting_dict["x_iteration"] = []  # total number of iterations so far
    plotting_dict["x_context"] = []    # number of contexts so far (i.e., context on which training just finished)
    return plotting_dict


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----GENERATION EVALUATION----####
####-----------------------------####

def show_samples(model, config, pdf=None, visdom=None, size=32, pdf_title="Generated images", visdom_title="Samples"):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Generate samples from the model
    sample = model.sample(size)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual_plt.plot_images_from_tensor(image_tensor, pdf, title=pdf_title, nrow=nrow)
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, title='{} ({})'.format(visdom_title, visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)


####--------------------------------------------------------------------------------------------------------------####

####---------------------------------####
####----RECONSTRUCTION EVALUATION----####
####---------------------------------####

def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, context=None):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Get data
    data_loader = get_data_loader(dataset, size, cuda=model._is_on_cuda())
    (data, labels) = next(iter(data_loader))
    data, labels = data.to(model._device()), labels.to(model._device())

    # Evaluate model
    with torch.no_grad():
        recon_batch = model(data, full=False)

    # Plot original and reconstructed images
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size*2)))
    # -make plots
    if pdf is not None:
        context_stm = "" if context is None else " (context {})".format(context)
        visual_plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=nrow, title="Reconstructions" + context_stm
        )
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, title='Reconstructions ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)
