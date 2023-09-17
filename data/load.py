import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS, NUM_CLASSES

from sklearn.model_selection import train_test_split
from functools import lru_cache

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None, all=False, none=False):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform, all=all, none=none, verbose=verbose)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

#----------------------------------------------------------------------------------------------------------#

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-context experiment.'''

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[name]
    config['output_units'] = config['classes']
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config

#----------------------------------------------------------------------------------------------------------#

def get_context_set(name, scenario, contexts, data_dir="./datasets", only_config=False, verbose=False,
                    exception=False, normalize=False, augment=False, singlehead=False, train_set_per_class=False, 
                    structure=3):
    '''Load, organize and return a context set (both train- and test-data) for the requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first context (permMNIST) or digits
                            are not shuffled before being distributed over the contexts (e.g., splitMNIST, CIFAR100)'''

    ## NOTE: options 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # Define data-type
    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST currently has no support for separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    elif name == "5GNIDD":
        data_type = '5GNIDD'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["CIFAR100_denorm"]
    # check for number of contexts
    # if contexts > config['classes'] and not name=="permMNIST":
    if False and contexts > config['classes'] and not name=="permMNIST":
    #if contexts > config['classes'] and not name=="permMNIST" and not scenario == 'domain':
        raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, config['classes']))
    # -how many classes per context?
    classes_per_context = 10 if name=="permMNIST" else int(np.floor(config['classes'] / contexts))
    #classes_per_context = 2 if scenario == 'domain' else classes_per_context
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario=='domain' or
                                                    # (scenario=="task" and singlehead)) else classes_per_context*contexts
                                                    (scenario=="task" and singlehead)) else config['classes']
    # -if only config-dict is needed, return it
    if only_config:
        return config

    # Depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x*classes_per_context
            ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        
        # print(f"\nScenario: {structure}")
        train_datasets, test_datasets = [], []
        dataset = get_dataset(data_type, dir=data_dir, target_transform=target_transform,
                                verbose=True, augment=augment, normalize=normalize, all=True)
        X, Y = dataset.data, dataset.targets
        subsets = []
        included_classes = define_classes_context(structure, contexts)
        class_counts = {item[0]:[item[1],0] for item in count_classes(included_classes, contexts).items()}

        # split the train and test datasets up into sub-datasets
        for i in range(contexts):
            for j in range(classes):
                if class_counts[j][0]:
                    if structure == 1: # intra class
                        idx = np.array_split(np.where(Y == j)[0], 1)[0]
                    else: # inter class
                        idx = np.array_split(np.where(Y == j)[0], class_counts[j][0])[class_counts[j][1]]
                        class_counts[j][1] += 1 if (class_counts[j][1] + 1) < class_counts[j][0] else 0
                    if j == 0:
                        subset = (X[idx], Y[idx])
                    else:
                        subset = (np.concatenate((subset[0], X[idx]), axis=0), np.concatenate((subset[1], Y[idx]), axis=0))
            subsets.append(subset)
        
        # prepare train and test datasets with all classes
        for i in range(contexts):
            x_train, x_test, y_train, y_test = train_test_split(subsets[i][0], subsets[i][1])

            print(f'context {i+1}: ')

            trainset = get_dataset(data_type, dir=data_dir, verbose=False, none=True)
            trainset.data = x_train
            trainset.targets = y_train
            train_datasets.append(SubDataset(trainset, included_classes[i], verbose=verbose))

            testset = get_dataset(data_type, dir=data_dir, verbose=False, none=True)
            testset.data = x_test
            testset.targets = y_test
            test_datasets.append(SubDataset(testset, included_classes[i]))
            
    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets), config)

@lru_cache(maxsize=None)
def define_classes_context(struct, Ncontexts):
    all_contexts = []
    included_classes = []
    for i in range(Ncontexts):
        all_contexts += [ define_one_context_classes(struct, i, included_classes_so_far=included_classes) ]
        included_classes += all_contexts[-1]
    
    return all_contexts


def define_one_context_classes(structure, i, included_classes_so_far=[]):
    included_classes = list(set(included_classes_so_far))
    # add one class per context
    if structure == 1:
        if i == 0:
            included_classes = [0,1]
        elif i == 1:
            included_classes = [0,1,2]
        elif i == 2:
            included_classes = [0,1,2,3]
        elif i == 3:
            included_classes = [0,1,2,3,4]
        elif i == 4:
            included_classes = [0,1,2,3,4,5]
        elif i == 5:
            included_classes = [0,1,2,3,4,5,6]
        elif i == 6:
            included_classes = [0,1,2,3,4,5,6,7]
        elif i == 7:
            included_classes = [0,1,2,3,4,5,6,7,8]
    elif structure == 2:
        # this order gave 0.91 of accuracy
        if i == 0:
            included_classes = [0,5]
        elif i == 1:
            included_classes = [0,5,6]
        elif i == 2:
            included_classes = [0,5,6,3]
        elif i == 3:
            included_classes = [0,5,6,3,4]
        elif i == 4:
            included_classes = [0,5,6,3,4,7]
        elif i == 5:
            included_classes = [0,5,6,3,4,7,1]
        elif i == 6:
            included_classes = [0,5,6,3,4,7,1,8]
        elif i == 7:
            included_classes = [0,5,6,3,4,7,1,8,2]
    elif structure == 4:
        # for scenario 4 in the draft
        if i == 0:
            included_classes = [0,7]
        elif i == 1:
            included_classes = [0,7,8,5]
        elif i == 2:
            included_classes = [0,7,8,5,1,4]
        elif i == 3:
            included_classes = [0,7,8,5,1,4,2,3]
        elif i == 4:
            included_classes = [0,7,8,5,1,4,2,3,6]
    elif structure == 3:
        #this order gave 0.28 of accuracy with lr=0.01
        if i == 0:
            included_classes = [0,4]
        elif i == 1:
            included_classes = [0,4,6]
        elif i == 2:
            included_classes = [0,4,6,1]
        elif i == 3:
            included_classes = [0,4,6,1,3]
        elif i == 4:
            included_classes = [0,4,6,1,3,7]
        elif i == 5:
            included_classes = [0,4,6,1,3,7,8]
        elif i == 6:
            included_classes = [0,4,6,1,3,7,8,2]
        elif i == 7:
            included_classes = [0,4,6,1,3,7,8,2,5]
        elif i == 8:
            included_classes = [0,4,6,1,3,7,8,2,5]

    # add classes incrementally in a random manner
    # else:
    #     if i == 0:
    #         included_classes = [0, *choices(range(1,NUM_CLASSES)) ]
    #     else:
    #         for _ in range(structure - 2):
    #             if len(s:=set(range(NUM_CLASSES)).difference(included_classes)) != 0:
    #                 included_classes += choices(list(s)) 

    return included_classes

def count_classes(included_classes, Ncontexts):
    classes_count = {i:0 for i in range(NUM_CLASSES)}
    for i in range(Ncontexts):
        for c in range(NUM_CLASSES):
            classes_count[c] += 1 if c in included_classes[i] else 0
    return classes_count
