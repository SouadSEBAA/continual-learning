import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS, networkDataset

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split

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
                                                    (scenario=="task" and singlehead)) else classes_per_context*contexts
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
        # prepare train and test datasets with all classes
        # trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
        #                        verbose=verbose, augment=augment, normalize=normalize)
        # testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
        #                       augment=augment, normalize=normalize)
        # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
        labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        labels_per_dataset_test = [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        # split the train and test datasets up into sub-datasets

        # train_datasets = []
        # i = 0
        # print('\ntraining contexts:')
        # for labels in labels_per_dataset_train:
        #     target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
        #             scenario=='domain' or (scenario=='task' and singlehead)
        #     ) else None
        #     subdataset = SubDataset(trainset, labels, target_transform=target_transform)
        #     train_datasets.append(subdataset)
        #     print(f'context {i}: {labels}, number of samples = {len(subdataset)}')
        #     i += 1
        # i = 0
        # test_datasets = []
        # print('\ntest contexts:')
        # for labels in labels_per_dataset_test:
        #     target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
        #             scenario=='domain' or (scenario=='task' and singlehead)
        #     ) else None
        #     subdataset = SubDataset(testset, labels, target_transform=target_transform)
        #     test_datasets.append(subdataset)
        #     print(f'context {i}: {labels}, number of samples = {len(subdataset)}')
        #     i += 1
        
        print(f"\nScenario: {structure}")
        train_datasets, test_datasets = [], []
        dataset = get_dataset(data_type, dir=data_dir, target_transform=target_transform,
                                verbose=True, augment=augment, normalize=normalize, all=True)
        X, Y = dataset.data, dataset.targets
        subsets = []
        for i in range(contexts):
            for j in range(classes):
                idx = np.array_split(np.where(Y == j)[0], contexts)[i]
                if j == 0:
                    subset = (X[idx], Y[idx])
                else:
                    subset = (np.concatenate((subset[0], X[idx]), axis=0), np.concatenate((subset[1], Y[idx]), axis=0))
            subsets.append(subset)
                
        for i in range(contexts):
            x_train, x_test, y_train, y_test = train_test_split(subsets[i][0], subsets[i][1])

            # same distributions among contxts
            if structure == 1:
                included_classes = range(classes)

            # add one class per context
            if structure == 2:
                included_classes = [j for j in range(i+1)]

            # add classes incrementally in a random manner
            if structure == 3:
                if i == 0:
                    included_classes = [0, 1, 2, 3]
                elif i == 1:
                    included_classes = [0,1]
                elif i == 2:
                    included_classes = [0]
                elif i == 3:
                    included_classes = [0,2,3]
                elif i == 4:
                    included_classes = [0,1,2,3,4]
                elif i == 5:
                    included_classes = [0,3,5]
                elif i == 6:
                    included_classes = [0,2,6]
                elif i == 7:
                    included_classes = [0,2,3,7]

            print(f'\n\ncontext {i}: ')

            trainset = get_dataset(data_type, dir=data_dir, verbose=False, none=True)
            trainset.data = x_train
            trainset.targets = y_train
            train_datasets.append(SubDataset(trainset, included_classes, verbose=verbose))

            testset = get_dataset(data_type, dir=data_dir, verbose=False, none=True)
            testset.data = x_test
            testset.targets = y_test
            test_datasets.append(SubDataset(testset, included_classes))
            
            #uniques, counts = np.unique(y_train, return_counts=True)
            #percentages = dict(zip(uniques, counts * 100 / len(y_train)))
            #print(f'context {i}: len(train) = {len(x_train)}, len(test) = {len(x_test)}')
            #print(percentages)

    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets), config)

