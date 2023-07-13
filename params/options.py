import argparse
from functools import partial

##-------------------------------------------------------------------------------------------------------------------##

# Where to store the data / results / models / plots
store = "./store"

##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser

def add_general_options(parser, main=False, comparison=False, compare_hyper=False, pretrain=False, **kwargs):
    if main:
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='[first] random seed (for each random-module used)')
    if comparison and (not compare_hyper):
        parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    if pretrain:
        parser.add_argument('--convE-stag', type=str, metavar='STAG', default='none',
                            help="tag for saving convE-layers")
        parser.add_argument('--seed-to-stag', action='store_true', help="add seed to tag for saving convE-layers")
    if main:
        parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    parser.add_argument('--data-dir', type=str, default='{}/datasets'.format(store), dest='d_dir',
                        help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='{}/models'.format(store), dest='m_dir',
                        help="default: %(default)s")
    if not pretrain:
        parser.add_argument('--plot-dir', type=str, default='{}/plots'.format(store), dest='p_dir',
                            help="default: %(default)s")
        parser.add_argument('--results-dir', type=str, default='{}/results'.format(store), dest='r_dir',
                            help="default: %(default)s")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_eval_options(parser, main=False, comparison=False, pretrain=False, compare_replay=False, no_boundaries=False,
                     **kwargs):
    eval_params = parser.add_argument_group('Evaluation Parameters')
    if not pretrain:
        eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
    if main:
        eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
    eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    eval_params.add_argument('--visdom-env-name', type=str, help="visdom environment name")
    eval_params.add_argument('--results-dict', action='store_true', help="output dict with results after each task")
    if not comparison:
        eval_params.add_argument('--loss-log', type=int, metavar="N",
                                 help="# iters after which to plot loss (def: # iters)")
        eval_params.add_argument('--acc-log', type=int, metavar="N",
                                 help="# iters after which to plot accuracy (def: # iters)")
    eval_params.add_argument('--acc-n', type=int, default=1024,
                             help="# samples to evaluate accuracy (after each context)")
    if (not no_boundaries) and (not comparison) and (not pretrain):
        eval_params.add_argument('--sample-log', type=int, metavar="N",
                                 help="# iters after which to plot samples (def: # iters)")
    if (not no_boundaries) and (not pretrain) and (not compare_replay):
        eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")
        eval_params.add_argument('--no-samples', action='store_true', help="don't plot generated images")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_problem_options(parser, pretrain=False, no_boundaries=False, **kwargs):
    problem_params = parser.add_argument_group('Problem Specification')
    cl_protocols = ['splitMNIST', 'permMNIST', 'CIFAR10', 'CIFAR100', '5GNIDD']
    problem_params.add_argument('--experiment', type=str, default='CIFAR10' if pretrain else 'splitMNIST',
                             choices=['CIFAR10', 'CIFAR100', 'MNIST', 'MNIST32', '5GNIDD'] if pretrain else cl_protocols)
    if no_boundaries:
        problem_params.add_argument('--stream', type=str, default='fuzzy-boundaries',
                                    choices=['fuzzy-boundaries', 'academic-setting', 'random'])
        problem_params.add_argument('--fuzziness', metavar='ITERS', type=int, default=500, help='amount of fuzziness')
    if not pretrain:
        problem_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
        # add structures
        problem_params.add_argument('--structure', type=int, default=3, choices=[1,2,3,4,5,6])
        problem_params.add_argument('--contexts', type=int, metavar='N', help='number of contexts')
        problem_params.add_argument('--iters', type=int, help="# iterations (mini-batches) per context")
        problem_params.add_argument('--batch', type=int, help="mini batch size (# observations per iteration)")
    if pretrain:
        problem_params.add_argument('--augment', action='store_true',
                                    help="augment training data (random crop & horizontal flip)")
    problem_params.add_argument('--no-norm', action='store_false', dest='normalize',
                                help="don't normalize images (only for CIFAR)")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_model_options(parser, pretrain=False, compare_replay=False, **kwargs):
    model = parser.add_argument_group('Parameters Main Model')
    # -convolutional layers
    model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
    model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
    model.add_argument('--depth', type=int, default=None, help="# of convolutional layers (0 = only fc-layers)")
    model.add_argument('--reducing-layers', type=int, dest='rl', help="# of layers with stride (=image-size halved)")
    model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
    model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
    model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
    model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully connected layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, metavar="N", help="# of units in hidden fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    if (not pretrain) and (not compare_replay):
        model.add_argument('--z-dim', type=int, default=100, help='size of latent representation (if used, def=100)')
    if not pretrain:
        model.add_argument('--singlehead', action='store_true',
                           help="for Task-IL: use a 'single-headed' output layer  (instead of a 'multi-headed' one)")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_train_options(parser, main=False, no_boundaries=False, pretrain=False, compare_replay=False, **kwargs):

    ## Training hyperparameters
    train_params = parser.add_argument_group('Training Parameters')
    if pretrain:
        iter_epochs = train_params.add_mutually_exclusive_group(required=False)
        iter_epochs.add_argument('--epochs', type=int, default=10, metavar='N', help='# epochs (default: %(default)d)')
        iter_epochs.add_argument('--iters', type=int, metavar='N', help='# iterations (replaces "--epochs")')
        train_params.add_argument('--batch', type=int, help="mini batch size")
    train_params.add_argument('--lr', type=float, help="learning rate")
    if not pretrain:
        train_params.add_argument('--optimizer', type=str, default='adam',
                                  choices=['adam', 'sgd'] if no_boundaries else ['adam', 'adam_reset', 'sgd'])
        train_params.add_argument("--momentum", type=float, default=0., help="momentum (if using SGD optimizer)")
    # -initialization / pretraining
    train_params.add_argument('--pre-convE', action='store_true', help="use pretrained convE-layers")
    train_params.add_argument('--convE-ltag', type=str, metavar='LTAG', default='e100',
                              help="tag for loading convE-layers")
    train_params.add_argument('--seed-to-ltag', action='store_true', help="add seed to tag when loading convE-layers")
    train_params.add_argument('--freeze-convE', action='store_true', help="freeze convE-layers")
    # -for Class-IL, which output units should be set to 'active'?
    if (not pretrain) and (not no_boundaries):
        train_params.add_argument('--active-classes', type=str, default='all', choices=["all", "all-so-far", "current"],
                                  dest='neg_samples', help="for Class-IL: which classes to set to 'active'?")
        #--> the above command controls which output units will be set to "active" (the active classes can also
        #    be thought of as 'negative classes', see Li et al., 2020, https://arxiv.org/abs/2011.12216):
        #    - "all-so-far":        the output units of all classes seen so far are set to active
        #    - "all":               always the output units of all classes are set to active
        #    - "current":           only output units of the classes in the current context are set to active

    ## Loss function(s) to be used
    if (not pretrain) and (not compare_replay):
        loss_params = parser.add_argument_group('Loss Parameters')
        loss_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    if main:
        loss_params.add_argument('--bce', action='store_true',
                                 help="use binary (instead of multi-class) classification loss")
    if main and (not no_boundaries):
        loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                            ' examples (if --bce & --scenario="class")')
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_cl_options(parser, main=False, compare_all=False, compare_replay=False, compare_hyper=False,
                   no_boundaries=False, **kwargs):

    ## Baselines
    if main and (not no_boundaries):
        baseline_options = parser.add_argument_group('Baseline Options')
        baseline_options.add_argument('--joint', action='store_true', help="train once on data of all contexts")
        baseline_options.add_argument('--cummulative', action='store_true',
                                      help="train incrementally on data of all contexts so far")
        #---> Explanation for these two "upper-target" baselines:
        # - "joint":        means that the network is trained on a single dataset consisting of the data of all contexts
        # - "cummulative":  means that the network is incrementally trained on all contexts, whereby the training data
        #                   always consists of the training data from all contexts seen so far

    ## Stream-specific options
    if no_boundaries:
        stream_options = parser.add_argument_group('Stream Options')
        stream_options.add_argument('--update-every', metavar='N', type=int, default=100,
                                    help='after how many iterations to consolidate model')
        if compare_all:
            stream_options.add_argument('--replay-update', metavar='N', type=int, default=1,
                                        help='after how many iterations to start replaying observed samples')

    ## Context-specific components
    context_spec = parser.add_argument_group('Context-Specific Component')
    xdg_message = "use 'Context-dependent Gating' (Masse et al, 2018)" if main else "combine all methods with XdG"
    context_spec.add_argument('--xdg', action='store_true', help=xdg_message)
    context_spec.add_argument('--gating-prop', type=float, metavar="PROP",
                              help="-> XdG: prop neurons per layer to gate")
    if main:
        context_spec.add_argument('--separate-networks', action='store_true', help="train separate network per context")
    if compare_all:
        context_spec.add_argument('--fc-units-sep', type=int, metavar="N",
                                  help="# of hidden units with separate network per context")

    ## Parameter regularization
    if not compare_replay:
        param_reg = parser.add_argument_group('Parameter Regularization')
        if main and no_boundaries:
            # With the flexible, 'task-free' CL experiments, currently the only supported param reg option is SI
            param_reg.add_argument('--si', action='store_true', help="select defaults for 'SI' (Zenke et al, 2017)")
            param_reg.add_argument("--weight-penalty", action='store_true',
                                   help="penalize parameters important for past contexts")
            param_reg.add_argument('--reg-strength', type=float, metavar='LAMDA',
                                   help="regularisation strength for weight penalty")
        if main and not no_boundaries:
            # 'Convenience-commands' that select the defaults for specific methods
            param_reg.add_argument('--ewc', action='store_true',
                                   help="select defaults for 'EWC' (Kirkpatrick et al, 2017)")
            param_reg.add_argument('--si', action='store_true', help="select defaults for 'SI' (Zenke et al, 2017)")
            param_reg.add_argument("--ncl", action="store_true",
                                   help="select defaults for 'NCL' (Kao, Jensen et al., 2021)")
            param_reg.add_argument("--ewc-kfac", action="store_true",
                                   help="select defaults for 'KFAC-EWC' (Ritter et al. 2018)")
            param_reg.add_argument("--owm", action="store_true", help="select defaults for 'OWM' (Zeng et al. 2019)")
            # Custom commands for specifying how parameter regularization should be performed
            param_reg.add_argument("--weight-penalty", action='store_true',
                                   help="penalize parameters important for past contexts")
            param_reg.add_argument('--reg-strength', type=float, metavar='LAMDA',
                                   help="regularisation strength for weight penalty")
            param_reg.add_argument("--precondition", action='store_true',
                                   help="parameter regularization by gradient projection")
            param_reg.add_argument("--alpha", type=float, default=1e-10,
                                   help="small constant stabilizing inversion importance matrix")
            param_reg.add_argument("--importance-weighting", type=str, choices=['fisher', 'si', 'owm'])
        if not no_boundaries:
            param_reg.add_argument('--fisher-n', type=int, help="-> Fisher: sample size estimating Fisher Information")
            param_reg.add_argument('--fisher-batch', type=int, default=1, metavar='N',
                                   help="-> Fisher: batch size estimating FI (should be 1)")
            param_reg.add_argument('--fisher-labels', type=str, default='all', choices=['all', 'sample', 'pred', 'true'],
                                   help="-> Fisher: what labels to use to calculate FI?")
            param_reg.add_argument("--fisher-kfac", action='store_true',
                                   help="-> Fisher: use KFAC approximation rather than diagonal")
            param_reg.add_argument("--fisher-init", action='store_true', help="-> Fisher: start with prior (as in NCL)")
            param_reg.add_argument("--fisher-prior", type=float, metavar='SIZE', dest='data_size',
                                   help="-> Fisher: prior-strength in 'data_size' (as in NCL)")
        param_reg.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-> SI: dampening parameter")
        if main and not no_boundaries:
            param_reg.add_argument('--offline', action='store_true',
                                   help="separate penalty term per context (as original EWC)")
            param_reg.add_argument('--gamma', type=float, default=1.,
                                   help="forgetting coefficient Fishers (as in Online EWC)")
        # For the comparison script in which EWC and SI are both run, to enable different hyper-params for both:
        if compare_all and not no_boundaries:
            param_reg.add_argument('--lambda', type=float, dest="ewc_lambda", help="-> EWC: regularisation strength")
        if compare_all:
            param_reg.add_argument('--c', type=float, dest="si_c", help="-> SI: regularisation strength")

    ## Functional regularization
    func_reg = parser.add_argument_group('Functional Regularization')
    if main:
        func_reg.add_argument('--lwf', action='store_true', help="select defaults for 'LwF' (Li & Hoiem, 2017)")
        func_reg.add_argument('--distill', action='store_true', help="use distillation-loss for the replayed data")
    if not compare_replay:
        func_reg.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation loss")
    if main and not no_boundaries:
        func_reg.add_argument('--fromp', action='store_true', help="use 'FROMP' (Pan et al, 2020)")
    if (not compare_hyper) and not no_boundaries:
        func_reg.add_argument('--tau', type=float, help="-> FROMP: regularization strength")
    if compare_replay:
        func_reg.add_argument('--tau-per-budget', action='store_true',
                              help="-> FROMP: use separate tau for each different budget")

    ## Memory buffer parameters (if data is stored)
    buffer = parser.add_argument_group('Memory Buffer Parameters')
    if not compare_replay:
        buffer.add_argument('--budget', type=int, help="how many samples can be stored{}".format(
            " (total budget)" if no_boundaries else " of each class?"
        ), default=1000 if no_boundaries else None)
    if not no_boundaries:
        buffer.add_argument('--use-full-capacity', action='store_true',
                            help="use budget of future classes to initially store more")
    if main and not no_boundaries:
        buffer.add_argument('--sample-selection', type=str, choices=['random', 'herding', 'fromp'])
        buffer.add_argument('--add-buffer', action='store_true',
                            help="add memory buffer to current context's training data")

    ## Replay
    replay_params = parser.add_argument_group('Replay')
    if main:
        replay_choices = ['none', 'current', 'buffer'] if no_boundaries else ['none', 'all', 'generative',
                                                                              'current', 'buffer']
        replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
        replay_params.add_argument('--use-replay', type=str, default='normal', choices=['normal', 'inequality', 'both'])
        #---> Explanation for these three ways to use replay:
        # - "normal":      add the loss on the replayed data to the loss on the data of the current context
        # - "inequality":  use the gradient of the loss on the replayed data as an inequality constraint (as in A-GEM)
        # - "both":        do both of the above
        replay_params.add_argument('--agem', action='store_true',
                                   help="select defaults for 'A-GEM' (Chaudhry et al, 2019)")
    replay_params.add_argument('--eps-agem', type=float, default=1e-7,
                               help="parameter to ensure numerical stability of A-GEM")
    if (not compare_replay) and (not no_boundaries):
        # -parameters for the generative model (if it is a separate model)
        if not compare_hyper:
            replay_params.add_argument('--g-z-dim', type=int, help='size latent space generator (def: as classifier)')
            replay_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (def: as classifier)')
            replay_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (def: as classifier)')
            replay_params.add_argument('--g-iters', type=int, help="# batches to train generator (def: as classifier)")
            replay_params.add_argument('--lr-gen', type=float, help="learning rate generator (def: as classifier)")
        # -parameters for brain-inspired replay
        if main:
            replay_params.add_argument('--brain-inspired', action='store_true',
                                       help="select defaults for 'BI-R' (van de Ven et al, 2020)")
            replay_params.add_argument('--feedback', action="store_true",
                                       help="equip main model with feedback connections")
            replay_params.add_argument('--prior', type=str, default="standard", choices=["standard", "GMM"])
            replay_params.add_argument('--per-class', action='store_true',
                                       help="if selected, each class has its own modes")
        replay_params.add_argument('--n-modes', type=int, default=1,
                                   help="how many modes for prior (per class)? (def=1)")
        if main:
            replay_params.add_argument('--dg-gates', action='store_true', help="use context-specific gates in decoder")
        replay_params.add_argument('--dg-type', type=str, metavar="TYPE",
                                   help="decoder-gates: based on contexts or classes?")
        if not compare_hyper:
            replay_params.add_argument('--dg-prop', type=float, help="decoder-gates: masking-prop")
    if main:
        replay_params.add_argument('--hidden', action="store_true",
                                   help="gen models at 'internal level' (after conv-layers)")

    ## Template-based classification
    if not compare_replay:
        templ_cl = parser.add_argument_group('Template-Based Classification')
        if main:
            templ_cl.add_argument('--icarl', action='store_true',
                                  help="select defaults for '{}iCaRL' (Rebuffi et al, 2017)".format(
                                      'Modified ' if no_boundaries else ''
                                  ))
            templ_cl.add_argument('--prototypes', action='store_true', help="classify using nearest-exemplar-mean rule")
            templ_cl.add_argument('--gen-classifier', action='store_true',
                                  help="use 'Generative Classifier' (van de Ven et al, 2021)")
        if not compare_hyper:
            templ_cl.add_argument('--eval-s', type=int, default=50,
                                  help="-> Generative Classifier: number of importance samples")
        if compare_all:
            templ_cl.add_argument('--fc-units-gc', type=int, metavar="N",
                                  help="# of hidden units with generative classifier")
            templ_cl.add_argument('--fc-lay-gc', type=int, metavar="N", help="# fc-layers with generative classifier")
            templ_cl.add_argument('--z-dim-gc', type=int, metavar="N", help="size latent space generative classifier")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_fl_options(parser, **kwargs):
    fl_params = parser.add_argument_group('Federated Learning')
    # fl_params.add_argument("--fed-avg", action="store_true", help="use FedAvg algorithm for Federated Learning")
    fl_params.add_argument("--fl", action="store_true", help="use Federated Learning", default=False)
    fl_params.add_argument('--fl-num-clients', type=int, help='number of federated clients', default=10)
    fl_params.add_argument('--fl-global-iters', type=int, help='number of rounds of training (global)', default=10)
    fl_params.add_argument('--fl-frac', type=float, help='fraction of clients participating per round', default=1.0)
    fl_params.add_argument("--fl-iid", action="store_true", help="sample dataset in IID fashion (default)", default=True)
    fl_params.add_argument("--fl-non-iid", action="store_true", help="sample dataset in non-IID fashion", default=False)
    fl_params.add_argument("--fl-non-iid-2", action="store_true", help="sample dataset in non-IID fashion (2)", default=False)
    fl_params.add_argument("--fl-num-shards", type=int, help="number of shards for non-IID distribution", default=20)
    fl_params.add_argument("--fl-threaded", action="store_true", help="Use multithreading", default=False)
    fl_params.add_argument('--fl-min-val', type=float, help='minimum percentage value for non-IID distribution (2)', default=0.05)
    fl_params.add_argument('--fl-max-val', type=float, help='maximum percentage value for non-IID distribution (2)', default=0.15)
    split_comma = partial(str.split, sep=',')
    fl_params.add_argument('--fl-watch-clients', type=split_comma, help='FL clients to watch in visdom (example: --fl-watch-clients 0,1,3)', default=[])
    fl_params.add_argument('--fl-acc-n', type=int, default=1024, help="# samples to evaluate accuracy (after each global round)")
    fl_params.add_argument('--fl-acc-log', type=int, metavar="N", default=1, help="# global rounds after which to plot accuracy")
    fl_params.add_argument('--fl-loss-log', type=int, default=1, help="# global rounds after which to plot loss")
    fl_params.add_argument('--fl-vis-single-context', action="store_true", default=False, help="visualize contexts separately")
    return parser

##-------------------------------------------------------------------------------------------------------------------##

def add_bc_options(parser, **kwargs):
    bc_params = parser.add_argument_group('Blockchain')
    # bc_params.add_argument("--fed-avg", action="store_true", help="use FedAvg algorithm for Federated Learning")
    bc_params.add_argument("--bc", "--blockchain", action="store_true", help="use Blockchain", default=False)
    # bc_params.add_argument('--fl-num-clients', type=int, help='number of federated clients', default=10)
    # bc_params.add_argument('--fl-global-iters', type=int, help='number of rounds of training (global)', default=10)
    bc_params.add_argument("--bc-iid", action="store_true", help="sample dataset in IID fashion (default)", default=True)
    bc_params.add_argument("--bc-non-iid", action="store_true", help="sample dataset in non-IID fashion", default=False)
    # bc_params.add_argument("--fl-non-iid-2", action="store_true", help="sample dataset in non-IID fashion (2)", default=False)
    bc_params.add_argument("--bc-num-shards", type=int, help="number of shards for non-IID distribution", default=20)
    bc_params.add_argument('--bc-min-val', type=float, help='minimum percentage value for non-IID distribution (2)', default=0.05)
    bc_params.add_argument('--bc-max-val', type=float, help='maximum percentage value for non-IID distribution (2)', default=0.15)
    # split_comma = partial(str.split, sep=',')
    # bc_params.add_argument('--fl-watch-clients', type=split_comma, help='FL clients to watch in visdom (example: --fl-watch-clients 0,1,3)', default=[])
    # bc_params.add_argument('--fl-acc-n', type=int, default=1024, help="# samples to evaluate accuracy (after each global round)")
    # bc_params.add_argument('--fl-acc-log', type=int, metavar="N", default=1, help="# global rounds after which to plot accuracy")
    # bc_params.add_argument('--fl-loss-log', type=int, default=1, help="# global rounds after which to plot loss")
    # bc_params.add_argument('--fl-vis-single-context', action="store_true", default=False, help="visualize contexts separately")
    # debug attributes
    bc_params.add_argument(
        "-bc-g", "--bc-gpu", type=str, default="0", help="gpu id to use(e.g. 0,1,2,3)"
    )
    bc_params.add_argument(
        "-bc-c", "--bc-cpu", action="store_true", default=False, help="force usage of CPU instead of GPU"
    )
    bc_params.add_argument(
        "-bc-v", "--bc-verbose", type=int, default=1, help="print verbose debug log"
    )
    bc_params.add_argument(
        "-bc-sn",
        "--bc-save_network_snapshots",
        type=int,
        default=0,
        help="only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder",
    )
    bc_params.add_argument(
        "-bc-dtx",
        "--bc-destroy_tx_in_block",
        type=int,
        default=0,
        help="currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.",
    )
    bc_params.add_argument(
        "-bc-rp",
        "--bc-resume_path",
        type=str,
        default=None,
        help="resume from the path of saved network_snapshots; only provide the date",
    )
    bc_params.add_argument(
        "-bc-sf",
        "--bc-save_freq",
        type=int,
        default=5,
        help="save frequency of the network_snapshot",
    )
    bc_params.add_argument(
        "-bc-sm",
        "--bc-save_most_recent",
        type=int,
        default=2,
        help="in case of saving space, keep only the recent specified number of snapshops; 0 means keep all",
    )
    # FL attributes
    # bc_params.add_argument(
    #     "-B", "--batchsize", type=int, default=10, help="local train batch size"
    # ) -> batch
    # bc_params.add_argument(
    #     "-mn", "--model_name", type=str, default="mnist_cnn", help="the model to train"
    # ) -> experiment
    # bc_params.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=float,
    #     default=0.01,
    #     help="learning rate, use value from origin paper as default",
    # ) -> lr
    # bc_params.add_argument(
    #     "-op",
    #     "--optimizer",
    #     type=str,
    #     default="SGD",
    #     help="optimizer to be used, by default implementing stochastic gradient descent",
    # ) -> optimizer
    # bc_params.add_argument(
    #     "-iid", "--IID", type=int, default=0, help="the way to allocate data to devices"
    # ) -> bc_iid & bc_non_iid
    bc_params.add_argument(
        "-bc-max_ncomm",
        "--bc-max_num_comm",
        type=int,
        default=100,
        help="maximum number of communication rounds, may terminate early if converges",
    )
    bc_params.add_argument(
        "-bc-nd",
        "--bc-num_devices",
        type=int,
        default=20,
        help="numer of the devices in the simulation network",
    )
    # bc_params.add_argument(
    #     "-st",
    #     "--shard_test_data",
    #     type=int,
    #     default=0,
    #     help="it is easy to see the global models are consistent across devices when the test dataset is NOT sharded",
    # )
    bc_params.add_argument(
        "-bc-nm",
        "--bc-num_malicious",
        type=int,
        default=0,
        help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise",
    )
    bc_params.add_argument(
        "-bc-nv",
        "--bc-noise_variance",
        type=int,
        default=1,
        help="noise variance level of the injected Gaussian Noise",
    )
    # bc_params.add_argument(
    #     "-le",
    #     "--default_local_epochs",
    #     type=int,
    #     default=5,
    #     help="local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified",
    # ) -> iters
    # blockchain system consensus attributes
    bc_params.add_argument(
        "-bc-ur",
        "--bc-unit_reward",
        type=int,
        default=1,
        help="unit reward for providing data, verification of signature, validation and so forth",
    )
    bc_params.add_argument(
        "-bc-ko",
        "--bc-knock_out_rounds",
        type=int,
        default=6,
        help="a worker or validator device is kicked out of the device's peer list(put in black list) if it's identified as malicious for this number of rounds",
    )
    bc_params.add_argument(
        "-bc-lo",
        "--bc-lazy_worker_knock_out_rounds",
        type=int,
        default=10,
        help="a worker device is kicked out of the device's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)",
    )
    bc_params.add_argument(
        "-bc-pow",
        "--bc-pow_difficulty",
        type=int,
        default=0,
        help="if set to 0, meaning miners are using PoS",
    )

    # blockchain FL validator/miner restriction tuning parameters
    bc_params.add_argument(
        "-bc-mt",
        "--bc-miner_acception_wait_time",
        type=float,
        default=0.0,
        help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each device will just perform same amount(-le) of epochs per round like in FedAvg paper",
    )
    bc_params.add_argument(
        "-bc-ml",
        "--bc-miner_accepted_transactions_size_limit",
        type=float,
        default=0.0,
        help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size",
    )
    bc_params.add_argument(
        "-bc-mp",
        "--bc-miner_pos_propagated_block_wait_time",
        type=float,
        default=float("inf"),
        help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS",
    )
    bc_params.add_argument(
        "-bc-vh",
        "--bc-validator_threshold",
        type=float,
        default=1.0,
        help="a threshold value of accuracy difference to determine malicious worker",
    )
    bc_params.add_argument(
        "-bc-md",
        "--bc-malicious_updates_discount",
        type=float,
        default=0.0,
        help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat",
    )
    bc_params.add_argument(
        "-bc-mv",
        "--bc-malicious_validator_on",
        type=int,
        default=0,
        help="let malicious validator flip voting result",
    )


    # distributed system attributes
    bc_params.add_argument(
        "-bc-ns",
        "--bc-network_stability",
        type=float,
        default=1.0,
        help="the odds a device is online",
    )
    bc_params.add_argument(
        "-bc-els",
        "--bc-even_link_speed_strength",
        type=int,
        default=1,
        help="This variable is used to simulate transmission delay. Default value 1 means every device is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a device will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.",
    )
    bc_params.add_argument(
        "-bc-dts",
        "--bc-base_data_transmission_speed",
        type=float,
        default=70000.0,
        help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.",
    )
    bc_params.add_argument(
        "-bc-ecp",
        "--bc-even_computation_power",
        type=int,
        default=1,
        help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.",
    )

    # simulation attributes
    bc_params.add_argument(
        "-bc-ha",
        "--bc-hard_assign",
        type=str,
        default="*,*,*",
        help='hard assign number of roles in the network, order by worker, validator and miner. e.g. 12,5,3 assign 12 workers, 5 validators and 3 miners. "*,*,*" means completely random role-assigning in each communication round ',
    )
    bc_params.add_argument(
        "-bc-aio",
        "--bc-all_in_one",
        type=int,
        default=1,
        help="let all nodes be aware of each other in the network while registering",
    )
    bc_params.add_argument(
        "-bc-cs",
        "--bc-check_signature",
        type=int,
        default=1,
        help="if set to 0, all signatures are assumed to be verified to save execution time",
    )
    bc_params.add_argument(
        "-bc-cons",
        "--bc-consensus",
        type=str,
        default='poa',
        help="defines the consensus mechanism to use, -pow take effect only if this option is set to 'pow'",
    )

    # bc_params.add_argument('-bc-la', '--bc-least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')
    return parser

##-------------------------------------------------------------------------------------------------------------------##