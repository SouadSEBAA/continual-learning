from eval import evaluate
from visual import visual_visdom


def _fl_global_eval_cb(test_datasets, log=1, visdom=None, test_size=None, S='mean'):
    def global_eval_cb(classifier, global_round):
        if global_round % log == 0:
            n_contexts = len(test_datasets)
            if (S is not None) and hasattr(classifier, 'S'):
                classifier.S = S
            accs_per_context = []
            for i in range(n_contexts):
                context_acc = evaluate.test_acc(
                    model=classifier,
                    dataset=test_datasets[i],
                    test_size=test_size,
                    context_id=i,
                    allowed_classes=None,
                    cm=None,
                )
                accs_per_context.append(context_acc)
            average_acc = sum(accs_per_context) / n_contexts
            visual_visdom._visualize_scalars(
                [average_acc], names=["ave"], title=f"ave accuracy ({visdom['graph']})",
                iteration=global_round, env=visdom["env"], xlabel="Global rounds", ylabel="test accuracy"
            )
    
    return global_eval_cb if (visdom is not None) else None

def _fl_eval_cb(log, test_datasets, visdom=None, test_size=None, iters_per_context=None, S='mean'):
    def eval_cb_wrapper(global_round, client_id):
        def eval_cb(classifier, batch, context=1):
            iteration = batch if (context is None or context == 1) else (context - 1) * iters_per_context + batch
            if iteration % log == 0:
                if (S is not None) and hasattr(classifier, "S"):
                    classifier.S = S
                n_contexts = len(test_datasets)
                precs = []
                for i in range(n_contexts):
                    if (context is None) or (i+1 <= context):
                        precs.append(evaluate.test_acc(classifier, test_datasets[i], test_size=test_size, verbose=False, context_id=i))
                    else:
                        precs.append(0)
                if context is None:
                    context = i + 1
                names = [ f"context {i + 1}" for i in range(n_contexts) ]
                visual_visdom.visualize_scalars(
                    precs,
                    names=names,
                    title=f"accuracy (Client ID {client_id} - Global Round {global_round} - {visdom['graph']})",
                    iteration=iteration,
                    env=visdom["env"],
                    ylabel="test accuracy",
                )
        return eval_cb
    
    return eval_cb_wrapper if (visdom is not None) else None
