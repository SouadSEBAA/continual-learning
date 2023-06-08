from eval import evaluate
from visual import visual_visdom


def _global_eval_cb(test_datasets, n_classes, log=1, visdom=None, test_size=None, S='mean'):
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
                    active_classes=range(n_classes),
                )
                accs_per_context.append(context_acc)
            average_acc = sum(accs_per_context) / n_contexts
            visual_visdom._visualize_scalars(
                [average_acc], names=["ave"], title=f"ave accuracy ({visdom['graph']})",
                iteration=global_round, env=visdom["env"], xlabel="Global rounds", ylabel="test accuracy"
            )
    
    return global_eval_cb if (visdom is not None) else None
