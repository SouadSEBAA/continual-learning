from federated.utils import DatasetSplit


class LocalUpdate(object):
    def __init__(self, train_datasets, idxs, client_id, watch, train_fn, iters, batch_size, baseline, loss_cbs, eval_cbs, sample_cbs, context_cbs, generator, gen_iters, gen_loss_cbs, **kwargs):
        self.traindata = [ DatasetSplit(train_dataset, idxs[i]) for i, train_dataset in enumerate(train_datasets)]
        self.client_id = client_id
        self.watch = watch
        self.train_fn = train_fn
        self.iters = iters
        self.batch_size = batch_size
        self.baseline = baseline
        self.loss_cbs = loss_cbs
        self.eval_cbs = eval_cbs
        self.sample_cbs = sample_cbs
        self.context_cbs = context_cbs
        self.generator = generator
        self.gen_iters = gen_iters
        self.gen_loss_cbs = gen_loss_cbs
        self.kwargs = kwargs

    def update_weights(self, model, global_round):
        eval_cbs = [
            cb_wrapper(global_round, self.client_id) if self.watch else None
            for cb_wrapper in filter(lambda x: x is not None, self.eval_cbs)
        ]
        loss_dict = self.train_fn(
            model,
            self.traindata,
            iters=self.iters,
            batch_size=self.batch_size,
            baseline=self.baseline,
            loss_cbs=self.loss_cbs,
            eval_cbs=eval_cbs,
            sample_cbs=self.sample_cbs,
            context_cbs=self.context_cbs,
            generator=self.generator,
            gen_iters=self.gen_iters,
            gen_loss_cbs=self.gen_loss_cbs,
            **self.kwargs,
        )
        return model.state_dict(), loss_dict