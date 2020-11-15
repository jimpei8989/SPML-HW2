def get_attack_epochs(cfg, max_epoch):
    if cfg.type == "fixed":
        epochs = list(range(1, max_epoch + 1, cfg.args))
    elif cfg.type == "absolute":
        epochs = cfg.args
    elif cfg.type == "relative":
        epochs, cur_epoch = [], 1
        iter_seq = iter(cfg.args)

        while cur_epoch <= max_epoch:
            epochs.append(cur_epoch)
            try:
                cur_epoch += next(iter_seq)
            except StopIteration:
                cur_epoch += cfg.args[-1]
    elif cfg.type == "expand":
        epochs, cur_epoch = [], 1
        iter_seq = iter(sum(([r] * n for r, n in cfg.args), []))

        while cur_epoch <= max_epoch:
            epochs.append(cur_epoch)
            try:
                cur_epoch += next(iter_seq)
            except StopIteration:
                cur_epoch += cfg.args[-1][0]
    else:
        raise NotImplementedError()
    return set(epochs)


def print_verbose(desc, time, log, is_eval=False):
    print(
        f"{'⚔' if is_eval else '⚘'} {desc[:16].center(16):16s} [{time:6.2f}s] ~ "
        + " - ".join(f"{k}: {log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"] if log[k])
    )
