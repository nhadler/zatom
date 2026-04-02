python zatom/eval_fm.py \
    experiment=eval \
    logger=csv \
    ckpt_path=logs/train_fm/runs/train_tft_80M_tmqm_2026-04-01_20-19-13/checkpoints/49-16900.ckpt \
    data.datamodule.datasets.mp20.proportion=0.0 \
    data.datamodule.datasets.qm9.proportion=0.0 \
    data.datamodule.datasets.tmqm.proportion=1.0 \
    model.sampling.num_samples=100 \
    model.sampling.batch_size=10 \
    model.sampling.steps=100