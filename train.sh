python zatom/train_fm.py logger=csv \
  data.datamodule.datasets.mp20.proportion=0.0 \
  data.datamodule.datasets.qm9.proportion=0.0 \
  data.datamodule.datasets.tmqm.proportion=1.0 \
  data.datamodule.batch_size.train=16 \
  data.datamodule.batch_size.val=16 \
  data.datamodule.batch_size.test=16