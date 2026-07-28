# Data storage — lab S3 buckets and dataset fetching

How shared datasets (nuPlan bins, WOMD bins, checkpoints) are stored in the
lab's S3 buckets and pulled onto a machine for training and evaluation.

## Fetching data

Datasets are registered in `data_utils/datasets.yaml` and land under
`<repo>/data/` (gitignored), where the default config expects them.
Fetching elsewhere (`--data-root`) means pointing `env.map_dir` there too.

```bash
python data_utils/fetch_data.py          # the defaults: nuplan_mini_train + nuplan_mini_val (~10 GB each)
python data_utils/fetch_data.py --list   # what exists, what is local
python data_utils/fetch_data.py nuplan_train --data-root /scratch/$USER/data
```

A bare run fetches the entries marked `default: true` in the manifest — the
mini splits. The full splits run to hundreds of GB; fetch those by name.
Syncs are incremental.

## AWS access

Ask Eugene Vinitsky or Riccardo Savorgnan for an AWS user account. All access
is controlled through IAM groups — no direct bucket policies. Credentials are
only needed for private datasets.

## Buckets

| Bucket | Purpose | Versioning |
| --- | --- | --- |
| `pufferdrive-bins` | Production binaries (versioned deployments) | yes |
| `pufferdrive-data` | Operational data — likely locked read-only in the future | no |
| `pufferdrive-bins-test` | Shared testing deployments before promoting to prod | no |
| `pufferdrive-personal` | Individual experiments and temporary files (`<bucket>/<user>/...`) | no |

`pufferdrive-data` holds the pipeline stages upstream of the bins:
`raw-files/{nuplan,waymo-open-motion-dataset,qualcomm}` (original downloads)
and `py123d-conversions/{nuplan,waymo-open-motion-dataset}` (arrow stage).
Most users never need these — fetch the converted bins instead.

Production bins live in `pufferdrive-bins` under
`<dataset>/<conversion-version>/<split>/`:

```
s3://pufferdrive-bins/nuplan/0.3.2/train/        # full train, ~475 GB
s3://pufferdrive-bins/nuplan/0.3.2/val/          # full val,   ~48 GB
s3://pufferdrive-bins/nuplan/0.3.2-mini/train/   # 10 GB sample — default
s3://pufferdrive-bins/nuplan/0.3.2-mini/val/     # 10 GB sample — default
```

Adding a dataset = upload to the appropriate bucket, add an entry to
`data_utils/datasets.yaml` (source URI, description, license, size), and it
becomes fetchable by name.
