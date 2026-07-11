# Data storage — lab S3 buckets and dataset fetching

How shared datasets (nuPlan bins, WOMD bins, checkpoints) are stored in the
lab's S3 buckets and pulled onto a machine for training and evaluation.

## Fetching data

Datasets are registered in `data_utils/datasets.yaml` and land under a single
local data root — `$PUFFERDRIVE_DATA_ROOT`, defaulting to `<repo>/data/`
(gitignored):

```bash
python data_utils/fetch_data.py --list            # what exists, what is local
python data_utils/fetch_data.py nuplan_dev_maps   # sync one dataset
python data_utils/fetch_data.py nuplan_dev_maps --data-root /scratch/$USER/data
```

Syncs are incremental. On clusters, point `--data-root` (or export
`PUFFERDRIVE_DATA_ROOT`) at shared scratch so one copy serves all jobs, then
reference it from configs, e.g.
`--env.map-dir $PUFFERDRIVE_DATA_ROOT/nuplan_dev_maps`.

## AWS access

Ask Eugene Vinitsky or Riccardo Savorgnan for an AWS user account. All access
is controlled through IAM groups — no direct bucket policies.

Credentials are only needed for private datasets. A manifest entry marked
`public: true` lives in a public-read bucket and fetches without any AWS
account (the script issues unsigned requests). All buckets are IAM-only
today; if nuPlan-derived bins are promoted to a public-read location — which
their CC BY-NC-SA license permits — flipping the manifest entry to
`public: true` is the whole change. WOMD-derived data must stay IAM-gated
(see below).

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

Current temporary home for development nuPlan bins:
`s3://pufferdrive-personal/valentin/0.3.2/{nuplan_train,nuplan_val}` (the
`nuplan_dev_train` / `nuplan_dev_val` manifest entries — note the train split
is ~475 GB; start with val). Once the conversion stabilizes they move to
`pufferdrive-bins` under a proper version prefix.

Adding a dataset = upload to the appropriate bucket, add an entry to
`data_utils/datasets.yaml` (source URI, description, license, size), and it
becomes fetchable by name.

## Licensing constraints on redistribution

What each upstream license allows determines where converted data may live
and who may access it. Lab-internal S3 (IAM-gated) is fine for all of these;
the constraints below apply to publishing outside the lab.

- **nuPlan (Motional)** — CC BY-NC-SA 4.0, with Motional's dataset terms
  prevailing on conflict. Redistribution of the data and derivatives
  (including converted `.bin` files) is permitted if attributed to Motional,
  non-commercial, and shared under the same CC BY-NC-SA 4.0 license. Public
  hosting of converted nuPlan bins is therefore allowed with the right
  license notice attached.
- **Waymo Open Motion Dataset** — custom terms: copies and modifications
  (which includes format-converted bins) may only be distributed to people
  who have registered at waymo.com/open and agreed to Waymo's terms. Public
  ungated hosting of WOMD-derived bins is not permitted; use gated
  distribution or keep them lab-internal.
- **CARLA assets** — CC-BY 4.0. The town map bins are freely redistributable
  with attribution, which is why they ship directly in the repo.
