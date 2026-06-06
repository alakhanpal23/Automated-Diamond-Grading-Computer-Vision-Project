# Platform bridge — setup & upload runbook

How to get `src/packetizer.py` uploading capture packets to the Kara Data
Platform's S3 ingest. Work top-to-bottom: **dry run → MinIO → AWS**. Each step is
strictly safer than the next, so you never debug credentials and data shape at the
same time.

Packets land under one bucket, prefix-based:

```
s3://<bucket>/raw/parcels/{parcel_id}/{run_id}/{stone_id}/
    brightfield_0.jpg ... darkfield_8.jpg ...
    metadata.json      <- COMMIT MARKER, written last; platform ingest triggers on it
```

---

## Step 0 — Dry run (no AWS, no Docker)

Writes packets to local disk exactly as they'd land in S3. Use it to eyeball the
tree and diff `metadata.json` before any upload.

```bash
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 --out-dir /tmp/packets
python src/packetizer.py --check          # unit self-test of the metadata builder
```

---

## Step 1 — Local MinIO round-trip (test the upload path without AWS)

MinIO is an S3-compatible server; the packetizer talks to it with the same boto3
code it uses for real S3. Requires Docker.

```bash
# 1. Start MinIO (API :9000, console :9001 — login minioadmin/minioadmin)
docker compose -f docker-compose.minio.yml up -d

# 2. Create the bucket (either via the console at http://localhost:9001, or CLI):
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
  aws --endpoint-url http://localhost:9000 s3 mb s3://kara-captures

# 3. Upload a few packets through the packetizer:
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 \
    --bucket kara-captures --endpoint-url http://localhost:9000

# 4. Verify the objects landed (36 frames + 3 metadata.json for --sample 3):
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
  aws --endpoint-url http://localhost:9000 s3 ls --recursive s3://kara-captures/

# 5. Tear down (add -v to wipe the stored objects):
docker compose -f docker-compose.minio.yml down
```

> The boto3 upload path (parallel frames → 2xx confirm → metadata.json last,
> idempotent skip, no-commit-marker-on-failure) is verified end-to-end against an
> S3-compatible server. If MinIO works, real S3 is a credentials/permissions swap.

---

## Step 2 — AWS Phase 0 (account, bucket, credentials)

### 2.0 — Decide WHOSE account the bucket lives in (do this first)

This is the one architectural choice that needs the platform team:

- **Platform owns the bucket (recommended):** they create it, wire the ingest
  Lambda's `metadata.json` notification, and hand you an IAM user/role that can
  only `PutObject` under `raw/parcels/*`. You skip 2.2–2.3 and just configure the
  creds they give you (2.4). Cleanest — the trigger and bucket stay on their side.
- **You own the bucket:** you create it (2.2–2.3) and grant their Lambda
  cross-account read + notification rights. More moving parts; only if required.

Confirm which before creating anything. The rest assumes you're creating it.

### 2.1 — Account & region
- Use an existing AWS account or create one at https://aws.amazon.com/ (needs a
  billing method). Enable MFA on the root user; never use root for daily work.
- Pick **one region** and use it everywhere (bucket, Lambda, creds). Match the
  platform's region. Example below: `us-east-1`.

### 2.2 — Create the S3 bucket
```bash
# us-east-1 (no LocationConstraint):
aws s3api create-bucket --bucket kara-captures --region us-east-1

# any other region needs the location constraint, e.g. eu-west-1:
# aws s3api create-bucket --bucket kara-captures --region eu-west-1 \
#   --create-bucket-configuration LocationConstraint=eu-west-1
```
Bucket names are globally unique — pick your own (e.g. `kara-captures-<org>`).

### 2.3 — Harden the bucket (block public access + encryption)
```bash
aws s3api put-public-access-block --bucket kara-captures \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-encryption --bucket kara-captures \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

### 2.4 — Least-privilege IAM for the packetizer
The packetizer needs only `PutObject` (upload) and `GetObject` (the idempotency
`head_object` check) under the parcels prefix, plus `ListBucket` (optional).

`packetizer-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "WriteAndCheckPackets",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::kara-captures/raw/parcels/*"
    },
    {
      "Sid": "ListBucketScoped",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::kara-captures",
      "Condition": {"StringLike": {"s3:prefix": "raw/parcels/*"}}
    }
  ]
}
```
```bash
aws iam create-policy --policy-name KaraPacketizerWrite \
  --policy-document file://packetizer-policy.json

aws iam create-user --user-name kara-packetizer
aws iam attach-user-policy --user-name kara-packetizer \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/KaraPacketizerWrite

aws iam create-access-key --user-name kara-packetizer   # capture AccessKeyId + SecretAccessKey
```
> Prefer a **role** (assumed via SSO/STS) over a long-lived user key if your org
> uses IAM Identity Center. The packetizer's `--profile` works with either.

### 2.5 — Configure credentials locally
```bash
aws configure --profile kara
#   AWS Access Key ID     : <from create-access-key>
#   AWS Secret Access Key : <from create-access-key>
#   Default region name   : us-east-1
#   Default output format : json

# sanity check:
aws --profile kara s3 ls s3://kara-captures/
```
Alternatives the packetizer honors: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
env vars, or the default credential chain (instance/SSO) with no `--profile`.

---

## Step 3 — Real upload
```bash
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 \
    --bucket kara-captures --profile kara

# scale up once happy (stratified sample, or an explicit list):
python src/packetizer.py --sample 200 --seed 42 --stratify-col shape_group \
    --bucket kara-captures --profile kara
python src/packetizer.py --stone-list my_stones.txt --bucket kara-captures --profile kara
```
Add `--schema /path/to/platform/schemas/metadata.schema.json` to validate every
packet against the platform's schema before it's sent.

## Step 4 — Verify & operate
- `aws --profile kara s3 ls --recursive s3://kara-captures/raw/parcels/ | tail`
- Per-run log: `data/processed/packetizer_log.csv` (stone_id, n_frames,
  n_darkfield, bytes_uploaded, dest, status, error, timestamp).
- Re-runs are **idempotent** (skip stones whose `metadata.json` already exists);
  `--force` re-sends, which is safe because the platform's ingest is idempotent.

---

## Notes / gotchas
- **The platform configures the ingest trigger** (S3 → Lambda on suffix
  `metadata.json`). The bridge's only job is to land objects in the right prefix
  with metadata.json last. Don't add the notification on the bridge side unless
  you own the bucket (2.0).
- **Demo side effects that are NOT bugs:** the stock-report "Days in stock" reads
  ~0 (it's `export_date − captured_at`, and captured_at = upload time); cert
  fields export as "N/A" under the default `kara_native` source (correct for the
  blind-grading demo).
- **Cost:** a few hundred packets is pennies of S3 storage + PUTs. Frames are
  ~50 KB each; ~12 per stone.
