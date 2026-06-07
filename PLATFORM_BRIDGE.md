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

## Step 2 — AWS Phase 0 (DECISION: the platform owns the bucket)

The bucket and the ingest trigger live in the **platform's** AWS account. The
bridge side creates **nothing** in AWS — we only receive credentials and use them.
(If that ever changes, see the Appendix for the create-it-yourself steps.)

### 2.1 — Request access from the platform team

Send them this exact ask:

> We're ready to upload capture packets to your ingest bucket. Please provide:
> 1. **Bucket name** and **region**.
> 2. **Credentials** scoped to write packets — either
>    (a) an IAM access key for a user, or
>    (b) a **role ARN** we can assume (preferred if you use IAM Identity Center),
>    with this least-privilege policy on `arn:aws:s3:::<bucket>/raw/parcels/*`:
>    `s3:PutObject`, `s3:GetObject` (we do a `head_object` idempotency check), and
>    `s3:ListBucket` on the bucket scoped to prefix `raw/parcels/*`.
> 3. Confirm the **ingest notification is wired** (S3 `ObjectCreated` → Lambda,
>    suffix `metadata.json`) so dropping a packet auto-creates labeling tasks.
> 4. A copy of `schemas/metadata.schema.json` (commit `19ec4fe`) so we can
>    validate every packet locally with `--schema` before upload.
> 5. The `parcel_id` / `run_id` convention you want us to use (we default to
>    `PARCEL_AARIN01` / `RUN001`).

The least-privilege policy they should attach (for their reference):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "WriteAndCheckPackets", "Effect": "Allow",
     "Action": ["s3:PutObject", "s3:GetObject"],
     "Resource": "arn:aws:s3:::<BUCKET>/raw/parcels/*"},
    {"Sid": "ListBucketScoped", "Effect": "Allow",
     "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::<BUCKET>",
     "Condition": {"StringLike": {"s3:prefix": "raw/parcels/*"}}}
  ]
}
```

### 2.2 — Configure the credentials they give you

**If they give an access key:**
```bash
aws configure --profile kara
#   AWS Access Key ID     : <from platform team>
#   AWS Secret Access Key : <from platform team>
#   Default region name   : <their region, e.g. us-east-1>
#   Default output format : json
```

**If they give a role ARN to assume**, add to `~/.aws/config`:
```ini
[profile kara]
role_arn = arn:aws:iam::<THEIR_ACCOUNT_ID>:role/<RoleName>
source_profile = <your-base-profile>   # or credential_source = Environment / Ec2InstanceMetadata
region = us-east-1
```

**Sanity check** (lists only our prefix — full bucket list may be denied, which is
fine):
```bash
aws --profile kara s3 ls s3://<BUCKET>/raw/parcels/
```
The packetizer also honors `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env vars or
the default credential chain (no `--profile`).

---

## Step 3 — Real upload
Use the bucket name the platform gave you (shown as `<BUCKET>` below).
```bash
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 \
    --bucket <BUCKET> --profile kara

# scale up once happy (stratified sample, or an explicit list):
python src/packetizer.py --sample 200 --seed 42 --stratify-col shape_group \
    --bucket <BUCKET> --profile kara
python src/packetizer.py --stone-list my_stones.txt --bucket <BUCKET> --profile kara
```
Add `--schema /path/to/platform/schemas/metadata.schema.json` to validate every
packet against the platform's schema before it's sent.

## Step 4 — Verify & operate
- `aws --profile kara s3 ls --recursive s3://<BUCKET>/raw/parcels/ | tail`
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

---

## Appendix — if WE ever own the bucket (not the current plan)

If you run BOTH sides yourself (you are the platform team), you create the bucket
here. The bootstrap script does the bucket + hardening (+ optional scoped IAM user)
in one idempotent shot once `aws configure` has working creds:

```bash
bash scripts/aws_bootstrap.sh <bucket-name> <region> <profile> [--with-iam]
```

Manual equivalent of what that script runs:

```bash
# Region us-east-1 (other regions need --create-bucket-configuration LocationConstraint=...)
aws s3api create-bucket --bucket <BUCKET> --region us-east-1

# Block public access + default encryption
aws s3api put-public-access-block --bucket <BUCKET> \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket <BUCKET> \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# Packetizer IAM (same least-privilege policy as Step 2.1, attached to a user you own)
aws iam create-policy --policy-name KaraPacketizerWrite --policy-document file://packetizer-policy.json
aws iam create-user --user-name kara-packetizer
aws iam attach-user-policy --user-name kara-packetizer \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/KaraPacketizerWrite
aws iam create-access-key --user-name kara-packetizer
```
You'd then coordinate with the platform team to add their Lambda's execution role
to a bucket policy and configure the S3 → Lambda notification on suffix
`metadata.json`. (Not needed under the current platform-owned plan.)
