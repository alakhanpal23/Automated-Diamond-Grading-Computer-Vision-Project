#!/usr/bin/env bash
# Bootstrap the S3 ingest bucket for the platform bridge.
#
# Run this ONCE, after `aws configure` has working credentials on this machine.
# It is idempotent: safe to re-run. It creates the bucket, blocks public access,
# turns on default encryption, and (optionally) creates a least-privilege IAM user
# whose access key the packetizer can use instead of your admin creds.
#
#   bash scripts/aws_bootstrap.sh <bucket-name> [region] [profile] [--with-iam]
#
# examples:
#   bash scripts/aws_bootstrap.sh kara-captures
#   bash scripts/aws_bootstrap.sh kara-captures us-east-1 default --with-iam
set -euo pipefail

BUCKET="${1:?usage: aws_bootstrap.sh <bucket-name> [region] [profile] [--with-iam]}"
REGION="${2:-us-east-1}"
PROFILE="${3:-default}"
WITH_IAM=false
for a in "$@"; do [ "$a" = "--with-iam" ] && WITH_IAM=true; done

AWS=(aws --profile "$PROFILE" --region "$REGION")

echo "== identity =="
"${AWS[@]}" sts get-caller-identity --output table

echo "== bucket: $BUCKET ($REGION) =="
if "${AWS[@]}" s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "  already exists — skipping create"
elif [ "$REGION" = "us-east-1" ]; then
  "${AWS[@]}" s3api create-bucket --bucket "$BUCKET" >/dev/null
  echo "  created"
else
  "${AWS[@]}" s3api create-bucket --bucket "$BUCKET" \
    --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  echo "  created"
fi

echo "== block public access =="
"${AWS[@]}" s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
echo "  done"

echo "== default encryption (SSE-S3) =="
"${AWS[@]}" s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
echo "  done"

if [ "$WITH_IAM" = true ]; then
  echo "== least-privilege IAM user: kara-packetizer =="
  POLICY_DOC='{
    "Version":"2012-10-17",
    "Statement":[
      {"Sid":"WriteAndCheckPackets","Effect":"Allow",
       "Action":["s3:PutObject","s3:GetObject"],
       "Resource":"arn:aws:s3:::'"$BUCKET"'/raw/parcels/*"},
      {"Sid":"ListBucketScoped","Effect":"Allow",
       "Action":"s3:ListBucket","Resource":"arn:aws:s3:::'"$BUCKET"'",
       "Condition":{"StringLike":{"s3:prefix":"raw/parcels/*"}}}
    ]}'
  "${AWS[@]}" iam create-user --user-name kara-packetizer 2>/dev/null \
    && echo "  user created" || echo "  user exists"
  "${AWS[@]}" iam put-user-policy --user-name kara-packetizer \
    --policy-name KaraPacketizerWrite --policy-document "$POLICY_DOC"
  echo "  inline policy attached"
  echo "  creating access key (store these in a profile, they print ONCE):"
  "${AWS[@]}" iam create-access-key --user-name kara-packetizer \
    --query 'AccessKey.{AccessKeyId:AccessKeyId,SecretAccessKey:SecretAccessKey}' \
    --output table
fi

echo
echo "DONE. Smoke-test the upload:"
echo "  python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 --bucket $BUCKET --profile $PROFILE"
